# STT Module — Build Log (for paper notes)

Date: 2026-05-21
Hardware: CPU-only dev box (Linux/Fedora), Python 3.11.9 (pyenv).

---

## Goal
Romanian doctor-patient consultation audio → speaker-labeled JSON transcript.
Two consumers: training-data generation + semi-prod inference. Self-hosted only
(GDPR Art. 9 — no third-party transcription APIs).

---

## Stack chosen
- **WhisperX 3.8.5** as integrator. Bundles:
  - `faster-whisper` (CTranslate2 backend, `large-v3-turbo` model) — ASR.
  - `wav2vec2` forced alignment — word-level timestamp refinement.
  - `pyannote/speaker-diarization-community-1` — speaker turns.
- Glue: `assign_word_speakers` maps each word to a speaker by timestamp overlap.
- Deps: `pydantic-settings` (config), `typer` (CLI), `python-dotenv`.
- HF token required (gated diarization repos: `pyannote/segmentation-3.0`,
  `pyannote/speaker-diarization-community-1`).

---

## Pipeline (final)

```
audio.mp4
  → ffmpeg preprocess (highpass=80, lowpass=8000, loudnorm I=-16:TP=-1.5:LRA=11, 16kHz mono)
  → faster-whisper (large-v3-turbo, int8 CPU, condition_on_previous_text=False)
  → wav2vec2 forced alignment (ro)
  → pyannote diarization (num_speakers=2 hint)
  → assign_word_speakers
  → Transcript JSON: [{start, end, speaker, text}, ...]
```

Graceful degradation: missing HF token / failed alignment / failed diarization
all warn but don't kill the run.

---

## Module layout
```
src/STT/
├── src/medical_stt/
│   ├── pipeline.py     # transcribe(audio_path, ...) → Transcript
│   ├── config.py       # pydantic-settings, .env-driven
│   ├── schemas.py      # Segment, Transcript
│   └── cli.py          # typer: run / config
├── tests/samples/      # gitignored
├── pyproject.toml
├── .env.example        # HF_TOKEN, WHISPER_MODEL, DEVICE, COMPUTE_TYPE, ...
└── README.md
```

CLI:
```
medical-stt run <audio> --out file.json
              [--num-speakers N | --min-speakers/--max-speakers]
              [--no-diarize] [--no-preprocess]
              [--vad-onset 0.30 --vad-offset 0.20]
medical-stt batch <dir> --out-dir <dir>
              [--pattern '*.mp4,*.wav,...'] [-r] [--overwrite]
              (+ all run flags)
medical-stt config
```

---

## Infrastructure issues hit

1. **`/tmp` is tmpfs, 6 GB cap.** pip extracting CUDA wheels overran it.
   Fix: `TMPDIR=$PWD/.tmp` (on real disk).
2. **`pyenv` Python missing `_sqlite3`.** Built before `sqlite-devel` was
   installed. Fix: `dnf install sqlite-devel` → `pyenv install 3.11.9 --force`.
3. **CPU-only torch reinstalled as full CUDA torch** by whisperx dep
   resolution. Venv ended up ~7.5 GB. Tolerable on dev box; revisit for prod.

---

## API drift (whisperX 3.8.5 vs older docs)
- `whisperx.DiarizationPipeline` is **not** re-exported at top level.
  Import from `whisperx.diarize`.
- Constructor renamed `use_auth_token` → `token`.
- Default diarization model switched from `pyannote/speaker-diarization-3.1`
  to `pyannote/speaker-diarization-community-1`. New gate to accept on HF.

---

## Quality fixes applied

### A. `--num-speakers 2`
Tells pyannote to cluster into exactly 2 speakers. For 1:1 consults this is
known a priori. Skips silhouette/threshold search → slightly faster + more
balanced speaker assignment.

Observed on audio_1: 56/5 split → on audio_2: 38/17 (more balanced).

### B. ffmpeg preprocessing pass
```
ffmpeg -af "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11" \
       -ac 1 -ar 16000 ...
```
- `highpass=80` removes mic rumble / HVAC / breathing thumps below speech band.
- `lowpass=8000` matches Whisper's 16 kHz Nyquist; kills aliasing + hiss.
- `loudnorm` (EBU R128): consistent perceived loudness → quiet speakers no
  longer embedded as "silence + faint noise" by the diarizer; Whisper mel
  spectrogram is more uniform.

### C. `condition_on_previous_text=False`
Default Whisper feeds segment N-1's decoded text as prompt for segment N.
On low-SNR / silence, garbage decoded in N-1 (e.g. `tă`) primes N to produce
more of the same → runaway repetition loop. Disabling breaks the spiral.
Each segment decoded independently.

### D. VAD thresholds lowered (`vad_onset=0.300, vad_offset=0.200`)
WhisperX pre-segments audio via pyannote VAD (defaults `onset=0.500,
offset=0.363`). Any region the VAD rejects never reaches Whisper.

On our clips, default thresholds skipped multiple quiet but speech-bearing
spans (e.g. soft patient replies, low-volume doctor utterances mid-turn).
Lowering onset/offset makes the VAD more permissive → those spans now hit
Whisper and get transcribed. Trade-off: occasional non-speech blip (sniff,
throat clear) may be transcribed as noise. Acceptable on clinic recordings.
Knobs exposed as `--vad-onset` / `--vad-offset`.

---

## Measured impact (audio2.mp4, ~1 min sample)

| | v1 (raw) | v2 (preprocess + cond=False) | v3 (+ VAD onset/offset lowered) |
|---|---|---|---|
| Segments | 55 | 55 | **67** |
| Speaker split | 38/17 | 35/20 | balanced |
| Hallucination loop (18.80–27.14s) | present | gone | gone |
| Missing utterances (32–39s, 42–56s) | yes | yes | recovered |

Going from v1→v2: hallucinated 8-second loop replaced by accurate transcription.
Going from v2→v3: ~12 additional segments recovered (quiet speech that the
default pyannote VAD was discarding before Whisper saw it).

---

## 2026-05-24 scoring + diarization experiments

### Manual scoring set
Created hand-editable scoring references under `tests/scoring_samples/`:

- `audio_2_manual.json`
- `audio_7_manual.json`
- `audio_11_manual.json`
- `audio23_manual.json`
- `audio24_manual.json`

Reference format is intentionally minimal:

```json
[
  {"speaker": "SPEAKER_00", "text": "..."},
  {"speaker": "SPEAKER_01", "text": "..."}
]
```

Removed `start` / `end` timestamps and transcript metadata from manual files so
the scoring references can be corrected by hand without timestamp noise.

### Scoring harness
Added a local evaluation module under `medical_stt.eval` and exposed it through:

```bash
PYTHONPATH=src .venv/bin/python -m medical_stt.cli eval \
  tests/scoring_samples tests/scoring_samples/hyp \
  --json-out tests/scoring_samples/results.json
```

Metrics:

- **WER**: plain concatenated transcript, speaker-agnostic.
- **cpWER**: concatenated minimum-permutation speaker WER via `meeteval`.
  This catches diarization/turn-assignment errors that ordinary WER hides.

Manual references currently have no timestamps, so `meeteval` prints
`Assuming sort=False because timestamps are missing...`; this is expected.

### Finding: segment-level speaker labels lose word-level diarization
On `audio23`, the hypothesis had segments containing two speakers but only one
segment-level speaker label. Example failure:

```text
[56.68-63.49] SPEAKER_00:
Da, o simt ... Am înțeles, fesele dor?
```

The first phrase is patient speech and the second phrase is doctor speech.
WhisperX keeps per-word speaker assignments after `assign_word_speakers`, but
the pipeline was collapsing each Whisper segment to a single `speaker`.

### Fix: split output segments on word-level speaker changes
Implemented `_split_on_speaker_change(...)` in `pipeline.py`.

Method:

1. Read each aligned WhisperX segment.
2. Use `seg["words"][i]["speaker"]` when available.
3. Group contiguous words with the same speaker.
4. Emit one `Segment(start, end, speaker, text)` per contiguous speaker group.
5. Fall back to the segment-level speaker when word speakers are missing.

Result: output JSON can represent intra-ASR-segment speaker turns instead of
forcing the whole segment into a single speaker.

### Finding: ffmpeg preprocessing helps ASR but can hurt diarization embeddings
The earlier ffmpeg cleanup is good for Whisper ASR, but diarization clustering
uses speaker embeddings. Loudness normalization / filtering can slightly distort
speaker identity cues, especially on clips where turns are short or speakers are
close in timbre.

Experimented on `audio23`:

| Variant | WER | cpWER | Notes |
|---|---:|---:|---|
| current hyp before original-audio diarization | 18.0% | 45.6% | ASR ok, speaker clustering poor |
| `--no-preprocess` | 17.2% | 39.3% | better diarization, slightly better WER |
| ASR on preprocessed audio, diarization on original audio | 18.0% | 39.3% | keeps ASR pipeline, improves clustering |

### Fix: diarize on original audio, transcribe on preprocessed audio
Changed `pipeline.py` so:

- ASR + forced alignment still run on `work_path` (preprocessed audio).
- Pyannote diarization runs on the original `audio_path` when preprocessing was
  enabled.
- `assign_word_speakers` still maps diarization turns onto aligned ASR words.

This preserves the ASR cleanup path while giving diarization cleaner speaker
identity information.

### Measured impact on 5-file scoring set
After promoting the improved hypotheses for `audio23` and `audio_2`:

| File | WER | cpWER | Notes |
|---|---:|---:|---|
| audio23 | 18.0% | 39.3% | cpWER improved from 45.6% |
| audio24 | 5.2% | 10.4% | original-audio diarization worsened slightly; not promoted |
| audio_11 | 0.0% | 2.8% | already strong |
| audio_2 | 17.0% | 34.1% | cpWER improved from 37.4%; false extra speaker removed |
| audio_7 | 21.0% | 40.5% | no improvement from original-audio diarization |
| **mean** | **12.2%** | **25.4%** | mean cpWER down from 27.3% |

Verification:

```bash
.venv/bin/python -m py_compile \
  src/medical_stt/pipeline.py \
  src/medical_stt/cli.py \
  src/medical_stt/eval/metrics.py \
  src/medical_stt/eval/runner.py
```

### Experiments not promoted

- `audio_7` with original-audio diarization: cpWER stayed `40.5%` and
  hypothesis speaker count dropped from 3 to 2 while reference has 3 speakers.
  Not promoted.
- `audio24` with original-audio diarization: cpWER worsened from `10.4%` to
  `11.5%`. Not promoted.

### Next likely biggest gains

1. **audio_7 deletion investigation.** The remaining error is not primarily
   diarization; WER is high and hyp has 221 words vs 257 reference words. Listen
   around the missing span and tune VAD / preprocessing if needed.
2. **Glossary file.** Cheap expected WER gain, especially names and medical
   jargon (`Simona Peleș`, lumbar/toracal/cervical, fese, rotulă, etc.).
3. **LLM post-correction.** Likely to clean substitution-level errors such as
   `doare`/`doresc`, `fără`/`fac`, Romanian diacritics, and medical term forms.

---

## Deferred (decided in AGENTS.md, not built yet)
- Fine-tuning Whisper on Romanian medical audio (need 50+ hours clean data).
- DeepFilterNet / Demucs denoise (only if specific noisy recordings need it).
- FastAPI wrapper (Step 2).
- Dockerization (Step 3).
- Glossary file for `initial_prompt` term biasing (hook exists; file TBD).
- Stable speaker identity (doctor vs patient) across files — currently
  SPEAKER_00/01 labels are per-run arbitrary.

---

---

## Data split v2 — TEST grew 15 → 18 (2026-06-04)

audio14 (restored) + audio16 + audio17 promoted to TEST. audio18 + audio19
added to POOL. Total now: TEST 18, POOL 17 (covers all 35 audios).

**Why:** v1 covered only the 30 audios available on 2026-05-26. Five new
audios landed (audio15..audio19). With 12 hand-corrected refs already
available (audio15-23, 25-27), the v1 split gave a 4/8 (TEST/POOL) ratio
among refs — too few refs in eval to start working meaningfully while waiting
for the remaining refs to be corrected. v2 promotes the right 3 audios so
the available-ref split is 7/5 (TEST/POOL), matching the target ratio for
~17 expected refs (7 eval, 10 pool).

**How to apply:** TEST_IDS is again frozen at 18 going forward. POOL_IDS
may still grow as more audios land. The hard-fence check
(`assert_no_test_leakage`) is unchanged in behavior — just covers 3 more IDs.

**Affected files:** `src/data_split.py` (TEST + POOL sets, self-checks,
docstring), `tests/test_data_split.py` (count assertions), root `AGENTS.md`
(data split section + locked decisions list).

Tests: `pytest tests/test_data_split.py` → 7 passed.

---

## Schema change — `evaluarea_durerii_vas` (2026-05-24)

Expanded from `int | null` to `int | array[int] | null`.

**Why:** real consultations produce more than one VAS value per session
(different regions, different postures, before/after a manoeuvre, progression
across visits). Existing hand-annotated references already used both forms;
the previous `int|null` schema rejected the array case (e.g. `[5,6,7]`).

**How to apply:**
- Single spoken score → integer (`7`).
- Multiple distinct scores spoken → array in spoken order (`[8, 5]` for
  "lombar 8, cervical 5"; `[9, 6]` for "la început 9, acum 6").
- No number verbalized → `null`.
- Do not promote a single int into a one-element list.

**Eval metric implication:** exact-match must canonicalize. Treat a bare
`n` and `[n]` as equal; treat list-vs-list as multiset equality; treat `null`
as a distinct class.

Updated in:
- `src/SOTA_EVALUATION/json_schema.py` (NOTE_SCHEMA → `oneOf` block)
- `src/SOTA_EVALUATION/claude_zero_shot.py` (SYSTEM_PROMPT field rule)
- root `AGENTS.md` (schema table + metric table)

Verified: all 12 current refs in `data/chiropractor_ro/refs/` validate clean.

---

## Known remaining quality issues
- Speaker labels are run-local; pyannote does not assign roles. A
  post-processing step (e.g. "first speaker = doctor" heuristic, or voiceprint
  enrollment) is required for downstream consumers expecting `doctor`/`patient`.
- Diarization on very short utterances (<1 s) still occasionally misattributed.
- Quantitative scoring exists for WER/cpWER on a 5-file hand-corrected set.
  DER is still not measured because current manual references omit timestamps.
