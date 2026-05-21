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

## Measured impact (audio_2.mp4, ~1 min sample)

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

## Deferred (decided in AGENTS.md, not built yet)
- Fine-tuning Whisper on Romanian medical audio (need 50+ hours clean data).
- Quantitative WER/DER evaluation on hand-labeled samples.
- DeepFilterNet / Demucs denoise (only if specific noisy recordings need it).
- FastAPI wrapper (Step 2).
- Dockerization (Step 3).
- Glossary file for `initial_prompt` term biasing (hook exists; file TBD).
- Stable speaker identity (doctor vs patient) across files — currently
  SPEAKER_00/01 labels are per-run arbitrary.

---

## Known remaining quality issues
- Speaker labels are run-local; pyannote does not assign roles. A
  post-processing step (e.g. "first speaker = doctor" heuristic, or voiceprint
  enrollment) is required for downstream consumers expecting `doctor`/`patient`.
- Diarization on very short utterances (<1 s) still occasionally misattributed.
- No quantitative WER/DER yet — assessment so far is qualitative on small N.
