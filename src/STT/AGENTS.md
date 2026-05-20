# STT Module — Agent Context & Instructions

This file orients any agent (Claude Code, future contributors, future me) working
inside the `STT/` subfolder. Read it before writing code.

---

## What this module is

A speech-to-text + speaker diarization pipeline for Romanian medical conversations
between a specialist and a patient.

**Input:** audio file of a consultation.
**Output:** speaker-labeled transcript with timestamps, as JSON.

This module does **not** generate medical notes. That is a separate module in the
parent repo. The STT module's only job is to turn audio into a structured
transcript that downstream models can consume.

---

## Why this module exists (two consumers)

The STT output feeds two different pipelines, both built on the same core:

1. **Training data pipeline** — batch processing of recorded consultations to
   produce (transcript, note) pairs for training the medical notes model.
   Priorities: accuracy, especially correct speaker attribution. Latency
   irrelevant. Cost per hour of audio not a concern.

2. **Production inference pipeline** — real-time-ish step in an end-to-end
   product where a clinician records a consultation and gets a draft note.
   Priorities: low latency, predictable cost, good-enough accuracy (clinician
   reviews the note before saving, which catches STT errors).

The current build serves the **shared core** of both. The split between the two
pipelines is deferred until the core works.

---

## Decisions already made (do not relitigate without reason)

- **Language:** Romanian. Some English medical terms may appear inline.
- **Stack:** WhisperX (wraps `faster-whisper` + `pyannote.audio` + forced
  alignment). Starting point only — may switch to `faster-whisper` + `pyannote`
  directly if WhisperX maintenance issues bite.
- **Model:** `large-v3-turbo` by default. Empirically validated at ~90–95% on
  Romanian conversational audio.
- **Diarization:** `pyannote/speaker-diarization-3.1`. Requires HF token and
  accepting the model license on HuggingFace (do this once, store token in
  `.env`).
- **Deployment target (eventually):** self-hosted. No third-party transcription
  APIs. This is a hard constraint — medical audio is Article 9 special category
  data under GDPR. Sending it to OpenAI / Deepgram / AssemblyAI / etc. is off
  the table.
- **Python deps:** managed by `uv`.
- **Output format (v1):** JSON, list of segments. Each segment has:
  `start`, `end`, `speaker`, `text`. Word-level timestamps deferred (one config
  flag away if needed later).
- **Repo placement:** this module lives in the `STT/` subfolder of the parent
  training/notebooks repo. It is *not* its own repo.

---

## Decisions deliberately deferred

Do not build any of these until explicitly asked:

- Fine-tuning Whisper on Romanian medical audio. Revisit only with 50+ hours of
  cleanly reviewed audio. Prompt biasing + post-hoc LLM correction comes first.
- Ground-truth evaluation harness (WER, DER on hand-labeled samples).
- Streaming / real-time transcription.
- Authentication, job queues, databases.
- Training data review UI.
- Splitting the training-data pipeline from the production pipeline.

---

## Build order

1. **Local CLI.** Python module + thin CLI wrapper. Takes an audio file path,
   returns speaker-labeled JSON. No API, no Docker. Must run on Mac (dev) and
   Linux+CUDA (eventual cloud).
2. **FastAPI wrapper** around the same pipeline module. Single endpoint, sync.
3. **Dockerization.** CUDA base image for cloud deploy. Mac dev stays bare-metal
   (no Docker for local iteration — too slow).

We are currently at **step 1**. Do not jump ahead.

---

## Architecture principles

- **Pipeline is a standalone module.** Importable from CLI, future FastAPI, a
  notebook, a Celery worker, whatever. It must not import FastAPI or any web
  framework. Web layer is thin, sits on top.
- **Configuration via env vars** (HF token, model size, device, compute type).
  Use `pydantic-settings`. Defaults should work for local Mac dev.
- **No premature plumbing.** No database, no queue, no auth until there is a
  concrete reason. The most common failure mode for "let's set it up properly
  from the start" is ending up with infrastructure that has to be reworked
  before it's ever used.
- **CLI is for iteration**, not just for show. Running the API to test
  transcription on a sample file is annoying overhead. The CLI must be the
  fastest path from "I have an audio file" to "I see the output."

---

## Suggested layout (proposal, not gospel)

```
STT/
├── src/medical_stt/
│   ├── __init__.py
│   ├── pipeline.py       # Core: audio path → segments. No web deps.
│   ├── config.py         # pydantic-settings
│   ├── schemas.py        # Segment, Transcript dataclasses / pydantic models
│   └── cli.py            # typer CLI, calls pipeline.py
├── tests/
│   └── samples/          # gitignored audio samples
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

Adjust if there's a reason. Don't adjust for the sake of it.

---

## Working agreement

This module is built collaboratively between the developer and Claude Code.
A separate Claude instance (in chat) is used for planning, stress-testing
decisions, and discussing tradeoffs. Claude Code writes the code.

When in doubt about *what* to build or *why*, the developer will have already
discussed it in chat. When in doubt about *how* to build something the brief
already covers, default to the simplest thing that satisfies the brief and
flag the choice in the PR / commit message rather than asking.

When something in this file no longer reflects reality, update this file in
the same change. Stale agent docs are worse than no agent docs.

---

## House rules

- Romanian medical terminology will be passed via Whisper's `initial_prompt`
  parameter. There will eventually be a glossary file. Don't hardcode the
  glossary into pipeline code; load it from a file.
- The HuggingFace token is a secret. It belongs in `.env`, never in code,
  never in commits. `.env.example` documents which keys are expected.
- Sample audio in `tests/samples/` is real patient data or synthetic stand-ins.
  Treat it as confidential. Never commit. `.gitignore` enforces this; do not
  weaken it.
- Mac dev runs on CPU or MPS. Cloud runs on CUDA. The same Python code must
  work in both places; differences live in the env config and (eventually)
  the Dockerfile.
- Default to `faster-whisper`'s `compute_type="int8"` on Mac for speed, and
  `"float16"` on CUDA. Make this configurable, not hardcoded.