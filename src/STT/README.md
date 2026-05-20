# medical-stt

Romanian medical STT + diarization. See `AGENTS.md` for context.

## Setup

```bash
cd src/STT
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # then fill HF_TOKEN
```

Accept license on HF for `pyannote/speaker-diarization-3.1` (once).

## Use

```bash
medical-stt run path/to/audio.wav --out out.json
medical-stt run path/to/audio.wav --no-diarize     # skip diarization
medical-stt config                                  # show resolved settings
```
