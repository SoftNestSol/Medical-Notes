#!/usr/bin/env python3
"""Condition NOUA (Gemini ICL): few-shot extraction cu Gemini Flash.

Analogul cu Gemini al conditiei 4 (Claude few-shot). Adaugat 2026-06-23 pentru ca
Gemini Flash s-a dovedit surprinzator de bun la zero-shot, deci merita testat si
cu ICL.

Foloseste EXACT acelasi system prompt + exemple ICL ca si claude_few_shot
(`build_system_prompt`) si acelasi parser, ca sa comparam modele, nu prompturi.
Singura diferenta fata de claude_few_shot.py este clientul API.

Usage:
    python src/ICL/gemini_few_shot.py data/chiropractor_ro/conversations/audio5.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SOTA = SRC / "SOTA_EVALUATION"
ICL = SRC / "ICL"
SCRIPTS = ROOT / "scripts"
for p in (SOTA, ICL, SCRIPTS):
    sys.path.insert(0, str(p))

from build_real_icl_examples import (  # noqa: E402
    DEFAULT_MANIFEST,
    read_conversation_as_transcript,
)
from claude_few_shot import build_system_prompt  # noqa: E402
from claude_zero_shot import MAX_TOKENS  # noqa: E402
from parser import ParseError, SchemaError, parse_note  # noqa: E402

load_dotenv(ROOT / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

RETRYABLE_STATUS = {429, 503}
MAX_RETRIES = 5
BACKOFF_SECONDS = [15, 30, 45, 60, 60]


def extract_note(transcription: str, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Call Gemini API with audited few-shot examples and return validated JSON."""
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_KEY not found in environment / .env file")

    client = genai.Client(api_key=api_key)
    user_message = f"--- TRANSCRIEREA CONSULTATIEI ---\n\n{transcription}"

    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(manifest_path),
        max_output_tokens=MAX_TOKENS,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    response = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=user_message, config=config
            )
            break
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            if code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                delay = BACKOFF_SECONDS[attempt]
                print(
                    f"[retry] {code} attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"sleeping {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue
            raise

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] prompt={usage.prompt_token_count} "
            f"output={usage.candidates_token_count}",
            file=sys.stderr,
        )

    return parse_note(response.text)


def read_input_transcript(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return read_conversation_as_transcript(path)
    return path.read_text(encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <transcript.json|.txt>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        note = extract_note(read_input_transcript(input_path))
    except ParseError as e:
        print(f"[parse error] {e}", file=sys.stderr)
        sys.exit(2)
    except SchemaError as e:
        print(f"[schema error] {e}", file=sys.stderr)
        sys.exit(3)

    print(json.dumps(note, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
