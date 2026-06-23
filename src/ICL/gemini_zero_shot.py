#!/usr/bin/env python3
"""Condition 1: zero-shot extraction cu Gemini Flash (cheap API floor).

Foloseste EXACT acelasi SYSTEM_PROMPT si parser ca si conditia 2 (Claude), ca
sa comparam modele, nu prompturi. Singura diferenta fata de claude_zero_shot.py
este clientul API.

Usage:
    python src/ICL/gemini_zero_shot.py data/chiropractor_ro/conversations/audio5.json
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
sys.path.insert(0, str(SOTA))
sys.path.insert(0, str(ICL))
sys.path.insert(0, str(SCRIPTS))

from build_real_icl_examples import read_conversation_as_transcript  # noqa: E402
from claude_zero_shot import MAX_TOKENS, SYSTEM_PROMPT  # noqa: E402
from parser import ParseError, SchemaError, parse_note  # noqa: E402

load_dotenv(ROOT / ".env")

# Override cu GEMINI_MODEL daca se schimba alegerea de model API.
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Retry pe 429 (rate limit / free-tier quota) si 503 (overload tranzitoriu).
RETRYABLE_STATUS = {429, 503}
MAX_RETRIES = 5
BACKOFF_SECONDS = [15, 30, 45, 60, 60]


def extract_note(transcription: str) -> dict:
    """Call Gemini API and return the parsed + schema-validated JSON note."""
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_KEY not found in environment / .env file")

    client = genai.Client(api_key=api_key)

    user_message = f"--- TRANSCRIEREA CONSULTATIEI ---\n\n{transcription}"

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_TOKENS,
        response_mime_type="application/json",
        # gemini-2.5-* au "thinking" pornit by default, care consuma bugetul de
        # output si trunchiaza JSON-ul. Il dezactivam: zero-shot pur = cheap floor.
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

    raw = response.text
    return parse_note(raw)


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

    output_json = json.dumps(note, ensure_ascii=False, indent=2)
    print(output_json)


if __name__ == "__main__":
    main()
