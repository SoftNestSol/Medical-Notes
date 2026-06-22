#!/usr/bin/env python3
"""Condition 4: few-shot ICL extraction with Claude Opus.

Uses the same extraction instructions as zero-shot, plus audited real examples
from src/ICL/real_examples_manifest.tsv.

Usage:
    python src/ICL/claude_few_shot.py data/chiropractor_ro/conversations/audio5.json
    python src/ICL/claude_few_shot.py path/to/transcript.txt --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SOTA = SRC / "SOTA_EVALUATION"
ICL = SRC / "ICL"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SOTA))
sys.path.insert(0, str(ICL))
sys.path.insert(0, str(SCRIPTS))

from build_real_icl_examples import (  # noqa: E402
    DEFAULT_MANIFEST,
    build_examples,
    format_prompt_block,
    read_conversation_as_transcript,
)
from claude_zero_shot import MAX_TOKENS, MODEL, SYSTEM_PROMPT  # noqa: E402
from parser import ParseError, SchemaError, parse_note  # noqa: E402

load_dotenv(ROOT / ".env")


def build_system_prompt(manifest_path: Path = DEFAULT_MANIFEST) -> str:
    examples = build_examples(manifest_path)
    return (
        SYSTEM_PROMPT
        + "\n\n## Exemple ICL reale auditate\n\n"
        + "Foloseste exemplele doar ca referinta de format si stil. Regula "
        + "\"daca nu este rostit, ramane gol\" are prioritate peste orice "
        + "tipar aparent din exemple.\n\n"
        + format_prompt_block(examples)
    )


def read_input_transcript(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return read_conversation_as_transcript(path)
    return path.read_text(encoding="utf-8")


def extract_note(transcription: str, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Call Claude API with audited few-shot examples and return validated JSON."""
    api_key = os.getenv("ANTHROPIC_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_KEY not found in environment / .env file")

    client = anthropic.Anthropic(api_key=api_key)
    user_message = f"--- TRANSCRIEREA CONSULTAȚIEI ---\n\n{transcription}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": build_system_prompt(manifest_path),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    usage = response.usage
    print(
        f"[cache] created={usage.cache_creation_input_tokens} "
        f"read={usage.cache_read_input_tokens} "
        f"uncached={usage.input_tokens} "
        f"output={usage.output_tokens}",
        file=sys.stderr,
    )

    raw = response.content[0].text
    return parse_note(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        note = extract_note(
            read_input_transcript(input_path),
            manifest_path=args.manifest.resolve(),
        )
    except ParseError as e:
        print(f"[parse error] {e}", file=sys.stderr)
        sys.exit(2)
    except SchemaError as e:
        print(f"[schema error] {e}", file=sys.stderr)
        sys.exit(3)

    output_json = json.dumps(note, ensure_ascii=False, indent=2)
    print(output_json)

    output_path = args.output.resolve() if args.output else input_path.with_suffix(".icl.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_json + "\n", encoding="utf-8")
    print(f"[saved] {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
