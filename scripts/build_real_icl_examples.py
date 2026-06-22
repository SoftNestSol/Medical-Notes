#!/usr/bin/env python3
"""Build audited real-pool ICL examples for conditions 3 and 4.

Default output uses only rows marked include_default=1 and status=ready in
src/ICL/real_examples_manifest.tsv. The manifest is intentionally conservative:
bad few-shot examples teach the model to violate the "if not spoken, leave
empty" rule.

Example:
    python scripts/build_real_icl_examples.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SOTA = SRC / "SOTA_EVALUATION"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SOTA))

from data_split import POOL_COMPLETE_PAIR_IDS, assert_no_test_leakage  # noqa: E402
from parser import parse_note  # noqa: E402

DEFAULT_MANIFEST = ROOT / "src" / "ICL" / "real_examples_manifest.tsv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "icl"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_conversation_as_transcript(path: Path) -> str:
    data = load_json(path)
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"conversation JSON has no segments list: {path}")

    lines = []
    for segment in segments:
        raw_speaker = segment.get("speaker") or "SPEAKER_00"
        speaker = str(raw_speaker).strip() or "SPEAKER_00"
        text = str(segment.get("text", "")).strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    required = {"conversation_id", "include_default", "status", "rationale"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"{path} must have TSV columns: {sorted(required)}")

    selected = [row for row in rows if row["include_default"] == "1"]
    not_ready = [row["conversation_id"] for row in selected if row["status"] != "ready"]
    if not_ready:
        raise ValueError(f"default ICL examples must be status=ready: {not_ready}")
    if not selected:
        raise ValueError(f"no default ICL examples selected in {path}")

    ids = [row["conversation_id"] for row in selected]
    assert_no_test_leakage(ids, context=f"{path} default ICL examples")

    non_pool = sorted(set(ids) - POOL_COMPLETE_PAIR_IDS)
    if non_pool:
        raise ValueError(
            f"default ICL examples must be in POOL_COMPLETE_PAIR_IDS: {non_pool}"
        )

    return selected


def build_examples(manifest_path: Path) -> list[dict[str, Any]]:
    rows = read_manifest(manifest_path)
    examples = []
    for n, row in enumerate(rows, start=1):
        cid = row["conversation_id"]
        conversation_path = ROOT / "data" / "chiropractor_ro" / "conversations" / f"{cid}.json"
        note_path = ROOT / "data" / "chiropractor_ro" / "refs" / f"{cid}.json"
        if not conversation_path.exists():
            raise FileNotFoundError(conversation_path)
        if not note_path.exists():
            raise FileNotFoundError(note_path)

        transcript = read_conversation_as_transcript(conversation_path)
        note = load_json(note_path)
        parse_note(json.dumps(note, ensure_ascii=False))
        examples.append(
            {
                "n": n,
                "conversation_id": cid,
                "conversation_path": str(conversation_path.relative_to(ROOT)),
                "note_path": str(note_path.relative_to(ROOT)),
                "rationale": row["rationale"],
                "transcript": transcript,
                "note": note,
            }
        )
    return examples


def format_prompt_block(examples: list[dict[str, Any]]) -> str:
    blocks = []
    for example in examples:
        blocks.append(
            "\n".join(
                [
                    f"### EXEMPLU ICL {example['n']} ({example['conversation_id']})",
                    "",
                    "TRANSCRIERE:",
                    example["transcript"].strip(),
                    "",
                    "JSON CORECT:",
                    json.dumps(example["note"], ensure_ascii=False, indent=2),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks).strip() + "\n"


def write_outputs(examples: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "real_pool_icl_examples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    prompt_path = out_dir / "real_pool_icl_prompt_block.txt"
    prompt_path.write_text(format_prompt_block(examples), encoding="utf-8")

    manifest_path = out_dir / "real_pool_icl_manifest.tsv"
    lines = [
        "conversation_id\tconversation_path\tnote_path\ttranscript_words\tpopulated_fields\trationale"
    ]
    for example in examples:
        populated = [
            field for field, value in example["note"].items() if is_populated(value)
        ]
        lines.append(
            "\t".join(
                [
                    example["conversation_id"],
                    example["conversation_path"],
                    example["note_path"],
                    str(word_count(example["transcript"])),
                    ",".join(populated),
                    example["rationale"],
                ]
            )
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples(args.manifest.resolve())
    write_outputs(examples, args.out_dir.resolve())
    print(f"examples={len(examples)}")
    print(f"ids={','.join(example['conversation_id'] for example in examples)}")
    print(f"out_dir={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
