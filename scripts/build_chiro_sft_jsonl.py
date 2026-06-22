#!/usr/bin/env python3
"""Build chat-style SFT JSONL from a synthetic chiropractor dataset.

Example:
    python scripts/build_chiro_sft_jsonl.py \
      --synthetic-root data/synthetic/Claude \
      --out artifacts/ft/synthetic_claude_sft_messages.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_split import assert_no_test_leakage  # noqa: E402
from SOTA_EVALUATION.json_schema import (  # noqa: E402
    ANTECEDENTE_ENUM,
    LOCALIZARE_ENUM,
    NOTE_SCHEMA,
)

SYSTEM_PROMPT = f"""Ești un sistem de extragere de informații clinice din transcrierile consultațiilor unui cabinet de chiropractică din România.

Primești o transcriere (dialog terapeut-pacient) și returnezi UN SINGUR obiect JSON cu informațiile extrase.

REGULA FUNDAMENTALĂ: dacă o informație NU este rostită explicit în conversație, câmpul rămâne gol (null sau listă goală). Nu deduce. Nu presupune. Nu completa pe baza intuiției clinice.

Schema exactă:
{{
  "motivul_prezentarii": <string sau null>,
  "evaluarea_durerii_vas": <int 0-10, listă de int 0-10 sau null>,
  "localizarea_durerii": <listă de slug-uri din enum>,
  "localizarea_durerii_alta": <string sau null>,
  "antecedente": <listă de slug-uri din enum>,
  "antecedente_altele": <string sau null>,
  "medicatie_actuala": <listă de obiecte {{"denumire": string, "doza": string sau null}}>,
  "evaluare_functionala_initiala": <string sau null>
}}

Slug-uri localizarea_durerii permise: {LOCALIZARE_ENUM}
Slug-uri antecedente permise: {ANTECEDENTE_ENUM}

Returnează DOAR JSON valid, fără markdown fences și fără text suplimentar. Folosește null, nu None. Listele goale sunt [].
"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_seed_manifest(synthetic_root: Path) -> None:
    seed_pairs = synthetic_root / "seed_pairs.tsv"
    if not seed_pairs.exists():
        return
    with seed_pairs.open(encoding="utf-8", newline="") as f:
        seed_ids = [row["seed_id"] for row in csv.DictReader(f, delimiter="\t")]
    assert_no_test_leakage(seed_ids, context=f"{seed_pairs} seed manifest")


def resolve_dataset_path(raw_path: str, synthetic_root: Path) -> Path:
    raw = Path(raw_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(ROOT / raw)
        candidates.append(synthetic_root / raw)
        parts = raw.parts
        if parts and parts[0] == "synthetic":
            candidates.append(synthetic_root / Path(*parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def rows_from_index(synthetic_root: Path) -> list[tuple[str, Path, Path]]:
    index_path = synthetic_root / "index.tsv"
    if not index_path.exists():
        rows = []
        for transcript_path in sorted((synthetic_root / "transcripts").glob("synth_*.txt")):
            note_path = synthetic_root / "refs" / f"{transcript_path.stem}.json"
            rows.append((transcript_path.stem, transcript_path, note_path))
        return rows

    rows = []
    with index_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            transcript_path = resolve_dataset_path(row["transcript_path"], synthetic_root)
            note_path = resolve_dataset_path(row["note_path"], synthetic_root)
            rows.append((row["conversation_id"], transcript_path, note_path))
    return rows


def make_sft_messages(transcript: str, note: dict[str, Any], validator: Draft7Validator) -> dict[str, Any]:
    errors = sorted(validator.iter_errors(note), key=lambda e: e.path)
    if errors:
        first = errors[0]
        raise ValueError(f"schema error at {list(first.path)}: {first.message}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "--- TRANSCRIEREA CONSULTAȚIEI ---\n\n" + transcript.strip()},
            {
                "role": "assistant",
                "content": json.dumps(note, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def build_dataset(synthetic_root: Path) -> list[dict[str, Any]]:
    validate_seed_manifest(synthetic_root)
    validator = Draft7Validator(NOTE_SCHEMA)
    rows = []
    missing = []
    for conversation_id, transcript_path, note_path in rows_from_index(synthetic_root):
        if not transcript_path.exists() or not note_path.exists():
            missing.append((conversation_id, transcript_path, note_path))
            continue
        rows.append(
            make_sft_messages(
                transcript_path.read_text(encoding="utf-8"),
                load_json(note_path),
                validator,
            )
        )
    if missing:
        preview = ", ".join(item[0] for item in missing[:5])
        raise FileNotFoundError(f"missing transcript/ref pairs for: {preview}")
    if not rows:
        raise RuntimeError(f"no synthetic rows found under {synthetic_root}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, default=ROOT / "data" / "synthetic" / "Claude")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "ft" / "synthetic_claude_sft_messages.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic_root = args.synthetic_root.resolve()
    rows = build_dataset(synthetic_root)
    write_jsonl(args.out.resolve(), rows)
    print(f"rows={len(rows)}")
    print(f"wrote={args.out.resolve()}")


if __name__ == "__main__":
    main()
