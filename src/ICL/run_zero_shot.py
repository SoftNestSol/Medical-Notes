#!/usr/bin/env python3
"""Batch runner pentru conditiile zero-shot (1 = Gemini Flash, 2 = Claude Opus).

Itereaza peste un split de conversatii, citeste fiecare conversatie ca transcript,
cheama extract_note() din backend-ul ales, si scrie cate o predictie JSON per
conversatie in layout-ul asteptat de eval.py:

    data/chiropractor_ro/predictions/<cond>/audioNN.json

Apoi rulezi eval.py pe directorul respectiv:

    python src/SOTA_EVALUATION/eval.py --pred-dir data/chiropractor_ro/predictions/<cond>

Usage:
    python src/ICL/run_zero_shot.py --condition claude
    python src/ICL/run_zero_shot.py --condition gemini --ids audio5,audio6
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SOTA = SRC / "SOTA_EVALUATION"
ICL = SRC / "ICL"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SOTA, ICL, SCRIPTS):
    sys.path.insert(0, str(p))

from build_real_icl_examples import read_conversation_as_transcript  # noqa: E402
from data_split import TEST_COMPLETE_PAIR_IDS  # noqa: E402
from parser import ParseError, SchemaError  # noqa: E402

CONVERSATIONS = ROOT / "data" / "chiropractor_ro" / "conversations"
PREDICTIONS = ROOT / "data" / "chiropractor_ro" / "predictions"

# condition key -> (backend module, default prediction subdir)
CONDITIONS = {
    "gemini": ("gemini_zero_shot", "cond1_gemini_zeroshot"),
    "claude": ("claude_zero_shot", "cond2_claude_zeroshot"),
}


def parse_ids(value: str) -> list[str]:
    ids = [v.strip() for v in value.split(",") if v.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("expected at least one conversation id")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        required=True,
        choices=sorted(CONDITIONS),
        help="Care backend zero-shot: gemini (cond.1) sau claude (cond.2).",
    )
    parser.add_argument(
        "--ids",
        type=parse_ids,
        help="IDs separate prin virgula (ex: audio5,audio6). Default: TEST_COMPLETE_PAIR_IDS.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Unde se scriu predictiile. Default: predictions/<cond default subdir>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-genereaza si predictiile care exista deja (default: skip).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module_name, default_subdir = CONDITIONS[args.condition]
    backend = importlib.import_module(module_name)

    ids = args.ids or sorted(TEST_COMPLETE_PAIR_IDS)
    out_dir = (args.out_dir or PREDICTIONS / default_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[run] condition={args.condition} model={getattr(backend, 'MODEL', '?')} "
        f"n={len(ids)} out={out_dir}",
        file=sys.stderr,
    )

    failures: list[tuple[str, str]] = []
    for i, conv_id in enumerate(ids, 1):
        out_path = out_dir / f"{conv_id}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(ids)}] {conv_id} -> skip (exists)", file=sys.stderr)
            continue

        conv_path = CONVERSATIONS / f"{conv_id}.json"
        if not conv_path.exists():
            print(f"[{i}/{len(ids)}] {conv_id} -> MISSING conversation", file=sys.stderr)
            failures.append((conv_id, "missing conversation file"))
            continue

        transcript = read_conversation_as_transcript(conv_path)
        t0 = time.time()
        try:
            note = backend.extract_note(transcript)
        except (ParseError, SchemaError) as e:
            print(f"[{i}/{len(ids)}] {conv_id} -> {type(e).__name__}: {e}", file=sys.stderr)
            failures.append((conv_id, f"{type(e).__name__}: {e}"))
            continue
        except Exception as e:  # noqa: BLE001 - vrem sa continuam restul lotului
            print(f"[{i}/{len(ids)}] {conv_id} -> ERROR: {e}", file=sys.stderr)
            failures.append((conv_id, f"{type(e).__name__}: {e}"))
            continue

        out_path.write_text(
            json.dumps(note, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{i}/{len(ids)}] {conv_id} -> saved ({time.time() - t0:.1f}s)", file=sys.stderr)

    print(
        f"[done] {len(ids) - len(failures)}/{len(ids)} ok, {len(failures)} failed",
        file=sys.stderr,
    )
    if failures:
        for conv_id, reason in failures:
            print(f"  FAIL {conv_id}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
