"""
Manual ternary scoring CLI for `motivul_prezentarii`.

Per 2026-06-16 decision (see AGENTS.md): no automatic metric for this field.
Two raters score independently, blind to source condition. This script is the
rater UI. Aggregation + Cohen's kappa happen separately on the resulting CSVs.

Usage:
    python manual_score_motivul.py --rater radu
    python manual_score_motivul.py --rater radu --conditions cond1_gemini_zeroshot,cond4_claude_fewshot
    python manual_score_motivul.py --rater radu --reveal   # show condition names

Rubric:
  1.0 = pred captures complaint + main anatomical zone; no invention.
  0.5 = zone correct; detail missing or invented.
  0.0 = misses complaint/zone, mass hallucination, or asymmetric empty.
  both null -> 1.0.

Storage:
  data/chiropractor_ro/manual_scores/motivul_<rater>.csv
  Columns: timestamp, rater, blind_id, condition, audio_id, ref_text, pred_text,
           score, notes
  Resume-safe: re-running skips (rater, condition, audio_id) already scored.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import random
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_split import TEST_COMPLETE_PAIR_IDS  # noqa: E402

DEFAULT_PRED_ROOT = ROOT / "data" / "chiropractor_ro" / "predictions"
DEFAULT_REF_DIR = ROOT / "data" / "chiropractor_ro" / "refs"
DEFAULT_OUT_DIR = ROOT / "data" / "chiropractor_ro" / "manual_scores"

VALID_SCORES = {"0": 0.0, "0.5": 0.5, "1": 1.0}
SKIP_CONDITIONS = {"mock_test"}

FIELDS = [
    "timestamp", "rater", "blind_id", "condition", "audio_id",
    "ref_text", "pred_text", "score", "notes",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_text(value) -> str:
    if value is None:
        return "<null>"
    return str(value).strip() or "<empty>"


def _collect_pairs(
    pred_root: Path,
    ref_dir: Path,
    ids: set[str],
    conditions: Optional[list[str]],
) -> list[tuple[str, str, str, str]]:
    """Returns list of (condition, audio_id, ref_text, pred_text)."""
    pairs: list[tuple[str, str, str, str]] = []
    if conditions:
        cond_dirs = [pred_root / c for c in conditions]
    else:
        cond_dirs = sorted(
            p for p in pred_root.iterdir()
            if p.is_dir() and p.name not in SKIP_CONDITIONS
        )

    for cond_dir in cond_dirs:
        if not cond_dir.exists():
            print(f"[skip] no such condition dir: {cond_dir}", file=sys.stderr)
            continue
        for audio_id in sorted(ids):
            pred_path = cond_dir / f"{audio_id}.json"
            ref_path = ref_dir / f"{audio_id}.json"
            if not pred_path.exists():
                continue
            if not ref_path.exists():
                continue
            try:
                pred = _load_json(pred_path)
                ref = _load_json(ref_path)
            except Exception as exc:
                print(f"[skip] {audio_id} in {cond_dir.name}: {exc}", file=sys.stderr)
                continue
            pairs.append((
                cond_dir.name,
                audio_id,
                _safe_text(ref.get("motivul_prezentarii")),
                _safe_text(pred.get("motivul_prezentarii")),
            ))
    return pairs


def _load_existing(csv_path: Path, rater: str) -> set[tuple[str, str]]:
    """Return {(condition, audio_id)} already scored by this rater."""
    done: set[tuple[str, str]] = set()
    if not csv_path.exists():
        return done
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("rater") == rater:
                done.add((row["condition"], row["audio_id"]))
    return done


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _hr(width: int = 80) -> str:
    return "─" * width


def _render(idx: int, total: int, blind_id: str, condition: str, audio_id: str,
            ref: str, pred: str, reveal: bool) -> None:
    print()
    print(_hr())
    header = f"[{idx + 1}/{total}]  blind={blind_id}"
    if reveal:
        header += f"  ({condition} / {audio_id})"
    print(header)
    print(_hr())
    print("REF :", ref)
    print()
    print("PRED:", pred)
    print(_hr())
    print("Rubrica: 1.0 = match | 0.5 = zonă ok, detaliu lipsă/invent | 0.0 = greșit | both null → 1.0")


def _prompt_score() -> Optional[tuple[float, str]]:
    """Returns (score, notes) or None to skip/quit."""
    while True:
        raw = input("score [0 / 0.5 / 1, s=skip, q=quit, ?=help]: ").strip().lower()
        if raw == "q":
            return None
        if raw in ("s", "skip"):
            return (-1.0, "")  # sentinel: skip without saving
        if raw == "?":
            print("  0   -> 0.0  pred ratează acuza/zona, halucinează, sau asimetric empty")
            print("  0.5 -> 0.5  zona corectă, detaliu lipsește sau e inventat")
            print("  1   -> 1.0  pred captează acuza și zona; nu inventează")
            print("  s   -> skip this item without saving")
            print("  q   -> quit (everything saved so far is on disk)")
            continue
        if raw in VALID_SCORES:
            score = VALID_SCORES[raw]
            notes = input("notes (optional, Enter to skip): ").strip()
            return (score, notes)
        print("  invalid. enter 0, 0.5, 1, s, q, or ?")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rater", required=True, help="Rater identifier (e.g. radu).")
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--output", type=Path, help="CSV output path (default: data/.../manual_scores/motivul_<rater>.csv)")
    parser.add_argument("--conditions", help="Comma-separated condition subdirs to score (default: all).")
    parser.add_argument("--ids", help="Comma-separated audio IDs (default: TEST_COMPLETE_PAIR_IDS).")
    parser.add_argument("--reveal", action="store_true", help="Show condition + audio_id (defeats blinding).")
    parser.add_argument("--no-shuffle", action="store_true", help="Score in deterministic order.")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed (default: hash of rater).")
    args = parser.parse_args(argv)

    ids = set(args.ids.split(",")) if args.ids else set(TEST_COMPLETE_PAIR_IDS)
    conditions = args.conditions.split(",") if args.conditions else None
    csv_path = args.output or (DEFAULT_OUT_DIR / f"motivul_{args.rater}.csv")

    pairs = _collect_pairs(args.predictions_root, args.ref_dir, ids, conditions)
    if not pairs:
        print("no (pred, ref) pairs found.", file=sys.stderr)
        return 1

    already = _load_existing(csv_path, args.rater)
    todo = [(c, a, r, p) for (c, a, r, p) in pairs if (c, a) not in already]
    if not todo:
        print(f"already scored all {len(pairs)} pairs for rater={args.rater}. nothing to do.")
        return 0

    if not args.no_shuffle:
        seed = args.seed if args.seed is not None else hash(args.rater) & 0xFFFFFFFF
        random.Random(seed).shuffle(todo)

    print(f"rater={args.rater} | to score: {len(todo)} (already done: {len(already)}) | output: {csv_path}")
    if not args.reveal:
        print("blind mode ON (condition + audio_id hidden). Use --reveal to show.")

    try:
        for idx, (cond, audio_id, ref_text, pred_text) in enumerate(todo):
            blind_id = f"item_{idx + 1:03d}"
            _render(idx, len(todo), blind_id, cond, audio_id, ref_text, pred_text, args.reveal)
            result = _prompt_score()
            if result is None:
                print("\nquit. progress saved.")
                break
            score, notes = result
            if score < 0:
                continue  # skip without saving
            _append_row(csv_path, {
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "rater": args.rater,
                "blind_id": blind_id,
                "condition": cond,
                "audio_id": audio_id,
                "ref_text": ref_text,
                "pred_text": pred_text,
                "score": score,
                "notes": notes,
            })
        else:
            print(f"\ndone. all {len(todo)} items scored.")
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted. progress saved.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
