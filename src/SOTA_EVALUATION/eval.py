"""Evaluation pipeline for the Romanian chiropractor dialogue-to-note task.

Layer 1 functions score one field for one conversation.
Layer 2 functions aggregate those raw scores across a condition.
The CLI loads prediction/reference JSON files, validates them against the
locked schema, runs the scorers, and writes machine-readable reports.

See repo-root EVAL_DECISIONS.md for the metric contract.

Note: motivul_prezentarii has NO automatic metric. It is extracted by the
model and saved to prediction JSON, but scored later by a manual ternary
pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SOTA = SRC / "SOTA_EVALUATION"
for path in (SRC, SOTA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from json_schema import ANTECEDENTE_ENUM, LOCALIZARE_ENUM  # noqa: E402
from parser import parse_note  # noqa: E402


NOTE_FIELDS = [
    "motivul_prezentarii",
    "evaluarea_durerii_vas",
    "localizarea_durerii",
    "localizarea_durerii_alta",
    "antecedente",
    "antecedente_altele",
    "medicatie_actuala",
    "evaluare_functionala_initiala",
]


def score_localizare(pred: list[str], ref: list[str]) -> dict[str, dict]:
    """Multi-label scoring on the 19 anatomical regions enum."""
    return score_multilabel(pred, ref, LOCALIZARE_ENUM)


def score_antecedente(pred: list[str], ref: list[str]) -> dict[str, dict]:
    """Multi-label scoring on the 10 medical history conditions enum."""
    return score_multilabel(pred, ref, ANTECEDENTE_ENUM)



def _canon_vas(v):
    """Canonicalize VAS to a sorted tuple of ints, or None.

    Schema accepts int | list[int] | null. Single int and one-element list
    compare equal. Lists are multiset-equal (order-insensitive).
    """
    if v is None:
        return None
    if isinstance(v, int):
        return (v,)
    if isinstance(v, list):
        return tuple(sorted(int(x) for x in v))
    raise TypeError(f"unexpected VAS type: {type(v).__name__}")


def score_vas(pred, ref) -> dict:
    """Exact match on VAS (int | list[int] | null) after canonicalization."""
    return {"correct": _canon_vas(pred) == _canon_vas(ref)}


def score_multilabel(pred: list[str], ref: list[str], vocab: list[str]) -> dict[str, dict]:
    """
    Multi-label classification scoring against a closed vocabulary.

    For each label in vocab, computes TP/FP/FN as a binary decision
    (present in pred? present in ref?). Returns per-class counts so the
    aggregator in layer 2 can compute both micro and macro F1.

    pred and ref are lists of labels from `vocab`. Labels outside vocab
    are not possible here — schema validation guarantees enum compliance.
    """
    pred_set = set(pred)
    ref_set = set(ref)

    counts = {}
    for label in vocab:
        in_pred = label in pred_set
        in_ref = label in ref_set
        counts[label] = {
            "tp": int(in_pred and in_ref),
            "fp": int(in_pred and not in_ref),
            "fn": int(not in_pred and in_ref),
        }
    return counts


def _set_f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    pred_set = set(pred)
    ref_set = set(ref)
    if not pred_set and not ref_set:
        return 1.0
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    return _f1(tp, fp, fn)


def _score_set(pred: Sequence[str], ref: Sequence[str]) -> dict[str, Any]:
    pred_set = set(pred)
    ref_set = set(ref)
    return {
        "sample_f1": _set_f1(pred, ref),
        "exact_match": pred_set == ref_set,
        "pred_empty": not pred_set,
        "ref_empty": not ref_set,
    }


def _normalize_med_name(name: str) -> str:
    """Lowercase + strip whitespace. No brand→generic mapping (declared limitation)."""
    return name.lower().strip()


def score_medicatie(pred: list[dict], ref: list[dict]) -> dict:
    """
    Open-vocabulary multi-label on medication names + dose accuracy on matched.

    Input format (per schema):
        [{"denumire": "nurofen", "doza": "400mg"}, {"denumire": "concor", "doza": null}]

    Returns:
        tp, fp, fn: counts on normalized name set (for micro F1)
        doza_matched: number of TP medications (i.e., name agrees)
        doza_correct: subset of doza_matched where dose also agrees exactly

    Dose comparison rule: exact match on the raw dose string (null included).
    Two TPs with both doses null = correct. One null one populated = incorrect.
    """
    # Build name → dose dicts. If duplicate names exist (schema doesn't forbid),
    # last one wins. Acceptable: in practice meds are listed once.
    pred_doses = {_normalize_med_name(m["denumire"]): m["doza"] for m in pred}
    ref_doses = {_normalize_med_name(m["denumire"]): m["doza"] for m in ref}

    pred_names = set(pred_doses.keys())
    ref_names = set(ref_doses.keys())

    matched = pred_names & ref_names
    tp = len(matched)
    fp = len(pred_names - ref_names)
    fn = len(ref_names - pred_names)

    doza_correct = sum(1 for name in matched if pred_doses[name] == ref_doses[name])

    name_sample = _score_set(sorted(pred_names), sorted(ref_names))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "doza_matched": tp,           # alias, mai citibil în agregare
        "doza_correct": doza_correct,
        "name_sample_f1": name_sample["sample_f1"],
        "name_exact_match": name_sample["exact_match"],
        "pred_empty": name_sample["pred_empty"],
        "ref_empty": name_sample["ref_empty"],
    }



# ---------------------------------------------------------------------------
# BERTScore: lazy-loaded encoder (single load across all conversations)
# ---------------------------------------------------------------------------

_BERTSCORE_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


def _normalize_text_for_exact(value: Optional[str]) -> Optional[str]:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    return normalized.lower()


def _bertscore_f1(pred: str, ref: str) -> float:
    """
    Returns BERTScore F1 between two non-empty Romanian strings.

    First call loads the model (~500MB download on first run).
    Subsequent calls reuse the cached model.
    """
    from bert_score import score as _bs_score

    _, _, f1 = _bs_score(
        cands=[pred],
        refs=[ref],
        model_type=_BERTSCORE_MODEL,
        num_layers=12,
        verbose=False,
        rescale_with_baseline=False,
    )
    return float(f1.item())


# score_text rămâne folosit pentru evaluare_functionala_initiala (Scenario B).
# NU mai e folosit pentru motivul_prezentarii — acel câmp e scored manual.
# Vezi EVAL_DECISIONS.md secțiunea 5.
def score_text(pred: Optional[str], ref: Optional[str]) -> dict:
    """
    BERTScore F1 for free-text fields, with empty handling per EVAL_DECISIONS.md.

    Rules:
    - both None → 1.0 (correctly empty)
    - one None, other populated → 0.0 (miss or hallucination)
    - both populated → BERTScore F1 (Romanian encoder)

    Used for `evaluare_functionala_initiala`.
    """
    pred_norm = _normalize_optional_text(pred)
    ref_norm = _normalize_optional_text(ref)
    if pred_norm is None and ref_norm is None:
        return {"bertscore_f1": 1.0}
    if pred_norm is None or ref_norm is None:
        return {"bertscore_f1": 0.0}
    return {"bertscore_f1": _bertscore_f1(pred_norm, ref_norm)}


def score_evaluare_functionala(pred: Optional[str], ref: Optional[str]) -> dict:
    """
    Free-text scoring for evaluare_functionala_initiala.

    Per the 2026-06-16 decision (see AGENTS.md), we use Scenario B
    (BERTScore F1, same as motivul_prezentarii) because real refs have
    this field consistently populated with diagnostic + objectives +
    functional observations.
    """
    return score_text(pred, ref)


def score_nullable_text_exact(pred: Optional[str], ref: Optional[str]) -> dict[str, Any]:
    """Strict normalized exact match for secondary nullable text fields."""
    pred_norm = _normalize_text_for_exact(pred)
    ref_norm = _normalize_text_for_exact(ref)
    return {
        "correct": pred_norm == ref_norm,
        "pred_empty": pred_norm is None,
        "ref_empty": ref_norm is None,
    }


# ---------------------------------------------------------------------------
# Layer 2: aggregation
# ---------------------------------------------------------------------------


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 1.0
    return (2 * tp) / denom


def _accuracy(correct: int, total: int) -> Optional[float]:
    if total == 0:
        return None
    return correct / total


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(fmean(values))


def aggregate_vas(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(scores)
    correct = sum(1 for score in scores if score["correct"])
    return {
        "accuracy": _accuracy(correct, total),
        "correct": correct,
        "total": total,
    }


def aggregate_nullable_text_exact(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(scores)
    correct = sum(1 for score in scores if score["correct"])
    empty_ref_total = sum(1 for score in scores if score["ref_empty"])
    empty_ref_correct = sum(
        1 for score in scores if score["ref_empty"] and score["pred_empty"]
    )
    return {
        "accuracy": _accuracy(correct, total),
        "correct": correct,
        "total": total,
        "empty_ref_accuracy": _accuracy(empty_ref_correct, empty_ref_total),
        "empty_ref_correct": empty_ref_correct,
        "empty_ref_total": empty_ref_total,
    }


def aggregate_multilabel(
    scores: Sequence[dict[str, Any]],
    vocab: Sequence[str],
) -> dict[str, Any]:
    per_label = {
        label: {"tp": 0, "fp": 0, "fn": 0}
        for label in vocab
    }
    for score in scores:
        for label in vocab:
            label_counts = score["per_label"][label]
            per_label[label]["tp"] += label_counts["tp"]
            per_label[label]["fp"] += label_counts["fp"]
            per_label[label]["fn"] += label_counts["fn"]

    tp = sum(counts["tp"] for counts in per_label.values())
    fp = sum(counts["fp"] for counts in per_label.values())
    fn = sum(counts["fn"] for counts in per_label.values())

    per_label_f1 = {
        label: _f1(counts["tp"], counts["fp"], counts["fn"])
        for label, counts in per_label.items()
    }
    active_labels = [
        label
        for label, counts in per_label.items()
        if counts["tp"] + counts["fp"] + counts["fn"] > 0
    ]

    exact_correct = sum(1 for score in scores if score["exact_match"])
    empty_ref_total = sum(1 for score in scores if score["ref_empty"])
    empty_ref_correct = sum(
        1 for score in scores if score["ref_empty"] and score["pred_empty"]
    )

    macro_values = [per_label_f1[label] for label in active_labels]
    macro_f1 = _mean(macro_values)

    return {
        "micro_f1": _f1(tp, fp, fn),
        "macro_f1": 1.0 if macro_f1 is None else macro_f1,
        "macro_f1_all_labels": _mean(list(per_label_f1.values())),
        "sample_f1_mean": _mean([score["sample_f1"] for score in scores]),
        "exact_match_accuracy": _accuracy(exact_correct, len(scores)),
        "exact_match_correct": exact_correct,
        "total": len(scores),
        "empty_ref_accuracy": _accuracy(empty_ref_correct, empty_ref_total),
        "empty_ref_correct": empty_ref_correct,
        "empty_ref_total": empty_ref_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "active_labels": active_labels,
        "per_label": {
            label: {**counts, "f1": per_label_f1[label]}
            for label, counts in per_label.items()
        },
    }


def aggregate_medicatie(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(score["tp"] for score in scores)
    fp = sum(score["fp"] for score in scores)
    fn = sum(score["fn"] for score in scores)
    exact_correct = sum(1 for score in scores if score["name_exact_match"])
    doza_matched = sum(score["doza_matched"] for score in scores)
    doza_correct = sum(score["doza_correct"] for score in scores)
    empty_ref_total = sum(1 for score in scores if score["ref_empty"])
    empty_ref_correct = sum(
        1 for score in scores if score["ref_empty"] and score["pred_empty"]
    )

    return {
        "name_micro_f1": _f1(tp, fp, fn),
        "name_sample_f1_mean": _mean([score["name_sample_f1"] for score in scores]),
        "name_exact_match_accuracy": _accuracy(exact_correct, len(scores)),
        "name_exact_match_correct": exact_correct,
        "total": len(scores),
        "empty_ref_accuracy": _accuracy(empty_ref_correct, empty_ref_total),
        "empty_ref_correct": empty_ref_correct,
        "empty_ref_total": empty_ref_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "doza_accuracy_on_matched": _accuracy(doza_correct, doza_matched),
        "doza_correct": doza_correct,
        "doza_matched": doza_matched,
    }


def aggregate_text(scores: Sequence[Optional[dict[str, Any]]]) -> dict[str, Any]:
    scored = [score for score in scores if score is not None]
    return {
        "bertscore_f1_mean": _mean([score["bertscore_f1"] for score in scored]),
        "scored": len(scored),
        "total": len(scores),
        "skipped": len(scores) - len(scored),
    }


def score_note_pair(
    pred: dict[str, Any],
    ref: dict[str, Any],
    *,
    include_text: bool = True,
) -> dict[str, Any]:
    localizare_set = _score_set(
        pred["localizarea_durerii"],
        ref["localizarea_durerii"],
    )
    antecedente_set = _score_set(pred["antecedente"], ref["antecedente"])

    scores: dict[str, Any] = {
        "evaluarea_durerii_vas": score_vas(
            pred["evaluarea_durerii_vas"],
            ref["evaluarea_durerii_vas"],
        ),
        "localizarea_durerii": {
            "per_label": score_localizare(
                pred["localizarea_durerii"],
                ref["localizarea_durerii"],
            ),
            **localizare_set,
        },
        "localizarea_durerii_alta": score_nullable_text_exact(
            pred["localizarea_durerii_alta"],
            ref["localizarea_durerii_alta"],
        ),
        "antecedente": {
            "per_label": score_antecedente(pred["antecedente"], ref["antecedente"]),
            **antecedente_set,
        },
        "antecedente_altele": score_nullable_text_exact(
            pred["antecedente_altele"],
            ref["antecedente_altele"],
        ),
        "medicatie_actuala": score_medicatie(
            pred["medicatie_actuala"],
            ref["medicatie_actuala"],
        ),
        "evaluare_functionala_initiala": (
            score_evaluare_functionala(
                pred["evaluare_functionala_initiala"],
                ref["evaluare_functionala_initiala"],
            )
            if include_text
            else None
        ),
        "motivul_prezentarii": {
            "automatic_metric": None,
            "status": "manual_ternary_deferred",
        },
    }
    return scores


def aggregate_case_scores(case_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    field_scores = [row["scores"] for row in case_rows]
    return {
        "n_cases": len(case_rows),
        "ids": [row["id"] for row in case_rows],
        "fields": {
            "evaluarea_durerii_vas": aggregate_vas(
                [scores["evaluarea_durerii_vas"] for scores in field_scores]
            ),
            "localizarea_durerii": aggregate_multilabel(
                [scores["localizarea_durerii"] for scores in field_scores],
                LOCALIZARE_ENUM,
            ),
            "localizarea_durerii_alta": aggregate_nullable_text_exact(
                [scores["localizarea_durerii_alta"] for scores in field_scores]
            ),
            "antecedente": aggregate_multilabel(
                [scores["antecedente"] for scores in field_scores],
                ANTECEDENTE_ENUM,
            ),
            "antecedente_altele": aggregate_nullable_text_exact(
                [scores["antecedente_altele"] for scores in field_scores]
            ),
            "medicatie_actuala": aggregate_medicatie(
                [scores["medicatie_actuala"] for scores in field_scores]
            ),
            "evaluare_functionala_initiala": aggregate_text(
                [
                    scores["evaluare_functionala_initiala"]
                    for scores in field_scores
                ]
            ),
            "motivul_prezentarii": {
                "automatic_metric": None,
                "status": "manual_ternary_deferred",
            },
        },
    }


# ---------------------------------------------------------------------------
# Driver / CLI
# ---------------------------------------------------------------------------


def load_valid_note(path: Path) -> dict[str, Any]:
    """Load a JSON note and validate it against the locked schema."""
    return parse_note(path.read_text(encoding="utf-8"))


def find_prediction_path(pred_dir: Path, conversation_id: str) -> Path:
    exact = pred_dir / f"{conversation_id}.json"
    if exact.exists():
        return exact

    matches = sorted(pred_dir.glob(f"{conversation_id}*.json"))
    if not matches:
        raise FileNotFoundError(
            f"missing prediction for {conversation_id} in {pred_dir}"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"multiple prediction files for {conversation_id} in {pred_dir}: {names}"
        )
    return matches[0]


def infer_prediction_ids(pred_dir: Path) -> list[str]:
    ids = set()
    pattern = re.compile(r"^(audio\d+)")
    for path in pred_dir.glob("*.json"):
        match = pattern.match(path.name)
        if match:
            ids.add(match.group(1))
    return sorted(ids, key=_natural_audio_key)


def _natural_audio_key(conversation_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([a-zA-Z_]+)(\d+)", conversation_id)
    if not match:
        return (conversation_id, -1)
    return (match.group(1), int(match.group(2)))


def evaluate_condition(
    *,
    pred_dir: Path,
    ref_dir: Path,
    ids: Sequence[str],
    include_text: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_rows = []
    for conversation_id in ids:
        pred_path = find_prediction_path(pred_dir, conversation_id)
        ref_path = ref_dir / f"{conversation_id}.json"
        if not ref_path.exists():
            raise FileNotFoundError(f"missing reference for {conversation_id}: {ref_path}")

        pred = load_valid_note(pred_path)
        ref = load_valid_note(ref_path)
        scores = score_note_pair(pred, ref, include_text=include_text)
        case_rows.append(
            {
                "id": conversation_id,
                "prediction_path": str(pred_path),
                "reference_path": str(ref_path),
                "scores": scores,
            }
        )

    summary = aggregate_case_scores(case_rows)
    summary["prediction_dir"] = str(pred_dir)
    summary["reference_dir"] = str(ref_dir)
    summary["include_text_metrics"] = include_text
    return summary, case_rows


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"per_label", "ids", "active_labels"}:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else key
            _flatten(next_prefix, nested, out)
    elif isinstance(value, list):
        out[prefix] = ",".join(str(item) for item in value)
    else:
        out[prefix] = value


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    flat: dict[str, Any] = {}
    _flatten("", summary, flat)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(flat))
        writer.writeheader()
        writer.writerow(flat)


def parse_ids(value: str) -> list[str]:
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("expected at least one conversation id")
    return ids


def resolve_eval_ids(args: argparse.Namespace) -> list[str]:
    if args.ids:
        return sorted(args.ids, key=_natural_audio_key)
    if args.ids_file:
        ids = [
            line.strip()
            for line in args.ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return sorted(ids, key=_natural_audio_key)
    if args.split == "pred-dir":
        return infer_prediction_ids(args.pred_dir)

    from data_split import TEST_COMPLETE_PAIR_IDS  # noqa: WPS433

    return sorted(TEST_COMPLETE_PAIR_IDS, key=_natural_audio_key)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory containing one prediction JSON per conversation id.",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=ROOT / "data" / "chiropractor_ro" / "refs",
        help="Directory containing reference JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write summary.json, summary.csv, and per_case.jsonl. Defaults to <pred-dir>/eval.",
    )
    parser.add_argument(
        "--split",
        choices=["test-complete", "pred-dir"],
        default="test-complete",
        help="Which IDs to evaluate when --ids/--ids-file is not provided.",
    )
    parser.add_argument(
        "--ids",
        type=parse_ids,
        help="Comma-separated conversation IDs to evaluate, e.g. audio5,audio6.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Text file with one conversation ID per line.",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore for fast structural checks.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pred_dir = args.pred_dir.resolve()
    ref_dir = args.ref_dir.resolve()
    if not pred_dir.exists():
        raise FileNotFoundError(f"prediction directory does not exist: {pred_dir}")
    if not ref_dir.exists():
        raise FileNotFoundError(f"reference directory does not exist: {ref_dir}")

    ids = resolve_eval_ids(args)
    if not ids:
        raise RuntimeError("no conversation IDs selected for evaluation")

    summary, per_case = evaluate_condition(
        pred_dir=pred_dir,
        ref_dir=ref_dir,
        ids=ids,
        include_text=not args.skip_bertscore,
    )

    output_dir = (args.output_dir or (pred_dir / "eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_case.jsonl", per_case)
    write_summary_csv(output_dir / "summary.csv", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
