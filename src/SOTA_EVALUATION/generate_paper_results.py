"""
Final paper-results aggregator.

Run AFTER:
  1. `eval.py` has been run on every condition (writes <pred-dir>/eval/summary.json).
  2. `manual_score_motivul.py` has been completed by both raters
     (writes CSVs to data/chiropractor_ro/manual_scores/).

Produces a self-contained context bundle an analyzing agent can read:
  - paper_results.md   tables + pointers to context files + analysis brief
  - paper_results.csv  same numbers, flat

The script does NOT call any LLM. The "Conclusions" section is left for an
agent (e.g. a separate Claude Code session) to write after reading this file
plus the linked context (AGENTS.md, MANUAL_SCORING.md, per-case JSONLs, etc.).

Usage:
    python generate_paper_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_split import TEST_COMPLETE_PAIR_IDS  # noqa: E402

DEFAULT_PRED_ROOT = ROOT / "data" / "chiropractor_ro" / "predictions"
DEFAULT_MANUAL_DIR = ROOT / "data" / "chiropractor_ro" / "manual_scores"
DEFAULT_OUTPUT_MD = ROOT / "paper_results.md"
DEFAULT_OUTPUT_CSV = ROOT / "paper_results.csv"
SKIP_CONDITION_DIRS = {"mock_test", "eval"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def discover_conditions(pred_root: Path) -> list[str]:
    return sorted(
        p.name for p in pred_root.iterdir()
        if p.is_dir() and p.name not in SKIP_CONDITION_DIRS
    )


def load_auto_summary(pred_root: Path, condition: str) -> Optional[dict]:
    path = pred_root / condition / "eval" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_json_validity(pred_root: Path, condition: str) -> dict:
    """Walk .meta.json files for the condition, count parse/schema status."""
    cond_dir = pred_root / condition
    total = ok = coerced = failed = 0
    coercions_total = 0
    for meta_path in cond_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        total += 1
        status = (meta.get("prediction_status") or "").lower()
        if status in ("ok", "valid", "success"):
            ok += 1
        elif "coerce" in status or meta.get("coercions"):
            coerced += 1
            coercions_total += len(meta.get("coercions") or [])
        elif meta.get("parse_or_schema_error"):
            failed += 1
        else:
            ok += 1  # assume valid if status field missing but no errors
    return {
        "total_with_meta": total,
        "ok": ok,
        "coerced": coerced,
        "failed": failed,
        "coercion_events": coercions_total,
        "ok_rate": (ok / total) if total else None,
        "coerced_rate": (coerced / total) if total else None,
    }


def load_manual_rows(manual_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not manual_dir.exists():
        return rows
    for csv_path in sorted(manual_dir.glob("motivul_*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    row["score"] = float(row["score"])
                except (KeyError, ValueError):
                    continue
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def manual_mean_per_condition(rows: list[dict]) -> dict[str, dict]:
    """Mean per condition, averaging across raters for dual-rated items first."""
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_pair[(r["condition"], r["audio_id"])].append(r["score"])
    per_cond: dict[str, list[float]] = defaultdict(list)
    for (cond, _aid), scores in by_pair.items():
        per_cond[cond].append(sum(scores) / len(scores))
    out: dict[str, dict] = {}
    for cond, values in per_cond.items():
        out[cond] = {
            "n": len(values),
            "mean": sum(values) / len(values) if values else None,
        }
    return out


def cohen_kappa_per_dual_condition(rows: list[dict]) -> dict[str, dict]:
    """For conditions scored by 2+ raters, compute Cohen's kappa on the shared subset."""
    by_rater: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for r in rows:
        by_rater[r["rater"]][(r["condition"], r["audio_id"])] = r["score"]

    if len(by_rater) < 2:
        return {}

    raters = sorted(by_rater.keys())
    # Per condition, find shared items across the raters
    per_condition_shared: dict[str, list[tuple[list[float], list[float]]]] = defaultdict(list)
    a_name, b_name = raters[0], raters[1]
    a_map, b_map = by_rater[a_name], by_rater[b_name]
    shared_keys = set(a_map.keys()) & set(b_map.keys())
    by_cond: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for cond, aid in shared_keys:
        by_cond[cond].append((a_map[(cond, aid)], b_map[(cond, aid)]))

    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return {
            cond: {"n_shared": len(pairs), "kappa": None, "note": "sklearn not installed"}
            for cond, pairs in by_cond.items()
        }

    out: dict[str, dict] = {}
    for cond, pairs in by_cond.items():
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        exact = sum(x == y for x, y in zip(a, b))
        try:
            kappa = float(cohen_kappa_score(a, b))
        except Exception:
            kappa = None
        out[cond] = {
            "n_shared": len(pairs),
            "kappa": kappa,
            "exact_agreement": exact / len(pairs) if pairs else None,
            "raters": [a_name, b_name],
        }
    return out


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def build_main_table(
    conditions: list[str],
    auto_summaries: dict[str, dict],
    manual_mean: dict[str, dict],
    kappa_per_cond: dict[str, dict],
) -> str:
    lines = []
    lines.append("| Condition | n | VAS | Localizare F1 | Antecedente F1 | Medicatie (no-halluc \\| catch) | EvalFunc BERTScore | motivul (manual) | κ |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cond in conditions:
        summary = auto_summaries.get(cond)
        n = summary["n_cases"] if summary else 0
        f = (summary or {}).get("fields", {})
        vas = (f.get("evaluarea_durerii_vas") or {}).get("accuracy")
        loc = (f.get("localizarea_durerii") or {}).get("micro_f1")
        ant = (f.get("antecedente") or {}).get("micro_f1")
        med_field = f.get("medicatie_actuala") or {}
        empty_total = med_field.get("empty_ref_total") or 0
        empty_correct = med_field.get("empty_ref_correct") or 0
        tp = med_field.get("tp") or 0
        fn = med_field.get("fn") or 0
        positive_total = tp + fn
        med_cell = (
            f"{empty_correct}/{empty_total} \\| {tp}/{positive_total}"
            if (empty_total or positive_total) else "—"
        )
        ef = (f.get("evaluare_functionala_initiala") or {}).get("bertscore_f1_mean")
        manual = (manual_mean.get(cond) or {}).get("mean")
        k = (kappa_per_cond.get(cond) or {}).get("kappa")
        lines.append(
            f"| `{cond}` | {n} | {_fmt(vas)} | {_fmt(loc)} | {_fmt(ant)} | "
            f"{med_cell} | {_fmt(ef)} | {_fmt(manual)} | {_fmt(k)} |"
        )
    return "\n".join(lines)


def build_validity_table(
    conditions: list[str],
    validity: dict[str, dict],
) -> str:
    lines = []
    lines.append("| Condition | meta files | ok | coerced | failed | ok rate | coerced rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for cond in conditions:
        v = validity.get(cond) or {}
        lines.append(
            f"| `{cond}` | {v.get('total_with_meta', 0)} | {v.get('ok', 0)} | "
            f"{v.get('coerced', 0)} | {v.get('failed', 0)} | "
            f"{_pct(v.get('ok_rate'))} | {_pct(v.get('coerced_rate'))} |"
        )
    return "\n".join(lines)


def write_flat_csv(
    path: Path,
    conditions: list[str],
    auto_summaries: dict[str, dict],
    manual_mean: dict[str, dict],
    kappa_per_cond: dict[str, dict],
    validity: dict[str, dict],
) -> None:
    fields = [
        "condition", "n_cases",
        "vas_accuracy", "localizare_micro_f1", "antecedente_micro_f1",
        "medicatie_no_halluc_rate", "medicatie_empty_correct", "medicatie_empty_total",
        "medicatie_caught", "medicatie_positive_total", "medicatie_fp",
        "evalfunc_bertscore_f1_mean",
        "motivul_manual_mean", "motivul_kappa",
        "meta_total", "json_ok_rate", "json_coerced_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond in conditions:
            s = auto_summaries.get(cond) or {}
            fields_dict = s.get("fields", {})
            v = validity.get(cond) or {}
            w.writerow({
                "condition": cond,
                "n_cases": s.get("n_cases", 0),
                "vas_accuracy": (fields_dict.get("evaluarea_durerii_vas") or {}).get("accuracy"),
                "localizare_micro_f1": (fields_dict.get("localizarea_durerii") or {}).get("micro_f1"),
                "antecedente_micro_f1": (fields_dict.get("antecedente") or {}).get("micro_f1"),
                "medicatie_no_halluc_rate": (fields_dict.get("medicatie_actuala") or {}).get("empty_ref_accuracy"),
                "medicatie_empty_correct": (fields_dict.get("medicatie_actuala") or {}).get("empty_ref_correct"),
                "medicatie_empty_total": (fields_dict.get("medicatie_actuala") or {}).get("empty_ref_total"),
                "medicatie_caught": (fields_dict.get("medicatie_actuala") or {}).get("tp"),
                "medicatie_positive_total": ((fields_dict.get("medicatie_actuala") or {}).get("tp") or 0) + ((fields_dict.get("medicatie_actuala") or {}).get("fn") or 0),
                "medicatie_fp": (fields_dict.get("medicatie_actuala") or {}).get("fp"),
                "evalfunc_bertscore_f1_mean": (fields_dict.get("evaluare_functionala_initiala") or {}).get("bertscore_f1_mean"),
                "motivul_manual_mean": (manual_mean.get(cond) or {}).get("mean"),
                "motivul_kappa": (kappa_per_cond.get(cond) or {}).get("kappa"),
                "meta_total": v.get("total_with_meta"),
                "json_ok_rate": v.get("ok_rate"),
                "json_coerced_rate": v.get("coerced_rate"),
            })


# ---------------------------------------------------------------------------
# Context discovery for the analyzing agent
# ---------------------------------------------------------------------------

def collect_meta_errors(
    pred_root: Path,
    conditions: list[str],
    max_per_condition: int = 5,
) -> dict[str, list[dict]]:
    """For each condition with .meta.json files, surface failed/coerced cases."""
    out: dict[str, list[dict]] = {}
    for cond in conditions:
        cond_dir = pred_root / cond
        items: list[dict] = []
        for meta_path in sorted(cond_dir.glob("*.meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            status = (meta.get("prediction_status") or "").lower()
            has_error = bool(meta.get("parse_or_schema_error")) or "coerce" in status or bool(meta.get("coercions"))
            if not has_error:
                continue
            stem = meta_path.name[:-len(".meta.json")]
            items.append({
                "audio_id": stem,
                "status": meta.get("prediction_status"),
                "error": meta.get("parse_or_schema_error"),
                "n_coercions": len(meta.get("coercions") or []),
                "meta_path": str(meta_path.relative_to(ROOT)),
                "raw_path": str((cond_dir / f"{stem}.raw.txt").relative_to(ROOT)) if (cond_dir / f"{stem}.raw.txt").exists() else None,
                "pred_path": str((cond_dir / f"{stem}.json").relative_to(ROOT)) if (cond_dir / f"{stem}.json").exists() else None,
            })
        if items:
            out[cond] = items[:max_per_condition]
    return out


def context_file_pointers(pred_root: Path, manual_dir: Path, conditions: list[str]) -> list[str]:
    """List repo files an agent should consult, with relative paths."""
    pointers: list[str] = []
    for p in [
        ROOT / "AGENTS.md",
        Path(__file__).parent / "MANUAL_SCORING.md",
        Path(__file__).parent / "json_schema.py",
        Path(__file__).parent / "eval.py",
        ROOT / "src" / "data_split.py",
    ]:
        if p.exists():
            pointers.append(str(p.relative_to(ROOT)))
    for cond in conditions:
        summary = pred_root / cond / "eval" / "summary.json"
        per_case = pred_root / cond / "eval" / "per_case.jsonl"
        if summary.exists():
            pointers.append(str(summary.relative_to(ROOT)))
        if per_case.exists():
            pointers.append(str(per_case.relative_to(ROOT)))
    if manual_dir.exists():
        for csv_path in sorted(manual_dir.glob("motivul_*.csv")):
            pointers.append(str(csv_path.relative_to(ROOT)))
    return pointers


ANALYSIS_BRIEF = """\
You are reading this file as an analyzing agent. Your job is to write the
conclusions section of a paper using ONLY the data in this file plus the
linked context files. Do not invent numbers. If a cell shows `—`, say the
metric is unavailable.

Required deliverables (write them at the bottom of this same file under a
`## 5. Conclusions` heading):

1. **Per-metric winners** — for each column in section 1 (VAS, Localizare,
   Antecedente, Medicatie, EvalFunc, motivul-manual), name the winning
   condition and the runner-up. Cite numbers.

   For **Medicatie**, do NOT pretend models are clean here. The format is
   `correctly-empty / total-empty-refs | caught / total-positive-refs`.
   Be explicit: with current refs there is only 1 positive case (audio17:
   Hepatofin), so the catch column is degenerate. Models DO hallucinate
   medications in the negative class (FP count visible in the CSV column
   `medicatie_fp`). Report the no-halluc rate honestly and note that the
   positive-case metric is unstable at N=1.

2. **JSON validity vs accuracy tradeoff** — compare API conditions
   (no `.meta.json` → schema enforced by the SDK or assumed valid) against
   fine-tuned local models (visible coerced/failed rates). Discuss whether
   accuracy gains from one offset JSON-validity costs in the other.

3. **Manual scoring on `motivul_prezentarii`** — report mean per condition,
   the Cohen's kappa where available, and what the kappa value implies
   for trust in those rankings (use the rubric in MANUAL_SCORING.md).

4. **Limitations to declare** — explicit list. At minimum: N=15 per condition,
   dual-rated subset size, raters are engineers not clinicians, blind
   randomization eliminates per-condition bias but not common bias, schema
   coercion may inflate metrics for failed-output cases.

5. **Deployment recommendation** — given the locked decision to favor a
   cheap-deployable model with an expensive LLM as teacher/ceiling
   (AGENTS.md), recommend which condition is the deployment candidate and
   which is the reporting ceiling. Be explicit about confidence given the
   sample size.

Important signals to surface explicitly:

- **Cond 3 (RoLlama ICL) vs cond 5b (RoLlama FT)** is a planned
  ICL-vs-FT ablation on the same base model. Highlight this comparison
  directly — it is the core low-resource evidence in the paper. If cond 3
  shows much weaker structured-field scores than cond 5b on the same base,
  say so: ICL alone is not enough for the small RO-native model; FT on
  synthetic data is what closes the gap.
- **Medicatie 14/14 no-halluc is NOT a compliment** when the rest of the
  row shows very low structured-field scores. It means the model returned
  empty lists by default. The high no-halluc rate is then a side-effect
  of low overall extraction, not a sign of good calibration. Call this out
  if it applies.

Style: academic but direct, ~400-700 words, English.

When writing, link to specific files using markdown like `[AGENTS.md](AGENTS.md)`
or `[per_case.jsonl](data/chiropractor_ro/predictions/cond1.../eval/per_case.jsonl)`.

Do not edit sections 1-4 above. Only append section 5.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--manual-scores-dir", type=Path, default=DEFAULT_MANUAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args(argv)

    conditions = discover_conditions(args.predictions_root)
    if not conditions:
        print("no conditions found under predictions root.", file=sys.stderr)
        return 1

    auto_summaries: dict[str, dict] = {}
    validity: dict[str, dict] = {}
    for cond in conditions:
        summary = load_auto_summary(args.predictions_root, cond)
        if summary is not None:
            auto_summaries[cond] = summary
        validity[cond] = compute_json_validity(args.predictions_root, cond)

    manual_rows = load_manual_rows(args.manual_scores_dir)
    manual_mean = manual_mean_per_condition(manual_rows)
    kappa_per_cond = cohen_kappa_per_dual_condition(manual_rows)

    main_md = build_main_table(conditions, auto_summaries, manual_mean, kappa_per_cond)
    validity_md = build_validity_table(conditions, validity)

    # Compose results block (used both in output and as LLM input).
    parts: list[str] = []
    parts.append("# Paper results")
    parts.append("")
    parts.append(f"_Generated by `generate_paper_results.py` over {len(conditions)} conditions._")
    parts.append("")
    parts.append("## 1. Main comparison (per-condition × per-field)")
    parts.append("")
    parts.append(main_md)
    parts.append("")
    parts.append("Legenda metrici:")
    parts.append("- `VAS` = exact-match accuracy (int / list / null canonicalized)")
    parts.append("- `Localizare F1`, `Antecedente F1` = micro F1 across enum vocabulary")
    parts.append("- `Medicatie (no-halluc | catch)` = `correctly-empty / total-empty-refs | caught-meds / total-positive-refs`. With current refs only 1 positive case exists (audio17: Hepatofin); the no-halluc rate (left) reflects how often models correctly returned `[]` when ref was `[]`. False positives = hallucinated medications.")
    parts.append("- `EvalFunc BERTScore` = mean BERTScore F1 for `evaluare_functionala_initiala` (RO encoder)")
    parts.append("- `motivul (manual)` = mean of two-rater-averaged ternary scores per condition (0 / 0.5 / 1)")
    parts.append("- `κ` = Cohen's kappa on the dual-rated subset for this condition (if applicable)")
    parts.append("")
    parts.append("## 2. JSON validity per condition")
    parts.append("")
    parts.append(validity_md)
    parts.append("")
    parts.append("Conditions without `.meta.json` files (typically API conditions) report 0 meta files; their predictions enter the metrics pipeline directly.")
    parts.append("")
    if kappa_per_cond:
        parts.append("## 3. Inter-rater agreement detail")
        parts.append("")
        parts.append("| Condition | n shared | κ | exact agreement | raters |")
        parts.append("|---|---|---|---|---|")
        for cond, k in sorted(kappa_per_cond.items()):
            parts.append(
                f"| `{cond}` | {k.get('n_shared', 0)} | {_fmt(k.get('kappa'))} | "
                f"{_pct(k.get('exact_agreement'))} | {', '.join(k.get('raters', []))} |"
            )
        parts.append("")

    # Section 4: context bundle for the analyzing agent
    parts.append("## 4. Context for the analyzing agent")
    parts.append("")
    parts.append("### 4.1 Files to read before writing conclusions")
    parts.append("")
    for rel in context_file_pointers(args.predictions_root, args.manual_scores_dir, conditions):
        parts.append(f"- [{rel}]({rel})")
    parts.append("")

    meta_errors = collect_meta_errors(args.predictions_root, conditions)
    if meta_errors:
        parts.append("### 4.2 Sample JSON-failure / coercion cases per FT condition")
        parts.append("")
        parts.append("Use these to ground discussion of JSON-validity tradeoff. Open the `.raw.txt` for the original model output, `.meta.json` for the failure mode, `.json` for the coerced version that entered the evaluator.")
        parts.append("")
        for cond, items in meta_errors.items():
            parts.append(f"**`{cond}`** ({len(items)} sample case(s) shown)")
            parts.append("")
            parts.append("| audio_id | status | coercions | meta | raw | pred |")
            parts.append("|---|---|---|---|---|---|")
            for it in items:
                meta_link = f"[meta]({it['meta_path']})" if it['meta_path'] else "—"
                raw_link = f"[raw]({it['raw_path']})" if it['raw_path'] else "—"
                pred_link = f"[pred]({it['pred_path']})" if it['pred_path'] else "—"
                status_txt = (it.get("status") or "—")
                err = (it.get("error") or "")
                err_short = (err[:60] + "…") if isinstance(err, str) and len(err) > 60 else err
                parts.append(
                    f"| `{it['audio_id']}` | {status_txt}{(' / ' + str(err_short)) if err_short else ''} "
                    f"| {it['n_coercions']} | {meta_link} | {raw_link} | {pred_link} |"
                )
            parts.append("")

    parts.append("### 4.3 Analysis brief")
    parts.append("")
    parts.append(ANALYSIS_BRIEF)
    parts.append("")
    parts.append("## 5. Conclusions")
    parts.append("")
    parts.append("_To be written by an analyzing agent following the brief in §4.3._")
    parts.append("")

    final = "\n".join(parts)

    write_flat_csv(args.csv, conditions, auto_summaries, manual_mean, kappa_per_cond, validity)
    args.output.write_text(final, encoding="utf-8")
    print(f"[wrote] {args.output}", file=sys.stderr)
    print(f"[wrote] {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
