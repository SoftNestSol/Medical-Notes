"""
Evaluation script for Romanian chiropractor dialogue-to-note task.

Per-field metric functions return raw counts/booleans per conversation.
Aggregation into final metrics (F1, accuracy, etc.) happens in layer 2.

See EVAL_DECISIONS.md for metric choices per field.

Note: motivul_prezentarii has NO automatic metric (manual ternary scoring
per EVAL_DECISIONS.md section 5). It is extracted by the model and saved
to pred JSON, but not scored automatically.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Layer 1: per-field metric functions
# ---------------------------------------------------------------------------


from json_schema import LOCALIZARE_ENUM, ANTECEDENTE_ENUM


def score_localizare(pred: list[str], ref: list[str]) -> dict[str, dict]:
    """Multi-label scoring on the 19 anatomical regions enum."""
    return score_multilabel(pred, ref, LOCALIZARE_ENUM)


def score_antecedente(pred: list[str], ref: list[str]) -> dict[str, dict]:
    """Multi-label scoring on the 10 medical history conditions enum."""
    return score_multilabel(pred, ref, ANTECEDENTE_ENUM)



def score_vas(pred: Optional[int], ref: Optional[int]) -> dict:
    """
    Exact match on integer pain score (0-10) or null.

    Both null = correct. Any mismatch (including null vs number) = wrong.
    Penalizes miss and hallucination symmetrically per EVAL_DECISIONS.md.
    """
    return {"correct": pred == ref}

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

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "doza_matched": tp,           # alias, mai citibil în agregare
        "doza_correct": doza_correct,
    }



# ---------------------------------------------------------------------------
# BERTScore: lazy-loaded encoder (single load across all conversations)
# ---------------------------------------------------------------------------

_BERTSCORE_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"


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
    if pred is None and ref is None:
        return {"bertscore_f1": 1.0}
    if pred is None or ref is None:
        return {"bertscore_f1": 0.0}
    return {"bertscore_f1": _bertscore_f1(pred, ref)}


def score_evaluare_functionala(pred: Optional[str], ref: Optional[str]) -> dict:
    """
    Free-text scoring for evaluare_functionala_initiala.

    Per the 2026-06-16 decision (see AGENTS.md), we use Scenario B
    (BERTScore F1, same as motivul_prezentarii) because real refs have
    this field consistently populated with diagnostic + objectives +
    functional observations.
    """
    return score_text(pred, ref)