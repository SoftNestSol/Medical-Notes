# Evaluation Decisions

This document is the metric contract for the Romanian chiropractor dialogue-to-note task.

Core rule: **if it was not spoken, it must be empty**. Evaluation must reward correctly empty predictions and penalize hallucinated filled fields.

## Evaluation Set

Primary automatic evaluation uses `TEST_COMPLETE_PAIR_IDS` from `src/data_split.py`: the frozen test IDs that currently have structured reference notes available locally.

`TEST_IDS` remains the full frozen holdout set. Pending test references are not evaluated until their structured refs exist.

## Field Metrics

| Field | Metric | Notes |
|---|---|---|
| `evaluarea_durerii_vas` | Exact-match accuracy | `int` and one-element `list[int]` compare equal. Multi-value lists compare as multisets. `null == null` is correct. |
| `localizarea_durerii` | Multi-label F1 + exact-set accuracy + sample F1 | Closed enum. Sample F1 treats both-empty as `1.0`, so correct empty cases are rewarded. |
| `localizarea_durerii_alta` | Secondary exact/empty accuracy | Lowercased, whitespace-normalized text match. Not a headline metric. |
| `antecedente` | Multi-label F1 + exact-set accuracy + sample F1 | Closed enum. Sample F1 treats both-empty as `1.0`. |
| `antecedente_altele` | Secondary exact/empty accuracy | Lowercased, whitespace-normalized text match. Not a headline metric. |
| `medicatie_actuala` | Medication-name micro F1 + exact-set accuracy + dose accuracy on matched names | Name normalization is lowercase + strip only. No brand-to-generic mapping in v1. Dose accuracy is computed only for medication names matched in both prediction and reference. |
| `evaluare_functionala_initiala` | Mean BERTScore F1 | Romanian encoder: `dumitrescustefan/bert-base-romanian-cased-v1`. Both empty = `1.0`; one empty = `0.0`. |
| `motivul_prezentarii` | Manual ternary, deferred | No automatic metric. Use randomized blind manual scoring later: `0`, `0.5`, `1`; both empty = `1.0`. |

## Multi-label Aggregation

For `localizarea_durerii` and `antecedente`, report:

- `micro_f1`: global F1 over summed TP/FP/FN.
- `macro_f1`: mean per-label F1 over active labels only, where active means the label appears in either predictions or references.
- `macro_f1_all_labels`: diagnostic value over the full enum; absent labels with no FP/FN score `1.0`, so this can be inflated and should not be used as the headline.
- `sample_f1_mean`: mean per-case set F1; both-empty sets score `1.0`.
- `exact_match_accuracy`: exact set match per case.
- `empty_ref_accuracy`: among cases where the reference set is empty, the fraction where the prediction is also empty.

If a field has no active labels at all and all cases are empty, F1 is `1.0`: the model correctly predicted absence throughout.

## Text Normalization

For nullable text fields, `null` and blank/whitespace-only strings are treated as empty during scoring.

For secondary exact text fields (`*_alta`), populated strings are compared after lowercasing and collapsing whitespace. This is intentionally strict and only diagnostic.

## Non-goals for v1

- No automatic metric for `motivul_prezentarii`.
- No medication synonym table or brand-to-generic mapping.
- No clinical inference beyond the reference note.
- No single headline aggregate over all fields; report per-field metrics.
