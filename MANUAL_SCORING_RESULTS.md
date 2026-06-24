# Manual Scoring Results — `motivul_prezentarii`

> Handoff for the paper-writing agent. These are the final manual-ternary
> results for the one field with no automatic metric. Numbers are real (scored
> 2026-06-23). The interpretation section tells you what is claimable and what
> is noise — do not exceed it.

## Method (already locked, see `EVAL_DECISIONS.md` + `MANUAL_SCORING.md`)

- `motivul_prezentarii` has **no automatic metric**. BERTScore/ROUGE-L tested
  empirically on the real ground truth (dominant length 2–5 tokens) → gap
  between similar/different pairs < 0.05, pure noise (decision log 2026-06-16).
- Replaced with a **ternary rubric** scored by humans: `1.0` = captures
  complaint + main anatomical zone, no invention; `0.5` = zone correct, a
  detail missing or invented; `0.0` = misses complaint/zone, hallucination, or
  asymmetric empty; both-empty → `1.0`.
- **Two raters** (`andrei`, `radu`), **blind** to source condition (each item
  shown as `REF` / `PRED` / `blind_id` only, per-rater shuffle).
- All 6 conditions × 15 = **90 unique items** scored. Two conditions
  (`cond1_gemini_zeroshot`, `cond5_qwen`) **dual-rated by both** for Cohen's κ
  → 30 overlapping items.

## Results

| Condition | mean (0–1) | dual-rated | per-cond κ |
|---|---|---|---|
| cond2 Claude ZS | 0.600 | no | — |
| cond4 Claude FS | 0.567 | no | — |
| cond5_RoLlama (5b) | 0.567 | no | — |
| cond3 RoLlama ICL | 0.533 | no | — |
| cond1 Gemini ZS | 0.517 | yes | 0.479 |
| cond5_qwen (5a) | 0.367 | yes | 0.545 |

**Inter-rater agreement** (30 dual-rated items, cond1 + cond5_qwen): overall
Cohen's κ = **0.519**, exact agreement **70%** (21/30).

## How to write this — claimable vs noise

1. **κ = 0.519 is "moderate"** (band 0.40–0.60). At N=30 it is directional.
   Claim: *raters broadly agree, the rubric is shared.* Do **not** use κ to
   defend fine-grained rankings. State N=30 alongside it.

2. **The top four are within noise.** 0.600 → 0.533 spans ~0.07, which the
   stated limitation (N=15, CI width) says is **indistinguishable**. Do **not**
   write "cond2 wins motivul." Write: *prompting and large-model conditions,
   plus RoLlama (both ICL and FT), cluster around 0.52–0.60 with no separation.*

3. **The one real signal: `cond5_qwen` (5a) is the clear low outlier (0.367)** —
   ~0.15–0.23 below the cluster, outside the noise band. Consistent with its
   weakest structured fields (Localizare 0.511, lowest FT BERTScore 0.577).
   Claim: *the compact 4B fine-tuned deployment candidate pays its largest
   quality cost on free-text complaint summarization, not only on structured
   fields.*

4. **The ICL-vs-FT collapse does NOT repeat on this field.** On structured
   fields cond3 (RoLlama ICL) collapsed vs cond5b (RoLlama FT): Localizare
   0.148 → 0.638, Antecedente 0.000 → 0.545. On `motivul` they are tied (0.533
   vs 0.567). Claim: *the ICL failure of the small RO-native model is specific
   to the constrained-schema fields; free-text generation survives ICL.* This
   is a clean nuance, not a contradiction — surface it explicitly.

## Limitations to declare (in addition to the per-field eval limitations)

- N=15 per condition; differences under ~0.07 are not distinguishable.
- Only 2 of 6 conditions dual-rated; κ measures agreement, not correctness.
- Raters are engineers, not clinicians.
- Blind randomization removes per-condition rater bias but not common bias
  (both raters could share a systematic tendency).
- Means mix dual-rated (two-rater average) and single-rated (one rater)
  conditions; only the two dual conditions have a reliability estimate.

## Provenance

- Raw ratings: `data/chiropractor_ro/manual_scores/motivul_andrei.csv` (60),
  `motivul_radu.csv` (60).
- Aggregation: scripts in `MANUAL_SCORING.md §5` (mean per condition; Cohen's κ
  via `sklearn.metrics.cohen_kappa_score` on the 3 labels {0, 0.5, 1}).
- Main results table with all fields: `paper_results.md §1` (this column now
  filled).
