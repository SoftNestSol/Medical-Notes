# Project Context — Romanian Chiropractor Dialogue-to-Note (AI Assistant Context)

> This file gives an AI assistant (Claude, ChatGPT, etc.) full context to help with coding, research, or planning tasks on this project. Read it before answering anything project-specific.

---

## Project identity

**What:** Build a system that takes Romanian chiropractor-patient conversations (audio → transcript) and produces a structured medical note matching a real Romanian clinic's template (Osteopath Concept).

**Why:** University bioNLP course project. Two co-occurring deliverables:
1. **Comparative experiment paper** — a **low-resource study**: when in-domain data is scarce on an under-represented language (Romanian), which method best extracts a structured clinical note from a chiropractor-patient conversation? (Reframed 2026-06-21 — see decision log. Cost is a *secondary* justification, not the headline; cost differences between candidate models were too small to anchor the paper.)
2. **MVP** — working pipeline that the chiropractor team can eventually use. MVP is a byproduct of the winning experimental condition, not built separately.

**Who:** Two-person team. Has access to a chiropractor team for data collection and clinical validation. Course paper only, not for academic publication.

**Timeline:** Compressed (~2 weeks active work). Speed prioritized over polish.

---

## Communication style preferences

The lead user prefers:
- **Concise responses, no fluff.** No "great question," no "you're absolutely right," no preamble.
- **Skeptical mode by default.** Before agreeing, stress-test the idea. Identify the weakest part.
- **Small sequential steps, verified before moving on.** Never dump a full plan unprompted.
- **Partnership tone.** Talk like a collaborator, not a service provider.
- **Push back when needed.** If a decision is weak, say so directly with reasoning.
- **Approval-only replies are fine** when no action is requested.

---

## Task definition

**Input:** Romanian conversation transcript between a chiropractor/osteopath/kinesiotherapist and a patient. Source audio runs ~3+ minutes typically. Transcription is done by an in-house WhisperX pipeline (not the focus of this project).

**Output:** JSON object with the following fields, matching the Osteopath Concept clinic's intake form:

| Field | Type | Source |
|-------|------|--------|
| `motivul_prezentarii` | free text | summarization of patient's main complaint |
| `evaluarea_durerii_vas` | int 0-10, list[int 0-10], or null | extraction of verbalized pain number(s); list when multiple distinct scores are spoken (different regions/postures/moments) |
| `localizarea_durerii` | multi-select from ~20 regions | extraction (cervical, toracal, lombar, sacral, umar_dr, umar_stg, ...) |
| `antecedente` | multi-select from ~10 conditions + free text | extraction (hipertensiune, diabet, hernia_disc, scolioza, ...) |
| `medicatie_actuala` | list of {nume, doza} | extraction of medications mentioned |
| `evaluare_functionala_initiala` | free text or null | verbalized diagnostic + verbalized therapeutic objectives + verbalized functional observations (posture, biomechanics, mobility); may be empty if not spoken |

`Diagnostic` and `Obiective_terapeutice` from the original form are **absorbed into `evaluare_functionala_initiala`** — in practice chiropractors write them there. No separate fields in v1. See Decision log entry 2026-06-16.

**The non-negotiable rule:**

> "If it's not spoken, leave empty."

No hallucinations, no clinical inference, no completion from probable context. Absent information → empty/null. This rule is enforced in:
- Every prompt
- Every annotation guideline given to chiropractors
- Every evaluation metric (correct empties must be rewarded)

---

## Data inventory

| Resource | Size | Status | Use |
|----------|------|--------|-----|
| Real RO chiropractor conversations | 30 → 50 | 30 by day 2, +20 in week 1 | Test set + ICL pool + synthetic seeds |
| Reference notes for real conversations | 30 → 50 | Chiropractors annotating now | Ground truth |
| Translated MTS dataset (RO) | ~1700 pairs | Already produced (machine translation) | Fine-tuning condition |
| Synthetic chiropractor data | TBD | To be generated during project | Fine-tuning condition + ICL condition |

**Data split (locked):**
- **18 test** — frozen, evaluated once, never touched during development (v2 on 2026-06-04; was 15 in v1)
- **0 formal dev** — prompt iteration on a small subset of pool, disclosed in paper
- **17 pool** — used as ICL examples AND synthetic seeds (same set, different uses per condition)

Across the 12 currently hand-corrected refs (audio15-23, 25-27): 7 in TEST, 5 in POOL.
See `src/data_split.py` docstring for v2 provenance.

**Test set protection requirement:** the 18 test conversation IDs must be hard-fenced in code. They cannot appear as:
- ICL few-shot examples
- Synthetic generation seeds
- Fine-tuning training data
- Manual prompt iteration material

---

## Experimental design (locked)

Seven conditions, framed as a **low-resource comparison**: with scarce in-domain RO data, which method extracts the note best? Two API models bracket the range (Gemini Flash floor, Claude Opus ceiling); a single fixed open-source small model carries the prompting-vs-fine-tuning comparison.

| # | Condition | Base model | Training data | Research question |
|---|-----------|-----------|---------------|-------------------|
| 1 | Zero-shot prompting | Gemini Flash (cheap API) | none | Cheap API floor |
| 2 | Zero-shot prompting | Claude Opus (big API) | none | Ceiling (NOT a parity claim) |
| 3 | Few-shot ICL with real examples | Open-source small (fixed) | 3-5 examples in prompt | Does ICL help the small model? |
| 4 | Few-shot ICL with real examples | Claude Opus (big API) | 3 examples in prompt | ICL ceiling |
| 4b | Few-shot ICL with real examples | Gemini Flash (cheap API) | 3 examples in prompt | Does ICL help the cheap floor? (added 2026-06-23 — Gemini ZS was unexpectedly strong) |
| 5 | Fine-tuned on synthetic chiropractor data | Open-source small (fixed) | LLM-generated RO chiropractor pairs | Main deployment candidate |
| 6 | Fine-tuned on translated MTS | Open-source small (fixed) | ~1700 translated MTS pairs | Does general medical pretraining help? |
| 7 | Fine-tuned on MTS → then synthetic chiropractor | Open-source small (fixed) | MTS first, then synthetic | Stacked domain adaptation |

Open-source small candidates under evaluation: Qwen3-4B / RoLlama3-8B / Gemma3-4B (pick not yet locked).

**Cross-condition consistency requirements:**
- Conditions 3, 5, 6, 7 share the same fixed open-source small base model (cond. 1 is a separate cheap API, Gemini Flash)
- Conditions 2 and 4 share the same big API model (Claude Opus)
- Conditions 5 and 7 use the same synthetic data (don't regenerate between conditions)
- ICL conditions (3, 4, 4b) use the same N examples and the same selection strategy
- **ICL example set LOCKED (2026-06-23): N=3, `audio18, audio19, audio1`** (all POOL, never test). Source of truth: `src/ICL/real_examples_manifest.tsv` (rows with `include_default=1` + `status=ready`); `build_real_icl_examples.build_examples()` reads it and asserts no test leakage. To change the set, edit only the `include_default` column. The `fix_before_use` pool pairs (audio4/10/21/24/26/29) have documented-wrong raw refs — do NOT add them as ICL examples.

---

## Reference work

### Foundational

- **MEDIQA-Chat 2023 shared task** (Ben Abacha et al., ACL ClinicalNLP 2023) — defines doctor-patient dialogue → clinical note task formulation. Includes MTS-Dialog dataset. https://github.com/abachaa/MEDIQA-Chat-2023
- **WangLab @ MEDIQA-Chat 2023** (Giorgi et al., arXiv:2305.02220) — GPT-4 ICL beat fine-tuned models and matched human-written notes in expert evaluation. Key precedent for our ICL-vs-fine-tuning comparison.
- **GersteinLab @ MEDIQA-Chat 2023** (Tang et al., arXiv:2305.05001) — compared fine-tuning and ICL pipelines.

### Newer / closer to our task

- **K-SOAP** (Li et al., PAKDD 2025) — keyword-augmented SOAP format guides LLMs to generate more structured notes. Relevant to our strict-format requirement.
- **NoteChat** (arXiv:2310.15959) — multi-agent synthetic dialogue generation from clinical notes.
- **SynDial** (arXiv:2408.06285) — iterative LLM with feedback loop for synthetic dialogue generation. Relevant if condition 5/7 synthetic generation needs more rigor.

### Romanian / low-resource

- **MedQARo / RoMedQA** (Rogoz, Ionescu et al., 2025) — first large-scale Romanian medical QA benchmark (~100k QA pairs). Key finding: **fine-tuned smaller LLMs beat GPT-5.2 and Gemini 3 Flash on Romanian medical text.** Strongest evidence for our low-resource framing. NOTE: results are NOT directly comparable to ours — MedQARo is extractive QA, ours is multi-field JSON note generation. Cite as motivation, not as a baseline.

### Caveats from the field

- Translated medical datasets often produce stilted phrasing that doesn't match native clinical writing. The MedQARo team explicitly avoided translation. We're using translated MTS anyway, but should flag this as a known concern.
- Synthetic-data fine-tuning risks distilling teacher errors. Need manual quality gate (committed).

---

## Annotation guideline (status: to write, day 1)

Short doc for chiropractors before they create reference notes. Must cover:

1. **The "only-if-spoken" rule** with concrete Romanian examples.
2. **Section disambiguation** — what goes in `motivul_prezentarii` vs `evaluare_functionala_initiala`, etc.
3. **VAS rule** — only fill if a number is verbalized. "Doare rău" stays empty.
4. **Multi-select rules** — patient confirming a therapist's question ("Aveți hipertensiune?" "Da") counts as spoken.
5. **Medication rule** — name without dose is acceptable; do not invent doses.
6. **Edge cases** — to be collected as they emerge during annotation.

---

## Evaluation framework (status: to lock, phase 1)

Per-field metrics, not a single aggregate:

| Field | Metric |
|-------|--------|
| `evaluarea_durerii_vas` | Exact match on canonical form (single int and `[int]` of length 1 compare equal; lists compared as multisets; null vs number distinct) |
| `localizarea_durerii` | Multi-label F1 with empty-set handling |
| `antecedente` | Multi-label F1 with empty-set handling |
| `medicatie_actuala` | TBD — exact match vs normalized name match |
| `motivul_prezentarii` | Manual ternary (0 / 0.5 / 1), mean per condition; Cohen's kappa on dual-rated subset. No automatic metric — see decision log 2026-06-16 (manual ternary) |
| `evaluare_functionala_initiala` | ROUGE-L + BERTScore |

**Critical: "correct empty" handling.** A metric that only scores filled fields silently rewards hallucination. Need explicit handling: if both ref and pred are empty, that's a correct prediction and counts toward score.

**Aggregation:** keep per-section breakdowns in the paper. One headline number hides the story.

**Human evaluation:** light qualitative chiropractor review on a sample of outputs. Not formal Likert-scale, not formal inter-rater agreement — just "does this look right." Used to support quantitative findings, not replace them.

---

## Locked decisions

1. Philosophy: "if not spoken, leave empty" — enforced everywhere
2. Output format: 6 fields from the Osteopath Concept template (4 structured, 2 free-text)
3. Data split: 18 test / 0 dev / 17 pool (v2 on 2026-06-04; was 15/0/15 in v1)
4. 7 experimental conditions as listed above
5. Low-resource framing: scarce in-domain RO data is the central problem; which method wins under that constraint. Cost is a SECONDARY justification only (reframed 2026-06-21; cost diffs < ~$50/yr at clinic volume, too small to anchor the paper). Gemini Flash = cheap floor, Claude Opus = ceiling (not a parity claim).
6. Manual quality gate on synthetic data — committed
7. Paper venue: university course (not for publication) — small N is acceptable
8. Same small base model across all fine-tuning conditions
9. Same big model across conditions 2 and 4
10. Test set hard-fenced — no leakage into ICL examples, synthetic seeds, or training data

## Open decisions (still to close)

1. Compute setup (Colab free vs paid, GPU specs)
2. Open-source small model pick — narrowed to Qwen3-4B / RoLlama3-8B / Gemma3-4B (research in progress)
3. API models — DECIDED: Gemini Flash (cond. 1 floor), Claude Opus (cond. 2/4 ceiling)
4. Exact metrics per field (especially medication normalization)
5. Number of few-shot examples in ICL conditions — DECIDED 2026-06-23: N=3, locked set `audio18, audio19, audio1` (see condition table notes + `src/ICL/real_examples_manifest.tsv`)
6. Synthetic data volume to generate
7. Whether `evaluare_functionala_initiala` is mostly verbalized (verify with chiropractors)

## Decisions parked

1. VAS edge cases — vague descriptions, threshold for inference (stay empty per locked rule, but corner cases TBD)
2. Medication normalization rules
3. Production deployment beyond MVP

---

## Parallelism map

**Parallelizable:**
- Annotation guideline writing ‖ chiropractor annotation (after guideline delivered)
- Prompt + parser work ‖ eval script work
- Synthetic data generation ‖ fine-tuning infra setup
- ICL conditions (1-4) ‖ fine-tuning conditions (5-7), once infra is up
- Writing related-work section ‖ running experiments
- Chiropractor sanity check ‖ paper writing (final phase)

**Strictly sequential:**
- Template lock → annotation guideline → chiropractor annotation → test set frozen → any evaluation
- Eval framework decisions → eval script → running any condition
- Synthetic seeds chosen → synthetic data generated → conditions 5 and 7 trained
- Condition 6 (MTS fine-tune) → condition 7 (MTS + synthetic) — condition 7 starts from condition 6's checkpoint
- All conditions evaluated → paper results section

---

## Risks (tracked)

| Risk | Mitigation |
|------|------------|
| Synthetic data quality | Professional manual review gate (committed) |
| Fine-tuning silent failure (loss looks fine, outputs garbage) | Sanity check: valid JSON in correct schema before trusting metrics |
| Test set leakage | Hard-fence test IDs in code; assertion before training/ICL example selection |
| Schedule slip on fine-tuning | If conditions 5-7 don't produce valid output by end of phase 3, ship with ICL only |
| Chiropractor delays | Daily check-ins on both annotation and additional 20 conversations |
| Translated MTS produces non-native Romanian phrasing | Disclosed as known limitation; result is data point not just confound |
| Small N (15 test) gives wide CIs | Acceptable for course paper; report directional findings, not significance claims |

---

## What's out of scope

- Audio capture / VAD / speaker diarization (WhisperX pipeline already built, not the focus)
- Production deployment infrastructure beyond an MVP demo
- Multi-turn conversational memory
- ICD code mapping, billing, insurance
- Patient-facing UI
- Real-time / streaming generation

---

## When helping with this project

- **Coding tasks:** keep things small and verified. Default to Python. Test on small examples before scaling.
- **Research tasks:** verify recency, especially anything about Romanian models or 2024-2026 medical NLP work.
- **Planning tasks:** push back on scope creep. Two weeks, two people. Every new idea must justify displacing an existing one.
- **Writing tasks:** match the user's voice — direct, no fluff, skeptical-by-default.
- **Don't suggest things outside the locked scope** unless flagging an actual risk.

---

## Decision log

### 2026-06-21 — Reframe: cost-efficiency → low-resource study

Headline framing schimbat. Anterior paper-ul era un **cost-efficiency study**
("cât de ieftin aproximăm un LLM scump"). Abandonat ca headline: diferențele
de cost între modelele candidate sunt prea mici ca să susțină claim-ul
(sub ~$50/an la volumul unei clinici). Costul rămâne în paper ca justificare
**secundară**, completat după research-ul de modele.

Noul framing: **low-resource problem**. Întrebarea centrală: când ai date
in-domain puține pe o limbă sub-reprezentată (RO), ce metodă extrage cel mai
bine nota clinică structurată? Comparăm prompting vs fine-tuning sub această
constrângere.

Titlu fixat: *Structured Clinical Note Extraction from Romanian
Chiropractor-Patient Conversations: A Low-Resource Comparison*.

Model picks pentru API: DECISE — Gemini Flash (cond. 1, floor ieftin),
Claude Opus (cond. 2/4, ceiling — NU claim de paritate). Open-source small
îngustat la Qwen3-4B / RoLlama3-8B / Gemma3-4B (research în curs).

Capcane de framing (de evitat în intro): nu scrie "cost-efficiency" /
"cheap deployment" ca scop principal; nu reduce la "zero-shot vs few-shot"
(fine-tuning 5-7 e jumătate din experiment); nu spune "matches GPT-4/Opus";
rezultatele NU sunt direct comparabile cu MedQARo (QA extractiv vs JSON
multi-câmp).

Status experimente la data reframe-ului: NErulate. Doar
`data/chiropractor_ro/predictions/mock_test/audio18.json` există (mock).
Vezi `PAPER_BRIEF.md` pentru ce e scriibil acum vs ce așteaptă rezultate.

Modificat: AGENTS.md (project identity, experimental design, locked decisions,
open decisions, related work caveat), PAPER_BRIEF.md (nou).

### 2026-06-16 — `evaluare_functionala_initiala` include diagnostic și obiective

Schema și prompt-ul defineau anterior câmpul îngust ("postură, biomecanică,
mobilitate"). În ground truth-ul livrat de chiropracticieni, câmpul conține
în mod consistent și diagnosticul verbalizat și obiectivele terapeutice
verbalizate. Decizia v1 din TEAMMATE_HANDOFF de a scoate
`Diagnostic` / `Obiective_terapeutice` ca scope era cosmetic — chiroii le
scriu oricum aici.

Aliniem schema, SYSTEM_PROMPT și documentația cu practica reală a
chiropracticienilor. Regula "if not spoken, leave empty" rămâne intactă —
nu schimbăm decât tipul de informație acceptat, nu și regula de extragere.

Modificat: `src/SOTA_EVALUATION/json_schema.py` (description),
`src/SOTA_EVALUATION/claude_zero_shot.py` (SYSTEM_PROMPT section),
`data/mts_dialog_ro_augmented/HANDOFF.md` (note added).

### 2026-06-16 — motivul_prezentarii: drop ROUGE-L, keep BERTScore only

Ground truth-ul livrat de chiropracticieni pentru motivul_prezentarii are
variabilitate naturală mare de lungime (de la 2 cuvinte la 2 propoziții),
reflectând practica reală: uneori scriu telegrafic ("cot dr."), uneori
descriptiv. Această variabilitate e reală și nu se poate forța să dispară
fără ground truth artificial.

ROUGE-L e lexical și penalizează diferența de lungime între ref și pred
chiar când extracția e semantic corectă. Drop ROUGE-L pentru acest câmp.

BERTScore (encoder RO `dumitrescustefan/bert-base-romanian-cased-v1`)
rămâne ca headline singur. Tolerează parafrazarea și e mai puțin sensibil
la diferența de lungime, deși nu insensibil — limitare declarată în paper.

Prompt-ul NU se schimbă. Modelul are voie să producă output de orice
lungime plauzibilă.

Modificat: EVAL_DECISIONS.md (secțiunea motivul_prezentarii și tabel
recapitulare).

> **Superseded by entry below (same day):** BERTScore testat empiric pe
> ground truth-ul real (text dominant 2-5 tokeni) → gap între similar/different
> sub 0.05 (zgomot). Drop BERTScore și el. Vezi entry-ul următor.

### 2026-06-16 — motivul_prezentarii: manual ternary scoring, no automatic metric

Testat empiric BERTScore (dumitrescustefan/bert-base-romanian-cased-v1)
pe text românesc scurt:
- 3-4 tokeni: similar 0.47 vs different 0.47 (gap 0.00, zgomot pur)
- ~6 tokeni: similar 0.67 vs different 0.56 (gap 0.11)
- ~13 tokeni: similar 0.73 vs different 0.56 (gap 0.17)

Ground truth chiropractor e dominant 2-5 cuvinte (audio18: "cot dr.",
audio19: "durere lombara"). ROUGE-L la fel de slab pe astfel de text.

Decizie: NU se calculează metric automată. Înlocuit cu manual ternary
scoring (0 / 0.5 / 1) pe toate cele 7 condiții × 15 conv = 105 ratings.
Dual-rated de cei doi co-autori pe 2 condiții reprezentative (Cond 1
+ Cond 5, 30 ratings) pentru Cohen's kappa. Restul single-rated,
~75 împărțite. Blind to source condition prin randomization.

Rubrica:
- 1.0: pred captează acuza și zona; nu inventează
- 0.5: zona corectă, lipsește/inventează un detaliu
- 0.0: ratează acuza/zona, halucinează masiv, sau asimetric empty
- both null → 1.0

Pipeline-ul de scoring manual (randomization + rater UI + IRA aggregator)
se construiește după ce sunt pred-urile produse.

Modificat: EVAL_DECISIONS.md (secțiunea 5 rescrisă, tabel update).
