# Gamma Presentation Package

> Two parts. **Part A** is the deck — paste it straight into Gamma's
> "Paste in text" import. **Part B** is your speaker script with a
> 10-minute clock (do NOT paste into Gamma; it's for you).
> All numbers are verified against the final results.

## How to feed Gamma

1. Gamma → Create → **Paste in text** → paste everything in Part A only.
2. Settings: **12 cards**, format *Presentation*, tone *Professional / academic*,
   "preserve my text" (don't let it rewrite your numbers).
3. Theme: clean, light, one accent color. Avoid stock-photo clutter — this is
   a data talk. Let the one chart (Slide 8) carry the visual weight.
4. **CRITICAL — Gamma fabricates chart numbers.** In a prior render it invented
   a bar chart with values that exist nowhere in this deck, the paper, or the
   eval (e.g. it showed Gemini localizare 0.643 / antecedente 0.667 — the real
   values are 0.683 / 0.833). Do **NOT** let Gamma auto-generate any chart.
   - Turn OFF "generate visuals / charts" for the import.
   - For Slide 8, **insert the finished image** `artifacts/headline_localizare.png`
     manually — do not let Gamma redraw it.
   - After generation, diff **every number** on the slides against Table 2 below.
     If Gamma changed or "rounded" any, fix it by hand. The numbers are final.
5. Gamma also paraphrases and drops hedges. Re-check that the conclusion slide
   still says "directional, N=15" and "specific to structured fields," and does
   NOT say fine-tuning "outperforms" or "closes the gap" without those qualifiers.

### Locked numbers — source of truth (do not let any tool alter these)

| Condition | Localizare F1 | Antecedente F1 | VAS | Med no-halluc | EvalFunc BERT | motivul |
|---|---|---|---|---|---|---|
| Gemini ZS | 0.683 | 0.833 | 0.533 | 12/14 | 0.465 | 0.517 |
| Claude ZS | 0.698 | 0.833 | 0.533 | 12/14 | 0.542 | 0.600 |
| RoLlama ICL | 0.148 | 0.000 | 0.333 | 14/14† | 0.512 | 0.533 |
| Claude FS | 0.634 | 0.769 | 0.533 | 12/14 | 0.657 | 0.567 |
| RoLlama FT | 0.638 | 0.545 | 0.533 | 11/14 | 0.679 | 0.567 |
| Qwen FT | 0.511 | 0.625 | 0.467 | 11/14 | 0.577 | 0.367 |

†14/14 is degeneracy (emitted no meds), not best-in-class. Medication catch = 0/1 for all. κ=0.519.

---

# PART A — DECK CONTENT (paste into Gamma)

---

# Prompting or Fine-Tuning?
### Structured Clinical Note Extraction from Romanian Chiropractor–Patient Dialogue under Data Scarcity

Caracas Radu-Nicolae · Iordache Andrei-Tudor — University of Bucharest

*A low-resource comparison: when in-domain data is scarce, which method writes the better clinical note?*

---

## The problem

- Every first chiropractor visit runs the same way: 5–8 standard questions → a **fixed intake form**, written by hand after the visit.
- The conversation is unstructured. The note is not — **the same fields, every time.**
- Modern LLMs make it plausible to fill that form directly from the transcript, leaving the clinician to review and sign.
- This is one component of a clinic management system we are building.

---

## The task — and the one hard rule

- **Input:** transcript of one consultation. **Output:** a real clinic's 6-field intake template.
- The schema is **not ours** — it's the clinic's, fixed.
- **The rule that governs everything:** *if it is not spoken, the field stays empty.*
- No inference, no clinical guesswork. **A missed mention and an invented one are equally wrong**; a correctly-empty field is a correct answer.
- A note that reads well but invents a symptom is worse than no note — a clinician may sign it.

---

## The real question: method, not cost

- Tempting framing: cost. Big proprietary APIs are accurate but priced for scale; small self-hosted models are cheap but uncertain on Romanian medical text.
- We checked — **cost doesn't separate the viable options.** At clinic volume, inference is cheap for every self-hostable route and the cheap API alike.
- What actually matters: **data governance** (self-hosting keeps GDPR Art. 9 health data off external APIs) and **quality after fine-tuning on scarce data**.
- → The real question: **prompting a large model vs fine-tuning a small one, under data scarcity.**

---

## Data

- **Mock consultations** (real audio blocked pending GDPR encryption): a chiropractor + a team member playing the patient.
- Transcribed by **self-hosted WhisperX**; models read **raw ASR**, not cleaned text — so the deployed pipeline, but ASR error is entangled with extraction error.
- **15 annotated test** consultations (from an 18-conversation frozen holdout) · **10 annotated pool** (of 19) · no dev set. Test set hard-fenced, used once.
- **6 target fields, very uneven density** — sparse fields give thin estimates:

| Field | Type | Non-empty (of 15) |
|---|---|---|
| motivul prezentării | short free text | 14 |
| VAS (pain score) | numeric | 10 |
| localizare durere | multi-label, 19 regions | 14 |
| antecedente | multi-label, 10 conditions | 3 |
| medicație | drug name + dose | 1 |
| evaluare funcțională | long free text | 15 |

---

## Six conditions: two method families

| # | Method | Base model | Role |
|---|---|---|---|
| 1 | Zero-shot | Gemini Flash | cheap API floor |
| 2 | Zero-shot | Claude Opus | quality ceiling |
| 3 | Few-shot ICL (N=3) | RoLlama3-8B | small-model ICL |
| 4 | Few-shot ICL (N=3) | Claude Opus | ICL ceiling |
| 5a | Fine-tune (synthetic) | Qwen3-4B | deployment candidate |
| 5b | Fine-tune (synthetic) | RoLlama3-8B | **same base as cond. 3** |

- **Core ablation:** cond 3 vs 5b share the *same* base (RoLlama3-8B) → isolates **adaptation method** from model choice.
- 5a vs 5b share *training data* → isolates **base model** from data.
- FT data = **344 synthetic pairs**, generated by Claude Sonnet, seeded from **10 real pool pairs** (never the test set), passed through a validation + de-biasing gate.

---

## Evaluation — built around the empty rule

- **Per field, never one aggregate** — fields are too heterogeneous; one number hides the story.
- **Symmetric empties:** correctly-empty = correct; wrongly-filled = error, just like a miss. A metric scoring only filled fields would silently reward hallucination.
- **Parse rate reported alongside accuracy** — a model that fails on its hard cases can look strong on the survivors.
- **Presenting complaint → human rating.** BERTScore/ROUGE-L are *noise* on 2–5-token Romanian phrases (similar-vs-different gap < 0.05). Replaced with a **manual ternary (0/0.5/1)**, 2 raters, blind to condition; Cohen's κ on a dual-rated subset.

---

## Headline result: the central ablation

**Same Romanian-native base (RoLlama3-8B). Only the adaptation method changes.**

> **Visual:** insert `artifacts/headline_localizare.png` here (a finished chart with
> verified values). Do NOT let Gamma generate its own bar chart for this slide.

| | Localizare F1 | Antecedente F1 | Clean JSON |
|---|---|---|---|
| Cond 3 — **ICL** | 0.148 | 0.000 | 26.7% |
| Cond 5b — **Fine-tuned** | **0.638** | **0.545** | **86.7%** |

- In-context prompting **collapses** on structured extraction for the small RO model.
- Fine-tuning on synthetic in-domain data **recovers** it — same base, so it's the *method*, not the model.
- We lean on **localizare** (dense: 14/15) for the claim; antecedente has only 3 positives.
- **This is the paper's central finding.**

---

## Fine-tuned small model vs the API ceiling

- Fine-tuned RoLlama (5b) reaches the **neighborhood** of zero-shot APIs on structured fields: localizare **0.638** vs Gemini 0.683 / Claude 0.698.
- On the long free-text assessment, 5b's BERTScore (**0.679**) is the **highest of any condition** — above few-shot Claude (0.657).
- Compact Qwen (5a) trails (localizare 0.511, BERTScore 0.577) but stays well clear of collapsed ICL.
- 5b > 5a, but the bases differ in **both size (8B vs 4B) and Romanian specialization** — we can't separate them. Say "Romanian-native base **and/or** larger size," not "Romanian pretraining wins."
- **Not a parity claim** — APIs are a ceiling. The reading: FT on synthetic in-domain data reaches the same neighborhood as a large model zero-shot.

---

## Two honesty checks

**The advantage is schema-specific.**
- On the free-text complaint, RoLlama ICL (0.533) and FT (0.567) are **tied** — even though they differ sharply on localizare (0.148 → 0.638).
- ICL failure is confined to **rigid-structure fields**. Free-text generation survives ICL.
- Honest claim: *fine-tuning wins where structure is required* — most of this template — **not** "fine-tuning wins."

**Medication is unscoreable here.**
- Exactly **1** drug-present reference in the test set. **No condition caught it (0/1 everywhere.)**
- Cond 3's "perfect" 14/14 no-hallucination is **degeneracy** — it emitted no drugs at all, so it couldn't hallucinate. Not best-in-class.

---

## Conclusion

- Framed as a **low-resource problem**: prompting vs fine-tuning on a real clinic template, strict leave-empty rule.
- **Fine-tuning a small RO-native model on synthetic in-domain data recovers structured extraction that ICL of the same base fails to produce** — and ties it on free text, locating the failure precisely in the schema-constrained fields.
- FT small models reach the **neighborhood** of zero-shot large APIs without matching them.
- Cost is a wash → the case for self-hosting rests on **data governance + post-FT quality**, not price.
- **Directional, not significant** (N=15). Next: real GDPR-compliant consultations, larger test set, medication with genuine positive cases.

---

# PART B — SPEAKER SCRIPT + 10-MINUTE CLOCK (do not paste into Gamma)

Target ≈ 9:30, leaving buffer. 12 slides. Numbers in **bold** are the ones to say out loud; everything else is support you can drop if time is short.

| # | Slide | Time | Land this one line |
|---|---|---|---|
| 1 | Title | 0:20 | "Scarce data, under-represented language — which method writes the better note?" |
| 2 | Problem | 0:40 | "Unstructured conversation, but the note is always the same fields." |
| 3 | Task + rule | 1:00 | "If it's not spoken, it stays empty — **omission and hallucination are equally wrong**." |
| 4 | Method not cost | 0:50 | "Cost doesn't separate the options; **method under scarcity** does." |
| 5 | Data | 1:00 | "**15 test, hard-fenced, used once** — and the fields are very uneven: medication has **1** positive case." |
| 6 | Six conditions | 1:00 | "Cond 3 and 5b share a base — that's the clean **ICL-vs-fine-tuning** test." |
| 7 | Evaluation | 0:50 | "Per field, **correct empties count**, and we dropped BERTScore for a human rating where it was noise." |
| 8 | Headline | 1:30 | "Same base: ICL **0.148**, fine-tuned **0.638**; JSON **27% → 87%**. It's the method." |
| 9 | vs ceiling | 0:50 | "Fine-tuned small model reaches the **neighborhood** of the big APIs — not parity." |
| 10 | Honesty checks | 0:50 | "The win is **schema-specific**; medication is unscoreable at 1 positive." |
| 11 | Conclusion | 0:50 | "Self-hosting wins on **governance + quality**, not price. Directional, N=15." |
| 12 | Q&A buffer | — | (hold) |

**Pacing rules**
- Spend your time on **Slides 5, 6, 8** — that's the spine. Everything else is setup.
- Slide 8 is the only slide you slow down on. Read the three numbers, pause, say "same base — so it's the method."
- If you're over time, cut Slide 9 to one sentence and merge Slide 10's two checks into "two caveats: the win is schema-specific, and medication has one case."

**Likely questions — short answers ready**
- *"Why N=15?"* → mock data pending GDPR; frozen holdout, used once; we report directional, not significant.
- *"Isn't synthetic data circular?"* → teacher-generated, so yes there's distillation risk — that's why the validation + de-biasing gate, and we flag it as a limitation.
- *"κ = 0.52 is only moderate."* → correct; it supports the coarse separation, not fine rankings — we say exactly that.
- *"Why is medication so bad?"* → it isn't scoreable: one drug-present reference. We report no-hallucination only and call the 14/14 what it is — degeneracy.
- *"Did you beat the big models?"* → No, and we don't claim to. They're a ceiling; the small fine-tuned model reaches their neighborhood.
