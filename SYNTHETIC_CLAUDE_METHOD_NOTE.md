# Synthetic Claude Data Generation - Method Note

This note summarizes how the Romanian chiropractor synthetic dataset was generated, so the method can be described consistently in the paper.

## Goal

We generated synthetic Romanian chiropractor-patient conversations paired with structured clinical notes for the fine-tuning conditions. The purpose was not to create new test data, but to create additional in-domain training examples under the low-resource setting.

The synthetic examples follow the same target format as the real Osteopath Concept references:

- `motivul_prezentarii`
- `evaluarea_durerii_vas`
- `localizarea_durerii`
- `localizarea_durerii_alta`
- `antecedente`
- `antecedente_altele`
- `medicatie_actuala`
- `evaluare_functionala_initiala`

The same project rule was enforced: if information is not spoken in the conversation, it must remain empty/null in the note.

## Generator

Synthetic data was generated with Anthropic Claude through `scripts/generate_synthetic_claude_data.py`.

Current model used by the script:

```text
claude-sonnet-4-6
```

The script outputs data under:

```text
data/synthetic/Claude/
```

For each accepted synthetic example, the script writes:

- `transcripts/synth_XXX.txt` - flat transcript with `SPEAKER_00` / `SPEAKER_01`
- `conversations/synth_XXX.json` - WhisperX-like segment JSON with approximate timestamps
- `refs/synth_XXX.json` - structured note
- `audits/synth_XXX.json` - evidence quotes and token usage
- `raw/synth_XXX*.txt` - raw Claude attempts

The final currently retained set has:

```text
344 valid synthetic pairs
```

All retained pairs passed schema and evidence validation after generation.

## Seed Examples

Claude was seeded with real Romanian chiropractor examples from the pool split, not from the protected test split.

The fixed seed pool used by the script is:

```text
audio1, audio4, audio10, audio18, audio19,
audio21, audio23, audio24, audio26, audio29
```

These seed IDs are complete real conversation-reference pairs from the pool. The script calls `assert_no_test_leakage(...)` before using them, so protected test conversations cannot be used as synthetic seeds.

The seed examples are inserted into the system prompt as full flattened transcripts plus corrected structured notes. They are used for two things:

1. Conversation style: Romanian spoken rhythm, short turns, confirmations, vague patient language, therapist-led questioning.
2. Mapping discipline: how colloquial spoken information should or should not populate each JSON field.

The prompt explicitly tells Claude that the seed examples are a mapping/style contract, not a license to copy the cases.

## Style Extraction

Before building the prompt, the script computes a small style profile from the selected seed conversations:

- number of segments
- total word count
- number of very short turns
- number of long turns
- speaker alternations
- observed oral markers such as `păi`, `adică`, `nu știu`, `parcă`, `ok`
- examples of short fragments

This measured profile is inserted into the system prompt. The goal is to preserve the mechanics of Romanian clinical conversation rather than produce polished medical prose.

## Prompt Structure

Claude is asked to return exactly one JSON object:

```json
{
  "conversation": "...",
  "note": {...},
  "evidence": {...}
}
```

Important prompt constraints:

- Only `SPEAKER_00` and `SPEAKER_01` are allowed.
- The therapist leads with short questions and clarifications.
- The patient should sound like a patient, not like a clinician.
- Functional evaluation should appear as spoken interaction, not as a written report read aloud.
- Every populated note field must have at least one exact evidence quote from the conversation.
- Empty fields must have empty evidence lists.
- No information may be inferred from clinical plausibility alone.

## Generation Plan

The plan is deterministic and controls broad form, not exact clinical labels.

For each synthetic ID, the script assigns:

- conversation length/register:
  - `scurt_rar`: 550-850 words
  - `detaliat_lung`: 800-1200 words
- patient style:
  - `tacut`
  - `vorbaret`
  - `vag`
  - `anxios`
- background profile:
  - mostly no relevant antecedents and no current medication
  - occasional OTC medication
  - occasional minor history without current medication
  - occasional non-hypertension chronic condition
  - rare hypertension-permitted cases
- region profile:
  - lombar
  - cervical/ceafă
  - sacral/coccis
  - toracal
  - genunchi
  - cot/pumn
  - șold

The background and region profiles were added after initial QA showed too many synthetic cases with hypertension, medication, and right shoulder pain. The final balancing run explicitly generated more non-shoulder cases.

## Seed Rotation and Parallel Generation

The script uses a fixed list of 10 seed examples, but each generation block sees only a sliding subset of 6 seeds.

```text
BLOCK_SIZE = 10
subset_size = 6
```

For each block of 10 synthetic IDs, the seed window advances by one position. This avoids using the exact same seed context for every generated example.

Parallel generation is supported with `--workers`. Jobs are submitted round-robin across seed blocks, so workers do not all start from the same seed subset. In practice:

- 40 workers hit Anthropic input-token/minute rate limits.
- 15 workers was stable.

The script also includes rate-limit backoff for 429/API rate limit errors.

## Validation

Each Claude attempt is accepted only if it passes all validation checks.

Validation includes:

1. The root object must contain exactly `conversation`, `note`, and `evidence`.
2. The note must pass the project parser and JSON schema.
3. `motivul_prezentarii` must be populated.
4. The transcript must use valid speaker labels.
5. Conversation length must be between 320 and 1300 words.
6. Each populated field must have evidence.
7. Each empty field must have no evidence.
8. Every evidence quote must be an exact substring of the generated conversation after normalization.
9. Medication names must be spoken in the conversation.
10. Medication doses must be supported by the conversation/evidence. Exact string match is accepted, but the validator also allows minor paraphrase/diacritic differences by checking important content tokens. Numeric dose tokens must still be present.

Invalid attempts are retried. If all attempts fail, the ID is left out of the retained dataset and logged in `_failures.tsv`.

## Manual QA and Cleanup

After generation, we scanned the retained refs for distributional problems.

Initial synthetic runs were schema-valid but had clear bias:

- too many cases with `hipertensiune_arteriala`
- too many current medications
- too many right shoulder complaints

We then changed the prompt/plan and generated additional balancing data. We also deleted 29 of the most biased old examples: cases combining hypertension, medication, right shoulder pain, and old prompt behavior.

Current retained set:

```text
344 valid pairs
```

Current QA snapshot:

- all retained pairs have refs/transcripts/audits/conversations
- full validation passed: no invalid retained pair
- no exact duplicate transcripts
- no near-duplicate transcripts above the checked shingle-overlap threshold
- word count range: approximately 320-1184 words
- median word count: approximately 563 words

Distribution after cleanup and balancing:

- `lombar`: 69
- `cervical`: 128
- `cap_ceafa`: 43
- `sacral_coccis`: 12
- `toracal`: 21
- `genunchi_dr`: 9
- `cot_dr`: 14
- `umar_dr`: 160
- `umar_stg`: 75
- `antecedente` empty: 191/344
- `medicatie_actuala` empty: 131/344
- `hipertensiune_arteriala`: 109/344

The last balancing batch was explicitly non-shoulder and added 44 valid examples:

- `lombar`: 13
- `cervical`: 10
- `sacral_coccis`: 10
- `cap_ceafa`: 9
- `toracal`: 5
- `genunchi_dr`: 4
- `sold_dr`: 3
- `pumn_dr`: 3
- `cot_dr`: 2
- no shoulder localizations in that batch
- 32/44 without antecedents
- 26/44 without medication
- hypertension only 3/44

## How to Describe This in the Paper

A concise paper description could be:

> We generated synthetic Romanian chiropractor-patient dialogue-note pairs using Claude Sonnet. The generator was seeded only with real in-domain examples from the development/pool split, never from the protected test set. Each prompt included full real conversation-note pairs, a measured dialogue-style profile extracted from the seed transcripts, the target JSON schema, and strict evidence requirements. Claude returned a synthetic transcript, a structured note, and field-level evidence quotes. Outputs were retained only if they passed schema validation, speaker-label validation, word-count constraints, exact evidence grounding, and medication support checks. We iteratively inspected aggregate distributions and removed or counterbalanced biased synthetic examples, especially early overproduction of hypertension, medication, and shoulder complaints.

Important caveat:

> The synthetic dataset is not treated as real clinical data. It is a controlled in-domain augmentation source for the fine-tuning condition, and we report its generation and validation procedure because teacher-model artifacts and distributional bias are known risks.

## Practical Recommendation

For the first serious fine-tuning run, use the current cleaned Claude set, but mention in the paper that a manual quality gate and distributional cleanup were applied. The set is good enough for the synthetic fine-tuning condition, but it should not be described as epidemiologically representative of the clinic.

