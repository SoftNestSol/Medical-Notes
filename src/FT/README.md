# Fine-Tuning Experiments

Colab entry point:

https://colab.research.google.com/github/SoftNestSol/Medical-Notes/blob/main/src/FT/ft_colab_bootstrap.ipynb

Training notebook:

https://colab.research.google.com/github/SoftNestSol/Medical-Notes/blob/main/src/FT/synthetic_claude_qlora_train.ipynb

First cloud run: set `SMOKE_TEST = True`, run end-to-end, and check that
`parse/schema sanity` is nonzero. Then switch `SMOKE_TEST = False` for the
actual condition 5 adapter.

This folder is for conditions 5-7:

- synthetic chiropractor fine-tuning
- translated MTS fine-tuning
- MTS followed by synthetic chiropractor fine-tuning

Synthetic Claude data flow:

```bash
.venv/bin/python scripts/generate_synthetic_claude_data.py --dry-run --n 1
ANTHROPIC_API_KEY=... .venv/bin/python scripts/generate_synthetic_claude_data.py --n 50 --workers 5
.venv/bin/python scripts/build_chiro_sft_jsonl.py \
  --synthetic-root data/synthetic/Claude \
  --out artifacts/ft/synthetic_claude_sft_messages.jsonl
```

The generator defaults to `claude-sonnet-4-6`. Override with `--model` or
`CLAUDE_MODEL` if the API model choice changes.

For ~500 synthetic examples, prefer 10 batches of 50 with rotated pool seeds
instead of one stylistically homogeneous run. Complete pool seed pairs currently
available: `audio1`, `audio4`, `audio10`, `audio18`, `audio19`, `audio21`,
`audio23`, `audio24`, `audio26`, `audio29`.

Example batch pattern:

```bash
.venv/bin/python scripts/generate_synthetic_claude_data.py --start 1 --n 50 \
  --output-root data/synthetic/Claude_500 --workers 5 \
  --seed-id audio18 --seed-id audio19 --seed-id audio21 --seed-id audio23 --seed-id audio26

.venv/bin/python scripts/generate_synthetic_claude_data.py --start 51 --n 50 \
  --output-root data/synthetic/Claude_500 --workers 5 \
  --seed-id audio1 --seed-id audio4 --seed-id audio10 --seed-id audio24 --seed-id audio29

.venv/bin/python scripts/generate_synthetic_claude_data.py --start 101 --n 50 \
  --output-root data/synthetic/Claude_500 --workers 5 \
  --seed-id audio10 --seed-id audio18 --seed-id audio23 --seed-id audio24 --seed-id audio29
```

Continue with `--start 151`, `201`, ... until `451`. The generator expands the
50-row control plan into unique IDs such as `synth_051`, `synth_101`, etc.,
while preserving the structured-value coverage plan.

`--workers` controls concurrent independent API calls. Each job owns one
conversation ID and writes only `synth_NNN.*`; `index.tsv` and reports are
rebuilt after all jobs finish. Use `--workers 3-5` first; raise toward 10 only
if Anthropic rate limits and local validation failure rate stay acceptable.

Rules:

- Never train on `TEST_IDS` from `src/data_split.py`.
- Keep notebooks as execution surfaces only; split logic, schema, prompts, and reusable data prep should stay in repo code.
- First cloud run should only validate GPU, repo checkout, dataset paths, JSON schema, and leakage guards.
