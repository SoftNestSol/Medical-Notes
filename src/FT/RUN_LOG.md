# Fine-Tuning Run Log

## 2026-06-23 — Condition 5 A100 QLoRA run

Purpose: train the open-source small model condition on Claude-generated synthetic Romanian chiropractor data.

Condition:

- Experimental condition: condition 5, synthetic chiropractor fine-tuning.
- Base model: `Qwen/Qwen3-4B-Instruct-2507`.
- Fine-tuning method: QLoRA / PEFT LoRA.
- Training notebook: `src/FT/synthetic_claude_qlora_train.ipynb`.
- Dataset builder: `scripts/build_chiro_sft_jsonl.py`.
- Synthetic source: `data/synthetic/Claude`.
- SFT rows available: 344 synthetic transcript/note pairs.
- Test data was not used for training. Synthetic seeds are checked by `assert_no_test_leakage`.

Hardware/profile:

- Colab paid GPU profile: A100.
- `HARDWARE_PROFILE = "A100"`.
- `MAX_SEQ_LENGTH = 4096`.
- Expected effect: keep the full synthetic training set after token-length filtering.

Hyperparameters:

- `NUM_TRAIN_EPOCHS = 2`
- `PER_DEVICE_BATCH_SIZE = 2`
- `GRADIENT_ACCUMULATION_STEPS = 4`
- Effective batch size: 8
- `LEARNING_RATE = 2e-4`
- Scheduler: cosine
- Warmup ratio: `0.03`
- Optimizer: `paged_adamw_8bit`
- `LORA_R = 16`
- `LORA_ALPHA = 32`
- `LORA_DROPOUT = 0.05`
- `USE_LIGER_KERNEL = True`
- `group_by_length = True`
- `length_column_name = "length"`
- Gradient checkpointing enabled with `use_reentrant=False`.

Observed training output:

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 19 | 0.543300 | 0.431537 |
| 38 | 0.427400 | 0.341189 |
| 57 | 0.309000 | 0.317134 |
| 76 | 0.285100 | 0.311623 |

Interpretation:

- Validation loss decreased and then stabilized.
- No obvious severe overfitting signal from the logged loss curve.
- This is still not a quality result by itself; the adapter must be evaluated on held-out real test conversations.

Artifact handling:

- Adapter is saved by the notebook under `artifacts/ft/runs/<RUN_NAME>/adapter`.
- The run should be exported with the notebook cells:
  - `Export manifest + adapter zip`
  - `Backup run artifacts to Google Drive`
- Prediction generation should use `src/SOTA_EVALUATION/condition_testing_colab.ipynb`.
- Evaluation should be run locally with `src/SOTA_EVALUATION/eval.py` after importing the prediction zip.

Notes:

- Earlier T4 runs were technically valid smoke/baseline runs but context-limited and not the preferred condition-5 artifact.
- The A100 run is the preferred condition-5 fine-tuned adapter unless held-out evaluation shows a failure mode.

## 2026-06-23 — RoLlama3 A100 QLoRA exploratory run

Purpose: exploratory model-selection run on the same synthetic Claude dataset, using a Romanian Llama-3 based model.

Condition:

- Experimental status: exploratory candidate run, not automatically a locked main condition.
- Base model: `OpenLLM-Ro/RoLlama3-8b-Instruct`.
- Fine-tuning method: QLoRA / PEFT LoRA.
- Training notebook: `src/FT/rollama3_synthetic_claude_qlora_train.ipynb`.
- Dataset builder: `scripts/build_chiro_sft_jsonl.py`.
- Synthetic source: `data/synthetic/Claude`.
- SFT rows available: 344 synthetic transcript/note pairs.
- Test data was not used for training.

Hardware/profile:

- Colab paid GPU profile: A100.
- `HARDWARE_PROFILE = "A100"`.
- `MAX_SEQ_LENGTH = 4096`.
- Expected effect: keep the full synthetic training set after token-length filtering.

Hyperparameters:

- `NUM_TRAIN_EPOCHS = 2`
- `PER_DEVICE_BATCH_SIZE = 1`
- `GRADIENT_ACCUMULATION_STEPS = 8`
- Effective batch size: 8
- `LEARNING_RATE = 2e-4`
- Scheduler: cosine
- Warmup ratio: `0.03`
- Optimizer: `paged_adamw_8bit`
- `LORA_R = 16`
- `LORA_ALPHA = 32`
- `LORA_DROPOUT = 0.05`
- `USE_LIGER_KERNEL = True` in notebook; RoLlama notebook attempts Liger Llama patch and continues if unavailable.
- `group_by_length = True`
- `length_column_name = "length"`
- Gradient checkpointing enabled with `use_reentrant=False`.

Observed training output:

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 19 | 0.444500 | 0.370952 |
| 38 | 0.378700 | 0.314224 |
| 57 | 0.251500 | 0.294606 |
| 76 | 0.221000 | 0.286393 |

Interpretation:

- Validation loss decreased across the run.
- Loss curve looks at least as healthy as the Qwen A100 run, but this is not enough to pick the model.
- Need held-out real test predictions and field-level evaluation before deciding whether RoLlama replaces Qwen as the fixed open-source base model.

Artifact handling:

- User reported the run was saved successfully.
- Adapter should be backed up under the Drive FT run folder via notebook export cells.

## 2026-06-23 — Qwen 8B model-selection caveat

Current lookup found:

- `Qwen/Qwen3-4B-Instruct-2507` exists and was the Qwen 4B FT base used here.
- `Qwen/Qwen3-8B` exists.
- No official `Qwen/Qwen3-8B-Instruct-2507` was found.

Implication:

- Do not describe a potential `Qwen/Qwen3-8B` run as the exact 8B version of
  `Qwen/Qwen3-4B-Instruct-2507`.
- It would confound parameter count with checkpoint/instruction-tuning lineage.
- For clean reporting, prefer evaluating the already trained Qwen 4B FT and
  RoLlama3 8B FT on the held-out real test set before spending another A100 run.
