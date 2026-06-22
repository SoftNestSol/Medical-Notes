# Fine-Tuning Experiments

Colab entry point:

https://colab.research.google.com/github/SoftNestSol/Medical-Notes/blob/main/src/FT/ft_colab_bootstrap.ipynb

This folder is for conditions 5-7:

- synthetic chiropractor fine-tuning
- translated MTS fine-tuning
- MTS followed by synthetic chiropractor fine-tuning

Rules:

- Never train on `TEST_IDS` from `src/data_split.py`.
- Keep notebooks as execution surfaces only; split logic, schema, prompts, and reusable data prep should stay in repo code.
- First cloud run should only validate GPU, repo checkout, dataset paths, JSON schema, and leakage guards.
