# MTS Dialog Romanian Augmentation

This folder contains Romanian translated and augmented MTS-Dialog splits.

## Files

- `training_set_ro_augmented.csv`: combined augmented training dataset for source rows `0-1200` from `data/mts_dialog/training_set.csv`.
- `validation_set_ro_augmented.csv`: augmented validation dataset for source rows `0-99` from `data/mts_dialog/validation_set.csv`.
- `test_set_ro_augmented.csv`: augmented test dataset for source rows `0-199` from `data/mts_dialog/test_set.csv`.

## Schema

```text
ID,section_header,section_text,dialogue
```

## Validation

Latest validation for `training_set_ro_augmented.csv`:

- rows: `1201`
- IDs: `0-1200`
- missing IDs: `0`
- duplicate IDs: `0`
- empty required fields: `0`

Latest validation for `validation_set_ro_augmented.csv`:

- rows: `100`
- IDs: `0-99`
- missing IDs: `0`
- duplicate IDs: `0`
- empty required fields: `0`

Latest validation for `test_set_ro_augmented.csv`:

- rows: `200`
- IDs: `0-199`
- missing IDs: `0`
- duplicate IDs: `0`
- empty required fields: `0`

The former per-range bulk CSV shards and sample CSV were consolidated into the single combined file and removed.
