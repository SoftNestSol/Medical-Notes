# MTS Dialog Romanian Augmentation Handoff

Date: 2026-05-22

## Progress

- Source datasets:
  - `data/mts_dialog/training_set.csv`
  - `data/mts_dialog/validation_set.csv`
  - `data/mts_dialog/test_set.csv`
- Output folder: `data/mts_dialog_ro_augmented/`
- Final outputs:
  - `training_set_ro_augmented.csv`
  - `validation_set_ro_augmented.csv`
  - `test_set_ro_augmented.csv`
- Completed training rows: `0` through `1200`, inclusive
- Completed validation rows: `0` through `99`, inclusive
- Completed test rows: `0` through `199`, inclusive
- Training total rows: `1201`
- Validation total rows: `100`
- Test total rows: `200`
- Status: training, validation, and `test_set.csv` complete; `test_set2.csv` pending
- Bulk shard CSVs and sample CSV were consolidated into the final output and deleted.

## Schema

```text
ID,section_header,section_text,dialogue
```

## Validation Command

```bash
python - <<'PY'
import csv
from pathlib import Path

p = Path('data/mts_dialog_ro_augmented/training_set_ro_augmented.csv')
with p.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

ids = [int(r['ID']) for r in rows]
id_set = set(ids)
missing = [i for i in range(1201) if i not in id_set]
dups = sorted({i for i in ids if ids.count(i) > 1})
empty_fields = sum(
    any(not r.get(k) for k in ['ID', 'section_header', 'section_text', 'dialogue'])
    for r in rows
)

print('rows', len(rows))
print('ids', min(ids), max(ids))
print('header', list(rows[0].keys()))
print('missing', len(missing))
print('dups', len(dups))
print('empty_fields', empty_fields)
PY
```

## Schema scope note (2026-06-16)

Diagnostic and Obiective terapeutice are absorbed into
`evaluare_functionala_initiala` (in practice, chiropractors write them there).
No separate `diagnostic` / `obiective_terapeutice` fields in v1. See AGENTS.md
decision log entry 2026-06-16.
