# Manual scoring for `motivul_prezentarii`

Per decizia din 2026-06-16 (vezi `AGENTS.md` decision log), câmpul
`motivul_prezentarii` nu are metric automată. Toate cele 6 condiții finale
(1, 2, 3, 4, 5a, 5b — vezi `AGENTS.md` decision log 2026-06-23) × 15
conversații = 90 ratings se fac manual cu rubrica ternară (0 / 0.5 / 1),
blind to source condition.

Acest document e checklist-ul de la zero la "tabel de comparație între
condiții" în paper.

---

## 0. Preconditii

- [ ] Toate cele 6 condiții finale au pred-uri generate sub
      `data/chiropractor_ro/predictions/<condition>/<audio_id>.json`.
      Actual (2026-06-23): `cond1_gemini_zeroshot`, `cond2_claude_zeroshot`,
      `cond4_claude_fewshot`, `cond5_qwen` (cond 5a), `cond5_RoLlama`
      (cond 5b). Lipsă: cond 3 (RoLlama + ICL).
- [ ] Refs în `data/chiropractor_ro/refs/` pentru toate ID-urile din
      `TEST_COMPLETE_PAIR_IDS` (15 IDs).
- [ ] Cei doi raters au câte un identificator scurt (ex: `radu`, `andrei`).
- [ ] Au ales **2 condiții reprezentative** pentru dual-rating →
      Cohen's kappa. Default:
      - `cond1_gemini_zeroshot` (cheap floor)
      - `cond5_qwen` sau `cond5_RoLlama` (main deployment candidates).
        Recomandat: `cond1_gemini_zeroshot,cond5_qwen` (cheap floor vs compact FT deployment candidate).

---

## 1. Setup minim

Nimic de instalat suplimentar. Scriptul rulează cu venv-ul deja existent
(`src/STT/.venv/bin/python`).

Identificatori folosiți mai jos:
- `RATER_A` = numele primului rater (ex. `radu`)
- `RATER_B` = numele celui de-al doilea (ex. `andrei`)
- `DUAL_CONDS` = condițiile dual-rated, comma-separated
  (ex. `cond1_gemini_zeroshot,cond5_qwen`)

---

## 2. Rundă dual-rated (pentru Cohen's kappa)

Ambii raters scorează **independent** aceleași 2 condiții × 15 conversații
= 30 ratings fiecare. Nu vorbesc între ei despre scoruri până la final.

### Rater A
```bash
.venv/bin/python src/SOTA_EVALUATION/manual_score_motivul.py \
    --rater $RATER_A \
    --conditions $DUAL_CONDS
```

### Rater B (la fel)
```bash
.venv/bin/python src/SOTA_EVALUATION/manual_score_motivul.py \
    --rater $RATER_B \
    --conditions $DUAL_CONDS
```

Ce trebuie să știe fiecare rater:
- Vede doar `REF`, `PRED`, și un `blind_id` (ex. `item_003`).
  Nu vede condiția sau audio_id. Ordinea e shuffled per-rater.
- Rubrica e afișată la fiecare item. `?` o reafișează detaliat.
- `s` = skip fără salvare (revine data viitoare). `q` = quit, progresul
  e salvat pe disc.
- Output: `data/chiropractor_ro/manual_scores/motivul_<rater>.csv`.

---

## 3. Rundă single-rated (restul de 4 condiții × 15 = 60 ratings)

Cele 4 condiții rămase se împart între cei doi raters. O posibilitate:
2 condiții la fiecare. Decizia practică, nu rigidă.

### Rater A
```bash
.venv/bin/python src/SOTA_EVALUATION/manual_score_motivul.py \
    --rater $RATER_A \
    --conditions cond2_claude_zeroshot,cond3_xxx,cond5_RoLlama
```

### Rater B
```bash
.venv/bin/python src/SOTA_EVALUATION/manual_score_motivul.py \
    --rater $RATER_B \
    --conditions cond4_claude_fewshot
```

Resume-safe: dacă un rater oprește la mijloc, relansează aceeași comandă —
sare peste tot ce-a scorat deja.

---

## 4. Sanity check după scoring

```bash
.venv/bin/python -c "
import csv, sys
from collections import Counter
for r in ['$RATER_A', '$RATER_B']:
    p = f'data/chiropractor_ro/manual_scores/motivul_{r}.csv'
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    by_cond = Counter(row['condition'] for row in rows)
    print(f'rater={r}: total={len(rows)}')
    for cond, n in sorted(by_cond.items()):
        print(f'  {cond}: {n}')
"
```

Verifică:
- Total raters A + B = 90 + 30 (dual) = 120 ratings.
- Fiecare condiție acoperită complet de cel puțin un rater (15 ratings).
- Cele 2 condiții dual au 15 ratings de la fiecare.

---

## 5. Agregare: mean per condiție + Cohen's kappa

### Mean per condiție (headline metric pentru paper)

```bash
.venv/bin/python <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

base = Path("data/chiropractor_ro/manual_scores")
all_rows = []
for csv_path in base.glob("motivul_*.csv"):
    all_rows.extend(csv.DictReader(open(csv_path, encoding="utf-8")))

# Mean per condition. Dual-rated items: average across raters per (cond, audio_id).
by_pair = defaultdict(list)  # (cond, audio_id) -> [scores]
for r in all_rows:
    by_pair[(r["condition"], r["audio_id"])].append(float(r["score"]))

per_cond = defaultdict(list)
for (cond, _audio), scores in by_pair.items():
    per_cond[cond].append(sum(scores) / len(scores))

print(f'{"condition":40s}  n   mean')
print("-" * 60)
for cond, vals in sorted(per_cond.items()):
    print(f'{cond:40s}  {len(vals):2d}   {sum(vals)/len(vals):.3f}')
PY
```

Output ăsta e tabelul direct pentru paper: o linie per condiție, mean +/- N.

### Cohen's kappa pe subsetul dual-rated

```bash
.venv/bin/pip install scikit-learn -q
.venv/bin/python <<'PY'
import csv
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

RATER_A = "$RATER_A"  # înlocuiește
RATER_B = "$RATER_B"

base = Path("data/chiropractor_ro/manual_scores")
rows_a = list(csv.DictReader(open(base / f"motivul_{RATER_A}.csv", encoding="utf-8")))
rows_b = list(csv.DictReader(open(base / f"motivul_{RATER_B}.csv", encoding="utf-8")))

score_a = {(r["condition"], r["audio_id"]): r["score"] for r in rows_a}
score_b = {(r["condition"], r["audio_id"]): r["score"] for r in rows_b}

shared = sorted(set(score_a) & set(score_b))
print(f"dual-rated items: {len(shared)}")
if not shared:
    raise SystemExit("no shared items — did both raters cover the same conditions?")

a = [score_a[k] for k in shared]
b = [score_b[k] for k in shared]

# Cohen's kappa on the 3 categorical labels {0.0, 0.5, 1.0}.
kappa = cohen_kappa_score(a, b)
print(f"Cohen's kappa: {kappa:.3f}")

# Agreement breakdown
exact = sum(x == y for x, y in zip(a, b))
print(f"exact agreement: {exact}/{len(shared)} = {exact/len(shared):.1%}")
PY
```

Interpretare brută:
- κ < 0.20 → slab. Probabil rubrica nu e clară. Discuție între raters, refac.
- 0.20–0.40 → fair.
- 0.40–0.60 → moderate.
- 0.60–0.80 → substantial. OK pentru paper.
- > 0.80 → almost perfect.

---

## 6. Tabel final de comparație în paper

Output-ul din pasul 5 (mean per condiție) → tabel direct. Adaugă coloana
κ ca footnote pe condițiile dual-rated.

Exemplu format paper-ready:

```
Condition                          motivul (manual ternary)
----------------------------------------------------------------
cond1_gemini_zeroshot              0.62  (κ=0.71, dual-rated)
cond2_claude_zeroshot              0.78
cond3_*                            0.55
cond4_claude_fewshot               0.71
cond5_qwen                         0.83  (κ=0.71, dual-rated)
cond5_RoLlama                      0.66
```

Combinat cu metrici automate din `eval.py` summary.json pentru celelalte
6 câmpuri, ai tabelul complet de comparație între condiții.

---

## 7. Limitări declarate în paper

- N=15 per condiție e mic. CI sunt largi; diferențele sub ~0.07 între
  condiții nu sunt distinguishable.
- Doar 2 condiții dual-rated. κ măsoară agreement, nu corectitudine
  absolută. Raters sunt engineers, nu clinicieni.
- Blind randomization elimină bias de condiție per rater, dar nu bias-ul
  comun (ambii raters pot sub/supraevalua un tipar specific).
- Decizia de scor manual e justificată empiric: BERTScore pe text scurt
  (median 2-5 tokeni în ground truth) returnează gap < 0.05 între perechi
  similar/different — zgomot pur. Vezi decision log 2026-06-16.

---

## Quick reference

| Pas | Cine | Comandă |
|-----|------|---------|
| Dual-rate cond1+cond5_qwen | A & B independent | `python manual_score_motivul.py --rater <X> --conditions cond1_gemini_zeroshot,cond5_qwen` |
| Single-rate restul | A & B împart | `python manual_score_motivul.py --rater <X> --conditions <other>` |
| Mean per condiție | oricine | scriptul de la pas 5 |
| Cohen's κ | oricine | scriptul de la pas 5 |

Output final: un CSV per rater + un tabel mean / condiție + un număr κ.
Asta e tot ce trebuie pentru secțiunea de eval în paper pentru
`motivul_prezentarii`.
