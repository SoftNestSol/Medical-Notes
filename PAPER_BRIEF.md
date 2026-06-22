# Paper Brief — pentru coautor (start writing)

> Ghid de pornire pentru redactarea paper-ului. Spune explicit **ce e scriibil
> acum** și **ce încă nu e făcut**. Citește înainte să te apuci. Framing-ul a
> fost reformulat pe 2026-06-21 (vezi §1).

---

## 1. Framing-ul corect (low-resource, NU cost)

Întrebarea centrală a paper-ului:

> *Când ai puține date in-domain pe o limbă sub-reprezentată (română), ce
> metodă extrage cel mai bine o notă clinică structurată dintr-o conversație
> chiropractor-pacient?*

Comparăm **prompting vs fine-tuning** sub constrângerea de date scăzute. NU e
despre cost, NU e despre cât context bagi în prompt.

**Titlu fixat:** *Structured Clinical Note Extraction from Romanian
Chiropractor-Patient Conversations: A Low-Resource Comparison*

**Formulare de obiectiv (EN, gata de adaptat):**
> We study structured clinical note extraction from Romanian
> chiropractor-patient conversations as a low-resource problem: limited
> in-domain data and an under-represented language. We compare prompting-based
> and fine-tuning-based methods to identify which approach performs best when
> in-domain data is scarce.

**Costul** rămâne în paper, dar ca **justificare secundară**, nu headline
(diferențele de cost între modelele candidate < ~$50/an la volumul unei
clinici — prea mici ca să ancoreze paper-ul). Secțiunea de cost se completează
DUPĂ research-ul de modele.

---

## 2. Regula non-negociabilă (de pus în obiectiv, nu îngropat în metodă)

> **"If it's not spoken, leave empty."** Fără halucinare, fără inferență
> clinică. Câmp absent → gol/null.

E parte din ce face task-ul greu și e și criteriu de evaluare (vezi §6
"correct empty"). Menționeaz-o în intro.

---

## 3. Ce compară concret (7 condiții)

| # | Condiție | Model | Întrebare |
|---|----------|-------|-----------|
| 1 | Zero-shot | Gemini Flash (API ieftin) | floor ieftin |
| 2 | Zero-shot | Claude Opus (API mare) | ceiling (NU paritate) |
| 3 | Few-shot ICL | open-source small (fix) | ajută ICL modelul mic? |
| 4 | Few-shot ICL | Claude Opus | ceiling ICL |
| 5 | Fine-tune pe sintetic | open-source small (fix) | candidat principal de deployment |
| 6 | Fine-tune pe MTS tradus | open-source small (fix) | ajută pretraining medical general? |
| 7 | Fine-tune MTS → sintetic | open-source small (fix) | adaugă stacking-ul ceva? |

Open-source small îngustat la: **Qwen3-4B / RoLlama3-8B / Gemma3-4B** (pick
neînchis încă).

Întrebări de research: ajută ICL un model mic? poate fine-tuning pe date
puține egala API-urile mari? e data sintetică in-domain mai bună decât medical
general (MTS)? adaugă stacking-ul valoare?

---

## 4. CE E SCRIIBIL ACUM (nu depinde de rezultate)

Astea au dovezi în repo, apucă-te de ele:

| Secțiune | Sursa |
|----------|-------|
| **Intro / Objective** | §1-§2 de mai sus |
| **Related Work** — 3 axe: dialog→note (MEDIQA-Chat, WangLab), synthetic dialogue (NoteChat, SynDial), RO/low-resource (MedQARo) | `AGENTS.md` → Reference work |
| **Task & Data** — schema 6 câmpuri, template Osteopath Concept, split 18 test/0 dev/17 pool hard-fenced | `AGENTS.md`, `src/data_split.py` |
| **STT layer** (context, nu focus) — WhisperX self-hosted, GDPR Art.9, ffmpeg preprocess, fixuri | `src/STT/CHANGELOG_PAPER.md` |
| **Synthetic data generation** — 50 perechi + quality gate (coverage/faithfulness/diversity/distribution) | `synthetic/`, `scripts/generate_synthetic_chiro_data.py` |
| **Evaluation framework** — per-field metrics, "correct empty" handling, manual ternary | `src/SOTA_EVALUATION/eval.py`, decision log în `AGENTS.md` |

**Diferențiatorii — dă-le spațiu disproporționat:**
1. **RO low-resource medical** — citează MedQARo (fine-tuned small bate
   GPT-5.2/Gemini 3 Flash pe RO medical) ca *motivație*. ATENȚIE: NU e baseline
   direct comparabil (ei = QA extractiv, noi = JSON multi-câmp).
2. **Decizia de metrică pe `motivul_prezentarii`** — am testat empiric
   BERTScore + ROUGE-L pe ground truth scurt (2-5 tokeni), gap similar/different
   < 0.05 (zgomot), deci am trecut pe manual ternary (0/0.5/1) + Cohen's kappa
   pe subset dual-rated. Scrie-o ca **contribuție metodologică**, nu ca scuză.

---

## 5. CE NU E ÎNCĂ FĂCUT (marchează clar, nu inventa)

- ❌ **Experimentele NU sunt rulate.** Există doar
  `data/chiropractor_ro/predictions/mock_test/audio18.json` (un mock). Niciuna
  din cele 7 condiții nu are predicții complete pe test set.
- ❌ **Tabelul central de rezultate e gol** — nicio comparație cantitativă.
- ❌ **Fine-tuning (cond. 5/6/7)** — infra neconfirmată în repo.
- ❌ **Secțiunea de cost** — așteaptă research-ul de modele open-source.
- ⚠️ **Reference notes** — chiropracticienii încă adnotează. Doar ~12 refs
  hand-corrected (audio15-23, 25-27). Fișierele din
  `data/chiropractor_ro/notes_by_chatGPT/` sunt **drafturi ChatGPT cu marcaje
  de confidence, NU ground truth.** Nu raporta cifre din ele.
- ⚠️ **Open decisions:** pick model open-source small, N few-shot, volum
  sintetic final.

**Recomandare:** scrie acum §4. Pentru Results/Discussion fă **schelet cu
placeholder-e** — tabel cu coloane definite (7 condiții × per-field metrics),
celule goale. Nu inventa cifre.

---

## 6. Evaluare — ce să accentuezi

- **Per-field, NU un singur număr agregat** ("one headline number hides the
  story"). Păstrează breakdown-ul pe câmpuri în paper.
- **"Correct empty" handling** — o metrică ce scorează doar câmpuri pline
  recompensează tăcut halucinația. Dacă ref și pred sunt ambele goale → corect.
- **Manual ternary** pe `motivul_prezentarii` (vezi §4 pct. 2).
- **Human eval** — review calitativ light de la chiropractor pe un eșantion.
  NU Likert formal, NU inter-rater formal. Susține findings, nu le înlocuiește.

---

## 7. Capcane de formulare (skeptic mode — evită-le)

- NU "cost-efficiency study" / "cheap deployment" ca scop principal.
- NU reduce povestea la "zero-shot vs few-shot" — fine-tuning (5-7) e jumătate
  din experiment.
- NU "matches GPT-4 / Opus quality" — Opus e **ceiling**, nu claim de paritate.
- NU compara direct cu MedQARo (alt task).
- NU pretinde semnificație statistică — N=18 test → CI-uri largi, findings
  direcționale.
- Declară limitările onest: MTS tradus poate produce RO nenaturală (confound
  declarat, nu ascuns); fine-tuning pe sintetic riscă distilarea erorilor
  teacher-ului (de aici quality gate-ul).
