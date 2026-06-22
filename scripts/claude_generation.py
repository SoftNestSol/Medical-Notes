#!/usr/bin/env python3
"""
Synthetic data generation pipeline.

Generates 50 (conversation, note) pairs using Claude Opus 4.7 with few-shot
prompting from 3 real seed pairs. Each call produces ONE pair. The note is
validated against the locked Osteopath Concept v1 schema; invalid outputs
are logged and skipped (no retry to the model — failure rate is a metric).

Usage:
    python generate_synthetic.py

Configure SEEDS and N_SAMPLES below before running.
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from json_schema import ANTECEDENTE_ENUM, LOCALIZARE_ENUM
from parser import ParseError, SchemaError, parse_note

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# =============================================================================
# CONFIG — edit paths and counts here
# =============================================================================

MODEL = "claude-opus-4-7"
MAX_TOKENS = 4096  # higher than zero-shot: must fit conversation + note
N_SAMPLES = 50

# 3 seed pairs. Each entry: (path_to_transcript_txt, path_to_note_json).
# Pick seeds that DIFFER in pathology, length, edge-case coverage.
SEEDS = [
    ("../../data/chiropractor_ro/transcriptions/SEED_1.txt",
     "../../data/chiropractor_ro/notes/SEED_1.json"),
    ("../../data/chiropractor_ro/transcriptions/SEED_2.txt",
     "../../data/chiropractor_ro/notes/SEED_2.json"),
    ("../../data/chiropractor_ro/transcriptions/SEED_3.txt",
     "../../data/chiropractor_ro/notes/SEED_3.json"),
]

OUTPUT_DIR = Path("../../data/chiropractor_ro/synthetic")

# =============================================================================
# PROMPT
# =============================================================================

INSTRUCTION = f"""Ești un generator de date sintetice pentru un cabinet de chiropractică din România.

Sarcina ta: generează O SINGURĂ pereche (conversație, notă structurată) care imită stilul transcrierilor reale primite ca exemple.

## Constrângeri obligatorii pentru conversație

1. **Limbă:** română colocvială, registru oral. Fără jargon literar.
2. **Lungime:** 300-600 cuvinte. Aproximativ 3 minute de dialog vorbit.
3. **Stil:** imitează exemplele — ezitări ("ăăă", "păi"), repetări, întreruperi, propoziții incomplete. Conversația vine dintr-un transcript automat (WhisperX), deci poate avea mici neclarități.
4. **Roluri:** terapeutul pune întrebări și verbalizează (uneori) observații; pacientul răspunde colocvial, NU folosește termeni medicali tehnici pe care un nespecialist nu i-ar ști.
5. **Antecedente familiale:** dacă apar ("mama are diabet"), NU trebuie să intre în câmpul `antecedente`.

## Constrângere pe varietate

Fiecare conversație generată trebuie să fie **DISTINCTĂ** ca patologie principală, demografie pacient, structură dialog. NU varia pe aceeași temă (ex: nu toate să fie lombalgie). Acoperă diferite zone: cervical, toracal, umăr, genunchi, șold, cap/ceafă, sciatică, durere multi-zonă etc.

## Edge cases — fiecare conversație TREBUIE să exhibe 1-2 din lista de mai jos

Alege EXPLICIT 1-2 cazuri și asigură-te că nota le reflectă corect (câmp gol/null pentru cele neverbalizate):

- VAS verbalizat ca număr concret (ex: "vreo 7") → câmp populat
- Durere descrisă fără număr ("doare rău", "insuportabil") → VAS=null
- Pacient zice "spate" vag fără a preciza lombar/toracal → `localizarea_durerii_alta`="spate", nu mapa la lombar
- Medicament fără doză rostită → `doza`=null, NU inventa
- Pacient spune că nu ia medicamente → `medicatie_actuala`=[]
- Antecedente familiale dar nu personale → `antecedente`=[]
- Pacient incert ("cred că am avut...", "poate") → câmp gol
- Confirmare la întrebarea terapeutului ("Aveți hipertensiune?" → "Da") → contează ca rostit
- Terapeut examinează în tăcere, nu verbalizează observații → `evaluare_functionala_initiala`=null
- Durere în mai multe zone → listă multi-element în `localizarea_durerii`
- Zonă anatomică în afara enum-ului (fesier, coapsă, antebraț) → `localizarea_durerii_alta`

## REGULA FUNDAMENTALĂ pentru notă

**Dacă o informație NU este rostită explicit în conversația generată de tine, câmpul rămâne gol în notă.**

Nota trebuie să fie 100% extractibilă din conversație. Nu adăuga în notă lucruri pe care nu le-ai pus în conversație. Nu lăsa în conversație lucruri pe care le-ai omis din notă (decât dacă sunt irelevante pentru schemă).

## Schema notei

Slug-uri permise `localizarea_durerii`: {LOCALIZARE_ENUM}
Slug-uri permise `antecedente`: {ANTECEDENTE_ENUM}

Returnează EXACT acest obiect JSON, fără text înainte sau după, fără markdown fences:

{{
  "conversatie": "<transcrierea generată, 300-600 cuvinte, cu marcaje T: și P: pentru terapeut/pacient>",
  "nota": {{
    "motivul_prezentarii": <string sau null>,
    "evaluarea_durerii_vas": <int 0-10 sau null>,
    "localizarea_durerii": <listă din enum>,
    "localizarea_durerii_alta": <string sau null>,
    "antecedente": <listă din enum>,
    "antecedente_altele": <string sau null>,
    "medicatie_actuala": <listă de {{"denumire": string, "doza": string sau null}}>,
    "evaluare_functionala_initiala": <string sau null>
  }}
}}
"""


def _load_seeds(seed_paths: list[tuple[str, str]]) -> str:
    """Load seed pairs and format them as few-shot examples for the prompt."""
    blocks = []
    for i, (transcript_path, note_path) in enumerate(seed_paths, start=1):
        tpath = Path(transcript_path)
        npath = Path(note_path)
        if not tpath.exists():
            raise FileNotFoundError(f"seed transcript not found: {tpath}")
        if not npath.exists():
            raise FileNotFoundError(f"seed note not found: {npath}")

        transcript = tpath.read_text(encoding="utf-8").strip()
        note = json.loads(npath.read_text(encoding="utf-8"))

        block = (
            f"### EXEMPLU {i}\n\n"
            f"CONVERSAȚIE:\n{transcript}\n\n"
            f"NOTĂ:\n{json.dumps(note, ensure_ascii=False, indent=2)}\n"
        )
        blocks.append(block)
    return "\n---\n\n".join(blocks)


def build_system_prompt(seed_paths: list[tuple[str, str]]) -> str:
    examples = _load_seeds(seed_paths)
    return (
        INSTRUCTION
        + "\n\n## Exemple de perechi reale (folosește ca referință de stil)\n\n"
        + examples
    )


def generate_one(client: anthropic.Anthropic, system_prompt: str, idx: int) -> dict:
    """Single API call → parsed and validated pair."""
    user_message = (
        f"Generează perechea numărul {idx} din {N_SAMPLES}. "
        f"Asigură-te că diferă de tot ce ai generat anterior. "
        f"Alege explicit 1-2 edge cases și aplică-le."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    usage = response.usage
    print(
        f"[{idx:02d}] cache_created={usage.cache_creation_input_tokens} "
        f"cache_read={usage.cache_read_input_tokens} "
        f"input={usage.input_tokens} output={usage.output_tokens}",
        file=sys.stderr,
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    pair = json.loads(raw)

    if "conversatie" not in pair or "nota" not in pair:
        raise ValueError(f"missing 'conversatie' or 'nota' in output: {list(pair.keys())}")
    if not isinstance(pair["conversatie"], str) or not pair["conversatie"].strip():
        raise ValueError("'conversatie' is empty or not a string")

    # Validate the note against the locked schema (reuses parser logic).
    note_raw = json.dumps(pair["nota"], ensure_ascii=False)
    validated_note = parse_note(note_raw)
    pair["nota"] = validated_note

    return pair


def main() -> None:
    api_key = os.getenv("ANTHROPIC_KEY")
    if not api_key:
        print("ANTHROPIC_KEY missing", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt(SEEDS)
    print(f"[setup] system prompt: {len(system_prompt)} chars", file=sys.stderr)

    client = anthropic.Anthropic(api_key=api_key)

    successes = 0
    failures: list[tuple[int, str]] = []

    for idx in range(1, N_SAMPLES + 1):
        out_path = OUTPUT_DIR / f"synth_{idx:03d}.json"
        if out_path.exists():
            print(f"[{idx:02d}] skip (exists)", file=sys.stderr)
            successes += 1
            continue

        try:
            pair = generate_one(client, system_prompt, idx)
        except (json.JSONDecodeError, ValueError, ParseError, SchemaError) as e:
            print(f"[{idx:02d}] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append((idx, f"{type(e).__name__}: {e}"))
            continue
        except anthropic.APIError as e:
            print(f"[{idx:02d}] API error: {e}", file=sys.stderr)
            failures.append((idx, f"APIError: {e}"))
            time.sleep(2)
            continue

        out_path.write_text(json.dumps(pair, ensure_ascii=False, indent=2), encoding="utf-8")
        successes += 1
        print(f"[{idx:02d}] saved {out_path.name}", file=sys.stderr)

    print(f"\n=== DONE: {successes}/{N_SAMPLES} successful, {len(failures)} failed ===",
          file=sys.stderr)
    if failures:
        log_path = OUTPUT_DIR / "_failures.log"
        log_path.write_text(
            "\n".join(f"{i}\t{msg}" for i, msg in failures), encoding="utf-8"
        )
        print(f"failure log: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()