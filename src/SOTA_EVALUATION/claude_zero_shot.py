#!/usr/bin/env python3
"""
Condition 2: zero-shot extraction cu Claude Opus 4.7 (expensive ceiling).

Usage:
    python claude.py <path/to/transcribe.txt>
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from json_schema import ANTECEDENTE_ENUM, LOCALIZARE_ENUM
from parser import ParseError, SchemaError, parse_note

load_dotenv(Path(__file__).parent.parent.parent / ".env")

MODEL = "claude-opus-4-7"
MAX_TOKENS = 2048

SYSTEM_PROMPT = f"""Ești un sistem de extragere de informații clinice din transcrierile consultațiilor unui cabinet de chiropractică din România.

Primești o transcriere (dialog terapeut-pacient) și returnezi UN SINGUR obiect JSON cu informațiile extrase.

## REGULA FUNDAMENTALĂ — NU SE NEGOCIAZĂ

**Dacă o informație NU este rostită explicit în conversație, câmpul rămâne gol (null sau listă goală).**

- Nu deduce. Nu presupune. Nu completa pe baza intuiției clinice.
- Nu reformula în termeni clinici ce nu a fost spus.
- Dacă pacientul confirmă o întrebare a terapeutului ("Aveți hipertensiune?" → "Da"), informația contează ca rostită.
- Dacă pacientul folosește incertitudine ("cred că", "poate"), lasă câmpul gol.

## Schema JSON

Returnează EXACT acest obiect, fără text înainte sau după, fără markdown fences:

{{
  "motivul_prezentarii": <string sau null>,
  "evaluarea_durerii_vas": <int 0-10, sau listă de int 0-10, sau null>,
  "localizarea_durerii": <listă de slug-uri din enum, poate fi goală>,
  "localizarea_durerii_alta": <string sau null>,
  "antecedente": <listă de slug-uri din enum, poate fi goală>,
  "antecedente_altele": <string sau null>,
  "medicatie_actuala": <listă de obiecte {{"denumire": string, "doza": string sau null}}, poate fi goală>,
  "evaluare_functionala_initiala": <string sau null>
}}

## Reguli pe câmp

### motivul_prezentarii (text liber)
- Rezumat al acuzei principale a pacientului, în cuvintele lui (nu în jargon clinic).
- Null dacă nu este verbalizat un motiv clar.

### evaluarea_durerii_vas (int 0-10, listă de int 0-10, sau null)
- Doar dacă pacientul rostește un NUMĂR explicit pe scala 0-10.
- Un singur scor rostit → int (ex: "8" → 8).
- Mai multe scoruri distincte rostite (zone diferite, posturi diferite, momente diferite, evoluție în timp) → listă în ordinea în care apar (ex: "lombar 8, cervical 5" → [8, 5]; "la început 9, acum 6" → [9, 6]).
- "Doare rău", "doare tare", "insuportabil" → null.
- Număr în afara scalei ("15 din 10") → null.
- Nu inventa o listă dintr-un singur scor. Lista se folosește doar când pacientul rostește mai multe valori.

### localizarea_durerii (multi-select din enum)
Slug-uri permise: {LOCALIZARE_ENUM}
- Doar zone numite explicit sau confirmate de pacient.
- "Spate" vag fără precizare → nu mapa la lombar; folosește localizarea_durerii_alta = "spate" sau lasă gol.
- "Ambii umeri" → ["umar_dr", "umar_stg"].

### localizarea_durerii_alta (text liber sau null)
- Pentru zone menționate dar care nu se potrivesc enum-ului (ex: "fesier", "coapsă", "spate" vag).
- Null dacă toate zonele rostite intră în enum.

### antecedente (multi-select din enum)
Slug-uri permise: {ANTECEDENTE_ENUM}
- Doar afecțiuni confirmate de pacient (sau confirmate la întrebarea terapeutului).
- Antecedente familiale ("mama are diabet") → NU intră aici.

### antecedente_altele (text liber sau null)
- Pentru afecțiuni rostite dar care nu se potrivesc enum-ului.
- Null dacă nu există.

### medicatie_actuala (listă de obiecte)
- Format: [{{"denumire": "ibuprofen", "doza": "400mg"}}, {{"denumire": "magneziu", "doza": null}}]
- Doza poate fi null dacă nu a fost rostită. NU INVENTA doze.
- Listă goală dacă pacientul nu menționează medicamente sau spune că nu ia.

### evaluare_functionala_initiala (text liber sau null)
- DOAR observații rostite cu voce tare de terapeut despre postură, biomecanică, mobilitate.
- Dacă terapeutul examinează în tăcere și nu verbalizează nimic → null.
- Nu include teste sau diagnostic.

## Format ieșire

- DOAR obiectul JSON, valid, parsabil cu json.loads().
- Fără ```json, fără comentarii, fără text suplimentar.
- Folosește null (nu None, nu "").
- Listele goale sunt [] (nu null) pentru câmpurile de listă.
"""


def extract_note(transcription: str) -> dict:
    """Call Claude API and return the parsed + schema-validated JSON note."""
    api_key = os.getenv("ANTHROPIC_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_KEY not found in environment / .env file")

    client = anthropic.Anthropic(api_key=api_key)

    user_message = f"--- TRANSCRIEREA CONSULTAȚIEI ---\n\n{transcription}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    usage = response.usage
    print(
        f"[cache] created={usage.cache_creation_input_tokens} "
        f"read={usage.cache_read_input_tokens} "
        f"uncached={usage.input_tokens} "
        f"output={usage.output_tokens}",
        file=sys.stderr,
    )

    raw = response.content[0].text
    return parse_note(raw)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <transcription_file.txt>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    transcription = input_path.read_text(encoding="utf-8")

    try:
        note = extract_note(transcription)
    except ParseError as e:
        print(f"[parse error] {e}", file=sys.stderr)
        sys.exit(2)
    except SchemaError as e:
        print(f"[schema error] {e}", file=sys.stderr)
        sys.exit(3)

    output_json = json.dumps(note, ensure_ascii=False, indent=2)
    print(output_json)

    output_path = input_path.with_suffix(".json")
    output_path.write_text(output_json, encoding="utf-8")
    print(f"[saved] {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()