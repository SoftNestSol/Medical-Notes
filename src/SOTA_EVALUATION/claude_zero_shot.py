#!/usr/bin/env python3
"""
Condition 2: big-model zero-shot extraction of Romanian chiro dialogues
into structured JSON matching the Osteopath Concept v1 schema.

Usage:
    python claude.py <path/to/transcribe.txt>
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

MODEL = "claude-sonnet-4-6"  # swap to "claude-opus-4-7" for the absolute ceiling

LOCALIZARE_ENUM = [
    "cervical", "toracal", "lombar", "sacral_coccis",
    "umar_dr", "umar_stg", "cot_dr", "cot_stg",
    "pumn_dr", "pumn_stg", "sold_dr", "sold_stg",
    "genunchi_dr", "genunchi_stg", "glezna_dr", "glezna_stg",
    "cap_ceafa", "abdomen", "torace",
]

ANTECEDENTE_ENUM = [
    "hipertensiune_arteriala", "diabet_zaharat", "boli_cardiovasculare",
    "osteoporoza", "artrita_artroza", "hernia_disc", "scolioza_cifoza",
    "epilepsie", "cancer_neoplasm", "boli_autoimune",
]

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
  "evaluarea_durerii_vas": <int 0-10 sau null>,
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

### evaluarea_durerii_vas (int 0-10 sau null)
- Doar dacă pacientul rostește un NUMĂR explicit pe scala 0-10.
- "Doare rău", "doare tare", "insuportabil" → null.
- Interval ("4-5", "între 6 și 8") → null (lasă gol până clarificăm regula).
- Număr în afara scalei ("15 din 10") → null.

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
- "Am avut hernie" vs "am hernie" → ambele intră (până clarificăm regula trecut/prezent).

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
    """Call Claude API and return the parsed JSON note."""
    api_key = os.getenv("ANTHROPIC_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_KEY not found in environment / .env file")

    client = anthropic.Anthropic(api_key=api_key)

    user_message = f"--- TRANSCRIEREA CONSULTAȚIEI ---\n\n{transcription}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
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

    raw = response.content[0].text.strip()

    # Defensive: strip code fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[error] model returned invalid JSON:\n{raw}", file=sys.stderr)
        raise RuntimeError(f"JSON parse failed: {e}")

    return parsed


def validate_schema(note: dict) -> list[str]:
    """Light schema check. Returns list of issues (empty = clean)."""
    issues = []
    expected_keys = {
        "motivul_prezentarii", "evaluarea_durerii_vas",
        "localizarea_durerii", "localizarea_durerii_alta",
        "antecedente", "antecedente_altele",
        "medicatie_actuala", "evaluare_functionala_initiala",
    }
    missing = expected_keys - note.keys()
    extra = note.keys() - expected_keys
    if missing:
        issues.append(f"missing keys: {missing}")
    if extra:
        issues.append(f"unexpected keys: {extra}")

    vas = note.get("evaluarea_durerii_vas")
    if vas is not None and not (isinstance(vas, int) and 0 <= vas <= 10):
        issues.append(f"VAS not int 0-10 or null: {vas!r}")

    loc = note.get("localizarea_durerii", [])
    if isinstance(loc, list):
        bad = [v for v in loc if v not in LOCALIZARE_ENUM]
        if bad:
            issues.append(f"localizarea_durerii has out-of-enum values: {bad}")

    ant = note.get("antecedente", [])
    if isinstance(ant, list):
        bad = [v for v in ant if v not in ANTECEDENTE_ENUM]
        if bad:
            issues.append(f"antecedente has out-of-enum values: {bad}")

    return issues


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <transcription_file.txt>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    transcription = input_path.read_text(encoding="utf-8")
    note = extract_note(transcription)

    issues = validate_schema(note)
    if issues:
        print("[schema warnings]", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)

    output_json = json.dumps(note, ensure_ascii=False, indent=2)
    print(output_json)

    output_path = input_path.with_suffix(".json")
    output_path.write_text(output_json, encoding="utf-8")
    print(f"[saved] {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()