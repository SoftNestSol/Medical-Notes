"""
JSON parser pentru output-ul LLM.

Strategie:
1. Încearcă json.loads direct.
2. Fallback (b): repară gunoiul tipic (markdown fences, text înainte/după)
   și încearcă din nou. Dacă tot eșuează → ridică ParseError.
3. Validează cu jsonschema. Dacă schema fail → ridică SchemaError cu detalii.

Nu face retry la model. Erorile sunt log-ate ca metric separat la eval.
"""

import json
import re
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

from json_schema import NOTE_SCHEMA


class ParseError(Exception):
    """JSON-ul nu a putut fi extras/parsat din răspunsul modelului."""


class SchemaError(Exception):
    """JSON valid, dar nu respectă schema Osteopath Concept v1."""

    def __init__(self, message: str, errors: list[ValidationError]):
        super().__init__(message)
        self.errors = errors


_VALIDATOR = Draft7Validator(NOTE_SCHEMA)


def _strip_code_fences(text: str) -> str:
    """Scoate ```json ... ``` sau ``` ... ``` dacă modelul le adaugă."""
    text = text.strip()
    if text.startswith("```"):
        # scoate prima linie (``` sau ```json)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        # scoate ``` final
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Ultim fallback: găsește primul { și ultimul } și ia ce e între.
    Funcționează când modelul adaugă text de tip "Iată JSON-ul: {...}.".
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ParseError(f"no JSON object braces found in output: {text[:200]!r}")
    return text[start : end + 1]


def parse_note(raw_output: str) -> dict[str, Any]:
    """
    Parsează răspunsul brut al modelului și returnează dict validat.

    Raises:
        ParseError: JSON malformat irecuperabil.
        SchemaError: JSON valid dar nu respectă schema.
    """
    # Pasul 1: direct
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        # Pasul 2: repair
        cleaned = _strip_code_fences(raw_output)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Pasul 3: extragere brute force între { și }
            extracted = _extract_json_object(cleaned)
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError as e:
                raise ParseError(
                    f"JSON parse failed after repair attempts: {e}. "
                    f"Raw (truncated): {raw_output[:300]!r}"
                )

    # Pasul 4: validare schema
    errors = sorted(_VALIDATOR.iter_errors(parsed), key=lambda e: e.path)
    if errors:
        summary = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors
        )
        raise SchemaError(f"schema validation failed: {summary}", errors)

    return parsed