#!/usr/bin/env python3
"""Generate synthetic Romanian chiropractor pairs with the Claude API.

Output layout matches data/synthetic/GPT5.5 so fine-tuning prep can swap the
root directory without changing the expected transcript/ref structure.

Default run:
    python scripts/generate_synthetic_claude_data.py --n 50

Smoke check without API calls:
    python scripts/generate_synthetic_claude_data.py --dry-run --n 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - dry-run can still work without SDK
    anthropic = None

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SOTA = SRC / "SOTA_EVALUATION"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SOTA))

from data_split import POOL_COMPLETE_PAIR_IDS, assert_no_test_leakage  # noqa: E402
from parser import ParseError, SchemaError, parse_note  # noqa: E402
from SOTA_EVALUATION.json_schema import (  # noqa: E402
    ANTECEDENTE_ENUM,
    LOCALIZARE_ENUM,
    NOTE_SCHEMA,
)

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_PLAN = ROOT / "data" / "synthetic" / "GPT5.5" / "plan.tsv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "synthetic" / "Claude"
DEFAULT_SEED_IDS = ["audio18", "audio19", "audio23", "audio26", "audio21"]

NOTE_FIELDS = [
    "motivul_prezentarii",
    "evaluarea_durerii_vas",
    "localizarea_durerii",
    "localizarea_durerii_alta",
    "antecedente",
    "antecedente_altele",
    "medicatie_actuala",
    "evaluare_functionala_initiala",
]

REGION_LABELS = {
    "cervical": "zona cervicala / gat / ceafa",
    "toracal": "zona toracala / mijlocul spatelui",
    "lombar": "zona lombara / partea de jos a spatelui",
    "sacral_coccis": "sacru / coccis",
    "umar_dr": "umarul drept",
    "umar_stg": "umarul stang",
    "cot_dr": "cotul drept",
    "cot_stg": "cotul stang",
    "pumn_dr": "pumnul drept",
    "pumn_stg": "pumnul stang",
    "sold_dr": "soldul drept",
    "sold_stg": "soldul stang",
    "genunchi_dr": "genunchiul drept",
    "genunchi_stg": "genunchiul stang",
    "glezna_dr": "glezna dreapta",
    "glezna_stg": "glezna stanga",
    "cap_ceafa": "cap / ceafa",
    "abdomen": "abdomen",
    "torace": "torace",
}

WORD_TARGETS = {
    "short_sparse": "450-750 cuvinte",
    "detailed_long": "800-1200 cuvinte",
}

ANTECEDENTE_LABELS = {
    "hipertensiune_arteriala": "hipertensiune arteriala",
    "diabet_zaharat": "diabet zaharat",
    "boli_cardiovasculare": "boala cardiovasculara",
    "osteoporoza": "osteoporoza",
    "artrita_artroza": "artrita sau artroza",
    "hernia_disc": "hernie de disc",
    "scolioza_cifoza": "scolioza sau cifoza",
    "epilepsie": "epilepsie",
    "cancer_neoplasm": "cancer / neoplasm",
    "boli_autoimune": "boala autoimuna",
}


class GenerationValidationError(Exception):
    """Raised when Claude output is parseable but not usable as training data."""


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_vas_plan(value: str) -> Any:
    value = value.strip()
    if value == "null":
        return None
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return int(value)


def parse_plan_rows(path: Path, *, start: int, n: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        base_rows = list(csv.DictReader(f, delimiter="\t"))
    if start < 1:
        raise ValueError("--start is 1-based and must be >= 1")
    if not base_rows:
        raise ValueError(f"empty plan: {path}")

    selected = []
    extended = start + n - 1 > len(base_rows)
    for global_index in range(start, start + n):
        template = dict(base_rows[(global_index - 1) % len(base_rows)])
        if extended:
            template["plan_template_id"] = template["conversation_id"]
            template["repeat_cycle"] = str((global_index - 1) // len(base_rows) + 1)
            template["conversation_id"] = f"synth_{global_index:03d}"
            template["batch"] = str((global_index - 1) // len(base_rows) + 1)
            template["generation_variation_hint"] = (
                f"Varianta {global_index}; păstrează valorile structurate din "
                "template, dar schimbă formulările, ritmul conversației, ordinea "
                "întrebărilor unde e natural și detaliile conversaționale."
            )
        selected.append(template)
    return selected


def expected_note_constraints(row: dict[str, str]) -> dict[str, Any]:
    antecedente = [] if row["antecedente_plan"] == "none" else row["antecedente_plan"].split(",")
    for item in antecedente:
        if item not in ANTECEDENTE_ENUM:
            raise ValueError(f"unknown antecedent in plan {row['conversation_id']}: {item}")

    if row["include_location"] == "yes":
        locations = row["regions"].split(",")
    else:
        locations = []
    for item in locations:
        if item not in LOCALIZARE_ENUM:
            raise ValueError(f"unknown location in plan {row['conversation_id']}: {item}")

    meds: list[dict[str, Any]]
    if row["meds_plan"] == "named":
        meds = json.loads(row["meds_detail"])
    else:
        meds = []

    return {
        "evaluarea_durerii_vas": parse_vas_plan(row["vas"]),
        "localizarea_durerii": locations,
        "localizarea_durerii_alta": row["localizare_alta_plan"] or None,
        "antecedente": antecedente,
        "antecedente_altele": row["antecedente_altele_plan"] or None,
        "medicatie_actuala": meds,
        "evaluare_functionala_initiala_required": row["func_eval"] == "populated",
    }


def read_conversation_as_transcript(path: Path) -> str:
    data = load_json(path)
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"conversation JSON has no segments list: {path}")
    lines = []
    for segment in segments:
        speaker = str(segment.get("speaker", "SPEAKER_00")).strip() or "SPEAKER_00"
        text = str(segment.get("text", "")).strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def seed_paths(seed_ids: list[str]) -> list[tuple[str, Path, Path, Optional[Path]]]:
    unknown = sorted(set(seed_ids) - POOL_COMPLETE_PAIR_IDS)
    if unknown:
        raise ValueError(
            "seed ids must be pool IDs with guaranteed local conversation+ref "
            f"pairs. Invalid now: {unknown}. Allowed: {sorted(POOL_COMPLETE_PAIR_IDS)}"
        )
    assert_no_test_leakage(seed_ids, context="Claude synthetic seed selection")
    paths = []
    for seed_id in seed_ids:
        conversation_path = ROOT / "data" / "chiropractor_ro" / "conversations" / f"{seed_id}.json"
        ref_path = ROOT / "data" / "chiropractor_ro" / "refs" / f"{seed_id}.json"
        matches = sorted((ROOT / "data" / "from_chiro").glob(f"{seed_id}_fisa.*"))
        from_chiro_path = matches[0] if matches else None
        if not conversation_path.exists():
            raise FileNotFoundError(conversation_path)
        if not ref_path.exists():
            raise FileNotFoundError(ref_path)
        paths.append((seed_id, conversation_path, ref_path, from_chiro_path))
    return paths


def build_seed_examples(seed_ids: list[str]) -> str:
    blocks = []
    for index, (seed_id, conversation_path, ref_path, _) in enumerate(seed_paths(seed_ids), start=1):
        transcript = read_conversation_as_transcript(conversation_path)
        note = load_json(ref_path)
        parse_note(json.dumps(note, ensure_ascii=False))
        blocks.append(
            "\n".join(
                [
                    f"<example id=\"{seed_id}\" n=\"{index}\">",
                    "<source>full real WhisperX conversation flattened from all segments + corrected structured ref</source>",
                    "<transcript>",
                    transcript.strip(),
                    "</transcript>",
                    "<note>",
                    json.dumps(note, ensure_ascii=False, indent=2),
                    "</note>",
                    "</example>",
                ]
            )
        )
    return "\n\n".join(blocks)


def write_seed_manifest(out_root: Path, seed_ids: list[str]) -> None:
    lines = ["seed_id\tconversation_path\tstructured_ref_path\tfrom_chiro_path"]
    for seed_id, conversation_path, ref_path, from_chiro_path in seed_paths(seed_ids):
        from_chiro = "" if from_chiro_path is None else str(from_chiro_path.relative_to(ROOT))
        lines.append(
            "\t".join(
                [
                    seed_id,
                    str(conversation_path.relative_to(ROOT)),
                    str(ref_path.relative_to(ROOT)),
                    from_chiro,
                ]
            )
        )
    (out_root / "seed_pairs.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_system_prompt(seed_ids: list[str]) -> str:
    seed_examples = build_seed_examples(seed_ids)
    schema = json.dumps(NOTE_SCHEMA, ensure_ascii=False, indent=2)
    return f"""Ești un generator de date sintetice pentru un studiu de bioNLP.

Generezi conversații sintetice în română între terapeut chiropractor/osteopat și pacient, apoi nota structurată care se poate extrage din conversație.

REGULA FUNDAMENTALĂ:
Dacă informația nu este rostită explicit în conversație, câmpul din notă rămâne gol: null sau [].

Interdicții:
- Nu deduce clinic.
- Nu completa antecedente, medicație, diagnostic sau obiective terapeutice din probabilitate.
- Nu transforma "spate" vag în lombar/toracal.
- Nu include antecedente familiale în antecedente personale.
- Nu inventa doze.
- Nu copia conversațiile reale; folosește-le doar ca stil.
- Nu produce dialoguri curate de manual. Imită transcript real: fragmente scurte,
  reveniri, întrebări reluate, răspunsuri monosilabice, confirmări "da", "ok",
  reformulări, mici erori ASR și dezacorduri gramaticale tolerabile.
- Nu face pacientul să vorbească literar sau clinic. Pacientul spune "mă ține",
  "mă jenează", "aici", "pe partea asta", "nu știu exact", "păi", "adică",
  "cum să zic"; terapeutul clarifică și verifică.

Stil conversație:
- Română orală, naturală, cu ezitări moderate, răspunsuri incomplete, suprapuneri
  sugerate prin propoziții neterminate și mici imperfecțiuni de transcript ASR.
- Folosește doar etichete SPEAKER_00 și SPEAKER_01.
- Lungime realistă, nu mini-dialog: cazurile short_sparse au de obicei 450-750 cuvinte; cazurile detailed_long au de obicei 800-1200 cuvinte.
- Pacientul vorbește ca pacient, nu ca medic.
- Terapeutul poate verbaliza observații funcționale, diagnostic sau obiective doar dacă planul cere evaluare funcțională populată.

Checklist intern de stil, fără să îl afișezi:
- menține cadență de consultație reală românească: întrebare scurtă, răspuns
  dezordonat, clarificare, confirmare;
- include expresii orale românești specifice, dar nu caricaturale;
- păstrează imperfecțiuni ASR moderate: cuvinte ușor stâlcite sau fraze tăiate,
  fără să distrugi informația clinică;
- alternează replici foarte scurte cu replici lungi ale pacientului;
- evită conversația prea simetrică și prea politicos redactată.

Schema notei:
{schema}

Trebuie să returnezi exact un obiect JSON cu forma:
{{
  "conversation": "<transcript cu linii SPEAKER_00/SPEAKER_01>",
  "note": {{... conform schema ...}},
  "evidence": {{
    "motivul_prezentarii": ["citat(e) din conversație sau []"],
    "evaluarea_durerii_vas": ["citat(e) din conversație sau []"],
    "localizarea_durerii": ["citat(e) din conversație sau []"],
    "localizarea_durerii_alta": ["citat(e) din conversație sau []"],
    "antecedente": ["citat(e) din conversație sau []"],
    "antecedente_altele": ["citat(e) din conversație sau []"],
    "medicatie_actuala": ["citat(e) din conversație sau []"],
    "evaluare_functionala_initiala": ["citat(e) din conversație sau []"]
  }}
}}

Reguli pentru evidence:
- Pentru fiecare câmp non-empty din note, evidence trebuie să conțină cel puțin un citat exact copiat din conversație.
- Pentru fiecare câmp empty/null/[], evidence trebuie să fie [].
- Citatele trebuie să fie substrings reale din transcript.

Exemple reale de stil și notă:
{seed_examples}
"""


def build_user_prompt(row: dict[str, str]) -> str:
    constraints = expected_note_constraints(row)
    word_target = WORD_TARGETS.get(row["richness"], "600-1000 cuvinte")
    interpreted = {
        "conversation_id": row["conversation_id"],
        "required_structured_values": {
            key: value for key, value in constraints.items() if key != "evaluare_functionala_initiala_required"
        },
        "functional_eval": (
            "must be a spoken therapist observation/diagnostic/objective"
            if constraints["evaluare_functionala_initiala_required"]
            else "must be null; do not verbalize therapist diagnostic/objectives/functional observations"
        ),
        "style_targets": {
            "duration": row["duration"],
            "age": row["age"],
            "onset": row["onset"],
            "richness": row["richness"],
            "patient_style": row["patient_style"],
            "opening": row["opening"],
            "activity": row["activity"],
            "dialogue_act_order": row["dialogue_act_order"],
            "note_shape": row["note_shape"],
            "target_length": word_target,
        },
        "region_theme": row["regions"],
        "meds_plan": row["meds_plan"],
    }
    return f"""Generează conversația sintetică și nota pentru {row['conversation_id']}.

Plan controlat:
{json.dumps(row, ensure_ascii=False, indent=2)}

Interpretare obligatorie a planului:
{json.dumps(interpreted, ensure_ascii=False, indent=2)}

Constrângeri concrete:
- Valorile structurate din note trebuie să respecte exact required_structured_values.
- Lungimea conversației trebuie să fie aproximativ {word_target}.
- Prioritatea stilistică este să semene cu seed-urile reale: oral, fragmentat,
  română colocvială, cu mici artefacte de transcript. Nu scrie proză curată.
- motivul_prezentarii trebuie să fie scurt, natural și extractibil din reclamația pacientului.
- Dacă VAS este null, pacientul nu trebuie să spună nicio cifră de durere pe scala 0-10.
- Dacă meds_plan este "unnamed-pills", pacientul poate spune că a luat pastile fără nume, dar medicatie_actuala trebuie să fie [].
- Dacă include_location este "no", conversația nu trebuie să numească explicit regiunile din region_theme ca localizare a durerii.
- Nu include markdown fences. Returnează doar JSON valid.
"""


def extract_text_from_response(response: Any) -> str:
    chunks = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def validate_plan_compatibility(note: dict[str, Any], row: dict[str, str]) -> None:
    constraints = expected_note_constraints(row)
    for field in [
        "evaluarea_durerii_vas",
        "localizarea_durerii",
        "localizarea_durerii_alta",
        "antecedente",
        "antecedente_altele",
        "medicatie_actuala",
    ]:
        if note[field] != constraints[field]:
            raise GenerationValidationError(
                f"{row['conversation_id']} field {field}={note[field]!r}, "
                f"expected {constraints[field]!r}"
            )

    func_required = constraints["evaluare_functionala_initiala_required"]
    if func_required and not is_populated(note["evaluare_functionala_initiala"]):
        raise GenerationValidationError(f"{row['conversation_id']} missing required functional eval")
    if not func_required and note["evaluare_functionala_initiala"] is not None:
        raise GenerationValidationError(f"{row['conversation_id']} functional eval must be null")
    if not is_populated(note["motivul_prezentarii"]):
        raise GenerationValidationError(f"{row['conversation_id']} missing motivul_prezentarii")


def validate_evidence(pair: dict[str, Any], row: dict[str, str]) -> None:
    conversation = pair["conversation"]
    note = pair["note"]
    evidence = pair["evidence"]
    normalized_conversation = normalize_text(conversation)

    if set(evidence) != set(NOTE_FIELDS):
        raise GenerationValidationError(
            f"{row['conversation_id']} evidence fields mismatch: {sorted(evidence)}"
        )

    for field in NOTE_FIELDS:
        quotes = evidence[field]
        if not isinstance(quotes, list) or not all(isinstance(item, str) for item in quotes):
            raise GenerationValidationError(f"{row['conversation_id']} evidence.{field} must be list[str]")
        populated = is_populated(note[field])
        if populated and not quotes:
            raise GenerationValidationError(f"{row['conversation_id']} missing evidence for {field}")
        if not populated and quotes:
            raise GenerationValidationError(f"{row['conversation_id']} empty field {field} has evidence")
        for quote in quotes:
            if len(quote.strip()) < 4:
                raise GenerationValidationError(f"{row['conversation_id']} evidence quote too short for {field}")
            if normalize_text(quote) not in normalized_conversation:
                raise GenerationValidationError(
                    f"{row['conversation_id']} evidence quote not found for {field}: {quote!r}"
                )

    for med in note["medicatie_actuala"]:
        if normalize_text(med["denumire"]) not in normalized_conversation:
            raise GenerationValidationError(f"{row['conversation_id']} medication name not spoken: {med}")
        if med["doza"] and normalize_text(med["doza"]) not in normalized_conversation:
            raise GenerationValidationError(f"{row['conversation_id']} medication dose not spoken: {med}")


def validate_conversation_shape(conversation: str, row: dict[str, str]) -> None:
    lines = [line for line in conversation.splitlines() if line.strip()]
    if not lines:
        raise GenerationValidationError(f"{row['conversation_id']} empty conversation")
    bad_lines = [line for line in lines if not re.match(r"^SPEAKER_0[01]:\s+\S", line)]
    if bad_lines:
        raise GenerationValidationError(
            f"{row['conversation_id']} bad speaker labels: {bad_lines[:3]}"
        )
    words = re.findall(r"\w+", conversation, flags=re.UNICODE)
    if not 350 <= len(words) <= 1300:
        raise GenerationValidationError(
            f"{row['conversation_id']} word count {len(words)} outside 350-1300"
        )
    if "```" in conversation:
        raise GenerationValidationError(f"{row['conversation_id']} contains markdown fence")


def parse_and_validate_pair(raw: str, row: dict[str, str]) -> dict[str, Any]:
    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise GenerationValidationError(f"{row['conversation_id']} root is not object")
    if set(parsed) != {"conversation", "note", "evidence"}:
        raise GenerationValidationError(f"{row['conversation_id']} unexpected root keys: {sorted(parsed)}")
    if not isinstance(parsed["conversation"], str):
        raise GenerationValidationError(f"{row['conversation_id']} conversation must be string")

    note = parse_note(json.dumps(parsed["note"], ensure_ascii=False))
    parsed["note"] = note
    validate_conversation_shape(parsed["conversation"], row)
    validate_plan_compatibility(note, row)
    validate_evidence(parsed, row)
    return parsed


def call_claude(
    client: Any,
    *,
    model: str,
    max_tokens: int,
    system_prompt: str,
    row: dict[str, str],
) -> tuple[str, Any]:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": build_user_prompt(row)}],
    )
    return extract_text_from_response(response), getattr(response, "usage", None)


def write_plan(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(out_root: Path, rows: list[dict[str, str]], successful_ids: list[str]) -> None:
    notes = [load_json(out_root / "refs" / f"{cid}.json") for cid in successful_ids]
    lines = ["field\tpositive_count\trate"]
    for field in NOTE_FIELDS:
        count = sum(1 for note in notes if is_populated(note[field]))
        rate = count / len(notes) if notes else 0.0
        lines.append(f"{field}\t{count}\t{rate:.2f}")
    (out_root / "coverage_report.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plan_by_id = {row["conversation_id"]: row for row in rows}
    plan_lines = ["conversation_id\tvas\tregions\tantecedente_plan\tmeds_plan\tfunc_eval\tstatus"]
    for cid in successful_ids:
        row = plan_by_id[cid]
        plan_lines.append(
            "\t".join(
                [
                    cid,
                    row["vas"],
                    row["regions"],
                    row["antecedente_plan"],
                    row["meds_plan"],
                    row["func_eval"],
                    "ok",
                ]
            )
        )
    (out_root / "generation_report.tsv").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")


def existing_pair_ok(out_root: Path, row: dict[str, str]) -> bool:
    cid = row["conversation_id"]
    transcript_path = out_root / "transcripts" / f"{cid}.txt"
    note_path = out_root / "refs" / f"{cid}.json"
    audit_path = out_root / "audits" / f"{cid}.json"
    if not transcript_path.exists() or not note_path.exists() or not audit_path.exists():
        return False
    try:
        pair = {
            "conversation": transcript_path.read_text(encoding="utf-8"),
            "note": load_json(note_path),
            "evidence": load_json(audit_path)["evidence"],
        }
        parse_and_validate_pair(json.dumps(pair, ensure_ascii=False), row)
    except Exception:
        return False
    return True


def usage_to_dict(usage: Any) -> Optional[dict[str, Any]]:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    fields = [
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ]
    return {field: getattr(usage, field) for field in fields if hasattr(usage, field)}


def transcript_to_conversation_json(conversation: str, *, model: str) -> dict[str, Any]:
    """Convert speaker-line text to the same broad shape as WhisperX JSON.

    Timestamps are approximate metadata for compatibility only; SFT uses the
    flattened transcript text.
    """
    segments = []
    cursor = 0.0
    for line in conversation.strip().splitlines():
        speaker, sep, text = line.partition(":")
        if not sep:
            continue
        text = text.strip()
        if not text:
            continue
        word_count = max(1, len(re.findall(r"\w+", text, flags=re.UNICODE)))
        duration = max(0.8, word_count * 0.38)
        start = round(cursor, 3)
        end = round(cursor + duration, 3)
        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker.strip(),
                "text": text,
            }
        )
        cursor = end + 0.12
    return {
        "audio_path": None,
        "language": "ro",
        "model": model,
        "segments": segments,
    }


def save_pair(out_root: Path, cid: str, pair: dict[str, Any], raw: str, usage: Any, *, model: str) -> None:
    (out_root / "transcripts" / f"{cid}.txt").write_text(
        pair["conversation"].strip() + "\n", encoding="utf-8"
    )
    (out_root / "conversations" / f"{cid}.json").write_text(
        json.dumps(
            transcript_to_conversation_json(pair["conversation"], model=model),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_root / "refs" / f"{cid}.json").write_text(
        json.dumps(pair["note"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "conversation_id": cid,
        "evidence": pair["evidence"],
        "usage": usage_to_dict(usage),
    }
    (out_root / "audits" / f"{cid}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_root / "raw" / f"{cid}.txt").write_text(raw.strip() + "\n", encoding="utf-8")


def build_index(out_root: Path, successful_ids: list[str]) -> None:
    available_ids = {
        path.stem
        for path in (out_root / "transcripts").glob("synth_*.txt")
        if (out_root / "refs" / f"{path.stem}.json").exists()
    }
    available_ids.update(successful_ids)

    def sort_key(cid: str) -> tuple[int, str]:
        match = re.search(r"\d+", cid)
        return (int(match.group()) if match else 10**9, cid)

    lines = ["conversation_id\ttranscript_path\tnote_path\tconversation_path"]
    for cid in sorted(available_ids, key=sort_key):
        transcript_path = out_root / "transcripts" / f"{cid}.txt"
        note_path = out_root / "refs" / f"{cid}.json"
        conversation_path = out_root / "conversations" / f"{cid}.json"
        if not transcript_path.exists() or not note_path.exists():
            continue
        lines.append(
            f"{cid}\t{transcript_path.relative_to(ROOT)}\t"
            f"{note_path.relative_to(ROOT)}\t{conversation_path.relative_to(ROOT)}"
        )
    (out_root / "index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def api_key_from_env() -> Optional[str]:
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="number of synthetic rows to generate")
    parser.add_argument("--start", type=int, default=1, help="1-based start row in the plan TSV")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between successful calls")
    parser.add_argument("--seed-id", action="append", dest="seed_ids", help="pool seed id; repeatable")
    parser.add_argument("--overwrite", action="store_true", help="regenerate rows even if files exist")
    parser.add_argument("--dry-run", action="store_true", help="validate setup and print prompt preview only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_ids = args.seed_ids or DEFAULT_SEED_IDS
    rows = parse_plan_rows(args.plan, start=args.start, n=args.n)

    seed_paths(seed_ids)
    for row in rows:
        expected_note_constraints(row)

    system_prompt = build_system_prompt(seed_ids)
    out_root = args.output_root.resolve()

    eprint(f"[setup] model={args.model}")
    eprint(f"[setup] output_root={out_root}")
    eprint(f"[setup] plan={args.plan}")
    eprint(f"[setup] seed_ids={', '.join(seed_ids)}")
    eprint(f"[setup] system_prompt_chars={len(system_prompt)}")

    if args.dry_run:
        preview = build_user_prompt(rows[0])
        eprint("[dry-run] first user prompt preview:")
        eprint(preview[:4000])
        return

    if anthropic is None:
        raise RuntimeError("anthropic package is not installed")
    api_key = api_key_from_env()
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY or ANTHROPIC_KEY in environment/.env")

    for subdir in ["transcripts", "conversations", "refs", "audits", "raw"]:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)
    write_seed_manifest(out_root, seed_ids)
    write_plan(out_root / "plan.tsv", rows)

    client = anthropic.Anthropic(api_key=api_key)
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for row in rows:
        cid = row["conversation_id"]
        if not args.overwrite and existing_pair_ok(out_root, row):
            eprint(f"[{cid}] skip existing valid pair")
            successes.append(cid)
            continue

        last_error = ""
        for attempt in range(1, args.max_retries + 2):
            try:
                raw, usage = call_claude(
                    client,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    system_prompt=system_prompt,
                    row=row,
                )
                (out_root / "raw").mkdir(parents=True, exist_ok=True)
                (out_root / "raw" / f"{cid}.attempt{attempt}.txt").write_text(
                    raw.strip() + "\n", encoding="utf-8"
                )
                pair = parse_and_validate_pair(raw, row)
                save_pair(out_root, cid, pair, raw, usage, model=args.model)
                successes.append(cid)
                usage_text = ""
                if usage is not None:
                    usage_text = (
                        f" input={getattr(usage, 'input_tokens', '?')}"
                        f" output={getattr(usage, 'output_tokens', '?')}"
                        f" cache_read={getattr(usage, 'cache_read_input_tokens', '?')}"
                    )
                eprint(f"[{cid}] saved attempt={attempt}{usage_text}")
                time.sleep(args.sleep)
                break
            except (json.JSONDecodeError, ParseError, SchemaError, GenerationValidationError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                eprint(f"[{cid}] invalid attempt={attempt}: {last_error}")
            except anthropic.APIError as exc:
                last_error = f"APIError: {exc}"
                eprint(f"[{cid}] api attempt={attempt}: {last_error}")
                time.sleep(max(args.sleep, 2.0))
        else:
            failures.append((cid, last_error))

    build_index(out_root, successes)
    write_reports(out_root, rows, successes)
    if failures:
        failure_lines = ["conversation_id\terror"]
        failure_lines.extend(f"{cid}\t{message}" for cid, message in failures)
        (out_root / "_failures.tsv").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    eprint(f"[done] successes={len(successes)}/{len(rows)} failures={len(failures)}")
    if failures:
        eprint(f"[done] failure log: {out_root / '_failures.tsv'}")


if __name__ == "__main__":
    main()
