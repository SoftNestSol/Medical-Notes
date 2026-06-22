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
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_PLAN = ROOT / "data" / "synthetic" / "GPT5.5" / "plan.tsv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "synthetic" / "Claude"
# Full POOL_COMPLETE_PAIR_IDS in a fixed numeric order so the per-block sliding
# window is deterministic. Rotation draws a subset of these per block.
DEFAULT_SEED_IDS = [
    "audio1", "audio4", "audio10", "audio18", "audio19",
    "audio21", "audio23", "audio24", "audio26", "audio29",
]

# Generations that share the same seed subset before the window slides on.
BLOCK_SIZE = 10

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
    "short_sparse": "550-850 cuvinte",
    "detailed_long": "800-1200 cuvinte",
}

ORAL_MARKERS = [
    "păi", "pai", "adică", "adica", "ăă", "mmm", "nu știu", "nu stiu",
    "cum să zic", "cum sa zic", "așa", "asa", "deci", "ok", "da", "nu",
    "înțeleg", "inteleg", "pe aici", "parcă", "parca",
]

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
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def content_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    stop = {
        "de", "la", "pe", "cu", "si", "și", "o", "un", "una", "unu", "unul",
        "cand", "când", "mai", "zi", "zile", "saptamana", "săptămână",
        "saptamani", "săptămâni", "ori", "doar", "cam", "vreo", "pentru",
    }
    return [token for token in tokens if len(token) > 1 and token not in stop]


def dose_is_supported(doza: str, support_text: str) -> bool:
    normalized_dose = normalize_text(doza)
    normalized_support = normalize_text(support_text)
    if normalized_dose in normalized_support:
        return True

    dose_tokens = content_tokens(doza)
    if not dose_tokens:
        return False
    support_tokens = set(content_tokens(support_text))

    numeric_tokens = [token for token in dose_tokens if any(char.isdigit() for char in token)]
    if numeric_tokens:
        return all(token in support_tokens for token in numeric_tokens)

    # For non-numeric doses ("un sfert de pastilă dimineața", "una singură"),
    # require most content words to appear in medication evidence/conversation.
    matched = sum(1 for token in dose_tokens if token in support_tokens)
    return matched >= max(1, min(len(dose_tokens), 2))


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


def transcript_lines(transcript: str) -> list[tuple[str, str]]:
    lines = []
    for raw in transcript.splitlines():
        speaker, sep, text = raw.partition(":")
        if sep and text.strip():
            lines.append((speaker.strip(), text.strip()))
    return lines


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def seed_style_profile(seed_ids: list[str]) -> str:
    blocks = []
    total_words = []
    for seed_id, conversation_path, _, _ in seed_paths(seed_ids):
        transcript = read_conversation_as_transcript(conversation_path)
        lines = transcript_lines(transcript)
        counts = [word_count(text) for _, text in lines]
        total_words.append(sum(counts))
        short_turns = sum(1 for count in counts if count <= 3)
        long_turns = sum(1 for count in counts if count >= 18)
        alternations = sum(
            1 for (speaker_a, _), (speaker_b, _) in zip(lines, lines[1:])
            if speaker_a != speaker_b
        )
        text_lc = transcript.lower()
        marker_hits = {
            marker: len(re.findall(rf"\b{re.escape(marker)}\b", text_lc))
            for marker in ORAL_MARKERS
        }
        marker_hits = {key: value for key, value in marker_hits.items() if value > 0}
        fragment_examples = [
            text for _, text in lines
            if len(text.split()) <= 8 or text.endswith("...") or text.lower() in {"da.", "nu.", "ok."}
        ][:8]
        blocks.append(
            "\n".join(
                [
                    f"- {seed_id}: {len(lines)} segments, {sum(counts)} words, "
                    f"{short_turns} very short turns, {long_turns} long turns, "
                    f"{alternations} speaker alternations",
                    f"  oral markers observed: {json.dumps(marker_hits, ensure_ascii=False)}",
                    f"  fragment examples: {json.dumps(fragment_examples, ensure_ascii=False)}",
                ]
            )
        )
    if total_words:
        blocks.append(
            f"- seed word-count range: {min(total_words)}-{max(total_words)}; "
            f"mean: {sum(total_words) // len(total_words)}"
        )
    return "\n".join(blocks)


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


def seed_subset_for_block(
    seed_ids: list[str], block_index: int, subset_size: int = 6
) -> list[str]:
    """Deterministic sliding window over seed_ids, wrapping at the end.

    block 0 -> indices 0..subset_size-1, block 1 -> 1..subset_size, etc.
    (mod len(seed_ids)). No randomness, fully reproducible. If subset_size
    is >= the pool size, the whole pool is returned (no rotation possible).
    """
    n = len(seed_ids)
    if n == 0:
        return []
    if subset_size >= n:
        return list(seed_ids)
    start = block_index % n
    return [seed_ids[(start + i) % n] for i in range(subset_size)]


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
    style_profile = seed_style_profile(seed_ids)
    schema = json.dumps(NOTE_SCHEMA, ensure_ascii=False, indent=2)
    return f"""Ești un generator de date sintetice pentru un studiu de bioNLP în limba română.

Produci perechi conversație–notă: o conversație sintetică între un terapeut (chiropractor/osteopat) și un pacient, plus nota structurată care se poate extrage EXCLUSIV din acea conversație.

## Regula fundamentală
Dacă o informație nu este rostită explicit în conversație, câmpul corespunzător din notă rămâne gol (null sau []). Fără inferență clinică, fără completare din context probabil. Dacă pacientul spune ceva vag care nu identifică clar o valoare, câmpul rămâne gol.

## Distribuția cazurilor
Inventează cazuri realiste, așa cum apar de fapt la un cabinet de chiropractică. Nu forța varietate artificială și nu încerca să acoperi afecțiuni rare. Lasă cazul să fie ce ar fi plauzibil: majoritatea pacienților vin cu dureri lombare, cervicale, de genunchi, umăr, apărute din efort, postură sau activitate. Antecedentele și medicația apar doar când e firesc pentru caz, nu ca să bifezi categorii.
Nu transforma întrebarea de screening ("aveți tensiune, diabet...?") într-un motiv ca pacientul să confirme mereu ceva. În multe consultații răspunsul natural este "nu", iar `antecedente` și `medicatie_actuala` rămân goale. Hipertensiunea NU este default: folosește-o rar, doar când profilul cerut pentru exemplu permite explicit asta. Evită să repeți aceleași combinații de medicamente (Enalapril/Prestarium/Nurofen/Ibuprofen) ca șablon.
Chiar și când antecedentele și medicația sunt negative, conversația nu trebuie să devină telegrafică: păstrează anamneză, clarificări despre debut, factori agravanți, funcție, teste simple și observații funcționale rostite.

## Exemplele seed — contractul tău de mapare
Mai jos ai perechi REALE conversație→notă, extrase din cabinet. Ele nu sunt doar mostre de stil. Sunt contractul care îți arată CUM se mapează vorbirea colocvială în câmpuri structurate. Studiază în special:
- ce a fost rostit dar a rămas necompletat în notă (vag, neclar, nesigur);
- ce formulare colocvială concretă a justificat completarea unui câmp;
- cum a fost (sau NU a fost) reflectat în `localizarea_durerii` un pacient care spune ceva ambiguu despre unde îl doare ("mă doare spatele", "pe aici");
- cum sună un VAS rostit vs. o durere descrisă fără cifră (care rămâne null);
- diferența dintre o boală pe care pacientul o confirmă clar și una atinsă în treacăt.
Reproduci aceeași disciplină de mapare în perechile pe care le generezi.
Dar incearca sa fii creativ și să inventezi cazuri noi, nu să copiezi exemplele seed.

## Stilul conversației
Imită mecanica dialogurilor seed, nu o idee abstractă de "oralitate":
- terapeutul conduce: întreabă scurt, clarifică, confirmă; pacientul răspunde colocvial, uneori incomplet, uneori pe lângă subiect;
- păstrează fragmentele scurte, confirmările monosilabice, reluările de întrebări și micile imperfecțiuni de transcript ASR pe care le vezi în seed-uri;
- pacientul vorbește ca pacient ("mă ține", "aici", "pe partea asta", "nu știu exact"), nu ca medic;
- terapeutul NU recită fișa cu voce tare. Observațiile funcționale apar ca interacțiune reală ("stai puțin pe piciorul drept... te ține?"), nu ca propoziție de raport medical citită pacientului;
- doar SPEAKER_00 (terapeut) și SPEAKER_01 (pacient).

## Extragerea notei
După ce scrii conversația, completezi nota DOAR din ce s-a rostit. Pentru fiecare câmp populat dai cel puțin un citat exact (substring real din conversație) care îl justifică. Pentru fiecare câmp gol, citatul e []. Dacă nu poți cita, nu completezi.

## Schema notei
{schema}

## Format de ieșire
Returnezi exact un obiect JSON, fără markdown, fără text în jur:
{{
  "conversation": "<linii SPEAKER_00/SPEAKER_01>",
  "note": {{... conform schemei ...}},
  "evidence": {{
    "motivul_prezentarii": ["citat exact din conversație", ...] sau [],
    "evaluarea_durerii_vas": [...] sau [],
    "localizarea_durerii": [...] sau [],
    "localizarea_durerii_alta": [...] sau [],
    "antecedente": [...] sau [],
    "antecedente_altele": [...] sau [],
    "medicatie_actuala": [...] sau [],
    "evaluare_functionala_initiala": [...] sau []
  }}
}}

Reguli evidence:
- fiecare câmp non-empty din notă are cel puțin un citat care e substring real din conversație;
- fiecare câmp gol are [];
- citatele sunt copiate exact, nu parafrazate.

## Profil de stil măsurat din seed-uri
{style_profile}

## Exemple seed (conversație + notă reală)
{seed_examples}
"""


def build_plan(n: int, *, start: int = 1) -> list[dict[str, str]]:
    """Minimal form-only plan: controls conversation length and patient register,
    plus broad background density. Clinical content is invented freely by the model.

    Distributions chosen to resemble a real clinic, not a gallery of extremes:
    - richness: ~60% short, ~40% long (most real visits are brief and to the point)
    - patient_style: weighted toward neutral (tacut 40 / vorbaret 30 / vag 20 / anxios 10)
    """
    if start < 1:
        raise ValueError("--start is 1-based and must be >= 1")
    richness_cycle = ["scurt_rar"] * 6 + ["detaliat_lung"] * 4
    style_cycle = (
        ["tacut"] * 4 + ["vorbaret"] * 3 + ["vag"] * 2 + ["anxios"] * 1
    )
    background_cycle = (
        ["fara_antecedente_fara_medicatie"] * 4
        + ["fara_antecedente_otc_rar"] * 2
        + ["istoric_minor_fara_medicatie"] * 2
        + ["cronic_non_hta"] * 1
        + ["hta_permisa_rar"] * 1
    )
    region_cycle = (
        ["lombar"] * 3
        + ["cervical_ceafa"] * 2
        + ["sacral_coccis"] * 1
        + ["toracal"] * 1
        + ["genunchi"] * 1
        + ["cot_pumn"] * 1
        + ["sold"] * 1
    )
    rows = []
    for i in range(start, start + n):
        rows.append(
            {
                "conversation_id": f"synth_{i:03d}",
                "richness": richness_cycle[(i - 1) % len(richness_cycle)],
                "patient_style": style_cycle[(i - 1) % len(style_cycle)],
                "background_profile": background_cycle[(i - 1) % len(background_cycle)],
                "region_profile": region_cycle[(i - 1) % len(region_cycle)],
            }
        )
    return rows


def build_user_prompt(row: dict[str, str]) -> str:
    richness_hint = {
        "scurt_rar": "scurt_rar = 550-850 cuvinte, consultație simplă, la obiect, dar cu anamneză suficientă",
        "detaliat_lung": "detaliat_lung = 800-1200 cuvinte, mai multe schimburi, anamneză mai bogată",
    }.get(row["richness"], "450-1000 cuvinte")
    style_hint = {
        "vorbaret": "vorbăreț: divaghează, dă context nesolicitat, revine la temeri",
        "tacut": "tăcut: răspunsuri scurte, monosilabice, terapeutul trebuie să insiste",
        "anxios": "anxios: îngrijorat, întreabă dacă e grav, nesigur pe răspunsuri",
        "vag": "vag: descrie impreciz, 'pe aici', 'nu știu exact', greu de fixat pe valori clare",
    }.get(row["patient_style"], "natural")
    background_hint = {
        "fara_antecedente_fara_medicatie": (
            "fără antecedente relevante și fără medicație actuală; la screening pacientul răspunde natural negativ"
        ),
        "fara_antecedente_otc_rar": (
            "fără antecedente relevante; poate menționa rar/ocazional un antiinflamator sau gel folosit la nevoie, dar nu forța"
        ),
        "istoric_minor_fara_medicatie": (
            "poate avea un istoric minor verbalizat (operație veche, fractură veche, scolioză ușoară), dar fără tratament curent"
        ),
        "cronic_non_hta": (
            "dacă apare o afecțiune cronică, evită hipertensiunea; poți folosi realist tiroidă, gastrită, colesterol, astm sau altceva comun"
        ),
        "hta_permisa_rar": (
            "hipertensiunea este permisă în acest exemplu, dar nu obligatorie; dacă apare, variază formularea și tratamentul"
        ),
    }.get(row["background_profile"], "natural, fără default de hipertensiune")
    region_hint = {
        "lombar": "durere lombară / lombosacrală; NU folosi umăr ca acuza principală",
        "cervical_ceafa": "durere cervicală, ceafă sau cap-ceafă; NU folosi umăr ca acuza principală",
        "sacral_coccis": "durere sacrală, coccis sau bazin jos; NU folosi umăr ca acuza principală",
        "toracal": "durere toracală / mijlocul spatelui; NU folosi umăr ca acuza principală",
        "genunchi": "durere de genunchi; NU folosi umăr ca acuza principală",
        "cot_pumn": "durere de cot sau pumn; NU folosi umăr ca acuza principală",
        "sold": "durere de șold; NU folosi umăr ca acuza principală",
    }.get(row["region_profile"], "evită să repeți umărul drept ca acuza principală")
    return f"""Generează o pereche conversație–notă nouă.

Formă țintă pentru acest exemplu (afectează DOAR lungimea și registrul, NU valorile clinice):
- bogăție: {richness_hint}
- stil pacient: {style_hint}
- profil antecedente/medicație: {background_hint}
- profil localizare: {region_hint}

Inventează liber cazul clinic (ce îl doare, de când, dacă ia medicație, ce antecedente are) astfel încât să fie plauzibil și natural pentru un cabinet de chiropractică. Lasă distribuția să fie realistă — nu forța afecțiuni rare.

Completează nota structurată STRICT din ce ai pus în conversație. Dacă o informație nu e rostită explicit, câmpul rămâne gol. Pentru fiecare câmp populat, dă un citat exact din conversație în evidence.

Nu include markdown fences. Returnează doar JSON valid cu cheile conversation, note, evidence.
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

    medication_support = "\n".join(evidence.get("medicatie_actuala", [])) + "\n" + conversation
    for med in note["medicatie_actuala"]:
        if normalize_text(med["denumire"]) not in normalized_conversation:
            raise GenerationValidationError(f"{row['conversation_id']} medication name not spoken: {med}")
        if med["doza"] and not dose_is_supported(med["doza"], medication_support):
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
    if not 320 <= len(words) <= 1300:
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
    if not is_populated(note["motivul_prezentarii"]):
        raise GenerationValidationError(f"{row['conversation_id']} missing motivul_prezentarii")
    validate_conversation_shape(parsed["conversation"], row)
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
    plan_lines = ["conversation_id\trichness\tpatient_style\tstatus"]
    for cid in successful_ids:
        row = plan_by_id[cid]
        plan_lines.append(
            "\t".join(
                [
                    cid,
                    row["richness"],
                    row["patient_style"],
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
    parser.add_argument("--start", type=int, default=1, help="1-based start index for synth_NNN ids")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="parallel API calls; keep <= Anthropic rate limits")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between successful calls")
    parser.add_argument("--seed-id", action="append", dest="seed_ids", help="pool seed id; repeatable")
    parser.add_argument("--overwrite", action="store_true", help="regenerate rows even if files exist")
    parser.add_argument("--dry-run", action="store_true", help="validate setup and print prompt preview only")
    return parser.parse_args()


def process_row(
    *,
    row: dict[str, str],
    out_root: Path,
    api_key: str,
    model: str,
    max_tokens: int,
    max_retries: int,
    sleep_seconds: float,
    system_prompt: str,
) -> tuple[str, bool, str]:
    cid = row["conversation_id"]
    client = anthropic.Anthropic(api_key=api_key)
    last_error = ""
    for attempt in range(1, max_retries + 2):
        try:
            raw, usage = call_claude(
                client,
                model=model,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                row=row,
            )
            (out_root / "raw").mkdir(parents=True, exist_ok=True)
            (out_root / "raw" / f"{cid}.attempt{attempt}.txt").write_text(
                raw.strip() + "\n", encoding="utf-8"
            )
            pair = parse_and_validate_pair(raw, row)
            save_pair(out_root, cid, pair, raw, usage, model=model)
            usage_text = ""
            if usage is not None:
                usage_text = (
                    f" input={getattr(usage, 'input_tokens', '?')}"
                    f" output={getattr(usage, 'output_tokens', '?')}"
                    f" cache_read={getattr(usage, 'cache_read_input_tokens', '?')}"
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return cid, True, f"saved attempt={attempt}{usage_text}"
        except (json.JSONDecodeError, ParseError, SchemaError, GenerationValidationError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            eprint(f"[{cid}] invalid attempt={attempt}: {last_error}")
        except anthropic.APIError as exc:
            last_error = f"APIError: {exc}"
            eprint(f"[{cid}] api attempt={attempt}: {last_error}")
            wait_seconds = max(sleep_seconds, 2.0)
            if "rate_limit" in str(exc).lower() or "429" in str(exc):
                wait_seconds = max(wait_seconds, min(60.0, 10.0 * attempt))
                eprint(f"[{cid}] rate limited; sleeping {wait_seconds:.0f}s before retry")
            time.sleep(wait_seconds)
    return cid, False, last_error


def round_robin_jobs_by_block(
    jobs: list[tuple[int, dict[str, str]]]
) -> list[tuple[int, dict[str, str]]]:
    jobs_by_block: dict[int, list[tuple[int, dict[str, str]]]] = {}
    for job in jobs:
        jobs_by_block.setdefault(job[0], []).append(job)

    ordered: list[tuple[int, dict[str, str]]] = []
    block_ids = sorted(jobs_by_block)
    while any(jobs_by_block.values()):
        for block_id in block_ids:
            if jobs_by_block[block_id]:
                ordered.append(jobs_by_block[block_id].pop(0))
    return ordered


def main() -> None:
    args = parse_args()
    seed_ids = args.seed_ids or DEFAULT_SEED_IDS
    rows = build_plan(args.n, start=args.start)
    row_ids = [row["conversation_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError(f"duplicate conversation IDs in generation plan: {row_ids}")

    seed_paths(seed_ids)

    out_root = args.output_root.resolve()

    eprint(f"[setup] model={args.model}")
    eprint(f"[setup] output_root={out_root}")
    eprint(f"[setup] seed pool ({len(seed_ids)}): {', '.join(seed_ids)}")
    eprint(f"[setup] block_size={BLOCK_SIZE} subset_size=6")
    eprint(f"[setup] workers={args.workers}")

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

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    prompt_by_block: dict[int, str] = {}
    for offset, _row in enumerate(rows):
        global_row_number = args.start + offset
        block_index = (global_row_number - 1) // BLOCK_SIZE
        if block_index not in prompt_by_block:
            block_seeds = seed_subset_for_block(seed_ids, block_index)
            prompt_by_block[block_index] = build_system_prompt(block_seeds)
            eprint(f"[block {block_index}] seeds: {', '.join(block_seeds)}")

    jobs: list[tuple[int, dict[str, str]]] = []
    for offset, row in enumerate(rows):
        global_row_number = args.start + offset
        block_index = (global_row_number - 1) // BLOCK_SIZE
        cid = row["conversation_id"]
        if not args.overwrite and existing_pair_ok(out_root, row):
            eprint(f"[{cid}] skip existing valid pair")
            successes.append(cid)
            continue
        jobs.append((block_index, row))
    jobs = round_robin_jobs_by_block(jobs)

    workers = max(1, args.workers)
    if workers == 1:
        for block_index, row in jobs:
            cid, ok, message = process_row(
                row=row,
                out_root=out_root,
                api_key=api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep,
                system_prompt=prompt_by_block[block_index],
            )
            if ok:
                successes.append(cid)
                eprint(f"[{cid}] {message}")
            else:
                failures.append((cid, message))
    else:
        eprint(f"[parallel] submitting {len(jobs)} jobs with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_cid = {
                executor.submit(
                    process_row,
                    row=row,
                    out_root=out_root,
                    api_key=api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    sleep_seconds=args.sleep,
                    system_prompt=prompt_by_block[block_index],
                ): row["conversation_id"]
                for block_index, row in jobs
            }
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    result_cid, ok, message = future.result()
                except Exception as exc:  # defensive: preserve other jobs
                    failures.append((cid, f"{type(exc).__name__}: {exc}"))
                    eprint(f"[{cid}] worker crash: {type(exc).__name__}: {exc}")
                    continue
                if ok:
                    successes.append(result_cid)
                    eprint(f"[{result_cid}] {message}")
                else:
                    failures.append((result_cid, message))

    build_index(out_root, successes)
    write_reports(out_root, rows, successes)
    failure_path = out_root / "_failures.tsv"
    if failures:
        failure_lines = ["conversation_id\terror"]
        failure_lines.extend(f"{cid}\t{message}" for cid, message in failures)
        failure_path.write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    elif failure_path.exists():
        failure_path.unlink()

    eprint(f"[done] successes={len(successes)}/{len(rows)} failures={len(failures)}")
    if failures:
        eprint(f"[done] failure log: {failure_path}")


if __name__ == "__main__":
    main()
