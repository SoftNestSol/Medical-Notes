"""Generate diversified synthetic Romanian chiropractor note/transcript pairs.

The generator is intentionally plan-first:
1. measure target populate rates from all available non-test pool refs;
2. write a 50-row variation plan to synthetic/plan.tsv;
3. render conversations from unique dialogue-act sequences;
4. validate schema, extraction contract, coverage floors, and masked diversity.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_split import POOL_IDS, TEST_IDS, assert_no_test_leakage  # noqa: E402
from SOTA_EVALUATION.json_schema import ANTECEDENTE_ENUM, LOCALIZARE_ENUM, NOTE_SCHEMA  # noqa: E402

try:
    from jsonschema import validate
except ImportError:  # pragma: no cover
    validate = None


SEED_IDS = ["audio18", "audio19", "audio23", "audio26", "audio21"]
FIELDS_FOR_RATES = [
    "motivul_prezentarii",
    "evaluarea_durerii_vas",
    "localizarea_durerii",
    "localizarea_durerii_alta",
    "antecedente",
    "antecedente_altele",
    "medicatie_actuala",
    "evaluare_functionala_initiala",
]

ACTS = [
    "greeting",
    "age",
    "complaint",
    "duration",
    "onset",
    "aggravating_factors",
    "daily_limitation",
    "VAS",
    "antecedente",
    "meds",
    "exam_movement",
    "exam_palpation",
    "functional_obs",
    "closing",
]

REGION_LABELS = {
    "cervical": "zona cervicală",
    "toracal": "zona toracală",
    "lombar": "zona lombară",
    "sacral_coccis": "sacru/coccis",
    "umar_dr": "umărul drept",
    "umar_stg": "umărul stâng",
    "cot_dr": "cotul drept",
    "cot_stg": "cotul stâng",
    "pumn_dr": "pumnul drept",
    "pumn_stg": "pumnul stâng",
    "sold_dr": "șoldul drept",
    "sold_stg": "șoldul stâng",
    "genunchi_dr": "genunchiul drept",
    "genunchi_stg": "genunchiul stâng",
    "glezna_dr": "glezna dreaptă",
    "glezna_stg": "glezna stângă",
    "cap_ceafa": "cap/ceafă",
    "abdomen": "abdomen",
    "torace": "torace",
}

REGION_PLANS = [
    ["cervical", "cap_ceafa"], ["toracal"], ["lombar"], ["sacral_coccis"],
    ["umar_dr"], ["umar_stg"], ["cot_dr"], ["cot_stg"], ["pumn_dr"],
    ["pumn_stg"], ["sold_dr"], ["sold_stg"], ["genunchi_dr"],
    ["genunchi_stg"], ["glezna_dr"], ["glezna_stg"], ["abdomen"],
    ["torace"], ["cervical", "umar_dr"], ["toracal", "lombar"],
    ["lombar", "sold_stg"], ["sacral_coccis", "lombar"],
    ["umar_stg", "cot_stg"], ["cot_dr", "pumn_dr"],
    ["pumn_stg", "umar_stg"], ["sold_dr", "genunchi_dr"],
    ["sold_stg", "glezna_stg"], ["genunchi_dr", "glezna_dr"],
    ["genunchi_stg", "sold_stg"], ["cap_ceafa", "toracal"],
    ["abdomen", "lombar"], ["torace", "umar_dr"], ["cervical"],
    ["lombar"], ["umar_stg"], ["glezna_dr"], ["cot_stg"],
    ["genunchi_dr"], ["sold_dr", "lombar"], ["toracal"],
    ["cap_ceafa", "cervical"], ["pumn_dr"], ["pumn_stg"],
    ["sacral_coccis"], ["abdomen"], ["torace"], ["umar_dr", "cervical"],
    ["genunchi_stg", "glezna_stg"], ["lombar", "toracal"],
    ["glezna_stg", "pumn_dr"],
]

ACTIVITIES = [
    "grădinărit", "condus pe distanțe lungi", "lucru la laptop", "alergare ușoară",
    "ridicat copilul", "curățenie în casă", "mers pe munte", "fotbal recreațional",
    "gătit mult în picioare", "cărat dosare", "mers cu bicicleta", "stat în ședințe",
    "dans", "lucru în atelier", "înot", "navetă cu autobuzul", "exerciții acasă",
    "mutat mobilă", "tenis", "plimbări lungi", "pescuit", "zugrăvit", "cumpărături",
    "ridicat greutăți", "îngrijit curtea", "lucru la casă", "urcat scări", "volan",
    "yoga", "mers rapid", "cărat rucsac", "lucru în depozit", "stat la birou",
    "joc cu copiii", "drum cu trenul", "antrenament de forță", "mers pe tocuri",
    "lucru cu mouse-ul", "curierat", "reparații mici", "spălat geamuri",
    "schimbat roată", "scris la tastatură", "împins cărucior", "somn pe canapea",
    "cărat apă", "stat aplecat", "transport bagaje", "drumeție scurtă", "lucru manual",
]

MED_PLANS = {
    5: [{"denumire": "Nurofen", "doza": "200 mg"}],
    8: [{"denumire": "Paracetamol", "doza": "500 mg"}],
    11: [{"denumire": "Arcoxia", "doza": "60 mg"}],
    14: [{"denumire": "Voltaren gel", "doza": "aplicare locală de două ori pe zi"}],
    17: [{"denumire": "Aspenter", "doza": "75 mg"}],
    20: [{"denumire": "Magne B6", "doza": "1 comprimat pe zi"}],
    23: [{"denumire": "Ibuprofen", "doza": "400 mg"}],
    26: [{"denumire": "Diclac gel", "doza": "aplicare locală seara"}],
    29: [{"denumire": "Algocalmin", "doza": "500 mg"}],
    32: [{"denumire": "Euthyrox", "doza": "50 mcg"}],
    35: [{"denumire": "Naproxen", "doza": None}],
    38: [{"denumire": "Detralex", "doza": None}],
}

ANTECEDENTE_PLANS = {
    3: ["hipertensiune_arteriala"],
    6: ["diabet_zaharat"],
    9: ["hernia_disc"],
    12: ["scolioza_cifoza"],
    15: ["artrita_artroza"],
    18: ["osteoporoza"],
    21: ["boli_cardiovasculare"],
    24: ["epilepsie"],
    27: ["cancer_neoplasm"],
    30: ["boli_autoimune"],
    33: ["hipertensiune_arteriala", "diabet_zaharat"],
    36: ["hernia_disc", "scolioza_cifoza"],
    39: ["artrita_artroza", "osteoporoza"],
}

ANTECEDENTE_LABELS = {
    "hipertensiune_arteriala": "hipertensiune arterială",
    "diabet_zaharat": "diabet zaharat",
    "boli_cardiovasculare": "boală cardiovasculară",
    "osteoporoza": "osteoporoză",
    "artrita_artroza": "artrită sau artroză",
    "hernia_disc": "hernie de disc",
    "scolioza_cifoza": "scolioză sau cifoză",
    "epilepsie": "epilepsie",
    "cancer_neoplasm": "cancer/neoplasm",
    "boli_autoimune": "boală autoimună",
}

ANTECEDENTE_ALTELE = {
    13: "tiroidită Hashimoto",
    28: "astm bronșic",
    44: "gastrită cronică",
}

LOCALIZARE_ALTA = {
    16: "coapsa dreaptă",
    31: "fesier stâng",
    47: "antebraț drept",
}

DURATION_DETAIL = {
    "acut": ["de ieri", "de două zile", "de aproximativ trei zile"],
    "subacut": ["de aproape două săptămâni", "de vreo zece zile", "cam de trei săptămâni"],
    "cronic": ["de câteva luni", "de mai bine de jumătate de an", "de mult timp, cu episoade mai rele"],
}

ONSET_NOTE = {
    "efort": ["după ridicare de bagaje", "după exerciții intense", "după cărat cumpărături pe scări"],
    "postură-birou": ["după lucru prelungit la calculator", "după ședere lungă pe scaun", "după utilizare prelungită a laptopului"],
    "cădere": ["după alunecare pe trepte", "după sprijin greșit al piciorului", "după căzătură ușoară"],
    "treptat": ["cu debut progresiv", "apărută treptat", "cu instalare lentă"],
    "accident": ["după frână bruscă în mașină", "după lovitură minoră la sport", "după incident la bicicletă"],
    "necunoscut": ["fără declanșator clar", "cu debut neclar", "fără cauză identificată"],
}

STYLE_FILLERS = {
    "talkative": ["păi", "adică", "ca să explic"],
    "terse": ["da", "cam așa", "pe scurt"],
    "anxious": ["sincer", "mă cam sperie", "mi-e teamă"],
    "vague": ["nu știu exact", "cumva", "pe aici"],
    "precise": ["mai exact", "în special", "la mișcarea asta"],
}

CUES = [
    "în evaluarea de azi",
    "la verificarea inițială",
    "în discuția curentă",
    "pentru fișa de acum",
    "în relatarea de azi",
    "la acest consult",
    "în testarea de față",
    "pentru episodul actual",
    "în cabinet acum",
    "la controlul de astăzi",
    "în descrierea pacientului",
    "în examinarea inițială",
    "la proba de mobilitate",
    "în notarea de azi",
    "pentru această ședință",
    "în dialogul de acum",
    "la prima verificare",
    "în observația curentă",
    "pentru consemnarea actuală",
    "în analiza de azi",
    "la testarea scurtă",
    "în etapa de evaluare",
    "pentru tabloul actual",
    "în consultul prezent",
    "la verificarea de început",
]


def is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return len(value) > 0
    return True


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def measure_pool_rates() -> tuple[list[str], dict[str, float]]:
    ref_dir = ROOT / "data" / "chiropractor_ro" / "refs"
    # "All available pool notes" means every structured ref that is not held out.
    # Today this resolves to the five hand-corrected POOL refs, but the scan is not seed-based.
    pool_ref_ids = sorted(p.stem for p in ref_dir.glob("*.json") if p.stem not in TEST_IDS)
    assert_no_test_leakage(pool_ref_ids, context="rate measurement")
    if not pool_ref_ids:
        raise RuntimeError("No accessible pool refs found; cannot measure target rates.")
    rates = {}
    for field in FIELDS_FOR_RATES:
        rates[field] = sum(is_populated(load_json(ref_dir / f"{audio_id}.json")[field]) for audio_id in pool_ref_ids) / len(pool_ref_ids)
    return pool_ref_ids, rates


def join_regions(regions: list[str]) -> str:
    labels = [REGION_LABELS[r] for r in regions]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " și " + labels[-1]


def transcript_region_text(row: dict[str, Any]) -> str:
    if row["include_location"] == "no":
        return "zona dureroasă"
    return join_regions(row["regions"].split(","))


def region_subject(row: dict[str, Any]) -> tuple[str, str]:
    region = transcript_region_text(row)
    if "," in row["regions"] or " și " in region:
        return region, "se liniștesc"
    if region == "abdomen":
        return "abdomenul", "se liniștește"
    return region, "se liniștește"


def spoken_region(row: dict[str, Any]) -> str:
    base = transcript_region_text(row)
    if row["localizare_alta_plan"]:
        return f"{base} și {row['localizare_alta_plan']}"
    return base


def duration_line(row: dict[str, Any]) -> str:
    options = DURATION_DETAIL[row["duration"]]
    return options[int(row["conversation_id"][-2:]) % len(options)]


def sentence_start(text: str) -> str:
    return text[:1].upper() + text[1:]


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def cue(row: dict[str, Any], act: str) -> str:
    act_offset = ACTS.index(act) if act in ACTS else 0
    cid = int(row["conversation_id"][-2:])
    return CUES[(cid + act_offset * 3) % len(CUES)]


def decorate_line(line: str, row: dict[str, Any], act: str) -> str:
    speaker, sep, text = line.partition(": ")
    if not sep or len(text.split()) < 2:
        return line
    marker = cue(row, act)
    if text.endswith("?"):
        text = f"{sentence_start(marker)}, {lower_first(text)}"
    elif text.endswith("."):
        text = f"{text[:-1]}, {marker}."
    else:
        text = f"{text}, {marker}"
    return f"{speaker}: {text}"


def decorate_lines(lines: list[str], row: dict[str, Any], act: str) -> list[str]:
    return [decorate_line(line, row, act) for line in lines]


def neutral_complaint(row: dict[str, Any]) -> str:
    regions = row["regions"].split(",")
    onset = ONSET_NOTE[row["onset"]][int(row["conversation_id"][-2:]) % 3]
    activity = row["activity"]
    if row["include_location"] == "no":
        return f"durere musculo-scheletală difuză, {onset}, asociată cu {activity}"
    location = join_regions(regions)
    if row["localizare_alta_plan"]:
        location = f"{location} și {row['localizare_alta_plan']}"
    return f"durere la {location}, {onset}, asociată cu {activity}"


def functional_observation(row: dict[str, Any]) -> str | None:
    if row["func_eval"] == "null":
        return None
    regions = row["regions"].split(",")
    region = join_regions(regions)
    n = int(row["conversation_id"][-2:])
    if any(r in {"lombar", "sacral_coccis", "sold_dr", "sold_stg"} for r in regions):
        variants = [
            f"La aplecarea trunchiului, durerea la {region} se reproduce înainte de capătul mișcării; palparea locală accentuează simptomul, {cue(row, 'functional_obs')}.",
            f"Mobilitatea trunchiului este limitată de durere la flexie, cu sensibilitate clară la presiune în {region}, {cue(row, 'functional_obs')}.",
            f"La transferul greutății și la presiune locală, durerea de la {region} apare constant și oprește mișcarea, {cue(row, 'functional_obs')}.",
        ]
    elif any(r in {"cervical", "toracal", "cap_ceafa", "torace", "abdomen"} for r in regions):
        variants = [
            f"Rotirea controlată este limitată de durere în {region}; palparea reproduce simptomul descris, {cue(row, 'functional_obs')}.",
            f"La mișcarea ghidată apare limitare de amplitudine, cu sensibilitate locală la testarea pentru {region}, {cue(row, 'functional_obs')}.",
            f"Respirația amplă sau rotația lentă accentuează durerea din {region}, iar presiunea locală o reproduce, {cue(row, 'functional_obs')}.",
        ]
    else:
        variants = [
            f"Durerea la {region} apare la mișcare activă și se accentuează la testarea contra-rezistență, {cue(row, 'functional_obs')}.",
            f"Presiunea locală pe {region} este tolerată parțial, însă mișcarea provocată reproduce simptomul descris, {cue(row, 'functional_obs')}.",
            f"La comparația bilaterală pentru {region}, partea simptomatică are durere mai clară la rezistență ușoară, {cue(row, 'functional_obs')}.",
        ]
    return variants[n % len(variants)]


def required_acts(row: dict[str, Any]) -> set[str]:
    acts = {"complaint"}
    if row["vas"] != "null":
        acts.add("VAS")
    if row["antecedente_plan"] != "none" or row["antecedente_altele_plan"]:
        acts.add("antecedente")
    if row["meds_plan"] in {"named", "unnamed-pills"}:
        acts.add("meds")
    if row["func_eval"] == "populated":
        acts.update({"exam_movement", "exam_palpation", "functional_obs"})
    return acts


def act_sequence(row: dict[str, Any]) -> str:
    cid = int(row["conversation_id"][-2:])
    req = required_acts(row)
    disabled: set[str] = set()
    if row["func_eval"] == "null":
        disabled.update({"exam_movement", "exam_palpation", "functional_obs"})
    drop_count = 2 + (cid % 3)
    protected = set(req)
    if row["vas"] == "null":
        protected.add("VAS")  # tempting empty, no number
    if row["meds_plan"] == "unnamed-pills":
        protected.add("meds")
    optional = [act for act in ACTS if act not in protected and act not in disabled]
    drops = {optional[(cid * 5 + j * 7) % len(optional)] for j in range(drop_count)}
    selected = [act for act in ACTS if act not in drops and act not in disabled]

    # Move blocks around at plan level; do not just rotate one fixed script.
    transforms = cid % 8
    if transforms == 0:
        order = selected
    elif transforms == 1:
        order = ["greeting", "complaint", "age", "duration", "VAS", "aggravating_factors", "antecedente", "meds", "exam_movement", "exam_palpation", "functional_obs", "daily_limitation", "closing"]
    elif transforms == 2:
        order = ["complaint", "greeting", "duration", "onset", "age", "daily_limitation", "VAS", "meds", "antecedente", "exam_palpation", "exam_movement", "functional_obs", "closing"]
    elif transforms == 3:
        order = ["greeting", "age", "duration", "complaint", "onset", "antecedente", "aggravating_factors", "VAS", "exam_movement", "meds", "exam_palpation", "functional_obs", "daily_limitation", "closing"]
    elif transforms == 4:
        order = ["complaint", "duration", "daily_limitation", "greeting", "age", "VAS", "aggravating_factors", "meds", "antecedente", "exam_movement", "exam_palpation", "functional_obs", "closing"]
    elif transforms == 5:
        order = ["greeting", "onset", "complaint", "aggravating_factors", "duration", "meds", "VAS", "age", "antecedente", "exam_palpation", "exam_movement", "functional_obs", "closing"]
    elif transforms == 6:
        order = ["age", "greeting", "complaint", "VAS", "duration", "daily_limitation", "antecedente", "exam_movement", "exam_palpation", "meds", "functional_obs", "closing"]
    else:
        order = ["greeting", "daily_limitation", "complaint", "duration", "aggravating_factors", "age", "meds", "antecedente", "VAS", "exam_movement", "exam_palpation", "functional_obs", "closing"]

    final = [act for act in order if act in selected]
    for act in selected:
        if act not in final:
            final.append(act)
    return ",".join(final)


def build_plan(target_rates: dict[str, float]) -> list[dict[str, Any]]:
    durations = ["acut", "subacut", "cronic"]
    onsets = ["efort", "postură-birou", "cădere", "treptat", "accident", "necunoscut"]
    styles = ["talkative", "terse", "anxious", "vague", "precise"]
    richness = ["short_sparse", "detailed_long"]
    vas_values = ["null"] * 5 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4 + [0]
    sparse_shapes = {
        1: "motiv_only",
        2: "motiv_vas",
        4: "motiv_location",
        7: "motiv_vas_location",
        10: "motiv_location_func",
        41: "motiv_vas",
    }
    rows = []
    sequences: set[str] = set()
    for i in range(50):
        one_based = i + 1
        regions = REGION_PLANS[i]
        assert all(region in LOCALIZARE_ENUM for region in regions)
        meds_plan = "none"
        if one_based in MED_PLANS:
            meds_plan = "named"
        elif one_based in {6, 19, 34, 43}:
            meds_plan = "unnamed-pills"
        note_shape = sparse_shapes.get(one_based, "rich" if one_based in {13, 16, 28, 31, 33, 36, 44, 47} else "standard")
        include_location = note_shape not in {"motiv_only", "motiv_vas"}
        include_vas = note_shape not in {"motiv_only", "motiv_location", "motiv_location_func"}
        include_func = note_shape not in {"motiv_only", "motiv_vas", "motiv_location", "motiv_vas_location"}
        row = {
            "conversation_id": f"synth_{one_based:02d}",
            "batch": str(i // 10 + 1),
            "regions": ",".join(regions),
            "vas": str(vas_values[i]) if include_vas else "null",
            "duration": durations[(i + i // 5) % len(durations)],
            "age": str(18 + ((i * 17) % 63)),
            "onset": onsets[(i * 2 + i // 7) % len(onsets)],
            "antecedente_plan": ",".join(ANTECEDENTE_PLANS.get(one_based, [])) or "none",
            "antecedente_altele_plan": ANTECEDENTE_ALTELE.get(one_based, ""),
            "localizare_alta_plan": LOCALIZARE_ALTA.get(one_based, "") if include_location else "",
            "meds_plan": meds_plan,
            "meds_detail": json.dumps(MED_PLANS.get(one_based, []), ensure_ascii=False),
            "func_eval": "populated" if include_func else "null",
            "include_location": "yes" if include_location else "no",
            "note_shape": note_shape,
            "richness": richness[(i + i // 3) % 2],
            "patient_style": styles[(i * 3 + i // 4) % len(styles)],
            "opening": "patient leads" if i % 5 in {1, 4} else "therapist leads",
            "activity": ACTIVITIES[i],
        }
        seq = act_sequence(row)
        suffix = 0
        while seq in sequences:
            row["opening"] = f"variant_{suffix}"
            seq = act_sequence(row)
            acts = seq.split(",")
            if len(acts) > 4:
                acts[1], acts[-2] = acts[-2], acts[1]
            seq = ",".join(acts)
            suffix += 1
        row["dialogue_act_order"] = seq
        sequences.add(seq)
        rows.append(row)
    assert len(sequences) == 50
    return rows


def note_from_plan(row: dict[str, Any]) -> dict[str, Any]:
    antecedente = [] if row["antecedente_plan"] == "none" else row["antecedente_plan"].split(",")
    for item in antecedente:
        assert item in ANTECEDENTE_ENUM
    note = {
        "motivul_prezentarii": neutral_complaint(row),
        "evaluarea_durerii_vas": None if row["vas"] == "null" else int(row["vas"]),
        "localizarea_durerii": row["regions"].split(",") if row["include_location"] == "yes" else [],
        "localizarea_durerii_alta": row["localizare_alta_plan"] or None,
        "antecedente": antecedente,
        "antecedente_altele": row["antecedente_altele_plan"] or None,
        "medicatie_actuala": json.loads(row["meds_detail"]),
        "evaluare_functionala_initiala": functional_observation(row),
    }
    assert "după ce am" not in note["motivul_prezentarii"]
    assert "după ce a " not in note["motivul_prezentarii"]
    assert not re.search(r"\bzonei zona\b|\bzona zona\b", json.dumps(note, ensure_ascii=False), re.IGNORECASE)
    return note


def filler(row: dict[str, Any]) -> str:
    options = STYLE_FILLERS[row["patient_style"]]
    return options[int(row["conversation_id"][-2:]) % len(options)]


def act_lines(row: dict[str, Any], act: str, note: dict[str, Any]) -> list[str]:
    cid = int(row["conversation_id"][-2:])
    region = transcript_region_text(row)
    full_region = spoken_region(row)
    duration = duration_line(row)
    f = filler(row)
    if act == "greeting":
        variants = [
            ["SPEAKER_00: Bună ziua, poftiți.", "SPEAKER_01: Bună ziua."],
            ["SPEAKER_01: Bună ziua, am intrat pentru consultație.", "SPEAKER_00: Sigur, luăm pe rând ce vă supără."],
            ["SPEAKER_00: Vă rog, luați loc câteva secunde.", "SPEAKER_01: Mulțumesc, încerc să stau comod."],
            ["SPEAKER_01: Bună, cred că sunt programat acum.", "SPEAKER_00: Da, haideți să începem discuția."],
        ]
        return variants[cid % len(variants)]
    if act == "age":
        variants = [
            [f"SPEAKER_00: Îmi spuneți vârsta, vă rog?", f"SPEAKER_01: Am {row['age']} de ani."],
            [f"SPEAKER_01: Ca date generale, am {row['age']} de ani.", "SPEAKER_00: Am notat."],
            [f"SPEAKER_00: Pentru fișă, ce vârstă trecem?", f"SPEAKER_01: {row['age']} de ani împliniți."],
        ]
        return variants[cid % len(variants)]
    if act == "complaint":
        # The activity is intentionally confined to this one complaint/onset line.
        if row["include_location"] == "yes":
            return [
                f"SPEAKER_01: {sentence_start(f)}, motivul principal este durerea la {full_region}, apărută după {row['activity']}.",
                "SPEAKER_00: Deci reținem doar zona pe care o indicați clar ca dureroasă.",
            ]
        return [
            f"SPEAKER_01: {sentence_start(f)}, mă doare difuz, fără să pot alege o zonă exactă, după {row['activity']}.",
            "SPEAKER_00: Dacă nu este o localizare precisă, nu o transformăm într-o bifă anatomică.",
        ]
    if act == "duration":
        variants = [
            [f"SPEAKER_00: De cât timp se simte durerea?", f"SPEAKER_01: {sentence_start(duration)} o simt mai clar."],
            [f"SPEAKER_01: Ca timp, e cam {duration}.", "SPEAKER_00: Bun, păstrăm durata ca relatare separată."],
            [f"SPEAKER_00: Este ceva recent sau mai vechi?", f"SPEAKER_01: Aș zice {duration}, cu variații pe parcursul zilei."],
        ]
        return variants[cid % len(variants)]
    if act == "onset":
        variants = [
            ["SPEAKER_00: A fost un moment clar de debut?", "SPEAKER_01: Da, pot lega începutul de un episod concret."],
            ["SPEAKER_01: Nu a pornit ca o durere mare, a crescut pe parcurs.", "SPEAKER_00: Am înțeles, debutul a fost relatat separat de intensitate."],
            ["SPEAKER_00: S-a instalat brusc sau treptat?", "SPEAKER_01: Mai degrabă treptat, apoi a devenit greu de ignorat."],
        ]
        return variants[cid % len(variants)]
    if act == "aggravating_factors":
        variants = [
            [f"SPEAKER_00: Ce o face să se accentueze în {region}?", "SPEAKER_01: Mișcarea repetată și menținerea poziției o fac mai supărătoare."],
            [f"SPEAKER_01: Dacă forțez zona, durerea din {region} se aprinde mai repede.", "SPEAKER_00: Deci solicitarea o crește, nu doar atingerea."],
            [f"SPEAKER_00: În repaus scade sau rămâne la fel în {region}?", "SPEAKER_01: Scade în repaus, dar revine când reiau mișcarea."],
        ]
        return variants[cid % len(variants)]
    if act == "daily_limitation":
        variants = [
            [f"SPEAKER_00: Ce ați început să evitați din cauza durerii la {region}?", "SPEAKER_01: Evit mișcările ample și fac pauze mai dese."],
            [f"SPEAKER_01: Mă încurcă la lucruri obișnuite, nu doar la efort mare.", "SPEAKER_00: Bun, notăm că vă limitează funcțional în ziua curentă."],
            [f"SPEAKER_00: Vă trezește noaptea sau doar vă oprește ziua?", "SPEAKER_01: Mai mult ziua mă oprește; noaptea nu e partea principală."],
        ]
        return variants[cid % len(variants)]
    if act == "VAS":
        if row["vas"] != "null":
            variants = [
                [f"SPEAKER_00: Pe o scală de la zero la zece, cât are durerea acum?", f"SPEAKER_01: Aș spune {row['vas']}."],
                [f"SPEAKER_01: Dacă trebuie să aleg un număr, pun {row['vas']} din zece.", "SPEAKER_00: Perfect, acesta este scorul verbalizat."],
                [f"SPEAKER_00: Ce număr trecem pentru intensitatea actuală?", f"SPEAKER_01: {row['vas']}, nu mai mult în momentul ăsta."],
            ]
            return variants[cid % len(variants)]
        return [
            "SPEAKER_00: Ați putea pune durerea pe o scară numerică?",
            f"SPEAKER_01: Nu prea pot să aleg un număr; {f}, știu doar că mă deranjează.",
        ]
    if act == "antecedente":
        antecedente = [] if row["antecedente_plan"] == "none" else row["antecedente_plan"].split(",")
        if antecedente or row["antecedente_altele_plan"]:
            spoken = [ANTECEDENTE_LABELS[item] for item in antecedente]
            if row["antecedente_altele_plan"]:
                spoken.append(row["antecedente_altele_plan"])
            variants = [
                ["SPEAKER_00: Aveți afecțiuni cunoscute pe care ar trebui să le trecem la antecedente?", f"SPEAKER_01: Da, am {', '.join(spoken)}."],
                [f"SPEAKER_01: La istoricul meu medical apar {', '.join(spoken)}.", "SPEAKER_00: Am notat exact ce ați spus."],
                ["SPEAKER_00: Îmi confirmați antecedentele importante?", f"SPEAKER_01: Confirm: {', '.join(spoken)}."],
            ]
            return variants[cid % len(variants)]
        variants = [
            ["SPEAKER_00: Aveți boli cunoscute, operații sau accidente relevante?", "SPEAKER_01: Nu, nimic confirmat."],
            ["SPEAKER_01: La antecedente nu am ceva important de trecut.", "SPEAKER_00: Bun, lăsăm câmpul gol."],
        ]
        return variants[cid % len(variants)]
    if act == "meds":
        if row["meds_plan"] == "unnamed-pills":
            return [
                "SPEAKER_01: Am luat niște pastile din casă, dar nu mai știu cum se numeau.",
                "SPEAKER_00: Fără nume, nu trec medicație concretă.",
            ]
        if row["meds_plan"] == "named":
            meds = json.loads(row["meds_detail"])
            med_text = ", ".join(f"{m['denumire']} {m['doza']}" if m["doza"] else m["denumire"] for m in meds)
            variants = [
                ["SPEAKER_00: Luați acum medicamente sau suplimente cu nume clar?", f"SPEAKER_01: Da, iau {med_text}."],
                [f"SPEAKER_01: Ca medicație actuală, iau {med_text}.", "SPEAKER_00: Am trecut numele și doza așa cum le-ați spus."],
                ["SPEAKER_00: Există ceva administrat curent?", f"SPEAKER_01: Da, {med_text}, atât."],
            ]
            return variants[cid % len(variants)]
        variants = [
            ["SPEAKER_00: Luați ceva cu nume și doză în perioada asta?", "SPEAKER_01: Nu iau nimic de trecut."],
            ["SPEAKER_01: Nu folosesc medicamente acum.", "SPEAKER_00: Atunci medicația rămâne necompletată."],
        ]
        return variants[cid % len(variants)]
    if act == "exam_movement":
        variants = [
            [f"SPEAKER_00: Faceți mișcarea lent, fără să forțați {region}.", f"SPEAKER_01: Aici apare durerea la {region}, înainte să ajung la capăt."],
            [f"SPEAKER_00: Ridicați și coborâți ușor segmentul, apoi ne oprim la prima durere.", f"SPEAKER_01: La coborâre se simte mai clar în {region}."],
            [f"SPEAKER_00: Încercăm mișcarea contra rezistență foarte ușoară.", f"SPEAKER_01: Contra rezistenței doare mai precis în {region}."],
        ]
        return variants[cid % len(variants)]
    if act == "exam_palpation":
        variants = [
            [f"SPEAKER_00: Palpez ușor zona indicată; spuneți dacă reproduce durerea.", f"SPEAKER_01: Da, presiunea de aici reproduce durerea din {region}."],
            [f"SPEAKER_00: Compar stânga-dreapta la presiune locală.", f"SPEAKER_01: Partea dureroasă reacționează mai repede decât cealaltă."],
            [f"SPEAKER_00: Mențin presiunea câteva secunde, fără să apăs tare.", f"SPEAKER_01: Se simte local și rămâne în {region}."],
        ]
        return variants[cid % len(variants)]
    if act == "functional_obs":
        return [f"SPEAKER_00: {note['evaluare_functionala_initiala']}"]
    if act == "closing":
        variants = [
            ["SPEAKER_00: Oprim testarea aici și continuăm doar dacă durerea rămâne tolerabilă.", "SPEAKER_01: Da, așa este mai clar pentru mine."],
            ["SPEAKER_01: Am înțeles ce ați verificat.", "SPEAKER_00: Bun, rămânem la ce s-a spus și observat astăzi."],
            ["SPEAKER_00: Dacă apare o schimbare clară, îmi spuneți înainte să continuăm.", "SPEAKER_01: Sigur, vă anunț imediat."],
        ]
        return variants[cid % len(variants)]
    raise KeyError(act)


def support_lines(row: dict[str, Any], act: str) -> list[str]:
    cid = int(row["conversation_id"][-2:])
    region = transcript_region_text(row)
    subject_region, settle_verb = region_subject(row)
    f = filler(row)
    variants = {
        "greeting": [
            ["SPEAKER_00: Începem calm și opriți răspunsul dacă ceva nu este clar.", "SPEAKER_01: Da, prefer să luăm lucrurile pe rând."],
            ["SPEAKER_01: Încerc să descriu simplu, fără termeni medicali.", "SPEAKER_00: E suficient, eu formulez clinic doar ce aud."],
        ],
        "age": [
            ["SPEAKER_00: Vârsta rămâne doar informație administrativă pentru fișă.", "SPEAKER_01: În regulă, nu are legătură directă cu durerea."],
            ["SPEAKER_01: Restul datelor personale sunt neschimbate.", "SPEAKER_00: Perfect, continuăm cu partea care vă supără."],
        ],
        "complaint": [
            ["SPEAKER_00: Vreau să separăm durerea clară de senzațiile vagi din zona descrisă ca dureroasă.", f"SPEAKER_01: {sentence_start(f)}, pot indica doar ce simt sigur."],
            [f"SPEAKER_01: Uneori pare mai largă, dar durerea importantă rămâne cum am spus.", "SPEAKER_00: Atunci nu extindem localizarea doar din presupunere."],
        ],
        "duration": [
            ["SPEAKER_00: Durata o păstrăm ca relatare, nu ca măsurătoare exactă.", "SPEAKER_01: Da, aproximarea asta este cea mai corectă."],
            ["SPEAKER_01: Nu am ținut un jurnal, dar perioada descrisă este realistă.", "SPEAKER_00: E suficient pentru contextul anamnezei."],
        ],
        "onset": [
            ["SPEAKER_00: Dacă debutul nu este complet sigur, îl lăsăm descris ca atare.", "SPEAKER_01: Da, nu vreau să inventez un moment precis."],
            ["SPEAKER_01: Pot spune doar cum am perceput începutul.", "SPEAKER_00: Exact, ne bazăm pe ce vă amintiți clar."],
        ],
        "aggravating_factors": [
            [f"SPEAKER_00: Când testăm {region}, urmărim ce reproduce durerea, nu ce pare doar oboseală.", "SPEAKER_01: Diferența se simte destul de clar."],
            [f"SPEAKER_01: Dacă mă opresc, {subject_region} {settle_verb} parțial.", "SPEAKER_00: Asta ajută să delimităm reacția la mișcare."],
        ],
        "daily_limitation": [
            ["SPEAKER_00: Limitarea zilnică o notăm doar ca descriere funcțională.", "SPEAKER_01: Da, nu spun că nu pot face nimic, doar că mă oprește."],
            ["SPEAKER_01: Am schimbat ritmul, dar încă mă descurc cu pauze.", "SPEAKER_00: Bun, nu exagerăm formularea."],
        ],
        "VAS": [
            ["SPEAKER_00: Scorul contează doar dacă îl spuneți explicit.", "SPEAKER_01: Am înțeles, fără număr ales de dumneavoastră."],
            ["SPEAKER_01: Intensitatea variază, dar răspund pentru momentul actual.", "SPEAKER_00: Exact, nu facem medie din presupuneri."],
        ],
        "antecedente": [
            ["SPEAKER_00: La antecedente trecem numai ce este confirmat sau spus clar.", "SPEAKER_01: Da, nu adaug alte lucruri nesigure."],
            ["SPEAKER_01: Dacă nu sunt sigur, prefer să nu fie trecut.", "SPEAKER_00: Corect, câmpul nu trebuie completat forțat."],
        ],
        "meds": [
            ["SPEAKER_00: Pentru medicație contează numele, iar doza doar dacă o știți.", "SPEAKER_01: Da, nu vreau să ghicesc ceva greșit."],
            ["SPEAKER_01: Dacă nu îmi amintesc denumirea, o spun așa.", "SPEAKER_00: Atunci nu transformăm pastilele vagi în medicație."],
        ],
        "exam_movement": [
            [f"SPEAKER_00: Mișcarea rămâne lentă; nu urmărim să provocăm durere mare în {region}.", "SPEAKER_01: Mă opresc imediat când apare clar."],
            [f"SPEAKER_01: Simt diferența între întindere și durere în {region}.", "SPEAKER_00: Perfect, această diferență este importantă la test."],
        ],
        "exam_palpation": [
            ["SPEAKER_00: Presiunea este ușoară, doar cât să comparăm reacția locală.", "SPEAKER_01: Da, nu e doar disconfort de la apăsare."],
            [f"SPEAKER_01: Când apăsați acolo, recunosc senzația din {region}.", "SPEAKER_00: Am notat reacția, fără să o transform în altceva."],
        ],
        "functional_obs": [
            ["SPEAKER_01: Deci ce ați spus este doar ce s-a văzut la testare.", "SPEAKER_00: Exact, rămânem strict la observația funcțională verbalizată."],
            ["SPEAKER_00: Observația rămâne funcțională și legată de mișcare.", "SPEAKER_01: Am înțeles diferența."],
        ],
        "closing": [
            ["SPEAKER_00: Nu mai adăugăm alte câmpuri dacă nu au fost discutate.", "SPEAKER_01: Da, e mai corect așa."],
            ["SPEAKER_01: Dacă îmi amintesc altceva, vă spun separat.", "SPEAKER_00: Sigur, dar pentru nota de acum rămânem la ce s-a verbalizat."],
        ],
    }
    return variants[act][cid % len(variants[act])]


def transcript_from_plan(row: dict[str, Any], note: dict[str, Any]) -> str:
    lines: list[str] = []
    for act in row["dialogue_act_order"].split(","):
        if act == "functional_obs":
            lines.extend(act_lines(row, act, note))
        else:
            lines.extend(decorate_lines(act_lines(row, act, note), row, act))
        lines.extend(decorate_lines(support_lines(row, act), row, act))
    transcript = "\n".join(lines)
    assert transcript.count(row["activity"]) == 1, (row["conversation_id"], row["activity"])
    return transcript


def mask_sentence(sentence: str) -> str:
    masked = sentence
    region_terms = sorted(set(REGION_LABELS.values()) | set(LOCALIZARE_ALTA.values()) | {"zona dureroasă"}, key=len, reverse=True)
    for term in region_terms:
        masked = re.sub(re.escape(term), "BODY_REGION", masked, flags=re.IGNORECASE)
    for activity in sorted(ACTIVITIES, key=len, reverse=True):
        masked = re.sub(re.escape(activity), "ACTIVITY", masked, flags=re.IGNORECASE)
    drug_names = {med["denumire"] for meds in MED_PLANS.values() for med in meds}
    for drug in sorted(drug_names, key=len, reverse=True):
        masked = re.sub(re.escape(drug), "DRUG", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b\d+\b(?:\s*(?:mg|mcg|comprimate?|ori|zile|săptămâni|luni))?", "NUMBER", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b(zero|unu|doi|două|trei|patru|cinci|șase|sase|șapte|sapte|opt|nouă|noua|zece)\b", "NUMBER", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b\d+\b", "NUMBER", masked)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked


def sentence_counts(texts: list[str], *, masked: bool = True) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        seen_in_text: set[str] = set()
        for raw in re.split(r"(?<=[.!?])\s+|\n+", text):
            sent = re.sub(r"^SPEAKER_\d+:\s*", "", raw.strip())
            sent = re.sub(r"\s+", " ", sent)
            if masked:
                sent = mask_sentence(sent)
            if len(sent.split()) >= 3:
                seen_in_text.add(sent)
        counts.update(seen_in_text)
    return counts


def masked_act_sequence(transcript: str) -> str:
    acts = []
    for raw in transcript.lower().splitlines():
        line = re.sub(r"^speaker_\d+:\s*", "", raw)
        act = ""
        if "bună ziua" in line or "vă ascult" in line or "luați loc" in line or "programat" in line or "începem" in line:
            act = "greeting"
        elif re.search(r"\b\d+\s+de ani\b", line):
            act = "age"
        elif "durere" in line and ("după" in line or "problema principală" in line or "cuvintele dumneavoastră" in line):
            act = "complaint"
        elif "de când" in line or "ați remarcat" in line or "aproximativ" in line or "recent sau mai vechi" in line or "o simt" in line:
            act = "duration"
        elif "moment exact" in line or "început clar" in line or "a pornit" in line or "prima relatare" in line or "altă cauză" in line:
            act = "onset"
        elif "aprinde" in line or "repaus" in line or "mișcare" in line and "clar" in line:
            act = "aggravating_factors"
        elif "zi obișnuită" in line or "fac pauze" in line or "gesturile mari" in line or "schimbat" in line:
            act = "daily_limitation"
        elif "scor" in line or "număr" in line or "zero" in line or "zece" in line or "cifra" in line:
            act = "VAS"
        elif "istoric" in line or "afecțiuni" in line or "confirmat" in line:
            act = "antecedente"
        elif "administrat" in line or "denumire" in line or "pastile" in line or "iau " in line or "nume" in line:
            act = "meds"
        elif "mișcarea" in line and ("opri" in line or "încercați" in line) or "durerea cunoscută" in line:
            act = "exam_movement"
        elif "palpez" in line or "apăs" in line or "presiune" in line:
            act = "exam_palpation"
        elif "limit" in line or "contra-rezistență" in line or "mișcare activă" in line or "comparația bilaterală" in line or "sensibilitate" in line:
            act = "functional_obs"
        elif "ne oprim" in line or "verificarea" in line and "azi" in line or "reprodus la test" in line:
            act = "closing"
        if act and (not acts or acts[-1] != act):
            acts.append(act)
    return ",".join(acts)


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "conversation_id", "batch", "regions", "vas", "duration", "age", "onset",
        "antecedente_plan", "antecedente_altele_plan", "localizare_alta_plan",
        "meds_plan", "meds_detail", "func_eval", "include_location", "note_shape",
        "richness", "patient_style", "opening", "activity", "dialogue_act_order",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_seed_manifest(out_root: Path) -> None:
    seed_lines = ["seed_id\tconversation_path\tstructured_ref_path\tfrom_chiro_path"]
    for seed_id in SEED_IDS:
        conversation_path = ROOT / "data" / "chiropractor_ro" / "conversations" / f"{seed_id}.json"
        ref_path = ROOT / "data" / "chiropractor_ro" / "refs" / f"{seed_id}.json"
        from_chiro = next((ROOT / "data" / "from_chiro").glob(f"{seed_id}_fisa.*"))
        if not conversation_path.exists() or not ref_path.exists() or not from_chiro.exists():
            raise FileNotFoundError(seed_id)
        seed_lines.append(f"{seed_id}\t{conversation_path.relative_to(ROOT)}\t{ref_path.relative_to(ROOT)}\t{from_chiro.relative_to(ROOT)}")
    (out_root / "seed_pairs.tsv").write_text("\n".join(seed_lines) + "\n", encoding="utf-8")


def read_transcripts_from_disk(transcripts_dir: Path) -> list[str]:
    return [path.read_text(encoding="utf-8") for path in sorted(transcripts_dir.glob("synth_*.txt"))]


def diversity_report(rows: list[dict[str, Any]], transcripts_dir: Path) -> str:
    transcripts = read_transcripts_from_disk(transcripts_dir)
    vas_counts = Counter(row["vas"] for row in rows)
    used_regions = sorted({region for row in rows for region in row["regions"].split(",")})
    repeated_sentences = {sentence: count for sentence, count in sentence_counts(transcripts).items() if count > 2}
    seq_counts = Counter(row["dialogue_act_order"] for row in rows)
    masked_seq_counts = Counter(masked_act_sequence(text) for text in transcripts)
    duplicate_plan_seq = {seq: count for seq, count in seq_counts.items() if count > 1}
    duplicate_masked_seq = {seq: count for seq, count in masked_seq_counts.items() if count > 1 and seq}
    lines = ["metric\tvalue\tstatus"]
    lines.append(f"vas_distribution\t{dict(sorted(vas_counts.items()))}\t{'FLAG' if max(vas_counts.values()) > 6 else 'ok'}")
    lines.append(f"distinct_regions\t{len(used_regions)} ({','.join(used_regions)})\t{'FLAG' if len(used_regions) < 12 else 'ok'}")
    lines.append(f"repeated_masked_sentence_skeletons_gt2\t{len(repeated_sentences)}\t{'FLAG' if repeated_sentences else 'ok'}")
    lines.append(f"duplicate_plan_act_sequences\t{len(duplicate_plan_seq)}\t{'FLAG' if duplicate_plan_seq else 'ok'}")
    lines.append(f"duplicate_masked_act_sequences\t{len(duplicate_masked_seq)}\t{'FLAG' if duplicate_masked_seq else 'ok'}")
    for sentence, count in sorted(repeated_sentences.items(), key=lambda item: (-item[1], item[0]))[:20]:
        lines.append(f"repeated_sentence\t{count}x {sentence}\tFLAG")
    for seq, count in sorted(duplicate_masked_seq.items(), key=lambda item: (-item[1], item[0]))[:10]:
        lines.append(f"repeated_act_sequence\t{count}x {seq}\tFLAG")
    return "\n".join(lines)


def coverage_report(notes: list[dict[str, Any]]) -> str:
    floors = {
        "antecedente": 12,
        "medicatie_actuala": 10,
        "localizarea_durerii_alta": 3,
        "antecedente_altele": 3,
    }
    positive_counts = {field: sum(1 for note in notes if is_populated(note[field])) for field in FIELDS_FOR_RATES}
    lines = ["field\tpositive_count\tfloor\tstatus"]
    for field in FIELDS_FOR_RATES:
        floor = floors.get(field, 0)
        lines.append(f"{field}\t{positive_counts[field]}\t{floor}\t{'ok' if positive_counts[field] >= floor else 'FLAG'}")
    antecedente_used = sorted({item for note in notes for item in note["antecedente"]})
    med_named_dose = sum(1 for note in notes for med in note["medicatie_actuala"] if med["doza"])
    lines.append(f"antecedente_enum_coverage\t{len(antecedente_used)} ({','.join(antecedente_used)})\t10\t{'ok' if len(antecedente_used) >= 10 else 'FLAG'}")
    lines.append(f"meds_with_real_dose\t{med_named_dose}\t10\t{'ok' if med_named_dose >= 10 else 'FLAG'}")
    return "\n".join(lines)


def validate_contract(note: dict[str, Any], transcript: str, cid: str) -> None:
    text = transcript.lower()
    if note["evaluare_functionala_initiala"]:
        assert note["evaluare_functionala_initiala"].lower() in text, cid
    if note["evaluarea_durerii_vas"] is not None:
        assert str(note["evaluarea_durerii_vas"]) in text, cid
    else:
        assert "aș spune " not in text and "din zece" not in text, cid
    for region in note["localizarea_durerii"]:
        assert REGION_LABELS[region].lower() in text, (cid, region)
    if not note["localizarea_durerii"]:
        assert not any(label.lower() in text for label in REGION_LABELS.values()), cid
    if note["localizarea_durerii_alta"]:
        assert note["localizarea_durerii_alta"].lower() in text, cid
    for antecedent in note["antecedente"]:
        assert ANTECEDENTE_LABELS[antecedent].lower() in text, (cid, antecedent)
    if not note["antecedente"]:
        assert not any(label.lower() in text for label in ANTECEDENTE_LABELS.values()), cid
    if note["antecedente_altele"]:
        assert note["antecedente_altele"].lower() in text, cid
    drug_names = {med["denumire"] for meds in MED_PLANS.values() for med in meds}
    for med in note["medicatie_actuala"]:
        assert med["denumire"].lower() in text, (cid, med)
        if med["doza"]:
            assert med["doza"].lower() in text, (cid, med)
    if not note["medicatie_actuala"]:
        assert not any(drug.lower() in text for drug in drug_names), cid


def validate_grammar_artifacts(text: str, cid: str) -> None:
    banned = ["zonei zona", "zona zona", "Diagnostic", "diagnostic", "Obiectiv", "obiectiv"]
    for item in banned:
        assert item not in text, (cid, item)
    for line in text.splitlines():
        words = re.findall(r"[A-Za-zĂÂÎȘȚăâîșț]+", line.lower())
        for first, second in zip(words, words[1:]):
            assert first != second, (cid, line)


def main() -> None:
    assert set(SEED_IDS).issubset(POOL_IDS)
    assert_no_test_leakage(SEED_IDS, context="synthetic seed selection")
    pool_ref_ids, target_rates = measure_pool_rates()

    out_root = ROOT / "synthetic"
    refs_dir = out_root / "refs"
    transcripts_dir = out_root / "transcripts"
    refs_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    rows = build_plan(target_rates)
    write_plan(out_root / "plan.tsv", rows)
    write_seed_manifest(out_root)

    index_lines = ["conversation_id\ttranscript_path\tnote_path"]
    realized_counts = {field: 0 for field in FIELDS_FOR_RATES}
    all_notes: list[dict[str, Any]] = []

    for row in rows:
        cid = row["conversation_id"]
        note = note_from_plan(row)
        if validate is not None:
            validate(instance=note, schema=NOTE_SCHEMA)
        transcript = transcript_from_plan(row, note)
        validate_contract(note, transcript, cid)
        note_text = json.dumps(note, ensure_ascii=False)
        assert "zonei zona" not in note_text and "zona zona" not in note_text, cid
        validate_grammar_artifacts(transcript, cid)
        for field in FIELDS_FOR_RATES:
            realized_counts[field] += int(is_populated(note[field]))
        all_notes.append(note)
        note_path = refs_dir / f"{cid}.json"
        transcript_path = transcripts_dir / f"{cid}.txt"
        note_path.write_text(json.dumps(note, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        transcript_path.write_text(transcript + "\n", encoding="utf-8")
        index_lines.append(f"{cid}\t{transcript_path.relative_to(ROOT)}\t{note_path.relative_to(ROOT)}")

    (out_root / "index.tsv").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    realized_rates = {field: realized_counts[field] / len(rows) for field in FIELDS_FOR_RATES}
    report_lines = [
        f"Pool refs used for target rates: {', '.join(pool_ref_ids)}",
        f"Seed pairs used: {', '.join(SEED_IDS)}",
        "",
        "field\ttarget_rate\trealized_rate",
    ]
    for field in FIELDS_FOR_RATES:
        report_lines.append(f"{field}\t{target_rates[field]:.2f}\t{realized_rates[field]:.2f}")
    distribution_report = "\n".join(report_lines)
    (out_root / "distribution_report.tsv").write_text(distribution_report + "\n", encoding="utf-8")

    diversity = diversity_report(rows, transcripts_dir)
    (out_root / "diversity_report.tsv").write_text(diversity + "\n", encoding="utf-8")
    coverage = coverage_report(all_notes)
    (out_root / "coverage_report.tsv").write_text(coverage + "\n", encoding="utf-8")

    print(distribution_report)
    print()
    print(diversity)
    print()
    print(coverage)


if __name__ == "__main__":
    main()
