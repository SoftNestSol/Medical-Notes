"""Generate diversified synthetic Romanian chiropractor note/transcript pairs.

The generator is intentionally plan-first:
1. measure target populate rates from POOL refs only;
2. write a 50-row variation plan to synthetic/plan.tsv;
3. generate 5 batches of 10 from that plan;
4. validate schema, leakage, distribution, and diversity.
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

ONSET_PHRASES = {
    "efort": [
        "după ce a ridicat bagaje la mutare",
        "după exerciții făcute mai intens decât de obicei",
        "după ce a cărat cumpărături pe scări",
    ],
    "postură-birou": [
        "după lucru prelungit la calculator",
        "după multe ore de stat pe scaun în ședințe",
        "după o perioadă cu laptopul ținut jos",
    ],
    "cădere": [
        "după o alunecare pe trepte",
        "după ce a călcat strâmb în curte",
        "după o căzătură ușoară pe partea dureroasă",
    ],
    "treptat": [
        "cu debut progresiv, fără un moment clar",
        "apărută treptat în ultimele săptămâni",
        "instalată lent, inițial doar ca jenă",
    ],
    "accident": [
        "după o frână bruscă în mașină",
        "după o lovitură minoră la sport",
        "după un incident la bicicletă",
    ],
    "necunoscut": [
        "fără cauză clară relatată de pacient",
        "fără declanșator identificat",
        "cu debut neclar pentru pacient",
    ],
}

OPENINGS = [
    ("SPEAKER_00: Bună ziua, poftiți, cu ce vă pot ajuta astăzi?", "greeting first"),
    ("SPEAKER_01: Bună ziua, vin direct cu problema, mă supără destul de tare zona asta.", "patient leads"),
    ("SPEAKER_00: Eram la completarea fișei; spuneți-mi, vă rog, vârsta și apoi discutăm durerea.", "mid-intake"),
    ("SPEAKER_00: Ne-am mai văzut sau este prima vizită la Osteopath Concept?", "first visit vs returning"),
    ("SPEAKER_01: Am ajuns puțin mai devreme, dar durerea nu prea m-a lăsat să stau jos.", "patient leads"),
    ("SPEAKER_00: Haideți să începem cu motivul prezentării, apoi verificăm zona prin mișcare.", "greeting first"),
    ("SPEAKER_00: Bun, am fișa în față; ce anume vă deranjează cel mai mult?", "mid-intake"),
    ("SPEAKER_01: Revin pentru altă problemă, nu pentru cea de data trecută.", "first visit vs returning"),
    ("SPEAKER_00: Înainte să testăm, vreau să înțeleg exact unde apare durerea.", "mid-intake"),
    ("SPEAKER_01: Nu știu dacă e grav, dar aș vrea să îmi spuneți ce observați la testare.", "patient leads"),
]

DURATION_DETAIL = {
    "acut": ["de ieri", "de două zile", "de aproximativ trei zile"],
    "subacut": ["de aproape două săptămâni", "de vreo zece zile", "cam de trei săptămâni"],
    "cronic": ["de câteva luni", "de mai bine de jumătate de an", "de mult timp, cu episoade mai rele"],
}

STYLE_FILLERS = {
    "talkative": ["adică", "vă dați seama", "ca să explic mai clar"],
    "terse": ["da", "cam așa", "nu prea știu"],
    "anxious": ["sincer", "mă îngrijorează", "mi-e teamă să nu fie ceva"],
    "vague": ["pe aici", "nu știu exact", "cumva"],
    "precise": ["mai exact", "în special", "la mișcarea asta"],
}

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
    pool_ref_ids = sorted(
        p.stem for p in ref_dir.glob("*.json") if p.stem in POOL_IDS and p.stem not in TEST_IDS
    )
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
    for i in range(50):
        one_based = i + 1
        regions = REGION_PLANS[i]
        assert all(region in LOCALIZARE_ENUM for region in regions)
        onset = onsets[(i * 2 + i // 7) % len(onsets)]
        duration = durations[(i + i // 5) % len(durations)]
        age = 18 + ((i * 17) % 63)
        meds_plan = "none"
        if one_based in MED_PLANS:
            meds_plan = "named"
        elif one_based in {6, 19, 34, 43}:
            meds_plan = "unnamed-pills"
        note_shape = sparse_shapes.get(one_based, "rich" if one_based in {13, 16, 28, 31, 33, 36, 44, 47} else "standard")
        include_location = note_shape not in {"motiv_only", "motiv_vas"}
        include_vas = note_shape not in {"motiv_only", "motiv_location", "motiv_location_func"}
        include_func = note_shape not in {"motiv_only", "motiv_vas", "motiv_location", "motiv_vas_location"}
        rows.append({
            "conversation_id": f"synth_{i + 1:02d}",
            "batch": str(i // 10 + 1),
            "regions": ",".join(regions),
            "vas": str(vas_values[i]) if include_vas else "null",
            "duration": duration,
            "age": str(age),
            "onset": onset,
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
            "opening": OPENINGS[i % len(OPENINGS)][1],
            "activity": ACTIVITIES[i],
        })
    return rows


def complaint_from_plan(row: dict[str, Any]) -> str:
    regions = row["regions"].split(",")
    onset = row["onset"]
    phrase = ONSET_PHRASES[onset][int(row["conversation_id"][-2:]) % 3]
    activity = row["activity"]
    if row["include_location"] == "no":
        return f"durere musculo-scheletală descrisă de pacient, {phrase}, în context de {activity}"
    if row["localizare_alta_plan"]:
        return f"durere la {join_regions(regions)} și {row['localizare_alta_plan']}, {phrase}"
    return f"durere la {join_regions(regions)}, {phrase}, în context de {activity}"


def functional_observation(row: dict[str, Any]) -> str | None:
    if row["func_eval"] == "null":
        return None
    regions = row["regions"].split(",")
    region_text = join_regions(regions)
    activity = row["activity"]
    n = int(row["conversation_id"][-2:])
    if any(r in {"lombar", "sacral_coccis", "sold_dr", "sold_stg"} for r in regions):
        variants = [
            f"La testarea în ortostatism după {activity}, durerea la {region_text} se reproduce la aplecare și la presiune locală.",
            f"Mobilitatea trunchiului este limitată de durere când pacientul descrie {activity}, cu reacție la palparea zonei {region_text}.",
            f"Durerea de la {region_text} este provocată la flexie controlată și la comparația prin presiune, mai ales după {activity}.",
        ]
    elif any(r in {"cervical", "toracal", "cap_ceafa", "torace", "abdomen"} for r in regions):
        variants = [
            f"Rotirea și extensia sunt limitate de durere după {activity}, iar palparea reproduce simptomul în {region_text}.",
            f"La mișcarea ghidată apare limitare de amplitudine, cu sensibilitate locală la testarea pentru {region_text} după {activity}.",
            f"Respirația sau rotația controlată accentuează durerea descrisă la {region_text}, în context de {activity}.",
        ]
    else:
        variants = [
            f"Durerea la {region_text} apare la mișcare activă și se accentuează la testarea contra-rezistență după {activity}.",
            f"Presiunea locală pe {region_text} este tolerată parțial, însă mișcarea provocată reproduce simptomul descris la {activity}.",
            f"La comparația bilaterală pentru {region_text}, partea simptomatică are durere mai clară când pacientul reproduce {activity}.",
        ]
    return variants[n % len(variants)]


def note_from_plan(row: dict[str, Any]) -> dict[str, Any]:
    vas = None if row["vas"] == "null" else int(row["vas"])
    antecedente = [] if row["antecedente_plan"] == "none" else row["antecedente_plan"].split(",")
    for item in antecedente:
        assert item in ANTECEDENTE_ENUM
    medication = json.loads(row["meds_detail"])
    note = {
        "motivul_prezentarii": complaint_from_plan(row),
        "evaluarea_durerii_vas": vas,
        "localizarea_durerii": row["regions"].split(",") if row["include_location"] == "yes" else [],
        "localizarea_durerii_alta": row["localizare_alta_plan"] or None,
        "antecedente": antecedente,
        "antecedente_altele": row["antecedente_altele_plan"] or None,
        "medicatie_actuala": medication,
        "evaluare_functionala_initiala": functional_observation(row),
    }
    return note


def opening_lines(row: dict[str, Any]) -> list[str]:
    region_text = transcript_region_text(row)
    cid = int(row["conversation_id"][-2:])
    activity = row["activity"]
    dynamic_openings = [
        f"SPEAKER_00: Bună ziua, poftiți; începem cu ce simțiți la {region_text} după {activity}.",
        f"SPEAKER_01: Bună ziua, vin direct cu problema de la {region_text}, mai ales când fac {activity}.",
        f"SPEAKER_00: Eram la completarea fișei pentru vizita {cid}; trecem imediat la {region_text} și la episodul cu {activity}.",
        f"SPEAKER_00: Ne-am mai văzut sau discutăm prima dată despre {region_text} în context de {activity}?",
        f"SPEAKER_01: Am ajuns puțin mai devreme fiindcă {region_text} m-a deranjat încă de la {activity}.",
        f"SPEAKER_00: Haideți să începem cu motivul prezentării și apoi testăm {region_text} prin mișcări legate de {activity}.",
        f"SPEAKER_00: Bun, am fișa în față; spuneți-mi ce vă deranjează la {region_text} când apare {activity}.",
        f"SPEAKER_01: Revin pentru o problemă nouă, de data asta la {region_text}, apărută în jurul activității de {activity}.",
        f"SPEAKER_00: Înainte să testăm, vreau să înțeleg exact durerea de la {region_text} după {activity}.",
        f"SPEAKER_01: Nu știu dacă e grav, dar vreau să verificăm atent {region_text}, pentru că mă încurcă la {activity}.",
    ]
    opening = dynamic_openings[(cid - 1) % len(dynamic_openings)]
    if opening.startswith("SPEAKER_01"):
        return [opening, f"SPEAKER_00: Haideți să luăm pe rând vârsta, apoi verificăm precis {region_text} în context de {row['activity']}."]
    return [opening, "SPEAKER_01: Sigur, vă spun cum pot."]


def style_answer(row: dict[str, Any], base: str) -> str:
    fillers = STYLE_FILLERS[row["patient_style"]]
    return f"{fillers[int(row['conversation_id'][-2:]) % len(fillers)]}, {base}"


def vas_question(row: dict[str, Any]) -> str:
    region_text = transcript_region_text(row)
    activity = row["activity"]
    variants = [
        f"SPEAKER_00: După {activity}, dacă ar fi să dați o notă durerii de la {region_text}, între zero și zece, ce număr ați alege?",
        f"SPEAKER_00: Ca intensitate după episodul cu {activity}, ce scor îi dați astăzi pentru {region_text}, de la zero până la zece?",
        f"SPEAKER_00: Pentru fișă am nevoie de un număr al durerii actuale la {region_text}, mai ales când reluați {activity}; cât spuneți că este?",
        f"SPEAKER_00: Unde se așază durerea de la {region_text} pe scala zero-zece în momentul ăsta, după {activity}?",
        f"SPEAKER_00: Dacă zero înseamnă deloc și zece maxim, cât simțiți acum în {region_text} când vă gândiți la {activity}?",
    ]
    return variants[int(row["conversation_id"][-2:]) % len(variants)]


def duration_line(row: dict[str, Any]) -> str:
    options = DURATION_DETAIL[row["duration"]]
    return options[int(row["conversation_id"][-2:]) % len(options)]


def sentence_start(text: str) -> str:
    return text[:1].upper() + text[1:]


def exam_block(row: dict[str, Any]) -> list[str]:
    regions = row["regions"].split(",")
    region_text = join_regions(regions)
    activity = row["activity"]
    cid = int(row["conversation_id"][-2:])
    if any(r in {"lombar", "sacral_coccis", "sold_dr", "sold_stg"} for r in regions):
        blocks = [
            [
                f"SPEAKER_00: Ridicați-vă încet ca după {activity} și aplecați trunchiul doar până apare prima jenă în {region_text}.",
                f"SPEAKER_01: Aici se oprește, simt că se strânge spre {region_text} ca în timpul activității de {activity}.",
                f"SPEAKER_00: Apăs lângă {region_text}; spuneți dacă reproduce senzația din {activity}.",
                f"SPEAKER_01: Da, seamănă cu ce simt când solicit {region_text} la {activity}.",
            ],
            [
                f"SPEAKER_00: Stați cu tălpile paralele și mutați greutatea ușor, ca să comparăm {region_text} după {activity}.",
                f"SPEAKER_01: Pe partea dureroasă din {region_text} se simte mai repede decât m-aș aștepta la {activity}.",
                f"SPEAKER_00: Compar presiunea locală pe {region_text} cu partea opusă, în logica mișcării de {activity}.",
                f"SPEAKER_01: Diferența e clară; partea cealaltă nu mă deranjează ca {region_text} când fac {activity}.",
            ],
        ]
    elif any(r in {"cervical", "toracal", "cap_ceafa", "torace", "abdomen"} for r in regions):
        blocks = [
            [
                f"SPEAKER_00: Întoarceți lent zona implicată, ca atunci când faceți {activity}, fără să forțați {region_text}.",
                f"SPEAKER_01: Într-o parte merge, în cealaltă mă blochează la {region_text}, mai ales după {activity}.",
                f"SPEAKER_00: Palpez acum {region_text}; îmi spuneți dacă reproduce simptomul din {activity}.",
                f"SPEAKER_01: Da, la {region_text} e durerea cunoscută din {activity}, nu doar apăsare.",
            ],
            [
                f"SPEAKER_00: Ridicați ușor zona din jurul {region_text}, ca după {activity}, coborâți și inspirați mai amplu.",
                f"SPEAKER_01: La coborâre și la inspirație se simte mai tare în {region_text}, similar cu {activity}.",
                f"SPEAKER_00: Testez amplitudinea pe {region_text} și compar reacția la palpare după mișcarea de {activity}.",
                f"SPEAKER_01: Se simte local în {region_text}, nu pleacă în altă parte când repet {activity}.",
            ],
        ]
    else:
        blocks = [
            [
                f"SPEAKER_00: Sprijiniți segmentul dureros și faceți întâi mișcarea fără greutate pentru {region_text}, ca la {activity}.",
                f"SPEAKER_01: Fără greutate e suportabil la {region_text}, dar nu e liber când imit {activity}.",
                f"SPEAKER_00: Opun rezistență ușoară pentru {region_text}; vă opriți dacă înțeapă ca în {activity}.",
                f"SPEAKER_01: Acum apare mai clar în {region_text}, exact la mișcarea legată de {activity}.",
            ],
            [
                f"SPEAKER_00: Strângeți ușor și relaxați, vreau să văd diferența între repaus și efort la {region_text} după {activity}.",
                f"SPEAKER_01: În repaus e ok, la efortul asemănător cu {activity} se aprinde durerea din {region_text}.",
                f"SPEAKER_00: Palpez linia dureroasă la {region_text} și verific mișcarea contra mea, similar cu {activity}.",
                f"SPEAKER_01: Contra rezistenței e cel mai evident pentru {region_text}, mai ales după {activity}.",
            ],
        ]
    return blocks[cid % 2]


def daily_impact_block(row: dict[str, Any]) -> list[str]:
    cid = int(row["conversation_id"][-2:])
    region_text = transcript_region_text(row)
    style = row["patient_style"]
    duration = duration_line(row)
    impacts = {
        "talkative": "îmi schimb poziția des și parcă pierd timp până găsesc o variantă suportabilă",
        "terse": "mă opresc din mișcare și aștept să scadă",
        "anxious": "mă sperie când apare brusc și tind să protejez zona",
        "vague": "nu știu mereu ce o declanșează, dar evit mișcările mari",
        "precise": "apare mai ales la mișcarea descrisă și scade când o controlez",
    }
    activity = row["activity"]
    checks = [
        f"SPEAKER_00: Într-o zi cu {activity}, ce vă limitează concret durerea de la {region_text}?",
        f"SPEAKER_01: {sentence_start(duration)}, {impacts[style]} când fac {activity}.",
        f"SPEAKER_00: După ce vă opriți din {activity}, durerea se retrage imediat sau mai stă în {region_text}?",
        f"SPEAKER_01: Mai stă câteva minute în {region_text}, apoi reiau {activity} mai încet.",
    ]
    if row["richness"] == "detailed_long" or cid % 2 == 0:
        checks += [
            f"SPEAKER_00: Ați observat dacă dimineața, seara sau după {activity} se schimbă reacția dureroasă?",
            f"SPEAKER_01: După {activity} reacția se schimbă; când sunt obosit, simptomele din {region_text} reapar mai repede.",
        ]
    else:
        checks += [
            f"SPEAKER_00: Ați scos din program ceva legat de {activity} din cauza durerii de la {region_text}?",
            f"SPEAKER_01: Da, am evitat partea din {activity} care pornește durerea în {region_text}, mai ales când trebuie repetată.",
        ]
    return checks


def transcript_from_plan(row: dict[str, Any], previous_sentences: set[str]) -> str:
    regions = row["regions"].split(",")
    region_text = transcript_region_text(row)
    spoken_region = region_text
    if row["localizare_alta_plan"]:
        spoken_region = f"{region_text} și {row['localizare_alta_plan']}"
    onset_phrase = ONSET_PHRASES[row["onset"]][int(row["conversation_id"][-2:]) % 3]
    patient_onset = (
        onset_phrase
        .replace("după ce a ", "după ce am ")
        .replace("apărută treptat", "a apărut treptat")
        .replace("instalată lent", "s-a instalat lent")
    )
    onset_with_activity = f"{patient_onset}, în context de {row['activity']}"
    lines = opening_lines(row)
    cid = int(row["conversation_id"][-2:])
    if row["opening"] == "patient leads":
        lines += [
            f"SPEAKER_01: Problema principală este la {spoken_region}; {style_answer(row, onset_with_activity)}.",
            f"SPEAKER_00: Am înțeles. Ce vârstă aveți acum?",
            f"SPEAKER_01: {row['age']} de ani.",
        ]
    else:
        lines += [
            f"SPEAKER_00: Ce vârstă aveți?",
            f"SPEAKER_01: {row['age']} de ani.",
            f"SPEAKER_00: Care este motivul pentru care ați venit?",
            f"SPEAKER_01: {style_answer(row, f'mă doare la {spoken_region}, {onset_with_activity}')}.",
        ]
    if row["patient_style"] in {"talkative", "anxious"}:
        lines += [
            f"SPEAKER_01: Durează {duration_line(row)}, iar în ultimele zile am început să evit mișcarea legată de {row['activity']}.",
            f"SPEAKER_00: Deci durerea vă schimbă felul în care faceți {row['activity']}, nu doar apare la apăsare.",
        ]
    else:
        lines += [
            f"SPEAKER_00: De când o simțiți?",
            f"SPEAKER_01: {duration_line(row)}.",
        ]
    lines += daily_impact_block(row)
    if cid % 3 == 0:
        if row["include_location"] == "yes":
            lines += [
                f"SPEAKER_00: Pentru episodul legat de {row['activity']}, este doar {spoken_region} sau mai trecem încă o zonă clar dureroasă?",
                f"SPEAKER_01: Doar {spoken_region}; după {row['activity']} restul e mai mult disconfort vag, nu durere separată.",
            ]
        else:
            lines += [
                "SPEAKER_00: Îmi puteți indica o zonă anatomică precisă pe care să o trec?",
                "SPEAKER_01: Nu chiar, simt durerea difuz și nu aș ști să o localizez clar.",
            ]
    else:
        if row["include_location"] == "yes":
            lines += [
                f"SPEAKER_01: Localizarea sigură pe care o pot indica după {row['activity']} este {spoken_region}.",
                f"SPEAKER_00: Atunci în fișă păstrăm localizarea clară pentru episodul cu {row['activity']}: {spoken_region}.",
            ]
        else:
            lines += [
                f"SPEAKER_01: Nu pot să spun o localizare exactă după {row['activity']}; e mai mult o durere generală.",
                f"SPEAKER_00: În regulă, pentru episodul cu {row['activity']} nu forțăm o bifă anatomică dacă nu este clar spusă.",
            ]
    if row["vas"] != "null":
        lines += [
            vas_question(row),
            f"SPEAKER_01: Aș spune {row['vas']}, mai ales după {row['activity']}.",
        ]
    else:
        lines += [
            f"SPEAKER_00: Pentru {row['activity']}, ați putea pune durerea pe o scară numerică?",
            f"SPEAKER_01: Nu prea pot să aleg un număr pentru ce simt la {spoken_region}; după {row['activity']} doar știu că mă deranjează.",
        ]
    antecedente = [] if row["antecedente_plan"] == "none" else row["antecedente_plan"].split(",")
    if antecedente or row["antecedente_altele_plan"]:
        spoken = []
        labels = {
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
        spoken.extend(labels[item] for item in antecedente)
        if row["antecedente_altele_plan"]:
            spoken.append(row["antecedente_altele_plan"])
        lines += [
            f"SPEAKER_00: La antecedente, există afecțiuni confirmate relevante pentru evaluarea după {row['activity']}?",
            f"SPEAKER_01: Da, am {', '.join(spoken)}.",
        ]
    else:
        lines += [
            f"SPEAKER_00: Aveți boli cunoscute, operații sau accidente relevante pentru activitatea de {row['activity']}?",
            "SPEAKER_01: Nu, nimic confirmat care să conteze aici.",
        ]
    if cid % 4 == 0 and row["meds_plan"] == "none":
        lines += []
    elif row["meds_plan"] == "unnamed-pills":
        lines += [
            f"SPEAKER_01: Am încercat niște pastile din casă după {row['activity']}, dar nu mai știu denumirea.",
            f"SPEAKER_00: Dacă nu știm numele, pentru {spoken_region} după {row['activity']} nu trec medicație concretă în fișă.",
        ]
    elif row["meds_plan"] == "named":
        meds = json.loads(row["meds_detail"])
        med_text = ", ".join(
            f"{m['denumire']} {m['doza']}" if m["doza"] else m["denumire"]
            for m in meds
        )
        lines += [
            f"SPEAKER_00: Pentru perioada cu {row['activity']}, aveți medicamente sau suplimente luate acum, cu nume dacă le știți, inclusiv pentru {spoken_region}?",
            f"SPEAKER_01: Da, iau {med_text}, exact așa le am notate acasă pentru perioada cu {row['activity']}.",
        ]
    else:
        lines += [
            f"SPEAKER_00: Medicamente cu nume și doză pe care le luați acum pentru durerea apărută la {row['activity']}?",
            f"SPEAKER_01: Nu, pentru {row['activity']} nu iau nimic cu nume clar.",
        ]
    if row["richness"] == "detailed_long":
        lines += [
            f"SPEAKER_00: Pentru {row['activity']}, verific dacă durerea de la {region_text} apare mai mult la mișcare, presiune sau menținerea poziției.",
            f"SPEAKER_01: La mișcarea asemănătoare cu {row['activity']} e cel mai clar pentru {region_text}; dacă stau nemișcat scade.",
        ]
    if row["include_location"] == "yes" or row["func_eval"] == "populated":
        lines += exam_block(row)
    obs = functional_observation(row)
    if obs:
        lines += [
            f"SPEAKER_00: Spun acum observația funcțională pentru {region_text}, legată de {row['activity']}, ca să fie clar ce s-a văzut la testare.",
            f"SPEAKER_00: {obs}",
        ]
    lines += [
        f"SPEAKER_01: Am înțeles, atunci are sens că mă opresc când încerc {row['activity']}.",
        f"SPEAKER_00: Pentru restul ședinței rămânem la testare atentă după {row['activity']}; dacă se schimbă durerea, îmi spuneți.",
    ]
    transcript = "\n".join(lines)
    current = sentence_counts([transcript])
    repeated = {s for s, n in current.items() if n > 1 and s in previous_sentences}
    if repeated:
        transcript += f"\nSPEAKER_00: Reformulez pentru fișă: reacția principală rămâne legată de {region_text}."
    return transcript


def mask_sentence(sentence: str) -> str:
    masked = sentence
    region_terms = sorted(
        set(REGION_LABELS.values()) | set(LOCALIZARE_ALTA.values()),
        key=len,
        reverse=True,
    )
    for term in region_terms:
        masked = re.sub(re.escape(term), "BODY_REGION", masked, flags=re.IGNORECASE)
    drug_names = {med["denumire"] for meds in MED_PLANS.values() for med in meds}
    for drug in sorted(drug_names, key=len, reverse=True):
        masked = re.sub(re.escape(drug), "DRUG", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b\d+\s*(mg|mcg|comprimate?|ori|zile|săptămâni|luni)?\b", "NUMBER", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b(zero|unu|doi|două|trei|patru|cinci|șase|sase|șapte|sapte|opt|nouă|noua|zece)\b", "NUMBER", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked


def sentence_counts(texts: list[str], *, masked: bool = True) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for raw in re.split(r"(?<=[.!?])\s+|\n+", text):
            sent = re.sub(r"^SPEAKER_\d+:\s*", "", raw.strip())
            sent = re.sub(r"\s+", " ", sent)
            if masked:
                sent = mask_sentence(sent)
            if len(sent.split()) >= 8:
                counts[sent] += 1
    return counts


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "conversation_id", "batch", "regions", "vas", "duration", "age", "onset",
        "antecedente_plan", "antecedente_altele_plan", "localizare_alta_plan",
        "meds_plan", "meds_detail", "func_eval", "include_location", "note_shape",
        "richness", "patient_style", "opening", "activity",
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
        seed_lines.append(
            f"{seed_id}\t{conversation_path.relative_to(ROOT)}\t{ref_path.relative_to(ROOT)}\t{from_chiro.relative_to(ROOT)}"
        )
    (out_root / "seed_pairs.tsv").write_text("\n".join(seed_lines) + "\n", encoding="utf-8")


def diversity_report(rows: list[dict[str, Any]], transcripts: list[str]) -> str:
    vas_counts = Counter(row["vas"] for row in rows)
    used_regions = sorted({region for row in rows for region in row["regions"].split(",")})
    repeated_sentences = {
        sentence: count for sentence, count in sentence_counts(transcripts).items() if count > 2
    }
    lines = ["metric\tvalue\tstatus"]
    max_vas = max(vas_counts.values())
    lines.append(f"vas_distribution\t{dict(sorted(vas_counts.items()))}\t{'FLAG' if max_vas > 6 else 'ok'}")
    lines.append(f"distinct_regions\t{len(used_regions)} ({','.join(used_regions)})\t{'FLAG' if len(used_regions) < 12 else 'ok'}")
    lines.append(f"repeated_full_sentences_gt2\t{len(repeated_sentences)}\t{'FLAG' if repeated_sentences else 'ok'}")
    for sentence, count in sorted(repeated_sentences.items(), key=lambda item: (-item[1], item[0]))[:20]:
        lines.append(f"repeated_sentence\t{count}x {sentence}\tFLAG")
    return "\n".join(lines)


def coverage_report(notes: list[dict[str, Any]]) -> str:
    floors = {
        "antecedente": 12,
        "medicatie_actuala": 10,
        "localizarea_durerii_alta": 3,
        "antecedente_altele": 3,
    }
    positive_counts = {
        field: sum(1 for note in notes if is_populated(note[field]))
        for field in FIELDS_FOR_RATES
    }
    lines = ["field\tpositive_count\tfloor\tstatus"]
    for field in FIELDS_FOR_RATES:
        floor = floors.get(field, 0)
        status = "ok" if positive_counts[field] >= floor else "FLAG"
        lines.append(f"{field}\t{positive_counts[field]}\t{floor}\t{status}")
    antecedente_used = sorted({item for note in notes for item in note["antecedente"]})
    med_named_dose = sum(
        1 for note in notes for med in note["medicatie_actuala"] if med["doza"]
    )
    lines.append(f"antecedente_enum_coverage\t{len(antecedente_used)} ({','.join(antecedente_used)})\t10\t{'ok' if len(antecedente_used) >= 10 else 'FLAG'}")
    lines.append(f"meds_with_real_dose\t{med_named_dose}\t10\t{'ok' if med_named_dose >= 10 else 'FLAG'}")
    return "\n".join(lines)


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
    assert len(rows) == 50
    write_plan(out_root / "plan.tsv", rows)
    write_seed_manifest(out_root)

    index_lines = ["conversation_id\ttranscript_path\tnote_path"]
    realized_counts = {field: 0 for field in FIELDS_FOR_RATES}
    all_transcripts: list[str] = []
    all_notes: list[dict[str, Any]] = []
    previous_sentences: set[str] = set()

    for batch in range(1, 6):
        batch_rows = [row for row in rows if row["batch"] == str(batch)]
        for row in batch_rows:
            cid = row["conversation_id"]
            note = note_from_plan(row)
            if validate is not None:
                validate(instance=note, schema=NOTE_SCHEMA)
            transcript = transcript_from_plan(row, previous_sentences)
            if note["evaluare_functionala_initiala"] is not None:
                assert note["evaluare_functionala_initiala"] in transcript
            for field in FIELDS_FOR_RATES:
                realized_counts[field] += int(is_populated(note[field]))
            all_notes.append(note)
            note_path = refs_dir / f"{cid}.json"
            transcript_path = transcripts_dir / f"{cid}.txt"
            note_path.write_text(json.dumps(note, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            transcript_path.write_text(transcript + "\n", encoding="utf-8")
            index_lines.append(f"{cid}\t{transcript_path.relative_to(ROOT)}\t{note_path.relative_to(ROOT)}")
            all_transcripts.append(transcript)
            previous_sentences.update(sentence_counts([transcript]).keys())

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

    diversity = diversity_report(rows, all_transcripts)
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
