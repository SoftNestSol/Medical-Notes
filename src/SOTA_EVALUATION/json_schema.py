"""
Osteopath Concept Note v1 — schema JSON + enum-uri.
Sursa unică de adevăr pentru toate condițiile experimentale.
"""

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

NOTE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://chiro-ro/schemas/osteopath-concept-note-v1.json",
    "title": "Osteopath Concept Note v1",
    "description": (
        "Structured note extracted from a Romanian chiropractor-patient "
        "conversation. Rule: if not spoken, leave empty (null or empty list)."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": [
        "motivul_prezentarii",
        "evaluarea_durerii_vas",
        "localizarea_durerii",
        "localizarea_durerii_alta",
        "antecedente",
        "antecedente_altele",
        "medicatie_actuala",
        "evaluare_functionala_initiala",
    ],
    "properties": {
        "motivul_prezentarii": {
            "description": "Free-text summary of patient's main complaint. Null if not verbalized.",
            "type": ["string", "null"],
        },
        "evaluarea_durerii_vas": {
            "description": "Pain score 0-10. Only if a number is explicitly spoken. Null otherwise.",
            "type": ["integer", "null"],
            "minimum": 0,
            "maximum": 10,
        },
        "localizarea_durerii": {
            "description": "Multi-select of anatomical regions explicitly named or confirmed.",
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "enum": LOCALIZARE_ENUM},
        },
        "localizarea_durerii_alta": {
            "description": (
                "Free-text escape hatch for spoken pain locations not in the "
                "enum (e.g. 'fesier', 'coapsa'). Null if none."
            ),
            "type": ["string", "null"],
        },
        "antecedente": {
            "description": "Multi-select of patient's confirmed personal medical history.",
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "enum": ANTECEDENTE_ENUM},
        },
        "antecedente_altele": {
            "description": "Free-text escape hatch for spoken conditions not in the enum. Null if none.",
            "type": ["string", "null"],
        },
        "medicatie_actuala": {
            "description": (
                "List of current medications. Empty list if patient takes "
                "nothing or no medication is mentioned."
            ),
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["denumire", "doza"],
                "properties": {
                    "denumire": {"type": "string", "minLength": 1},
                    "doza": {"type": ["string", "null"]},
                },
            },
        },
        "evaluare_functionala_initiala": {
            "description": (
                "Therapist's verbalized observations about posture, "
                "biomechanics, mobility. Null if not spoken aloud."
            ),
            "type": ["string", "null"],
        },
    },
}