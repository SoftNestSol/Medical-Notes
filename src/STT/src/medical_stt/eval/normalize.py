import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\săâîșțĂÂÎȘȚ]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_ASCII_DIACRITIC_MAP = str.maketrans({
    "ş": "ș", "Ş": "Ș", "ţ": "ț", "Ţ": "Ț",
})


_ASCII_FOLD_MAP = str.maketrans({
    "ă": "a", "Ă": "A", "â": "a", "Â": "A",
    "î": "i", "Î": "I", "ș": "s", "Ș": "S",
    "ş": "s", "Ş": "S", "ț": "t", "Ț": "T",
    "ţ": "t", "Ţ": "T",
})


def normalize(text: str, ascii_fold: bool = False) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Romanian diacritics ă/â/î/ș/ț preserved by default; legacy ş/ţ remapped to
    ș/ț. With ascii_fold=True, all Romanian diacritics are stripped — useful
    when references are typed without diacritics.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ASCII_DIACRITIC_MAP)
    if ascii_fold:
        text = text.translate(_ASCII_FOLD_MAP)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
