import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from .normalize import normalize


def load_turns(path: Path) -> List[Tuple[str, str]]:
    """Load a transcript JSON. Returns list of (speaker, text), in order.

    Accepts:
      - manual format: list of {speaker, text}
      - pipeline format: {segments: [{start, end, speaker, text}, ...]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "segments" in data:
        items = data["segments"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"{path}: unknown transcript shape")
    return [(item.get("speaker") or "UNK", item.get("text", "")) for item in items]


def concat_full(turns: List[Tuple[str, str]], ascii_fold: bool = False) -> str:
    """Concatenate all text in order, ignoring speakers. For plain WER."""
    return normalize(" ".join(t for _, t in turns), ascii_fold=ascii_fold)


def concat_per_speaker(turns: List[Tuple[str, str]], ascii_fold: bool = False) -> List[str]:
    """Group text by speaker, return list of normalized per-speaker strings.

    Order is by first-appearance of the speaker. cpWER is permutation-invariant
    so order doesn't affect the score, only the row order in debug output.
    """
    bucket: "OrderedDict[str, List[str]]" = OrderedDict()
    for spk, text in turns:
        bucket.setdefault(spk, []).append(text)
    return [normalize(" ".join(chunks), ascii_fold=ascii_fold) for chunks in bucket.values()]
