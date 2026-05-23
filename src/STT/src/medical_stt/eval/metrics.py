from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import jiwer
from meeteval.wer import cp_word_error_rate

from .loaders import concat_full, concat_per_speaker, load_turns


@dataclass
class ScoreResult:
    name: str
    ref_path: str
    hyp_path: str
    wer: float
    cpwer: float
    ref_speakers: int
    hyp_speakers: int
    missed_speaker: int
    falarm_speaker: int
    scored_speaker: int
    ref_words: int
    hyp_words: int

    def as_dict(self) -> dict:
        return asdict(self)


def score_pair(
    ref_path: Path, hyp_path: Path, name: Optional[str] = None, ascii_fold: bool = False,
) -> ScoreResult:
    ref_turns = load_turns(ref_path)
    hyp_turns = load_turns(hyp_path)

    ref_text = concat_full(ref_turns, ascii_fold=ascii_fold)
    hyp_text = concat_full(hyp_turns, ascii_fold=ascii_fold)
    wer = float(jiwer.wer(ref_text, hyp_text)) if ref_text else float("nan")

    ref_per_spk = concat_per_speaker(ref_turns, ascii_fold=ascii_fold)
    hyp_per_spk = concat_per_speaker(hyp_turns, ascii_fold=ascii_fold)
    # meeteval skips empty strings as missing speakers; filter them out
    ref_per_spk = [s for s in ref_per_spk if s.strip()]
    hyp_per_spk = [s for s in hyp_per_spk if s.strip()] or [""]

    cp = cp_word_error_rate(ref_per_spk, hyp_per_spk)

    return ScoreResult(
        name=name or Path(ref_path).stem,
        ref_path=str(ref_path),
        hyp_path=str(hyp_path),
        wer=wer,
        cpwer=float(cp.error_rate) if cp.error_rate is not None else float("nan"),
        ref_speakers=len(ref_per_spk),
        hyp_speakers=len([s for s in hyp_per_spk if s.strip()]),
        missed_speaker=int(cp.missed_speaker),
        falarm_speaker=int(cp.falarm_speaker),
        scored_speaker=int(cp.scored_speaker),
        ref_words=len(ref_text.split()),
        hyp_words=len(hyp_text.split()),
    )
