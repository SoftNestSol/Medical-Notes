import json
import re
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from .metrics import ScoreResult, score_pair

_MANUAL_SUFFIX_RE = re.compile(r"_manual$")


def _pair_name(manual_path: Path) -> str:
    """audio_2_manual.json -> audio_2"""
    return _MANUAL_SUFFIX_RE.sub("", manual_path.stem)


def find_pairs(manual_dir: Path, hyp_dir: Path) -> List[tuple[Path, Path, str]]:
    pairs = []
    for ref in sorted(manual_dir.glob("*_manual.json")):
        name = _pair_name(ref)
        hyp = hyp_dir / f"{name}.json"
        if not hyp.exists():
            print(f"[skip] no hypothesis for {name} at {hyp}")
            continue
        pairs.append((ref, hyp, name))
    return pairs


def evaluate(manual_dir: Path, hyp_dir: Path, ascii_fold: bool = False) -> List[ScoreResult]:
    pairs = find_pairs(manual_dir, hyp_dir)
    if not pairs:
        raise SystemExit("no manual/hypothesis pairs found")
    return [score_pair(ref, hyp, name, ascii_fold=ascii_fold) for ref, hyp, name in pairs]


def render(results: List[ScoreResult], console: Optional[Console] = None) -> None:
    console = console or Console()
    table = Table(title="STT evaluation", show_lines=False)
    table.add_column("file", style="cyan")
    table.add_column("WER", justify="right")
    table.add_column("cpWER", justify="right")
    table.add_column("ref spk", justify="right")
    table.add_column("hyp spk", justify="right")
    table.add_column("miss/falarm", justify="right")
    table.add_column("ref words", justify="right")
    table.add_column("hyp words", justify="right")
    for r in results:
        table.add_row(
            r.name,
            f"{r.wer*100:5.1f}%",
            f"{r.cpwer*100:5.1f}%",
            str(r.ref_speakers),
            str(r.hyp_speakers),
            f"{r.missed_speaker}/{r.falarm_speaker}",
            str(r.ref_words),
            str(r.hyp_words),
        )
    n = len(results)
    if n:
        mean_wer = sum(r.wer for r in results) / n
        mean_cp = sum(r.cpwer for r in results) / n
        table.add_section()
        table.add_row(
            f"mean (n={n})",
            f"{mean_wer*100:5.1f}%",
            f"{mean_cp*100:5.1f}%",
            "", "", "", "", "",
        )
    console.print(table)


def dump_json(results: List[ScoreResult], path: Path) -> None:
    path.write_text(
        json.dumps([r.as_dict() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
