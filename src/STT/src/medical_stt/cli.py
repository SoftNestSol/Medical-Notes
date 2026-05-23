import json
import sys
from pathlib import Path
from typing import Optional

import typer

from .config import get_settings
from .pipeline import transcribe

app = typer.Typer(add_completion=False, help="Medical STT CLI")


@app.command()
def run(
    audio: Path = typer.Argument(..., exists=True, readable=True, help="Audio file path"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output JSON path (default: stdout)"),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip speaker diarization"),
    num_speakers: Optional[int] = typer.Option(None, "--num-speakers", help="Exact speaker count"),
    min_speakers: Optional[int] = typer.Option(None, "--min-speakers", help="Min speaker count"),
    max_speakers: Optional[int] = typer.Option(None, "--max-speakers", help="Max speaker count"),
    no_preprocess: bool = typer.Option(False, "--no-preprocess", help="Skip ffmpeg highpass/lowpass/loudnorm"),
    vad_onset: float = typer.Option(0.300, "--vad-onset", help="Pyannote VAD onset threshold (lower=more permissive)"),
    vad_offset: float = typer.Option(0.200, "--vad-offset", help="Pyannote VAD offset threshold"),
):
    """Transcribe an audio file to speaker-labeled JSON."""
    settings = get_settings()
    transcript = transcribe(
        audio, settings=settings, diarize=not no_diarize,
        num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers,
        preprocess=not no_preprocess,
        vad_onset=vad_onset, vad_offset=vad_offset,
    )
    payload = transcript.model_dump()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if out:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out} ({len(transcript.segments)} segments)", err=True)
    else:
        sys.stdout.write(text + "\n")


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory containing audio files"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", help="Output dir for JSON (default: alongside audio)"),
    pattern: str = typer.Option("*.mp4,*.mp3,*.wav,*.m4a,*.flac,*.ogg", "--pattern", help="Comma-separated glob patterns"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirs"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--overwrite", help="Skip files whose JSON already exists"),
    no_diarize: bool = typer.Option(False, "--no-diarize"),
    num_speakers: Optional[int] = typer.Option(None, "--num-speakers"),
    min_speakers: Optional[int] = typer.Option(None, "--min-speakers"),
    max_speakers: Optional[int] = typer.Option(None, "--max-speakers"),
    no_preprocess: bool = typer.Option(False, "--no-preprocess"),
    vad_onset: float = typer.Option(0.300, "--vad-onset"),
    vad_offset: float = typer.Option(0.200, "--vad-offset"),
):
    """Transcribe every audio file in a directory."""
    settings = get_settings()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    globber = input_dir.rglob if recursive else input_dir.glob
    files: list[Path] = []
    for pat in pattern.split(","):
        files.extend(globber(pat.strip()))
    files = sorted(set(files))

    if not files:
        typer.echo("no matching files", err=True)
        raise typer.Exit(1)

    typer.echo(f"found {len(files)} file(s)", err=True)
    ok = 0
    fail = 0
    for i, audio_path in enumerate(files, 1):
        target = (out_dir / f"{audio_path.stem}.json") if out_dir else audio_path.with_suffix(".json")
        if skip_existing and target.exists():
            typer.echo(f"[{i}/{len(files)}] skip {audio_path.name} (exists)", err=True)
            continue
        typer.echo(f"[{i}/{len(files)}] {audio_path.name} → {target}", err=True)
        try:
            transcript = transcribe(
                audio_path, settings=settings, diarize=not no_diarize,
                num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers,
                preprocess=not no_preprocess,
                vad_onset=vad_onset, vad_offset=vad_offset,
            )
            target.write_text(
                json.dumps(transcript.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            typer.echo(f"  → {len(transcript.segments)} segments", err=True)
            ok += 1
        except Exception as e:
            typer.echo(f"  ! failed: {e}", err=True)
            fail += 1

    typer.echo(f"done. ok={ok} fail={fail} skipped={len(files)-ok-fail}", err=True)
    if fail:
        raise typer.Exit(1)


@app.command(name="eval")
def eval_cmd(
    manual_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Dir with <name>_manual.json files"),
    hyp_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Dir with pipeline <name>.json outputs"),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Also write per-file results to JSON"),
    ascii_fold: bool = typer.Option(False, "--ascii-fold", help="Strip Romanian diacritics before scoring"),
):
    """Score pipeline outputs against hand-corrected references (WER + cpWER)."""
    from .eval.runner import evaluate, render, dump_json
    results = evaluate(manual_dir, hyp_dir, ascii_fold=ascii_fold)
    render(results)
    if json_out:
        dump_json(results, json_out)
        typer.echo(f"wrote {json_out}", err=True)


@app.command()
def config():
    """Print resolved settings (sans secrets)."""
    s = get_settings()
    d = s.model_dump()
    if d.get("hf_token"):
        d["hf_token"] = "***set***"
    typer.echo(json.dumps(d, indent=2, default=str))


if __name__ == "__main__":
    app()
