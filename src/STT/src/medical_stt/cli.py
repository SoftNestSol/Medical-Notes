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
):
    """Transcribe an audio file to speaker-labeled JSON."""
    settings = get_settings()
    transcript = transcribe(audio, settings=settings, diarize=not no_diarize)
    payload = transcript.model_dump()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if out:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out} ({len(transcript.segments)} segments)", err=True)
    else:
        sys.stdout.write(text + "\n")


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
