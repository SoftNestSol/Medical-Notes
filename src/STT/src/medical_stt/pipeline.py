import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import Settings, get_settings
from .schemas import Segment, Transcript


def _ffmpeg_clean(src: str) -> str:
    """Highpass 80Hz + lowpass 8kHz + loudnorm → 16kHz mono wav. Returns temp path."""
    if shutil.which("ffmpeg") is None:
        return src
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", src,
        "-af", "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ac", "1", "-ar", "16000", out,
    ]
    try:
        subprocess.run(cmd, check=True)
        return out
    except subprocess.CalledProcessError as e:
        print(f"[warn] ffmpeg preprocess failed: {e}; using original")
        return src


def _load_glossary_prompt(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    terms = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return ", ".join(terms) if terms else None


def transcribe(
    audio_path: str | Path,
    settings: Optional[Settings] = None,
    diarize: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    preprocess: bool = True,
    vad_onset: float = 0.300,
    vad_offset: float = 0.200,
) -> Transcript:
    import whisperx

    s = settings or get_settings()
    audio_path = str(audio_path)

    work_path = _ffmpeg_clean(audio_path) if preprocess else audio_path

    asr_options = {"condition_on_previous_text": False}
    initial_prompt = _load_glossary_prompt(s.glossary_path)
    if initial_prompt:
        asr_options["initial_prompt"] = initial_prompt

    model = whisperx.load_model(
        s.whisper_model,
        device=s.device,
        compute_type=s.compute_type,
        language=s.language,
        asr_options=asr_options or None,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
    )
    audio = whisperx.load_audio(work_path)
    result = model.transcribe(audio, batch_size=s.batch_size, language=s.language)

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=s.language, device=s.device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, s.device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"[warn] alignment skipped: {e}")

    if diarize:
        if not s.hf_token:
            print("[warn] HF_TOKEN missing; skipping diarization")
        else:
            try:
                from whisperx.diarize import DiarizationPipeline
                diar = DiarizationPipeline(
                    token=s.hf_token, device=s.device
                )
                diar_kwargs = {}
                if num_speakers is not None:
                    diar_kwargs["num_speakers"] = num_speakers
                if min_speakers is not None:
                    diar_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diar_kwargs["max_speakers"] = max_speakers
                diar_segments = diar(audio, **diar_kwargs)
                result = whisperx.assign_word_speakers(diar_segments, result)
            except Exception as e:
                print(f"[warn] diarization failed: {e}")

    segments = [
        Segment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            speaker=seg.get("speaker"),
            text=(seg.get("text") or "").strip(),
        )
        for seg in result.get("segments", [])
    ]

    return Transcript(
        audio_path=audio_path,
        language=s.language,
        model=s.whisper_model,
        segments=segments,
    )
