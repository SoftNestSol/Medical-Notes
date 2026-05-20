from typing import List, Optional
from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    speaker: Optional[str] = None
    text: str


class Transcript(BaseModel):
    audio_path: str
    language: str
    model: str
    segments: List[Segment]
