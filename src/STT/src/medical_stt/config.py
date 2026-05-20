from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: Optional[str] = None
    whisper_model: str = "large-v3-turbo"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "ro"
    batch_size: int = 8
    glossary_path: Optional[Path] = None


def get_settings() -> Settings:
    return Settings()
