from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"
    deepgram_language: str = "en"

    vad_threshold: float = 0.40
    vad_min_speech_ms: int = 150
    vad_min_silence_ms: int = 800

    audio_sample_rate: int = 16000
    audio_channels: int = 1
    chunk_duration_ms: int = 100
    max_buffer_seconds: int = 30

    translation_max_length: int = 512
    translation_batch_size: int = 1

    host: str = "0.0.0.0"
    port: int = 8000
    max_ws_connections: int = 50
    log_level: str = "info"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
