from pydantic import BaseModel, Field
from typing import Optional


class TranscriptChunk(BaseModel):
    text: str
    language: str
    language_probability: float = Field(ge=0.0, le=1.0)
    start_time: float
    end_time: float
    is_partial: bool = False


class TranslatedChunk(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    start_time: float
    end_time: float
    is_partial: bool = False


class StreamConfig(BaseModel):
    target_language: str = "en"
    source_language: Optional[str] = None
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    voice: str = "aura-2-arcas-en"


class HealthResponse(BaseModel):
    status: str
    stt_model: str
    device: str
    loaded_translation_pairs: list[str]


class LanguagesResponse(BaseModel):
    supported_targets: list[str]
    auto_detect_sources: bool = True


class TTSVoice:
    MALE = "aura-2-arcas-en"
    FEMALE = "aura-2-thalia-en"
    SUPPORTED = ["aura-2-arcas-en", "aura-2-thalia-en"]
