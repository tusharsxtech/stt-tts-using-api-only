# FastAPI application entry point for Audio Transcribe, Translate & TTS Service
# Lifespan manages Deepgram STT init, VAD singleton init, and translation model pre-warming on startup

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.websocket import router as ws_router, get_shared_vad
from app.core.transcriber import DeepgramTranscriber
from app.core.translator import registry as translator_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    settings = get_settings()
    logger.info("=== Audio Translate Service starting ===")

    # STT
    transcriber = DeepgramTranscriber(
        api_key=settings.deepgram_api_key,
        model_id=settings.deepgram_model,
        realtime_model_id=settings.deepgram_model,
    )
    await asyncio.to_thread(transcriber.load)
    app.state.transcriber = transcriber

    # FIX: Pre-load VAD singleton at startup (not per-session).
    # get_shared_vad() is idempotent — safe to call here and from websocket handlers.
    await asyncio.to_thread(get_shared_vad, settings)
    logger.info("Silero VAD pre-loaded at startup.")

    # Translation models
    common_pairs = [("en", "hi"), ("en", "fr"), ("en", "de"), ("en", "es"),
                    ("hi", "en"), ("fr", "en"), ("de", "en"), ("es", "en")]
    logger.info("Pre-warming common translation models...")
    await translator_registry.preload(common_pairs)

    logger.info("=== All models ready. Accepting requests. ===")
    yield

    logger.info("=== Audio Translate Service shutting down ===")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Audio Transcribe, Translate & TTS Service",
        description=(
            "Real-time audio → transcription → translation → TTS microservice.\n\n"
            "**WebSocket** `/ws/stream/{target_lang}` for live streaming.\n"
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ws_router, tags=["streaming"])

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "audio-translate-service",
            "version": "2.0.0",
            "docs": "/docs",
            "websocket": "/ws/stream/{target_lang}",
        }

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )