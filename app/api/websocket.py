from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.config import get_settings
from app.core.audio_buffer import AudioBuffer
from app.core.vad import SileroVAD, VADEvent
from app.core.transcriber import DeepgramTranscriber
from app.core.translator import registry as translator_registry
from app.models.schemas import TranslatedChunk, TTSVoice
from app.utils.audio import (
    decode_audio, AudioFormat, is_silent, chunk_audio, detect_format
)
from app.utils.lang_map import SUPPORTED_TARGET_LANGUAGES

logger = logging.getLogger(__name__)
router = APIRouter()

MIN_AUDIO_SECONDS  = 0.3
MAX_BUFFER_SECONDS = 10.0
IDLE_CLOSE_SECONDS = 120.0

_ENCODING_MAP: dict[str, AudioFormat] = {
    "pcm_s16le": AudioFormat.PCM_RAW,
    "pcm":       AudioFormat.PCM_RAW,
    "raw":       AudioFormat.PCM_RAW,
    "wav":       AudioFormat.WAV,
    "mp3":       AudioFormat.MP3,
    "ogg":       AudioFormat.OGG_OPUS,
    "opus":      AudioFormat.OGG_OPUS,
    "webm":      AudioFormat.WEBM_OPUS,
    "flac":      AudioFormat.FLAC,
    "m4a":       AudioFormat.M4A_AAC,
    "auto":      None,
}


async def _tts_task(
    text: str,
    voice: str,
    transcriber: DeepgramTranscriber,
    websocket: WebSocket,
) -> None:
    try:
        async for pcm_chunk in transcriber.stream_tts(text=text, voice=voice):
            try:
                await websocket.send_bytes(pcm_chunk)
            except Exception as e:
                logger.error(f"[TTS] Failed to send audio chunk to client: {e}")
                return
    except Exception as e:
        logger.error(f"[TTS] Task error: {e}", exc_info=True)


@router.websocket("/ws/stream/{target_lang}")
async def stream_endpoint(
    websocket: WebSocket,
    target_lang: str,
    source_lang: Optional[str] = Query(default=None),
    sample_rate: int = Query(default=16000),
    encoding: str = Query(default="auto"),
    voice: str = Query(default=TTSVoice.MALE),
):
    settings = get_settings()
    await websocket.accept()

    if target_lang not in SUPPORTED_TARGET_LANGUAGES:
        await websocket.send_json({
            "error": f"Unsupported target language: {target_lang}",
            "supported": SUPPORTED_TARGET_LANGUAGES,
        })
        await websocket.close(code=1003)
        return

    if voice not in TTSVoice.SUPPORTED:
        await websocket.send_json({
            "error": f"Unsupported voice: {voice}",
            "supported": TTSVoice.SUPPORTED,
        })
        await websocket.close(code=1003)
        return

    forced_fmt = _ENCODING_MAP.get(encoding.lower())
    transcriber: DeepgramTranscriber = websocket.app.state.transcriber

    logger.info(
        f"[WS] Session start | target={target_lang} source={source_lang} "
        f"sr={sample_rate} enc={encoding} voice={voice}"
    )

    vad = SileroVAD(
        threshold=settings.vad_threshold,
        min_speech_ms=settings.vad_min_speech_ms,
        min_silence_ms=settings.vad_min_silence_ms,
        sample_rate=16000,
    )
    vad.load()
    vad.reset_state()

    buffer           = AudioBuffer(sample_rate=16000, max_seconds=MAX_BUFFER_SECONDS)
    stream_start     = time.time()
    last_speech_time = time.time()
    in_speech        = False
    total_chunks     = 0
    session_closed   = False

    webm_header: bytes        = b""
    is_first_webm_chunk: bool = True

    result_task: Optional[asyncio.Task] = None

    async def stream_results_to_client() -> None:
        try:
            async for transcript in transcriber.aiter_results():
                if not transcript.text.strip():
                    continue

                try:
                    translated = await translator_registry.translate_one(
                        text=transcript.text,
                        source_lang=transcript.language,
                        target_lang=target_lang,
                    )
                except ValueError as e:
                    logger.warning(f"[RESULT] Lang pair not supported: {e}")
                    translated = transcript.text
                except Exception as e:
                    logger.error(f"[RESULT] Translation failed: {e}", exc_info=True)
                    translated = transcript.text

                chunk = TranslatedChunk(
                    original_text=transcript.text,
                    translated_text=translated,
                    source_language=transcript.language,
                    target_language=target_lang,
                    start_time=round(time.time() - stream_start, 3),
                    end_time=round(time.time() - stream_start, 3),
                    is_partial=transcript.is_partial,
                )

                label = "PARTIAL" if transcript.is_partial else "FINAL"
                logger.info(
                    f"[{label}] {transcript.language}→{target_lang} | "
                    f'"{transcript.text[:60]}" → "{translated[:60]}"'
                )

                try:
                    await websocket.send_json(chunk.model_dump())
                except Exception as e:
                    logger.error(f"[RESULT] WebSocket caption send failed: {e}")
                    return

                if not transcript.is_partial and translated.strip():
                    asyncio.create_task(
                        _tts_task(
                            text=translated,
                            voice=voice,
                            transcriber=transcriber,
                            websocket=websocket,
                        )
                    )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[RESULT] Streamer error: {e}", exc_info=True)

    async def on_speech_start() -> None:
        nonlocal result_task, in_speech
        if transcriber.is_stream_open:
            return

        try:
            await transcriber.open_stream(source_language=source_lang)
            result_task = asyncio.create_task(stream_results_to_client())
            in_speech = True
            logger.info("[WS] STT stream opened — speech started")
        except Exception as e:
            logger.error(f"[WS] Failed to open STT stream: {e}", exc_info=True)
            await websocket.send_json({"error": f"STT stream open failed: {e}"})

    async def on_speech_end() -> None:
        nonlocal in_speech
        if not transcriber.is_stream_open:
            in_speech = False
            return

        in_speech = False
        await transcriber.close_stream()

        if result_task and not result_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(result_task), timeout=8.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                result_task.cancel()

        logger.info("[WS] STT stream closed — speech ended")

    async def close_session_idle() -> None:
        nonlocal session_closed
        if session_closed:
            return
        session_closed = True

        logger.info(f"[WS] Idle timeout ({IDLE_CLOSE_SECONDS}s) — closing session")
        await on_speech_end()
        try:
            await websocket.send_json({
                "event": "silence_timeout",
                "message": f"Closed after {int(IDLE_CLOSE_SECONDS)}s of silence.",
            })
        except Exception:
            pass
        try:
            await websocket.close(code=1000)
        except Exception:
            pass

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=0.5
                )
            except asyncio.TimeoutError:
                now = time.time()

                if (now - last_speech_time) >= IDLE_CLOSE_SECONDS:
                    await close_session_idle()
                    break

                if transcriber.is_stream_open and buffer.duration_seconds >= MAX_BUFFER_SECONDS:
                    logger.info("[WS] Hard buffer limit — forcing speech end")
                    full_audio = await buffer.export_and_trim()
                    if len(full_audio) > 0:
                        await transcriber.send_audio(full_audio)
                    await on_speech_end()

                continue

            if message["type"] == "websocket.receive" and "text" in message:
                cmd = message["text"].strip().upper()
                if cmd == "STOP":
                    logger.info("[WS] Client sent STOP")
                    await on_speech_end()
                    break
                if cmd == "PING":
                    await websocket.send_text("PONG")
                continue

            if message["type"] == "websocket.receive" and "bytes" in message:
                raw_bytes = message["bytes"]
                if not raw_bytes:
                    continue

                total_chunks += 1

                detected = forced_fmt or detect_format(raw_bytes)
                if detected == AudioFormat.WEBM_OPUS:
                    if is_first_webm_chunk:
                        webm_header = raw_bytes
                        is_first_webm_chunk = False
                        decode_bytes = raw_bytes
                    else:
                        decode_bytes = webm_header + raw_bytes
                else:
                    decode_bytes = raw_bytes
                    is_first_webm_chunk = False

                try:
                    audio_f32 = decode_audio(
                        decode_bytes,
                        target_sr=16000,
                        fmt=forced_fmt,
                        pcm_input_sr=sample_rate,
                    )
                except Exception as e:
                    logger.error(f"[WS] Decode error: {e}")
                    continue

                if len(audio_f32) == 0 or is_silent(audio_f32):
                    continue

                await buffer.push(audio_f32)

                vad_speech_started = False
                vad_speech_ended   = False

                for vc in chunk_audio(audio_f32, vad.CHUNK_SIZE):
                    result = vad.process_chunk(vc)
                    if result.event == VADEvent.SPEECH_START:
                        vad_speech_started = True
                    elif result.event == VADEvent.SPEECH_END:
                        vad_speech_ended = True
                        break

                if vad_speech_started or (in_speech and not vad_speech_ended):
                    last_speech_time = time.time()

                if vad_speech_started and not in_speech:
                    await on_speech_start()

                if in_speech and transcriber.is_stream_open:
                    full_audio = await buffer.export_and_trim()
                    if len(full_audio) > 0:
                        await transcriber.send_audio(full_audio)

                if vad_speech_ended and in_speech:
                    logger.debug("[WS] VAD: speech ended → closing STT stream")
                    await on_speech_end()

            elif message["type"] == "websocket.disconnect":
                logger.info("[WS] Client disconnected.")
                break

    except WebSocketDisconnect:
        logger.info("[WS] WebSocket disconnected.")
    except Exception as e:
        logger.exception(f"[WS] Unexpected error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        if result_task and not result_task.done():
            result_task.cancel()
        await transcriber.close_stream()
        await buffer.clear()
        logger.info(
            f"[WS] Session ended | chunks={total_chunks} "
            f"duration={time.time()-stream_start:.1f}s"
        )
        try:
            await websocket.close()
        except Exception:
            pass
