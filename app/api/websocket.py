from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.config import get_settings
from app.core.audio_buffer import AudioBuffer
from app.core.vad import SileroVAD, VADEvent
from app.core.transcriber import DeepgramTranscriber, TranscriptionResult
from app.core.translator import registry as translator_registry
from app.models.schemas import TranslatedChunk, TTSVoice
from app.utils.audio import (
    decode_audio, AudioFormat, is_silent, chunk_audio, detect_format
)
from app.utils.lang_map import SUPPORTED_TARGET_LANGUAGES

logger = logging.getLogger(__name__)
router = APIRouter()

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

# ── VAD singleton — loaded once at module level, shared across sessions ───────
# FIX #4: Was re-loading Silero on every WebSocket connection (2s cold start per session).
# The model itself is stateless; per-session state is managed via vad.reset_state().
_vad_singleton: Optional[SileroVAD] = None

def get_shared_vad(settings) -> SileroVAD:
    global _vad_singleton
    if _vad_singleton is None:
        _vad_singleton = SileroVAD(
            threshold=settings.vad_threshold,
            min_speech_ms=settings.vad_min_speech_ms,
            min_silence_ms=settings.vad_min_silence_ms,
            sample_rate=16000,
        )
        _vad_singleton.load()
        logger.info("[VAD] Singleton loaded.")
    return _vad_singleton


async def _tts_task(
    text: str,
    voice: str,
    transcriber: DeepgramTranscriber,
    websocket: WebSocket,
    ws_open_flag: list[bool],          # FIX #3: mutable flag checked before each send
) -> None:
    try:
        async for pcm_chunk in transcriber.stream_tts(text=text, voice=voice):
            # FIX #3: Guard — don't attempt send if WebSocket already closed
            if not ws_open_flag[0]:
                logger.debug("[TTS] WebSocket closed, aborting TTS stream.")
                return
            try:
                await websocket.send_bytes(pcm_chunk)
            except Exception as e:
                logger.error(f"[TTS] Failed to send audio chunk: {e}")
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

    # FIX #4: Get shared VAD, reset per-session state only
    vad = get_shared_vad(settings)
    vad.reset_state()

    buffer           = AudioBuffer(sample_rate=16000, max_seconds=MAX_BUFFER_SECONDS)
    stream_start     = time.time()
    last_speech_time = time.time()
    in_speech        = False
    total_chunks     = 0
    session_closed   = False

    # FIX #3: Mutable flag so background TTS tasks know when WS is gone
    ws_open = [True]

    webm_header: bytes        = b""
    is_first_webm_chunk: bool = True

    result_task: Optional[asyncio.Task] = None
    last_partial: Optional[TranscriptionResult] = None

    async def stream_results_to_client() -> None:
        nonlocal last_partial
        try:
            async for item in transcriber.aiter_results():
                if item == "utterance_end":
                    if last_partial is not None:
                        final = last_partial
                        last_partial = None
                        await _emit(final, is_final=True)
                    continue

                result: TranscriptionResult = item
                if not result.text.strip():
                    continue

                if result.is_partial:
                    last_partial = result
                    await _emit(result, is_final=False)
                else:
                    last_partial = None
                    await _emit(result, is_final=True)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[RESULT] Streamer error: {e}", exc_info=True)

    async def _emit(result: TranscriptionResult, is_final: bool) -> None:
        try:
            translated = await translator_registry.translate_one(
                text=result.text,
                source_lang=result.language,
                target_lang=target_lang,
            )
        except ValueError as e:
            logger.warning(f"[RESULT] Lang pair not supported: {e}")
            translated = result.text
        except Exception as e:
            logger.error(f"[RESULT] Translation failed: {e}", exc_info=True)
            translated = result.text

        chunk = TranslatedChunk(
            original_text=result.text,
            translated_text=translated,
            source_language=result.language,
            target_language=target_lang,
            start_time=round(time.time() - stream_start, 3),
            end_time=round(time.time() - stream_start, 3),
            is_partial=not is_final,
        )

        label = "PARTIAL" if not is_final else "FINAL"
        logger.info(
            f"[{label}] {result.language}→{target_lang} | "
            f'"{result.text[:60]}" → "{translated[:60]}"'
        )

        # FIX #3: Only send if WebSocket is still open
        if not ws_open[0]:
            return

        try:
            await websocket.send_json(chunk.model_dump())
        except Exception as e:
            logger.error(f"[RESULT] WebSocket send failed: {e}")
            return

        if is_final and translated.strip() and ws_open[0]:
            asyncio.create_task(
                _tts_task(
                    text=translated,
                    voice=voice,
                    transcriber=transcriber,
                    websocket=websocket,
                    ws_open_flag=ws_open,   # pass the live flag
                )
            )

    async def ensure_stream_open() -> None:
        nonlocal result_task
        if transcriber.is_stream_open:
            return
        await transcriber.open_stream(source_language=source_lang)
        if result_task is None or result_task.done():
            result_task = asyncio.create_task(stream_results_to_client())
        logger.info("[WS] STT stream opened")

    # FIX #1 + #5: Flush audio and close the STT stream on SPEECH_END so
    # Deepgram finalizes the utterance. Reset in_speech so next speech
    # re-opens the stream cleanly.
    async def on_speech_end() -> None:
        nonlocal in_speech
        logger.info("[VAD] SPEECH_END — flushing audio, closing STT stream for finalization")
        in_speech = False

        remaining = await buffer.export_and_clear()
        if len(remaining) > 0 and transcriber.is_stream_open:
            await transcriber.send_audio(remaining)

        # Close the STT stream so Deepgram emits is_final=True for the utterance.
        # The result_task keeps draining until it gets the sentinel None from aiter_results.
        # A new stream will be opened on the next SPEECH_START.
        await transcriber.close_stream()
        logger.info("[VAD] STT stream closed after SPEECH_END — awaiting final result")

    async def close_session() -> None:
        nonlocal session_closed
        if session_closed:
            return
        session_closed = True

        remaining = await buffer.export_and_clear()
        if len(remaining) > 0 and transcriber.is_stream_open:
            await transcriber.send_audio(remaining)

        await transcriber.close_stream()

        if result_task and not result_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(result_task), timeout=8.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                result_task.cancel()

        logger.info("[WS] Session closed")

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=0.5
                )
            except asyncio.TimeoutError:
                now = time.time()
                if (now - last_speech_time) >= IDLE_CLOSE_SECONDS:
                    logger.info(f"[WS] Idle timeout — closing session")
                    await close_session()
                    try:
                        await websocket.send_json({
                            "event": "silence_timeout",
                            "message": f"Closed after {int(IDLE_CLOSE_SECONDS)}s of silence.",
                        })
                    except Exception:
                        pass
                    break
                continue

            if message["type"] == "websocket.receive" and "text" in message:
                cmd = message["text"].strip().upper()
                if cmd == "STOP":
                    logger.info("[WS] Client sent STOP")
                    await close_session()
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

                # FIX #1 + #2: Process ALL chunks — don't break early on SPEECH_START.
                # Also handle SPEECH_END to flush + finalize the utterance.
                speech_started_this_frame = False
                speech_ended_this_frame   = False

                for vc in chunk_audio(audio_f32, vad.CHUNK_SIZE):
                    result = vad.process_chunk(vc)

                    if result.event == VADEvent.SPEECH_START:
                        speech_started_this_frame = True
                        # Don't break — keep processing remaining chunks

                    elif result.event == VADEvent.SPEECH_END:
                        speech_ended_this_frame = True
                        # Don't break — there may be more chunks after silence

                # Act on events after processing the full frame
                if speech_started_this_frame and not in_speech:
                    in_speech = True
                    last_speech_time = time.time()
                    await ensure_stream_open()

                if speech_ended_this_frame and not in_speech:
                    # in_speech was already reset inside on_speech_end,
                    # but SPEECH_END from VAD may come first — handle cleanly
                    pass

                if in_speech:
                    last_speech_time = time.time()
                    await buffer.push(audio_f32)
                    audio_to_send = await buffer.export_and_clear()
                    if len(audio_to_send) > 0 and transcriber.is_stream_open:
                        await transcriber.send_audio(audio_to_send)

                # FIX #1: Trigger finalization after the frame is fully processed
                # and audio has been flushed to Deepgram.
                if speech_ended_this_frame:
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
        # FIX #3: Mark WS as closed BEFORE any remaining cleanup
        ws_open[0] = False

        if result_task and not result_task.done():
            result_task.cancel()
        if not session_closed:
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