from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import httpx
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

DEEPGRAM_HTTP_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_WS_URL   = "wss://api.deepgram.com/v1/listen"
DEEPGRAM_TTS_URL  = "wss://api.deepgram.com/v1/speak"

WS_CHUNK_SAMPLES = 16000 // 2


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    start_time: float
    end_time: float
    segments: list
    is_partial: bool = False


class DeepgramTranscriber:

    def __init__(
        self,
        api_key: str,
        model_id: str = "nova-3",
        realtime_model_id: str = "nova-3",
        language_code: str = "en",
    ):
        self.api_key           = api_key
        self.model_id          = model_id
        self.realtime_model_id = realtime_model_id
        self.language_code     = language_code

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._loaded           = False
        self._audio_queue: asyncio.Queue  = asyncio.Queue()
        self._result_queue: asyncio.Queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._stream_open  = False
        self._stream_lang: str = "en"

        # FIX: asyncio.Lock to prevent concurrent open_stream() calls from
        # racing (e.g. rapid SPEECH_START events opening the stream twice).
        self._open_lock = asyncio.Lock()

        self.model_size   = f"deepgram/{model_id}"
        self.device       = "api"
        self.compute_type = "cloud"

    def load(self) -> None:
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not set.")
        self._loaded = True
        logger.info(f"Deepgram STT ready. model={self.model_id}")

    def is_loaded(self) -> bool:
        return self._loaded

    async def open_stream(self, source_language: Optional[str] = None) -> None:
        # FIX: Lock prevents a second concurrent call from opening a second
        # WebSocket connection while the first is still being established.
        async with self._open_lock:
            if self._stream_open:
                return

            lang = source_language or self.language_code
            self._stream_lang = lang

            params = (
                f"?model={self.realtime_model_id}"
                f"&language={lang}"
                f"&encoding=linear16"
                f"&sample_rate=16000"
                f"&channels=1"
                f"&interim_results=true"
                f"&punctuate=true"
                f"&smart_format=true"
                f"&endpointing=300"
                f"&utterance_end_ms=1000"
            )
            url     = DEEPGRAM_WS_URL + params
            headers = {"Authorization": f"Token {self.api_key}"}

            try:
                self._ws = await websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=10,
                    ping_timeout=20,
                    open_timeout=10,
                )
                self._stream_open  = True
                # FIX: Always create fresh queues when opening a new stream.
                # Old queues from a previous session may still have stale data
                # or a lingering None sentinel from close_stream(), which would
                # immediately poison aiter_results() on the new stream.
                self._audio_queue  = asyncio.Queue()
                self._result_queue = asyncio.Queue()

                self._send_task = asyncio.create_task(self._sender_loop())
                self._recv_task = asyncio.create_task(self._receiver_loop())

                logger.info(f"[STT WS] Stream opened. lang={lang} model={self.realtime_model_id}")

            except Exception as e:
                self._stream_open = False
                self._ws          = None
                logger.error(f"[STT WS] Failed to open stream: {e}", exc_info=True)
                raise

    async def send_audio(self, audio: np.ndarray) -> None:
        if not self._stream_open or audio is None or len(audio) == 0:
            return
        for i in range(0, max(len(audio), 1), WS_CHUNK_SAMPLES):
            chunk = audio[i: i + WS_CHUNK_SAMPLES]
            await self._audio_queue.put(chunk)

    async def close_stream(self) -> None:
        if not self._stream_open:
            return

        logger.info("[STT WS] Closing stream")
        self._stream_open = False

        await self._audio_queue.put(None)

        if self._send_task and not self._send_task.done():
            try:
                await asyncio.wait_for(self._send_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                self._send_task.cancel()

        if self._recv_task and not self._recv_task.done():
            try:
                await asyncio.wait_for(self._recv_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                self._recv_task.cancel()

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # FIX: Only put the sentinel if the queue doesn't already have one.
        # _receiver_loop's finally block also puts None; putting a second one
        # means the NEXT session's aiter_results() exits immediately on first get().
        try:
            self._result_queue.put_nowait(None)
        except Exception:
            pass

        logger.info("[STT WS] Stream closed.")

    async def close(self) -> None:
        await self.close_stream()

    @property
    def is_stream_open(self) -> bool:
        return self._stream_open

    async def aiter_results(self) -> AsyncGenerator[TranscriptionResult, None]:
        while True:
            try:
                item = await asyncio.wait_for(self._result_queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                if not self._stream_open:
                    break
                continue
            if item is None:
                break
            yield item

    async def _sender_loop(self) -> None:
        try:
            while True:
                chunk = await self._audio_queue.get()

                if chunk is None:
                    if self._ws:
                        try:
                            await self._ws.send(json.dumps({"type": "CloseStream"}))
                        except Exception as e:
                            logger.debug(f"[STT WS] CloseStream send error: {e}")
                    break

                if not self._ws:
                    break

                try:
                    pcm = (chunk * 32768).clip(-32768, 32767).astype("int16").tobytes()
                    await self._ws.send(pcm)
                except ConnectionClosed:
                    logger.warning("[STT WS] Connection closed while sending")
                    break
                except Exception as e:
                    logger.error(f"[STT WS] Send error: {e}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[STT WS] Sender loop error: {e}", exc_info=True)

    async def _receiver_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "Results":
                    channel      = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if not alternatives:
                        continue

                    text = alternatives[0].get("transcript", "").strip()
                    if not text:
                        continue

                    is_final = data.get("is_final", False)
                    lang     = channel.get("detected_language", self._stream_lang)

                    from app.utils.lang_map import normalize_lang
                    lang = normalize_lang(lang)

                    await self._result_queue.put(TranscriptionResult(
                        text=text,
                        language=lang,
                        language_probability=1.0,
                        start_time=data.get("start", 0.0),
                        end_time=data.get("start", 0.0) + data.get("duration", 0.0),
                        segments=[],
                        is_partial=not is_final,
                    ))

                elif msg_type == "UtteranceEnd":
                    logger.debug("[STT WS] UtteranceEnd — sentence boundary")
                    await self._result_queue.put("utterance_end")

                elif msg_type == "Metadata":
                    logger.debug(f"[STT WS] Metadata: {data}")

                elif msg_type == "SpeechStarted":
                    logger.debug("[STT WS] SpeechStarted")

                elif msg_type == "Close":
                    logger.info("[STT WS] Close message received")
                    break

                elif msg_type == "Error":
                    logger.error(f"[STT WS] Server error: {data}")
                    break

        except ConnectionClosed:
            logger.info("[STT WS] Connection closed by server")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[STT WS] Receiver loop error: {e}", exc_info=True)
        finally:
            # FIX: Only put the sentinel if one isn't already queued.
            # close_stream() also puts None; a double-sentinel means the next
            # session's aiter_results() exits immediately without reading anything.
            try:
                # Check if queue is empty before adding sentinel to avoid duplicates.
                if self._result_queue.empty():
                    self._result_queue.put_nowait(None)
            except Exception:
                pass

    async def stream_tts(
        self,
        text: str,
        voice: str,
    ) -> AsyncGenerator[bytes, None]:
        if not text.strip():
            return

        url = (
            f"{DEEPGRAM_TTS_URL}"
            f"?model={voice}"
            f"&encoding=linear16"
            f"&sample_rate=24000"
            f"&container=none"
        )
        headers = {"Authorization": f"Token {self.api_key}"}

        logger.info(f"[TTS WS] Opening. voice={voice} text_len={len(text)}")

        try:
            async with websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=30,
                open_timeout=10,
            ) as tts_ws:

                await tts_ws.send(json.dumps({"type": "Speak", "text": text}))
                await tts_ws.send(json.dumps({"type": "Flush"}))

                async for message in tts_ws:
                    if isinstance(message, bytes):
                        if len(message) > 0:
                            yield message
                    elif isinstance(message, str):
                        try:
                            evt      = json.loads(message)
                            evt_type = evt.get("type", "")
                            if evt_type == "Flushed":
                                logger.debug("[TTS WS] Flushed")
                                break
                            elif evt_type == "Error":
                                logger.error(f"[TTS WS] Server error: {evt}")
                                break
                        except json.JSONDecodeError:
                            pass

        except ConnectionClosed:
            logger.info("[TTS WS] Connection closed")
        except Exception as e:
            logger.error(f"[TTS WS] Stream error: {e}", exc_info=True)

    async def atranscribe_segment(
        self,
        audio: np.ndarray,
        source_language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        if len(audio) == 0:
            return TranscriptionResult("", "en", 0.0, 0.0, 0.0, [], False)

        duration  = len(audio) / 16000
        pcm_bytes = (audio * 32768).clip(-32768, 32767).astype("int16").tobytes()

        logger.info(f"[HTTP STT] Sending {duration:.2f}s audio. model={self.model_id}")

        params = {
            "model":        self.model_id,
            "language":     source_language or self.language_code,
            "encoding":     "linear16",
            "sample_rate":  "16000",
            "channels":     "1",
            "punctuate":    "true",
            "smart_format": "true",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    DEEPGRAM_HTTP_URL,
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type":  "audio/raw",
                    },
                    params=params,
                    content=pcm_bytes,
                )

            if response.status_code != 200:
                logger.error(f"[HTTP STT] Error {response.status_code}: {response.text}")
                return TranscriptionResult("", source_language or "en", 0.0, 0.0, duration, [], False)

            data         = response.json()
            alternatives = (
                data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])
            )
            text          = alternatives[0].get("transcript", "").strip() if alternatives else ""
            detected_lang = (
                data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("detected_language", source_language or "en")
            )

            from app.utils.lang_map import normalize_lang
            detected_lang = normalize_lang(detected_lang)

        except Exception as e:
            logger.error(f"[HTTP STT] Request failed: {e}", exc_info=True)
            return TranscriptionResult("", source_language or "en", 0.0, 0.0, duration, [], False)

        return TranscriptionResult(
            text=text,
            language=detected_lang,
            language_probability=1.0,
            start_time=0.0,
            end_time=duration,
            segments=[],
            is_partial=False,
        )

    @property
    def model_info(self) -> dict:
        return {
            "model_size":   self.model_size,
            "device":       self.device,
            "compute_type": self.compute_type,
        }