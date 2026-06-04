"""
AudioBuffer — thread-safe ring buffer that accumulates incoming PCM(Pulse Code Modulator) chunks
and exposes complete speech-sized windows to the VAD/Whisper pipeline.

Design:
  - Client sends ~500ms PCM chunks over WebSocket
  - Buffer accumulates until VAD fires (speech detected)
  - On speech end, exports the speech segment (+ 200ms padding each side)
  - After export, keeps a short tail for context continuity
"""

import asyncio
import numpy as np
from collections import deque
from typing import Optional


class AudioBuffer:
    def __init__(
        self,
        sample_rate: int = 16000,
        max_seconds: int = 30,
        context_tail_seconds: float = 0.2,
    ):
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_seconds
        self.tail_samples = int(sample_rate * context_tail_seconds)

        self._buffer: deque[np.ndarray] = deque()
        self._total_samples: int = 0
        self._lock = asyncio.Lock()

    async def push(self, chunk: np.ndarray) -> None:
        """Append a new audio chunk. Evicts oldest samples if buffer is full."""
        async with self._lock:
            self._buffer.append(chunk)
            self._total_samples += len(chunk)
            # Evict oldest chunks if over capacity
            while self._total_samples > self.max_samples and self._buffer:
                evicted = self._buffer.popleft()
                self._total_samples -= len(evicted)

    async def get_all(self) -> np.ndarray:
        """Return full buffer as a contiguous float32 array (no mutation)."""
        async with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buffer))

    async def export_and_trim(self) -> np.ndarray:
        """
        Export all buffered audio for transcription, then keep only the
        context tail so Whisper has a tiny overlap for continuity.
        """
        async with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            full = np.concatenate(list(self._buffer))
            # Keep tail
            tail = full[-self.tail_samples :] if len(full) > self.tail_samples else full
            self._buffer.clear()
            self._total_samples = 0
            if len(tail):
                self._buffer.append(tail)
                self._total_samples = len(tail)
            return full

    async def clear(self) -> None:
        async with self._lock:
            self._buffer.clear()
            self._total_samples = 0

    @property
    def duration_seconds(self) -> float:
        return self._total_samples / self.sample_rate

    @property
    def is_empty(self) -> bool:
        return self._total_samples == 0