from __future__ import annotations

import asyncio
import numpy as np
from collections import deque


class AudioBuffer:
    def __init__(
        self,
        sample_rate: int = 16000,
        max_seconds: int = 30,
    ):
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_seconds

        self._buffer: deque[np.ndarray] = deque()
        self._total_samples: int = 0
        self._lock = asyncio.Lock()

    async def push(self, chunk: np.ndarray) -> None:
        async with self._lock:
            self._buffer.append(chunk)
            self._total_samples += len(chunk)
            while self._total_samples > self.max_samples and self._buffer:
                evicted = self._buffer.popleft()
                self._total_samples -= len(evicted)

    async def get_all(self) -> np.ndarray:
        async with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buffer))

    async def export_and_clear(self) -> np.ndarray:
        async with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            full = np.concatenate(list(self._buffer))
            self._buffer.clear()
            self._total_samples = 0
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