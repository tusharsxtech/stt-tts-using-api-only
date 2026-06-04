"""
VAD (Voice Activity Detection) using Silero VAD v5.

For live call captioning the key tuning is:
  - min_speech_ms = 150ms  → start capturing quickly, don't miss word beginnings
  - min_silence_ms = 800ms → natural pause before closing ElevenLabs stream
                              (too short = cuts mid-sentence; too long = delays captions)
  - threshold = 0.40       → slightly below default for sensitivity to soft speech

The min_silence_ms here is the gate on the ElevenLabs API connection:
  - While speech is detected  → ElevenLabs WS stays OPEN (billing active)
  - After 800ms of silence    → SPEECH_END fires → ElevenLabs WS is CLOSED (billing stops)
  - User starts speaking again → SPEECH_START fires → new ElevenLabs WS opened

This is different from the session idle timeout (120s) in websocket.py which handles
complete inactivity — no audio from browser at all.
"""

import logging
import numpy as np
import torch
from collections import deque
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class VADEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END   = "speech_end"
    SILENCE      = "silence"
    SPEECH       = "speech"


@dataclass
class VADResult:
    event: VADEvent
    speech_prob: float
    smoothed_prob: float
    num_samples: int
    speech_duration_s: float


class SileroVAD:
    """
    Silero VAD v5 wrapper tuned for live call captioning.

    Silence → Speech:  min_speech_ms  (default 150ms) fires SPEECH_START
    Speech → Silence:  min_silence_ms (default 800ms) fires SPEECH_END

    800ms silence is the ElevenLabs connection gate. Set it via config:
      VAD_MIN_SILENCE_MS=800
    """

    SILERO_REPO = "snakers4/silero-vad"
    CHUNK_SIZE  = 512    # samples @ 16kHz = 32ms — Silero required window
    SMOOTH_WINDOW = 5    # rolling average for stability

    def __init__(
        self,
        threshold: float = 0.40,
        min_speech_ms: int = 150,
        min_silence_ms: int = 800,    # IMPORTANT: 800ms for live captions
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_samples  = int(min_speech_ms  * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)
        self.sample_rate = sample_rate

        self._model  = None
        self._loaded = False

        self._in_speech: bool     = False
        self._speech_samples: int = 0
        self._silence_samples: int = 0
        self._segment_samples: int = 0

        self._prob_history: deque = deque(maxlen=self.SMOOTH_WINDOW)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        logger.info("Loading Silero VAD model...")
        self._model, _ = torch.hub.load(
            repo_or_dir=self.SILERO_REPO,
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        self._model.eval()
        self._loaded = True
        logger.info(
            f"Silero VAD loaded. threshold={self.threshold} "
            f"min_speech={self.min_speech_samples}smp "
            f"min_silence={self.min_silence_samples}smp"
        )

    def is_loaded(self) -> bool:
        return self._loaded

    # ── State management ──────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Full reset — call when a new WebSocket session starts."""
        if self._model:
            self._model.reset_states()
        self._reset_counters()
        self._prob_history.clear()

    def reset_segment(self) -> None:
        """Reset only segment counters after SPEECH_END (preserve model state)."""
        self._speech_samples   = 0
        self._silence_samples  = 0
        self._segment_samples  = 0
        self._in_speech = False

    def _reset_counters(self) -> None:
        self._in_speech        = False
        self._speech_samples   = 0
        self._silence_samples  = 0
        self._segment_samples  = 0

    # ── Core inference ────────────────────────────────────────────────────────

    def process_chunk(self, audio: np.ndarray) -> VADResult:
        """
        Score a single 512-sample (32ms) audio chunk.
        Returns VADResult with event and probabilities.

        Key events:
          SPEECH_START → open ElevenLabs stream
          SPEECH_END   → close ElevenLabs stream (stops billing)
        """
        if not self._loaded:
            raise RuntimeError("Silero VAD not loaded. Call .load() first.")

        if len(audio) < self.CHUNK_SIZE:
            audio = np.pad(audio, (0, self.CHUNK_SIZE - len(audio)))
        audio = audio[:self.CHUNK_SIZE]

        tensor = torch.FloatTensor(audio).unsqueeze(0)
        with torch.no_grad():
            raw_prob = float(self._model(tensor, self.sample_rate).item())

        self._prob_history.append(raw_prob)
        smoothed = float(np.mean(self._prob_history))
        num_samples = self.CHUNK_SIZE
        segment_duration_s = self._segment_samples / self.sample_rate

        if smoothed >= self.threshold:
            # Speech frame
            self._silence_samples = 0
            self._speech_samples += num_samples
            if self._in_speech:
                self._segment_samples += num_samples
            if not self._in_speech and self._speech_samples >= self.min_speech_samples:
                self._in_speech = True
                self._segment_samples = self._speech_samples
                return VADResult(VADEvent.SPEECH_START, raw_prob, smoothed,
                                 num_samples, segment_duration_s)
            return VADResult(VADEvent.SPEECH, raw_prob, smoothed,
                             num_samples, segment_duration_s)
        else:
            # Silence frame
            self._speech_samples = 0
            self._silence_samples += num_samples
            if self._in_speech and self._silence_samples >= self.min_silence_samples:
                # Enough silence → fire SPEECH_END → ElevenLabs WS will close
                event = VADEvent.SPEECH_END
                self.reset_segment()
                return VADResult(event, raw_prob, smoothed,
                                 num_samples, segment_duration_s)
            return VADResult(VADEvent.SILENCE, raw_prob, smoothed,
                             num_samples, segment_duration_s)

    def process_window(self, audio: np.ndarray) -> tuple[list[VADResult], float]:
        """
        Process a longer window by striding through CHUNK_SIZE chunks.
        Returns list of VADResult and aggregate smoothed probability.
        """
        results = []
        for i in range(0, max(len(audio), self.CHUNK_SIZE), self.CHUNK_SIZE):
            chunk = audio[i: i + self.CHUNK_SIZE]
            results.append(self.process_chunk(chunk))

        aggregate_prob = float(np.mean([r.smoothed_prob for r in results]))
        return results, aggregate_prob

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def current_smoothed_prob(self) -> float:
        if not self._prob_history:
            return 0.0
        return float(np.mean(self._prob_history))

    @property
    def segment_duration_seconds(self) -> float:
        return self._segment_samples / self.sample_rate