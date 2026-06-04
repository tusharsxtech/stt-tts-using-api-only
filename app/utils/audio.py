"""
 Audio utility helpers — universal format handling.

 Supported input formats:
  - Raw PCM (s16le)  — WebRTC / browser MediaRecorder raw
  - WAV              — standard .wav files
  - MP3              — .mp3 files
  - OGG/Opus         — .ogg or .opus (Firefox/Chrome MediaRecorder default)
  - WebM/Opus        — .webm (Chrome MediaRecorder default)
  - FLAC             — lossless
  - M4A / AAC        — mobile recordings

 Detection strategy:
  1. Inspect magic bytes header to identify format
  2. Route to the appropriate decoder
  3. Normalise → float32 mono @ target_sample_rate (16kHz for Whisper)

 Dependencies:
  - pydub (wraps ffmpeg) — handles mp3, ogg, webm, m4a
  - soundfile            — handles wav, flac (no ffmpeg needed)
  - torchaudio           — high-quality resampling
  - numpy
  - ffmpeg binary must be installed (apt-get install ffmpeg)
"""

import io
import logging
import numpy as np
import soundfile as sf
from enum import Enum
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ─── Format Enum ─────────────────────────────────────────────────────────────

class AudioFormat(str, Enum):
    PCM_RAW   = "pcm_raw"      # headerless raw s16le (WebRTC default)
    WAV       = "wav"
    MP3       = "mp3"
    OGG_OPUS  = "ogg_opus"     # .ogg container with Opus codec
    WEBM_OPUS = "webm_opus"    # .webm container — Chrome MediaRecorder default
    FLAC      = "flac"
    M4A_AAC   = "m4a_aac"      # mobile recordings
    UNKNOWN   = "unknown"


# ─── Magic byte signatures ───────────────────────────────────────────────────

# (magic_bytes, byte_offset, AudioFormat)
_MAGIC_SIGNATURES: list[tuple[bytes, int, AudioFormat]] = [
    (b"RIFF",               0, AudioFormat.WAV),
    (b"ID3",                0, AudioFormat.MP3),
    (b"\xff\xfb",           0, AudioFormat.MP3),    # MP3 sync word
    (b"\xff\xf3",           0, AudioFormat.MP3),
    (b"\xff\xf2",           0, AudioFormat.MP3),
    (b"\xff\xe0",           0, AudioFormat.MP3),    # MPEG-1 Layer 3
    (b"OggS",               0, AudioFormat.OGG_OPUS),
    (b"\x1a\x45\xdf\xa3",  0, AudioFormat.WEBM_OPUS),  # EBML header (WebM/MKV)
    (b"fLaC",               0, AudioFormat.FLAC),
]


def detect_format(data: bytes) -> AudioFormat:
    """
    Inspect magic bytes to identify the audio container format.
    Falls back to PCM_RAW if nothing matches (headerless WebRTC stream).
    """
    if len(data) < 4:
        return AudioFormat.PCM_RAW

    for magic, offset, fmt in _MAGIC_SIGNATURES:
        end = offset + len(magic)
        if len(data) >= end and data[offset:end] == magic:
            return fmt

    # M4A / MP4 ftyp box: 4 bytes size, then b"ftyp"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return AudioFormat.M4A_AAC

    # OGG: "OpusHead" may appear slightly after OggS in packet data
    if b"OpusHead" in data[:128] or b"opus" in data[:128].lower():
        return AudioFormat.OGG_OPUS

    # WebM: DocType element marker
    if b"webm" in data[:64].lower() or b"\x42\x82" in data[:32]:
        return AudioFormat.WEBM_OPUS

    # Default → raw PCM
    return AudioFormat.PCM_RAW


# ─── Main entry point ────────────────────────────────────────────────────────

def decode_audio(
    data: bytes,
    target_sr: int = 16000,
    fmt: Optional[AudioFormat] = None,
    pcm_input_sr: int = 16000,   # declared by client when sending raw PCM
    pcm_channels: int = 1,
) -> np.ndarray:
    """
    Universal decoder. Accepts any supported audio format as bytes.
    Returns a float32 mono numpy array resampled to target_sr.

    Args:
        data:          Raw bytes from WebSocket frame or file upload
        target_sr:     Output sample rate (16000 for Whisper)
        fmt:           Force a specific format; None = auto-detect
        pcm_input_sr:  Input sample rate when format is PCM_RAW
        pcm_channels:  Number of channels when format is PCM_RAW
    """
    if not data:
        return np.array([], dtype=np.float32)

    detected = fmt or detect_format(data)
    logger.debug(f"Format detected: {detected.value} | size={len(data)} bytes")

    # ── Decode ────────────────────────────────────────────────────────────────
    try:
        if detected == AudioFormat.PCM_RAW:
            audio, src_sr = _decode_pcm_raw(data, pcm_input_sr, pcm_channels)

        elif detected in (AudioFormat.WAV, AudioFormat.FLAC):
            audio, src_sr = _decode_soundfile(data)

        elif detected in (
            AudioFormat.MP3, AudioFormat.OGG_OPUS,
            AudioFormat.WEBM_OPUS, AudioFormat.M4A_AAC
        ):
            audio, src_sr = _decode_pydub(data, detected)

        else:
            audio, src_sr = _decode_fallback(data, pcm_input_sr)

    except Exception as exc:
        logger.warning(
            f"Primary decode failed for {detected.value}: {exc}. Trying fallback."
        )
        audio, src_sr = _decode_fallback(data, pcm_input_sr)

    if len(audio) == 0:
        return audio

    # ── Downmix multichannel → mono ──────────────────────────────────────────
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # ── Resample → target_sr ─────────────────────────────────────────────────
    audio = _resample(audio, src_sr, target_sr)

    return audio.astype(np.float32)


# ─── Format-specific decoders ────────────────────────────────────────────────

def _decode_pcm_raw(
    data: bytes,
    src_sr: int = 16000,
    channels: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Headerless signed 16-bit little-endian PCM.
    Used by WebRTC (browser getUserMedia → pcm stream).
    """
    if len(data) % 2 != 0:
        data = data[:-1]  # Drop incomplete final byte

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    # Deinterleave stereo if declared
    if channels == 2 and len(audio) % 2 == 0:
        audio = audio.reshape(-1, 2).mean(axis=1)

    return audio, src_sr


def _decode_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    """
    Decode WAV or FLAC using soundfile (no ffmpeg dependency).
    soundfile uses libsndfile which is extremely fast.
    """
    buf = io.BytesIO(data)
    with sf.SoundFile(buf) as f:
        sr = f.samplerate
        audio = f.read(dtype="float32", always_2d=False)
    return audio, sr


def _decode_pydub(
    data: bytes,
    fmt: AudioFormat,
) -> tuple[np.ndarray, int]:
    """
    Decode MP3, OGG/Opus, WebM/Opus, M4A via pydub + ffmpeg.
    pydub is a thin ffmpeg wrapper that handles virtually any format.
    ffmpeg must be installed on the system.
    """
    from pydub import AudioSegment

    _pydub_fmt_map = {
        AudioFormat.MP3:       "mp3",
        AudioFormat.OGG_OPUS:  "ogg",
        AudioFormat.WEBM_OPUS: "webm",
        AudioFormat.M4A_AAC:   "mp4",
    }
    pydub_fmt = _pydub_fmt_map.get(fmt)
    buf = io.BytesIO(data)

    try:
        seg = AudioSegment.from_file(buf, format=pydub_fmt)
    except Exception as e:
        logger.warning(f"pydub explicit format={pydub_fmt} failed: {e}. Auto-detecting.")
        buf.seek(0)
        seg = AudioSegment.from_file(buf)  # let ffmpeg auto-detect

    src_sr = seg.frame_rate
    # Keep original sample rate; resample later in decode_audio()
    samples = np.array(seg.get_array_of_samples())

    # Handle stereo
    if seg.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    audio = samples.astype(np.float32)
    # Normalise based on sample width
    max_val = float(2 ** (8 * seg.sample_width - 1))
    audio /= max_val

    return audio, src_sr


def _decode_fallback(
    data: bytes,
    pcm_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Last-resort chain: soundfile → pydub auto-detect → raw PCM.
    Never raises; always returns something.
    """
    # 1) soundfile
    try:
        return _decode_soundfile(data)
    except Exception:
        pass

    # 2) pydub full auto-detect
    try:
        from pydub import AudioSegment
        buf = io.BytesIO(data)
        seg = AudioSegment.from_file(buf)
        src_sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples())
        if seg.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        max_val = float(2 ** (8 * seg.sample_width - 1))
        return samples.astype(np.float32) / max_val, src_sr
    except Exception:
        pass

    # 3) Treat as raw PCM s16le
    logger.warning("All decoders failed. Treating data as raw PCM s16le.")
    audio, sr = _decode_pcm_raw(data, pcm_sr)
    return audio, sr


# ─── Resampling ──────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """
    Resample audio array from src_sr to tgt_sr.
    Uses torchaudio (sinc interpolation, best quality) if available,
    else falls back to numpy linear interpolation.
    """
    if src_sr == tgt_sr:
        return audio

    # torchaudio: high-quality sinc resampling
    try:
        import torch
        import torchaudio
        tensor = torch.from_numpy(audio).unsqueeze(0)      # [1, T]
        resampled = torchaudio.functional.resample(tensor, src_sr, tgt_sr)
        return resampled.squeeze(0).numpy().astype(np.float32)
    except Exception:
        pass

    # numpy linear interpolation fallback
    target_len = int(round(len(audio) * tgt_sr / src_sr))
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ─── Utility helpers ─────────────────────────────────────────────────────────

def float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 array back to raw PCM s16le bytes."""
    return (audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()


def chunk_audio(audio: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    """Split audio into fixed-size chunks. Last chunk may be shorter."""
    return [
        audio[i : i + chunk_samples]
        for i in range(0, len(audio), chunk_samples)
    ]


def is_silent(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """RMS-based silence check. Fast pre-VAD gate, no model needed."""
    if len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms < threshold


def get_duration_seconds(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """Duration of an audio array in seconds."""
    return len(audio) / sample_rate


def audio_info(data: bytes) -> dict:
    """
    Return metadata about audio bytes without fully decoding.
    Useful for logging and debug endpoints.
    """
    fmt = detect_format(data)
    info = {"format": fmt.value, "size_bytes": len(data)}
    if fmt in (AudioFormat.WAV, AudioFormat.FLAC):
        try:
            buf = io.BytesIO(data)
            with sf.SoundFile(buf) as f:
                info["sample_rate"] = f.samplerate
                info["channels"] = f.channels
                info["frames"] = f.frames
                info["duration_s"] = round(f.frames / f.samplerate, 3)
        except Exception:
            pass
    return info