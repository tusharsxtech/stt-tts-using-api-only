# Audio Translate Service

Real-time audio transcription and translation microservice.

```
Client Audio (any format)
        │
        ▼
FastAPI  /ws/stream/{target_lang}
        │
        ▼
AudioBuffer  (accumulates 500ms chunks)
        │
        ▼
Silero VAD  (speech / silence detection)
        │  fires on speech end
        ▼
ElevenLabs Scribe v2 Realtime  (STT — cloud API, ~150ms)
        │
        ▼
Lang Mapping  (ISO-639-1 pair → Helsinki model)
        │
        ▼
MarianMT / Helsinki-NLP  (translation — local CPU/GPU)
        │
        ▼
WebSocket emit  →  Client (partial + final captions)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | `python --version` to check |
| ffmpeg | Required for MP3 / OGG / WebM decoding |
| ElevenLabs API key | Get one at https://elevenlabs.io — free tier works |

Install ffmpeg:

# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install -y ffmpeg


---

## Setup

**1. Clone and enter the project**

cd translation_pipeline


**2. Create virtual environment**

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate


**3. Install PyTorch (CPU build — needed for Silero VAD)**

pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu


> If you have an NVIDIA GPU, replace cpu with cu121 for faster MarianMT inference.

**4. Install dependencies**

uv pip install -r requirements.txt


**5. Configure environment**

# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env


Open .env and set your ElevenLabs API key:

ELEVENLABS_API_KEY=sk_your_actual_key_here
ELEVENLABS_MODEL_ID=scribe_v2_realtime

VAD_THRESHOLD=0.5
VAD_MIN_SPEECH_MS=250
VAD_MIN_SILENCE_MS=200

AUDIO_SAMPLE_RATE=16000
CHUNK_DURATION_MS=100

MAX_WS_CONNECTIONS=50


**6. Pre-download MarianMT translation models**

MarianMT models are downloaded from HuggingFace on first use (~50–300 MB each).
Pre-download the ones you need so there is no delay on first request:


python scripts/download_models.py --langs en-hi en-fr en-de en-es


> No whisper flag needed anymore — STT is handled by ElevenLabs cloud API.

**7. Start the server**

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


---

## URLs

| URL                                   |                                             What you see |
|---|---|
| http://localhost:8000                 |                                        Service info JSON |
| http://localhost:8000/docs            |                          Swagger UI — test all endpoints |
| http://localhost:8000/health          |       ElevenLabs model status + loaded translation pairs |
| http://localhost:8000/languages       |                               Supported target languages |
| http://localhost:8000/languages/pairs |                             All supported language pairs |

---

## Testing

### REST — Single file via Swagger

1. Go to http://localhost:8000/docs
2. Click POST /pipeline` → **Try it out**
3. Upload any audio file (WAV, MP3, OGG, WebM, FLAC, M4A)
4. Set target_language to hi (or fr, de, es, etc.)
5. Click **Execute**

### WebSocket — Live streaming

Connect to:

ws://localhost:8000/ws/stream/{target_lang}?source_lang=en&encoding=pcm_s16le


Send raw PCM audio chunks (binary frames). You will receive JSON back:


// Partial — emitted every ~400ms while speaking (update caption live)
{
  "original_text": "Hello how are",
  "translated_text": "नमस्ते आप कैसे",
  "source_language": "en",
  "target_language": "hi",
  "is_partial": true
}

// Final — emitted when VAD detects end of speech (replace partial with this)
{
  "original_text": "Hello how are you doing today",
  "translated_text": "नमस्ते आप आज कैसे हैं",
  "source_language": "en",
  "target_language": "hi",
  "is_partial": false
}


Send STOP (text frame) to flush and close the session cleanly.

### Silence timeout

If no speech is detected for **2 minutes**, the ElevenLabs connection is automatically
closed and the client receives:

{ "event": "silence_timeout", "message": "Connection closed after 2 minutes of silence." }


---

## Supported Languages

| Code | Language |
|---|---|
| `hi` | Hindi |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `ar` | Arabic |
| `zh` | Chinese |
| `ja` | Japanese |
| `ru` | Russian |
| `pt` | Portuguese |
| `ko` | Korean |
| `it` | Italian |
| `tr` | Turkish |
| `nl` | Dutch |
| `pl` | Polish |
| `sv` | Swedish |
| `uk` | Ukrainian |

Full list at /languages/pairs.

---

## Expected Latency

| Condition | Lag |
|---|---|
| ElevenLabs STT (cloud) | ~150–400ms |
| MarianMT on CPU | ~300–600ms |
| MarianMT on GPU | ~20–80ms |
| VAD silence wait | ~200ms |
| **Total (CPU)** | **~800ms – 1.5s** |
| **Total (GPU)** | **~400ms – 700ms** |

For Google Meet / YouTube level captions (~500ms), swap MarianMT for a translation API (DeepL / Google Translate).

---

## What Changed from Original

| Before                                |                           After                       |
|---                                    |                                                    ---|
| faster-whisper (local Whisper model)  |             ElevenLabs Scribe v2 Realtime (cloud API) |
| Whisper VAD filter                    |             Silero VAD (external, chunk-level gating) |
| `--whisper` flag in download script   |                   Removed — no local STT model needed |
| `WHISPER_MODEL_SIZE`, `WHISPER_DEVICE` in `.env` |                                    Removed |
| Silence closes nothing                |    2-min silence closes ElevenLabs WS to stop billing |