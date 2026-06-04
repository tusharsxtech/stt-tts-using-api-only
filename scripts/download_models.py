"""
scripts/download_models.py

Pre-downloads all required models so the service starts instantly in production.

    python scripts/download_models.py [--langs en-hi en-fr] [--skip-vad] [--skip-translation] [--skip-tts]

Downloads:
  1. Silero VAD          (torch hub)
  2. MarianMT pairs      (HuggingFace transformers)
#   3. Kokoro TTS          (kokoro-v1.0.onnx + voices-v1.0.bin via wget/curl)
"""

import argparse
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_PAIRS = [
     ("en", "fr"),  ("en", "es"),
     
    ("fr", "en"), ("es", "en"),
]

# [
#     ("en", "hi"), ("en", "fr"), ("en", "de"), ("en", "es"),
#     ("en", "ar"), ("en", "zh"), ("en", "ru"), ("en", "ja"),
#     ("hi", "en"), ("fr", "en"), ("de", "en"), ("es", "en"),
# ]


# KOKORO_ONNX_URL   = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
# KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
# KOKORO_ONNX_FILE  = "kokoro-v1.0.onnx"
# KOKORO_VOICES_FILE = "voices-v1.0.bin"


# ─── Silero VAD ───────────────────────────────────────────────────────────────

def download_silero_vad() -> None:
    logger.info("Downloading Silero VAD...")
    t0 = time.perf_counter()
    import torch
    model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        force_reload=False, onnx=False, verbose=False,
    )
    del model
    logger.info(f"Silero VAD ready in {time.perf_counter() - t0:.1f}s")


# ─── MarianMT translation models ─────────────────────────────────────────────

def download_translation_model(src: str, tgt: str) -> None:
    sys.path.insert(0, ".")
    from app.utils.lang_map import get_model_name
    model_name = get_model_name(src, tgt)
    if not model_name:
        logger.warning(f"No MarianMT model for {src}→{tgt}. Skipping.")
        return
    logger.info(f"Downloading translation model: {model_name}")
    t0 = time.perf_counter()
    from transformers import MarianMTModel, MarianTokenizer
    MarianTokenizer.from_pretrained(model_name)
    MarianMTModel.from_pretrained(model_name)
    logger.info(f"{model_name} ready in {time.perf_counter() - t0:.1f}s")


# # ─── Kokoro TTS model files ───────────────────────────────────────────────────

# def _download_file(url: str, dest: str) -> None:
#     if os.path.exists(dest):
#         logger.info(f"Already exists: {dest} — skipping download.")
#         return

#     logger.info(f"Downloading {dest} ...")
#     t0 = time.perf_counter()

#     # Try wget first, then curl, then Python httpx/urllib
#     for cmd in [
#         ["wget", "-q", "--show-progress", "-O", dest, url],
#         ["curl", "-L", "--progress-bar", "-o", dest, url],
#     ]:
#         try:
#             result = subprocess.run(cmd, check=True)
#             logger.info(f"{dest} downloaded in {time.perf_counter() - t0:.1f}s")
#             return
#         except (subprocess.CalledProcessError, FileNotFoundError):
#             continue

#     # Pure-Python fallback
#     try:
#         import httpx
#         with httpx.stream("GET", url, follow_redirects=True) as r:
#             r.raise_for_status()
#             with open(dest, "wb") as f:
#                 for chunk in r.iter_bytes(chunk_size=1 << 20):
#                     f.write(chunk)
#         logger.info(f"{dest} downloaded in {time.perf_counter() - t0:.1f}s")
#         return
#     except Exception as e:
#         logger.error(f"httpx fallback failed: {e}")

#     # urllib last resort
#     import urllib.request
#     urllib.request.urlretrieve(url, dest)
#     logger.info(f"{dest} downloaded via urllib in {time.perf_counter() - t0:.1f}s")


# def download_kokoro_tts(dest_dir: str = ".") -> None:
#     onnx_path   = os.path.join(dest_dir, KOKORO_ONNX_FILE)
#     voices_path = os.path.join(dest_dir, KOKORO_VOICES_FILE)

#     logger.info("=== Downloading Kokoro TTS model files ===")
#     _download_file(KOKORO_ONNX_URL,   onnx_path)
#     _download_file(KOKORO_VOICES_URL, voices_path)

#     # Smoke-test: try to import kokoro_onnx and instantiate
#     try:
#         from kokoro_onnx import Kokoro
#         logger.info("Verifying Kokoro model loads correctly...")
#         t0 = time.perf_counter()
#         k = Kokoro(onnx_path, voices_path)
#         # Quick synthesis to warm ONNX runtime
#         samples, sr = k.create("Hello.", voice="am_echo", speed=1.0, lang="en-us")
#         logger.info(
#             f"Kokoro smoke-test OK — {len(samples)} samples @ {sr}Hz "
#             f"in {time.perf_counter() - t0:.2f}s"
#         )
#         del k
#     except ImportError:
#         logger.warning(
#             "kokoro-onnx not installed yet. "
#             "Run: pip install kokoro-onnx>=0.4.0 onnxruntime>=1.18.0"
#         )
#     except Exception as e:
#         logger.error(f"Kokoro smoke-test failed: {e}", exc_info=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download all service models")
    parser.add_argument(
        "--langs", nargs="*", default=None, metavar="SRC-TGT",
        help="MarianMT pairs to download, e.g. en-hi en-fr. Default: 12 common pairs.",
    )
    parser.add_argument("--skip-vad",         action="store_true", help="Skip Silero VAD download")
    parser.add_argument("--skip-translation", action="store_true", help="Skip MarianMT download")
    # parser.add_argument("--skip-tts",         action="store_true", help="Skip Kokoro TTS download")
    # parser.add_argument(
    #     "--tts-dir", default=".", metavar="DIR",
    #     help="Directory to save kokoro-v1.0.onnx and voices-v1.0.bin (default: project root)",
    # )
    args = parser.parse_args()

    logger.info("=== Model pre-download starting ===")
    total_start = time.perf_counter()

    if not args.skip_vad:
        download_silero_vad()

    if not args.skip_translation:
        pairs = DEFAULT_PAIRS
        if args.langs:
            pairs = []
            for pair_str in args.langs:
                parts = pair_str.split("-")
                if len(parts) != 2:
                    logger.warning(f"Invalid pair format '{pair_str}'. Use SRC-TGT e.g. en-hi")
                    continue
                pairs.append((parts[0], parts[1]))
        for src, tgt in pairs:
            download_translation_model(src, tgt)

    # if not args.skip_tts:
    #     download_kokoro_tts(dest_dir=args.tts_dir)

    logger.info(
        f"=== All models downloaded in {time.perf_counter() - total_start:.1f}s ==="
    )


if __name__ == "__main__":
    main()