"""
Translator — Helsinki-NLP MarianMT translation engine.

Design:
  - ModelRegistry: singleton that lazy-loads one MarianMT model per language pair
  - Models are cached in memory after first load (LRU eviction at max_models)
  - translate() and atranslate() are sync/async API respectively
  - All translation runs in a thread pool (asyncio.to_thread) to avoid blocking
    the FastAPI event loop during HuggingFace inference

Why Helsinki-NLP / MarianMT?
  - Tiny models (50–300 MB per pair) vs 1.5GB+ for NLLB or M2M-100
  - Sub-100ms per sentence on CPU, ~20ms on GPU
  - 1000+ language pairs on HuggingFace Hub
  - No API key, fully offline after download
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class _MarianModel:
    """Holds a loaded tokeniser + model pair."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def load(self) -> None:
        from transformers import MarianMTModel, MarianTokenizer

        logger.info(f"Loading translation model: {self.model_name}")
        t0 = time.perf_counter()
        self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self._model = MarianMTModel.from_pretrained(self.model_name)
        self._model.eval()
        elapsed = time.perf_counter() - t0
        logger.info(f"Loaded {self.model_name} in {elapsed:.1f}s")
        self._loaded = True

    def translate(self, texts: list[str], max_length: int = 512) -> list[str]:
        """Synchronous batch translation. Call via asyncio.to_thread."""
        if not self._loaded:
            self.load()
        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )
        return [
            self._tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class ModelRegistry:
    """
    Singleton registry of loaded MarianMT models.
    Lazy-loads models on first request; evicts LRU when max_models exceeded.
    Thread-safe via asyncio.Lock.
    
    """

    def __init__(self, max_models: int = 10, max_length: int = 512):
        self.max_models = max_models
        self.max_length = max_length
        # OrderedDict used as LRU cache
        self._cache: OrderedDict[str, _MarianModel] = OrderedDict()
        self._lock = asyncio.Lock()

    async def _get_or_load(self, model_name: str) -> _MarianModel:
        async with self._lock:
            if model_name in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(model_name)
                return self._cache[model_name]

            # Evict LRU if at capacity
            if len(self._cache) >= self.max_models:
                evicted_name, _ = self._cache.popitem(last=False)
                logger.info(f"Evicted translation model: {evicted_name}")

            model = _MarianModel(model_name)
            self._cache[model_name] = model

        # Load outside the lock so other requests aren't blocked during download
        if not model.is_loaded:
            await asyncio.to_thread(model.load)

        return model

    async def translate(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """
        Translate a list of strings from source_lang to target_lang.
        Returns translated strings in the same order.
        Raises ValueError if the language pair is not supported.
        """
        from app.utils.lang_map import get_model_name, is_same_language

        # No-op if src == tgt
        if is_same_language(source_lang, target_lang):
            return texts

        model_name = get_model_name(source_lang, target_lang)
        if model_name is None:
            raise ValueError(
                f"Translation pair not supported: {source_lang} → {target_lang}. "
                "Check /languages for supported pairs."
            )

        # Filter empty strings
        filtered = [t for t in texts if t.strip()]
        if not filtered:
            return [""] * len(texts)

        model = await self._get_or_load(model_name)
        translated = await asyncio.to_thread(
            model.translate, filtered, self.max_length
        )

        # Rebuild original-length list (re-insert empty strings)
        result = []
        t_iter = iter(translated)
        for t in texts:
            result.append(next(t_iter) if t.strip() else "")
        return result

    async def translate_one(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Convenience wrapper for single-string translation."""
        results = await self.translate([text], source_lang, target_lang)
        return results[0]

    @property
    def loaded_pairs(self) -> list[str]:
        """Return list of currently cached model names."""
        return list(self._cache.keys())

    async def preload(self, pairs: list[tuple[str, str]]) -> None:
        """
        Pre-warm models for a list of (src, tgt) pairs.
        Call at startup for common pairs to avoid first-request latency.
        """
        from app.utils.lang_map import get_model_name
        for src, tgt in pairs:
            model_name = get_model_name(src, tgt)
            if model_name:
                logger.info(f"Pre-loading: {src}→{tgt} ({model_name})")
                await self._get_or_load(model_name)


# ─── Module-level singleton ───────────────────────────────────────────────────
# Instantiated here; injected into FastAPI via app.state in main.py

registry = ModelRegistry(max_models=10, max_length=512)