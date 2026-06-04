"""
Maps (source_lang, target_lang) ISO-639-1 pairs to Helsinki-NLP/opus-mt model IDs.
Helsinki models use 2- or 3-letter codes; some are multi-source (e.g. opus-mt-ROMANCE-en).
"""

# fmt: off
LANG_MODEL_MAP: dict[tuple[str, str], str] = {
    # English → *
    ("en", "hi"):  "Helsinki-NLP/opus-mt-en-hi",
    ("en", "fr"):  "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"):  "Helsinki-NLP/opus-mt-en-de",
    ("en", "es"):  "Helsinki-NLP/opus-mt-en-es",
    ("en", "ar"):  "Helsinki-NLP/opus-mt-en-ar",
    ("en", "zh"):  "Helsinki-NLP/opus-mt-en-zh",
    ("en", "ja"):  "Helsinki-NLP/opus-mt-en-jap",
    ("en", "ru"):  "Helsinki-NLP/opus-mt-en-ru",
    ("en", "pt"):  "Helsinki-NLP/opus-mt-en-ROMANCE",  # covers pt, fr, es, it
    ("en", "ko"):  "Helsinki-NLP/opus-mt-en-ko",
    ("en", "it"):  "Helsinki-NLP/opus-mt-en-it",
    ("en", "tr"):  "Helsinki-NLP/opus-mt-en-tr",
    ("en", "nl"):  "Helsinki-NLP/opus-mt-en-nl",
    ("en", "pl"):  "Helsinki-NLP/opus-mt-en-pl",
    ("en", "sv"):  "Helsinki-NLP/opus-mt-en-sv",
    ("en", "uk"):  "Helsinki-NLP/opus-mt-en-uk",

    # * → English
    ("hi", "en"):  "Helsinki-NLP/opus-mt-hi-en",
    ("fr", "en"):  "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"):  "Helsinki-NLP/opus-mt-de-en",
    ("es", "en"):  "Helsinki-NLP/opus-mt-es-en",
    ("ar", "en"):  "Helsinki-NLP/opus-mt-ar-en",
    ("zh", "en"):  "Helsinki-NLP/opus-mt-zh-en",
    ("ja", "en"):  "Helsinki-NLP/opus-mt-ja-en",
    ("ru", "en"):  "Helsinki-NLP/opus-mt-ru-en",
    ("pt", "en"):  "Helsinki-NLP/opus-mt-ROMANCE-en",
    ("ko", "en"):  "Helsinki-NLP/opus-mt-ko-en",
    ("it", "en"):  "Helsinki-NLP/opus-mt-it-en",
    ("tr", "en"):  "Helsinki-NLP/opus-mt-tr-en",
    ("nl", "en"):  "Helsinki-NLP/opus-mt-nl-en",
    ("pl", "en"):  "Helsinki-NLP/opus-mt-pl-en",
    ("sv", "en"):  "Helsinki-NLP/opus-mt-sv-en",
    ("uk", "en"):  "Helsinki-NLP/opus-mt-uk-en",


}
# fmt: on

SUPPORTED_TARGET_LANGUAGES: list[str] = sorted(
    {tgt for (_, tgt) in LANG_MODEL_MAP.keys()}
)


# ElevenLabs returns ISO-639-3 (3-letter) codes; MarianMT needs ISO-639-1 (2-letter)
_ISO639_3_TO_1: dict[str, str] = {
    "eng": "en", "hin": "hi", "fra": "fr", "deu": "de", "spa": "es",
    "ara": "ar", "zho": "zh", "jpn": "ja", "rus": "ru", "por": "pt",
    "kor": "ko", "ita": "it", "tur": "tr", "nld": "nl", "pol": "pl",
    "swe": "sv", "ukr": "uk",
}


def normalize_lang(code: str) -> str:
    """
    Convert ISO-639-3 (3-letter) to ISO-639-1 (2-letter).
    ElevenLabs returns 3-letter codes; pass through if already 2-letter.
    """
    code = code.lower().strip()
    return _ISO639_3_TO_1.get(code, code)


def get_model_name(source: str, target: str) -> str | None:
    """Return HuggingFace model ID or None if pair not supported."""
    return LANG_MODEL_MAP.get((normalize_lang(source), normalize_lang(target)))


def is_same_language(source: str, target: str) -> bool:
    return normalize_lang(source) == normalize_lang(target)