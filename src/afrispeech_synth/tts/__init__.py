"""TTS backend registry.

Backends are looked up by name so `tts.backend: gemini` in a config file is all
it takes to swap one in. A new backend is a TTSBackend subclass registered here.
"""
from __future__ import annotations

from .base import Clip, RetryableTTSError, TTSBackend, TTSError, pcm_to_wav, wav_sample_rate

_BACKENDS = {}


def register(name: str, loader) -> None:
    _BACKENDS[name] = loader


def _load_gemini():
    from .gemini import GeminiTTS
    return GeminiTTS


def _load_gemini_live():
    from .gemini_live import GeminiLiveTTS
    return GeminiLiveTTS


register("gemini", _load_gemini)
register("gemini-live", _load_gemini_live)


def available() -> list:
    return sorted(_BACKENDS)


def get_backend(name: str, config, language) -> TTSBackend:
    if name not in _BACKENDS:
        raise TTSError(f"Unknown TTS backend {name!r}. Available: {', '.join(available())}")
    return _BACKENDS[name]()(config, language)


__all__ = ["Clip", "TTSBackend", "TTSError", "RetryableTTSError", "pcm_to_wav",
           "wav_sample_rate", "get_backend", "register", "available"]
