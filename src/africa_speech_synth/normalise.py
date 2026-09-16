"""Text normalisation through africa-g2p.

Two modes matter for TTS:

`grapheme` (default) rewrites the sentence into the language's own phoneme
units — multigraphs like `ny`, `kp`, `gb` kept whole. This is what the Twi
dataset was built with, and it trains better than IPA because the symbols stay
inside the language's own inventory.

`ipa` gives a cross-language inventory instead, which is the right choice when
one model is trained over several languages at once.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from .lang import Language, require_g2p

MODES = ("grapheme", "ipa", "none")


class Normaliser:
    """Wraps africa-g2p's AfricaPipeline, or passes text through unchanged."""

    def __init__(self, language: Language, mode: str = "grapheme", sep: str = ""):
        if mode not in MODES:
            raise ValueError(f"normalise must be one of {', '.join(MODES)}, got {mode!r}")
        self.language = language
        self.mode = mode
        self.sep = sep
        self._pipeline = None
        if mode != "none":
            from africa_g2p import AfricaPipeline
            self._pipeline = AfricaPipeline(lang=require_g2p(language), output=mode)

    @property
    def enabled(self) -> bool:
        return self._pipeline is not None

    def __call__(self, text: str) -> str:
        if self._pipeline is None:
            return text
        return self._pipeline.run(text, sep=self.sep)

    def batch(self, texts: List[str]) -> List[str]:
        if self._pipeline is None:
            return list(texts)
        return [self(text) for text in texts]

    def units(self, text: str) -> List[str]:
        """Phoneme units of a sentence — the coverage unit for selection."""
        if self._pipeline is None:
            return text.lower().split()
        return [unit for unit in self._pipeline.run(text, sep=" ").split(" ") if unit]


def describe(language: Language, mode: str) -> Optional[str]:
    """One line for the dataset card."""
    if mode == "none":
        return None
    import africa_g2p
    version = getattr(africa_g2p, "__version__", "unknown")
    return f"africa-g2p {version} — AfricaPipeline(lang={language.g2p_code!r}, output={mode!r})"
