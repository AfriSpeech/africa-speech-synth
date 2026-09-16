"""Text normalisation through africa-g2p.

Two modes matter for TTS:

`grapheme` (default) rewrites the sentence into the language's own phoneme
units — multigraphs like `ny`, `kp`, `gb` kept whole. This is what the Twi
dataset was built with, and it trains better than IPA because the symbols stay
inside the language's own inventory.

`universal` rewrites the sentence in the grapheme set most African languages
share — `ɔ` becomes `o`, `ɛ` becomes `e`. Same idea as IPA (one inventory across
languages) but written in plain letters rather than phonetic symbols. This is
what the Ghana Twi synthetic dataset was built with.

`ipa` gives a cross-language phonetic inventory instead, which is the right
choice when one model is trained over several languages at once.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from .lang import Language, require_g2p

MODES = ("grapheme", "universal", "ipa", "none")


class Normaliser:
    """Wraps africa-g2p's AfricaPipeline, or passes text through unchanged."""

    def __init__(self, language: Language, mode: str = "grapheme", sep: str = ""):
        if mode not in MODES:
            raise ValueError(f"normalise must be one of {', '.join(MODES)}, got {mode!r}")
        self.language = language
        self.mode = mode
        self.sep = sep
        self._pipeline = None
        self._converter = None
        self._units_pipeline = None
        if mode == "universal":
            from africa_g2p import UNIVERSAL, GraphemeConverter
            self._converter = GraphemeConverter(require_g2p(language), UNIVERSAL)
        elif mode != "none":
            from africa_g2p import AfricaPipeline
            self._pipeline = AfricaPipeline(lang=require_g2p(language), output=mode)

    @property
    def enabled(self) -> bool:
        return self._pipeline is not None or self._converter is not None

    def __call__(self, text: str) -> str:
        if self._converter is not None:
            return self._converter.convert(text, sep=self.sep)
        if self._pipeline is None:
            return text
        return self._pipeline.run(text, sep=self.sep)

    def batch(self, texts: List[str]) -> List[str]:
        if not self.enabled:
            return list(texts)
        return [self(text) for text in texts]

    def units(self, text: str) -> List[str]:
        """Phoneme units of a sentence — the coverage unit for selection.

        Universal mode respells the same sounds rather than changing them, so
        coverage is measured on the language's own phoneme units either way.
        """
        if self._converter is not None:
            if self._units_pipeline is None:
                from africa_g2p import AfricaPipeline
                self._units_pipeline = AfricaPipeline(lang=self.language.g2p_code,
                                                      output="grapheme")
            return [u for u in self._units_pipeline.run(text, sep=" ").split(" ") if u]
        if self._pipeline is None:
            return text.lower().split()
        return [unit for unit in self._pipeline.run(text, sep=" ").split(" ") if unit]


def describe(language: Language, mode: str) -> Optional[str]:
    """One line for the dataset card."""
    if mode == "none":
        return None
    import africa_g2p
    version = getattr(africa_g2p, "__version__", "unknown")
    if mode == "universal":
        return (f"africa-g2p {version} — "
                f"GraphemeConverter({language.g2p_code!r}, UNIVERSAL)")
    return f"africa-g2p {version} — AfricaPipeline(lang={language.g2p_code!r}, output={mode!r})"
