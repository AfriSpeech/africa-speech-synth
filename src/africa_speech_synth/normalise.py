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

import unicodedata
from typing import List, Optional

from .lang import Language, require_g2p

MODES = ("universal", "grapheme", "ipa", "none")

# Punctuation a speech model uses for phrasing. Everything else is stripped before
# synthesis: source corpora carry apostrophes, asterisks marking proper nouns,
# hyphens, colons and quotes, and a synthesiser reads each of them as a pause or
# spells them out. Anyin "Ɛ 'nwun ... *Abalahamʋn" was being spoken with breaks
# that are not in the language.
#
# Marks are removed rather than replaced by a space, so a word is never split in
# two; a stripped mark between letters leaves the letters adjacent.
KEPT_PUNCTUATION = ".?!,"

# Printed once when a mode other than the default is chosen, so the trade-off is
# visible at the point it is made rather than only in the README.
MODE_WARNINGS = {
    "grapheme": ("grapheme keeps this language's own characters (ɔ, ɛ, ŋ, stacked tone "
                 "diacritics). TTS voices often mispronounce or skip those — compare a "
                 "sample against the universal default before a full run."),
    "ipa": ("ipa sends phonetic symbols. Most TTS models were never trained to read them; "
            "use this for phoneme-level ASR work, not for speech generation."),
    "none": "none sends the raw text, including any punctuation and markup the source had.",
}


def strip_punctuation(text: str, keep: str = KEPT_PUNCTUATION) -> str:
    """Drop every punctuation and symbol character except `keep`.

    Unicode categories decide, not a character list, so this covers the curly
    quotes, dashes and brackets that vary by corpus. Letters and combining marks
    are untouched, which matters for IPA output: its modifier letters (ʰ, ʼ, ʷ)
    are letters, not punctuation, and must survive.
    """
    out = []
    for char in text:
        if char in keep or not unicodedata.category(char)[0] in ("P", "S"):
            out.append(char)
    # Stripping can leave doubled spaces or a space before a comma.
    cleaned = " ".join("".join(out).split())
    for mark in keep:
        cleaned = cleaned.replace(f" {mark}", mark)
    return cleaned


class Normaliser:
    """Wraps africa-g2p's AfricaPipeline, or passes text through unchanged."""

    def __init__(self, language: Language, mode: str = "universal", sep: str = "",
                 punctuation: str = KEPT_PUNCTUATION):
        if mode not in MODES:
            raise ValueError(f"normalise must be one of {', '.join(MODES)}, got {mode!r}")
        self.language = language
        self.mode = mode
        self.sep = sep
        self.punctuation = punctuation
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
            converted = self._converter.convert(text, sep=self.sep)
        elif self._pipeline is None:
            converted = text
        else:
            converted = self._pipeline.run(text, sep=self.sep)
        return strip_punctuation(converted, self.punctuation)

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
