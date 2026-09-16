"""Which languages actually work, and why.

Three different numbers matter to someone deciding whether to use this tool:

  400  languages africa-g2p can phonemise
  693  languages africa-corpus-builder has text for
  215  languages in both — these run with no input from you at all

The 185 with a G2P table but no corpus text still work; you just have to supply
the text yourself with a `file:` or `hf:` source. The 478 with text but no G2P
table work too, with `normalise: none` — the TTS model gets the raw orthography
instead of phoneme units.

Nothing here matches languages by name across the two registries. Doing so looks
tempting (it "recovers" 73 more) but pairs genuinely different languages that
share an alternative name — Tunisian Arabic text under an Algerian Arabic table,
Basa of Cameroon under Basa of Nigeria. A sample mislabelled that way is worse
than a missing one, so only exact code matches count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

_CACHE: Optional["Coverage"] = None


@dataclass
class Entry:
    code: str
    name: str
    family: str = ""
    region: str = ""
    has_g2p: bool = False
    has_text: bool = False

    @property
    def ready(self) -> bool:
        """Runs end to end with nothing but a language name."""
        return self.has_g2p and self.has_text

    @property
    def status(self) -> str:
        if self.ready:
            return "ready"
        if self.has_g2p:
            return "bring text"
        return "no g2p"


@dataclass
class Coverage:
    entries: Dict[str, Entry]

    def ready(self) -> List[Entry]:
        return [e for e in self.sorted() if e.ready]

    def sorted(self) -> List[Entry]:
        return sorted(self.entries.values(), key=lambda e: (e.name.lower(), e.code))

    def search(self, needle: str) -> List[Entry]:
        needle = needle.lower()
        return [e for e in self.sorted()
                if needle in e.code.lower() or needle in e.name.lower()]

    def counts(self) -> Dict[str, int]:
        values = self.entries.values()
        return {
            "g2p": sum(1 for e in values if e.has_g2p),
            "text": sum(1 for e in values if e.has_text),
            "ready": sum(1 for e in values if e.ready),
            "total": len(self.entries),
        }


def _corpus_languages() -> Dict[str, str]:
    """code -> name, for every language africa-corpus-builder can serve."""
    try:
        from .sources import _import_africa_corpus
        african, _ = _import_africa_corpus().list_languages()
        return {language.code: language.name for language in african}
    except Exception:
        # africa-corpus-builder is optional; without it every language simply
        # reports "bring text", which is the truth for that installation.
        return {}


def load(refresh: bool = False) -> Coverage:
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE

    from africa_g2p import registry

    entries: Dict[str, Entry] = {}
    for code, record in registry().items():
        regions = record.get("regions") or []
        entries[code] = Entry(
            code=code,
            name=str(record.get("name") or code),
            family=str(record.get("family") or ""),
            region=regions[0] if regions else "",
            has_g2p=True,
        )

    for code, name in _corpus_languages().items():
        if code in entries:
            entries[code].has_text = True
        else:
            entries[code] = Entry(code=code, name=name or code, has_text=True)

    _CACHE = Coverage(entries)
    return _CACHE


def ready_codes() -> List[str]:
    return [entry.code for entry in load().ready()]
