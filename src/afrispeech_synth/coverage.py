"""Which languages actually work, and why.

Three different numbers matter to someone deciding whether to use this tool:

  408  languages africa-g2p converts to universal orthography
  693  languages africa-corpus-builder has text for
  223  languages in both — these run with no input from you at all

Those counts are measured, not read off the registry: africa-g2p lists 400 keys,
but eight more corpus languages — Akan, Luo, Luwo, Mwan, Ngemba, Kamba, Malgache
and Tonga — reach a table through an alias. Matching on keys alone dropped them.

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
    # True when the language converts to universal orthography — by its own
    # registry entry or through an alias. Not merely "has a registry key".
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

    from africa_g2p import available_languages, registry

    # `registry()` is the metadata table and `available_languages()` is what
    # africa-g2p can actually convert — they are not the same size. Tables have
    # been added faster than metadata rows, so reading the registry alone hid
    # every language that had a table but no description of itself.
    meta = registry()
    entries: Dict[str, Entry] = {}
    for code in available_languages():
        record = meta.get(code) or {}
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
            # A table can exist without a metadata row, leaving the entry named
            # after its own code. The corpus knows the language's name, so a
            # gallery shows "Dagbani" rather than "dag".
            if name and entries[code].name == code:
                entries[code].name = name
        else:
            entries[code] = Entry(code=code, name=name or code, has_text=True)

    # Registry keys are not the whole story: a corpus code can reach a table
    # through an alias — `aka` (Akan), `luo`, `plt` (Malagasy) and five others
    # all convert to universal without appearing as keys themselves. Asking the
    # normaliser is the only answer that matches what a run will actually do,
    # so anything with text but no exact key is checked rather than assumed.
    for entry in entries.values():
        if entry.has_text and not entry.has_g2p and _universal_available(entry.code):
            entry.has_g2p = True

    _CACHE = Coverage(entries)
    return _CACHE


def _universal_available(code: str) -> bool:
    """Whether this language really converts to universal orthography."""
    try:
        from .lang import resolve
        from .normalise import Normaliser
        normaliser = Normaliser(resolve(code), "universal")
        if not normaliser.enabled:
            return False
        normaliser("test")
        return True
    except Exception:
        return False


def ready_codes() -> List[str]:
    return [entry.code for entry in load().ready()]
