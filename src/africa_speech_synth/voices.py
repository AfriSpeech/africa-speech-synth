"""The Gemini TTS voice catalogue, and how voices are spread across languages.

A Gemini voice is the same voice in every language — Zephyr does not become a
different speaker for Yoruba. What changes is how well that voice handles the
language's phonology. So for a samples gallery there is nothing to gain from
giving every language the same voice: assigning a different one to each
language covers the whole catalogue at no extra cost, and a listener browsing
the gallery hears all 30 rather than one repeated 215 times.

Source: https://ai.google.dev/gemini-api/docs/speech-generation
"""
from __future__ import annotations

from typing import Dict, List, Sequence

# name -> the characteristic Google documents for it
GEMINI_VOICES: Dict[str, str] = {
    "Zephyr": "Bright",
    "Puck": "Upbeat",
    "Charon": "Informative",
    "Kore": "Firm",
    "Fenrir": "Excitable",
    "Leda": "Youthful",
    "Orus": "Firm",
    "Aoede": "Breezy",
    "Callirrhoe": "Easy-going",
    "Autonoe": "Bright",
    "Enceladus": "Breathy",
    "Iapetus": "Clear",
    "Umbriel": "Easy-going",
    "Algieba": "Smooth",
    "Despina": "Smooth",
    "Erinome": "Clear",
    "Algenib": "Gravelly",
    "Rasalgethi": "Informative",
    "Laomedeia": "Upbeat",
    "Achernar": "Soft",
    "Alnilam": "Firm",
    "Schedar": "Even",
    "Gacrux": "Mature",
    "Pulcherrima": "Forward",
    "Achird": "Friendly",
    "Zubenelgenubi": "Casual",
    "Vindemiatrix": "Gentle",
    "Sadachbia": "Lively",
    "Sadaltager": "Knowledgeable",
    "Sulafat": "Warm",
}

ALL = list(GEMINI_VOICES)


def describe(voice: str) -> str:
    return GEMINI_VOICES.get(voice, "")


def unknown(voices: Sequence[str]) -> List[str]:
    """Voices not in the catalogue — a typo here costs a whole run."""
    return [v for v in voices if v not in GEMINI_VOICES]


def spread(codes: Sequence[str], voices: Sequence[str] = ()) -> Dict[str, str]:
    """Assign one voice per language, round-robin over sorted codes.

    Round-robin rather than hashing so the distribution is even by
    construction — with 215 languages and 30 voices every voice gets 7 or 8,
    and no voice is missing from the gallery by chance.
    """
    voices = list(voices) or ALL
    return {code: voices[index % len(voices)] for index, code in enumerate(sorted(codes))}


def distribution(assignment: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for voice in assignment.values():
        counts[voice] = counts.get(voice, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
