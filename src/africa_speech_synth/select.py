"""Sentence selection by greedy set cover.

The goal of a synthetic TTS corpus is not "many sentences" but "every sound the
language has, in enough contexts". Greedy set cover over coverage units gets
there with far fewer utterances — which matters directly, since every sentence
kept is a TTS API call paid for.

Coverage unit:
  phoneme  every phoneme in the language's inventory appears (best for TTS)
  word     every word above --min-freq appears (what the Twi run used)
  none     keep everything, just filter by length

The greedy loop is lazy (a max-heap with stale-key revalidation) rather than
rescanning candidates each round, which is what makes phoneme mode usable over
100k+ sentence corpora.
"""
from __future__ import annotations

import heapq
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class Selection:
    sentences: List[str]
    covered: int
    total_units: int
    uncovered: List[str]

    @property
    def coverage(self) -> float:
        return self.covered / self.total_units if self.total_units else 1.0


def word_units(text: str) -> List[str]:
    return WORD.findall(text.lower())


def filter_by_length(sentences: Sequence[str], min_chars: int, max_chars: int) -> List[str]:
    return [s for s in sentences if min_chars <= len(s) <= max_chars]


def greedy_cover(sentences: Sequence[str],
                 unit_fn: Callable[[str], List[str]],
                 min_freq: int = 1,
                 max_sentences: Optional[int] = None,
                 progress_every: int = 250) -> Selection:
    unit_sets = [set(unit_fn(s)) for s in sentences]

    counts = Counter()
    for units in unit_sets:
        counts.update(units)
    targets = {unit for unit, n in counts.items() if n >= min_freq}
    if not targets:
        return Selection(list(sentences), 0, 0, [])

    # Sentences that cover nothing we need can never be picked; dropping them
    # here keeps the heap small.
    candidates = [(i, units & targets) for i, units in enumerate(unit_sets)]
    candidates = [(i, units) for i, units in candidates if units]

    unit_to_sentences = defaultdict(list)
    for position, (_, units) in enumerate(candidates):
        for unit in units:
            unit_to_sentences[unit].append(position)

    uncovered = set(targets)
    heap = [(-len(units), position) for position, (_, units) in enumerate(candidates)]
    heapq.heapify(heap)
    gains = {position: len(units) for position, (_, units) in enumerate(candidates)}

    chosen: List[int] = []
    while uncovered and heap:
        if max_sentences and len(chosen) >= max_sentences:
            break
        negative_gain, position = heapq.heappop(heap)
        current = len(candidates[position][1] & uncovered)
        if current == 0:
            continue
        # Lazy greedy: a popped entry may be stale. Re-push with the true gain
        # and take the next one; only an entry that is still the best is used.
        if current != -negative_gain:
            gains[position] = current
            heapq.heappush(heap, (-current, position))
            continue
        chosen.append(candidates[position][0])
        uncovered -= candidates[position][1]
        if progress_every and len(chosen) % progress_every == 0:
            print(f"    selected {len(chosen)} sentences, {len(uncovered)} units uncovered",
                  flush=True)

    return Selection(
        sentences=[sentences[i] for i in sorted(chosen)],
        covered=len(targets) - len(uncovered),
        total_units=len(targets),
        uncovered=sorted(uncovered),
    )


def run(sentences: Sequence[str],
        cover: str = "phoneme",
        unit_fn: Optional[Callable[[str], List[str]]] = None,
        min_freq: int = 1,
        max_sentences: Optional[int] = None,
        min_chars: int = 8,
        max_chars: int = 240,
        seed: int = 0) -> Selection:
    pool = filter_by_length(sentences, min_chars, max_chars)
    print(f"  {len(pool)} of {len(sentences)} sentences within {min_chars}-{max_chars} chars",
          flush=True)

    if cover == "none":
        if max_sentences and len(pool) > max_sentences:
            order = {sentence: i for i, sentence in enumerate(pool)}
            sampled = random.Random(seed).sample(pool, max_sentences)
            pool = sorted(sampled, key=order.__getitem__)
        return Selection(pool, 0, 0, [])

    if cover == "word":
        unit_fn = word_units
    elif cover == "phoneme":
        if unit_fn is None:
            raise ValueError("phoneme cover needs a Normaliser-backed unit_fn")
    else:
        raise ValueError(f"cover must be phoneme, word or none — got {cover!r}")

    selection = greedy_cover(pool, unit_fn, min_freq=min_freq, max_sentences=max_sentences)
    print(f"  covered {selection.covered}/{selection.total_units} {cover} units "
          f"({selection.coverage:.1%}) with {len(selection.sentences)} sentences", flush=True)
    if selection.uncovered:
        print(f"  uncovered: {', '.join(selection.uncovered[:20])}"
              f"{' …' if len(selection.uncovered) > 20 else ''}", flush=True)
    return selection
