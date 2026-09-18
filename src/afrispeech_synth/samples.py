"""Build one sample clip per language for the showcase gallery.

Every language gets a *different* voice, assigned round-robin, so browsing the
gallery covers all 30 voices instead of hearing one repeated 215 times. A Gemini
voice is the same speaker in every language, so nothing is lost by spreading
them: what a listener learns from each clip is how that voice handles that
language's phonology.

One pass, one backend, one rate limiter — the per-utterance voice and language
overrides on Utterance let the normal synthesis runner do the work.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import coverage, voices as voices_module
from .lang import Language, resolve
from .normalise import Normaliser
from .sources import from_corpus
from .synth import Utterance, synthesise

SAMPLE_MIN_CHARS = 45
SAMPLE_MAX_CHARS = 150


def _ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _to_web_audio(wav_path: str, destination_stem: str, compress: bool = False) -> str:
    """Copy the clip for the gallery, compressing to mono MP3 only if asked.

    Lossless by default. Compression existed for when the clips shipped inside
    the Space, where a few hundred MB of WAV is page weight nobody wants. Once
    they live in a dataset repo that trade is the wrong way round: a dataset
    should hold what the model produced, and a lossy step baked in at publish
    time cannot be undone by whoever downloads it.
    """
    if compress and _ffmpeg():
        out = destination_stem + ".mp3"
        result = subprocess.run(
            [_ffmpeg(), "-y", "-loglevel", "error", "-i", wav_path,
             "-ac", "1", "-b:a", "64k", out],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.exists(out):
            return out
        print(f"    ffmpeg failed for {wav_path}, keeping WAV", flush=True)
    out = destination_stem + ".wav"
    shutil.copyfile(wav_path, out)
    return out


@dataclass
class Sample:
    code: str
    name: str
    family: str
    region: str
    voice: str
    text: str
    normalised_text: str
    audio: Optional[str] = None     # path relative to the gallery root

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _pick_sentence(sentences: Sequence[str]) -> Optional[str]:
    """First sentence in the length window — deterministic, so reruns match."""
    picked = _pick_sentences(sentences, 1)
    return picked[0] if picked else None


def _pick_sentences(sentences: Sequence[str], count: int) -> List[str]:
    """The first `count` distinct sentences in the length window.

    Deterministic, so a rerun produces the same dataset rather than a new one.
    Duplicates are skipped: corpus text repeats, and a language whose voices all
    read the same line twice is quietly less varied than its clip count claims.
    """
    out, seen = [], set()
    for sentence in sentences:
        if not (SAMPLE_MIN_CHARS <= len(sentence) <= SAMPLE_MAX_CHARS):
            continue
        key = sentence.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
        if len(out) >= count:
            break
    return out


def plan(codes: Optional[Sequence[str]] = None,
         voice_list: Optional[Sequence[str]] = None,
         limit: Optional[int] = None,
         normalise: str = "grapheme",
         all_voices: bool = False,
         distinct: bool = False) -> List[Sample]:
    """Choose a sentence and voice(s) for each language, without calling any API.

    By default each language gets one voice, spread round-robin so the gallery
    covers the catalogue. With `all_voices`, every language is read by every
    voice instead: the same sentence across the whole catalogue, which is what
    makes voices comparable — you hear the voice change and nothing else.
    """
    catalogue = coverage.load()
    entries = ([catalogue.entries[c] for c in codes if c in catalogue.entries]
               if codes else catalogue.ready())
    if limit:
        entries = entries[:limit]

    pool = list(voice_list or voices_module.ALL)
    assignment = voices_module.spread([e.code for e in entries], voice_list or ())
    samples: List[Sample] = []
    for position, entry in enumerate(entries, 1):
        try:
            sentences = from_corpus(entry.code, limit=400)
        except Exception as exc:
            print(f"  [{position}/{len(entries)}] {entry.code}: no text ({exc})", flush=True)
            continue
        wanted = len(pool) if (all_voices and distinct) else 1
        chosen_text = _pick_sentences(sentences, wanted)
        if not chosen_text:
            print(f"  [{position}/{len(entries)}] {entry.code}: no sentence in "
                  f"{SAMPLE_MIN_CHARS}-{SAMPLE_MAX_CHARS} chars", flush=True)
            continue
        if len(chosen_text) < wanted:
            # Fewer usable sentences than voices: take the voices we can fill
            # rather than repeating text, so "distinct" stays true.
            print(f"  [{position}/{len(entries)}] {entry.code}: only "
                  f"{len(chosen_text)} distinct sentences, using that many voices",
                  flush=True)
        sentence = chosen_text[0]

        language = resolve(entry.code)
        try:
            normalised = Normaliser(language, normalise)(sentence)
        except Exception as exc:
            # A language without a table for this mode still gets a sample; the
            # model is simply given its own orthography.
            print(f"  [{position}/{len(entries)}] {entry.code}: {normalise} unavailable "
                  f"({type(exc).__name__}), sending original text", flush=True)
            normalised = sentence
        if all_voices and distinct:
            pairs = list(zip(pool, chosen_text))
        elif all_voices:
            pairs = [(voice, sentence) for voice in pool]
        else:
            pairs = [(assignment[entry.code], sentence)]
        for voice, text in pairs:
            try:
                text_norm = Normaliser(language, normalise)(text)
            except Exception:
                text_norm = text
            samples.append(Sample(
                code=entry.code, name=entry.name, family=entry.family,
                region=entry.region, voice=voice, text=text,
                normalised_text=text_norm,
            ))
        if position % 25 == 0:
            print(f"  planned {len(samples)}/{position}", flush=True)
    return samples


def _stem(sample: Sample) -> str:
    return f"sample_{sample.code}_{sample.voice}"


def _utterances(samples: Sequence[Sample]) -> List[Utterance]:
    # Grouped by voice: a Live session's voice is fixed when it connects, so
    # consecutive utterances sharing one keeps the session pool warm instead of
    # reconnecting per clip. Order does not affect output — stems are unique.
    ordered = sorted(samples, key=lambda s: (s.voice, s.code))
    out = []
    for index, sample in enumerate(ordered):
        language = resolve(sample.code)
        out.append(Utterance(index=index, text=sample.text,
                             transcript=sample.normalised_text,
                             voice=sample.voice, language=language,
                             name=_stem(sample)))
    return out


def build(config, out_dir: str, codes: Optional[Sequence[str]] = None,
          limit: Optional[int] = None, resume: bool = True,
          all_voices: bool = False, compress: bool = False,
          distinct: bool = False) -> List[Sample]:
    """Plan, synthesise and collect samples into `out_dir/audio`."""
    print(f"Planning samples ({'all ready languages' if not codes else len(codes)}, "
          f"normalise={config.normalise}"
          f"{', every voice' if all_voices else ''}"
          f"{', distinct sentences' if distinct else ''})", flush=True)
    samples = plan(codes, config.tts.voices or None, limit=limit,
                   normalise=config.normalise, all_voices=all_voices,
                   distinct=distinct)
    print(f"  {len(samples)} languages with a usable sentence", flush=True)
    if not samples:
        return []

    per_voice: Dict[str, int] = {}
    for sample in samples:
        per_voice[sample.voice] = per_voice.get(sample.voice, 0) + 1
    languages = len({s.code for s in samples})
    print(f"  {languages} languages x {len(per_voice)} voices = {len(samples)} clips "
          f"({min(per_voice.values())}-{max(per_voice.values())} per voice)", flush=True)

    work_dir = os.path.join(out_dir, "work")
    result = synthesise(_utterances(samples), config.tts, resolve(config.language),
                        work_dir, resume=resume)
    print(f"  synthesised {result['done']}, skipped {result['skipped']}, "
          f"failed {result['failed']}", flush=True)

    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    by_name = {record.get("name"): record for record in result["workspace"].records()}
    kept: List[Sample] = []
    for sample in samples:
        record = by_name.get(_stem(sample))
        if not record:
            continue
        # One directory per language. A flat audio/ is simpler but a Hub repo
        # allows at most 10,000 files per directory, and an all-voices gallery
        # passes that at a few hundred languages — the push is rejected outright.
        language_dir = os.path.join(audio_dir, sample.code)
        os.makedirs(language_dir, exist_ok=True)
        destination = _to_web_audio(record["audio_path"],
                                    os.path.join(language_dir, sample.voice),
                                    compress=compress)
        sample.audio = f"audio/{sample.code}/{os.path.basename(destination)}"
        kept.append(sample)

    total_mb = sum(os.path.getsize(os.path.join(out_dir, s.audio)) for s in kept) / 1e6
    print(f"  audio: {total_mb:.1f}MB "
          f"({'MP3' if compress else 'WAV, as generated'})", flush=True)

    manifest = os.path.join(out_dir, "samples.json")
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump([s.to_dict() for s in kept], handle, ensure_ascii=False, indent=1)
    print(f"  {len(kept)} samples -> {out_dir}", flush=True)
    return kept


def load_manifest(out_dir: str) -> List[Sample]:
    path = os.path.join(out_dir, "samples.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [Sample(**row) for row in json.load(handle)]
