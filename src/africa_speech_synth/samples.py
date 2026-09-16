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


def _to_web_audio(wav_path: str, destination_stem: str) -> str:
    """Compress to mono MP3 for the gallery, or copy the WAV if ffmpeg is absent.

    215 raw 24 kHz WAVs is ~65 MB of page weight for clips a few seconds long.
    MP3 at 64 kbps mono is perceptually fine for speech and about a fifth the
    size; the dataset itself keeps the original WAV either way.
    """
    if _ffmpeg():
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
    for sentence in sentences:
        if SAMPLE_MIN_CHARS <= len(sentence) <= SAMPLE_MAX_CHARS:
            return sentence
    return None


def plan(codes: Optional[Sequence[str]] = None,
         voice_list: Optional[Sequence[str]] = None,
         limit: Optional[int] = None) -> List[Sample]:
    """Choose a sentence and a voice for each language, without calling any API."""
    catalogue = coverage.load()
    entries = ([catalogue.entries[c] for c in codes if c in catalogue.entries]
               if codes else catalogue.ready())
    if limit:
        entries = entries[:limit]

    assignment = voices_module.spread([e.code for e in entries], voice_list or ())
    samples: List[Sample] = []
    for position, entry in enumerate(entries, 1):
        try:
            sentences = from_corpus(entry.code, limit=400)
        except Exception as exc:
            print(f"  [{position}/{len(entries)}] {entry.code}: no text ({exc})", flush=True)
            continue
        sentence = _pick_sentence(sentences)
        if not sentence:
            print(f"  [{position}/{len(entries)}] {entry.code}: no sentence in "
                  f"{SAMPLE_MIN_CHARS}-{SAMPLE_MAX_CHARS} chars", flush=True)
            continue

        language = resolve(entry.code)
        try:
            normalised = Normaliser(language, "grapheme")(sentence)
        except Exception:
            normalised = sentence
        samples.append(Sample(
            code=entry.code, name=entry.name, family=entry.family, region=entry.region,
            voice=assignment[entry.code], text=sentence, normalised_text=normalised,
        ))
        if position % 25 == 0:
            print(f"  planned {len(samples)}/{position}", flush=True)
    return samples


def _utterances(samples: Sequence[Sample]) -> List[Utterance]:
    out = []
    for index, sample in enumerate(samples):
        language = resolve(sample.code)
        out.append(Utterance(index=index, text=sample.text,
                             transcript=sample.normalised_text,
                             voice=sample.voice, language=language,
                             name=f"sample_{sample.code}"))
    return out


def build(config, out_dir: str, codes: Optional[Sequence[str]] = None,
          limit: Optional[int] = None, resume: bool = True) -> List[Sample]:
    """Plan, synthesise and collect samples into `out_dir/audio`."""
    print(f"Planning samples ({'all ready languages' if not codes else len(codes)})", flush=True)
    samples = plan(codes, config.tts.voices if codes else None, limit=limit)
    print(f"  {len(samples)} languages with a usable sentence", flush=True)
    if not samples:
        return []

    assignment = {s.code: s.voice for s in samples}
    spread_counts = voices_module.distribution(assignment)
    print(f"  {len(spread_counts)} distinct voices, "
          f"{min(spread_counts.values())}-{max(spread_counts.values())} languages each",
          flush=True)

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
        record = by_name.get(f"sample_{sample.code}")
        if not record:
            continue
        destination = _to_web_audio(record["audio_path"],
                                    os.path.join(audio_dir, sample.code))
        sample.audio = f"audio/{os.path.basename(destination)}"
        kept.append(sample)

    total_mb = sum(os.path.getsize(os.path.join(out_dir, s.audio)) for s in kept) / 1e6
    print(f"  audio: {total_mb:.1f}MB"
          f"{'' if _ffmpeg() else ' (install ffmpeg to compress)'}", flush=True)

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
