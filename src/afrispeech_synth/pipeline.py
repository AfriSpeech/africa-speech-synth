"""The five stages, wired together.

    source ──► select ──► normalise ──► synthesise ──► package/publish

Each stage is callable on its own (the CLI exposes them as subcommands), and
each writes its output to the work directory, so a run can be stopped after any
stage and picked up later.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from . import card as card_module
from . import package as package_module
from . import publish as publish_module
from . import select as select_module
from . import sources as sources_module
from .config import RunConfig
from .lang import Language, resolve
from .normalise import MODE_WARNINGS, Normaliser
from .synth import build_utterances, synthesise

SENTENCES_FILE = "sentences.txt"
SELECTION_FILE = "selection.json"


def _banner(step: str, title: str) -> None:
    print(f"\n[{step}] {title}", flush=True)


def stage_sources(config: RunConfig, language: Language) -> List[str]:
    _banner("1/5", f"Source text ({len(config.sources)} source(s))")
    if not config.sources:
        raise ValueError("No sources configured. Add at least one, e.g. sources: [corpus:twi]")
    sentences = sources_module.collect(config.sources, language, sentence_split=True)
    print(f"  {len(sentences)} unique sentences", flush=True)
    return sentences


def stage_select(config: RunConfig, language: Language, sentences: List[str],
                 normaliser: Normaliser):
    _banner("2/5", f"Selection (cover={config.select.cover})")
    unit_fn = normaliser.units if config.select.cover == "phoneme" else None
    if config.select.cover == "phoneme" and not normaliser.enabled:
        raise ValueError(
            "cover: phoneme needs G2P, but normalise is 'none'. "
            "Use cover: word, or set normalise: grapheme."
        )
    return select_module.run(
        sentences,
        cover=config.select.cover,
        unit_fn=unit_fn,
        min_freq=config.select.min_freq,
        max_sentences=config.select.max_sentences,
        min_chars=config.select.min_chars,
        max_chars=config.select.max_chars,
        seed=config.select.seed,
    )


def stage_synthesise(config: RunConfig, language: Language, sentences: List[str],
                     normaliser: Normaliser, resume: bool = True) -> dict:
    _banner("3-4/5", f"Normalise ({config.normalise}) + synthesise ({config.tts.backend})")
    started = time.time()
    utterances = build_utterances(sentences, normaliser)
    print(f"  {len(utterances)} utterances prepared", flush=True)
    result = synthesise(utterances, config.tts, language, config.work_dir, resume=resume)
    print(f"  synthesised {result['done']}, skipped {result['skipped']}, "
          f"failed {result['failed']} in {time.time() - started:.0f}s", flush=True)
    return result


def stage_package(config: RunConfig, language: Language, selection=None) -> dict:
    _banner("5/5", f"Package -> {config.out}")
    records = len(package_module.Workspace(config.work_dir).records())
    rendered = card_module.render(config, language, records, selection=selection)
    result = package_module.build(config.work_dir, config.out, config, card=rendered)
    print(f"  packaged {result['clips']} clips into {config.out}", flush=True)

    if config.package.push_to:
        url = publish_module.push(config.out, config.package.push_to,
                                  private=config.package.private)
        result["url"] = url
    return result


def _save_sentences(config: RunConfig, sentences: List[str], selection) -> None:
    os.makedirs(config.work_dir, exist_ok=True)
    with open(os.path.join(config.work_dir, SENTENCES_FILE), "w", encoding="utf-8") as handle:
        handle.write("\n".join(sentences) + "\n")
    if selection is not None:
        with open(os.path.join(config.work_dir, SELECTION_FILE), "w", encoding="utf-8") as handle:
            json.dump({"covered": selection.covered, "total_units": selection.total_units,
                       "coverage": selection.coverage, "uncovered": selection.uncovered[:500],
                       "sentences": len(selection.sentences)}, handle, ensure_ascii=False, indent=2)


def load_sentences(config: RunConfig) -> Optional[List[str]]:
    path = os.path.join(config.work_dir, SENTENCES_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def run(config: RunConfig, resume: bool = True, dry_run: bool = False) -> dict:
    language = resolve(config.language)
    print(f"Language: {language.name} ({language.code})"
          f"{'' if language.has_g2p else ' — no africa-g2p table'}", flush=True)

    normaliser = Normaliser(language, config.normalise)
    warning = MODE_WARNINGS.get(config.normalise)
    if warning:
        print(f"  note: {warning}", flush=True)

    sentences = load_sentences(config) if resume else None
    selection = None
    if sentences:
        print(f"\nReusing {len(sentences)} selected sentences from {config.work_dir}", flush=True)
    else:
        sentences = stage_sources(config, language)
        selection = stage_select(config, language, sentences, normaliser)
        sentences = selection.sentences
        _save_sentences(config, sentences, selection)

    if dry_run:
        print(f"\nDry run: {len(sentences)} sentences selected, nothing synthesised.", flush=True)
        return {"sentences": len(sentences), "dry_run": True}

    stage_synthesise(config, language, sentences, normaliser, resume=resume)
    return stage_package(config, language, selection=selection)
