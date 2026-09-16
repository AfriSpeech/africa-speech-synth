"""Generate the dataset card.

Provenance is the whole point: a synthetic dataset is only trustworthy if a
reader can see which text it came from, how sentences were chosen, which G2P
produced the transcripts and which model and voices spoke them. All of that is
already in the run config, so the card is generated rather than hand-written.
"""
from __future__ import annotations

import datetime
from typing import Optional

from .lang import Language
from .normalise import describe as describe_g2p

TEMPLATE = """---
language:
- {lang_code}
license: cc-by-4.0
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- audio
- tts
- synthetic
- african-languages
- {lang_slug}
pretty_name: {title}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
---

# {title}

Synthetic speech for **{lang_name}** ({lang_code}), generated with
[`africa-speech-synth`](https://github.com/AfriSpeech/africa-speech-synth).

{counts}

> **Synthetic data.** Every clip here is machine-generated, not recorded from a
> speaker. It is meant to bootstrap and supplement TTS/ASR training for a
> low-resource language, not to replace recorded speech. Check a sample by ear
> before training on it.

## How it was built

| Stage | What ran |
|---|---|
| Source text | {sources} |
| Selection | {selection} |
| Normalisation | {normalisation} |
| Synthesis | {synthesis} |

### 1. Source text

{sources_detail}

### 2. Sentence selection

{selection_detail}

### 3. Normalisation

{normalisation_detail}

### 4. Synthesis

Audio was generated with **{tts_model}** (backend `{tts_backend}`), voice(s)
**{voices}**, prompted with:

```text
{prompt_example}
```

## Dataset structure

- `data/train-*.parquet` — audio bytes embedded inline (WAV, {sample_rate} Hz mono),
  so the dataset viewer plays each clip next to its text.
- `metadata.jsonl` — one record per clip: `index`, `text`, `normalised_text`,
  `voice`, `shard`, `file_name`.
- `sentences.txt` — the selected source sentences, one per line.

| Column | Description |
|---|---|
| `audio` | Generated speech, WAV @ {sample_rate} Hz mono |
| `text` | The original sentence |
| `normalised_text` | The transcript actually given to the TTS model |
| `voice` | Which voice spoke this clip |

## Reproducing it

```bash
pip install africa-speech-synth
africa-speech-synth run config.yaml
```

```yaml
{config_yaml}```

## License

Audio and transcripts: CC-BY-4.0. The source text keeps the licence of its
own corpus — see the source links above.

---

Built with [africa-speech-synth](https://github.com/AfriSpeech/africa-speech-synth) ·
[africa-g2p](https://github.com/AfriSpeech/africa-g2p) ·
[africa-corpus-builder](https://github.com/AfriSpeech/africa-corpus-builder) ·
[afriso](https://github.com/AfriSpeech/afriso) — generated {date}.
"""

SOURCE_LINKS = {
    "corpus": ("africa-corpus-builder",
               "https://huggingface.co/datasets/AfriSpeech/africa-corpus"),
    "hf": ("HuggingFace dataset", "https://huggingface.co/datasets/{target}"),
    "file": ("local file", None),
}


def _source_detail(uri: str) -> str:
    from .sources import _parse
    scheme, target, column, qualifier = _parse(uri)
    if scheme == "corpus":
        cap = f", capped at {qualifier} sentences" if qualifier else ""
        return (f"- **{target}** monolingual text from "
                f"[`AfriSpeech/africa-corpus`](https://huggingface.co/datasets/AfriSpeech/africa-corpus) "
                f"via [africa-corpus-builder](https://github.com/AfriSpeech/africa-corpus-builder){cap}.")
    if scheme == "hf":
        split = f", split `{qualifier}`" if qualifier else ", all splits"
        return (f"- Column `{column}` of "
                f"[`{target}`](https://huggingface.co/datasets/{target}){split}.")
    return f"- Local file `{target}`" + (f", column `{column}`" if column else "") + "."


def _selection_detail(select_config, selection) -> str:
    if select_config.cover == "none":
        return ("No coverage selection — every sentence within the length filter was kept"
                f" ({select_config.min_chars}–{select_config.max_chars} characters).")
    unit = "phoneme" if select_config.cover == "phoneme" else "word"
    detail = (
        f"Greedy **set cover** over {unit} units: the smallest set of sentences such that every "
        f"{unit} in the corpus appears at least once. This is what keeps a synthetic corpus "
        f"small without leaving sounds unheard — every sentence dropped is an API call saved, "
        f"and every {unit} kept is one the model gets to learn.\n\n"
        f"Sentences were filtered to {select_config.min_chars}–{select_config.max_chars} characters"
    )
    if select_config.min_freq > 1:
        detail += f", and only {unit}s occurring at least {select_config.min_freq} times were targeted"
    detail += "."
    if selection is not None and selection.total_units:
        detail += (f"\n\n**{selection.covered:,} of {selection.total_units:,} {unit} units covered "
                   f"({selection.coverage:.1%}) by {len(selection.sentences):,} sentences.**")
    return detail


def _normalisation_detail(language: Language, mode: str) -> str:
    if mode == "none":
        return "None — the raw source text was sent to the TTS model unchanged."
    if mode == "ipa":
        return (
            "Each sentence was converted to **IPA** with "
            "[`africa-g2p`](https://github.com/AfriSpeech/africa-g2p):\n\n"
            "```python\n"
            "from africa_g2p import AfricaPipeline\n"
            f"AfricaPipeline(lang={language.g2p_code!r}, output='ipa').run(text)\n"
            "```\n\n"
            "IPA puts every language on one shared symbol inventory, which is what you want "
            "when a single model is trained across several languages."
        )
    return (
        "Each sentence was rewritten into **native-orthography phoneme units** with "
        "[`africa-g2p`](https://github.com/AfriSpeech/africa-g2p):\n\n"
        "```python\n"
        "from africa_g2p import AfricaPipeline\n"
        f"AfricaPipeline(lang={language.g2p_code!r}).run(text)\n"
        "```\n\n"
        "Multigraphs (`ny`, `kp`, `gb`, …) stay whole and the text stays inside the language's "
        "own inventory, which trains TTS better than IPA for a single language. The result is "
        "stored as `normalised_text` and is the transcript the TTS model was actually given."
    )


def render(config, language: Language, clips: int, selection=None,
           title: Optional[str] = None) -> str:
    import yaml

    from .synth import render_prompt

    title = title or f"{language.name} Synthetic Speech"
    sources = ", ".join(f"`{uri}`" for uri in config.sources) or "—"
    counts = f"**{clips:,} clips** · {language.name} (`{language.code}`)"
    if language.family:
        counts += f" · {language.family}"

    safe_config = config.to_dict()
    safe_config["tts"].pop("api_key_env", None)

    return TEMPLATE.format(
        lang_code=language.code,
        lang_slug=language.name.lower().replace(" ", "-"),
        lang_name=language.name,
        title=title,
        counts=counts,
        sources=sources,
        selection=(f"greedy set cover ({config.select.cover})"
                   if config.select.cover != "none" else "length filter only"),
        normalisation=describe_g2p(language, config.normalise) or "none",
        synthesis=f"{config.tts.backend} / {config.tts.model}",
        sources_detail="\n".join(_source_detail(uri) for uri in config.sources) or "—",
        selection_detail=_selection_detail(config.select, selection),
        normalisation_detail=_normalisation_detail(language, config.normalise),
        tts_model=config.tts.model,
        tts_backend=config.tts.backend,
        voices=", ".join(config.tts.voices),
        prompt_example=render_prompt(config.tts, language, "<normalised_text>"),
        sample_rate=config.tts.sample_rate,
        config_yaml=yaml.safe_dump(safe_config, sort_keys=False, allow_unicode=True),
        date=datetime.date.today().isoformat(),
    )
