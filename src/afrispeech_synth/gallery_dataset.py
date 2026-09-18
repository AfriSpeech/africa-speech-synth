"""Publish a samples gallery as a loadable HuggingFace dataset.

The gallery's own output is a folder of clips plus `samples.json`, which is what
the page needs. That is not a dataset: `load_dataset` on it gets a pile of files
with no schema. This turns it into parquet shards with the audio embedded and
the language and voice on every row, so one `load_dataset` call gets everything
and a subset is a filter rather than a config nobody can guess.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .package import EXTRA_COLUMNS, to_parquet
from .samples import load_manifest

SHUFFLE_SEED = 0

# The dataset holds the clips; the Space is the demo you click through to hear
# them. Naming both "demo" would hide which one you can actually download.
DEFAULT_TITLE = "Synthetic Voice Samples · Africa"

CARD = """---
license: mit
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: {title}
size_categories:
- 10K<n<100K
tags:
- synthetic
- tts
- african-languages
- speech-synthesis
- multilingual
configs:
- config_name: default
  data_files: data/*.parquet
---

# {title}

**Synthetic speech. No human speaker was recorded for any clip in this dataset.**

Generated with [afrispeech-synth](https://github.com/AfriSpeech/afrispeech-synth): text from
[africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus), normalised to a
universal orthography with [africa-g2p](https://github.com/AfriSpeech/africa-g2p), spoken by
Google Gemini's Live API.

- **{clips} clips** · **{hours:.1f} hours** · **{languages} languages** · **{voices} voices**
- Every clip is a **distinct sentence** — no sentence is repeated
- Each language is read by up to {voices} different voices, one sentence per voice
- ~{per_voice:.2f} hours per voice

## Audio

Exactly as the model produced it — no resampling, no re-encoding, no conversion.

| | |
|---|---|
| Format | WAV, 16-bit signed PCM |
| Sample rate | 24,000 Hz |
| Channels | 1 (mono) |
| Clip length | {mean:.1f}s mean, {dmin:.1f}-{dmax:.1f}s |

## Two ways in

`data/*.parquet` carries the audio inline for training. `audio/` holds the same clips as
individual WAV files, which is what the browsable gallery streams and what you want if you
need one clip rather than the set. The clips are identical; only the packaging differs.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train")

twi = ds.filter(lambda r: r["language"] == "twi")     # one language
zephyr = ds.filter(lambda r: r["voice"] == "Zephyr")  # one voice
```

Rows are **shuffled**, so language and voice are spread across every shard: streaming the
first shard gives a cross-section rather than one language in one voice.

## Columns

| Column | |
|---|---|
| `audio` | the clip (24 kHz mono WAV) |
| `text` | the sentence as it appears in the corpus |
| `normalised_text` | what the model was actually asked to read (universal orthography) |
| `voice` | which Gemini voice spoke it |
| `language` | ISO 639-3 code |
| `language_name` | e.g. Dagbani |
| `family` | e.g. Niger-Congo |
| `region` | e.g. West Africa |

`text` and `normalised_text` differ because universal orthography maps a language's own
letters onto the shared set most African languages use — `ɔ` to `o`, `ɛ` to `e`, `gy` to `j`.
The model read `normalised_text`.

## What this is not

Recorded speech, a pronunciation reference, or evidence that a language *sounds* like this.
A synthetic clip is a model's guess at an orthography, and quality varies enormously by
language — these voices were built for widely-spoken languages and are being asked to read
hundreds of others. Treat it as a starting point for bootstrapping, never as ground truth.
"""


def _records(out_dir: str) -> List[dict]:
    """Join the synthesis records with the gallery's language metadata."""
    meta_dir = os.path.join(out_dir, "work", "meta")
    by_code: Dict[str, dict] = {}
    for sample in load_manifest(out_dir):
        by_code.setdefault(sample.code, {
            "language_name": sample.name, "family": sample.family, "region": sample.region})

    records = []
    for name in sorted(os.listdir(meta_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(meta_dir, name), encoding="utf-8") as handle:
            try:
                record = json.load(handle)
            except json.JSONDecodeError:
                continue
        if not os.path.exists(record.get("audio_path", "")):
            continue
        code = record.get("language", "")
        record.update(language=code, **by_code.get(code, {
            "language_name": code, "family": "", "region": ""}))
        records.append(record)
    return records


def build(out_dir: str, repo: str, title: str, shard_target_mb: int = 400) -> List[str]:
    records = _records(out_dir)
    if not records:
        raise SystemExit(f"No finished clips under {out_dir}/work — run `samples` first.")

    durations = [(r["bytes"] - 44) / ((r.get("sample_rate") or 24000) * 2) for r in records]
    total = sum(durations)
    stats = dict(
        title=title, repo=repo, clips=f"{len(records):,}",
        hours=total / 3600,
        languages=len({r["language"] for r in records}),
        voices=len({r["voice"] for r in records}),
        per_voice=total / 3600 / max(1, len({r["voice"] for r in records})),
        mean=total / len(records), dmin=min(durations), dmax=max(durations),
    )
    print(f"  {stats['clips']} clips, {stats['hours']:.1f} h, "
          f"{stats['languages']} languages, {stats['voices']} voices", flush=True)

    paths = to_parquet(records, out_dir, shard_target_mb=shard_target_mb,
                       extra=EXTRA_COLUMNS, shuffle_seed=SHUFFLE_SEED)
    card = os.path.join(out_dir, "DATASET_README.md")
    with open(card, "w", encoding="utf-8") as handle:
        handle.write(CARD.format(**stats))
    return paths


def push(out_dir: str, repo: str, private: bool = False,
         token: Optional[str] = None) -> str:
    """Upload shards, loose clips and the card; return the streaming base URL."""
    from huggingface_hub import HfApi

    from .publish import _token

    api = HfApi(token=_token(token))
    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"  uploading shards -> {repo}", flush=True)
    # Parquet for loading, loose WAVs for streaming one clip at a time — the
    # gallery page fetches individual files over HTTP, which parquet cannot
    # serve. Same audio twice, packaged for the two different jobs.
    api.upload_folder(folder_path=out_dir, repo_id=repo, repo_type="dataset",
                      allow_patterns=["data/*.parquet", "audio/**", "samples.json"])
    api.upload_file(path_or_fileobj=os.path.join(out_dir, "DATASET_README.md"),
                    path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    print(f"  live: https://huggingface.co/datasets/{repo}", flush=True)
    return f"https://huggingface.co/datasets/{repo}/resolve/main"
