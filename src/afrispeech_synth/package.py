"""Package a finished run into a training-ready dataset.

Parquet shards carry the audio *bytes* inline rather than a path. That is what
makes the HuggingFace dataset viewer play each clip next to its text, and it is
what makes the dataset usable straight from the Hub with no second download
step. Shards are sized by cumulative audio bytes to stay under the Hub's
comfortable per-file ceiling.
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
from typing import List, Optional, Sequence

from .synth import Workspace


def _shard_by_size(records: List[dict], target_bytes: int) -> List[List[dict]]:
    shards, current, size = [], [], 0
    for record in records:
        record_size = record.get("bytes") or os.path.getsize(record["audio_path"])
        if current and size + record_size > target_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(record)
        size += record_size
    if current:
        shards.append(current)
    return shards


# What the Hub reads to know a column holds audio. `datasets` writes exactly this
# key into the Arrow schema; the viewer and `load_dataset` both pick it up.
# String columns a caller can add beyond the four every run writes. A gallery
# spanning hundreds of languages needs the language on the row: the alternative
# is one dataset config per language, which makes `load_dataset` take an
# argument nobody can guess and hides cross-language work behind 500 configs.
EXTRA_COLUMNS = ("language", "language_name", "family", "region")

# Rows carry embedded audio at roughly 400 KB each, so this is about 40 MB of
# row group — well inside the 300 MB the Hub's viewer will scan.
ROW_GROUP_ROWS = 100


def _hf_features_metadata(sample_rate: int, extra: Sequence[str] = ()) -> dict:
    features = {
        "audio": {"sampling_rate": sample_rate, "_type": "Audio"},
        "text": {"dtype": "string", "_type": "Value"},
        "normalised_text": {"dtype": "string", "_type": "Value"},
        "voice": {"dtype": "string", "_type": "Value"},
    }
    for name in extra:
        features[name] = {"dtype": "string", "_type": "Value"}
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")}


def to_parquet(records: List[dict], out_dir: str, sample_rate: int = 24000,
               shard_target_mb: int = 190, extra: Sequence[str] = (),
               shuffle_seed: Optional[int] = None) -> List[str]:
    """Write shards with pyarrow rather than through `datasets`.

    The audio is already WAV bytes, so nothing needs encoding — but
    `Audio.encode_example` imports a codec unconditionally before it looks at what
    it was handed, and which codec that is has changed across releases (soundfile
    on datasets 2.x, torchcodec on 5.x). Writing the struct directly produces a
    byte-identical file, and the only dependency is pyarrow, which parquet needs
    anyway.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if shuffle_seed is not None:
        # Shards are read in order, so writing them grouped by language and
        # voice — which is how they were generated — puts every clip of one
        # language in one shard. A reader streaming the first shard would see
        # one language in one voice and think that was the dataset. Shuffling
        # once, with a fixed seed, spreads both across every shard and keeps
        # the file set reproducible.
        records = list(records)
        random.Random(shuffle_seed).shuffle(records)

    data_dir = os.path.join(out_dir, "data")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    schema = pa.schema(
        [
            pa.field("audio", pa.struct([pa.field("bytes", pa.binary()),
                                         pa.field("path", pa.string())])),
            pa.field("text", pa.string()),
            pa.field("normalised_text", pa.string()),
            pa.field("voice", pa.string()),
            *[pa.field(name, pa.string()) for name in extra],
        ],
        metadata=_hf_features_metadata(sample_rate, extra),
    )

    shards = _shard_by_size(records, shard_target_mb * 1024 * 1024)
    print(f"  {len(records)} clips -> {len(shards)} shard(s)", flush=True)
    paths = []
    for number, shard in enumerate(shards):
        name = f"train-{number:05d}-of-{len(shards):05d}.parquet"
        audio, text, normalised, voices = [], [], [], []
        extras = {name: [] for name in extra}
        for record in shard:
            with open(record["audio_path"], "rb") as handle:
                audio.append({"bytes": handle.read(),
                              "path": os.path.basename(record["audio_path"])})
            text.append(record["text"])
            normalised.append(record.get("transcript", record["text"]))
            voices.append(record.get("voice", ""))
            for field in extra:
                extras[field].append(str(record.get(field) or ""))
            record["shard"] = f"data/{name}"
            record["file_name"] = os.path.basename(record["audio_path"])
        table = pa.Table.from_pydict(
            {"audio": audio, "text": text, "normalised_text": normalised,
             "voice": voices, **extras},
            schema=schema,
        )
        path = os.path.join(data_dir, name)
        # One row group per shard is pyarrow's default and unreadable here: the
        # Hub's viewer scans a row group whole and refuses anything over 300 MB,
        # so a 400 MB shard fails to preview at all. Embedded audio makes rows
        # ~400 KB, so a few hundred of them is a comfortable group, and the page
        # index lets a reader seek to one row instead of decoding the group.
        pq.write_table(table, path, row_group_size=ROW_GROUP_ROWS,
                       write_page_index=True)
        paths.append(path)
        print(f"  [{number + 1}/{len(shards)}] {path} rows={table.num_rows} "
              f"size={os.path.getsize(path) / 1e6:.1f}MB", flush=True)
        del table, audio, text, normalised, voices, extras
    return paths


def write_manifest(records: List[dict], out_dir: str) -> str:
    path = os.path.join(out_dir, "metadata.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({
                "index": record["index"],
                "text": record["text"],
                "normalised_text": record.get("transcript", record["text"]),
                "voice": record.get("voice", ""),
                "shard": record.get("shard"),
                "file_name": record.get("file_name") or os.path.basename(record["audio_path"]),
            }, ensure_ascii=False) + "\n")
    return path


def write_sentences(records: List[dict], out_dir: str) -> str:
    path = os.path.join(out_dir, "sentences.txt")
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record["text"] + "\n")
    return path


def to_ljspeech(records: List[dict], out_dir: str) -> str:
    """wavs/ + metadata.csv — the layout Piper, VITS and MeloTTS all read."""
    root = os.path.join(out_dir, "ljspeech")
    wavs = os.path.join(root, "wavs")
    os.makedirs(wavs, exist_ok=True)
    with open(os.path.join(root, "metadata.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for record in records:
            stem = os.path.splitext(os.path.basename(record["audio_path"]))[0]
            shutil.copy2(record["audio_path"], os.path.join(wavs, f"{stem}.wav"))
            writer.writerow([stem, record["text"], record.get("transcript", record["text"])])
    print(f"  ljspeech export: {root}", flush=True)
    return root


def build(work_dir: str, out_dir: str, config, card: Optional[str] = None) -> dict:
    workspace = Workspace(work_dir)
    records = workspace.records()
    if not records:
        raise RuntimeError(f"No synthesised clips found in {work_dir}")
    os.makedirs(out_dir, exist_ok=True)

    result = {"clips": len(records), "files": []}
    if "parquet" in config.package.formats:
        result["files"] += to_parquet(records, out_dir,
                                      sample_rate=config.tts.sample_rate,
                                      shard_target_mb=config.package.shard_target_mb)
    if "ljspeech" in config.package.formats:
        to_ljspeech(records, out_dir)

    result["manifest"] = write_manifest(records, out_dir)
    result["sentences"] = write_sentences(records, out_dir)
    if card:
        readme = os.path.join(out_dir, "README.md")
        with open(readme, "w", encoding="utf-8") as handle:
            handle.write(card)
        result["card"] = readme
    return result
