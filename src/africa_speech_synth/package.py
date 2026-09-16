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
import shutil
from typing import List, Optional

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


def to_parquet(records: List[dict], out_dir: str, sample_rate: int = 24000,
               shard_target_mb: int = 190) -> List[str]:
    from datasets import Audio, Dataset, Features, Value

    data_dir = os.path.join(out_dir, "data")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    features = Features({
        "audio": Audio(sampling_rate=sample_rate),
        "text": Value("string"),
        "normalised_text": Value("string"),
        "voice": Value("string"),
    })

    shards = _shard_by_size(records, shard_target_mb * 1024 * 1024)
    print(f"  {len(records)} clips -> {len(shards)} shard(s)", flush=True)
    paths = []
    for number, shard in enumerate(shards):
        name = f"train-{number:05d}-of-{len(shards):05d}.parquet"
        audio, text, normalised, voices = [], [], [], []
        for record in shard:
            with open(record["audio_path"], "rb") as handle:
                audio.append({"bytes": handle.read(),
                              "path": os.path.basename(record["audio_path"])})
            text.append(record["text"])
            normalised.append(record.get("transcript", record["text"]))
            voices.append(record.get("voice", ""))
            record["shard"] = f"data/{name}"
            record["file_name"] = os.path.basename(record["audio_path"])
        dataset = Dataset.from_dict(
            {"audio": audio, "text": text, "normalised_text": normalised, "voice": voices},
            features=features,
        )
        path = os.path.join(data_dir, name)
        dataset.to_parquet(path)
        paths.append(path)
        print(f"  [{number + 1}/{len(shards)}] {path} rows={len(dataset)} "
              f"size={os.path.getsize(path) / 1e6:.1f}MB", flush=True)
        del dataset, audio, text, normalised, voices
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
