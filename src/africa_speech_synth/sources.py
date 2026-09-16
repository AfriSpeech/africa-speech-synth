"""Text sources.

A source is a short URI string. Several can be combined; the result is
deduplicated in first-seen order, which is what makes a run reproducible.

    corpus:twi                     africa-corpus-builder, monolingual
    corpus:twi@5000                ... capped at 5,000 sentences
    hf:org/dataset#column          any HuggingFace dataset column
    hf:org/dataset#column@train    ... one split only (default: all splits)
    file:sentences.txt             one sentence per line
    file:sentences.csv#text        a CSV column
"""
from __future__ import annotations

import csv
import os
import re
import sys
from typing import Iterable, List

from .lang import Language

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SourceError(Exception):
    pass


# --------------------------------------------------------------------------- corpus

def _import_africa_corpus():
    """africa-corpus-builder is a single module in a repo, not a PyPI package.

    Import it if installed, else look for a clone: AFRICA_CORPUS_PATH, then a
    sibling directory next to this checkout (the common layout).
    """
    try:
        import africa_corpus  # type: ignore
        return africa_corpus
    except ImportError:
        pass

    candidates = []
    env = os.environ.get("AFRICA_CORPUS_PATH")
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_parent = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    candidates.append(os.path.join(repo_parent, "africa-corpus-builder"))

    for path in candidates:
        path = path[:-len("/africa_corpus.py")] if path.endswith("africa_corpus.py") else path
        if os.path.exists(os.path.join(path, "africa_corpus.py")):
            sys.path.insert(0, path)
            import africa_corpus  # type: ignore
            return africa_corpus

    raise SourceError(
        "The `corpus:` source needs africa-corpus-builder, which is not on PyPI yet:\n"
        "    git clone https://github.com/AfriSpeech/africa-corpus-builder\n"
        "    export AFRICA_CORPUS_PATH=$PWD/africa-corpus-builder\n"
        "(or clone it next to this repository)."
    )


def from_corpus(code: str, limit=None, seed: int = 0) -> List[str]:
    """Monolingual sentences for one language from AfriSpeech/africa-corpus."""
    africa_corpus = _import_africa_corpus()
    try:
        return list(africa_corpus.monolingual(code, limit=limit, seed=seed))
    except Exception as exc:
        raise SourceError(f"africa-corpus-builder could not serve {code!r}: {exc}") from exc


# --------------------------------------------------------------------------- huggingface

def from_hf(repo: str, column: str, split=None, limit=None) -> List[str]:
    from datasets import load_dataset

    splits = [split] if split else ["train", "validation", "test"]
    out: List[str] = []
    errors = []
    for name in splits:
        try:
            dataset = load_dataset(repo, split=name)
        except Exception as exc:          # a split that does not exist is normal
            errors.append(f"{name}: {exc}")
            continue
        if column not in dataset.column_names:
            raise SourceError(
                f"{repo} has no column {column!r}. Available: {', '.join(dataset.column_names)}"
            )
        out.extend(text for text in dataset[column] if text)
        if limit and len(out) >= limit:
            return out[:limit]
    if not out:
        raise SourceError(f"No rows read from {repo}#{column}. Tried: {'; '.join(errors)}")
    return out[:limit] if limit else out


# --------------------------------------------------------------------------- file

def from_file(path: str, column=None, limit=None) -> List[str]:
    if not os.path.exists(path):
        raise SourceError(f"No such file: {path}")
    out: List[str] = []
    if column or path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if column and column not in (reader.fieldnames or []):
                raise SourceError(
                    f"{path} has no column {column!r}. Available: {', '.join(reader.fieldnames or [])}"
                )
            key = column or (reader.fieldnames or [None])[0]
            out = [row[key] for row in reader if row.get(key)]
    else:
        with open(path, encoding="utf-8") as handle:
            out = [line.strip() for line in handle if line.strip()]
    return out[:limit] if limit else out


# --------------------------------------------------------------------------- dispatch

def _parse(uri: str):
    """`scheme:target#column@qualifier` -> (scheme, target, column, qualifier)."""
    if ":" not in uri:
        raise SourceError(
            f"Source {uri!r} has no scheme. Expected one of corpus:, hf:, file:"
        )
    scheme, rest = uri.split(":", 1)
    qualifier = None
    if "@" in rest:
        rest, qualifier = rest.rsplit("@", 1)
    column = None
    if "#" in rest:
        rest, column = rest.split("#", 1)
    return scheme.strip().lower(), rest.strip(), column, qualifier


def load_source(uri: str, language: Language) -> List[str]:
    scheme, target, column, qualifier = _parse(uri)
    if scheme == "corpus":
        limit = int(qualifier) if qualifier else None
        return from_corpus(target or language.code, limit=limit)
    if scheme == "hf":
        if not column:
            raise SourceError(f"{uri!r} needs a column: hf:{target}#text")
        return from_hf(target, column, split=qualifier)
    if scheme == "file":
        return from_file(target, column=column,
                         limit=int(qualifier) if qualifier else None)
    raise SourceError(f"Unknown source scheme {scheme!r} in {uri!r}")


def clean(text: str) -> str:
    return " ".join(str(text).split()).strip().strip('"').strip("'")


def split_sentences(texts: Iterable[str]) -> List[str]:
    out = []
    for text in texts:
        for piece in SENTENCE_SPLIT.split(text):
            piece = clean(piece)
            if piece:
                out.append(piece)
    return out


def collect(uris: List[str], language: Language, sentence_split: bool = False) -> List[str]:
    """Load every source in order and deduplicate, first occurrence wins."""
    seen = {}
    for uri in uris:
        texts = load_source(uri, language)
        if sentence_split:
            texts = split_sentences(texts)
        added = 0
        for text in texts:
            text = clean(text)
            if text and text not in seen:
                seen[text] = True
                added += 1
        print(f"  {uri}: {len(texts)} rows, {added} new (total {len(seen)})", flush=True)
    return list(seen)
