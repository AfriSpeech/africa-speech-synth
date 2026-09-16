import json
import os

import pytest

import mock_backend  # noqa: F401  (registers the "mock" backend)
from africa_speech_synth import RunConfig, from_dict, resolve
from africa_speech_synth.lang import Language
from africa_speech_synth.normalise import Normaliser
from africa_speech_synth.select import run as select_run, word_units
from africa_speech_synth.sources import _parse, clean, collect, split_sentences
from africa_speech_synth.tts.base import pcm_to_wav, wav_sample_rate
from africa_speech_synth import pipeline

SENTENCES = [
    "Akwaaba mo nyinaa wo Ghana ha",
    "Me din de Kofi na mefiri Kumasi",
    "Wo ho te sen anopa yi",
    "Ɔkɔɔ sukuu no mu ntɛm ara",
    "Akwaaba mo nyinaa wo Ghana ha",
]


# --------------------------------------------------------------- language

def test_resolve_by_code_and_name():
    assert resolve("twi").g2p_code == "twi"
    assert resolve("Twi").code == "twi"
    assert resolve("Yoruba").g2p_code == "yor"


def test_resolve_unknown_has_no_g2p():
    assert resolve("definitelynotalanguage").has_g2p is False


# --------------------------------------------------------------- sources

def test_parse_source_uris():
    assert _parse("corpus:twi@5000") == ("corpus", "twi", None, "5000")
    assert _parse("hf:org/ds#transcript@train") == ("hf", "org/ds", "transcript", "train")
    assert _parse("file:a.csv#text") == ("file", "a.csv", "text", None)


def test_clean_and_split():
    assert clean('  "hello   world" ') == "hello   world".replace("   ", " ")
    assert split_sentences(["One. Two! Three?"]) == ["One.", "Two!", "Three?"]


def test_collect_dedupes_in_order(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("\n".join(["b", "a", "b"]), encoding="utf-8")
    out = collect([f"file:{path}"], resolve("twi"))
    assert out == ["b", "a"]


# --------------------------------------------------------------- selection

def test_word_cover_is_minimal_and_complete():
    selection = select_run(SENTENCES, cover="word", min_chars=5)
    assert selection.coverage == 1.0
    covered = set()
    for sentence in selection.sentences:
        covered |= set(word_units(sentence))
    assert covered == set(word_units(" ".join(SENTENCES)))
    # the duplicate sentence must not be selected twice
    assert len(selection.sentences) == len(set(selection.sentences))


def test_phoneme_cover_uses_g2p():
    normaliser = Normaliser(resolve("twi"), "grapheme")
    selection = select_run(SENTENCES, cover="phoneme", unit_fn=normaliser.units, min_chars=5)
    assert selection.coverage == 1.0
    assert selection.total_units > 0


def test_length_filter():
    selection = select_run(["short", "a long enough sentence here"], cover="none", min_chars=10)
    assert selection.sentences == ["a long enough sentence here"]


def test_max_sentences_caps_selection():
    selection = select_run(SENTENCES, cover="word", min_chars=5, max_sentences=1)
    assert len(selection.sentences) == 1


# --------------------------------------------------------------- normalisation

def test_grapheme_and_ipa_differ():
    grapheme = Normaliser(resolve("twi"), "grapheme")("Akwaaba")
    ipa = Normaliser(resolve("twi"), "ipa")("Akwaaba")
    assert grapheme and ipa and grapheme != ipa


def test_normalise_none_is_passthrough():
    normaliser = Normaliser(resolve("twi"), "none")
    assert normaliser("Akwaaba") == "Akwaaba"
    assert normaliser.enabled is False


def test_missing_g2p_table_is_a_clear_error():
    from africa_speech_synth.lang import LanguageNotSupported
    with pytest.raises(LanguageNotSupported):
        Normaliser(Language(token="x", code="xxx", name="X"), "grapheme")


# --------------------------------------------------------------- audio

def test_pcm_to_wav_reads_rate_from_mime():
    wav = pcm_to_wav(b"\x00\x00" * 100, "audio/L16;rate=16000")
    assert wav[:4] == b"RIFF"
    assert wav_sample_rate(wav) == 16000


# --------------------------------------------------------------- config

def test_config_roundtrip_and_unknown_key():
    config = from_dict({"language": "twi", "select": {"cover": "word"}})
    assert config.select.cover == "word"
    with pytest.raises(ValueError):
        from_dict({"nonsense": 1})


# --------------------------------------------------------------- end to end

def _config(tmp_path, sentences_file):
    return from_dict({
        "language": "twi",
        "sources": [f"file:{sentences_file}"],
        "normalise": "grapheme",
        "out": str(tmp_path / "out"),
        "select": {"cover": "word", "min_chars": 5},
        "tts": {"backend": "mock", "model": "mock-1", "voices": ["A", "B"],
                "rpm": 0, "concurrency": 4},
        "package": {"formats": ["parquet", "ljspeech"]},
    })


def test_end_to_end(tmp_path):
    source = tmp_path / "sentences.txt"
    source.write_text("\n".join(SENTENCES), encoding="utf-8")
    config = _config(tmp_path, source)

    result = pipeline.run(config)
    assert result["clips"] >= 4

    # parquet shards exist and carry embedded audio
    from datasets import load_dataset
    dataset = load_dataset("parquet", data_files=str(tmp_path / "out" / "data" / "*.parquet"),
                           split="train")
    assert set(dataset.column_names) == {"audio", "text", "normalised_text", "voice"}
    assert dataset[0]["audio"]["array"] is not None
    assert dataset[0]["normalised_text"] != dataset[0]["text"]
    assert {row["voice"] for row in dataset} == {"A", "B"}

    # manifest and card
    manifest = (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == result["clips"]
    assert json.loads(manifest[0])["shard"].startswith("data/train-")
    card = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    assert "africa-g2p" in card and "Twi" in card

    # ljspeech export
    assert (tmp_path / "out" / "ljspeech" / "metadata.csv").exists()


def test_resume_skips_finished_clips(tmp_path, capsys):
    source = tmp_path / "sentences.txt"
    source.write_text("\n".join(SENTENCES), encoding="utf-8")
    config = _config(tmp_path, source)

    pipeline.run(config)
    capsys.readouterr()
    pipeline.run(config)
    out = capsys.readouterr().out
    assert "already done" in out


def test_dry_run_calls_no_backend(tmp_path):
    source = tmp_path / "sentences.txt"
    source.write_text("\n".join(SENTENCES), encoding="utf-8")
    config = _config(tmp_path, source)
    config.tts.backend = "does-not-exist"
    result = pipeline.run(config, dry_run=True)
    assert result["dry_run"] is True
    assert not os.path.exists(os.path.join(config.work_dir, "audio", "000"))


def test_macrolanguage_falls_back_to_a_member():
    """`Akan` has no g2p table; Twi, one of its members, does."""
    pytest.importorskip("afriso")
    language = resolve("Akan")
    assert language.g2p_code == "twi"
    assert language.name == "Twi"
