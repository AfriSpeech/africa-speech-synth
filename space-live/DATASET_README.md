---
license: mit
task_categories:
- text-to-speech
language_creators:
- found
pretty_name: Synthetic Voice Samples · Africa
tags:
- synthetic
- tts
- synthetic
- african-languages
- speech-synthesis
---

# Synthetic Voice Samples · Africa

**Synthetic speech. No human speaker was recorded for any clip here.**

Every clip was generated with [afrispeech-synth](https://github.com/AfriSpeech/afrispeech-synth)
from Google Gemini, reading text from
[africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus).

- **16856 clips** across **562 languages**, in **30 Gemini voices**
- Each language is read by every voice, **the same sentence throughout**, so the voices are
  directly comparable — the voice changes and nothing else does
- WAV source, published here as mono MP3; the generator writes 24 kHz 16-bit PCM

`samples.json` carries one row per clip: language code and name, family, region, voice, the
original sentence, the normalised text that was actually spoken, and the audio path.

## What this is for

Choosing a voice before generating a dataset of your own, and hearing how far a synthetic
voice gets on a given language. Quality varies enormously by language — the voices were
built for widely-spoken languages and are being asked to read others.

## What this is not

Recorded speech, a pronunciation reference, or evidence that a language *sounds* like this.
A synthetic clip is a model's guess at an orthography. Treat it as a starting point for
bootstrapping, never as ground truth.

Browse and listen: **https://huggingface.co/spaces/AfriSpeech/afrispeech-synth-samples**
