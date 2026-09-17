"""Build the samples gallery as a static HuggingFace Space.

The Space is a build artefact of this repository, not a project of its own: the
page is generated here from `samples.json` plus the audio files, so it is always
consistent with whatever the pipeline currently produces.

Static Space, no backend — the page is one HTML file with the clips beside it,
or, with `--audio-repo`, one HTML file that streams them from a dataset repo.
Splitting the two matters once the gallery carries every voice for every
language: that is thousands of clips and hundreds of megabytes, which belongs
in a dataset people can load and reuse, not inside a Space nobody can cite.
"""
from __future__ import annotations

import html
import json
import os
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from . import coverage, voices as voices_module
from .samples import Sample

# The gallery is synthetic audio, and a listener should know that before they
# press play rather than after — "African Speech Samples" reads like a corpus
# of recorded speech, which is the one thing this is not. The title matches the
# dataset the clips live in, so the page and its audio are obviously the same
# artefact rather than two things that happen to look alike.
DEFAULT_TITLE = "Synthetic Voice Samples · Africa"

DATASET_README = """---
license: mit
task_categories:
- text-to-speech
language_creators:
- found
pretty_name: {title}
tags:
- synthetic
- tts
- synthetic
- african-languages
- speech-synthesis
---

# {title}

**Synthetic speech. No human speaker was recorded for any clip here.**

Every clip was generated with [afrispeech-synth](https://github.com/AfriSpeech/afrispeech-synth)
from Google Gemini, reading text from
[africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus).

- **{clips} clips** across **{languages} languages**, in **{voices} Gemini voices**
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

Browse and listen: **{space_url}**
"""

SPACE_README = """---
title: {title}
emoji: 🎧
colorFrom: yellow
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
tags:
- tts
- synthetic
- african-languages
- speech-synthesis
---

# {title}

**Synthetic speech — model-generated, not recorded.** No human speaker was recorded for any
clip here. Generated with
[afrispeech-synth](https://github.com/AfriSpeech/afrispeech-synth).

Voices come from Google Gemini's 30-voice catalogue — either one per language, or every
voice for every language when the gallery was built with `--all-voices`. Audio is
model-generated, not recorded speech.
"""

STYLE = """
:root {
  --bg: #fbfaf7; --card: #fff; --ink: #16130d; --muted: #6b6559;
  --line: #e7e2d8; --accent: #b3540f; --accent-soft: #fdf1e3; --ok: #2f6b3a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14120f; --card: #1c1a16; --ink: #f2eee6; --muted: #a39c8e;
    --line: #2e2a24; --accent: #e8944a; --accent-soft: #2a2119; --ok: #7fc08c;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 20px 80px; }
header { padding: 56px 0 28px; border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 10px; font-size: clamp(28px, 4.2vw, 40px); letter-spacing: -0.02em; }
.lede { margin: 0; max-width: 62ch; color: var(--muted); font-size: 16px; }
.lede a { color: var(--accent); }
.stats { display: flex; flex-wrap: wrap; gap: 28px; margin: 26px 0 0; padding: 0; list-style: none; }
.stats b { display: block; font-size: 26px; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.stats span { color: var(--muted); font-size: 13px; }
.controls { position: sticky; top: 0; z-index: 5; background: var(--bg);
  padding: 18px 0 14px; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
input[type=search], select {
  font: inherit; color: inherit; background: var(--card);
  border: 1px solid var(--line); border-radius: 9px; padding: 9px 12px;
}
input[type=search] { flex: 1 1 260px; min-width: 0; }
input[type=search]:focus, select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.count { color: var(--muted); font-size: 13px; margin-left: auto; }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); margin-top: 18px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 13px; padding: 15px 16px 13px; }
.card h3 { margin: 0; font-size: 16.5px; letter-spacing: -0.01em; }
.meta { color: var(--muted); font-size: 12.5px; margin: 3px 0 11px; }
.voice { display: inline-block; background: var(--accent-soft); color: var(--accent);
  border-radius: 999px; padding: 2px 9px; font-size: 12px; font-weight: 600; white-space: nowrap; }
audio { width: 100%; height: 34px; margin: 2px 0 11px; }
.text { font-size: 14px; margin: 0 0 6px; }
.norm { font-size: 12.5px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  word-break: break-word; margin: 0; }
details { margin-top: 34px; border-top: 1px solid var(--line); padding-top: 22px; }
summary { cursor: pointer; font-weight: 600; }
.pending { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.pending span { background: var(--card); border: 1px solid var(--line); border-radius: 7px;
  padding: 3px 8px; font-size: 12.5px; color: var(--muted); }
.empty { color: var(--muted); padding: 40px 0; text-align: center; }
footer { margin-top: 46px; padding-top: 22px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 13px; }
footer a { color: var(--accent); }
footer p { max-width: 70ch; }
"""

SCRIPT = """
const cards = Array.from(document.querySelectorAll('.card'));
const search = document.getElementById('q');
const voiceSel = document.getElementById('voice');
const regionSel = document.getElementById('region');
const count = document.getElementById('count');

function apply() {
  const q = search.value.trim().toLowerCase();
  const v = voiceSel.value, r = regionSel.value;
  let shown = 0;
  for (const card of cards) {
    const hit = (!q || card.dataset.search.includes(q))
      && (!v || card.dataset.voices.split(',').includes(v))
      && (!r || card.dataset.region === r);
    card.hidden = !hit;
    if (hit) {
      shown++;
      // Filtering to a voice should play that voice, not leave the card on
      // whichever one happened to be selected.
      const pick = card.querySelector('.voice-pick');
      if (v && pick && pick.value !== v) { pick.value = v; swap(card); }
    }
  }
  count.textContent = shown + (shown === 1 ? ' language' : ' languages');
  document.getElementById('empty').hidden = shown > 0;
  // Stop audio that has just been filtered away, so a hidden card is not
  // still playing with no visible way to pause it.
  for (const audio of document.querySelectorAll('audio')) {
    if (audio.closest('.card').hidden && !audio.paused) audio.pause();
  }
}
// Swapping voice reloads the clip in place; a playing one is stopped first so
// the new source does not start mid-way through the old one's progress bar.
function swap(card) {
  const audio = card.querySelector('audio');
  const pick = card.querySelector('.voice-pick');
  if (!audio || !pick) return;
  const src = JSON.parse(card.dataset.srcs)[pick.value];
  if (!src || audio.getAttribute('src') === src) return;
  audio.pause();
  audio.setAttribute('src', src);
  audio.load();
}
for (const pick of document.querySelectorAll('.voice-pick')) {
  pick.addEventListener('change', (e) => swap(e.target.closest('.card')));
}
// Only one clip at a time: starting a second pauses the first.
document.addEventListener('play', (event) => {
  for (const audio of document.querySelectorAll('audio')) {
    if (audio !== event.target) audio.pause();
  }
}, true);
search.addEventListener('input', apply);
voiceSel.addEventListener('change', apply);
regionSel.addEventListener('change', apply);
apply();
"""


def _card(group: Sequence[Sample]) -> str:
    """One card per language. Several voices become a picker, not more cards.

    With every voice for every language the gallery is 30x the clips but the
    same 200-odd languages, so the language stays the unit you browse and the
    voice becomes a control inside it.
    """
    sample = group[0]
    bits = [sample.code]
    if sample.family:
        bits.append(sample.family)
    if sample.region:
        bits.append(sample.region)
    voices = sorted(group, key=lambda s: s.voice)
    search_key = " ".join([sample.name, sample.code, sample.family or "",
                           sample.region or ""] + [s.voice for s in voices]).lower()
    srcs = json.dumps({s.voice: (s.audio or "") for s in voices})

    if len(voices) == 1:
        picker = (f'<span class="voice" title="{html.escape(voices_module.describe(sample.voice))}">'
                  f'{html.escape(sample.voice)}</span>')
    else:
        options = "".join(
            f'<option value="{html.escape(s.voice)}">{html.escape(s.voice)} — '
            f'{html.escape(voices_module.describe(s.voice))}</option>' for s in voices)
        picker = (f'<select class="voice-pick" aria-label="Voice for '
                  f'{html.escape(sample.name)}">{options}</select>')

    return f"""      <article class="card" data-voices="{html.escape(','.join(s.voice for s in voices))}"
        data-region="{html.escape(sample.region or '')}"
        data-srcs='{html.escape(srcs)}'
        data-search="{html.escape(search_key)}">
        <div class="row" style="justify-content:space-between;align-items:start;gap:8px">
          <div>
            <h3>{html.escape(sample.name)}</h3>
            <p class="meta">{html.escape(' · '.join(bits))}</p>
          </div>
          {picker}
        </div>
        <audio controls preload="none" src="{html.escape(voices[0].audio or '')}"></audio>
        <p class="text">{html.escape(sample.text)}</p>
        <p class="norm">{html.escape(sample.normalised_text)}</p>
      </article>"""


def render(samples: Sequence[Sample], title: str = DEFAULT_TITLE) -> str:
    groups: Dict[str, List[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.code, []).append(sample)
    ordered = sorted(groups.values(), key=lambda g: g[0].name.lower())
    used_voices = sorted({s.voice for s in samples})
    regions = sorted({s.region for s in samples if s.region})
    per_language = max((len(g) for g in ordered), default=0)
    lede_voices = (
        f"Every language is read by all <strong>{len(used_voices)}</strong> voices, in the same "
        "sentence — so switching voice on a card changes the voice and nothing else. Pick the "
        "one you want and name it in your own run."
        if per_language > 1 else
        "Each language is spoken by a <strong>different</strong> one of Gemini's 30 voices, so "
        "the gallery covers the whole catalogue — the voice badge tells you which one to ask "
        "for in your own run.")

    catalogue = coverage.load()
    pending = [e for e in catalogue.sorted() if e.has_g2p and not e.has_text]

    voice_options = "".join(
        f'<option value="{html.escape(v)}">{html.escape(v)} — '
        f'{html.escape(voices_module.describe(v))}</option>' for v in used_voices)
    region_options = "".join(
        f'<option value="{html.escape(r)}">{html.escape(r)}</option>' for r in regions)
    pending_chips = "".join(
        f"<span>{html.escape(e.name)} <code>{html.escape(e.code)}</code></span>" for e in pending)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- A Space is served inside an iframe on huggingface.co, so a link with no target
     tries to open in that frame and is blocked by its sandbox — every link on the
     page looked broken. One base rule sends them all to a new tab. -->
<base target="_blank">
<title>{html.escape(title)}</title>
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>{html.escape(title)}</h1>
  <p class="lede">One synthetic speech sample per African language, built with
    <a href="https://github.com/AfriSpeech/afrispeech-synth">afrispeech-synth</a>.
    {lede_voices}</p>
  <ul class="stats">
    <li><b>{len(ordered)}</b><span>languages</span></li>
    <li><b>{len(used_voices)}</b><span>voices</span></li>
    <li><b>{len(samples)}</b><span>clips</span></li>
    <li><b>{len(regions)}</b><span>regions</span></li>
  </ul>
</header>

<div class="controls">
  <div class="row">
    <input type="search" id="q" placeholder="Search language, code, family or voice…"
      aria-label="Search languages">
    <select id="voice" aria-label="Filter by voice">
      <option value="">All voices</option>{voice_options}
    </select>
    <select id="region" aria-label="Filter by region">
      <option value="">All regions</option>{region_options}
    </select>
    <span class="count" id="count"></span>
  </div>
</div>

<main class="grid">
{chr(10).join(_card(g) for g in ordered)}
</main>
<p class="empty" id="empty" hidden>No language matches that filter.</p>

<details>
  <summary>{len(pending)} more languages have a G2P table but no text yet</summary>
  <p class="meta">These can be synthesised today — the pipeline just needs sentences. Point it at
    your own with <code>--source file:sentences.txt</code>, or contribute text to
    <a href="https://github.com/AfriSpeech/africa-corpus-builder">africa-corpus-builder</a>
    so they appear here.</p>
  <div class="pending">{pending_chips}</div>
</details>

<footer>
  <p><strong>This audio is machine-generated.</strong> Every clip was produced by
    <a href="https://ai.google.dev/gemini-api/docs/speech-generation">Google Gemini TTS</a>
    (<code>gemini-3.1-flash-tts-preview</code>) from written text — no one recorded it, and no
    clip is a real speaker of the language. Quality varies a lot by language: the voices were
    not trained for most of these, and some will be plainly wrong. That is the point of showing
    them.</p>
  <p>Sentences come from
    <a href="https://github.com/AfriSpeech/africa-corpus-builder">africa-corpus-builder</a>,
    whose text is drawn from Bible translations, so the sample sentences are scripture verses.
    Phonemisation is <a href="https://github.com/AfriSpeech/africa-g2p">africa-g2p</a> — the
    monospace line under each sentence is what the model was actually asked to speak. Language
    metadata from <a href="https://github.com/AfriSpeech/afriso">afriso</a>.</p>
  <p>For <em>recorded</em> African speech, see
    <a href="https://github.com/AfriSpeech/afrispeech-selector">afrispeech-selector</a>.</p>
</footer>
</div>
<script>{SCRIPT}</script>
</body>
</html>
"""


def build(samples: Sequence[Sample], out_dir: str, title: str = DEFAULT_TITLE,
          audio_base: Optional[str] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if audio_base:
        # Rewrite to absolute URLs so the page works with the clips living in a
        # dataset repo rather than beside it.
        base = audio_base.rstrip("/")
        samples = [replace(s, audio=f"{base}/{s.audio}") if s.audio else s
                   for s in samples]
    page = os.path.join(out_dir, "index.html")
    with open(page, "w", encoding="utf-8") as handle:
        handle.write(render(samples, title=title))
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(SPACE_README.format(title=title))
    with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as handle:
        json.dump([s.to_dict() for s in samples], handle, ensure_ascii=False, indent=1)
    print(f"  built {page} ({len(samples)} samples)", flush=True)
    return page


def push_audio(out_dir: str, repo_id: str, title: str = DEFAULT_TITLE,
               space_repo: Optional[str] = None, private: bool = False,
               token: Optional[str] = None) -> str:
    """Upload the clips to a dataset repo; return the base URL to stream from."""
    from huggingface_hub import HfApi

    from .publish import _token
    from .samples import load_manifest

    samples = load_manifest(out_dir)
    languages = len({s.code for s in samples})
    voices = len({s.voice for s in samples})
    space_url = (f"https://huggingface.co/spaces/{space_repo}" if space_repo
                 else "https://huggingface.co/spaces/AfriSpeech")

    api = HfApi(token=_token(token))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    card = os.path.join(out_dir, "DATASET_README.md")
    with open(card, "w", encoding="utf-8") as handle:
        handle.write(DATASET_README.format(title=title, clips=len(samples),
                                           languages=languages, voices=voices,
                                           space_url=space_url))
    print(f"  uploading audio -> dataset {repo_id}", flush=True)
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="dataset",
                      allow_patterns=["audio/**", "samples.json"])
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    os.remove(card)
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    print(f"  audio: https://huggingface.co/datasets/{repo_id}", flush=True)
    return base


def push(out_dir: str, repo_id: str, private: bool = False,
         token: Optional[str] = None, audio_elsewhere: bool = False) -> str:
    from huggingface_hub import HfApi

    from .publish import _token

    api = HfApi(token=_token(token))
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static",
                    private=private, exist_ok=True)
    print(f"  uploading {out_dir} -> {repo_id}", flush=True)
    ignore = ["work/**", "*.tmp"]
    if audio_elsewhere:
        ignore.append("audio/**")
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="space",
                      ignore_patterns=ignore)
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"  live: {url}", flush=True)
    return url
