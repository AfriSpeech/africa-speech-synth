#!/usr/bin/env bash
# Build a small synthetic corpus for several languages in one go.
# Each language gets its own work directory, so any of them can be resumed
# independently after an interruption.
set -euo pipefail

: "${GEMINI_API_KEY:?export GEMINI_API_KEY first}"

for lang in twi yor hau swh ewe gaa; do
  afrispeech-synth run \
    --lang "$lang" \
    --source "corpus:$lang" \
    --cover phoneme \
    --max-sentences 1000 \
    --voices Zephyr,Puck,Kore \
    --out "out/$lang"
done
