#!/usr/bin/env bash
# Downloads inclusionAI/Ming-3.0-tts-0.5B's model.safetensors and tokenizer.json at the pinned
# revision into $1, checking model.safetensors' sha256. Skips files that are already present.
set -euo pipefail
DIR="$1"
REVISION=e5bd8da31ecf1b9276cdbdda821fa891068f1368
MODEL_SHA256=45c4d78d8ad6b1d942e3da86ba035fece02de569f7b2ca7e9d121dfaf18d674c
BASE="https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B/resolve/$REVISION"
mkdir -p "$DIR"
for f in tokenizer.json model.safetensors; do
  [ -s "$DIR/$f" ] || { echo "downloading $f"; curl -fL --retry 3 -o "$DIR/$f.part" "$BASE/$f" && mv "$DIR/$f.part" "$DIR/$f"; }
done
if [ ! -f "$DIR/.verified" ]; then
  echo "$MODEL_SHA256  $DIR/model.safetensors" | shasum -a 256 -c - && touch "$DIR/.verified"
fi
