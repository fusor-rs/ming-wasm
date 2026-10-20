#!/bin/bash
# Repeats scripts/node-smoke.mjs (the single-threaded wasm builds in Node) and collects a TSV.
#   REPEAT=3 benchmarks/node.sh [web-single web-single-relaxed]
set -euo pipefail
cd "$(dirname "$0")/.."
out=benchmarks/out/node; mkdir -p "$out"
REPEAT=${REPEAT:-3}
tsv="$out/node.tsv"
[ -f "$tsv" ] || printf 'build\trun\tload_s\tframes\taudio_s\tgen_s\trtf\tms_frame\tfirst_audio_s\tloadavg\n' > "$tsv"
for build in ${@:-web-single web-single-relaxed}; do
  for ((i = 0; i < REPEAT; i++)); do
    echo "=== $build run $i ($(uptime | sed 's/.*load/load/'))"
    line=$(nice -n 19 node scripts/node-smoke.mjs dist/model --build "$build" --out "$out/$build-$i.wav" 2>&1 | grep -F 'RTF')
    echo "$line"
    # "web-single: load 1.2s, 69 frames, 5.40s audio in 24.9s: RTF 4.61, 361 ms/frame, first audio 0.95s"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$build" "$i" \
      "$(sed -E 's/.*load ([0-9.]+)s.*/\1/' <<< "$line")" "$(sed -E 's/.*, ([0-9]+) frames.*/\1/' <<< "$line")" \
      "$(sed -E 's/.*frames, ([0-9.]+)s audio.*/\1/' <<< "$line")" "$(sed -E 's/.*audio in ([0-9.]+)s.*/\1/' <<< "$line")" \
      "$(sed -E 's/.*RTF ([0-9.]+).*/\1/' <<< "$line")" "$(sed -E 's/.*, ([0-9.]+) ms\/frame.*/\1/' <<< "$line")" \
      "$(sed -E 's/.*first audio ([0-9.]+)s.*/\1/' <<< "$line")" "$(sysctl -n vm.loadavg | awk '{print $2}')" | tee -a "$tsv"
  done
done
