#!/bin/bash
# Repeats e2e/run.mjs (a fresh Chrome profile each time: first-visit load, synthesis, ASR WER,
# revisit) and collects one TSV row per run from its results JSON.
#
#   benchmarks/browser-e2e.sh "<build> <threads> <preset>" ...    # e.g. "web-threaded-relaxed 8 quality"
#   REPEAT=3 PORT=8795 benchmarks/browser-e2e.sh "web-single 1 quality"
set -euo pipefail
cd "$(dirname "$0")/.."
out=benchmarks/out/browser; mkdir -p "$out"
REPEAT=${REPEAT:-3}
export PORT=${PORT:-8795}
tsv="$out/e2e.tsv"
[ -f "$tsv" ] || printf 'build\tthreads\tpreset\trun\tload_s\tframes\taudio_s\tgen_s\trtf\tms_frame\twer\trevisit_s\trequests\tloadavg\tresult\n' > "$tsv"
for cfg in "$@"; do
  set -- $cfg
  for ((i = 0; i < REPEAT; i++)); do
    echo "=== $cfg run $i ($(uptime | sed 's/.*load/load/'))"
    line=$(nice -n 19 node e2e/run.mjs --build "$1" --threads "$2" --preset "$3" 2>&1 | tee /dev/stderr | grep -oE 'e2e/results/[^ ]+\.json' | tail -1 || true)
    [ -n "$line" ] || { echo "no results file" >&2; continue; }
    node -e '
      const r = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
      const m = /([\d.]+) s of audio in ([\d.]+) s · real-time factor ([\d.]+) · (\d+) frames · (\d+) network/.exec(r.stats ?? "") ?? [];
      const wer = /WER=([\d.]+)/.exec(r.asr ?? "")?.[1] ?? "";
      const la = require("os").loadavg()[0].toFixed(1);
      console.log([process.argv[2], process.argv[3], process.argv[4], process.argv[5], (r.loadMs / 1000).toFixed(2), m[4], m[1], m[2], m[3], m[4] ? (1000 * m[2] / m[4]).toFixed(1) : "", wer, (r.revisitMs / 1000).toFixed(2), m[5], la, process.argv[1]].join("\t"));
    ' "$line" "$1" "$2" "$3" "$i" | tee -a "$tsv"
  done
done
