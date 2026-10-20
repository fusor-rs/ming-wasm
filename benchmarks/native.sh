#!/bin/bash
# Times the native CLI: `ming bench` (per-stage ms/frame) and `ming say` (RTF, first audio, load
# time, peak RSS via /usr/bin/time -l) for each bundle, thread count, preset and sentence.
#
#   benchmarks/native.sh [out-dir]            # default benchmarks/out/native
#   BUNDLES="dist/model" THREADS="8" PRESETS="quality" SENTENCES="S1" REPEAT=3 benchmarks/native.sh
#
# Writes <out>/bench-<bundle>-t<threads>.txt and one TSV row per `say` run to <out>/say.tsv.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-benchmarks/out/native}
mkdir -p "$out"
MING=${MING:-target/release/ming}
BUNDLES=${BUNDLES:-"dist/model dist/model-q7"}
THREADS=${THREADS:-"1 4 8"}
PRESETS=${PRESETS:-"quality balanced fast"}
SENTENCES=${SENTENCES:-"S1 S2 S3"}
REPEAT=${REPEAT:-3}
text() {
  case $1 in
    S1) echo "Please remember to water the plants on Tuesday, and call the dentist about the appointment at three o'clock." ;;
    S2) echo "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold." ;;
    S3) echo "Glaciers move slowly, carving valleys over thousands of years while the rivers below carry the silt to the sea." ;;
    *) echo "unknown sentence $1" >&2; exit 2 ;;
  esac
}
tsv="$out/say.tsv"
[ -f "$tsv" ] || printf 'bundle\tthreads\tpreset\tsentence\trun\tframes\taudio_s\tgen_s\trtf\tms_frame\tfirst_audio_s\tload_s\tpeak_rss_mb\tloadavg\n' > "$tsv"

for bundle in $BUNDLES; do
  name=$(basename "$bundle")
  for t in $THREADS; do
    echo "=== bench $name threads=$t ($(uptime | sed 's/.*load/load/'))"
    nice -n 19 "$MING" --threads "$t" bench "$bundle" 2>&1 | tee "$out/bench-$name-t$t.txt"
    for preset in $PRESETS; do
      for s in $SENTENCES; do
        for ((i = 0; i < REPEAT; i++)); do
          wav="$out/$name-t$t-$preset-$s-$i.wav"
          log=$(nice -n 19 /usr/bin/time -l "$MING" --threads "$t" say "$bundle" "$(text "$s")" -o "$wav" --preset "$preset" --seed 1895 2>&1)
          # "1 fragment(s), 69 frames, 5.40s audio in 1.30s: RTF 0.241, 18.9 ms/frame, first audio 0.21s -> out.wav"
          say=$(grep -F 'fragment(s)' <<< "$log")
          frames=$(sed -E 's/.*, ([0-9]+) frames.*/\1/' <<< "$say")
          audio=$(sed -E 's/.*frames, ([0-9.]+)s audio.*/\1/' <<< "$say")
          gen=$(sed -E 's/.*audio in ([0-9.]+)s.*/\1/' <<< "$say")
          rtf=$(sed -E 's/.*RTF ([0-9.]+).*/\1/' <<< "$say")
          msf=$(sed -E 's/.*RTF [0-9.]+, ([0-9.]+) ms\/frame.*/\1/' <<< "$say")
          first=$(sed -E 's/.*first audio ([0-9.]+)s.*/\1/' <<< "$say")
          load=$(grep -F 'loaded ' <<< "$log" | sed -E 's/.*in ([0-9.]+)s.*/\1/')
          rss=$(grep -F 'maximum resident set size' <<< "$log" | awk '{printf "%.0f", $1 / 1e6}')
          la=$(sysctl -n vm.loadavg | awk '{print $2}')
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "$t" "$preset" "$s" "$i" "$frames" "$audio" "$gen" "$rtf" "$msf" "$first" "$load" "$rss" "$la" | tee -a "$tsv"
        done
      done
    done
  done
done
