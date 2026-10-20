"""Prints Markdown tables (median and min over repeats) from the files under benchmarks/out/:
pt/*.json (pytorch_bench.py), native/say.tsv (native.sh), browser/e2e.tsv (browser-e2e.sh),
browser/*.json (browser.mjs), node/node.tsv (node.sh) and kokoro/*.json (kokoro.mjs).

  python3 benchmarks/summarize.py [benchmarks/out]
"""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def med_min(values, digits=2):
    values = [float(v) for v in values if v not in ("", None)]
    if not values:
        return "–"
    return f"{statistics.median(values):.{digits}f} ({min(values):.{digits}f})"


def table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


def group(items, key):
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups


# PyTorch
pt = []
for path in sorted(glob.glob(os.path.join(out, "pt", "*.json"))):
    pt.append(json.load(open(path)))
if pt:
    rows = []
    for s in pt:
        for sentence, runs in sorted(group(s["runs"], lambda r: r["sentence"]).items()):
            rows.append([s["dtype"], s["device"], s["threads"], sentence, runs[0]["frames"], runs[0]["audio_s"],
                         med_min([r["gen_s"] for r in runs]), med_min([r["rtf"] for r in runs]),
                         med_min([r["ms_per_frame"] for r in runs], 0), s["load_s"], s["peak_rss_mb"],
                         med_min([r["load_avg"] for r in runs], 1)])
    table("PyTorch reference", ["dtype", "device", "threads", "sentence", "frames", "audio s", "gen s", "RTF", "ms/frame",
                                "load s", "peak RSS MB", "load avg"], rows)

# Native
say = os.path.join(out, "native", "say.tsv")
if os.path.exists(say):
    rows = []
    items = list(csv.DictReader(open(say), delimiter="\t"))
    for key, runs in sorted(group(items, lambda r: (r["bundle"], int(r["threads"]), r["preset"], r["sentence"])).items(),
                            key=lambda kv: (kv[0][0], kv[0][1], ["quality", "balanced", "fast"].index(kv[0][2]), kv[0][3])):
        rows.append([*key, runs[0]["frames"], runs[0]["audio_s"], med_min([r["gen_s"] for r in runs]),
                     med_min([r["rtf"] for r in runs], 3), med_min([r["ms_frame"] for r in runs], 1),
                     med_min([r["first_audio_s"] for r in runs]), med_min([r["load_s"] for r in runs]),
                     med_min([r["peak_rss_mb"] for r in runs], 0), med_min([r["loadavg"] for r in runs], 1)])
    table("Native `ming say`", ["bundle", "threads", "preset", "sentence", "frames", "audio s", "gen s", "RTF", "ms/frame",
                                "first audio s", "load s", "peak RSS MB", "load avg"], rows)

# Browser e2e
e2e = os.path.join(out, "browser", "e2e.tsv")
if os.path.exists(e2e):
    rows = []
    items = list(csv.DictReader(open(e2e), delimiter="\t"))
    for key, runs in group(items, lambda r: (r["build"], r["threads"], r["preset"])).items():
        rows.append([*key, len(runs), med_min([r["load_s"] for r in runs], 1), med_min([r["revisit_s"] for r in runs], 1),
                     runs[0]["frames"], med_min([r["gen_s"] for r in runs], 1), med_min([r["rtf"] for r in runs]),
                     med_min([r["ms_frame"] for r in runs], 0), "/".join(r["wer"] for r in runs),
                     "/".join(r["requests"] for r in runs), med_min([r["loadavg"] for r in runs], 1)])
    table("Chrome, e2e/run.mjs (S1)", ["build", "threads", "preset", "runs", "first-visit load s", "revisit s", "frames",
                                        "gen s", "RTF", "ms/frame", "WER", "requests", "load avg"], rows)

# Browser in-session
rows = []
for path in sorted(glob.glob(os.path.join(out, "browser", "*.json"))):
    s = json.load(open(path))
    for sentence, runs in sorted(group(s["runs"], lambda r: r["sentence"]).items()):
        rows.append([s["build"], s["threads"], s["preset"], sentence, runs[0]["frames"], runs[0]["audio_s"],
                     med_min([r["gen_s"] for r in runs]), med_min([r["rtf"] for r in runs]),
                     med_min([r["ms_per_frame"] for r in runs], 0), med_min([r["first_audio_s"] for r in runs], 1),
                     f"{s['loadMs'] / 1000:.1f}"])
if rows:
    table("Chrome, benchmarks/browser.mjs (one session)", ["build", "threads", "preset", "sentence", "frames", "audio s",
                                                             "gen s", "RTF", "ms/frame", "first audio s", "load s"], rows)

# Node
node = os.path.join(out, "node", "node.tsv")
if os.path.exists(node):
    rows = []
    items = list(csv.DictReader(open(node), delimiter="\t"))
    for build, runs in group(items, lambda r: r["build"]).items():
        rows.append([build, len(runs), med_min([r["load_s"] for r in runs], 1), runs[0]["frames"], runs[0]["audio_s"],
                     med_min([r["gen_s"] for r in runs], 1), med_min([r["rtf"] for r in runs]),
                     med_min([r["ms_frame"] for r in runs], 0), med_min([r["first_audio_s"] for r in runs]),
                     med_min([r["loadavg"] for r in runs], 1)])
    table("Node, scripts/node-smoke.mjs (S2)", ["build", "runs", "load s", "frames", "audio s", "gen s", "RTF", "ms/frame",
                                                 "first audio s", "load avg"], rows)

# Kokoro
rows = []
for path in sorted(glob.glob(os.path.join(out, "kokoro", "*.json"))):
    s = json.load(open(path))
    if s.get("error"):
        print(f"\nkokoro {s['dtype']}: error {s['error'][:200]}")
        continue
    for sentence, runs in sorted(group(s["runs"], lambda r: r["sentence"]).items()):
        rows.append([s["dtype"], sentence, runs[0]["audio_s"], med_min([r["gen_s"] for r in runs]),
                     med_min([r["rtf"] for r in runs], 3), f"{s['loadMs'] / 1000:.1f}", s.get("threads")])
if rows:
    table("Kokoro.js in Chrome (a different, 82M model)", ["dtype", "sentence", "audio s", "gen s", "RTF", "load s (network)",
                                                            "hardwareConcurrency"], rows)
