"""Time the PyTorch reference (Ming-3.0-tts-0.5B) the way `ming say` is timed.

Loads the reference the way `MingTTS` does, synthesizes each sentence `--repeat` times (voice
US_James, task IP, zero speaker embedding, raw text, 10 CFG Euler steps: `ming say --preset
quality`), counts the 80 ms frames by wrapping the flow head's `sample`, and prints one line per
run plus a JSON summary: load time, generation time, audio seconds, RTF, ms per frame, frames,
and the process's peak RSS. The reference decodes the waveform after the last frame, so its first
audio arrives when generation ends.

Usage (from the ming-wasm checkout, with the reference's venv):
  ../ming-05b-experiment/.venv/bin/python benchmarks/pytorch_bench.py \
      --reference ../ming-05b-experiment --dtype float32 --threads 8 --repeat 3 --out benchmarks/out/pt
  ... --dtype bfloat16            # autocast bf16
  ... --device mps                # Apple GPU
`--reference` holds `Ming-3.0-tts/` (commit e999c39, with a small patch so autocast follows the
model's device and dtype instead of hard-coding cuda/bf16) and the checkpoint `Ming-3.0-tts-0.5b/`.
"""
import argparse
import json
import os
import resource
import sys
import time

SENTENCES = {
    "S1": "Please remember to water the plants on Tuesday, and call the dentist about the appointment at three o'clock.",
    "S2": "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.",
    "S3": "Glaciers move slowly, carving valleys over thousands of years while the rivers below carry the silt to the sea.",
}
SEED = 1895


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, help="torch.set_num_threads (default: torch's default)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--sentences", nargs="*", default=list(SENTENCES))
    ap.add_argument("--voice", default="US_James")
    ap.add_argument("--out", help="directory for WAVs and the JSON summary")
    args = ap.parse_args()

    reference = os.path.abspath(args.reference)
    if args.out:
        args.out = os.path.abspath(args.out)
    repo = os.path.join(reference, "Ming-3.0-tts")
    sys.path.insert(0, repo)
    os.chdir(repo)
    os.environ["MING_DTYPE"] = args.dtype

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    from ming_tts import MingTTS

    t0 = time.perf_counter()
    tts = MingTTS(os.path.join(reference, "Ming-3.0-tts-0.5b"), device=args.device, text_normalization=False)
    load_s = time.perf_counter() - t0
    tag = f"pytorch-{args.dtype}-{args.device}-t{torch.get_num_threads()}"
    print(f"[{tag}] torch {torch.__version__}, threads={torch.get_num_threads()}, dtype={tts.model.dtype}, "
          f"device={args.device}, load={load_s:.2f}s", flush=True)

    flow = tts.model.flowloss
    frames = {"n": 0}
    orig_sample = flow.sample

    def counted_sample(*a, **k):
        frames["n"] += 1
        return orig_sample(*a, **k)

    flow.sample = counted_sample

    runs = []
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    for key in args.sentences:
        text = SENTENCES[key]
        for i in range(args.repeat):
            frames["n"] = 0
            torch.manual_seed(SEED)
            wav_path = os.path.join(args.out, f"{tag}-{key}-{i}.wav") if args.out else None
            t1 = time.perf_counter()
            wav = tts.synthesize(task="IP", text=text, controls={"ip": args.voice}, use_zero_spk_emb=True,
                                 raw_text=True, output_path=wav_path)
            if args.device != "cpu":
                getattr(torch, args.device).synchronize()
            gen_s = time.perf_counter() - t1
            audio_s = wav.shape[-1] / tts.sample_rate
            run = {"sentence": key, "run": i, "frames": frames["n"], "audio_s": round(audio_s, 3),
                   "gen_s": round(gen_s, 3), "rtf": round(gen_s / audio_s, 3),
                   "ms_per_frame": round(1000 * gen_s / max(1, frames["n"]), 1), "load_avg": os.getloadavg()[0]}
            runs.append(run)
            print(f"[{tag}] {key} run {i}: {frames['n']} frames, {audio_s:.2f}s audio in {gen_s:.2f}s: "
                  f"RTF {run['rtf']:.3f}, {run['ms_per_frame']:.1f} ms/frame, load avg {run['load_avg']:.1f}", flush=True)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS
    summary = {"tag": tag, "torch": torch.__version__, "dtype": args.dtype, "device": args.device,
               "threads": torch.get_num_threads(), "load_s": round(load_s, 2), "peak_rss_mb": round(peak_rss_mb),
               "runs": runs}
    print(f"[{tag}] peak RSS {peak_rss_mb:.0f} MB")
    if args.out:
        with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
    else:
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
