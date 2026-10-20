# Benchmarks

How fast Ming-3.0-tts-0.5B speaks on one machine, in every runtime this project has: the PyTorch reference on the CPU and on the Apple GPU, the native `ming` CLI, the four WebAssembly builds in Chrome and the two single-threaded ones in Node, and, as a point of reference from a different and much smaller model, Kokoro.js in the same Chrome. Everything here was measured on 2026-10-20 on the machine below; nothing is quoted from elsewhere.

**Real-time factor (RTF)** is generation time divided by the duration of the audio produced: an RTF of 0.5 means five seconds of speech took two and a half seconds to compute, and anything below 1 is faster than real time. The model produces speech in 80 ms frames, so RTF is also ms per frame divided by 80. Alongside RTF, the tables give the **time to first audio** (how long you wait before playback can start), the **load time** of the model, the **memory** of the process, and the **word error rate (WER)** of faster-whisper's transcription of the output against the input, so that speed is never bought with unintelligible audio.

Three sentences were used across every runtime, all with the built-in voice US_James:

| | Sentence |
|---|---|
| S1 | Please remember to water the plants on Tuesday, and call the dentist about the appointment at three o'clock. (the page's end-to-end check sentence) |
| S2 | The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold. |
| S3 | Glaciers move slowly, carving valleys over thousands of years while the rivers below carry the silt to the sea. |

Every number is the **median of three runs**, with the minimum in parentheses, unless stated otherwise. Each produces 5–7 s of audio.

## The machine

| | |
|---|---|
| Computer | Apple M4 Pro, 14 cores (10 performance, 4 efficiency), 24 GB |
| OS | macOS 27.0.1 (Darwin 27.0.0) |
| Native CLI | `ming` 0.1.0, Rust 1.99.0, release build, fusor-ai 0.1.0 |
| Browser | Chrome headless shell 156.0.8078.4 (Playwright 1.64.1), Node 22.16.0 |
| PyTorch | torch 2.6.0, transformers 4.46.3, the reference at commit e999c39 with a two-line patch so that autocast follows the model's device and dtype instead of hard-coding `cuda` and bf16 (without it the reference does not run in fp32 or on MPS) |
| Kokoro.js | kokoro-js 1.2.1 (transformers.js, onnxruntime-web, wasm backend), model `onnx-community/Kokoro-82M-v1.0-ONNX` |
| ASR | faster-whisper 1.2.1, model `small`, int8 (`scripts/asr_check.py`) |

The machine was in use by other processes throughout. Every benchmark ran one at a time under `nice -n 19`, with at most 8 compute threads, and the 1-minute load average is recorded next to every measurement in the raw data (`benchmarks/out/`). The load average stayed between 5 and 11 during most runs; where it went higher the row says so.

## Headline

Sentence S1, preset `quality` (10 CFG Euler steps in the flow head, which is what the reference runs), mix bundle (4-bit language model and VAE, 7-bit flow head, 850 MB) unless noted.

| Runtime | Threads | RTF | First audio | Model load | Memory | WER |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch reference, fp32, CPU | 10 (default) | 2.60 (2.57) † | 14.3 s (no streaming) | 2.8 s | 6.1 GB | 0 |
| PyTorch reference, bf16 autocast, CPU | 10 (default) | 4.01 (3.88) † | 22.0 s | 0.6 s | 2.8 GB | 0 |
| PyTorch reference, fp32, CPU | 8 | 3.00 (2.60) † | 16.5 s | 7.7 s | 5.1 GB | 0 |
| PyTorch reference, bf16 autocast, CPU | 8 | 3.42 (3.34) † | 18.8 s | 0.6 s | 2.6 GB | 0 |
| PyTorch reference, fp32, Apple GPU (MPS) | – | 2.34 (2.34) | 12.4 s | 7.2 s | 5.1 GB | 0 |
| PyTorch reference, bf16, Apple GPU (MPS) | – | 2.27 (2.26) | 11.8 s | 3.6 s | 0.6 GB | 0 |
| `ming` native CLI | 1 | 1.48 (1.47) | 1.67 s | 0.61 s | 1.27 GB | 0 |
| `ming` native CLI | 4 | 0.42 (0.42) | 0.47 s | 0.64 s | 1.27 GB | 0 |
| `ming` native CLI | 8 | **0.28 (0.28)** | **0.30 s** | 0.64 s | 1.27 GB | 0 |
| `ming` native CLI, preset `fast` | 8 | 0.20 (0.19) | 0.23 s | 0.62 s | 1.27 GB | see note |
| Chrome, `web-threaded-relaxed` | 8 | **0.47 (0.45)** | **0.3 s** | 1.8 s first visit, 1.0 s revisit | 1.37 GB | 0 |
| Chrome, `web-threaded` (SIMD128) | 8 | 0.76 (0.73) | – | 1.6 s / 1.0 s | – | 0 |
| Chrome, `web-single-relaxed` | 1 | 2.15 (2.07) | 2.2 s | 1.6 s / 1.1 s | 1.21 GB | 0 |
| Chrome, `web-single` (SIMD128) | 1 | 3.97 (3.97) | – | 1.8 s / 1.0 s | – | 0 |
| Node, `web-single-relaxed` (S2) | 1 | 2.48 (2.46) | 2.47 s | 0.9 s | – | 0 |
| Node, `web-single` (S2) | 1 | 4.20 (4.19) | 4.22 s | 0.9 s | – | 0 |
| *Kokoro.js, Kokoro-82M fp32, Chrome (a different, 6× smaller model)* | ORT default | 0.39 (0.39) | = total, 2.5 s | 11.0 s from the network | – | 0 |
| *Kokoro.js, Kokoro-82M q8, Chrome (a different, 6× smaller model)* | ORT default | 0.71 (0.71) | = total, 4.7 s | 4.5 s from the network | – | 0 |

Memory is the peak resident set size of the process (`/usr/bin/time -l`, `getrusage`), or for Chrome the sum of the resident sets of every process of the browser profile right after synthesis. Model load times are with the files in the OS file cache; the PyTorch fp32 load varied between 2.8 and 7.7 s from run to run (the bf16 checkpoint is read and converted to fp32). † marks rows taken under a load average of 10–15, see the PyTorch section. "First audio" for the reference is the whole generation: it decodes the waveform after the last frame, while `ming` streams it per frame.

In short: on this machine the native CLI on 8 threads speaks 3.5× faster than real time and 9× faster than the PyTorch reference on the same CPU; the page in Chrome with threads and relaxed SIMD runs at about twice real time, 5× faster than the reference; without worker threads the page is 2–4× slower than real time. The Apple GPU does not help the reference much here: MPS lands at about the same RTF as the CPU (the model's loop is one token and one frame at a time, which leaves a GPU little to parallelize) while using a fraction of the memory in bf16.

## PyTorch reference

`benchmarks/pytorch_bench.py`: `MingTTS.synthesize(task="IP", controls={"ip": "US_James"}, use_zero_spk_emb=True, raw_text=True)`, text normalization off, seed 1895, 10 flow steps with CFG 2. Frames are counted by wrapping the flow head's `sample`; the reference's frame counts differ from `ming`'s by a few frames per sentence (its noise is drawn differently), which is why durations differ slightly.

| Precision | Device | Threads | Sentence | Frames | Audio | Generation | RTF | ms / frame | Load | Peak RSS | Load avg. |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | CPU | 10 | S1 | 71 | 5.5 s | 14.3 s (14.1) | 2.60 (2.57) | 202 (199) | 2.8 s | 6.1 GB | 13.6 |
| fp32 | CPU | 10 | S2 | 64 | 5.0 s | 11.8 s (11.6) | 2.35 (2.32) | 184 (181) | | | 13.5 |
| fp32 | CPU | 10 | S3 | 77 | 6.1 s | 17.2 s (16.0) | 2.82 (2.63) | 223 (208) | | | 15.0 |
| fp32 | CPU | 8 | S1 | 71 | 5.5 s | 16.5 s (14.3) | 3.00 (2.60) † | 232 (202) | 7.7 s | 5.1 GB | 8.6 |
| fp32 | CPU | 8 | S2 | 64 | 5.0 s | 13.3 s (13.2) | 2.66 (2.63) † | 208 (206) | | | 14.1 |
| fp32 | CPU | 8 | S3 | 77 | 6.1 s | 22.3 s (22.0) | 3.66 (3.61) † | 290 (286) | | | 14.9 |
| bf16 | CPU | 10 | S1 | 71 | 5.5 s | 22.0 s (21.3) | 4.01 (3.88) † | 310 (300) | 0.6 s | 2.8 GB | 13.4 |
| bf16 | CPU | 10 | S2 | 68 | 5.4 s | 20.5 s (20.1) | 3.80 (3.73) † | 302 (296) | | | 15.0 |
| bf16 | CPU | 10 | S3 | 80 | 6.4 s | 23.9 s (23.2) | 3.73 (3.63) † | 298 (290) | | | 14.2 |
| bf16 | CPU | 8 | S1 | 71 | 5.5 s | 18.8 s (18.4) | 3.42 (3.34) † | 265 (259) | 0.6 s | 2.6 GB | 12.5 |
| bf16 | CPU | 8 | S2 | 68 | 5.4 s | 18.1 s (17.9) | 3.34 (3.32) † | 265 (264) | | | 11.3 |
| bf16 | CPU | 8 | S3 | 80 | 6.4 s | 21.2 s (21.1) | 3.31 (3.30) † | 265 (264) | | | 10.7 |
| fp32 | MPS | – | S1 | 67 | 5.3 s | 12.4 s (12.4) | 2.34 (2.34) | 185 (185) | 7.2 s | 5.1 GB | 5.1 |
| fp32 | MPS | – | S2 | 68 | 5.3 s | 12.9 s (12.5) | 2.43 (2.37) | 189 (184) | | | 6.1 |
| fp32 | MPS | – | S3 | 76 | 6.0 s | 14.8 s (14.5) | 2.46 (2.41) | 194 (190) | | | 6.8 |
| bf16 | MPS | – | S1 | 66 | 5.2 s | 11.8 s (11.8) | 2.27 (2.26) | 179 (178) | 3.6 s | 0.6 GB | 10.6 |
| bf16 | MPS | – | S2 | 66 | 5.2 s | 11.7 s (11.6) | 2.25 (2.23) | 177 (176) | | | 7.4 |
| bf16 | MPS | – | S3 | 75 | 5.9 s | 13.3 s (13.3) | 2.25 (2.25) | 177 (177) | | | 6.5 |

† Taken while another process held the load average at 10–15 (the fp32 10-thread rows too, at 13–15). The rows marked † were run twice, both times under that load; the table shows the second pass, and the first (load average 12–33, `benchmarks/out/pt-heavy-load/`) gave RTF 3.6 for fp32 at 8 threads, 3.9 for bf16 at 10 and 3.2 for bf16 at 8. The per-run minimum is the better guide: fp32 at 8 threads reached 2.60 on S1, the same as the 10-thread median, so on an idle machine the CPU reference is RTF ≈ 2.3–2.6 in fp32 and ≈ 3.3 in bf16. An earlier single fp32 8-thread run at load 7 gave 2.73.

bf16 autocast on the CPU is slower than fp32 on this chip: the measured time went up in every bf16 CPU configuration, and the model's activations are rounded to bf16 at every autocast boundary without a kernel that is faster for it here. On the Apple GPU, bf16 cuts the memory to a tenth and gives the lowest RTF the reference reaches on this machine, but still above 2.

## `ming` native CLI

### Per-stage cost of a frame (`ming bench`)

Each stage timed in isolation over 20 calls after a warm-up, on sentence S2's prompt (158 tokens). The frame total is the sum of the stages; `say` is a little slower than this total because it also pays the prefill and the first-frame latency.

| ms per frame, mix bundle | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| LLM decode (one token) | 6.9 | 2.2 | 1.6 |
| flow head, `quality` (10 steps) | 74.9 | 20.8 | 13.7 |
| flow head, `balanced` | 57.4 | 15.9 | 11.0 |
| flow head, `fast` | 38.4 | 10.6 | 7.0 |
| aggregator | 2.9 | 0.9 | 0.7 |
| vocoder | 18.4 | 5.5 | 3.5 |
| **frame, `quality`** | **103.0 (RTF 1.29)** | **29.5 (RTF 0.37)** | **19.6 (RTF 0.24)** |
| frame, `balanced` | 85.5 (RTF 1.07) | 24.5 (RTF 0.31) | 16.8 (RTF 0.21) |
| frame, `fast` | 66.5 (RTF 0.83) | 19.2 (RTF 0.24) | 12.9 (RTF 0.16) |
| prefill, 158 tokens | 818 | 227 | 160 |

| ms per frame, q7 bundle | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| LLM decode | 7.6 | 2.5 | 2.4 |
| flow head, `quality` / `balanced` / `fast` | 75.1 / 55.8 / 37.5 | 21.0 / 16.2 / 10.7 | 13.7 / 10.6 / 7.7 |
| aggregator | 2.7 | 0.9 | 0.8 |
| vocoder | 21.4 | 5.5 | 7.1 |
| **frame, `quality`** | **106.7 (RTF 1.33)** | **29.8 (RTF 0.37)** | **24.0 (RTF 0.30)** |
| frame, `balanced` / `fast` | 87.5 / 69.1 | 24.9 / 19.4 | 20.8 / 18.0 |
| prefill, 158 tokens | 908 | 225 | 209 |

The flow head is 70% of a frame on one thread and still 65–70% on eight. Going from 4 to 8 threads buys 1.5× on the mix bundle and less on q7, whose 7-bit vocoder and decode do not scale past 4 threads here (the 8-thread q7 vocoder was measured slower than the 4-thread one, at load average 6–8).

### End to end (`ming say`)

`--seed 1895`, peak RSS from `/usr/bin/time -l`. Load is the time to read and hash the bundle (files in the OS cache).

| Bundle | Threads | Preset | Sentence | Frames | Audio | RTF | ms / frame | First audio | Load | Peak RSS |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| mix | 1 | quality | S1 | 69 | 5.4 s | 1.48 (1.47) | 116 | 1.67 s | 0.61 s | 1.27 GB |
| mix | 1 | quality | S2 | 66 | 5.1 s | 1.50 (1.50) | 116 | 1.35 s | 0.63 s | 1.27 GB |
| mix | 1 | quality | S3 | 75 | 5.9 s | 1.47 (1.45) | 115 | 1.39 s | 0.63 s | 1.28 GB |
| mix | 1 | balanced | S1 / S2 / S3 | 66 / 65 / 88 | 5.2 / 5.1 / 7.0 s | 1.25 / 1.25 / 1.20 | 98 / 98 / 95 | 1.28 / 1.27 / 1.38 s | 0.62 s | 1.27 GB |
| mix | 1 | fast | S1 / S2 / S3 | 54 / 67 / 88 | 4.3 / 5.3 / 6.8 s | 1.03 / 0.99 / 0.99 | 82 / 79 / 76 | 1.26 / 1.16 / 1.26 s | 0.63 s | 1.27 GB |
| mix | 4 | quality | S1 | 69 | 5.4 s | 0.42 (0.42) | 33.0 | 0.47 s | 0.64 s | 1.27 GB |
| mix | 4 | quality | S2 | 66 | 5.1 s | 0.43 (0.43) | 33.5 | 0.38 s | 0.64 s | 1.27 GB |
| mix | 4 | quality | S3 | 75 | 5.9 s | 0.43 (0.42) | 33.5 | 0.39 s | 0.63 s | 1.27 GB |
| mix | 4 | balanced | S1 / S2 / S3 | 66 / 65 / 88 | 5.2 / 5.1 / 7.0 s | 0.36 / 0.36 / 0.35 | 28.7 / 28.5 / 27.8 | 0.36 / 0.35 / 0.39 s | 0.63 s | 1.27 GB |
| mix | 4 | fast | S1 / S2 / S3 | 54 / 67 / 88 | 4.3 / 5.3 / 6.8 s | 0.30 / 0.29 / 0.29 | 23.9 / 23.0 / 22.7 | 0.36 / 0.33 / 0.36 s | 0.63 s | 1.27 GB |
| mix | 8 | quality | S1 | 69 | 5.4 s | **0.28 (0.28)** | 22.0 | **0.30 s** | 0.64 s | 1.27 GB |
| mix | 8 | quality | S2 | 66 | 5.1 s | 0.29 (0.28) | 22.5 | 0.24 s | 0.63 s | 1.27 GB |
| mix | 8 | quality | S3 | 75 | 5.9 s | 0.29 (0.29) | 22.6 | 0.26 s | 0.63 s | 1.27 GB |
| mix | 8 | balanced | S1 / S2 / S3 | 66 / 65 / 88 | 5.2 / 5.1 / 7.0 s | 0.24 / 0.26 / 0.23 | 19.0 / 20.1 / 18.4 | 0.24 / 0.25 / 0.25 s | 0.63 s | 1.27 GB |
| mix | 8 | fast | S1 / S2 / S3 | 54 / 67 / 88 | 4.3 / 5.3 / 6.8 s | 0.20 / 0.20 / 0.20 | 15.9 / 15.6 / 15.1 | 0.23 / 0.22 / 0.23 s | 0.62 s | 1.27 GB |
| q7 | 1 | quality | S1 | 63 | 4.9 s | 1.54 (1.51) | 120 | 1.42 s | 0.83 s | 1.63 GB |
| q7 | 4 | quality | S1 | 63 | 4.9 s | 0.43 (0.42) | 33.1 | 0.40 s | 0.82 s | 1.63 GB |
| q7 | 8 | quality | S1 | 63 | 4.9 s | 0.30 (0.29) | 23.0 | 0.26 s | 0.84 s | 1.63 GB |
| q7 | 8 | quality | S2 / S3 | 67 / 75 | 5.3 / 5.9 s | 0.30 (0.28) / 0.35 (0.31) | 23.6 / 27.6 | 0.26 / 0.24 s | 0.85 s | 1.63 GB |
| q7 | 8 | balanced | S1 / S2 / S3 | 66 / 71 / 78 | 5.1 / 5.5 / 6.1 s | 0.27 / 0.29 / 0.26 | 20.5 / 22.4 / 20.6 | 0.26 / 0.23 / 0.22 s | 0.87 s | 1.63 GB |
| q7 | 8 | fast | S1 / S2 / S3 | 65 / 68 / 75 | 5.2 / 5.4 / 6.0 s | 0.29 (0.22) / 0.34 (0.28) / 0.21 (0.21) | 23.1 / 27.1 / 16.6 | 0.23 / 0.25 / 0.21 s | 0.89 s | 1.63 GB |

The `say` RTF is above the `bench` frame total by 10–15% at 8 threads: the prefill (160 ms) and the first frame are paid once per sentence, and the per-frame loop also moves audio out. The first audio arrives after the prefill plus the streaming vocoder's look-ahead of a few frames: 0.3 s on 8 threads (0.16 s prefill), 1.7 s on one (0.82 s prefill). The q7 8-thread `fast` rows have a wide spread (min 0.22, median 0.29 on S1): those runs coincided with a load average of 7.5–8.

**Intelligibility.** One WAV per row above was transcribed; every one has WER 0 except the mix bundle's `fast` preset on S1 with seed 1895, which stops two words early (54 frames, "…about the appointment." with "at three o'clock" missing, WER 0.167, identically at 1, 4 and 8 threads, since the seed fixes the noise). Five other seeds of the same case gave WER 0 four times and one garbled phrase (seed 5, WER 0.222). `quality` and `balanced` had WER 0 for every sentence, bundle, thread count and seed tried; `fast` trades some reliability for 30% less time per frame.

## WebAssembly in Chrome

`e2e/run.mjs`: a fresh Chrome profile per run, the page served from localhost, the 850 MB bundle fetched and written to OPFS on the first visit and read back on the revisit; sentence S1; the WER of the page's downloaded WAV; the count of network requests made during synthesis (always 0). Three runs per row.

| Build | Threads | Preset | First-visit load | Revisit load | Frames | Generation | RTF | ms / frame | WER | Load avg. |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `web-threaded-relaxed` | 8 | quality | 1.8 s (1.8) | 1.0 s | 69 | 2.6 s (2.4) | **0.47 (0.45)** | 38 (35) | 0 / 0 / 0 | 7.0 |
| `web-threaded` | 8 | quality | 1.6 s (1.6) | 1.0 s | 66 | 4.0 s (3.8) | 0.76 (0.73) | 61 (58) | 0 / 0 / 0 | 7.1 |
| `web-single-relaxed` | 1 | quality | 1.6 s (1.6) | 1.1 s | 69 | 11.6 s (11.2) | 2.15 (2.07) | 168 (162) | 0 / 0 / 0 | 9.8 |
| `web-single` | 1 | quality | 1.8 s (1.8) | 1.0 s | 66 | 20.7 s (20.6) | 3.97 (3.97) | 314 (312) | 0 / 0 / 0 | 11.5 |
| `web-threaded-relaxed` | 8 | balanced | 1.6 s (1.6) | 1.0 s | 66 | 2.3 s (2.2) | 0.45 (0.42) | 35 (33) | 0 / 0 / 0 | 10.7 |
| `web-threaded-relaxed` | 8 | fast | 1.6 s (1.3) | 1.0 s | 63 | 2.0 s (1.8) | 0.41 (0.35) | 32 (29) | 0 / 0 / 0 | 10.3 |

Relaxed SIMD (which gives the kernels a single-instruction int8 dot product) is worth 1.6× over plain SIMD128 with threads and 1.85× without; worker threads are worth 4.6× over one thread. The `balanced` and `fast` presets gain less in the browser than natively (0.47 → 0.45 → 0.41, against 0.28 → 0.24 → 0.20) because those rows ran under a higher load average and because the browser's per-frame overhead outside the flow head (messages between the engine worker and the page, scheduling the audio) does not shrink with the preset.

In the same Chrome, one session (`benchmarks/browser.mjs`: load once, then every sentence three times; first audio is the page's own "in N s" of the first progress line that carries audio, 0.1 s resolution):

| Build | Threads | Sentence | Frames | Audio | RTF | ms / frame | First audio | Memory (all Chrome processes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `web-threaded-relaxed` | 8 | S1 | 69 | 5.4 s | 0.43 (0.42) | 33 (32) | 0.4 s (0.3) | 1.37 GB |
| `web-threaded-relaxed` | 8 | S2 | 67 | 5.3 s | 0.42 (0.41) | 33 (33) | 0.3 s (0.3) | |
| `web-threaded-relaxed` | 8 | S3 | 79 | 6.2 s | 0.42 (0.40) | 33 (32) | 0.3 s (0.3) | |
| `web-threaded-relaxed` | 4 | S1 | 69 | 5.4 s | 0.65 (0.65) | 51 (51) | 0.5 s (0.5) | – |
| `web-single-relaxed` | 1 | S1 (one run) | 69 | 5.4 s | 2.06 | 161 | 2.0 s | 1.21 GB |

The page's time to first audio is the same as the native CLI's on 8 threads, 0.3 s; the page then starts playback once enough audio is buffered to stay ahead of generation.

### WebAssembly in Node, one thread (`scripts/node-smoke.mjs`, sentence S2)

| Build | Load | Frames | Audio | Generation | RTF | ms / frame | First audio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `web-single` (SIMD128) | 0.9 s | 67 | 5.3 s | 22.3 s (22.2) | 4.20 (4.19) | 332 | 4.22 s |
| `web-single-relaxed` | 0.9 s | 67 | 5.3 s | 13.1 s (13.0) | 2.48 (2.46) | 196 | 2.47 s |

Node's V8 is the same engine as Chrome's, so the single-threaded figures are within 15% of the browser's (S2 here against S1 there). First audio is the single-threaded prefill (about 2.5 s) plus the vocoder's look-ahead frames at 200–330 ms each. WER 0 for both.

### Sizes

| | |
|---|---:|
| mix bundle (`dist/model`) | 811 MiB (850 MB) |
| q7 bundle (`dist/model-q7`) | 1,154 MiB (1.2 GB) |
| f32 bundle (`dist/model-f32`, parity checks only) | 4,055 MiB (4.25 GB) |
| runtime module, `web-single` / `web-single-relaxed` | 2.93 / 3.00 MB |
| runtime module, `web-threaded` / `web-threaded-relaxed` | 3.01 / 3.03 MB |
| page (`app/dist`, without the runtimes) | 12 MB |

## Kokoro.js, for scale

[Kokoro.js](https://github.com/hexgrad/kokoro/tree/main/kokoro.js) runs **Kokoro-82M**, a different and about six times smaller text-to-speech model (no language model, no diffusion; fixed voices from a voice bank). It is included so that readers have a familiar in-browser reference point, not as a like-for-like comparison. `benchmarks/kokoro/index.html` loads kokoro-js 1.2.1 from jsdelivr and the model from Hugging Face on every run, on the `wasm` backend of onnxruntime-web with its default thread count, voice `af_heart`; one warm-up, then each sentence three times (`benchmarks/kokoro.mjs`).

| Precision | Sentence | Audio | Generation | RTF | Load (from the network) |
|---|---|---:|---:|---:|---:|
| fp32 | S1 | 6.5 s | 2.54 s (2.52) | 0.39 (0.39) | 11.0 s |
| fp32 | S2 | 6.3 s | 2.42 s (2.40) | 0.38 (0.38) | |
| fp32 | S3 | 7.3 s | 2.83 s (2.81) | 0.39 (0.39) | |
| q8 | S1 | 6.5 s | 4.65 s (4.65) | 0.71 (0.71) | 4.5 s |
| q8 | S2 | 6.3 s | 4.47 s (4.43) | 0.71 (0.71) | |
| q8 | S3 | 7.2 s | 5.22 s (5.13) | 0.72 (0.71) | |

Kokoro produces a whole sentence at once, so its first audio is its total time (2.4–2.8 s for fp32, against 0.3 s for the ming-wasm page). Its q8 model, the one its README suggests for the browser, is slower than fp32 on this machine's wasm backend. All six WAVs transcribe with WER 0. So, on this machine, the 0.5-billion-parameter Ming in ming-wasm's own runtime (RTF 0.47) runs at about the speed of the 82-million-parameter Kokoro through onnxruntime-web (RTF 0.39–0.71).

## How to reproduce

From a checkout with the bundles compiled (`just build`, `PROFILE=q7 just model`) and the e2e dependencies installed (`cd e2e && npm install && npx playwright install chromium`). The raw outputs of every command (TSV, JSON, WAV, logs) land in `benchmarks/out/`, and `python3 benchmarks/summarize.py` prints the median/min tables from them.

```sh
# PyTorch reference: ../ming-05b-experiment holds Ming-3.0-tts/ (commit e999c39, patched as above),
# Ming-3.0-tts-0.5b/ (the checkpoint) and .venv/ with the reference's requirements.
../ming-05b-experiment/.venv/bin/python benchmarks/pytorch_bench.py --reference ../ming-05b-experiment \
    --dtype float32 --repeat 3 --out benchmarks/out/pt                  # CPU, torch's default threads
... --dtype float32 --threads 8      # CPU, 8 threads
... --dtype bfloat16                 # bf16 autocast
... --dtype bfloat16 --device mps    # Apple GPU

# Native CLI: ming bench per bundle and thread count, ming say per preset and sentence, 3 repeats
just native
benchmarks/native.sh                                                     # mix and q7, 1/4/8 threads
BUNDLES=dist/model THREADS=8 PRESETS=fast SENTENCES=S1 benchmarks/native.sh   # a subset
nice -n 19 target/release/ming --threads 8 bench dist/model
nice -n 19 /usr/bin/time -l target/release/ming --threads 8 say dist/model "Please remember…" -o out.wav --preset quality --seed 1895

# Chrome: a fresh profile per run (first-visit load, synthesis, WER, revisit), 3 repeats per configuration
just page
REPEAT=3 PORT=8795 benchmarks/browser-e2e.sh "web-threaded-relaxed 8 quality" "web-threaded 8 quality" \
    "web-single-relaxed 1 quality" "web-single 1 quality" "web-threaded-relaxed 8 balanced" "web-threaded-relaxed 8 fast"
PORT=8795 node e2e/run.mjs --build web-threaded-relaxed --threads 8 --preset quality   # one run by hand

# Chrome, one session: first audio per sentence and the browser's memory
PORT=8795 node benchmarks/browser.mjs --build web-threaded-relaxed --threads 8 --preset quality --repeat 3

# Node, the single-threaded builds
REPEAT=3 benchmarks/node.sh

# Kokoro.js (needs the network for the library and the model)
PORT=8796 node benchmarks/kokoro.mjs --dtype fp32
PORT=8796 node benchmarks/kokoro.mjs --dtype q8

# Intelligibility of any WAV
../ming-05b-experiment/.venv-asr/bin/python scripts/asr_check.py --text "Please remember…" out.wav

python3 benchmarks/summarize.py
```

## Caveats

- **One machine, shared and busy.** Everything was measured on a single Apple M4 Pro that other processes were using at the same time (load average 5–11 for most rows, up to 15 for the PyTorch fp32 rows). Medians of three runs are reported with the minimum beside them; the minimum is the better estimate of what an idle machine does. Expect ±10–20% between repeats of the browser rows, and different absolute numbers on other CPUs.
- **At most 8 threads.** The CLI and the page default to 8 compute threads on this 14-core machine; the reference used torch's default of 10. The 1-thread rows show the cost of the computation itself, free of parallel scaling.
- **Presets are not the same output.** `balanced` and `fast` run fewer flow steps than the reference's 10; only `quality` matches the reference's sampling. The `fast` preset lost words on one sentence and seed (see the native section).
- **Different sentences, slightly different lengths.** The reference and `ming` draw their noise differently, so the same sentence comes out a few frames longer or shorter in each; RTF and ms per frame normalize this, generation times do not.
- **The PyTorch reference is not optimized for this.** It runs eager PyTorch with the model's own loop, no `torch.compile`, no KV-cache tricks beyond the model's, on a checkout patched only to run on CPU/MPS in fp32. It is the correctness reference, and its speed is the starting point, not a competitor's best effort.
- **Kokoro.js is a different model.** Six times smaller, a different architecture, a different runtime (onnxruntime-web) and a different output sample rate (24 kHz against 44.1 kHz). The rows are there for scale only.
- **Load times are warm.** The bundles and the checkpoint were in the OS file cache, and the browser loads are from localhost: the first visit on a real connection is dominated by the 850 MB download.
- **Memory for Chrome** is the resident set of every process of the headless profile after synthesis, which includes Chrome's own baseline; the model itself is 850 MB in the worker's wasm memory.
