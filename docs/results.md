# Results

Machine: Apple M4 Pro (14 cores, 24 GB), macOS, Chrome headless shell (Playwright 1.64.1),
Node 22.16. The machine was shared with other heavy processes during the browser runs (load
average ≈ 8), so browser timings vary by about ±20% between identical runs; ranges are given.
"Frame" = one 80 ms latent frame; RTF = generation time / audio duration (below 1 is faster than
real time). Default sentence for timings: "The evening wind drifted down from the mountain pass,
and the distant clouds slowly turned gold." (US_James, seed 1895, `mix` bundle).

## Correctness against the PyTorch reference

`just parity` (`ming parity dist/model-f32 fixtures/goldens/ref_f32_A.safetensors`): the f32
bundle, each component teacher-forced on the fp32 reference's own inputs
(`scripts/dump_goldens.py`, sentence A, 158 prompt tokens, 69 frames).

| Check | Result |
|---|---|
| prompt token ids | identical (158) |
| `z` after prefill | rel L2 3.5e-5 |
| `z` per frame (LLM decode, 68 frames) | worst rel 1.3e-5 |
| flow-head latent (10 steps, CFG 2, injected noise, 69 frames) | worst rel 2.3e-4 |
| stop decision | 69/69 frames agree |
| aggregator output | worst rel 2.9e-5 |
| streaming vocoder vs one-shot reference decode (243,432 samples) | rel 2.8e-6, max abs 2.5e-6 |

Quantized bundles, same check: `mix` — `z` prefill 0.56, decode 0.30, flow 0.065, aggregator
0.0095, vocoder 0.15; `q7` — 0.11, 0.047, 0.065, 0.0095, 0.026 (see decisions D4).

## Intelligibility and voice

`scripts/asr_check.py` (faster-whisper small, int8) on native `ming say` output, `mix`:

| Text | Voice | WER |
|---|---|---|
| sentence A | US_James | 0.000 (also `q7`, and the wasm builds in Node) |
| "In 1998 the company sold 2,500 units, … forty percent …" | US_James | 0.000 |
| three-sentence paragraph (2 fragments, 18 s) | US_James | 0.000 |
| "Glaciers move slowly …" | US_Emma / voice design / free-form | 0.000 / 0.000 / 0.000 |
| "Please remember to water the plants … three o'clock." (in Chrome, every build) | US_James | 0.000 |

`scripts/voice_check.py` (CAM++ cosine to the reference's sentence-A audio): f32 bundle
0.72 / 0.72 / 0.64, `q7` 0.76 / 0.76 / 0.65, `mix` 0.67 / 0.73 / 0.65 (sentence A; sentence B
at seeds 7 and 11).

## Speed

### Native (`target/release/ming --threads N bench dist/model`)

| ms per frame | 1 thread | 8 threads |
|---|---:|---:|
| LLM decode | 6.6 | 1.6 |
| flow head, quality / balanced / fast | 72.9 / 55.3 / 37.9 | 13.2 / 9.9 / 6.7 |
| aggregator | 3.0 | 0.6 |
| vocoder | 18.9 | 3.6 |
| **frame, quality** | **101.4 (RTF 1.27)** | **19.0 (RTF 0.24)** |
| frame, balanced / fast | 83.8 / 66.4 | 15.7 / 12.6 (RTF 0.16) |
| prefill, 158 tokens | 799 | 137 |

History (8 threads, quality): first working version 36.3 ms/frame → two-column kernel 24.7 →
SIMD quantizer + vectorized activations 19.0 (decisions D7, D8). The PyTorch fp32 reference on
the same machine: RTF 2.46.

### WebAssembly in Node, one thread (`node scripts/node-smoke.mjs dist/model --build …`)

| Build | ms/frame | RTF |
|---|---:|---:|
| `web-single` (SIMD128) | 361 | 4.55 |
| `web-single-relaxed` | 183 | 2.36 |

### Chrome (`PORT=8787 node e2e/run.mjs [--build …] [--threads …] [--preset …]`)

Sentence "Please remember to water the plants on Tuesday, and call the dentist about the
appointment at three o'clock." (5.0–5.7 s of audio):

| Build | Threads | RTF |
|---|---:|---:|
| `web-threaded-relaxed` (the default in Chrome) | 8 | 0.49–0.67 |
| `web-threaded` | 8 | 0.88 |
| `web-single-relaxed` | 1 | 2.14 |
| `web-single` | 1 | 4.03 |

Presets with `web-threaded-relaxed`: quality 0.49–0.67, balanced 0.53–1.05, fast 0.62–0.64
(the spread is the shared machine; see D12). Every run: 0 network requests during synthesis,
WER 0.000.

Loading (localhost): 850 MB of parts in 1.3–2.3 s on a first visit (OPFS write included),
0.8–1.0 s on a revisit (all parts from OPFS, BLAKE3-checked by the runtime).

## Sizes

`mix` bundle: 571 tensors in 15 weight parts (838.6 MB) plus the tokenizer part (11.4 MB) (`q7`:
1,209 MB). Runtime modules after `wasm-opt`: 2.76 (`web-single`), 2.81 (`web-single-relaxed`),
2.85 (`web-threaded`), 2.88 MB (`web-threaded-relaxed`).
