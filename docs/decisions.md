# Decisions

Each entry: the decision, the evidence, and the part of [`plan.md`](plan.md) it implements or
changes. Measurements are in [`results.md`](results.md).

## D1. Target the dense Ming-3.0-tts-0.5B, not Ming-3.0-tts-tiny

`inclusionAI/Ming-3.0-tts-tiny` is a 7.9B-total / 1.3B-active MoE (128 experts, 18 GB bf16);
the "0.5B" model is the separate `inclusionAI/Ming-3.0-tts-0.5B` (Qwen2.5-0.5B backbone,
1.48B parameters in the checkpoint, about 1.06B on the synthesis path). Chosen by the user when
asked. Revision pinned: `e5bd8da3…`, sha256 of `model.safetensors` checked by
`scripts/fetch-checkpoint.sh`.

## D2. Built-in, described and free-form voices only (no voice cloning)

Tasks `IP`, `VOICE_DESIGN` and `FREEFORM` run with `use_zero_spk_emb=True`: the speaker row of
the prompt is zeros and the voice comes from the instruction caption. So the browser never needs
the VAE encoder (419M parameters), the speaker projection or the CAM++ ONNX model. Cloning from
a reference clip would need all three; it is not built.

## D3. Own runtime, no tensor framework (plan §0)

`ming-core` is f32 activations, two quantized weight formats, a handful of kernels and one
thread team. The flow head runs 10 steps × 8 blocks of 6-row matmuls per 80 ms frame: dispatch
and fork/join overhead would dominate in a framework (SealedKB D22/D25), and candle's wasm
pitfalls (SealedKB D3, D14) do not arise. Parity with the PyTorch reference (f32 bundle,
teacher-forced, `just ming-parity`): every component within 2.3e-4 relative L2, identical prompt ids
and stop decisions.

## D4. The `mix` bundle is the default (plan §3, confirmed)

4-bit LLM layers and VAE decoder, 7-bit flow head, aggregator and embedding table, 850 MB.
The LLM's conditioning vector moves a lot under 4-bit weights (prefill `z` relative L2 0.56 vs
0.11 with 7-bit; the fake-quantized PyTorch model gives the same numbers, 0.561 / 0.118, and
shows activation quantization alone costs 0.04). Every LLM layer contributes (0.07–0.26 when
quantized alone) and MSE-optimal 4-bit scales do not help (0.59). What matters downstream does
not move: WER 0.000 on every test sentence, and CAM++ voice similarity to the reference within
the run-to-run spread of the f32 bundle (mix 0.67/0.73/0.65, q7 0.76/0.76/0.65, f32
0.72/0.72/0.64). `PROFILE=q7 just ming-model` gives the 1.2 GB all-7-bit bundle.

## D5. Activations quantized per block of 32, weights' zero point folded into the dot

Per-token activation scales cost 3× the deviation of per-block ones (plan §3.1). The
`− zero_point · Σa` correction of a block is seeded into the first accumulator lane before the
dot instructions (`i32x4.relaxed_dot_i8x16_i7x16_add_s`, NEON `sdot`), so it costs nothing in
the inner loop. 7-bit weights keep the relaxed-SIMD operand in its defined range on every engine.

## D6. The thread team: one broadcast per component, spin barriers inside (plan §6.4)

Model code is SPMD over `team::Ctx`: static column splits, `SharedBuf`s written in disjoint
ranges, barriers between dependent phases; matmul inputs are normalized and quantized
redundantly by every thread rather than synchronized. Native scaling 1 → 8 threads: 5.6× per
frame. The same code with a one-thread team is the single-threaded wasm build.

## D7. Kernel: two output columns per pass, unchecked inner loads

Loading each activation block once for two weight rows and replacing per-load bounds checks by
one upfront assertion took the flow head from 158 to 93 ms/frame on one native core and wasm
relaxed from 345 to 218 ms/frame (`ming bench`, `node-smoke.mjs`).

## D8. SIMD activation quantization, vectorizable `exp`

The redundant per-thread work (RMSNorm, activation quantization) is serial time in a threaded
run; GELU's `tanhf` was 6% of wasm time. A SIMD quantizer (wasm `f32x4.nearest`, NEON `frintn`:
the same ties-to-even rounding), RMSNorm through the SIMD dot product and a branch-free
polynomial `exp` (≤ 4e-7 relative, tested against libm) for GELU (`x·σ(2u)`) and SiLU took the
Chrome threaded build from RTF 0.68 to 0.49 and native 8 threads from 24.7 to 19.1 ms/frame;
f32 parity unchanged.

## D9. The vocoder streams, exactly

The VAE backbone's sliding window (64) and the linear upsampling's one-frame lookahead let the
decoder run per latent frame with a 128-slot KV ring and exact overlap-add bookkeeping; its
output equals the reference's one-shot decode of the whole utterance (relative L2 2.8e-6).
Trailing-silence trimming is done while streaming by holding back silent 0.1 s chunks (same
result as `trim_trailing_silence`).

## D10. Text frontend: the reference's, minus the FST normalizer

`text.rs` ports `cut_text_by_semantic_length`, `normalize_numbers` (inflect's wording, checked
against inflect's outputs) and `normalize_text_kit`, with the reference's segmentation per task
(`MingTTS.prepare_text`): built-in voices cut into fragments of ≤ 50 semantic units; voice
design and free-form instructions keep the text whole (one utterance, so the designed voice
cannot drift between fragments); a target language other than Chinese/English cuts but never
normalizes. Differences from the reference, all deliberate:

- The pynini `TalkerTN` stage is not ported (no pynini here, and it is a large FST). In its
  place a hyphen inside a word becomes a space: otherwise the whitelist step turns
  "ninety-eight" into "ninety，eight" (a pause).
- `chinese_punc_en2cn` in the reference has a non-f-string regex (`r'([{ch_puncs}])\1+'`), so for
  any text containing CJK it collapses doubled letters `c h p u n s` ("Mississippi" →
  "Misisipi") and keeps "……"; the port implements the evident intent instead.
- Numbers above 10²¹ are left as digits (inflect continues to sextillion and beyond).

## D11. Static Node dev server instead of an axum server (plan §4)

The server only serves files with COOP/COEP and Range, so `scripts/serve.mjs` (no
dependencies) replaces the planned `ming serve`.

## D12. Speed presets

`quality` (reference sampler) is the default; `balanced` (CFG only for `t ∈ [0.05, 0.8)`) and
`fast` (7 steps + the same interval) are offered as measured in plan §3. Natively they cut the
flow head by 25% and 50%; in Chrome the gain is smaller and within run-to-run noise on this
(shared) machine, because with fewer rows per pass the threaded matmuls become bandwidth-bound.

## D13. A longer frame budget than the reference

The reference stops a fragment at `min(1000, 512 − prompt)` frames (its static 512-position
cache); the LLM cache here holds 1024 positions, so the budget is `min(1000, 1024 − prompt)`.
Built-in voices are cut into short fragments, where the stop head ends every fragment long before
either bound; whole-text voice design and free-form utterances can run longer than the
reference's ≈ 28 s. If the stop head never fires the fragment runs to the budget (Stop ends it).

## D14. Plan items not built

- Prefix KV snapshot and `warm()` (plan §5.3, §6.3, §13.2): each fragment prefills its whole
  prompt (137 ms natively for 158 tokens, a few hundred ms in Chrome), which is small next to
  the frames that follow.
- Last-block trimming of the flow head (§6.3, §11 #5, ≈ 8% of the flow head): not needed for
  faster-than-real-time in Chrome; the first remaining speed item.
- Spin-then-futex barrier (§6.4): the barrier spins, then `yield_now` (a no-op in wasm). Between
  component calls the Rayon workers park; inside a call phases are microseconds apart. The page
  never starts more compute workers than `navigator.hardwareConcurrency`.
- 3 GiB maximum memory (§4): the threaded builds reserve 2 GiB, enough for both bundles
  (`mix` 0.85 GB, `q7` 1.2 GB) plus the tokenizer and activations.
- `kbench`, goldens inside `cargo test`, `node-smoke --check` (§8): the kernels are tested
  against an f64 reference in `cargo test`, the model against the goldens by `ming parity`, and
  the wasm builds end to end by ASR in Node and Chrome.
- Separate `storage.js` / `audio.js` modules (§7): the loader and player are small enough to
  live in `worker.js` and `app.js`.

## D15. The generic parts live in fusor-ai

The kernels, the thread team, the Qwen2 stack (its dimensions now a `Shape`), the bundle format
and store (the tokenizer is a generic "blob" part), the safetensors reader and bundle writer,
the browser loader and compute pool, the runtime build script, the import check and the dev
server moved to [fusor-ai](https://github.com/fusor-rs/fusor-ai), pinned here as the git
submodule `vendor/fusor-ai` so the crates and the JavaScript move together. ming-core keeps what
is Ming's: the flow head and aggregator, the vocoder, the synthesis loop, the prompt and the
text frontend. The move changed no result: recompiled `mix` and `f32` bundles are
byte-identical (every part hash), `make parity` prints the same numbers, every runtime build
passes the import check, and the Chrome e2e passes (RTF 0.52 threaded, 2.05 single-threaded,
WER 0, 0 requests during synthesis).

## D16. Ming lives in fusor-ai, in layers

D15's split kept the model in its own repository. With a second model in view, the model moved
into fusor-ai instead, and both repositories' code was sorted into layers that only depend
downwards (`AGENTS.md`, "Repository"): primitives (`team`, `kernels`, `bundle`, `compile`),
blocks (`blocks`: the Qwen2 stack, the DiT block stack, flow-matching integration, the
streaming inverse STFT), models (`ming`, `ming-compile`) and hosts (`ming-runtime`,
`ming-cli`). A piece is a block only when a second, named model would use it unchanged: the DiT
stack (F5-TTS-style flow heads), the guided Euler integrator with sway sampling and the
Vocos-style iSTFT qualify; the aggregator's wiring, the EPSS knots, the silence gate, the
presets, the prompts and the text frontend stay in `ming`. `scripts/check-layers.mjs` enforces
the direction and keeps tensor names and fixed dimensions out of `blocks`.

Each component loads itself (`bundle::Load`, with a prefix and a shape), so `ming::model` is a
table of contents. Blocks therefore depend on `bundle`, which the first layering draft forbade;
the alternative, loaders in every model for every block, repeats the tensor layout of each block
in each model. The matmul is generic over an `arch::Backend` (relaxed SIMD, SIMD128, NEON,
scalar), so the f64 reference test runs the scalar backend beside the native one.
`ming::pipeline::Stage` lists the stages that `ming bench` and `ming parity` walk.

The move changed no result: `ming-compile` builds the same rename table (570 tensors, same
order), `just ming-parity` prints the same numbers, and native speed is within run-to-run noise
of the code before the move (best of six alternating runs: flow head 15.0 ms per frame on both).


## D17. The app lives in ming-wasm, the model in fusor-ai

D16 put everything Ming in fusor-ai. A toolkit should not own one model's product, so the
product came back here: the runtime surface, the CLI, the page, the goldens, the end-to-end and
smoke checks, these documents and the releases (the CLI archives, the page with its runtimes
and `install.sh`). The model (`ming`) and its compiler (`ming-compile`) stay in fusor-ai as
libraries on its blocks, published to crates.io with the toolkit; this repository consumes them
through the `vendor/fusor-ai` submodule. The split changed no result: every dependency resolves
to the same version, `just parity` prints the same numbers, and a bundle compiled here is
byte-identical (every part hash) to the one compiled before the split.

## D18. fusor-ai from crates.io

Since fusor-ai v0.1.0 is on crates.io, the model and the toolkit come from there as
`fusor-ai-*` crates, pinned exactly. The `vendor/fusor-ai` submodule stays, at the same
release's tag, for what crates.io does not carry: the browser loader and compute pool, the
runtime build script, the import check and the dev server. Switching changed no result:
`just parity` prints the same numbers.

