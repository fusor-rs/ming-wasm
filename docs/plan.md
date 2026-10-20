# Ming-3.0-tts-0.5B in the browser: engineering plan

The plan the implementation started from. Its repository layout and commands (§4) are those of
the original standalone repository; [README.md](README.md) has the current ones, and
[decisions.md](decisions.md) records where the implementation departed from the plan.

Target: `inclusionAI/Ming-3.0-tts-0.5B` (HF revision `e5bd8da31ecf1b9276cdbdda821fa891068f1368`,
GitHub reference `inclusionAI/Ming-3.0-tts` @ `e999c39`) running entirely on the CPU in Rust compiled to
WebAssembly, with WASM threads (shared memory + Rayon workers, the fusor recipe SealedKB already uses).
The server serves static files only. Done = a page where the user types text, picks a built-in voice,
presses Generate and gets playable audio, with zero network requests after the model has loaded.

Everything in §1–§2 was verified against the reference source and the fp32 PyTorch run on this machine
(Apple M4 Pro, 14 cores, 24 GB). Experiments in §3 were run with
`.cache/experiments/harness.py` (commands and raw logs in `.cache/experiments/logs/`).

---

## 0. Decisions in one screen

| Topic | Decision |
|---|---|
| Framework | **No candle.** A lean own runtime: `Vec<f32>` activations, two quantized weight formats, ~8 kernels, one static thread team. Reasons: SealedKB D3/D4/D14/D15/D22/D23 (candle 0.11 does not build for wasm+SIMD, 0.10.2 is unsound with owned buffers, opens one Rayon region per activation row, gemm re-packs the KV cache, `num_cpus`=1 in wasm, dequantizes the 151,936-row embedding to f32), k-quant super-blocks of 256 do not divide hidden 896, and this model's shapes are tiny and fixed (the flow head is 10 × 8 blocks of 6-row matmuls per 80 ms frame): framework dispatch and fork/join overhead would dominate. |
| Weight formats | `Q7B32` (7-bit values stored as `u8 = q + 64`, `q ∈ [−63, 63]`, blocks of 32 along K, one f32 scale per block) for the flow head, aggregator and embedding table; `Q4B32` (ggml Q4_0 layout: 16 B of nibbles `n = q + 8`, `q ∈ [−8, 7]`, f32 scale) for the LLM and VAE decoder **if §3 confirms quality**, else `Q7B32` everywhere. Activations are int8 symmetric per block of 32 with a per-block `Σa` (bias correction). 7-bit weights exist so the weight is the unsigned 7-bit operand of `i32x4.relaxed_dot_i8x16_i7x16_add_s` (Chrome 114+): one instruction per 16 MACs, implementation-defined only for values ≥ 128, and `pmaddubsw` on x86 cannot saturate (2·127·128 < 32767). Same bytes serve the SIMD128 (`extmul` + `dot_i16x8`) and native (`sdot`) kernels. |
| Threads | Rayon on the fusor recipe (`nightly-2025-11-15`, `-Z build-std`, `+atomics,+bulk-memory`, shared imported memory, page-owned workers through a copy of SealedKB's `thread-host.js`, `wasm-bindgen-rayon 1.3.0`). **One `rayon::broadcast` per component call** (flow-head sample, LLM token, aggregator, VAE frame) with static partitions and spin-then-futex barriers inside; never a Rayon region per matmul. Four builds: `{single, threaded} × {simd128, relaxed}`, picked by the loader like SealedKB D25. |
| Model compiler | Native Rust (`ming compile`) reading `model.safetensors` (bf16→f32, mmap), quantizing with the **same** `ming-core` quant code the runtime dequantizes, dropping `audio.encoder.*`, `model.lm_head`, `spk_head`, `campplus`, precomputing all constants (§5.3), writing hashed shards + `manifest.json`. Python is used only to dump golden tensors from the reference. |
| Speed knobs | Default sampler preset decided by §3: flow steps and CFG schedule from the measured table; everything exact-algebraic (last-block trimming, shared history embedding, constant time/uncond embeddings) is always on. |
| Vocoder | Streaming: the fp32 reference's one-shot VAE decode **does** apply the 64-position sliding window (measured: position 399 attends to exactly 336..399), and frame-by-frame decode with a KV cache equals one-shot to 1.3e-10. So a 64-slot KV ring, a 2-position upsample lag and the reference's iSTFT carry buffers give exact streaming audio ~160 ms behind the AR loop. |
| Audio out | PCM f32 44.1 kHz pushed per frame to the page; WebAudio playback starts when 1.5 s is buffered or generation is done; WAV download always. |
| Toolchain pins | stable `1.99.0` + `wasm32-unknown-unknown` (rust-toolchain.toml), `nightly-2025-11-15` with `rust-src` for the threaded builds, `wasm-bindgen 0.2.129`, binaryen `wasm-opt` (SealedKB `scripts/install-tools.sh` pattern into `.tools/`), Node ≥ 22, Playwright for e2e. |

---

## 1. The model, pinned down (verified)

Checkpoint: `model.safetensors`, 2,959,055,504 B, bf16, 1,479,454,868 parameters. `header.json` groups:

| Group | Params | Used in browser |
|---|---:|---|
| `model.model.*` (Qwen2.5-0.5B body) | 494.0M (358M layers + 136.1M `embed_tokens`) | yes; `lm_head` is tied (`_tied_weights_keys`) and never needed: only hidden states are used |
| `flowloss.cfm.model.*` (DiT flow head) | 103.1M | yes |
| `linear_proj_audio.*` (Aggregator) | 101.8M | yes |
| `audio.decoder.*` (VAE decoder) | 361.1M | yes |
| `audio.encoder.*` | 419.3M | **no** (only for voice cloning from reference audio) |
| `stop_head` (896→2, bias) | 1.8K | yes |
| `spk_head` (192→896, bias) | 173K | **no**: with `use_zero_spk_emb=True` the projected speaker row is zeroed, so the `<|vision_pad|>` row of the input embeddings is a **zero vector** |

### 1.1 Prompt (dense layout, `prompt_contract.build_input_ids`)

Pieces are tokenized **separately** with `add_special_tokens=False` and concatenated (so the runtime must encode piece-wise too). For the IP task with voice `V` the exact sequence (logged by the reference) is:

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n
Please generate speech based on the following description.\n          # DEFAULT_TTS_PROMPT (+ "Target language: X.\n" only if language_id given; we pass none for en/zh)
  speaker_1:<|vision_start|><|vision_pad|><|vision_end|>\n            # speaker block; the <|vision_pad|> embedding row is overwritten with zeros
 Text input:\n                                                        # TEXT_INPUT_PREFIX
{text fragment}
<|im_end|>\n<|im_start|>assistant\n
{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": null, "语速": null, "基频": null, "音量": null, "情感": null, "BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null, "ENV": null, "SNR": null}, "IP": "US_James"}]}<|im_end|>
<audio>
```

- IP and VOICE_DESIGN use the **zh schema** (`tts_tasks.build_task_instruction` defaults `schema="zh"` for everything but DIALECT); `style` fills `风格`, `ip` fills `IP`. FREEFORM uses the raw instruction string + `<|im_end|>`. Key order and `ensure_ascii=False` JSON spacing (`", "` and `": "`) are part of the contract.
- Pinned token ids: `<|endoftext|>` 151643, `<|im_start|>` 151644, `<|im_end|>` 151645, `<|vision_start|>` 151652, `<|vision_end|>` 151653, `<|vision_pad|>` 151654, `<audioPatch>` 151705, `<audio>` 151706.
- English built-in voices: `US_Emma`, `US_James` (default), `US_Stella`, `UK_George`, `UK_Charlotte`; Chinese recommended: `灵小柔`, `灵楷` (full list in `cookbooks/instruction_en.md`).
- Text pipeline before the prompt (`ming_tts.MingTTS.prepare_text`): strip markdown backticks → `add_period` → `cut_text_by_semantic_length(max 50 semantic units, min_tail 5)` → per fragment `normalize_numbers` (English) → FST TN (pynini, **not ported**) → `normalize_text_kit`. Each fragment is synthesized separately and trailing silence is trimmed per fragment (0.1 s frames, mean |x| ≤ 1e-3 is silent, keep last non-silent frame + 0.3 s), then concatenated.

### 1.2 LLM (Qwen2, `llm_config`)

hidden 896, 24 layers, 14 heads / 2 KV heads (GQA group 7), head_dim 64, intermediate 4864, SiLU gated MLP, RMSNorm eps 1e-6 (`x · rsqrt(mean(x²)+eps) · w`, statistics in f32), q/k/v **with bias**, o_proj without, `rope_theta` 1e6, no sliding window (`use_sliding_window: false`), tied embeddings, vocab 151,936. RoPE = HF `rotate_half`: pairs `(i, i+32)`, `inv_freq_i = 1e6^(−2i/64)`, `q' = q·cos + rotate_half(q)·sin`. Positions `0..L−1` for the prompt, then `+1` per frame. The conditioning vector `z` is `hidden_states[-1][:, -1]` which in transformers 4.46 is taken **after** `model.norm` (the final RMSNorm). Prefill of the whole prompt, then one token per frame whose input embedding is the aggregator output. The reference uses a 512-position cache (`MAX_CACHE_LEN`); decode budget `min(1000, 512 − prompt_len)`.

### 1.3 Flow head (`flowloss.cfm.model`: DiT) and sampler

- `x_embedder` Linear 64→1024 (bias); `c_embedder.cond_embedder` Linear 896→1024 (bias); `t_embedder`: `SinusPositionEmbedding(256)` → Linear 256→1024 → SiLU → Linear 1024→1024. Sinusoid: `half=128`, `freq_i = exp(−i · ln(10000)/127)`, `e = 1000·t·freq`, `emb = [sin(e), cos(e)]`.
- Sequence of **3 tokens**: `[t_emb + c_emb(z), x_emb(latent_history), x_emb(x)]`, positions 0,1,2, no mask (`attn_mask_enabled: false` is swallowed by `**kwargs`; `mask=None`), **non-causal** SDPA, scale `1/8`.
- 8 `DiTBlock`s, hidden 1024, 16 heads × 64, pre-RMSNorm (eps 1e-6) → attention (`to_q/to_k/to_v/to_out.0` all with bias) → residual → RMSNorm → MLP `ff.0.0` 1024→4096 (bias) → **GELU tanh approximation** → `ff.2` 4096→1024 (bias) → residual. `final_layer`: RMSNorm → Linear 1024→64 (bias). Output = last token only (`[:, -1:]`).
- RoPE is **x-transformers**: interleaved pairs `(2i, 2i+1)`, `inv_freq_i = 10000^(−2i/64)` (shipped as `rotary_embed.inv_freq`, F32[32]); `freqs = repeat_interleave(p·inv_freq, 2)`; `rotate_half(x)[2i] = −x[2i+1]`, `[2i+1] = x[2i]`; `out = x·cos + rotate_half(x)·sin`, scale 1 (no xpos). Applied to q and k.
- CFG batch of 2: rows `[cond, uncond]` with `c_uncond = 0` ⇒ `y_uncond = t_emb + c_embedder.bias` (constant per timestep). `v = pred + (pred − null)·2.0`.
- Timesteps (`get_epss_timesteps` then sway with coef −1): knots `k/32` for `k ∈ {0,2,4,6,8,12,16,20,24,28,32}` (10 steps; presets exist for 5, 6, 7, 12, 16), warped by `t ← t + (−1)·(cos(πt/2) − 1 + t) = 1 − cos(πt/2)`. 10-step knots: `0, .0048, .0192, .0431, .0761, .1685, .2929, .4444, .6173, .8049, 1`. 6-step: `0, .0048, .0192, .0431, .0761, .2929, 1` (the last step is one Euler jump of 0.707). Euler: `y ← y + v(t_s, y)·(t_{s+1} − t_s)`; the SDE term is `sigma·sqrt(temperature)·…` = 0 at temperature 0, but the reference still draws `sde_noise` (`randn(steps,1,1,64)`) **after** the initial `noise = randn(1,1,64)`; we do not reproduce torch's RNG (parity tests inject recorded noise).
- Per frame: `latent_history` is zeros (1×1×64) at step 0, then the previous frame's latent. `patch_size = history_patch_size = 1`.

### 1.4 Aggregator (`linear_proj_audio`)

`x_embedder` 64→1024 (bias) on the sampled latent; sequence `[word_embedder.weight[0] (cls), x_emb]`, positions 0,1; same `DiTBlock` × 8 and RoPE as the flow head; `final_layer` RMSNorm → Linear 1024→896 (bias); output = position 0 (cls) only → the next LLM input embedding (896). Only the cls row needs `final_layer`.

### 1.5 Stop rule

`stop_head(z)` = 2 logits; stop iff `softmax[1] > 0.5` ⇔ `logit[1] > logit[0]`, and `step > 3` (zero-based; so at least 5 frames). The frame sampled at the stopping step **is** emitted.

### 1.6 VAE decoder (`audio.decoder`) and iSTFT

- `fc1` Linear 64→896 (bias) per frame → linear upsample ×4 (`align_corners=False`): for frame `i` and `j ∈ 0..3`, `src = i + (2j−3)/8`, so positions `4i+j` = `0.375·h[i−1] + 0.625·h[i]`, `0.125·h[i−1] + 0.875·h[i]`, `0.875·h[i] + 0.125·h[i+1]`, `0.625·h[i] + 0.375·h[i+1]`, with `h[−1] := h[0]` and `h[T] := h[T−1]` (edge clamp). Streaming needs **one frame of lookahead** (positions `4i+2, 4i+3` wait for frame `i+1`; `4i, 4i+1` are final immediately).
- Qwen2Model backbone: same shape as the LLM (896, 24 layers, 14/2 heads, 4864, RMSNorm 1e-6, rope 1e6, q/k/v bias), `vocab_size 1` (`embed_tokens` [1,896] unused: inputs are embeddings), **causal with sliding window 64** (`use_sliding_window: true, max_window_layers: 0`): position `i` attends to `j ∈ (i−64, i]`, i.e. exactly 64 keys. Verified on the CPU/SDPA path (mask built by `_prepare_4d_causal_attention_mask_with_cache_position`). No final projection other than `norm`.
- `ISTFTHead`: `out` Linear 896→3530 (bias). Split the 3530 into `mag = exp(x[:1765])` **clipped to max 1e2**, `phase = x[1765:]`; `S = mag·(cos φ + i sin φ)` (1765 = 3528/2 + 1 bins). `irfft(S, n=3528, norm="backward")` (1/n scaling), multiply by the window, overlap-add with hop 882, "same" padding (`pad = (3528−882)/2 = 1323` trimmed from both ends), divide by the window envelope `Σ w²` built the same way (assert > 1e-11). The **window is the checkpoint tensor `audio.decoder.head.istft.window`** (BF16[3528]): a Hann window rounded to bf16, which the fp32 reference upcasts, so ship that tensor as f32 rather than recomputing `hann`. `T` positions → exactly `882·T` samples at 44.1 kHz; one AR frame = 4 positions = 3528 samples = 80 ms.
- Streaming semantics to mirror (`ISTFT.__buffer_process`): first chunk drops `pad` leading samples, carry `buffer_len = 2646` tail samples (audio and envelope) into the next chunk, last chunk drops `pad` from the end. Measured: streaming one frame per call equals one-shot (max abs diff 1.3e-10).

### 1.7 Compute per 80 ms frame (MACs, exact from shapes)

| Component | Rows | MACs | Notes |
|---|---:|---:|---|
| LLM, one token | 1 | 357.8M | weight-bandwidth bound (24 × 14.9M) |
| Flow head, one step | 6 (3 tokens × CFG 2) | 604M | 100.7M params/row; 10 steps = **6.04G** (75% of the frame) |
| Aggregator | 2 | 201M | |
| VAE decoder, 4 positions | 4 | 1,431M + 12.6M head | attention ≤ 64 keys: negligible |
| Stop head, embedders, RoPE, norms, iSTFT | | < 20M | 4 real FFTs of 3528 = 2³·3²·7² |
| **Total (10 steps)** | | **8.0 GMAC = 16 GFLOP** | realtime needs 200 GFLOPS |
| Total (6 steps) | | 5.6 GMAC | 140 GFLOPS |
| Total (6 steps, CFG on 3 of them) | | 4.7 GMAC | 118 GFLOPS |

Native PyTorch fp32 reference on this machine (orchestrator's `profile_parts.py`, 14 threads): 13.77 s for 5.60 s of audio, **RTF 2.46**; flow head 138.5 ms/call (71%), LLM 23.4 ms/token, aggregator 15.0 ms, VAE decode 445 ms once. Our baseline run (`harness.py gen --tag baseline`): sentence A 69 frames, 5.4 s audio, 13.4 s, RTF 2.48; sentence B 70 frames, 5.5 s, RTF 2.62.

---

## 2. Lessons from SealedKB applied here

1. **Pin the engine, or drop it.** candle 0.11 does not compile for wasm32+SIMD (D3); 0.10.2 has a use-after-free in `QStorage::from_data` (D14) and one Rayon region per activation row (D22). We keep SealedKB's *kernels and recipe* (relaxed/SIMD128 dot products, fusor threads, four builds, OPFS loader, push-style part loading, import check) and drop the tensor framework.
2. **Quantization layout must fit the shapes** (D1: 896 is not a multiple of 256). Our blocks are 32 wide, and every K here (896, 1024, 4096, 4864) is a multiple of 32; 64-wide inputs (latent) stay f32.
3. **Fork/join is the decode-time enemy**: 141 regions × ~45 µs per token was 60% of an 8-thread decode token (D25 profile; `optimizations.md` §6.1 estimates 74 → 110 tok/s from one region per token). This model has ~13× more matmul calls per frame than a Qwen3 decode token, so the static team with barriers is not optional; it is the design.
4. **Relaxed SIMD is a 2–3× kernel win** (Q4_K 18.7 → 39.3 GMAC/s on one core) and must be probed with `WebAssembly.validate` and shipped as a separate build (Safari rejects the whole module).
5. **Direct attention loops, never gemm with re-packing** (D23); here attention is tiny (≤ 64 keys, 3 tokens) and must be fused into the team phases.
6. **Explicit SIMD in wasm**: LLVM does not vectorise accumulator-array loops for wasm (D25 "explicit SIMD attention loops": 21.4 → 14.2 s first token). Write `core::arch::wasm32` intrinsics for every hot f32 loop (RMSNorm, RoPE, GELU/SiLU, axpy, exp).
7. **Load path**: one copy from shard bytes into the arena (SealedKB cleanup pass: 971 → 941 MiB), ranged parallel fetches (6 × 8 MB), OPFS by content hash, progress per part, model warm-up (prefix prefill) at load (D4 snapshot/rewind).
8. **Deterministic getrandom** (`--cfg getrandom_backend="custom"`) so `tokenizers` does not import `crypto.getRandomValues`; own seeded RNG for sampling (D5).
9. **The threaded module's imports** are only the shared memory, `startWorkers` and the handles compute workers need (D24). Reuse `scripts/check-wasm-imports.mjs` as is.
10. **Measure on the host it affects**: native, Node (single-thread wasm) and headless Chrome (threaded) all run the same Rust; parity across hosts is asserted (SealedKB got the same answer on all three).
11. **8 workers, not 12** (`min(8, cores − 2)`): 12 ran hotter for no gain.
12. **f16 in wasm is slow** (no conversion instructions): scales are f32, activations f32, KV caches f32.

---

## 3. Experiments (fp32 CPU reference, US_James, `cfg=2.0`)

Scripts: `.cache/experiments/harness.py` (modes `gen` and `head`), `asr_wer.py` (faster-whisper small int8 WER vs the input text), `probe_vae_mask.py`, `run_matrix.sh`, `run_matrix2.sh`. Sentences: A "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold." (69 frames), B "Please remember to water the plants on Tuesday, and call the dentist about the appointment at three o'clock." (70 frames), C "In 1998 the company sold 2,500 units, which was a forty percent increase over the previous year." Noise is re-seeded per frame (`manual_seed(1895 + frame)`), so every configuration draws identical noise and the **teacher-forced** mode (`head`) re-runs a modified flow head on the baseline's recorded `(z, history)` per frame and reports the relative L2 error of the frame latent against the baseline (no autoregressive divergence). WER is on the generated audio (autoregressive, end to end).

### 3.1 Results

Teacher-forced flow-head deviation (`head` mode; relative L2 of the frame latent vs the 10-step fp32 baseline, mean over 69/70 frames, sentences A / B) and end-to-end WER (`gen` + `asr_wer.py`, faster-whisper small):

| Configuration | Flow-head row-passes / frame | rel L2 mean A / B | rel L2 p90 A / B | WER A / B (/ C) | Frames A / B | Notes |
|---|---:|---:|---:|---:|---:|---|
| **baseline**: 10 steps, CFG every step, fp32 | 60 | 0 | 0 | 0.000 / 0.000 / 0.000 | 69 / 70 (/ 80) | reference; sentence C 80 frames, RTF 3.13 |
| 7 steps | 42 | 0.227 / 0.202 | 0.354 / 0.291 | 0.000 / 0.000 / **0.143** | 67 / 70 (/ 82) | C: "2,500" spoken as "20,500"; last two steps are 0.29→0.62→1.0 |
| 6 steps | 36 | 0.561 / 0.562 | 0.779 / 0.789 | 0.000 / **0.167** | 68 / **61** | B stopped early: "…about the appointment." (dropped "at three o'clock") |
| 5 steps | 30 | 0.561 / 0.562 | 0.780 / 0.789 | 0.000 / 0.056 | 68 / 70 | same endpoint as 6 steps (identical last jump 0.29→1.0); B: "30 o'clock" |
| CFG only steps 0–4 (`first:5`, t < 0.17) | 45 | 0.327 / 0.293 | 0.449 / 0.387 | 0.000 / 0.000 | 78 / 66 | longer audio (prosody drifts) |
| CFG only steps 0–2 (`first:3`) | 39 | 0.374 / 0.332 | 0.575 / 0.501 | — | — | |
| **CFG only for t ∈ [0.05, 0.8)** (steps 4–8 of 10) | 45 | **0.129 / 0.128** | 0.162 / 0.157 | **0.000 / 0.000 / 0.000** | 68 / 71 (/ 79) | smallest deviation of any non-exact variant; frame counts within ±1 of baseline |
| uncond recomputed every other step (`alt`) | 45 | 0.259 / 0.243 | 0.342 / 0.279 | 0.000 / 0.000 | 68 / 58 | |
| flow head W8A8 (blocks of 32) | 60 | 0.0194 / 0.0192 | 0.025 / 0.023 | — | — | quantization noise floor |
| flow head **W7A8** (blocks of 32) | 60 | 0.0230 / 0.0231 | 0.030 / 0.029 | — | — | 1 bit less costs +0.4 pp: relaxed-SIMD operand is free |
| flow head W4A8 (Q4_0 blocks) | 60 | 0.105 / 0.117 | 0.148 / 0.154 | — | — | as bad as dropping to 7 steps |
| flow head W8 per-row weights + **per-token** A8 (`w8a8r`) | 60 | 0.062 / 0.061 | 0.073 / 0.076 | — | — | per-token activation scales lose 3× vs per-32 blocks |
| flow head W7 per-row weights + per-token A8 (`w7a8r`) | 60 | 0.063 / 0.063 | 0.077 / 0.075 | — | — | same: activations, not weights, are the problem |
| flow head W7 **per-row weights** + per-32 A8 (`w7ra8`) | 60 | 0.027 / 0.026 | 0.035 / 0.034 | — | — | per-row weight scales are nearly free (+0.4 pp) |
| all W7A8 (+emb W8) | 60 | — | — | 0.000 / 0.000 | 67 / 70 | |
| **mix**: LLM+VAE W4A8, flow+agg W7A8, emb W8 | 60 | — | — | 0.000 / 0.000 | 68 / 66 | the shipping profile |
| all W4A8 (+emb W4) | 60 | — | — | 0.000 / 0.000 | 68 / 69 | intelligible even here |
| combo: 6 steps + `first:3` + mix | 27 | ≈0.56 (6-step dominated) | | 0.000 / 0.000 / 0.000 | 81 / 71 / 90 | intelligible but 15% longer, slower prosody |
| 7 steps + CFG t∈[0.05,0.8) (CFG on steps 4–6) | 30 | 0.229 / 0.203 | 0.360 / 0.293 | 0.000 / 0.000 / 0.000 | 66 / 70 (/ 82) | "Fast" preset: half the flow-head row-passes |
| 7 steps + interval + mix quantization | 30 | — | — | 0.000 / 0.000 / **0.143** | 70 / 63 (/ 79) | C again "20,500": 7 steps is fragile on numbers (2 of 4 seven-step runs) |

Reference timings on this machine (PyTorch fp32, 10 threads): 10 steps RTF 2.5–2.6; 6 steps RTF 1.4–1.65; 5 steps 1.3–1.4 (the flow head is ~70% of the time, so steps scale almost linearly). Mask probe (`probe_vae_mask.py`): the VAE backbone runs with `sdpa`, sliding window 64 applied in one-shot decode (64 attended keys from position 399); streaming per frame equals one-shot to 1.3e-10.

### 3.2 What the numbers decide

1. **Quantization is not the quality risk; step count is.** W7A8 on the flow head deviates 2.3% (vs 1.9% for W8A8), and every quantized end-to-end configuration, including all-W4, scored WER 0.00. Dropping to 6 steps deviates 56% and lost the end of sentence B. So: ship the **`mix` profile** (LLM + VAE decoder Q4B32, flow head + aggregator Q7B32, embeddings Q7B32) and keep **10 steps as the default**.
2. **Flow-head W4 is off the table** (10–12%, the same damage as removing 3 steps) — it is also where the compute is, so bandwidth savings would not matter.
3. **CFG interval `t ∈ [0.05, 0.8)` is the cheapest speed knob**: it removes the uncond branch from 5 of the 10 steps (−25% flow-head row-passes) and moves the latent less (0.13) than any step reduction, less than `alt` (0.25) and `first:k` (0.3+). End to end it scored WER 0.000 on A/B/C with frame counts within ±1 of the baseline: it is the **"Balanced"** preset.
4. **7 steps** (0.20–0.23) is the floor for a **"Fast"** preset, combined with the interval (CFG on steps 4–6 of 7: 30 row-passes, half of the baseline's 60). It stays intelligible on prose (WER 0 on A/B in all four 7-step runs) but mispronounced "2,500" as "20,500" in 2 of 4 runs on sentence C, so the UI labels it "fast, may slur numbers" and it is never the default. 6 and 5 steps are not offered: their schedule ends with a single Euler jump from t = 0.29 to 1.0 and the early stop on B shows it.
5. **Teacher-forced relative L2 is the regression metric for the Rust runtime**: a correct implementation of the `mix` bundle must land at ≈ 0.02–0.03 against the fp32 goldens (the quantization floor), and ≤ 5e-3 against goldens dumped from the fake-quantized reference.
6. **Activations need per-32-block scales** (per-token scales triple the deviation to 6.2%), so the dot kernel keeps one f32 scale-and-accumulate per 32-element block (ggml Q8_0 style); weight scales could be per row at +0.4 pp but that saves only 11% of bytes for a second format, so weights stay per block too. The pure-integer inner loop (one float op per row) is only reachable through SmoothQuant-style folding (§11 #8), kept as a measured follow-up.
7. Prosody differences (frame counts 61–81 for the same text) show the AR loop amplifies any sampler change; the UI exposes the preset and the seed, and the default is reference-exact.

---

## 4. Repository layout, toolchain, Makefile

```
ming-wasm/
  Cargo.toml                 workspace; [profile.wasm] opt-level 3, lto fat, codegen-units 1, panic abort
  rust-toolchain.toml        channel 1.99.0, targets wasm32-unknown-unknown, rustfmt + clippy
  .cargo/config.toml         wasm32: -C target-feature=+simd128, --cfg getrandom_backend="custom"
  Makefile                   build / wasm / threaded / native / check / bench / kbench / smoke / e2e / serve / compile / goldens
  crates/
    ming-core/               no I/O, builds for native and wasm: quant formats, kernels (simd128 / relaxed / neon / scalar),
                             Team executor, the three stacks, sampler, vocoder (+ realfft), text frontend, prompt builder, RNG, WAV
    ming-bundle/             manifest (serde), shard reader/writer, BLAKE3, tensor directory; arena loader
    ming-runtime/            wasm-bindgen surface (cdylib) + the same API as an rlib for native hosts
    ming/                    native CLI: compile | say | bench | parity | serve (axum + tower-http with COOP/COEP + Range)
  packages/web/
    src/worker.js            loader: manifest → feature probes → build choice → ranged fetch + OPFS → push parts → generate loop
    src/thread-host.js       SealedKB's host for wasm-bindgen-rayon (copied verbatim, renamed message types)
    src/storage.js           OPFS store (hash-keyed), memory fallback
    src/audio.js             WebAudio streaming sink + WAV encoder
    pages/index.html, app.js, ui.css
    dist/<build>/            generated by `make wasm`
  scripts/
    install-tools.sh         pinned wasm-bindgen 0.2.129 + binaryen into .tools/
    check-wasm-imports.mjs   SealedKB's import check (threaded allowlist, --no-relaxed)
    dump_goldens.py          runs the PyTorch reference (ming-05b-experiment/.venv) → fixtures/goldens/*.npy
    node-smoke.mjs           single-thread wasm in Node: generate sentence A, write WAV, print RTF
    asr_check.py             faster-whisper WER on a WAV (ming-05b-experiment/.venv-asr)
  e2e/run.mjs                Playwright: load page → generate → assert audio, no network after ready → WER
  fixtures/goldens/          dumped tensors (git-ignored if > 5 MB; regenerated by `make goldens`)
  research/plan.md, decisions.md, results.md
```

Makefile targets (all cargo under `nice -n 19`, `CARGO_BUILD_JOBS=4`, never two builds at once):

- `make tools` – rustup target, nightly with rust-src, wasm-bindgen/wasm-opt into `.tools/`.
- `make compile` – `ming compile --safetensors …/Ming-3.0-tts-0.5b --out dist/model --profile mix` (writes shards + manifest; `--profile q7` for all-7-bit).
- `make goldens` – `dump_goldens.py` (needs the reference venv).
- `make native` / `make wasm` (two single builds + verify imports) / `make threaded` (two threaded builds on the nightly) / `make check` (fmt, clippy native+wasm, tests incl. parity against goldens, all four builds, import check).
- `make bench` (native per-component ms + RTF), `make kbench` (kernel GMAC/s native and in Node for both wasm variants, barrier/broadcast cost), `make smoke` (Node single-thread), `make e2e` (Playwright, threaded, `PORT=8787`), `make serve` (static server with `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`, `Accept-Ranges`).

Threaded rustflags (from SealedKB's Makefile, max memory raised): `-C target-feature=+atomics,+bulk-memory -C link-arg=--shared-memory -C link-arg=--max-memory=3221225472 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base`, `-Z build-std=std,panic_abort`, wasm-opt `-O3 --enable-simd --enable-bulk-memory --enable-threads [--enable-relaxed-simd]`. 3 GiB max memory: the all-Q7 bundle is ~1.25 GB in the arena; browsers allow up to 4 GiB for wasm32; the loader allocates the shared `WebAssembly.Memory` itself and reports a clear error if it cannot.

---

## 5. Model compiler (`ming compile`)

### 5.1 Why native Rust

The quantizer and the dequantizer are one `ming-core::quant` module, so the bundle can never drift from what the runtime computes; the compiler doubles as the native parity harness (it can load f32 tensors for A/B tests); no Python in the build. The reference venv is only used by `dump_goldens.py`.

### 5.2 Steps

1. Parse the safetensors header; map the file (`memmap2`, native only); stream tensors one at a time (bf16 → f32).
2. Keep the groups in §1 marked "yes"; drop `audio.encoder.*`, `spk_head.*`, `audio.decoder.decoder.embed_tokens.weight`, both `rotary_embed.inv_freq` (recomputed exactly as `1/10000^(2i/64)`; keep a test that the shipped values match to 1e-7).
3. Fuse at compile time: Qwen2 `q/k/v` weights (and biases) into one `[1152, 896]` matrix per layer, `gate/up` into `[9728, 896]`; DiT `to_q/to_k/to_v` into `[3072, 1024]`. One matmul call, one weight stream, one barrier.
4. Quantize per the profile: `mix` = Q4B32 for LLM layers and VAE decoder layers (+ `head.out`), Q7B32 for flow head, aggregator and `embed_tokens`, f32 for everything with K = 64 or 256 (`x_embedder`, `fc1`, `t_embedder`, `final_layer.linear` of the flow head, `stop_head`, norms, biases, window). `q7` profile = Q7B32 for every K ≥ 896 matrix. Rounding: `round_ties_even` (torch's `round`); scale = absmax/63 (Q7) or ggml Q4_0's `d = max_signed / −8`.
5. Precompute constants (§5.3) with `ming-core`'s own f32 code.
6. Write shards of ≤ 48 MB cut at tensor boundaries, each with its BLAKE3; `manifest.json` lists tensors `{name, dtype, shape, shard, offset, bytes}`, shards `{file, bytes, blake3}`, constants, voice table, tokenizer part, format version and the source revision.
7. Expected size: `mix` ≈ 830 MB (LLM 224, embeddings 153, flow 116, aggregator 114, VAE 226), `q7` ≈ 1.25 GB; tokenizer.json 11.4 MB; the four runtimes ≈ 1–2 MB each.

### 5.3 Precomputed constants (all exact functions of the weights)

- Sampler presets `{5, 6, 7, 10}`: warped knots `t_s`, `dt_s`, `t_emb[s] ∈ R^1024` (= `time_mlp(sinusoid(t_s))`), `y_uncond[s] = t_emb[s] + c_embedder.bias`. (The uncond token of the flow head depends only on the step.)
- RoPE tables: flow head `cos/sin[3][64]`, aggregator `cos/sin[2][64]` (interleaved layout); Qwen2 `inv_freq[32]` for LLM/VAE (tables for positions < 1024 built at load).
- Prompt pieces as token ids: system+user prefix, default prompt, speaker block, `TEXT_INPUT_PREFIX`, turn end, `<audio>`, `<|im_end|>`, and the full instruction block for each shipped voice (IP captions in the zh schema) – produced with the `tokenizers` crate natively, so the runtime encodes only the user's text (and a FREEFORM/VOICE_DESIGN instruction when given).
- iSTFT: the bf16-rounded window as f32, `w²`, and the steady-state envelope `env[m] = Σ_{j<4} w²[m + 882j]` (period 882) plus the exact ramp envelopes for the first and last 3 hops.
- Nothing is precomputed for the LLM prefix KV: the prefix (≈ 37 tokens up to ` Text input:\n`) is prefilled once at load in the browser (~0.1 s) and the cache is rewound to it per fragment (SealedKB D4), which keeps the bundle voice-agnostic.

---

## 6. Runtime design (`ming-core`)

### 6.1 Tensors and memory

- Weights live in one `Vec<u8>` **arena** filled directly by `write_part` (no second copy); a tensor is `(dtype, shape, offset)`. Activations are `&mut [f32]` scratch buffers owned by per-component `Workspace` structs sized once (6×1024, 6×4096, 4×896, 4×4864, 1×3530, …); no allocation in the per-frame path.
- `Q7B32` block = 32 × u8 + f32 scale (36 B); `Q4B32` block = 16 × u8 (low nibble = element `i`, high nibble = element `i+16`, as Q4_0) + f32 scale (20 B). `A8B32` activation block = 32 × i8 + f32 scale + i32 `sum` (40 B). Quantizing a row: per block `s = absmax/127`, `q = round_ties_even(x/s)`, `sum = Σq`.
- Dot of one weight row with one activation row: `Σ_blocks (d_w·d_a) · (Σ u·q − 64·sum)` (Q7) or `(Σ n·q − 8·sum)` (Q4), accumulated in f32 per block (as ggml Q8_0×Q8_0 does). Kernels: `relaxed` (`i32x4.relaxed_dot_i8x16_i7x16_add_s`, 2 ops per 32 MACs), `simd128` (`i16x8.extmul_{low,high}_i8x16_s` + `i32x4.dot_i16x8_s(·, splat(1))`), `neon` (`vdotq_s32`), `scalar`. Multi-row tiles: each weight block is loaded/unpacked once for all M activation rows (M ≤ 6 for the flow head, 4 for the VAE), keeping M i32x4 accumulators in registers; output columns are the parallel dimension.
- KV caches: LLM `[24][2 kv heads][1024 positions][64] f32` for K and V (12.6 MB), position-major per head so decode attention streams contiguous rows; snapshot = remembered length. VAE: 64-slot ring per layer (0.8 MB).

### 6.2 Kernels (all with a wasm SIMD128 variant written in intrinsics)

`qmatmul(M rows)` (the only quantized kernel, 2 weight formats × 3 SIMD flavours), `quantize_rows` (A8B32), `rmsnorm`, `rope_half` (Qwen2) and `rope_interleaved` (DiT), `attention_decode_gqa` (1 query row, ≤ 1024 keys, 7 queries per KV head), `attention_window` (VAE: 4 query positions, ≤ 64 keys, causal), `attention_small` (DiT: 3 or 2 tokens, 16 heads, non-causal — unrolled), `silu_mul`, `gelu_tanh`, `exp_f32x4` (polynomial), `axpy`, `irfft3528` (realfft/rustfft), `overlap_add`. Exact 1/√d scaling, f32 softmax with max subtraction.

### 6.3 The three stacks

- **LLM**: `prefill(tokens) → z`, `decode(embed) → z`. Fused QKV matmul → split → RoPE → write K,V at the position → attention → o_proj → residual → RMSNorm → fused gate/up → SiLU·up → down → residual. Final `norm` applied to the last position only (prefill) or the one row (decode): `z`.
- **Flow head**: `sample(z, history, noise, preset) → latent`. Per frame: `c = c_embedder(z)` (1 row), `h_hist = x_embedder(history)` (shared by both CFG halves and all steps). Per step: rows `[t_emb[s]+c, h_hist, x_emb(x)]` and `[y_uncond[s], h_hist, x_emb(x)]` → 8 blocks on 6 rows; in block 8 only the two `x` rows go through `to_out`, residual, MLP (the other 4 rows are not needed) → `final_layer` on those 2 rows → `v = 3·pred − 2·null` → Euler step. Exact algebra, ~8% fewer MACs.
- **Aggregator**: `next_embed(latent) → e ∈ R^896`: 2 rows through 8 blocks; `final_layer` on the cls row only.
- **Vocoder**: `push(latent) → Option<&[f32]>`: `fc1` → upsample with the one-frame lag → 4 backbone positions through 24 windowed layers (ring KV) → `head.out` → `exp`/clip, `cos`/`sin` → irfft → window → OLA + envelope carry → emits the completed 3528 samples (first call after warm-up emits fewer: `pad` is dropped); `finish()` flushes the tail (clamped lookahead, last-chunk trim).
- **Generator** (`ming-core::tts::Session`): `prepare(request)` → fragments, prompt ids per fragment; `step()` executes one AR frame: `z = llm.decode(e)` (or prefill on the first step of a fragment), `latent = flow.sample(z, hist, rng)`, `stop = stop_head(z)`, `e = agg.next_embed(latent)`, `audio = vocoder.push(latent)`; on stop or budget: `vocoder.finish()`, trim trailing silence, start the next fragment (LLM cache rewound to the prefix snapshot). Returns `{fragment, frame, done, new_samples}`; the host pulls PCM with `take_audio()`.

### 6.4 Threading: the `Team`

`Team::run(n_threads, |ctx| …)` wraps `rayon::broadcast` (rayon-core 1.13, present in the registry; wasm-bindgen-rayon 1.3.0 builds a standard pool) so one broadcast per component call runs the same closure on every pool thread; `ctx.index/ctx.count` select static partitions; `ctx.barrier()` is a sense-reversing barrier that spins ~2000 iterations then parks on a futex (`std::sync::Condvar` works under `+atomics`). Partitions: matmuls by output columns (N/threads, ≥ 16 columns, aligned to heads where attention follows: 1024 → 128 = 2 heads per thread), attention by heads (then KV heads), VAE positions × heads, MLP intermediate by columns with the activation applied in place before the barrier. Row statistics (RMSNorm) and activation quantization are computed **redundantly on every thread** (6 × 1024 elements, ~1 µs) instead of a barrier + broadcast. Expected barriers per frame: ~550 (flow 32 per step, LLM/VAE 96, aggregator 32) at 1–3 µs each on 8 threads = 1–2 ms; broadcasts: 4 per frame. With 1 thread the same code runs without barriers (the single-threaded build). Measured by `make kbench` (`regions`-style micro-benchmark: broadcast cost, barrier cost at 1/4/8 threads, native and wasm).

Why not overlap the vocoder with the AR loop on a second pool: both are CPU-bound and the machine has no idle ALUs when 8 workers run; sequential phases have the same throughput and simpler correctness. Latency to first audio stays ~2 frames.

### 6.5 Hosts

- Native (`ming say`, `bench`, `parity`): Rayon's global pool with `min(8, cores − 2)` threads.
- wasm: `ming-runtime` exports `Engine::new(manifest_json)`, `begin_part/write_part/end_part` (BLAKE3 checked in `end_part`), `warm()` (prefix prefill), `prepare(json)`, `step() → json`, `take_audio() → Float32Array`, `cancel()`, `stats()`, `last_panic()`, `init_thread_pool(n)` (threaded builds). Values cross as strings, numbers and byte/float arrays; no `js_sys`/`web_sys`; import section limited to wasm-bindgen intrinsics (+ the threaded allowances), checked by the script.
- Determinism: `xoshiro256**` seeded from the request (default 1895), Box–Muller normals; the native CLI, Node and Chrome produce bit-identical latents for the same seed on the same build flavour (relaxed and simd128 differ only in f32 accumulation order of block sums; parity tolerances below cover both).

### 6.6 Text frontend (Rust port, English + pass-through Chinese)

`preprocess` (strip backticks), `add_period`, `cut_text_by_semantic_length` (sentence split on `。！？` after mapping `.!?,`, merge to ≤ 50 semantic units = CJK chars + Latin words, `min_tail 5`, protect decimals and `U.S.A.`), `normalize_numbers` (commas, units, £/$, fractions, %, ordinals, versions `v2`/`gpt-4`, decimals, negatives; cardinals ≤ 10¹² to words with inflect's formatting: `"one thousand, nine hundred ninety-eight"`, `andword=""` for plain numbers and `"and"` in ordinals/versions), `normalize_text_kit` (whitelist replacement with `，`, `chinese_punc_en2cn` only when the text contains CJK, `clean_text_formatting`, whitespace collapse), paralinguistic tags protected. The pynini FST (`TalkerTN`) is not ported; `text_normalization=False` is how the published benchmarks ran (`raw_text=True`). Unit tests compare against `number_en.py` outputs for ~40 cases dumped by `dump_goldens.py`.

---

## 7. Web package

- **Worker-owned engine**: `worker.js` runs the loader and the engine in a module Worker (Atomics.wait is allowed there); the page owns the compute workers (`thread-host.js` asks the page to spawn them, SealedKB D24).
- **Load**: fetch `manifest.json` → probe SIMD / relaxed SIMD (`WebAssembly.validate` byte probes from SealedKB) / `crossOriginIsolated` → choose build → allocate shared memory (threaded) → compile runtime → fetch shards with 6 parallel 8 MB range requests, hash-check, store in OPFS by BLAKE3, push into the engine as they complete (manifest order: tokenizer, LLM, flow, aggregator, VAE) → `warm()` → ready. Revisit: OPFS only. Progress per part and MB/s in the UI.
- **UI** (`index.html`): textarea (default sentence A), voice `<select>` (5 English + 2 Chinese built-ins, "Freeform instruction…" and "Voice design…" revealing a second textarea), Speed preset (`Quality 10 steps` / `Balanced` / `Fast` from §3), seed field, Generate / Stop, progress (frames, seconds generated, RTF live), waveform strip, Play / Pause, Download WAV, a status line with build name, threads, model bytes, and a "network requests since ready: 0" counter (from a `PerformanceObserver`). Minimal CSS, no framework.
- **Audio**: `audio.js` collects f32 chunks; an `AudioContext` at 44,100 Hz schedules `AudioBufferSourceNode`s back to back; playback auto-starts when buffered ≥ 1.5 s or on `done` (so slow machines never underrun); WAV = 16-bit PCM encoder (clamp, dither off).
- **Fallbacks**: no `crossOriginIsolated` → single-threaded build + banner "serve with COOP/COEP for threads"; no relaxed SIMD (Safari, Firefox < 139) → plain build; shared memory allocation failure → retry single-threaded build (non-shared memory can grow lazily).

---

## 8. Correctness strategy

Golden tensors (`scripts/dump_goldens.py`, fp32 reference, sentence A, voice US_James, seed 1895, **weights fake-quantized exactly as the bundle** via the same block math, so parity is tight; plus a second set from the unquantized fp32 model for the quality bar):

| Golden | Shape | Rust test | Tolerance |
|---|---|---|---|
| prompt token ids | [L] | `tokenizer::prompt_ids_match` | exact |
| input embeddings | [L, 896] | `embed_rows_match` | max abs 1e-5 (same dequantized table) |
| `z` after prefill | [896] | `llm::prefill_matches` | rel L2 ≤ 2e-3 |
| `z` at frames 1, 2, 5 (teacher-forced: the reference's own latents fed back) | [896] | `llm::decode_matches` | rel L2 ≤ 2e-3 |
| flow head: `noise_k`, `z_k`, `hist_k` → `latent_k` for k ∈ {0, 1, 5} with the 10-step and 6-step presets | [64] | `flow::sample_matches` (noise injected) | rel L2 ≤ 5e-3 |
| aggregator output for `latent_k` | [896] | `aggregator::matches` | rel L2 ≤ 2e-3 |
| stop logits per frame | [2] | `stop_matches` | abs 1e-3 and same decision |
| VAE decode of the first 20 reference latents, one-shot | [20·3528] | `vocoder::streaming_matches_oneshot` | rel L2 ≤ 1e-2, max abs ≤ 2e-3 |
| whole fragment with injected noise sequence | wav | `e2e_latents_match` | per-frame latent rel L2 ≤ 1e-2 up to frame 10 (AR drift grows after) |
| intelligibility | wav | `asr_check.py` | WER ≤ 0.10 on A/B, ≤ 0.20 on C (numbers) |

The same tests run natively (`cargo test`), in Node on both single-threaded wasm builds (`node-smoke.mjs --check` loads the goldens and asserts), and the e2e asserts the Chrome WAV's WER. `make kbench --check` compares each wasm kernel's dot products with the scalar reference on random blocks (tolerance 2e-5 relative, as SealedKB).

Measurement: `ming bench` prints per-component ms/frame and RTF natively (1 and 8 threads); `node-smoke.mjs` prints RTF for the single-thread builds; `e2e/run.mjs` records load time, RTF, time to first audio, bytes fetched after ready (must be 0), and the WER, writing `research/results.json`.

---

## 9. Performance budget

| Host / build | Flow head (10 st.) | LLM | Agg | VAE | Frame total | RTF (10 st.) | RTF (preset from §3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chrome, 8 workers, relaxed | 30–40 ms | 6–10 | 3 | 8–12 | 50–65 ms | **0.6–0.8** | 0.4–0.6 |
| Chrome, 8 workers, simd128 | 60–75 | 8–12 | 4 | 12–18 | 85–110 | 1.1–1.4 | 0.8–1.0 |
| Chrome / Node, 1 thread, relaxed | 160 | 25 | 12 | 45 | 240 | 3.0 | 2.1 |
| Native (M4 Pro, 8 threads, sdot) | 15–20 | 4 | 2 | 6 | 30 | 0.35 | 0.25 |

Assumptions (to be replaced by `make kbench` numbers in M0): relaxed int8 dot 40 GMAC/s per core, simd128 19 GMAC/s (SealedKB measured 39.3 / 18.7 for Q4_K including unpacking), 8 workers at 65% scaling, weight bandwidth 40 GB/s shared, 1–2 ms of barriers per frame. Target for "done": RTF ≤ 1.0 in the threaded-relaxed build on this machine; stretch: ≤ 0.6. Every milestone records its numbers in `research/results.md` with the command.

---

## 10. Milestones (execution order; each leaves something runnable)

**M0 – Skeleton and kernels (budget check).** Workspace, pins, Makefile, tools, `ming-core::quant` + `qmatmul` for Q7B32/Q4B32 with scalar/neon/simd128/relaxed flavours, `Team` with barriers, `kbench` native + Node (`kbench.mjs`) printing GMAC/s per kernel and µs per broadcast/barrier at 1/4/8 threads. *Accept:* all kernel variants match the scalar reference (rel 2e-5); the table in §9 is re-baselined; `make check` green (no model yet).

**M1 – Compiler and goldens.** `ming compile` writes the `mix` and `q7` bundles (sizes recorded); `dump_goldens.py` writes the fixtures (both fake-quant and fp32 sets); a Python reader round-trips a shard's blocks to prove the format. *Accept:* every kept tensor present with the right shape; tied/unused tensors absent; precomputed `t_emb` equals the reference's `t_embedder(t)` to 1e-5.

**M2 – LLM stack natively.** Tokenizer + prompt builder, embeddings gather, prefill/decode with KV cache and prefix snapshot. *Accept:* token ids exact; `z` goldens within tolerance; `ming bench --llm` prints ms/token at 1 and 8 threads.

**M3 – Flow head, aggregator, stop head, AR loop natively.** `ming say --no-audio` writes the latent sequence. *Accept:* flow/aggregator/stop goldens pass (10 and 6 steps); frame-0..10 latents of the full loop match the reference with injected noise; frame count for sentence A within ±2 of 69.

**M4 – Vocoder natively → first audio.** Upsample lag, windowed backbone with ring KV, iSTFT head, OLA/envelope carry, silence trim, WAV. *Accept:* streaming equals the one-shot golden; `ming say "…" -o a.wav` passes `asr_check.py` (WER ≤ 0.10); `ming bench` prints the per-component table and RTF.

**M5 – Native speed.** Team partitions for every phase, fused QKV/gate-up, last-block trimming, SIMD f32 loops, 4-row VAE tiles. *Accept:* native 8-thread RTF ≤ 0.4; latents unchanged (goldens still pass).

**M6 – wasm single-threaded in Node.** `ming-runtime` exports, `getrandom` custom backend, `node-smoke.mjs --check` (goldens) and RTF for `web-single` and `web-single-relaxed`; import check passes `--none`-style allowlist. *Accept:* identical latents to native for the same seed (per flavour), WER on the Node WAV ≤ 0.10.

**M7 – Threaded wasm + browser loader.** Threaded builds, `thread-host.js`, `worker.js` loader (probes, ranged fetch, OPFS, push, warm), `ming serve`, Playwright e2e in headless Chrome measuring RTF and asserting zero post-ready requests. *Accept:* e2e green; Chrome threaded-relaxed RTF recorded; revisit fetches 0 bytes.

**M8 – UI and text frontend.** Page per §7, voice/instruction modes, presets, streaming playback, WAV download, number normalization and fragmenting, Chinese pass-through. *Accept:* sentence C ("In 1998 … 2,500 units … forty percent") is spoken with the numbers verbalised; a 3-sentence paragraph is split and concatenated with per-fragment silence trim; Stop cancels within one frame.

**M9 – Speed presets and fallbacks.** Wire the §3 presets, measure each in Chrome, pick the default; verify the single-threaded fallback path in a non-isolated page and the plain build in Node; document limits (Safari, memory). *Accept:* `research/results.md` has the RTF table for all four builds and the presets; README with the two commands (`make build`, `make serve`).

**M-final** = M8 + M9: text → audio in the browser with no network requests after load.

---

## 11. Crazy ideas, ranked (gain on the frame budget vs quality risk)

Frame budget at 10 steps = 8.0 GMAC, 75% in the flow head. "Gain" is on that budget; "exact" means bit-for-bit the reference computation.

| # | Idea | Gain | Quality risk | Status |
|---|---|---:|---|---|
| 1 | **Static thread team + barriers instead of Rayon regions** (§6.4) | the difference between ~60% and ~95% of 8 cores on 6-row matmuls (SealedKB measured 26% of decode time waiting on fork/join, ~45 µs × 140 regions) | none | design |
| 2 | **Relaxed-SIMD int8 dot with 7-bit weights** | 2.1× on every matmul vs SIMD128 (39.3 vs 18.7 GMAC/s per core, SealedKB kbench) | 2.3% vs 1.9% deviation (measured) | design |
| 3 | **CFG only on t ∈ [0.05, 0.8)** | −25% flow-head MACs (−19% frame) | 0.13 rel L2, WER 0.000 on A/B/C (measured) | Balanced preset |
| 4 | **7 EPSS steps + interval CFG** | flow head −50% (−38% frame) | 0.20–0.23 rel L2; WER 0 on prose, numbers slurred in 2 of 4 runs (measured) | Fast preset, opt-in |
| 5 | **Exact trims**: block 8 computes `to_out`/MLP for the 2 `x` rows only, aggregator `final_layer` on cls only, `x_embedder(history)` once per frame, `t_emb` and `y_uncond` precomputed | ≈ 8% of the flow head, ≈ 6% of the frame | none (exact) | design |
| 6 | **Q4 for LLM and VAE decoder** (bandwidth-bound parts) | halves their weight traffic (~−8 ms/frame at 8 workers), bundle 1.25 GB → 0.83 GB | WER 0.00 (measured); not measured: MOS | design (`mix`) |
| 7 | **Per-token activation scales → pure-integer dot** for the flow head (one f32 op per 1024/4096 MACs instead of one per 32: ~3× fewer inner-loop ops, flow head −30–40% on compute-bound builds) | large | **measured 6.2% deviation** (vs 2.3%): activation outliers | rejected as is; see #8 |
| 8 | **SmoothQuant folding** (per-channel factors `s_c = max|a_c|^α / max|w_c|^(1−α)` folded into the preceding RMSNorm weight and the Linear columns, offline, calibration on 10 sentences) | makes #7 viable; no runtime cost | low (exact transformation, only the quantization error moves); to be measured with `harness.py` (`--quant flow=w7a8r` after folding) | later |
| 9 | **Uncond branch reuse every other step** (`alt`) | −25% flow head | 0.25 rel L2 (measured) — dominated by #3 at equal cost | rejected |
| 10 | **Step-level block caching (FORA/Δ-DiT-style)**: reuse blocks 2–7 outputs from the previous step on odd steps | up to −40% flow head | expected ≥ 7-step damage on an 8-block DiT; untested | later, measurable with `harness.py` in one hour |
| 11 | **Decode the vocoder two frames per call** (8 rows instead of 4) | VAE weight traffic halved (~−5 ms/frame), +80 ms latency | none (exact) | M5 option |
| 12 | **Batch fragments** of long texts through the flow head together (12–18 rows) | better arithmetic intensity on bandwidth-bound builds; no gain where compute-bound | none | later |
| 13 | **6 or 5 steps** | −40%/−50% flow head | 0.56 rel L2, early stop on B (measured) | rejected as a preset |
| 14 | Prefix KV precomputed offline and shipped | saves ~0.1 s at load | none | rejected: warm-up at load is simpler and voice-agnostic |
| 15 | Overlap vocoder and AR loop on separate thread groups | 0 throughput gain on a CPU-bound pipeline; only latency | none | rejected; streaming vocoder on the same team gives the latency |
| 16 | f16 weight scales, f16 KV | −6% bytes; slower in wasm (no f16 ops) | none | rejected (SealedKB) |

---

## 12. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| Realtime missed in wasm (RTF 1.2–1.5 at 10 steps on slower laptops) | Presets from §3 (6–7 steps, CFG interval) are the UX default on machines where the measured RTF > 1; the UI shows progress and plays when done; "done" does not require RTF < 1. |
| 3 GiB shared memory refused (low-RAM devices, 32-bit-limited engines) | `mix` bundle keeps the arena ≈ 0.85 GB; loader retries with the single-threaded build (non-shared memory, lazily grown); clear error otherwise. |
| No `crossOriginIsolated` (hosting without COOP/COEP) | single-threaded builds; `ming serve` sets the headers; document the service-worker header trick for static hosts. |
| Safari: no relaxed SIMD, OPFS quirks, threads only in isolated pages | plain SIMD128 builds; memory store fallback; tested in M9 on Safari if available, otherwise documented. |
| Relaxed-SIMD result differences across CPUs | only 7-bit × 8-bit products are used, so `pmaddubsw` cannot saturate; the loader probes the exact instruction; kernel checks run in Node on this machine (ARM) and the executor should run `kbench --check` once on x86 if available. |
| Quantization quality (Q4 on LLM/VAE) | decided by §3 measurements; `q7` profile is one flag away; the compiler could add per-block MSE-optimal scales or SmoothQuant folding later (§11). |
| Tokenization drift between Python and the `tokenizers` crate | exact-match golden on token ids for A/B/C and the constant pieces; the compiler precomputes every constant piece. |
| Long texts exceed the 512-position reference budget | fragments ≤ 50 semantic units keep prompts ≈ 100–180 tokens; KV capacity 1024; frame budget `min(600, 1024 − prompt_len)`. |
| Fork/join or barrier spinning burns CPU between frames | workers park (futex) after 2000 spins; between `step()` calls the pool is idle (no broadcast in flight). |
| `realfft`/`rustfft` on wasm | pure Rust, builds for wasm32 (rustfft has a `wasm_simd` feature); fallback is a 3528-point mixed-radix of our own (2³·3²·7²) – only if the crates fail to build. |
| Browser throttling of background tabs slows generation | generation runs in workers; the UI reports RTF; nothing to do. |

---

## 13. Interfaces (sketches the executor should keep to)

### 13.1 `manifest.json`

```json
{
  "format": 1,
  "model": {"repo": "inclusionAI/Ming-3.0-tts-0.5B", "revision": "e5bd8da31ecf1b9276cdbdda821fa891068f1368", "profile": "mix"},
  "shards": [{"file": "shards/000.bin", "bytes": 50331648, "blake3": "…"}],
  "tensors": {
    "llm.layers.0.qkv.weight": {"dtype": "q4b32", "shape": [1152, 896], "shard": 0, "offset": 0, "bytes": 645120},
    "llm.layers.0.qkv.bias":   {"dtype": "f32",   "shape": [1152], "shard": 0, "offset": 645120, "bytes": 4608},
    "flow.blocks.0.qkv.weight": {"dtype": "q7b32", "shape": [3072, 1024], "shard": 7, "offset": 0, "bytes": 3538944}
  },
  "constants": {
    "presets": {"10": {"t": [0, 0.0048, …, 1.0], "cfg_steps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
                "balanced": {"t": "…10 knots…", "cfg_steps": [4, 5, 6, 7, 8]},
                "fast": {"t": "…7 knots…", "cfg_steps": [4, 5, 6]}},
    "flow.t_emb": "tensor name", "flow.y_uncond": "tensor name", "flow.rope": "…", "agg.rope": "…",
    "istft.window": "tensor name", "istft.env_steady": "tensor name"
  },
  "prompt": {"prefix_ids": [151644, 8948, …], "text_prefix_ids": [...], "turn_end_ids": [...], "im_end_id": 151645, "audio_id": 151706,
             "voices": {"US_James": {"kind": "ip", "instruction_ids": [...]}, "灵小柔": {...}}},
  "tokenizer": {"file": "tokenizer.json", "bytes": 11431389, "blake3": "…"},
  "sample_rate": 44100, "frame_samples": 3528
}
```

Tensor names are renamed to the runtime's vocabulary (`llm.*`, `flow.*`, `agg.*`, `vae.*`, `stop.*`) by one table in `ming-bundle`; the compiler asserts every expected name is present and nothing else is written.

### 13.2 wasm surface (`ming-runtime`)

```
new Engine(manifest_json: string)
engine.begin_part(name, bytes) / write_part(name, offset, bytes: Uint8Array) / end_part(name) -> "{\"ready\":bool}"
engine.warm()                                   // prefix prefill (~37 tokens), idempotent
engine.prepare(request_json) -> plan_json       // {"text","voice":"US_James"|{"freeform":"…"}|{"design":"…"},"preset":"quality|balanced|fast","seed":1895,"max_frames":600}
                                                // -> {"fragments":[{"text":"…","prompt_tokens":123}],"estimated_frames":…}
engine.step() -> "{\"fragment\":0,\"frame\":12,\"done\":false,\"stopped\":false,\"new_samples\":3528}"
engine.take_audio() -> Float32Array             // PCM since the last call, 44.1 kHz
engine.cancel(); engine.stats() -> timings json (frame counts; wall-clock timing is added by the host, there is no clock in the module)
engine.last_panic() -> string; threaded builds: initThreadPool(n)
```

The worker runs `step()` once per macrotask (SealedKB's `ask_next` pattern) so cancel and UI messages interleave; it forwards `take_audio()` chunks to the page as transferable buffers.

### 13.3 Team sketch

```rust
pub struct Team { threads: usize, barrier: SenseBarrier }
impl Team {
    pub fn run<F: Fn(&Ctx) + Sync>(&self, f: F) {           // one rayon::broadcast; with threads == 1 calls f directly
        rayon::broadcast(|b| f(&Ctx { index: b.index(), count: b.num_threads(), barrier: &self.barrier }));
    }
}
impl Ctx<'_> {
    pub fn columns(&self, n: usize, align: usize) -> Range<usize> { /* static split of n output columns, multiple of align */ }
    pub fn barrier(&self) { /* spin ~2000 iterations on an AtomicUsize generation counter, then futex wait (Condvar) */ }
}
// A DiT block on 6 rows, inside one broadcast:
//   every thread: rmsnorm + quantize all 6 rows into its private A8 buffer (no sync)
//   qkv for my columns (heads h0..h1) -> rope -> attention for my heads -> write attn_out[:, my cols]   | barrier
//   every thread: quantize attn_out; to_out for my columns + residual                                     | barrier
//   every thread: rmsnorm + quantize; ff.0 for my 512 columns + gelu                                         | barrier
//   every thread: quantize ff hidden (its own copy); ff.2 for my columns + residual                          | barrier
```

Scratch buffers that several threads write (`attn_out`, `h`, `ff_hidden`) are plain `&mut [f32]` split into disjoint column ranges via `split_at_mut`-style views created before the broadcast (`UnsafeCell` slices with a `// SAFETY:` note on the disjointness), the only `unsafe` outside the SIMD kernels.

### 13.4 Experiment harness reuse

`.cache/experiments/harness.py` is the model-side tool for any future "is this still intelligible?" question: `gen` writes WAV + per-frame latents + manifest, `head` scores a modified flow head teacher-forced against a recorded baseline, `asr_wer.py` scores WAVs. `dump_goldens.py` (M1) should import its `fq_weight`/`fq_act` so the fake-quantized reference uses the same block math as the bundle.
