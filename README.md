# ming-wasm

Text-to-speech with [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)
running entirely in the browser: Rust compiled to WebAssembly, on the CPU, with WASM threads.
Type text, press Generate, get 44.1 kHz speech. The server only serves static files; after the
model has loaded the page makes no network requests.

On an Apple M4 Pro, Chrome generates speech 1.5–2× faster than real time (8 threads, relaxed
SIMD); the numbers for every build are in [`docs/results.md`](docs/results.md).

ming-wasm is the app; the model is a library in [fusor-ai](https://github.com/fusor-rs/fusor-ai),
the toolkit it is built with, which this repository pins as the git submodule `vendor/fusor-ai`.

## Run a release

A release holds the `ming` CLI (macOS on Apple silicon, Linux on x86-64) and
`ming-web-<version>.tar.gz`: the page with its four runtimes, a dev server and the checkpoint
download. Without Rust, with Node.js ≥ 20, curl, about 4 GB of disk and 5 GB of free RAM:

```sh
curl -fsSL https://github.com/fusor-rs/ming-wasm/releases/latest/download/install.sh | sh
version=0.1.0
curl -fsSL https://github.com/fusor-rs/ming-wasm/releases/download/v$version/ming-web-$version.tar.gz | tar -xz
cd ming-web-$version
./fetch-checkpoint.sh checkpoint    # 3 GB, sha256-checked
ming compile checkpoint model       # the 850 MB bundle the page loads from model/
node serve.mjs --mount /=.          # http://localhost:8787/
```

`install.sh` puts `ming` in `~/.ming/bin` (`MING_INSTALL` changes it) and takes a version
(`sh install.sh v0.1.0`); rerunning it upgrades. Any static server works in place of
`serve.mjs` if it sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (without them the page uses the single-threaded
builds).

## Run it from source

Prerequisites: Rust via [rustup](https://rustup.rs) (`rust-toolchain.toml` pins 1.99.0; the
threaded builds use `nightly-2025-11-15` with `rust-src`, installed by `just tools`), Node.js
≥ 20, [`just`](https://github.com/casey/just), curl, about 4 GB of disk for the checkpoint and
the bundle, about 5 GB of free RAM while the model compiles (the compiler reads the 3 GB
checkpoint into memory), and a recent desktop browser (Chrome/Edge 114+ for the fastest build;
any browser with WebAssembly SIMD works).

```sh
git clone --recurse-submodules https://github.com/fusor-rs/ming-wasm && cd ming-wasm
just build    # tools, native CLI, downloads + compiles the model (850 MB), four browser runtimes
just serve    # http://localhost:8787/
```

`just build` downloads the pinned checkpoint (3 GB, sha256-checked) into
`.cache/Ming-3.0-tts-0.5B` unless `CHECKPOINT=<dir>` already holds `model.safetensors` and
`tokenizer.json`. `PORT=… just serve` changes the port.

## What to expect

The page loads the model (progress per MB; a revisit loads it from the browser's
origin-private storage in about a second), then offers:

- a text box (long text is split into fragments and synthesized one after another; English
  numbers, money, percentages and units are read out),
- a voice: five English and two Chinese built-in voices, **Describe a voice…** (a voice
  designed from a description) or **Free-form instruction…** (voice and delivery in words),
- a speed preset (quality, balanced, fast) and a seed,
- Generate / Stop. Audio plays while it is generated when generation is faster than real time,
  otherwise when it is done; **Download WAV** saves it.

The footer names the runtime build in use and counts network requests since the model loaded.
`?threads=1` forces the single-threaded build, `?build=web-single` pins a build, `?model=<url>`
loads another bundle (another origin must allow CORS and send `Cross-Origin-Resource-Policy`,
since the page is cross-origin isolated).

## How it works

```
text → fragments → prompt (built-in voice caption) → Qwen2-0.5B prefill
     → per 80 ms frame: z → flow-matching DiT (10 Euler steps, CFG) → 64-d latent
                         latent → aggregator DiT → next LLM input → z
                         latent → streaming VAE decoder (Qwen2 backbone + iSTFT) → 3528 samples
```

The model, `ming` in fusor-ai's `crates/models/ming`, names these stages in
`ming::pipeline::Stage`; `ming bench` times them and `ming parity` checks them one by one. Its
compiler, `ming-compile`, turns the checkpoint into a bundle. This repository holds the app
around them:

- `crates/ming-runtime`: the wasm-bindgen surface (parts in, PCM out). Its modules import no
  network, storage, clock or entropy capability (fusor-ai's `scripts/check-wasm-imports.mjs`).
- `crates/ming-cli`: the native CLI `ming`: `compile`, `say`, `bench`, `parity`.
- `packages/web`: the page and the engine worker (generation loop); loading and the compute
  workers are fusor-ai's `packages/host`, copied to `packages/web/dist/host/` by
  `just runtimes`.
- `scripts/`: the checkpoint download, the golden dump, the Node smoke test and the ASR check;
  `e2e/run.mjs`: the headless Chrome run.
- `docs/`: [`plan.md`](docs/plan.md) (the design), [`decisions.md`](docs/decisions.md),
  [`results.md`](docs/results.md).

## Checks

```sh
just check     # fmt, tests, clippy (native + wasm, -D warnings), docs, host scripts
just runtimes  # the four runtime builds and their import checks
just parity    # every stage against the PyTorch goldens (needs `just goldens` first)
just smoke     # both single-threaded runtimes in Node
just e2e       # headless Chrome: load, generate, 0 requests during synthesis, ASR check, revisit from cache
```

`just goldens` runs the PyTorch reference: `REFERENCE=<dir>` must hold a checkout of
[inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) at `e999c39` as
`Ming-3.0-tts/`, the checkpoint as `Ming-3.0-tts-0.5b/`, and a `.venv/` with that repository's
requirements. `just parity` then compiles a 4.3 GB f32 bundle. The e2e's ASR check uses
faster-whisper from `ASR_PYTHON` (skipped if absent).

Native use: `target/release/ming say dist/model "Hello there." -o hello.wav
[--voice UK_George] [--design "…"] [--freeform "…"] [--preset fast]`,
`target/release/ming bench dist/model`.

## Limits

- No voice cloning from a reference recording (it needs the VAE encoder and the CAM++ speaker
  model in the browser; decisions D2).
- The reference's FST text normalizer is not ported (decisions D10): dates, times and
  abbreviations are read as written.
- Speed depends on the device: without cross-origin isolation (COOP/COEP) or relaxed SIMD the
  page falls back to slower builds (single-threaded SIMD128 runs at about a quarter of real
  time on an M4 Pro).
- The model's license (MIT) and its usage disclaimer apply: label synthetic speech as synthetic.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the setup, checks and releasing; coding agents follow
[AGENTS.md](AGENTS.md).

## License

Licensed under the [MIT License](LICENSE).
