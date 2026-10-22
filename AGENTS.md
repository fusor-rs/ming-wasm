# AGENTS.md

Rules for coding agents in this repository that are its own. fusor-ai's
`vendor/fusor-ai/AGENTS.md` is the standard and binds here too (`CLAUDE.md`
imports both); this file adds what is particular to this product.

ming-wasm is the Ming-3.0-tts-0.5B text-to-speech app: the model running in the
browser on the CPU, sealed from the network. The README describes what it does;
don't invent scope, features or options beyond the task.

## Repository

- `vendor/fusor-ai/models/ming/runtime`: the wasm-bindgen surface of the model
  (bundle parts in, PCM out), built as four browser runtimes.
- `vendor/fusor-ai/models/ming/cli`: the native CLI `ming` (`compile`, `say`,
  `bench`, `parity`).
- `app/`: the page, a [fusor](https://github.com/fusor-rs/fusor) app with its own
  Cargo workspace (fusor pins wasm-bindgen 0.2.117, the runtime 0.2.129): the
  form and progress as Rust signals in `src/app.rs`, the template in
  `web/index.html`, the engine bridge and player in `web/*.js`, and the engine
  worker in `engine/worker.js`, beside the runtimes `just runtimes` builds.
- `scripts/`: the checkpoint download, the Node smoke test and the ASR check;
  `e2e/run.mjs`: the headless Chrome run.
- `benchmarks.md` and `benchmarks/`: the measured speed against the PyTorch
  reference and other runtimes, with the harness that reproduces it.
- [fusor-ai](https://github.com/fusor-rs/fusor-ai): the model (`ming`), its
  compiler (`ming-compile`), the kernels, blocks and bundle format come from
  `vendor/fusor-ai`, a git submodule pinned to a commit pushed to fusor-ai, with
  the browser loader and compute pool, the runtime build, the import check and
  the dev server.

What belongs where: model code (dimensions, tensor names, wiring, sampling,
prompts, the text frontend) and anything another model could reuse go to
fusor-ai, following its `AGENTS.md`. This repository holds what turns the model
into an app: the page, the checks against the reference and the releases. A
change to fusor-ai lands there first; this repository then moves its submodule
to it.

Constraints that hold everywhere:

- fusor-ai comes only from `vendor/fusor-ai`, never a crates.io version, a
  `[patch]` table or a sibling checkout. The submodule moves to another pushed
  fusor-ai commit in a change of its own, with the parity numbers and the e2e
  result it gave.

## Checks

A change is done only when this passes:

```sh
just check
```

It runs formatting and Clippy on the page and syntax checks of the scripts;
fusor-ai's `just check` covers the runtime and the CLI. A change to the runtime,
the page, the CLI or the submodule also needs `just runtimes`, `just smoke`,
`just parity` (the same numbers as before unless the change is meant to move
them) and `just e2e`. The model is checked stage by stage against the PyTorch
goldens by `ming parity`; extend it rather than adding a parallel harness.
