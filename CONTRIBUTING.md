# Contributing

Use Rust 1.99 or later, edition 2024, Node.js 22 or later, and `just`. Clone with
`--recurse-submodules` (or run `just tools`, which initializes `vendor/fusor-ai`). `unsafe` is
denied except in the module that states why it needs it (see `AGENTS.md`).

Run `just check` before considering a change complete. It formats, tests and lints the crates
natively, the runtime for `wasm32-unknown-unknown` (SIMD128 and relaxed SIMD) and the page,
and syntax-checks the scripts. `just ui` builds the page and serves it; the page is a
[fusor](https://github.com/fusor-rs/fusor) app in `app/`, built with the fusor CLI that
`just tools` installs into `.tools/`. A change to the runtime, the page, the CLI or the
submodule also needs `just build` once, then `just runtimes`, `just smoke`, `just parity` and
`just e2e`. Say in the pull request what they showed (parity numbers, real-time factor, word
error rate).

The model, its compiler and the toolkit live in
[fusor-ai](https://github.com/fusor-rs/fusor-ai). Everything this repository takes from it
comes from `vendor/fusor-ai`, a submodule pinned to a commit pushed to fusor-ai: the
`fusor-ai-*` crates, as path dependencies in the root `Cargo.toml`, the browser host scripts,
the runtime build and the import check. To move to another fusor-ai commit, check it out in the
submodule (`git -C vendor/fusor-ai checkout <commit>`) in one change of its own, with the
checks above. Never add a crates.io pin, a `[patch]` table or a path to a fusor-ai checkout
beside this one.

The threaded runtimes need `nightly-2025-11-15` with `rust-src`; `just tools` installs it,
and fusor-ai's `scripts/install-tools.sh` the pinned wasm-bindgen and binaryen into `.tools/`.

## Releasing

Every crate shares one version: `version` under `[workspace.package]` in the root
`Cargo.toml`. A release attaches the `ming` CLI archives, the page with its runtimes and
`install.sh` to the GitHub release. No crate is published to crates.io; fusor-ai publishes
the model and its compiler.

1. Bump `version`, run `just check` so `Cargo.lock` follows, and merge that to `main` with
   CI passing.
2. On GitHub, publish a release whose tag is the version with a `v`, such as `v0.2.0`, on
   that commit.

The [Release workflow](.github/workflows/release.yml) then runs Check, checks that the tag
matches the version, and builds and tests every archive:

- `just archive` builds the `ming` CLI for the runner's target (macOS on Apple silicon, for
  every Apple silicon CPU; Linux on x86-64) and packages it as `ming-VERSION-TARGET.tar.gz`,
  holding `bin/ming`, `LICENSE` and `README.md`, with a `.sha256` file.
- `just archive-check` installs that archive with [install.sh](install.sh) from the local
  files (`MING_DOWNLOAD_BASE`), checks the installed version, and checks that a corrupt
  download is refused.
- `just web-archive` builds the four runtimes through the import check and packages the page
  with them, `serve.mjs`, `fetch-checkpoint.sh` and the README as `ming-web-VERSION.tar.gz`,
  checking that every runtime is in it.

It attaches the archives to the release, then `install.sh` last: its presence marks the
release complete. Running the workflow by hand builds and tests the archives without
attaching them.

## Deploying the site

[ming-wasm.fusor.build](https://ming-wasm.fusor.build) is the Vercel project `ming-wasm` in
the `pirela` team, deployed as prebuilt files: Vercel has no Rust toolchain. Run the
[Deploy site workflow](.github/workflows/deploy-site.yml) from the Actions tab (it needs a
`VERCEL_TOKEN` secret in a GitHub environment named `vercel`), or `just deploy` locally after
`vercel link --scope pirela --project ming-wasm`. [vercel.json](vercel.json) makes the page
cross-origin isolated, so it gets the threaded runtimes.

The same page is the static Hugging Face Space
[fusorlabs/ming-wasm](https://huggingface.co/spaces/fusorlabs/ming-wasm). Run the
[Deploy Space workflow](.github/workflows/deploy-space.yml) (it needs an `HF_TOKEN` secret in
a GitHub environment named `huggingface`), or `just deploy-space` locally with
[uv](https://docs.astral.sh/uv/) after `hf auth login`. It uploads the page with
[space/README.md](space/README.md), whose `custom_headers` make the Space cross-origin isolated.

Both load the published bundle,
[fusorlabs/ming-3.0-tts-0.5b-web](https://huggingface.co/fusorlabs/ming-3.0-tts-0.5b-web);
`?model=<url>` loads another one, such as `?model=model/` for `dist/model` under `just serve`.

## Using AI tools

You're welcome to use AI tools, but you're responsible for everything you submit. Read and
understand every line, make sure the tests you add would fail without your change, and be ready
to explain your reasoning in review. Please don't open pull requests or issues generated
without that review.

ming-wasm's own development uses AI tools heavily too, mainly Claude Code, and the maintainer
works under the same rules. Coding agents follow `AGENTS.md`, which is also the code standard
for every change.
