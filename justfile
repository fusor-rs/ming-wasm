# Ming-3.0-tts-0.5B in the browser. `just ui` to try it; `just build` then `just serve` to build
# everything, the model included, from source.

import 'vendor/fusor-ai/just/product.just'

name := "ming"
checkpoint := env_var_or_default("CHECKPOINT", ".cache/Ming-3.0-tts-0.5B")
reference := env_var_or_default("REFERENCE", "../ming-05b-experiment")
profile := env_var_or_default("PROFILE", "mix")

host:
    sh -n install.sh
    bash -n scripts/fetch-checkpoint.sh
    node --check app/web/app.js
    node --check app/web/player.js
    node --check app/engine/worker.js
    node --check scripts/node-smoke.mjs
    node --check e2e/run.mjs

# Downloads the pinned checkpoint (3 GB) unless it is already there, then compiles the bundle.
model: native
    scripts/fetch-checkpoint.sh {{checkpoint}}
    target/release/ming compile {{checkpoint}} {{model}} --profile {{profile}}

# Try it on localhost: the page with the bundle compiled here (`just model`), else the published one.
ui: tools
    test -f {{engine}}/web-single/ming_runtime_bg.wasm || just runtimes
    just page
    just serve

# Golden tensors from the fp32 PyTorch reference (see
# vendor/fusor-ai/models/ming/reference/dump_goldens.py).
goldens:
    {{reference}}/.venv/bin/python {{fusor_ai}}/models/ming/reference/dump_goldens.py --reference {{reference}} --out fixtures/goldens/ref_f32_A.safetensors

# Every stage against the goldens, with the f32 bundle (4.3 GB, native only).
parity: native
    test -f dist/model-f32/manifest.json || target/release/ming compile {{checkpoint}} dist/model-f32 --profile f32
    target/release/ming parity dist/model-f32 fixtures/goldens/ref_f32_A.safetensors

bench: native
    target/release/ming bench {{model}}

# Releases (CONTRIBUTING.md, "Releasing"): archives in target/release-archives/v<version>/.

# The `ming` CLI for this machine's target.
archive:
    #!/usr/bin/env bash
    set -euo pipefail
    target=$(rustc -vV | sed -n 's/^host: //p')
    # .cargo/config.toml tunes aarch64 builds for this machine's CPU; a release runs on every
    # Apple silicon Mac.
    if [ "$target" = aarch64-apple-darwin ]; then export RUSTFLAGS="-C target-cpu=apple-m1"; fi
    (cd {{fusor_ai}} && cargo build --locked --release --target "$target" -p ming-cli --target-dir "{{justfile_directory()}}/target")
    name="ming-{{version}}-$target"
    staging="target/release-archives/$name"
    out="target/release-archives/v{{version}}"
    rm -rf "$staging"
    mkdir -p "$staging/bin" "$out"
    cp "target/$target/release/ming" "$staging/bin/"
    cp LICENSE README.md "$staging/"
    COPYFILE_DISABLE=1 tar -czf "$out/$name.tar.gz" -C target/release-archives "$name"
    (cd "$out" && shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256")
    echo "$out/$name.tar.gz"

# Installs the archive from `just archive` with install.sh and checks that a corrupt download
# is refused.
archive-check:
    #!/usr/bin/env bash
    set -euo pipefail
    target=$(rustc -vV | sed -n 's/^host: //p')
    archives="$PWD/target/release-archives"
    archive="$archives/v{{version}}/ming-{{version}}-$target.tar.gz"
    work=$(mktemp -d)
    trap 'rm -r "$work"' EXIT
    export MING_DOWNLOAD_BASE="file://$archives"
    export MING_INSTALL="$work/installed ming"
    sh install.sh "v{{version}}"
    test "$("$MING_INSTALL/bin/ming" --version)" = "ming {{version}}"
    cp "$archive" "$work/intact.tar.gz"
    printf 'corrupt download' >> "$archive"
    if sh install.sh "v{{version}}" > "$work/rejected.log" 2>&1; then
        mv "$work/intact.tar.gz" "$archive"
        echo "install.sh accepted a corrupt archive" >&2
        exit 1
    fi
    mv "$work/intact.tar.gz" "$archive"
    grep -Fx 'error: checksum mismatch; download the release again' "$work/rejected.log"

# The page with its four runtimes, the dev server, the checkpoint download and how to run them.
web-archive: runtimes page
    #!/usr/bin/env bash
    set -euo pipefail
    name="ming-web-{{version}}"
    staging="target/release-archives/$name"
    out="target/release-archives/v{{version}}"
    rm -rf "$staging"
    mkdir -p "$staging" "$out"
    cp -R {{app}}/dist/. "$staging/"
    rm "$staging/.fusor-output.json"
    cp {{fusor_ai}}/scripts/serve.mjs scripts/fetch-checkpoint.sh LICENSE README.md "$staging/"
    COPYFILE_DISABLE=1 tar -czf "$out/$name.tar.gz" -C target/release-archives "$name"
    (cd "$out" && shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256")
    listing=$(tar -tzf "$out/$name.tar.gz")
    for variant in {{variants}}; do
        grep -qx "$name/engine/dist/$variant/ming_runtime_bg.wasm" <<< "$listing"
    done
    echo "$out/$name.tar.gz"

# The page to Vercel as prebuilt files: `production` (ming-wasm.fusor.build) or `preview`.
deploy environment="production":
    @case "{{environment}}" in production|preview) ;; *) echo "deploy: expected production or preview, got {{environment}}" >&2; exit 2 ;; esac
    vercel build {{ if environment == "production" { "--prod" } else { "" } }} --yes
    find .vercel/output/static -name '.fusor-*.json' -delete
    vercel deploy --prebuilt {{ if environment == "production" { "--prod" } else { "" } }} --yes

# The page to the Hugging Face Space fusorlabs/ming-wasm, configured by space/README.md.
deploy-space: runtimes page
    #!/usr/bin/env bash
    set -euo pipefail
    staging="target/space"
    rm -rf "$staging"
    mkdir -p "$staging"
    cp -R {{app}}/dist/. "$staging/"
    find "$staging" -name '.fusor-*.json' -delete
    cp space/README.md "$staging/"
    # Not `hf upload`: it first creates the repo as a Gradio Space, which Hugging Face refuses for
    # a free organization (402) even when the static Space exists.
    uv run --no-project --with huggingface_hub==0.36.2 python - "$staging" <<'EOF'
    import sys
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id="fusorlabs/ming-wasm",
        repo_type="space",
        folder_path=sys.argv[1],
        delete_patterns="*",
        commit_message="Deploy ming-wasm {{version}}",
    )
    EOF

# The fixed inputs the checks read.
fixtures:
