---
title: ming-wasm
short_description: Ming-3.0-tts-0.5B text to speech in the browser
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - fusorlabs/ming-3.0-tts-0.5b-web
  - inclusionAI/Ming-3.0-tts-0.5B
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

Text to speech with [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B),
computed on your own CPU in the browser: the model, compiled to WebAssembly, downloads once
(850 MB) and then runs sealed from the network. The source is
[github.com/fusor-rs/ming-wasm](https://github.com/fusor-rs/ming-wasm), and the same page runs
at [ming-wasm.fusor.build](https://ming-wasm.fusor.build).
