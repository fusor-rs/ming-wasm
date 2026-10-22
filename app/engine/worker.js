// The engine's home: a module Worker that loads the model and runs synthesis.
//
// Load: fusor-ai's worker kit (`dist/host/worker.js`) starts the best runtime build and streams
// every part of the bundle into it; the runtime checks each part's size and BLAKE3.
//
// Generate: one `step()` (one 80 ms frame) per macrotask, so a stop request is seen between
// frames; every step's new PCM goes to the page as a transferable Float32Array.
//
// After the model is loaded nothing else is fetched: synthesis is entirely local.

import { engine, load, post, requestsSinceReady, serve, yieldTask } from './dist/host/worker.js';

let stopRequested = false;

serve({
  load: ({ modelUrl, threads, build }) =>
    load({
      modelUrl,
      threads,
      build,
      glueName: 'ming_runtime',
      cacheName: 'ming-tts-parts',
      distUrl: new URL('./dist/', import.meta.url),
      onRuntime: (started) => post({ type: 'build', build: started.build.name, threads: started.threads }),
      onReady: (progress) =>
        post({ type: 'ready', ...progress, ms: Math.round(progress.ms), voices: JSON.parse(engine.glue.voices()), threads: engine.rt.threads() }),
    }),
  generate: ({ request }) => generate(request),
  stop: () => (stopRequested = true),
});

async function generate(request) {
  const { rt } = engine;
  stopRequested = false;
  const plan = JSON.parse(rt.start(JSON.stringify(request)));
  post({ type: 'plan', fragments: plan.fragments.length });
  const t0 = performance.now();
  for (;;) {
    if (stopRequested) {
      rt.cancel();
      post({ type: 'done', stopped: true, ms: performance.now() - t0, networkRequests: requestsSinceReady() });
      return;
    }
    const progress = JSON.parse(rt.step());
    const samples = rt.take_audio();
    if (samples.length) post({ type: 'audio', samples }, [samples.buffer]);
    post({ type: 'frame', ...progress, ms: performance.now() - t0 });
    if (progress.done) break;
    await yieldTask();
  }
  post({ type: 'done', stopped: false, ms: performance.now() - t0, networkRequests: requestsSinceReady() });
}
