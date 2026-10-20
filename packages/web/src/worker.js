// The engine's home: a module Worker that loads the model and runs synthesis.
//
// Load: fusor-ai's loader picks and starts the best runtime build (threaded builds get their
// compute workers from the page) and streams every part of the bundle into it, from OPFS when
// cached there, else from the network; the runtime checks each part's size and BLAKE3.
//
// Generate: one `step()` (one 80 ms frame) per macrotask, so a stop request is seen between
// frames; every step's new PCM goes to the page as a transferable Float32Array.
//
// After the model is loaded nothing else is fetched: synthesis is entirely local.

import { describe, fetchOk, loadParts, startRuntime } from '../dist/host/loader.js';

const CACHE_DIR = 'ming-tts-parts';
const DIST_URL = new URL('../dist/', import.meta.url);

const post = (message, transfer) => self.postMessage(message, transfer ?? []);
const yieldTask = () => new Promise((resolve) => setTimeout(resolve, 0));

let runtime = null;
let stopRequested = false;
// Resource-timing entries of this worker when the model was ready: every later one is a request
// made during synthesis (the page shows the count; it should stay 0).
let entriesAtReady = 0;
const requestsSinceReady = () => performance.getEntriesByType('resource').length - entriesAtReady;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') await load(data);
    else if (data.type === 'generate') await generate(data.request);
    else if (data.type === 'stop') stopRequested = true;
  } catch (error) {
    post({ type: 'error', message: describe(error, runtime?.glue) });
  }
};

async function load({ modelUrl, threads, build: requested }) {
  const t0 = performance.now();
  const manifestText = await (await fetchOk(new URL('manifest.json', modelUrl))).text();
  const started = await startRuntime({
    distUrl: DIST_URL,
    glueName: 'ming_runtime',
    threads,
    requested,
    onWarning: (message) => post({ type: 'warning', message }),
  });
  const rt = new started.glue.Runtime(manifestText);
  runtime = { rt, glue: started.glue };
  post({ type: 'build', build: started.build.name, threads: started.threads });
  const progress = await loadParts({
    runtime: rt,
    manifest: JSON.parse(manifestText),
    baseUrl: modelUrl,
    cacheName: CACHE_DIR,
    onProgress: (p) => post({ type: 'progress', ...p }),
  });
  entriesAtReady = performance.getEntriesByType('resource').length;
  post({ type: 'ready', ms: Math.round(performance.now() - t0), voices: JSON.parse(started.glue.voices()), threads: rt.threads(), ...progress });
}

async function generate(request) {
  const { rt } = runtime;
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
