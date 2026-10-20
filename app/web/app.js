import { ComputePool, SPAWN_COMPUTE } from '../../vendor/fusor-ai/packages/host/compute.js';
import { Player } from './player.js';

// The engine bridge. The page's state lives in Rust (src/app.rs); this module owns the engine
// worker and, in threaded builds, its compute workers (fusor-ai's ComputePool), plays the audio,
// reads the form from the `#[js]` inputs and reports back with `engine` events.

const params = new URLSearchParams(location.search);
const cores = navigator.hardwareConcurrency || 4;
// One compute worker per core minus two (the page and the engine worker), at most 8; `?threads=1`
// forces the single-threaded build. Never more than the cores: the barriers spin.
const requestedThreads = Number.parseInt(params.get('threads') ?? '', 10);
const THREADS = Math.min(cores, requestedThreads > 0 ? requestedThreads : Math.max(1, Math.min(8, cores - 2)));
const HOSTED_MODEL = 'https://huggingface.co/fusorlabs/ming-3.0-tts-0.5b-web/resolve/main/';
// `?model=<url>` loads another bundle than the published one; `?build=web-single` pins a build.
const MODEL_URL = new URL(params.get('model') ?? HOSTED_MODEL, document.baseURI).href;

const mb = (bytes) => `${(bytes / 1e6).toFixed(0)} MB`;

export function onMount({ root, inputs, onCleanup }) {
  const report = (detail) => root.dispatchEvent(new CustomEvent('engine', { detail: JSON.stringify(detail) }));
  const engine = new Worker(new URL('engine/worker.js', document.baseURI), { type: 'module', name: 'ming-engine' });
  const computePool = new ComputePool(crash);
  let loaded = false;
  let broken = false;
  let build = '';
  let notice = '';
  let job = null;
  let wavUrl = null;
  let requestsAtReady = 0;

  const handlers = {
    [SPAWN_COMPUTE]: (message) => computePool.spawn(message),
    build: ({ build: name, threads }) => (build = `${name} · ${threads} thread${threads > 1 ? 's' : ''}`),
    warning: ({ message }) => {
      console.warn(message);
      notice = ` (${message})`;
    },
    progress: ({ done, total, network, cache }) => {
      const source = cache && !network ? 'from this device' : cache ? `${mb(network)} downloaded, ${mb(cache)} cached` : 'downloading';
      report({ type: 'loading', fraction: done / total, detail: `${mb(done)} / ${mb(total)} · ${source}` });
    },
    ready: ({ ms, voices, cache, total }) => {
      loaded = true;
      requestsAtReady = performance.getEntriesByType('resource').length;
      const seconds = `${(ms / 1000).toFixed(1)} s`;
      const model = cache === total
        ? `model loaded from this device in ${seconds}`
        : `model ${mb(total)} in ${seconds}${cache ? ` (${mb(cache)} from this device)` : ''}`;
      report({ type: 'ready', voices, runtime: `${build}${notice} · ${model}` });
    },
    audio: ({ samples }) => job.player.push(samples),
    frame: ({ fragment, fragments, frame, ms }) => {
      job.frames++;
      const seconds = job.player.seconds();
      const status = `fragment ${fragment + 1}/${fragments} · frame ${frame + 1} · ${seconds.toFixed(1)} s of audio in ${(ms / 1000).toFixed(1)} s`;
      report({ type: 'progress', fraction: (fragment + 0.5) / fragments, status });
      job.player.maybeStart(seconds / (ms / 1000));
    },
    done: ({ stopped, ms, networkRequests }) => finish(stopped, ms, networkRequests),
    error: ({ message }) => (loaded ? report({ type: 'error', message }) : fail(message)),
  };

  engine.onmessage = ({ data }) => {
    if (!broken) handlers[data.type]?.(data);
  };
  engine.onerror = (event) => crash(event.message || 'the engine worker failed');
  engine.postMessage({ type: 'load', modelUrl: MODEL_URL, threads: THREADS, build: params.get('build') });

  function generate() {
    job?.player.close();
    job = { player: new Player(), frames: 0 };
    const voice = inputs.voice.get();
    const request = {
      text: inputs.text.get().trim(),
      voice: voice === 'design' ? { design: inputs.description.get().trim() } : { builtin: voice },
      preset: inputs.preset.get(),
    };
    engine.postMessage({ type: 'generate', request });
  }

  function finish(stopped, ms, workerRequests) {
    const seconds = job.player.seconds();
    job.player.maybeStart(Infinity);
    if (wavUrl) URL.revokeObjectURL(wavUrl);
    wavUrl = seconds ? URL.createObjectURL(new Blob([job.player.wav()], { type: 'audio/wav' })) : null;
    const requests = performance.getEntriesByType('resource').length - requestsAtReady + workerRequests;
    const rtf = seconds ? ms / 1000 / seconds : 0;
    const stats = `${seconds.toFixed(2)} s of audio in ${(ms / 1000).toFixed(1)} s · real-time factor ${rtf.toFixed(2)} · ${job.frames} frames · ${requests} network requests since the model loaded`;
    report({ type: 'done', status: stopped ? 'stopped' : 'done', audio: wavUrl, stats });
  }

  /** The model could not load, or the engine or a compute thread died. */
  function fail(message) {
    broken = true;
    engine.terminate();
    computePool.terminate();
    report({ type: 'broken', message });
  }

  function crash(message) {
    fail(`The engine stopped: ${message}.`);
  }

  onCleanup(whenChanged(inputs.generations, generate));
  onCleanup(whenChanged(inputs.stops, () => engine.postMessage({ type: 'stop' })));
  onCleanup(() => {
    engine.terminate();
    computePool.terminate();
    job?.player.close();
    if (wavUrl) URL.revokeObjectURL(wavUrl);
  });
}

/** Runs `action` on every change of a click counter (`subscribe` reports the current value first). */
function whenChanged(input, action) {
  let current = true;
  return input.subscribe(() => {
    if (current) current = false;
    else action();
  });
}
