import { ComputePool, SPAWN_COMPUTE } from '../dist/host/compute.js';

// The page: owns the engine worker and (threaded builds, through fusor-ai's ComputePool) its
// compute workers, the form, and playback. Audio is played while it is generated when
// generation outruns playback, otherwise once it is complete; the full utterance is offered as a
// WAV.

const SAMPLE_RATE = 44100;
const params = new URLSearchParams(location.search);
const cores = navigator.hardwareConcurrency || 4;
// One compute worker per core minus two (UI and the engine worker), at most 8; `?threads=1`
// forces the single-threaded build. Never more than the cores: the barriers spin.
const requestedThreads = Number.parseInt(params.get('threads') ?? '', 10);
const THREADS = Math.min(cores, requestedThreads > 0 ? requestedThreads : Math.max(1, Math.min(8, cores - 2)));
const MODEL_URL = new URL(params.get('model') ?? 'model/', location.href).href;
// Start streaming once this much audio is buffered, if generation runs faster than real time.
const STREAM_START_SECONDS = 1.0;
const STREAM_MIN_SPEED = 1.1;

const $ = (id) => document.getElementById(id);
const ui = {
  loadTitle: $('load-title'), loadDetail: $('load-detail'), loadBar: $('load-bar'), compose: $('compose'),
  text: $('text'), voice: $('voice'), preset: $('preset'), seed: $('seed'), instructionBox: $('instruction-box'),
  instructionLabel: $('instruction-label'), instruction: $('instruction'), generate: $('generate'), stop: $('stop'),
  genStatus: $('gen-status'), genBar: $('gen-bar'), output: $('output'), player: $('player'), download: $('download'),
  stats: $('stats'), runtime: $('runtime'), network: $('network'),
};

const engine = new Worker(new URL('./worker.js', import.meta.url), { type: 'module', name: 'ming-engine' });
const computePool = new ComputePool(crash);
const network = { page: 0, worker: 0 };
let job = null;
let loaded = false;
let stopped = false; // the engine failed for good: ignore its late messages
let notice = '';
let wavUrl = null;

engine.onmessage = ({ data }) => {
  const handler = HANDLERS[data.type];
  if (handler && !stopped) handler(data);
};
engine.onerror = (e) => crash(e.message || 'the engine worker failed');
// `?build=web-single` etc. pins a runtime build (when this browser can run it), for comparisons.
engine.postMessage({ type: 'load', modelUrl: MODEL_URL, threads: THREADS, build: params.get('build') });

const HANDLERS = {
  [SPAWN_COMPUTE]: (message) => computePool.spawn(message),
  build: ({ build, threads }) => {
    ui.runtime.textContent = `${build} · ${threads} thread${threads > 1 ? 's' : ''}${notice ? ` (${notice})` : ''}`;
  },
  warning: ({ message }) => {
    console.warn(message);
    notice = message;
  },
  progress: ({ done, total, network: net, cache }) => {
    ui.loadBar.value = done / total;
    const source = cache && !net ? 'from this device' : cache ? `${mb(net)} downloaded, ${mb(cache)} cached` : 'downloading';
    ui.loadDetail.textContent = `${mb(done)} / ${mb(total)} · ${source}`;
  },
  ready: ({ ms, voices, network: net, cache, total }) => {
    loaded = true;
    ui.loadTitle.textContent = 'Model ready';
    ui.loadDetail.textContent = `${mb(total)} in ${(ms / 1000).toFixed(1)} s${cache ? ` (${mb(cache)} from cache)` : ''}`;
    fillVoices(voices);
    ui.compose.setAttribute('aria-disabled', 'false');
    ui.generate.disabled = false;
    network.page = performance.getEntriesByType('resource').length;
    updateNetwork();
  },
  plan: ({ fragments }) => (job.fragments = fragments),
  audio: ({ samples }) => job?.player.push(samples),
  frame: ({ fragment, fragments, frame, ms }) => {
    job.frames++;
    const seconds = job.player.seconds();
    ui.genStatus.textContent = `fragment ${fragment + 1}/${fragments} · frame ${frame + 1} · ${seconds.toFixed(1)} s of audio in ${(ms / 1000).toFixed(1)} s`;
    ui.genBar.value = (fragment + 0.5) / fragments;
    job.player.maybeStart(seconds / (ms / 1000));
  },
  done: ({ stopped, ms, networkRequests }) => finish(stopped, ms, networkRequests),
  error: ({ message }) => fail(message),
};

function fillVoices(voices) {
  const options = voices.map((v) => new Option(v.label, `builtin:${v.id}`));
  options.push(new Option('Describe a voice…', 'design'), new Option('Free-form instruction…', 'freeform'));
  ui.voice.replaceChildren(...options);
}

ui.voice.addEventListener('change', () => {
  const mode = ui.voice.value;
  ui.instructionBox.hidden = mode.startsWith('builtin:');
  ui.instructionLabel.textContent = mode === 'design' ? 'Describe the voice' : 'How should it be spoken?';
  ui.instruction.placeholder = mode === 'design'
    ? 'A calm, deep male narrator with a slow, clear delivery.'
    : 'Speak in a warm, intimate storytelling voice, with a gentle pace.';
});

function voiceRequest() {
  const mode = ui.voice.value;
  if (mode.startsWith('builtin:')) return { builtin: mode.slice('builtin:'.length) };
  const text = ui.instruction.value.trim() || ui.instruction.placeholder;
  return mode === 'design' ? { design: text } : { freeform: text };
}

ui.generate.addEventListener('click', () => {
  const text = ui.text.value.trim();
  if (!text) return;
  job?.player.close();
  job = { player: new Player(), frames: 0, fragments: 1 };
  ui.generate.disabled = true;
  ui.stop.disabled = false;
  ui.output.hidden = true;
  ui.genBar.hidden = false;
  ui.genBar.value = 0;
  ui.genStatus.textContent = 'preparing…';
  const request = { text, voice: voiceRequest(), preset: ui.preset.value, seed: Number(ui.seed.value) || 0 };
  engine.postMessage({ type: 'generate', request });
});

ui.stop.addEventListener('click', () => {
  engine.postMessage({ type: 'stop' });
  ui.stop.disabled = true;
});

function finish(stopped, ms, networkRequests) {
  network.worker = networkRequests;
  updateNetwork();
  ui.generate.disabled = false;
  ui.stop.disabled = true;
  ui.genBar.hidden = true;
  const seconds = job.player.seconds();
  const rtf = seconds ? ms / 1000 / seconds : 0;
  ui.genStatus.textContent = stopped ? 'stopped' : 'done';
  job.player.maybeStart(Infinity);
  if (!seconds) return;
  if (wavUrl) URL.revokeObjectURL(wavUrl);
  wavUrl = URL.createObjectURL(new Blob([job.player.wav()], { type: 'audio/wav' }));
  ui.player.src = wavUrl;
  ui.download.href = wavUrl;
  ui.stats.textContent = `${seconds.toFixed(2)} s of audio in ${(ms / 1000).toFixed(1)} s · real-time factor ${rtf.toFixed(2)} · ${job.frames} frames`;
  ui.output.hidden = false;
}

/** An error the engine reported: loading ends; a synthesis error leaves the engine usable. */
function fail(message) {
  if (!loaded) {
    stopped = true;
    ui.loadTitle.textContent = 'Could not load the model';
    ui.loadDetail.textContent = message;
    return;
  }
  ui.genStatus.textContent = `error: ${message}`;
  ui.genBar.hidden = true;
  ui.generate.disabled = false;
  ui.stop.disabled = true;
}

/** The engine or a compute thread died: nothing more can run in this page. */
function crash(message) {
  stopped = true;
  engine.terminate();
  computePool.terminate();
  ui.generate.disabled = true;
  ui.stop.disabled = true;
  ui.genBar.hidden = true;
  const text = `The engine stopped (${message}). Reload the page to start again.`;
  if (loaded) ui.genStatus.textContent = text;
  else ui.loadDetail.textContent = text;
}

function updateNetwork() {
  const page = performance.getEntriesByType('resource').length - network.page;
  ui.network.textContent = `network requests since load: ${page + network.worker}`;
}

const mb = (bytes) => `${(bytes / 1e6).toFixed(0)} MB`;

/** Collects the PCM and plays it through Web Audio, chunk after chunk. */
class Player {
  #context = new AudioContext({ sampleRate: SAMPLE_RATE });
  #chunks = [];
  #scheduled = 0;
  #next = 0;
  #started = false;
  #length = 0;

  push(samples) {
    this.#chunks.push(samples);
    this.#length += samples.length;
    if (this.#started) this.#schedule();
  }

  seconds() {
    return this.#length / SAMPLE_RATE;
  }

  /** Starts playback when it can no longer run dry (`speed`: audio seconds per wall second). */
  maybeStart(speed) {
    if (this.#started || !this.#length) return;
    if (speed === Infinity || (speed >= STREAM_MIN_SPEED && this.seconds() >= STREAM_START_SECONDS)) {
      this.#started = true;
      this.#next = this.#context.currentTime + 0.05;
      this.#schedule();
    }
  }

  #schedule() {
    for (; this.#scheduled < this.#chunks.length; this.#scheduled++) {
      const samples = this.#chunks[this.#scheduled];
      const buffer = this.#context.createBuffer(1, samples.length, SAMPLE_RATE);
      buffer.copyToChannel(samples, 0);
      const source = this.#context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.#context.destination);
      this.#next = Math.max(this.#next, this.#context.currentTime + 0.02);
      source.start(this.#next);
      this.#next += buffer.duration;
    }
  }

  close() {
    this.#context.close();
  }

  /** 16-bit PCM WAV of everything pushed. */
  wav() {
    const out = new DataView(new ArrayBuffer(44 + this.#length * 2));
    const text = (offset, s) => [...s].forEach((c, i) => out.setUint8(offset + i, c.charCodeAt(0)));
    text(0, 'RIFF');
    out.setUint32(4, 36 + this.#length * 2, true);
    text(8, 'WAVEfmt ');
    out.setUint32(16, 16, true);
    out.setUint16(20, 1, true);
    out.setUint16(22, 1, true);
    out.setUint32(24, SAMPLE_RATE, true);
    out.setUint32(28, SAMPLE_RATE * 2, true);
    out.setUint16(32, 2, true);
    out.setUint16(34, 16, true);
    text(36, 'data');
    out.setUint32(40, this.#length * 2, true);
    let offset = 44;
    for (const chunk of this.#chunks) {
      for (const s of chunk) {
        out.setInt16(offset, Math.round(Math.max(-1, Math.min(1, s)) * 32767), true);
        offset += 2;
      }
    }
    return out.buffer;
  }
}
