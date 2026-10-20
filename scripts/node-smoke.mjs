// Runs a single-threaded wasm runtime build in Node against a bundle: pushes every part the way
// the browser loader does, synthesizes one sentence, writes a WAV and prints the real-time factor.
//
//   node scripts/node-smoke.mjs dist/model --build web-single [--text "..."] [--out out.wav]

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    build: { type: 'string', default: 'web-single' },
    text: { type: 'string', default: 'The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.' },
    voice: { type: 'string', default: 'US_James' },
    preset: { type: 'string', default: 'quality' },
    out: { type: 'string' },
  },
});
const bundle = positionals[0] ?? 'dist/model';
const dist = new URL(`../packages/web/dist/${values.build}/`, import.meta.url);
const glue = await import(new URL('ming_runtime.js', dist));
glue.initSync({ module: readFileSync(new URL('ming_runtime_bg.wasm', dist)) });

const SAMPLE_RATE = 44100;
const CHUNK = 8 << 20;
const t0 = performance.now();
const manifest = readFileSync(join(bundle, 'manifest.json'), 'utf8');
const runtime = new glue.Runtime(manifest);
let ready = false;
JSON.parse(manifest).parts.forEach((part, index) => {
  const bytes = readFileSync(join(bundle, part.file));
  for (let o = 0; o < bytes.length; o += CHUNK) runtime.write_part(index, bytes.subarray(o, o + CHUNK));
  ready = runtime.end_part(index);
});
if (!ready) throw new Error('engine not ready after every part');
const loadMs = performance.now() - t0;

const t1 = performance.now();
runtime.start(JSON.stringify({ text: values.text, voice: { builtin: values.voice }, preset: values.preset }));
const chunks = [];
let frames = 0;
let firstAudioMs = null;
for (;;) {
  const progress = JSON.parse(runtime.step());
  frames++;
  const audio = runtime.take_audio();
  if (audio.length) {
    chunks.push(audio);
    firstAudioMs ??= performance.now() - t1;
  }
  if (progress.done) break;
}
const genMs = performance.now() - t1;
const samples = chunks.reduce((n, c) => n + c.length, 0);
const seconds = samples / SAMPLE_RATE;
console.log(`${values.build}: load ${(loadMs / 1000).toFixed(1)}s, ${frames} frames, ${seconds.toFixed(2)}s audio in ${(genMs / 1000).toFixed(1)}s: RTF ${(genMs / 1000 / seconds).toFixed(2)}, ${(genMs / frames).toFixed(0)} ms/frame, first audio ${(firstAudioMs / 1000).toFixed(2)}s`);

if (values.out) {
  const pcm = new Int16Array(samples);
  let o = 0;
  for (const c of chunks) for (const s of c) pcm[o++] = Math.round(Math.max(-1, Math.min(1, s)) * 32767);
  const header = Buffer.alloc(44);
  header.write('RIFF', 0); header.writeUInt32LE(36 + pcm.byteLength, 4); header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
  header.writeUInt32LE(SAMPLE_RATE, 24); header.writeUInt32LE(SAMPLE_RATE * 2, 28); header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34); header.write('data', 36); header.writeUInt32LE(pcm.byteLength, 40);
  writeFileSync(values.out, Buffer.concat([header, Buffer.from(pcm.buffer)]));
  console.log(`wrote ${values.out}`);
}
