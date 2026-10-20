// End-to-end run in headless Chrome: open the page in a fresh profile, wait for the model, type a
// sentence, generate, and check that (1) audio came out, (2) no request was made after the model
// was ready, (3) the WAV the page offers is intelligible (faster-whisper WER, when the ASR venv
// exists). Then reload: the revisit must load every part from OPFS.
//
//   PORT=8787 node e2e/run.mjs [--threads 8] [--build web-threaded] [--preset quality] [--text "..."]
//   node e2e/run.mjs --url https://ming-wasm.fusor.build    # a deployed page, no local server
//
// `--model <url>` is the page's `?model=`: `model/` (dist/model) on the local server, the page's
// own default (the published bundle) with `--url`.
//
// Results (JSON, WAV, screenshots) go to e2e/results/<timestamp>-*.
// Env: ASR_PYTHON (default ../ming-05b-experiment/.venv-asr/bin/python).

import { chromium } from 'playwright';
import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const { values } = parseArgs({
  options: {
    threads: { type: 'string' },
    build: { type: 'string' },
    preset: { type: 'string', default: 'quality' },
    text: { type: 'string', default: 'Please remember to water the plants on Tuesday, and call the dentist about the appointment at three o\'clock.' },
    'max-wer': { type: 'string', default: '0.15' },
    url: { type: 'string' },
    model: { type: 'string' },
  },
});
const PORT = process.env.PORT ?? '8787';
const BASE = (values.url ?? `http://127.0.0.1:${PORT}`).replace(/\/$/, '');
const MODEL = values.model ?? (values.url ? undefined : 'model/');
const ASR_PYTHON = process.env.ASR_PYTHON ?? join(ROOT, '../ming-05b-experiment/.venv-asr/bin/python');
const LOAD_DEADLINE_MS = 10 * 60 * 1000;
const GENERATE_DEADLINE_MS = 10 * 60 * 1000;
const RUN_DIR = join(ROOT, 'e2e/results');
const STAMP = new Date().toISOString().replace(/[:.]/g, '-');
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const up = async () => {
  try {
    return (await fetch(`${BASE}/model/manifest.json`)).ok;
  } catch {
    return false;
  }
};
let server = null;
if (!values.url && !(await up())) {
  log(`starting fusor-ai's serve.mjs on :${PORT}`);
  const serve = join(ROOT, 'vendor/fusor-ai/scripts/serve.mjs');
  const mounts = ['--mount', '/=app/dist', '--mount', '/model/=dist/model'];
  server = spawn(process.execPath, [serve, '--port', PORT, ...mounts], { cwd: ROOT, stdio: 'inherit' });
  for (let i = 0; i < 50 && !(await up()); i++) await new Promise((r) => setTimeout(r, 200));
}

mkdirSync(RUN_DIR, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), 'ming-e2e-'));
// The page's state machine is `main[data-phase]`: loading, ready, generating, broken.
const settled = () => /^(ready|broken)$/.test(document.querySelector('main')?.dataset.phase ?? '');
const phase = (page) => page.getAttribute('main', 'data-phase');
const results = { started: new Date().toISOString(), base: BASE, preset: values.preset, text: values.text };
const failures = [];
const check = (ok, message) => {
  log(`${ok ? 'ok  ' : 'FAIL'} ${message}`);
  if (!ok) failures.push(message);
};

try {
  const context = await chromium.launchPersistentContext(profile, { headless: true });
  const requests = [];
  context.on('request', (r) => requests.push(r.url()));
  const page = context.pages()[0] ?? (await context.newPage());
  page.on('console', (m) => m.type() === 'error' && log('console:', m.text()));
  page.on('pageerror', (e) => log('pageerror:', e.message));
  const query = `?${new URLSearchParams(Object.entries({ threads: values.threads, build: values.build, model: MODEL }).filter(([, v]) => v))}`;

  const t0 = Date.now();
  await page.goto(`${BASE}/${query}`);
  await page.waitForFunction(settled, null, { timeout: LOAD_DEADLINE_MS, polling: 250 });
  results.loadMs = Date.now() - t0;
  results.runtime = await page.textContent('#runtime');
  check((await phase(page)) === 'ready', `model loaded in ${(results.loadMs / 1000).toFixed(1)} s (${results.runtime})`);
  await page.screenshot({ path: join(RUN_DIR, `${STAMP}-ready.png`), fullPage: true });

  const requestsAtReady = requests.length;
  await page.fill('#text', values.text);
  // The radios are visually hidden behind their labels.
  await page.check(`input[name=preset][value=${values.preset}]`, { force: true });
  const t1 = Date.now();
  await page.click('#generate');
  await page.waitForFunction(() => /^(done|stopped|error)/.test(document.getElementById('status').textContent), null, { timeout: GENERATE_DEADLINE_MS, polling: 250 });
  results.generateMs = Date.now() - t1;
  results.status = await page.textContent('#status');
  results.stats = await page.textContent('#stats');
  results.requestsDuringSynthesis = requests.slice(requestsAtReady).filter((u) => !u.startsWith('blob:'));
  check(results.status === 'done', `generation finished: ${results.stats}`);
  check(results.requestsDuringSynthesis.length === 0, `requests during synthesis: ${results.requestsDuringSynthesis.length}`);
  await page.screenshot({ path: join(RUN_DIR, `${STAMP}-done.png`), fullPage: true });

  const wavBase64 = await page.evaluate(async () => {
    const bytes = new Uint8Array(await (await fetch(document.getElementById('download').href)).arrayBuffer());
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(s);
  });
  const wavPath = join(RUN_DIR, `${STAMP}.wav`);
  writeFileSync(wavPath, Buffer.from(wavBase64, 'base64'));
  results.wavBytes = Buffer.from(wavBase64, 'base64').length;
  check(results.wavBytes > 44 + 44100, `WAV ${(results.wavBytes / 1e6).toFixed(2)} MB -> ${wavPath}`);

  if (existsSync(ASR_PYTHON)) {
    const out = execFileSync(ASR_PYTHON, [join(ROOT, 'scripts/asr_check.py'), '--text', values.text, wavPath], { encoding: 'utf8' });
    results.asr = out.trim().split('\n').at(-1);
    const wer = Number(/WER=([\d.]+)/.exec(results.asr)[1]);
    check(wer <= Number(values['max-wer']), `ASR ${results.asr}`);
  } else {
    log(`skipped ASR (no ${ASR_PYTHON})`);
  }

  const t2 = Date.now();
  await page.reload();
  await page.waitForFunction(settled, null, { timeout: LOAD_DEADLINE_MS, polling: 250 });
  results.revisitMs = Date.now() - t2;
  results.revisitDetail = await page.textContent('#runtime');
  check(/model loaded from this device/.test(results.revisitDetail), `revisit in ${(results.revisitMs / 1000).toFixed(1)} s: ${results.revisitDetail}`);
  await context.close();
} finally {
  rmSync(profile, { recursive: true, force: true });
  server?.kill();
}

results.failures = failures;
writeFileSync(join(RUN_DIR, `${STAMP}.json`), JSON.stringify(results, null, 2));
log(failures.length ? `e2e FAILED (${failures.length})` : 'e2e passed', `-> e2e/results/${STAMP}.json`);
process.exit(failures.length ? 1 : 0);
