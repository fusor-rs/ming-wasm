// Times the page in headless Chrome within one session: loads the model once, then synthesizes
// each sentence `--repeat` times and records, per run, the time to the first audio (the page's
// own "in N s" of the first progress line that has audio), the total time, the frames and the
// real-time factor. The first run's WAV of each sentence is kept for the intelligibility check.
//
//   PORT=8795 node benchmarks/browser.mjs [--build web-threaded-relaxed] [--threads 8] [--preset quality]
//                                        [--repeat 3] [--sentences S1,S2,S3] [--out benchmarks/out/browser]
//
// Companion of e2e/run.mjs (which measures a fresh visit, the revisit and the ASR WER per run).

import { chromium } from '../e2e/node_modules/playwright/index.mjs';
import { execFileSync, spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const SENTENCES = {
  S1: "Please remember to water the plants on Tuesday, and call the dentist about the appointment at three o'clock.",
  S2: 'The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.',
  S3: 'Glaciers move slowly, carving valleys over thousands of years while the rivers below carry the silt to the sea.',
};
const { values } = parseArgs({
  options: {
    threads: { type: 'string' },
    build: { type: 'string', default: 'web-threaded-relaxed' },
    preset: { type: 'string', default: 'quality' },
    repeat: { type: 'string', default: '3' },
    sentences: { type: 'string', default: 'S1,S2,S3' },
    out: { type: 'string', default: 'benchmarks/out/browser' },
  },
});
const PORT = process.env.PORT ?? '8795';
const BASE = `http://127.0.0.1:${PORT}`;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const up = async () => {
  try {
    return (await fetch(`${BASE}/model/manifest.json`)).ok;
  } catch {
    return false;
  }
};
let server = null;
if (!(await up())) {
  const serve = join(ROOT, 'vendor/fusor-ai/scripts/serve.mjs');
  server = spawn(process.execPath, [serve, '--port', PORT, '--mount', '/=app/dist', '--mount', '/model/=dist/model'], { cwd: ROOT, stdio: 'ignore' });
  for (let i = 0; i < 50 && !(await up()); i++) await new Promise((r) => setTimeout(r, 200));
}

const outDir = join(ROOT, values.out);
mkdirSync(outDir, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), 'ming-bench-'));
const tag = `${values.build}-t${values.threads ?? 'auto'}-${values.preset}`;
const results = { tag, build: values.build, threads: values.threads ?? 'auto', preset: values.preset, runs: [] };
try {
  const context = await chromium.launchPersistentContext(profile, { headless: true });
  const page = context.pages()[0] ?? (await context.newPage());
  page.on('pageerror', (e) => log('pageerror:', e.message));
  const query = `?${new URLSearchParams(Object.entries({ threads: values.threads, build: values.build }).filter(([, v]) => v))}`;
  const t0 = Date.now();
  await page.goto(`${BASE}/${query}`);
  await page.waitForFunction(() => /^(ready|broken)$/.test(document.querySelector('main')?.dataset.phase ?? ''), null, { timeout: 600000, polling: 100 });
  results.loadMs = Date.now() - t0;
  results.runtime = await page.textContent('#runtime');
  results.chrome = context.browser()?.version?.() ?? null;
  log(`loaded in ${(results.loadMs / 1000).toFixed(1)} s (${results.runtime})`);
  await page.check(`input[name=preset][value=${values.preset}]`, { force: true });

  for (const key of values.sentences.split(',')) {
    for (let i = 0; i < Number(values.repeat); i++) {
      await page.fill('#text', SENTENCES[key]);
      const t1 = Date.now();
      await page.click('#generate');
      // The first progress line with audio: "fragment 1/1 · frame N · 0.3 s of audio in 0.4 s".
      const first = await page.evaluate(
        () =>
          new Promise((resolve) => {
            const status = document.getElementById('status');
            const t = performance.now();
            const tick = () => {
              const m = /([\d.]+) s of audio in ([\d.]+) s/.exec(status.textContent);
              if (m && Number(m[1]) > 0) return resolve({ worker: Number(m[2]), wall: (performance.now() - t) / 1000 });
              if (/^(done|stopped|error)/.test(status.textContent)) return resolve(null);
              setTimeout(tick, 10);
            };
            tick();
          }),
      );
      await page.waitForFunction(() => /^(done|stopped|error)/.test(document.getElementById('status').textContent), null, { timeout: 600000, polling: 50 });
      const wallMs = Date.now() - t1;
      // Resident set of every Chrome process of this profile (the renderer, its workers, the GPU
      // and utility processes), right after synthesis, in MB.
      results.rss_mb ??= [];
      try {
        // The main process carries --user-data-dir=<profile>; the renderer, its workers and the
        // helper processes are its descendants.
        const rows = execFileSync('ps', ['-axo', 'pid=,ppid=,rss=,command='], { encoding: 'utf8' })
          .split('\n').map((l) => l.trim().split(/\s+/)).filter((f) => f.length >= 4)
          .map(([pid, ppid, rss, ...cmd]) => ({ pid, ppid, rss: Number(rss), cmd: cmd.join(' ') }));
        const main = rows.find((r) => r.cmd.includes(`--user-data-dir=${profile}`));
        const pids = new Set(main ? [main.pid] : []);
        for (let grew = true; grew; ) {
          grew = false;
          for (const r of rows) if (pids.has(r.ppid) && !pids.has(r.pid)) (pids.add(r.pid), (grew = true));
        }
        const rss = rows.filter((r) => pids.has(r.pid)).reduce((n, r) => n + r.rss, 0);
        results.rss_mb.push(Math.round(rss / 1024));
      } catch {}
      const stats = await page.textContent('#stats');
      const m = /([\d.]+) s of audio in ([\d.]+) s · real-time factor ([\d.]+) · (\d+) frames/.exec(stats);
      const run = {
        sentence: key, run: i, audio_s: Number(m[1]), gen_s: Number(m[2]), rtf: Number(m[3]), frames: Number(m[4]),
        ms_per_frame: Math.round((1000 * Number(m[2])) / Number(m[4]) * 10) / 10,
        first_audio_s: first?.worker ?? null, first_audio_wall_s: first ? Math.round(first.wall * 100) / 100 : null,
        wall_s: wallMs / 1000,
      };
      results.runs.push(run);
      log(`${key} run ${i}: ${run.frames} frames, ${run.audio_s} s audio in ${run.gen_s} s, RTF ${run.rtf}, first audio ${run.first_audio_s} s`);
      if (i === 0) {
        const b64 = await page.evaluate(async () => {
          const bytes = new Uint8Array(await (await fetch(document.getElementById('download').href)).arrayBuffer());
          let s = '';
          for (let o = 0; o < bytes.length; o += 0x8000) s += String.fromCharCode(...bytes.subarray(o, o + 0x8000));
          return btoa(s);
        });
        writeFileSync(join(outDir, `${tag}-${key}.wav`), Buffer.from(b64, 'base64'));
      }
    }
  }
  await context.close();
} finally {
  rmSync(profile, { recursive: true, force: true });
  server?.kill();
}
writeFileSync(join(outDir, `${tag}.json`), JSON.stringify(results, null, 2));
log(`-> ${join(values.out, `${tag}.json`)}`);
