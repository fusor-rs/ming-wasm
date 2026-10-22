// Runs benchmarks/kokoro/index.html (Kokoro-82M through kokoro-js, onnxruntime-web, wasm) in
// headless Chrome and writes the timings and the first WAV per sentence. The model and the
// library come from the network (jsdelivr, Hugging Face) on each run.
//
//   PORT=8796 node benchmarks/kokoro.mjs [--dtype q8|fp32] [--repeat 3] [--out benchmarks/out/kokoro]
import { chromium } from '../vendor/fusor-ai/packages/e2e/node_modules/playwright/index.mjs';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const { values } = parseArgs({ options: { dtype: { type: 'string', default: 'q8' }, repeat: { type: 'string', default: '3' }, out: { type: 'string', default: 'benchmarks/out/kokoro' } } });
const PORT = process.env.PORT ?? '8796';
const serve = join(ROOT, 'vendor/fusor-ai/scripts/serve.mjs');
const server = spawn(process.execPath, [serve, '--port', PORT, '--mount', '/=benchmarks/kokoro'], { cwd: ROOT, stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 800));
const outDir = join(ROOT, values.out);
mkdirSync(outDir, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), 'kokoro-bench-'));
try {
  const context = await chromium.launchPersistentContext(profile, { headless: true });
  const page = context.pages()[0] ?? (await context.newPage());
  page.on('console', (m) => m.type() === 'error' && console.log('console:', m.text()));
  await page.goto(`http://127.0.0.1:${PORT}/?dtype=${values.dtype}&repeat=${values.repeat}`);
  await page.waitForFunction(() => document.body.dataset.done === '1', null, { timeout: 900000, polling: 250 });
  const results = await page.evaluate(() => window.__results);
  console.log(await page.textContent('#log'));
  for (const [key, b64] of Object.entries(results.wavs)) writeFileSync(join(outDir, `kokoro-${values.dtype}-${key}.wav`), Buffer.from(b64, 'base64'));
  delete results.wavs;
  results.chrome = context.browser()?.version?.() ?? null;
  writeFileSync(join(outDir, `kokoro-${values.dtype}.json`), JSON.stringify(results, null, 2));
  await context.close();
} finally {
  rmSync(profile, { recursive: true, force: true });
  server.kill();
}
