// Renders the social posts: `node assets/social/render.mjs` captures speed.html frame by frame
// (window.render(t)) into ming-wasm-speed.mp4 and memory.html into ming-wasm-memory.png. FFMPEG
// names the ffmpeg binary (default `ffmpeg`); Playwright comes from the e2e's node_modules.
import { chromium } from '../../vendor/fusor-ai/packages/e2e/node_modules/playwright/index.mjs';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = (file) => fileURLToPath(new URL(file, import.meta.url));
const FPS = 30;
const SECONDS = 13;
const stills = process.argv.includes('--stills');

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
await page.goto(`file://${here('speed.html')}`);
await page.evaluate(() => document.fonts.ready);
if (stills) {
  for (const t of [0.4, 3.5, 9.0]) {
    await page.evaluate((time) => window.render(time), t);
    await page.screenshot({ path: here(`still-${t}.png`) });
  }
} else {
  const frames = mkdtempSync(join(tmpdir(), 'ming-speed-'));
  for (let frame = 0; frame < FPS * SECONDS; frame++) {
    await page.evaluate((time) => window.render(time), frame / FPS);
    await page.screenshot({ path: join(frames, `${String(frame).padStart(4, '0')}.png`) });
  }
  execFileSync(process.env.FFMPEG ?? 'ffmpeg', [
    '-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', join(frames, '%04d.png'),
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    here('ming-wasm-speed.mp4'),
  ]);
  rmSync(frames, { recursive: true });
}
const card = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 2 });
await card.goto(`file://${here('memory.html')}`).catch(() => null);
if (!stills) await card.screenshot({ path: here('ming-wasm-memory.png') });
await browser.close();
