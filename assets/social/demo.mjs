// Records the demo post: the real page, served locally with dist/model, types a sentence, picks the
// balanced preset and speaks it. Chrome's screencast gives the frames; the page's own WAV, placed
// where its player started it, gives the sound; endcard.html closes the video. FFMPEG names ffmpeg.
//
//   FFMPEG=<path> node assets/social/demo.mjs   ->  assets/social/ming-wasm-demo.mp4
import { chromium } from '../../e2e/node_modules/playwright/index.mjs';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = (file) => fileURLToPath(new URL(file, import.meta.url));
const root = fileURLToPath(new URL('../../', import.meta.url));
const FFMPEG = process.env.FFMPEG ?? 'ffmpeg';
const PORT = 8797;
const PAGE = `http://127.0.0.1:${PORT}/?model=model/`;
const TEXT = "Hi, I'm Ming. I'm speaking to you from inside this browser tab, with no server and no GPU, just your computer's CPU.";
const FPS = 30;
const ENDCARD_SECONDS = 3;
const step = (message) => console.log(new Date().toISOString().slice(11, 19), message);
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Drawn cursor (headless Chrome renders none) and the moment the page's player starts the audio.
const INSTRUMENT = () => {
  const start = AudioBufferSourceNode.prototype.start;
  AudioBufferSourceNode.prototype.start = function (when = 0, ...rest) {
    if (window.audioStartedAt === undefined) {
      window.audioStartedAt = Date.now() + Math.max(0, when - this.context.currentTime) * 1000;
    }
    return start.call(this, when, ...rest);
  };
  addEventListener('DOMContentLoaded', () => {
    const cursor = document.createElement('div');
    cursor.innerHTML = '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2l18 10-8 2.5L9.5 23z" fill="#272a2c" stroke="#faf9f6" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(cursor.style, {
      position: 'fixed', left: '0', top: '0', zIndex: 99, pointerEvents: 'none',
      transform: 'translate(760px, 470px)', transition: 'transform 0.7s cubic-bezier(.2,.8,.2,1)',
    });
    document.body.append(cursor);
    window.moveCursor = (x, y) => (cursor.style.transform = `translate(${x - 4}px, ${y - 3}px)`);
  });
};

async function clickOn(page, selector) {
  const box = await page.locator(selector).boundingBox();
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await page.evaluate(([cx, cy]) => window.moveCursor(cx, cy), [x, y]);
  await pause(800);
  await page.mouse.click(x, y);
  await pause(250);
}

const ready = () => document.querySelector('main')?.dataset.phase === 'ready';
const server = spawn(process.execPath, [join(root, 'vendor/fusor-ai/scripts/serve.mjs'), '--port', String(PORT),
  '--mount', '/=app/dist', '--mount', '/model/=dist/model'], { cwd: root, stdio: 'ignore' });
const profile = mkdtempSync(join(tmpdir(), 'ming-demo-'));
const work = mkdtempSync(join(tmpdir(), 'ming-demo-frames-'));
try {
  await pause(800);
  const context = await chromium.launchPersistentContext(profile, {
    headless: true, viewport: { width: 960, height: 540 }, deviceScaleFactor: 2,
    args: ['--autoplay-policy=no-user-gesture-required'],
  });
  step('warming the cache');
  const warm = await context.newPage();
  warm.on('console', (m) => step(`warm console: ${m.text()}`));
  await warm.goto(PAGE);
  await warm.waitForFunction(ready, null, { timeout: 600000 });
  await warm.close();

  const page = await context.newPage();
  await page.addInitScript(INSTRUMENT);
  const cdp = await context.newCDPSession(page);
  const frames = [];
  cdp.on('Page.screencastFrame', async ({ data, metadata, sessionId }) => {
    frames.push({ data, at: metadata.timestamp * 1000 });
    await cdp.send('Page.screencastFrameAck', { sessionId });
  });
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 95, maxWidth: 1920, maxHeight: 1080 });
  step('recording');
  page.on('pageerror', (e) => step(`pageerror: ${e.message}`));
  await page.goto(PAGE);
  await page.waitForFunction(ready, null, { timeout: 600000 });
  await pause(1200);

  step('typing');
  await clickOn(page, '#text');
  await page.keyboard.press('Meta+A');
  await page.keyboard.press('Backspace');
  await page.keyboard.type(TEXT, { delay: 28 });
  await pause(500);
  await page.evaluate(() => document.querySelector('#generate').scrollIntoView({ behavior: 'smooth', block: 'center' }));
  await pause(1100);
  await clickOn(page, 'label:has(input[name=preset][value=balanced]) span');
  await clickOn(page, '#generate');
  step('generating');
  await page.waitForFunction(() => /^(done|stopped|error)/.test(document.getElementById('status').textContent), null, { timeout: 600000 });
  await pause(400);
  await page.evaluate(() => document.querySelector('.result').scrollIntoView({ behavior: 'smooth', block: 'center' }));
  const wav = Buffer.from(await page.evaluate(async () => {
    const bytes = new Uint8Array(await (await fetch(document.getElementById('download').href)).arrayBuffer());
    let text = '';
    for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(text);
  }), 'base64');
  const audioStartedAt = await page.evaluate(() => window.audioStartedAt);
  const audioSeconds = (wav.length - 44) / 2 / 44100;
  const tail = audioStartedAt + audioSeconds * 1000 + 1200 - Date.now();
  await pause(Math.max(tail, 1500));
  await cdp.send('Page.stopScreencast');
  const stoppedAt = Date.now();
  console.log(await page.textContent('#stats'));
  await context.close();

  // The screencast only sends a frame when the page changes: each frame lasts until the next.
  const list = frames.map((frame, index) => {
    const file = join(work, `${String(index).padStart(5, '0')}.jpg`);
    writeFileSync(file, Buffer.from(frame.data, 'base64'));
    const next = frames[index + 1]?.at ?? stoppedAt;
    return `file '${file}'\nduration ${((next - frame.at) / 1000).toFixed(4)}`;
  });
  writeFileSync(join(work, 'frames.txt'), `${list.join('\n')}\nfile '${join(work, `${String(frames.length - 1).padStart(5, '0')}.jpg`)}'\n`);
  writeFileSync(join(work, 'speech.wav'), wav);

  const card = await chromium.launch();
  const cardPage = await card.newPage({ viewport: { width: 1920, height: 1080 } });
  await cardPage.goto(`file://${here('endcard.html')}`);
  await cardPage.screenshot({ path: join(work, 'endcard.png') });
  await card.close();

  const delay = Math.round(audioStartedAt - frames[0].at);
  const total = ((stoppedAt - frames[0].at) / 1000 + ENDCARD_SECONDS).toFixed(3);
  step('encoding');
  execFileSync(FFMPEG, [
    '-y', '-loglevel', 'error',
    '-f', 'concat', '-safe', '0', '-i', join(work, 'frames.txt'),
    '-loop', '1', '-t', String(ENDCARD_SECONDS), '-i', join(work, 'endcard.png'),
    '-i', join(work, 'speech.wav'),
    '-filter_complex',
    `[0:v]scale=1920:1080:out_range=tv,fps=${FPS},format=yuv420p,setsar=1[page];` +
    `[1:v]scale=1920:1080:out_range=tv,fps=${FPS},format=yuv420p,setsar=1,fade=in:st=0:d=0.4[card];` +
    `[page][card]concat=n=2:v=1:a=0[video];` +
    `[2:a]adelay=${delay}|${delay},apad=whole_dur=${total}[audio]`,
    '-map', '[video]', '-map', '[audio]', '-t', total,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-color_range', 'tv',
    '-c:a', 'aac', '-b:a', '192k',
    '-movflags', '+faststart', here('ming-wasm-demo.mp4'),
  ]);
  console.log(`audio starts ${(delay / 1000).toFixed(2)} s into the video; ${audioSeconds.toFixed(2)} s of speech; ${frames.length} frames`);
} finally {
  server.kill();
  rmSync(profile, { recursive: true, force: true });
  rmSync(work, { recursive: true, force: true });
}
