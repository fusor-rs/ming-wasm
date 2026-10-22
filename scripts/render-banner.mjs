// Renders assets/brand/banner.html into banner.png and banner-dark.png with the e2e's Playwright
// (`cd vendor/fusor-ai/packages/e2e && npm ci` first).
import { chromium } from '../vendor/fusor-ai/packages/e2e/node_modules/playwright/index.mjs';
import { fileURLToPath } from 'node:url';

const page = fileURLToPath(new URL('../assets/brand/banner.html', import.meta.url));
const browser = await chromium.launch();
const tab = await browser.newPage({ viewport: { width: 1280, height: 400 }, deviceScaleFactor: 2 });
for (const [theme, file] of [['light', 'banner.png'], ['dark', 'banner-dark.png']]) {
  await tab.goto(`file://${page}?theme=${theme}`);
  await tab.screenshot({ path: fileURLToPath(new URL(`../assets/brand/${file}`, import.meta.url)) });
}
await browser.close();
