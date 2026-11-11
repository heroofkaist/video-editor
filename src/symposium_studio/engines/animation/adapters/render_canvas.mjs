import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '');
    args[key] = argv[i + 1];
  }
  return args;
}

const args = parseArgs(process.argv);
const htmlPath = path.resolve(args.html);
const outDir = path.resolve(args.out);
const width = Number(args.width || 1920);
const height = Number(args.height || 1080);
const fps = Number(args.fps || 30);
const durationSec = Number(args.duration || 6);
const omitBackground = String(args.transparent || 'true') === 'true';
const pwModule = path.resolve(args.pwModule);
const executablePath = args.executable && args.executable.length > 0 ? args.executable : undefined;

fs.mkdirSync(outDir, { recursive: true });

const pw = await import(pathToFileURL(pwModule).href);
const chromium = pw.chromium || pw.default?.chromium;
if (!chromium) {
  throw new Error(`Could not resolve chromium export from ${pwModule}`);
}

const launchOpts = { headless: true };
if (executablePath) {
  launchOpts.executablePath = executablePath;
}

const browser = await chromium.launch(launchOpts);
try {
  const page = await browser.newPage({ viewport: { width, height } });
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'load' });
  const totalFrames = Math.max(1, Math.round(durationSec * fps));
  for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
    const tSec = frameIdx / fps;
    await page.evaluate(
      ([t, dur]) => window.renderAtTime(t, dur),
      [tSec, durationSec],
    );
    const frameFile = path.join(outDir, `frame_${String(frameIdx).padStart(5, '0')}.png`);
    await page.screenshot({ path: frameFile, omitBackground });
  }
} finally {
  await browser.close();
}
