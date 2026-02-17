#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

function arg(name, fallback = null) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0 || i + 1 >= process.argv.length) return fallback;
  return process.argv[i + 1];
}

function hasFlag(name) {
  return process.argv.includes(`--${name}`);
}

function toNum(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function slug(v) {
  return String(v || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'case';
}

function nowTag() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function choice(rng, arr) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  const i = Math.floor(rng() * arr.length);
  return arr[Math.max(0, Math.min(arr.length - 1, i))];
}

function quantized(rng, lo, hi, step = 0.05) {
  const s = Math.max(1e-6, Number(step) || 0.05);
  const n = lo + rng() * (hi - lo);
  return Math.round(n / s) * s;
}

function ensureDir(p) {
  mkdirSync(p, { recursive: true });
}

function writeJson(p, obj) {
  writeFileSync(p, `${JSON.stringify(obj, null, 2)}\n`, 'utf8');
}

function readJson(p) {
  return JSON.parse(readFileSync(p, 'utf8'));
}

function wait(ms) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, Math.max(0, Number(ms) || 0)));
}

async function loadPlaywright(repoRoot) {
  const candidates = [
    resolve(repoRoot, 'sim-server/node_modules/playwright-core/index.mjs'),
    resolve(repoRoot, 'sim-server/node_modules/playwright-core/index.js'),
  ];

  let lastErr = null;
  for (const c of candidates) {
    if (!existsSync(c)) continue;
    try {
      return await import(pathToFileURL(c).href);
    } catch (err) {
      lastErr = err;
    }
  }

  const msg = lastErr ? String(lastErr?.message || lastErr) : 'playwright-core not found';
  throw new Error([
    `Unable to load playwright-core (${msg}).`,
    'Install it once with:',
    '  cd sim-server && npm install --save-dev playwright-core',
  ].join('\n'));
}

function buildRandomPreset(rng, idx) {
  const fixtureType = choice(rng, ['rigid-line', 'soft-line']) || 'rigid-line';
  const motionMode = rng() < 0.7 ? 'pinned' : 'circle';
  const dyeChannel = choice(rng, ['r', 'g', 'b']) || 'r';
  const dyeMode = choice(rng, ['noop', 'eat']) || 'noop';
  const velocityMode = rng() < 0.7 ? 'block' : 'pass';
  const momentum = Math.max(0, Math.min(1, quantized(rng, 0.2, 1.0, 0.05)));
  const circleRadius = motionMode === 'circle' ? Math.max(8, Math.min(22, Math.round(quantized(rng, 8, 22, 1)))) : 12;
  const circleSpeed = motionMode === 'circle' ? Math.max(0.35, Math.min(1.8, quantized(rng, 0.35, 1.8, 0.05))) : 0.8;

  const id = `random-${String(idx + 1).padStart(2, '0')}`;
  return {
    id,
    name: `Random generated scenario ${idx + 1}`,
    description: `Auto-generated: ${fixtureType}, ${motionMode}, ${dyeChannel.toUpperCase()} ${dyeMode.toUpperCase()}, velocity ${velocityMode.toUpperCase()}.`,
    fixtureType,
    motionMode,
    circleRadius,
    circleSpeed,
    dyeChannel,
    dyeMode,
    velocityMode,
    momentum,
    source: 'random',
  };
}

function makeCases({ truthTable, randomCount, seed }) {
  const rng = mulberry32(seed >>> 0);
  const out = [];

  for (const s of truthTable) {
    out.push({
      id: String(s?.id || `scenario-${out.length + 1}`),
      name: s?.name || s?.id || `scenario-${out.length + 1}`,
      description: s?.description || '',
      fixtureType: s?.fixtureType || 'rigid-line',
      motionMode: s?.motionMode || 'pinned',
      circleRadius: toNum(s?.circleRadius, 12),
      circleSpeed: toNum(s?.circleSpeed, 0.8),
      dyeChannel: s?.dyeChannel || 'r',
      dyeMode: s?.dyeMode || 'noop',
      velocityMode: s?.velocityMode || 'block',
      momentum: toNum(s?.momentum, 1),
      source: 'truth-table',
    });
  }

  const randomN = Math.max(0, Math.floor(randomCount));
  for (let i = 0; i < randomN; i++) out.push(buildRandomPreset(rng, i));
  return out;
}

async function waitForGpuApi(frame, timeoutMs = 30000) {
  await frame.waitForFunction(() => {
    const api = window.__gpuLabApi;
    return !!api && typeof api.getStatus === 'function' && typeof api.captureCanvasMetrics === 'function';
  }, { timeout: timeoutMs });
}

async function waitForTick(frame, targetTick, timeoutMs) {
  await frame.waitForFunction((target) => {
    const api = window.__gpuLabApi;
    if (!api || typeof api.getStatus !== 'function') return false;
    const status = api.getStatus();
    const tick = Number(status?.frame) || 0;
    return tick >= target;
  }, targetTick, { timeout: timeoutMs });
}

async function applyPresetToInteractionLab(page, preset) {
  await page.selectOption('#fixtureType', String(preset.fixtureType || 'rigid-line'));
  await page.selectOption('#motionMode', String(preset.motionMode || 'pinned'));
  await page.selectOption('#dyeChannel', String(preset.dyeChannel || 'r'));
  await page.selectOption('#dyeMode', String(preset.dyeMode || 'noop'));
  await page.selectOption('#velocityMode', String(preset.velocityMode || 'block'));

  await page.fill('#circleRadius', String(toNum(preset.circleRadius, 12)));
  await page.fill('#circleSpeed', String(toNum(preset.circleSpeed, 0.8)));
  await page.fill('#momentum', String(toNum(preset.momentum, 1)));

  // Keep overlays visible for debug screenshots by default.
  await page.locator('#showSegmentIds').setChecked(true);
  await page.locator('#showExtraVisuals').setChecked(true);

  await page.click('#applyBtn');

  // Wait until payload panel is populated with JSON.
  await page.waitForFunction(() => {
    const text = document.querySelector('#scenarioOut')?.textContent || '';
    return text.includes('"truthTableRow"') && text.includes('"payload"');
  }, { timeout: 20000 });
}

async function readScenarioPanelJson(page) {
  let last = '';
  for (let i = 0; i < 20; i++) {
    const txt = await page.textContent('#scenarioOut');
    last = String(txt || '').trim();
    if (!last) {
      await wait(100);
      continue;
    }
    try {
      return JSON.parse(last);
    } catch {
      await wait(120);
    }
  }
  throw new Error(`Failed to parse #scenarioOut JSON. Last value: ${last.slice(0, 240)}`);
}

function writeManifest(manifestPath, entries) {
  const lines = entries.map((e) => JSON.stringify(e));
  writeFileSync(manifestPath, `${lines.join('\n')}\n`, 'utf8');
}

async function main() {
  const repoRoot = resolve(process.cwd(), '..');
  const baseUrl = String(arg('baseUrl', 'http://127.0.0.1:8787')).replace(/\/$/, '');
  const randomCount = Math.max(0, Math.floor(toNum(arg('randomCount', '6'), 6)));
  const tickEvery = Math.max(1, Math.floor(toNum(arg('tickEvery', '100'), 100)));
  const maxTick = Math.max(tickEvery, Math.floor(toNum(arg('maxTick', '1000'), 1000)));
  const tickTimeoutMs = Math.max(5_000, Math.floor(toNum(arg('tickTimeoutMs', '120000'), 120000)));
  const headed = hasFlag('headed');
  const seed = (Math.floor(toNum(arg('seed', String(Date.now() >>> 0)), Date.now())) >>> 0);

  const scenariosPath = resolve(repoRoot, 'sim-server/public/interaction-scenarios.json');
  const truthTable = readJson(scenariosPath);
  if (!Array.isArray(truthTable) || truthTable.length === 0) {
    throw new Error(`No truth-table scenarios found in ${scenariosPath}`);
  }

  // Quick reachability check.
  const healthUrl = `${baseUrl}/interaction-lab.html`;
  let healthOk = false;
  try {
    const res = await fetch(healthUrl, { method: 'GET' });
    healthOk = res.ok;
  } catch {}
  if (!healthOk) {
    throw new Error([
      `Interaction Lab is not reachable at ${healthUrl}`,
      'Start the sim server first, e.g.:',
      '  cd sim-server && npm start',
    ].join('\n'));
  }

  const batchName = slug(arg('name', `truth-table-plus-random-${randomCount}`));
  const outRoot = resolve(repoRoot, arg('out', `artifacts/interaction-pipeline/${nowTag()}-${batchName}`));
  const casesRoot = join(outRoot, 'cases');
  ensureDir(outRoot);
  ensureDir(casesRoot);

  const cases = makeCases({ truthTable, randomCount, seed });

  const manifest = cases.map((c, i) => ({
    caseIndex: i + 1,
    caseId: c.id,
    caseName: c.name,
    source: c.source,
    caseDir: join(casesRoot, `${String(i + 1).padStart(2, '0')}-${slug(c.id)}`),
    scenarioJson: 'scenario.json',
    scenarioConfigPath: 'scenario-config.json',
    screenshotPath: null,
    screenshotPaths: [],
    metricsPath: null,
    status: 'pending_capture',
    error: null,
  }));

  writeManifest(join(outRoot, 'manifest.jsonl'), manifest);

  const { chromium } = await loadPlaywright(repoRoot);
  const launchOptions = {
    headless: !headed,
    channel: 'chrome',
  };

  let browser = null;
  try {
    browser = await chromium.launch(launchOptions);
  } catch {
    // Fallback path for macOS Chrome app, in case channel launch is unavailable.
    browser = await chromium.launch({
      headless: !headed,
      executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    });
  }

  const context = await browser.newContext({ viewport: { width: 1500, height: 980 } });
  const page = await context.newPage();
  await page.goto(`${baseUrl}/interaction-lab.html`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#windFrame', { timeout: 30000 });

  const frameHandle = await page.$('#windFrame');
  const frame = await frameHandle?.contentFrame();
  if (!frame) throw new Error('Failed to acquire windFrame content frame.');
  await waitForGpuApi(frame, 40000);

  const targets = [];
  for (let t = tickEvery; t <= maxTick; t += tickEvery) targets.push(t);

  for (let i = 0; i < manifest.length; i++) {
    const entry = manifest[i];
    const preset = cases[i];
    const caseDir = entry.caseDir;
    const shotsDir = join(caseDir, 'screenshots');
    ensureDir(caseDir);
    ensureDir(shotsDir);

    const scenarioJsonPath = join(caseDir, 'scenario.json');
    const scenarioConfigPath = join(caseDir, 'scenario-config.json');
    const metricsPath = join(caseDir, 'metrics.json');

    const scenarioDoc = {
      caseIndex: entry.caseIndex,
      caseId: entry.caseId,
      caseName: entry.caseName,
      source: entry.source,
      createdAt: new Date().toISOString(),
      sourcePreset: preset,
      runConfig: {
        baseUrl,
        tickEvery,
        maxTick,
        tickTargets: targets,
      },
    };
    writeJson(scenarioJsonPath, scenarioDoc);

    try {
      await applyPresetToInteractionLab(page, preset);
      const panel = await readScenarioPanelJson(page);
      writeJson(scenarioConfigPath, panel);

      const captures = [];
      for (const targetTick of targets) {
        await waitForTick(frame, targetTick, tickTimeoutMs);

        const metrics = await frame.evaluate(() => {
          const api = window.__gpuLabApi;
          const status = api?.getStatus?.() || null;
          const frameMetrics = api?.captureCanvasMetrics?.() || null;
          return { status, frame: frameMetrics };
        });

        const actualTick = Math.max(targetTick, Number(metrics?.status?.frame) || 0);
        const shotName = `tick-${String(actualTick).padStart(6, '0')}.png`;
        const shotRel = `screenshots/${shotName}`;
        const shotAbs = join(caseDir, shotRel);

        await frame.locator('#view').screenshot({ path: shotAbs });

        captures.push({
          targetTick,
          actualTick,
          screenshotPath: shotRel,
          frame: metrics?.frame || null,
          status: metrics?.status || null,
          capturedAt: new Date().toISOString(),
        });
      }

      const last = captures[captures.length - 1] || null;
      const metricsDoc = {
        caseIndex: entry.caseIndex,
        caseId: entry.caseId,
        caseName: entry.caseName,
        source: entry.source,
        tickEvery,
        maxTick,
        captures,
        // Compatibility with analyzeBatch.mjs expectation.
        frame: {
          totalRGB: Number(last?.frame?.totalRGB) || 0,
          nonZeroPixels: Number(last?.frame?.nonZeroPixels) || 0,
          width: Number(last?.frame?.width) || 0,
          height: Number(last?.frame?.height) || 0,
        },
      };
      writeJson(metricsPath, metricsDoc);

      entry.screenshotPath = last?.screenshotPath || null;
      entry.screenshotPaths = captures.map((c) => c.screenshotPath);
      entry.metricsPath = 'metrics.json';
      entry.status = captures.length > 0 ? 'captured' : 'capture_failed';
      entry.error = captures.length > 0 ? null : 'no-captures';
    } catch (err) {
      entry.status = 'capture_failed';
      entry.error = String(err?.message || err);
      entry.metricsPath = null;
      entry.screenshotPath = null;
      entry.screenshotPaths = [];
    }

    writeManifest(join(outRoot, 'manifest.jsonl'), manifest);
  }

  await context.close();
  await browser.close();

  const summary = {
    ok: true,
    outRoot,
    baseUrl,
    totalCases: manifest.length,
    capturedCases: manifest.filter((m) => m.status === 'captured').length,
    failedCases: manifest.filter((m) => m.status !== 'captured').map((m) => ({
      caseIndex: m.caseIndex,
      caseId: m.caseId,
      error: m.error,
    })),
    tickEvery,
    maxTick,
    randomCount,
    seed,
  };

  writeJson(join(outRoot, 'capture-summary.json'), summary);
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((err) => {
  const msg = err && err.stack ? err.stack : String(err);
  console.error(msg);
  process.exitCode = 1;
});
