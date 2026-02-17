#!/usr/bin/env node
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';

function arg(name, fallback = null) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0 || i + 1 >= process.argv.length) return fallback;
  return process.argv[i + 1];
}

const batchDir = resolve(process.cwd(), arg('batch', ''));
if (!batchDir || !existsSync(batchDir)) {
  throw new Error('Pass --batch <artifacts/interaction-pipeline/...>');
}

const manifestPath = join(batchDir, 'manifest.jsonl');
if (!existsSync(manifestPath)) throw new Error(`Missing ${manifestPath}`);

const lines = readFileSync(manifestPath, 'utf8').split('\n').map((l) => l.trim()).filter(Boolean);
const entries = lines.map((l) => JSON.parse(l));

const verdicts = [];
const notes = [];

for (const e of entries) {
  const caseDir = e.caseDir;
  const scenarioPath = join(caseDir, e.scenarioJson || 'scenario.json');
  const metricsPath = e.metricsPath ? join(caseDir, e.metricsPath) : null;
  const screenshotPath = e.screenshotPath ? join(caseDir, e.screenshotPath) : null;

  const scenario = existsSync(scenarioPath) ? JSON.parse(readFileSync(scenarioPath, 'utf8')) : null;
  const metrics = metricsPath && existsSync(metricsPath) ? JSON.parse(readFileSync(metricsPath, 'utf8')) : null;

  const hasScreenshot = !!(screenshotPath && existsSync(screenshotPath));
  const nonBlank = Number(metrics?.frame?.totalRGB || 0) > 1000;
  const captureReady = hasScreenshot && nonBlank;

  const verdict = captureReady ? 'NEEDS_HUMAN_REVIEW' : 'INCOMPLETE_ARTIFACTS';
  const reason = captureReady
    ? 'Artifacts present; run visual contract analysis.'
    : `Missing or weak artifacts (screenshot=${hasScreenshot}, nonBlank=${nonBlank}).`;

  verdicts.push({
    caseIndex: e.caseIndex,
    caseId: e.caseId,
    verdict,
    reason,
    screenshotPath: hasScreenshot ? screenshotPath : null,
    metricsPath: metricsPath && existsSync(metricsPath) ? metricsPath : null,
  });

  notes.push(`### ${e.caseIndex}. ${scenario?.caseName || e.caseName || e.caseId}\n- Case ID: \`${e.caseId}\`\n- Status: **${verdict}**\n- Reason: ${reason}\n- Screenshot: ${hasScreenshot ? `\`${screenshotPath}\`` : 'MISSING'}\n- Metrics: ${metricsPath && existsSync(metricsPath) ? `\`${metricsPath}\`` : 'MISSING'}\n`);
}

writeFileSync(join(batchDir, 'verdicts.json'), JSON.stringify(verdicts, null, 2));
const report = `# Interaction Truth-Table Analysis\n\nBatch: ${batchDir}\nGenerated: ${new Date().toISOString()}\n\n${notes.join('\n')}\n`;
writeFileSync(join(batchDir, 'analysis.md'), report);

console.log(JSON.stringify({ ok: true, batchDir, cases: verdicts.length }, null, 2));
