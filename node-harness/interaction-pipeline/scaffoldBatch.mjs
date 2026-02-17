#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';

function arg(name, fallback = null) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0 || i + 1 >= process.argv.length) return fallback;
  return process.argv[i + 1];
}

function slug(v) {
  return String(v || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'batch';
}

function nowTag() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

const repoRoot = resolve(process.cwd(), '..');
const scenariosPath = resolve(repoRoot, 'sim-server/public/interaction-scenarios.json');
const scenarios = JSON.parse(readFileSync(scenariosPath, 'utf8'));
if (!Array.isArray(scenarios) || scenarios.length === 0) {
  throw new Error(`No scenarios found in ${scenariosPath}`);
}

const batchName = slug(arg('name', 'truth-table'));
const outRoot = resolve(repoRoot, arg('out', `artifacts/interaction-pipeline/${nowTag()}-${batchName}`));
mkdirSync(outRoot, { recursive: true });
mkdirSync(join(outRoot, 'cases'), { recursive: true });

const createdAt = new Date().toISOString();
const manifestLines = [];
const verdicts = [];

for (let i = 0; i < scenarios.length; i++) {
  const s = scenarios[i];
  const id = String(s?.id || `scenario-${i + 1}`);
  const caseDir = join(outRoot, 'cases', `${String(i + 1).padStart(2, '0')}-${slug(id)}`);
  mkdirSync(caseDir, { recursive: true });

  const scenarioDoc = {
    caseIndex: i + 1,
    caseId: id,
    caseName: s?.name || id,
    createdAt,
    sourcePreset: s,
    runConfig: {
      settleMs: 6500,
      captureMode: 'fullPage',
      notes: 'Capture stage should load this preset in interaction-lab, wait settleMs, then save screenshot + metrics.json.',
    },
  };

  writeFileSync(join(caseDir, 'scenario.json'), JSON.stringify(scenarioDoc, null, 2));

  const manifestEntry = {
    caseIndex: i + 1,
    caseId: id,
    caseName: s?.name || id,
    caseDir,
    scenarioJson: 'scenario.json',
    screenshotPath: null,
    metricsPath: null,
    status: 'pending_capture',
  };
  manifestLines.push(JSON.stringify(manifestEntry));

  verdicts.push({
    caseIndex: i + 1,
    caseId: id,
    verdict: 'PENDING',
    confidence: null,
    notes: '',
    expected: '',
    observed: '',
    screenshotPath: null,
  });
}

writeFileSync(join(outRoot, 'manifest.jsonl'), `${manifestLines.join('\n')}\n`);
writeFileSync(join(outRoot, 'verdicts.json'), JSON.stringify(verdicts, null, 2));

const analysisTemplate = `# Interaction Truth-Table Analysis\n\nBatch: ${outRoot}\nCreated: ${createdAt}\n\n## Rubric\n- Contract alignment\n- Visual plausibility\n- Segment interaction timing\n- Artifact quality (non-blank frame)\n\n## Case Notes\n\n${scenarios.map((s, i) => `### ${i + 1}. ${s?.name || s?.id}\n- Case ID: \`${s?.id}\`\n- Expected:\n- Observed:\n- Verdict: PENDING\n- Confidence:\n- Screenshot:\n- Metrics:\n`).join('\n')}\n`;

writeFileSync(join(outRoot, 'analysis.md'), analysisTemplate);

const runInfo = {
  ok: true,
  outRoot,
  cases: scenarios.length,
  files: ['manifest.jsonl', 'verdicts.json', 'analysis.md', 'cases/*/scenario.json'],
};
console.log(JSON.stringify(runInfo, null, 2));
