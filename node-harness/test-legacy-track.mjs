import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';

const all = readdirSync('node-harness/tests').filter((f) => f.endsWith('.test.mjs'));
const gpuSet = new Set([
  'creatureSpec.test.mjs',
  'fieldToStructureCompiler.test.mjs',
]);
const legacyTests = all
  .filter((f) => !gpuSet.has(f))
  .map((f) => `node-harness/tests/${f}`)
  .sort();

const result = spawnSync(process.execPath, ['--test', ...legacyTests], {
  stdio: 'inherit',
  cwd: process.cwd(),
});

process.exit(result.status ?? 1);
