import { spawnSync } from 'node:child_process';

const tests = [
  'node-harness/tests/creatureSpec.test.mjs',
  'node-harness/tests/fieldToStructureCompiler.test.mjs',
  'node-harness/tests/rigidWeldSolver.test.mjs',
];

const result = spawnSync(process.execPath, ['--test', ...tests], {
  stdio: 'inherit',
  cwd: process.cwd(),
});

process.exit(result.status ?? 1);
