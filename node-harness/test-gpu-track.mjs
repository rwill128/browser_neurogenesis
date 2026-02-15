import { spawnSync } from 'node:child_process';

const tests = [
  'node-harness/tests/creatureSpec.test.mjs',
  'node-harness/tests/creatureSpecV2Contract.test.mjs',
  'node-harness/tests/fieldToStructureCompiler.test.mjs',
  'node-harness/tests/rigidDecompositionQuality.test.mjs',
  'node-harness/tests/rigidConcaveCollision.test.mjs',
  'node-harness/tests/rigidRigidPolygonCollision.test.mjs',
  'node-harness/tests/dyeBarrierIntegration.test.mjs',
  'node-harness/tests/softXpbdTopologyGuards.test.mjs',
];

const result = spawnSync(process.execPath, ['--test', ...tests], {
  stdio: 'inherit',
  cwd: process.cwd(),
});

process.exit(result.status ?? 1);
