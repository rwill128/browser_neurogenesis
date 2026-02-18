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
  'node-harness/tests/softDeformationMetrics.test.mjs',
  'node-harness/tests/softClusterKinematics.test.mjs',
  'node-harness/tests/runtimeSolverRigidGpuOnlyParity.test.mjs',
  'node-harness/tests/softFluidCouplingGpuOnlyParity.test.mjs',
  'node-harness/tests/softIntegrateGpuOnlyParity.test.mjs',
  'node-harness/tests/rigidCollisionGpuOnlyParity.test.mjs',
  'node-harness/tests/rigidSoftCollisionGpuOnlyParity.test.mjs',
  'node-harness/tests/softMembranePressureGpuOnlyParity.test.mjs',
  'node-harness/tests/softSpringXpbdGpuOnlyParity.test.mjs',
  'node-harness/tests/softSpringXpbdGpuOnlySource.test.mjs',
  'node-harness/tests/softRestRecoveryGpuOnlyParity.test.mjs',
  'node-harness/tests/softRestRecoveryGpuOnlySource.test.mjs',
  'node-harness/tests/softAreaXpbdGpuOnlyParity.test.mjs',
  'node-harness/tests/softAreaXpbdGpuOnlySource.test.mjs',
  'node-harness/tests/bodyFluidInjectionGpuOnlySource.test.mjs',
  'node-harness/tests/hybridConstraintsGpuOnlySource.test.mjs',
  'node-harness/tests/softFluidCouplingGpuOnlySource.test.mjs',
  'node-harness/tests/gpuLabDeformationSource.test.mjs',
  'node-harness/tests/miniScenarioWatchdogSource.test.mjs',
];

const result = spawnSync(process.execPath, ['--test', ...tests], {
  stdio: 'inherit',
  cwd: process.cwd(),
});

process.exit(result.status ?? 1);
