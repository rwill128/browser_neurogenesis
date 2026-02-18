import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js'), 'utf8');

test('body-fluid injection gpu-only builds deterministic gather layout metadata for upcoming WGSL reduction stage', () => {
  assert.match(
    source,
    /function buildBodyFluidInjectionGatherLayout\(\{[\s\S]*cellOffsets = new Uint32Array\(cellCount \+ 1\);[\s\S]*contribPointIndex = new Uint32Array\(totalContrib\);[\s\S]*contribWeight = new Float32Array\(totalContrib\);/,
    'expected gather layout helper with deterministic CSR-style cell offsets and contribution streams',
  );

  assert.match(
    source,
    /wgslOffload\.state\.preparedGatherLayout = gatherLayout;[\s\S]*wgslOffload\.state\.lastPreparedGatherContributionCount = gatherLayout\.contributionCount;/,
    'expected cpu-prepared offload state to publish gather layout telemetry for next WGSL stage',
  );

  assert.match(
    source,
    /Gather layout now exists for a deterministic[\s\S]*reduction shader itself is still pending\./,
    'expected explicit blocker annotation tying this prep task to the next WGSL milestone',
  );
});
