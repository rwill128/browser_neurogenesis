import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js'), 'utf8');

test('body-fluid injection gpu-only wires deterministic gather layout metadata into a WGSL gather proposal dispatch', () => {
  assert.match(
    source,
    /function buildBodyFluidInjectionGatherLayout\(\{[\s\S]*cellOffsets = new Uint32Array\(cellCount \+ 1\);[\s\S]*contribPointIndex = new Uint32Array\(totalContrib\);[\s\S]*contribWeight = new Float32Array\(totalContrib\);/,
    'expected gather layout helper with deterministic CSR-style cell offsets and contribution streams',
  );

  assert.match(
    source,
    /wgslOffload\.state\.preparedGatherLayout = gatherLayout;[\s\S]*wgslOffload\.state\.lastPreparedGatherContributionCount = gatherLayout\.contributionCount;/,
    'expected offload state to publish gather layout telemetry for WGSL gather proposal dispatch',
  );

  assert.match(
    source,
    /dispatchBodyFluidInjectionGatherProposal\(\{[\s\S]*gatherLayout,[\s\S]*couplingLimit,[\s\S]*n,[\s\S]*\}\);/,
    'expected gather proposal dispatch to consume deterministic gather layout streams',
  );

  assert.match(
    source,
    /wgslOffload\.state\.lastMode = wgslGatherRan \? 'wgsl-gather-proposal' : 'cpu-prepared';/,
    'expected mode telemetry to report wgsl gather proposal dispatch versus cpu-prepared fallback',
  );
});
