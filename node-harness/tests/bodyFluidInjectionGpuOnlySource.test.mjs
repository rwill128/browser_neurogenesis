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
    /void dispatchBodyFluidInjectionGatherProposal\(\{[\s\S]*gatherLayout,[\s\S]*couplingLimit,[\s\S]*n,[\s\S]*\}\)\.then\(/,
    'expected gather proposal dispatch to consume deterministic gather layout streams via async WGSL readback pipeline',
  );

  assert.match(
    source,
    /function computeBodyFluidInjectionCellDeltasFromGatherLayout\(\{ gatherLayout, couplingLimit, n \}\)[\s\S]*cellDeltaVx\[cell\] = clampComponent\(sumX, couplingLimit\);[\s\S]*cellDeltaVy\[cell\] = clampComponent\(sumY, couplingLimit\);/,
    'expected deterministic CPU gather-delta helper that mirrors WGSL gather proposal ownership',
  );

  assert.match(
    source,
    /wgslOffload\.state\.lastCpuGatherDeltaVx = cpuGatherDelta\.cellDeltaVx;[\s\S]*wgslOffload\.state\.lastCpuGatherDeltaVy = cpuGatherDelta\.cellDeltaVy;/,
    'expected offload telemetry to publish CPU gather delta arrays for upcoming WGSL parity/readback checks',
  );

  assert.match(
    source,
    /encoder\.copyBufferToBuffer\(state\.gatherProposalDeltaVx, 0, state\.gatherProposalDeltaVxReadback, 0, cellBytes\)[\s\S]*mapAsync\(globalThis\.GPUMapMode\.READ, 0, cellBytes\)[\s\S]*state\.lastGatherProposalDeltaVx = deltaVx;[\s\S]*state\.lastGatherProposalDeltaVy = deltaVy;/,
    'expected WGSL gather proposal stage to read back deterministic cell-delta telemetry for parity checks',
  );

  assert.match(
    source,
    /if \(wgslOffload\.state\.wgslInFlight\) \{[\s\S]*wgslSkippedWhileBusy[\s\S]*\} else \{[\s\S]*wgslInFlight = true;[\s\S]*lastCompletedWgslRunId/,
    'expected gpu-only body-fluid WGSL dispatch to serialize map/readback work and avoid overlapping mapAsync races',
  );
});
