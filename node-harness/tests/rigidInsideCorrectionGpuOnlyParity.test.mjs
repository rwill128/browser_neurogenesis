import test from 'node:test';
import assert from 'node:assert/strict';

import { applyRigidInsideCorrectionPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidInsideCorrectionGpuOnly.js';

function buildFixture() {
  const rigidBodies = [
    { x: 0, y: 0, mass: 2, insideCorrectionEnabled: true },
  ];
  const soft = {
    nodes: [
      { x: 0.15, y: 0.15, vx: -0.3, vy: -0.1, r: 0.4, mass: 1 },
      { x: 2.2, y: 2.2, vx: 0.1, vy: 0.05, r: 0.4, mass: 1 },
    ],
  };
  const square = [
    { x: -1, y: -1 },
    { x: 1, y: -1 },
    { x: 1, y: 1 },
    { x: -1, y: 1 },
  ];
  return {
    rigidBodies,
    soft,
    hybridAttachedByRigid: new Map(),
    correctionIters: 1,
    correctionSlop: 0.02,
    getRigidPolysWorld: () => [square],
  };
}

test('rigid-inside correction parity: wgsl-prep-enabled path preserves CPU authoritative behavior and emits deterministic prep telemetry', () => {
  const base = buildFixture();
  const withPrep = buildFixture();

  const baselineCorrected = applyRigidInsideCorrectionPassGpuOnly({ ...base, wgslOffload: null });

  const offload = { enabled: true, state: {} };
  const prepCorrected = applyRigidInsideCorrectionPassGpuOnly({ ...withPrep, wgslOffload: offload });

  assert.equal(prepCorrected, baselineCorrected);
  assert.deepEqual(withPrep.soft.nodes, base.soft.nodes);
  assert.equal(offload.state.lastPreparedInsideNodeCount, 2);
  assert.equal(offload.state.lastPreparedInsideRigidCount, 1);
  assert.equal(offload.state.lastPreparedInsidePolyCount, 1);
  assert.equal(offload.state.lastPreparedInsideEligibleNodeCount, 2);
  assert.ok(Number.isInteger(offload.state.lastPreparedInsideLayoutSignature));
  assert.ok((offload.state.lastPreparedInsideLayoutBytes || 0) > 0);
  assert.equal(offload.state.lastSourceRoute, 'cpu-rigid-inside-authoritative');
  assert.equal(offload.state.lastAuthoritativeInsideSource, 'cpu-rigid-inside-authoritative');
});
