import test from 'node:test';
import assert from 'node:assert/strict';

import { stabilizeRigidPostIntegrateGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidPostIntegrateGpuOnly.js';

function makeBodies() {
  return [
    { x: 2, y: 3, vx: 7.2, vy: -1.1, omega: 0.41 },
    { x: 4, y: 9, vx: -2.4, vy: 1.8, omega: -0.77 },
    { x: 8, y: 1, vx: 0.2, vy: -0.15, omega: 0.06 },
    { x: 5, y: 7, vx: Number.NaN, vy: Infinity, omega: -Infinity },
  ];
}

function baselineRigidPostIntegrateClamp(rigidBodies, velocityCap = 4.0, omegaCap = 0.22) {
  for (const rb of rigidBodies) {
    const vmag = Math.hypot(rb.vx, rb.vy);
    if (vmag > velocityCap) {
      rb.vx = (rb.vx / vmag) * velocityCap;
      rb.vy = (rb.vy / vmag) * velocityCap;
    }
    rb.omega = Math.max(-omegaCap, Math.min(omegaCap, rb.omega || 0));
  }
}

test('gpu-only rigid post-integrate stabilization matches baseline clamp semantics', () => {
  const baselineBodies = makeBodies();
  const gpuBodies = makeBodies();

  baselineRigidPostIntegrateClamp(baselineBodies, 4.0, 0.22);
  stabilizeRigidPostIntegrateGpuOnly({ rigidBodies: gpuBodies, velocityCap: 4.0, omegaCap: 0.22 });

  assert.deepEqual(gpuBodies, baselineBodies);
});
