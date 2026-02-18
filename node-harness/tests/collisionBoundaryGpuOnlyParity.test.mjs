import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCollisionBoundaryPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepCollisionBoundaryGpuOnly.js';

function makeState() {
  return {
    bodies: {
      rigid: [
        { x: -0.5, y: 3.1, vx: -0.8, vy: 0.2, omega: 0.1 },
        { x: 31.2, y: 30.6, vx: 0.7, vy: -1.1, omega: -0.08 },
      ],
    },
    soft: {
      nodes: [
        { x: 0.2, y: -0.4, vx: 0.5, vy: -0.6, clusterId: 1 },
        { x: 31.8, y: 12.5, vx: -0.3, vy: 0.4, clusterId: 2 },
        { x: 15.5, y: 31.1, vx: 0.1, vy: 0.9, clusterId: 2 },
      ],
    },
    n: 32,
  };
}

function bounceStub(body, n, bounce) {
  const min = 0.5;
  const max = n - 1.5;
  if (body.x < min) {
    body.x = min;
    if (body.vx < 0) body.vx = -body.vx * bounce;
  } else if (body.x > max) {
    body.x = max;
    if (body.vx > 0) body.vx = -body.vx * bounce;
  }
  if (body.y < min) {
    body.y = min;
    if (body.vy < 0) body.vy = -body.vy * bounce;
  } else if (body.y > max) {
    body.y = max;
    if (body.vy > 0) body.vy = -body.vy * bounce;
  }
}

function runBaselineInline(state) {
  for (const rb of state.bodies.rigid) bounceStub(rb, state.n, 0.84);
  for (const sn of state.soft.nodes) bounceStub(sn, state.n, 0.78);
}

test('collision boundary parity: gpu-only module matches baseline rigid+soft boundary sweep', () => {
  const baseline = makeState();
  const gpuOnly = makeState();

  runBaselineInline(baseline);
  applyCollisionBoundaryPassGpuOnly({
    rigidBodies: gpuOnly.bodies.rigid,
    soft: gpuOnly.soft,
    n: gpuOnly.n,
    rigidBounce: 0.84,
    softBounce: 0.78,
    applyBounceBoundary: bounceStub,
  });

  assert.deepEqual(gpuOnly, baseline);
});
