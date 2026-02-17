import test from 'node:test';
import assert from 'node:assert/strict';
import { integrateSoftBodiesGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftIntegrateGpuOnly.js';

function applyBounceBoundary(body, n, damping = 0.8) {
  const r = Number(body?.r) || 0;
  const min = r;
  const max = n - r;
  if (body.x < min) {
    body.x = min;
    body.vx = Math.abs(body.vx || 0) * damping;
  } else if (body.x > max) {
    body.x = max;
    body.vx = -Math.abs(body.vx || 0) * damping;
  }
  if (body.y < min) {
    body.y = min;
    body.vy = Math.abs(body.vy || 0) * damping;
  } else if (body.y > max) {
    body.y = max;
    body.vy = -Math.abs(body.vy || 0) * damping;
  }
}

function runBaselineSoftIntegration({ soft, n, dt, scale, cap }) {
  for (const node of soft.nodes) {
    const vmag = Math.hypot(node.vx, node.vy);
    if (vmag > cap) {
      node.vx = (node.vx / vmag) * cap;
      node.vy = (node.vy / vmag) * cap;
    }
    node.x = node.x + node.vx * dt * scale;
    node.y = node.y + node.vy * dt * scale;
    applyBounceBoundary(node, n, 0.78);
  }
}

test('soft integration parity: baseline loop and gpu-only module produce matching node states', () => {
  const n = 128;
  const dt = 0.013;
  const softIntegrationScale = 24;
  const hybridNodeVCap = 3.2;

  const seedNodes = [
    { x: 10, y: 12, vx: 2.2, vy: -1.8, r: 1.4 },
    { x: 126, y: 60, vx: 5.9, vy: 1.1, r: 2.0 },
    { x: 50, y: 3, vx: -0.8, vy: -6.2, r: 1.2 },
    { x: 90, y: 127, vx: 0.4, vy: 4.4, r: 1.5 },
  ];

  const baseline = { nodes: structuredClone(seedNodes) };
  const gpuOnly = { nodes: structuredClone(seedNodes) };

  runBaselineSoftIntegration({ soft: baseline, n, dt, scale: softIntegrationScale, cap: hybridNodeVCap });
  integrateSoftBodiesGpuOnly({
    soft: gpuOnly,
    n,
    dt,
    softIntegrationScale,
    hybridNodeVCap,
    applyBounceBoundary,
  });

  assert.equal(gpuOnly.nodes.length, baseline.nodes.length);
  for (let i = 0; i < baseline.nodes.length; i++) {
    const b = baseline.nodes[i];
    const g = gpuOnly.nodes[i];
    assert.ok(Math.abs(g.x - b.x) < 1e-9, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-9, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
    assert.ok(Math.abs(g.vx - b.vx) < 1e-9, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-9, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
  }
});
