import test from 'node:test';
import assert from 'node:assert/strict';
import { applySoftAreaXPBDVelocityGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftAreaXpbdGpuOnly.js';

const SOFT_AREA_XPBD_ITERS = 6;
const SOFT_AREA_BASE_COMPLIANCE = 0.0009;

function signedAreaPredicted(nodes, indices, dtPos) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    const ax = a.x + a.vx * dtPos;
    const ay = a.y + a.vy * dtPos;
    const bx = b.x + b.vx * dtPos;
    const by = b.y + b.vy * dtPos;
    s += ax * by - bx * ay;
  }
  return 0.5 * s;
}

function applySoftAreaXPBDVelocityBaseline(sim, soft, loops, dtPos, stiffnessScale) {
  if (!loops.length) return;
  const alpha = (SOFT_AREA_BASE_COMPLIANCE / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < SOFT_AREA_XPBD_ITERS; iter++) {
    for (const loop of loops) {
      const ids = loop.indices;
      const m = ids.length;
      if (m < 3) continue;
      const restArea = sim.softAreaRest.get(loop.clusterId);
      if (!Number.isFinite(restArea)) continue;

      const area = signedAreaPredicted(soft.nodes, ids, dtPos);
      const C = area - restArea;

      const gradX = new Array(m);
      const gradY = new Array(m);
      let sumWGrad2 = 0;

      for (let k = 0; k < m; k++) {
        const prev = soft.nodes[ids[(k - 1 + m) % m]];
        const next = soft.nodes[ids[(k + 1) % m]];
        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nx = next.x + next.vx * dtPos;
        const ny = next.y + next.vy * dtPos;
        const gx = 0.5 * (ny - py);
        const gy = 0.5 * (px - nx);
        gradX[k] = gx;
        gradY[k] = gy;

        const node = soft.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        sumWGrad2 += w * (gx * gx + gy * gy);
      }

      if (sumWGrad2 <= 1e-10) continue;

      const lambdaPrev = Number(sim.softAreaLambda.get(loop.clusterId)) || 0;
      let dl = (-C - alpha * lambdaPrev) / (sumWGrad2 + alpha);
      if (!Number.isFinite(dl)) continue;
      dl = Math.max(-2.0, Math.min(2.0, dl));
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      sim.softAreaLambda.set(loop.clusterId, lambdaNext);

      for (let k = 0; k < m; k++) {
        const node = soft.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        node.vx += (w * gradX[k] * dl) / dtPos;
        node.vy += (w * gradY[k] * dl) / dtPos;
      }
    }
  }
}

test('soft area XPBD parity: baseline stepping and gpu-only module produce matching soft states', () => {
  const dtPos = 0.16;
  const stiffnessScale = 3.4;
  const softSeed = {
    nodes: [
      { x: 14, y: 18, vx: 0.3, vy: -0.2, mass: 1.0, clusterId: 7 },
      { x: 22, y: 17, vx: -0.2, vy: 0.1, mass: 0.9, clusterId: 7 },
      { x: 26, y: 24, vx: 0.4, vy: 0.3, mass: 1.2, clusterId: 7 },
      { x: 19, y: 29, vx: -0.3, vy: -0.1, mass: 1.4, clusterId: 7 },
      { x: 34, y: 31, vx: 0.1, vy: -0.25, mass: 0.95, clusterId: 11 },
      { x: 41, y: 30, vx: -0.35, vy: 0.22, mass: 1.05, clusterId: 11 },
      { x: 39, y: 38, vx: 0.18, vy: 0.33, mass: 1.3, clusterId: 11 },
    ],
  };
  const loops = [
    { clusterId: 7, indices: [0, 1, 2, 3] },
    { clusterId: 11, indices: [4, 5, 6] },
    { clusterId: 99, indices: [0, 2] },
  ];

  const baselineSoft = structuredClone(softSeed);
  const gpuOnlySoft = structuredClone(softSeed);
  const baselineSim = {
    softAreaRest: new Map([[7, 44.2], [11, 29.4]]),
    softAreaLambda: new Map([[7, 0.07], [11, -0.04]]),
  };
  const gpuOnlySim = {
    softAreaRest: new Map([[7, 44.2], [11, 29.4]]),
    softAreaLambda: new Map([[7, 0.07], [11, -0.04]]),
  };
  const wgslOffload = { enabled: true, state: {} };

  applySoftAreaXPBDVelocityBaseline(baselineSim, baselineSoft, loops, dtPos, stiffnessScale);
  applySoftAreaXPBDVelocityGpuOnly({
    sim: gpuOnlySim,
    soft: gpuOnlySoft,
    loops,
    dtPos,
    stiffnessScale,
    softAreaXpbdIters: SOFT_AREA_XPBD_ITERS,
    softAreaBaseCompliance: SOFT_AREA_BASE_COMPLIANCE,
    wgslOffload,
  });

  assert.deepEqual(
    Array.from(gpuOnlySim.softAreaLambda.entries()),
    Array.from(baselineSim.softAreaLambda.entries()),
    'cluster lambda map should match baseline area stepping exactly',
  );

  assert.equal(gpuOnlySoft.nodes.length, baselineSoft.nodes.length);

  assert.equal(wgslOffload.state.lastMode, 'cpu-prepared', 'expected gpu-only area pass to publish WGSL-prepared layout mode');
  assert.equal(wgslOffload.state.lastPreparedClusterCount, 3, 'expected prepared cluster count to include all loops');
  assert.equal(wgslOffload.state.lastPreparedEndpointCount, 9, 'expected prepared endpoint count to match flattened loop endpoints');
  assert.ok((wgslOffload.state.lastPreparedLayoutBytes || 0) > 0, 'expected prepared WGSL layout byte footprint to be tracked');

  for (let i = 0; i < baselineSoft.nodes.length; i++) {
    const b = baselineSoft.nodes[i];
    const g = gpuOnlySoft.nodes[i];
    assert.ok(Math.abs(g.vx - b.vx) < 1e-12, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-12, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    assert.ok(Math.abs(g.x - b.x) < 1e-12, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-12, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
  }
});
