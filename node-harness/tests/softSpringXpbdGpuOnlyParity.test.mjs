import test from 'node:test';
import assert from 'node:assert/strict';
import { applySoftSpringsXPBDVelocityGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js';

const SOFT_XPBD_ITERS = 10;
const SOFT_XPBD_BASE_COMPLIANCE = 0.0012;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function applySoftSpringsXPBDVelocityBaseline(soft, dtPos, stiffnessScale, lambdaCache, { skipClusterSet = null } = {}) {
  if (!soft?.nodes?.length || !soft?.springs?.length) return;
  const alpha = (SOFT_XPBD_BASE_COMPLIANCE / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < SOFT_XPBD_ITERS; iter++) {
    for (let si = 0; si < soft.springs.length; si++) {
      const [i, j, rest] = soft.springs[si];
      const a = soft.nodes[i];
      const b = soft.nodes[j];
      if (!a || !b) continue;
      if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

      const ax = a.x + a.vx * dtPos;
      const ay = a.y + a.vy * dtPos;
      const bx = b.x + b.vx * dtPos;
      const by = b.y + b.vy * dtPos;

      const dx = bx - ax;
      const dy = by - ay;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = dx / d;
      const ny = dy / d;
      const strainCap = Math.max(0.05, Math.abs(rest) * 0.45);
      const C = clamp(d - rest, -strainCap, strainCap);

      const wA = 1 / Math.max(0.02, a.mass || 1);
      const wB = 1 / Math.max(0.02, b.mass || 1);
      const wSum = wA + wB;
      if (wSum <= 1e-9) continue;

      const lambdaPrev = Number(lambdaCache[si]) || 0;
      let dl = (-C - alpha * lambdaPrev) / (wSum + alpha);
      if (!Number.isFinite(dl)) continue;
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      lambdaCache[si] = lambdaNext;

      const corrAx = -wA * dl * nx;
      const corrAy = -wA * dl * ny;
      const corrBx = wB * dl * nx;
      const corrBy = wB * dl * ny;

      a.vx += corrAx / dtPos;
      a.vy += corrAy / dtPos;
      b.vx += corrBx / dtPos;
      b.vy += corrBy / dtPos;
    }
  }
}

test('soft spring XPBD parity: baseline stepping and gpu-only module produce matching soft states', () => {
  const dtPos = 0.18;
  const stiffnessScale = 3.0;
  const softSeed = {
    nodes: [
      { x: 20, y: 25, vx: 0.2, vy: -0.1, mass: 1.1, clusterId: 1 },
      { x: 30, y: 21, vx: -0.3, vy: 0.4, mass: 0.8, clusterId: 1 },
      { x: 39, y: 28, vx: 0.5, vy: 0.2, mass: 1.4, clusterId: 2 },
      { x: 48, y: 20, vx: -0.4, vy: -0.35, mass: 1.0, clusterId: 3 },
    ],
    springs: [
      [0, 1, 11.2],
      [1, 2, 10.8],
      [2, 3, 12.1],
      [0, 2, 19.4],
    ],
  };
  const skipClusterSet = new Set([3]);

  const baseline = structuredClone(softSeed);
  const gpuOnly = structuredClone(softSeed);
  const baselineLambda = new Float32Array(softSeed.springs.length);
  const gpuLambda = new Float32Array(softSeed.springs.length);

  applySoftSpringsXPBDVelocityBaseline(baseline, dtPos, stiffnessScale, baselineLambda, { skipClusterSet });
  applySoftSpringsXPBDVelocityGpuOnly({
    soft: gpuOnly,
    dtPos,
    stiffnessScale,
    lambdaCache: gpuLambda,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    skipClusterSet,
  });

  assert.deepEqual(Array.from(gpuLambda), Array.from(baselineLambda), 'lambda cache should match baseline stepping exactly');
  assert.equal(gpuOnly.nodes.length, baseline.nodes.length);
  for (let i = 0; i < baseline.nodes.length; i++) {
    const b = baseline.nodes[i];
    const g = gpuOnly.nodes[i];
    assert.ok(Math.abs(g.vx - b.vx) < 1e-12, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-12, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    assert.ok(Math.abs(g.x - b.x) < 1e-12, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-12, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
  }
});
