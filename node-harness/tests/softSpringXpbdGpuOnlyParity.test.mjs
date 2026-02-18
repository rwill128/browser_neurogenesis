import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applySoftSpringsXPBDVelocityGpuOnly,
  buildSoftSpringXpbdWgslPlan,
  buildSoftSpringXpbdWgslLayout,
} from '../../sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js';

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


test('soft spring XPBD WGSL plan builder emits deterministic CSR endpoint ownership for active springs', () => {
  const soft = {
    nodes: [
      { clusterId: 0 },
      { clusterId: 1 },
      { clusterId: 2 },
      { clusterId: 3 },
    ],
    springs: [
      [0, 1, 10],
      [1, 2, 9],
      [2, 3, 8],
      [0, 99, 7],
      [3, 0, Number.NaN],
    ],
  };

  const plan = buildSoftSpringXpbdWgslPlan({
    soft,
    skipClusterSet: new Set([3]),
  });

  assert.equal(plan.nodeCount, 4);
  assert.equal(plan.springCount, 5);
  assert.equal(plan.activeSpringCount, 2, 'should include only valid, non-skipped springs');
  assert.equal(plan.endpointCount, 4, 'each active spring should contribute two endpoints');
  assert.deepEqual(Array.from(plan.activeSpringIndices), [0, 1]);

  assert.deepEqual(
    Array.from(plan.nodeEndpointOffsets),
    [0, 1, 3, 4, 4],
    'CSR offsets should group endpoint ownership by node index',
  );
  assert.deepEqual(
    Array.from(plan.endpointNodeIndices),
    [0, 1, 1, 2],
    'endpoints should be stably grouped by node index',
  );
  assert.deepEqual(
    Array.from(plan.endpointSpringIndices),
    [0, 0, 1, 1],
    'endpoint spring indices should map grouped endpoints back to active-spring rows',
  );
  assert.deepEqual(
    Array.from(plan.endpointSigns),
    [-1, 1, -1, 1],
    'endpoint signs should preserve i/j ownership direction for gather-reduce kernels',
  );
});

test('soft spring XPBD WGSL layout builder emits deterministic spring SoA + endpoint sign buffers', () => {
  const soft = {
    nodes: [
      { mass: 1.25, clusterId: 0 },
      { mass: 0.5, clusterId: 1 },
      { mass: 2.0, clusterId: 2 },
      { mass: 1.0, clusterId: 3 },
    ],
    springs: [
      [0, 1, 4.5],
      [1, 2, 3.25],
      [2, 3, 5.75],
    ],
  };

  const plan = buildSoftSpringXpbdWgslPlan({
    soft,
    skipClusterSet: new Set([3]),
  });
  const layout = buildSoftSpringXpbdWgslLayout({ soft, plan });

  assert.deepEqual(Array.from(layout.springNodeA), [0, 1]);
  assert.deepEqual(Array.from(layout.springNodeB), [1, 2]);
  assert.deepEqual(Array.from(layout.springRest), [4.5, 3.25]);
  assert.deepEqual(
    Array.from(layout.springInvMassA).map((v) => Number(v.toFixed(6))),
    [0.8, 2],
  );
  assert.deepEqual(
    Array.from(layout.springInvMassB).map((v) => Number(v.toFixed(6))),
    [2, 0.5],
  );
  assert.deepEqual(Array.from(layout.endpointSignsI32), [-1, 1, -1, 1]);
  assert.equal(layout.byteLength > 0, true);
});

test('soft spring XPBD stores WGSL-prep state while preserving cpu parity outputs', () => {
  const dtPos = 0.18;
  const stiffnessScale = 3.0;
  const seed = {
    nodes: [
      { x: 20, y: 25, vx: 0.2, vy: -0.1, mass: 1.1, clusterId: 1 },
      { x: 30, y: 21, vx: -0.3, vy: 0.4, mass: 0.8, clusterId: 1 },
      { x: 39, y: 28, vx: 0.5, vy: 0.2, mass: 1.4, clusterId: 2 },
    ],
    springs: [
      [0, 1, 11.2],
      [1, 2, 10.8],
    ],
  };

  const softA = structuredClone(seed);
  const softB = structuredClone(seed);
  const lambdaA = new Float32Array(seed.springs.length);
  const lambdaB = new Float32Array(seed.springs.length);

  applySoftSpringsXPBDVelocityGpuOnly({
    soft: softA,
    dtPos,
    stiffnessScale,
    lambdaCache: lambdaA,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
  });

  const wgslState = {};
  applySoftSpringsXPBDVelocityGpuOnly({
    soft: softB,
    dtPos,
    stiffnessScale,
    lambdaCache: lambdaB,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: {
      enabled: true,
      state: wgslState,
    },
  });

  assert.deepEqual(Array.from(lambdaB), Array.from(lambdaA), 'wgsl prep mode should preserve cpu lambda outputs');
  assert.deepEqual(softB.nodes, softA.nodes, 'wgsl prep mode should preserve cpu node outputs');
  assert.equal(wgslState.lastMode, 'cpu-prepared');
  assert.equal(wgslState.lastPreparedSpringCount, 2);
  assert.equal(wgslState.lastPreparedEndpointCount, 4);
  assert.equal(wgslState.preparedPlan?.nodeEndpointOffsets?.length, seed.nodes.length + 1);
  assert.equal(wgslState.preparedLayout?.springNodeA?.length, 2);
  assert.equal(wgslState.preparedLayout?.endpointSignsI32?.length, 4);
  assert.equal(wgslState.lastPreparedLayoutBytes, wgslState.preparedLayout?.byteLength);
});
