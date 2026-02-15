import test from 'node:test';
import assert from 'node:assert/strict';

import {
  sanitizeSoftSprings,
  ensureLambdaCacheSize,
  buildSoftClusterBoundaryLoops,
  decayLambdaCache,
  recoverSoftSpringRests,
} from '../../sim-server/public/soft-xpbd.js';
import { compileFieldToMesh } from '../../sim-server/public/field-to-structure-core.js';
import { createCreatureSpecFromMesh, buildBodiesFromCreatureSpec } from '../../sim-server/public/creature-spec.js';

test('sanitizeSoftSprings drops malformed/out-of-range entries deterministically', () => {
  const raw = [
    [0, 1, 2.5, 1, [1, 1, 1]],   // keep
    [6, 1, 1.0, 1, [1, 1, 1]],   // drop (a out of range)
    [1, -2, 1.0, 1, [1, 1, 1]],  // drop (b out of range)
    [1, 1, 1.0, 1, [1, 1, 1]],   // drop (degenerate)
    ['x', 2, 1.0, 1, [1, 1, 1]], // drop (non-integer)
    { a: 0, b: 2, rest: 1.2, edgeBodyMode: 1, edgeDyeMode: [1,1,1] }, // drop (non-array strict mode)
    [2, 3, NaN, 0, [0, 0, 0]],   // keep with rest fallback
  ];

  const out = sanitizeSoftSprings(raw, 5, { edgeBodyPass: 0, edgeBodyBlock: 1, restFloor: 1e-3 });
  assert.equal(out.springs.length, 2);
  assert.equal(out.dropped, 5);

  assert.deepEqual(out.springs[0].slice(0, 4), [0, 1, 2.5, 1]);
  assert.deepEqual(out.springs[1].slice(0, 2), [2, 3]);
  assert.ok(out.springs[1][2] >= 1e-3, 'rest should be clamped to finite floor');
});

test('sanitizeSoftSprings de-duplicates undirected spring pairs to avoid double constraints', () => {
  const raw = [
    [0, 2, 1.0, 1, [1, 1, 1]],
    [2, 0, 1.0, 1, [1, 1, 1]], // duplicate reversed edge
    [0, 2, 1.0, 0, [0, 0, 0]], // duplicate same orientation, different mode
    [2, 3, 1.5, 1, [1, 1, 1]],
  ];

  const out = sanitizeSoftSprings(raw, 5, { edgeBodyPass: 0, edgeBodyBlock: 1, restFloor: 1e-3 });
  assert.equal(out.springs.length, 2);
  assert.equal(out.dropped, 2);
  assert.deepEqual(out.springs[0].slice(0, 4), [0, 2, 1, 1]);
  assert.deepEqual(out.springs[1].slice(0, 4), [2, 3, 1.5, 1]);
});

test('buildSoftClusterBoundaryLoops prefers boundary BLOCK cycle and ignores interior nodes', () => {
  const nodes = [
    { x: 0, y: 0, clusterId: 0 },  // 0
    { x: 2, y: 0, clusterId: 0 },  // 1
    { x: 2, y: 2, clusterId: 0 },  // 2
    { x: 0, y: 2, clusterId: 0 },  // 3
    { x: 1, y: 1, clusterId: 0 },  // 4 interior node
  ];

  const springs = [
    [0, 1, 2, 1],
    [1, 2, 2, 1],
    [2, 3, 2, 1],
    [3, 0, 2, 1],
    [0, 4, 1.4, 0],
    [1, 4, 1.4, 0],
    [2, 4, 1.4, 0],
    [3, 4, 1.4, 0],
  ];

  const loops = buildSoftClusterBoundaryLoops(nodes, springs, { blockMode: 1 });
  assert.equal(loops.length, 1);
  assert.equal(loops[0].source, 'boundary');
  assert.equal(loops[0].indices.length, 4);
  assert.ok(!loops[0].indices.includes(4), 'interior node should not be part of boundary area loop');
});

test('buildSoftClusterBoundaryLoops falls back to convex hull when boundary cycle is unavailable', () => {
  const nodes = [
    { x: 0, y: 0, clusterId: 1 },
    { x: 2, y: 0, clusterId: 1 },
    { x: 2, y: 2, clusterId: 1 },
    { x: 0, y: 2, clusterId: 1 },
    { x: 1, y: 1, clusterId: 1 },
  ];

  // No BLOCK perimeter springs => force hull fallback.
  const springs = [
    [0, 4, 1.4, 0],
    [1, 4, 1.4, 0],
    [2, 4, 1.4, 0],
    [3, 4, 1.4, 0],
  ];

  const loops = buildSoftClusterBoundaryLoops(nodes, springs, { blockMode: 1 });
  assert.equal(loops.length, 1);
  assert.equal(loops[0].source, 'hull');
  assert.equal(loops[0].indices.length, 4);
  assert.ok(!loops[0].indices.includes(4), 'hull should discard interior node');
});

test('ensureLambdaCacheSize preserves warm-start values across resize and sanitizes non-finite entries', () => {
  const prev = new Float32Array([1.25, -2.5, Number.NaN, 99]);
  const next = ensureLambdaCacheSize(prev, 6, 20);
  assert.equal(next.length, 6);
  assert.equal(next[0], 1.25);
  assert.equal(next[1], -2.5);
  assert.equal(next[2], 0);
  assert.equal(next[3], 20);
  assert.equal(next[4], 0);
  assert.equal(next[5], 0);

  const shrunk = ensureLambdaCacheSize(next, 2, 20);
  assert.deepEqual([...shrunk], [1.25, -2.5]);
});

test('decayLambdaCache damps stale warm-start impulses while keeping finite clamp guarantees', () => {
  const cache = new Float32Array([10, -8, 0.00004, Number.NaN, Number.POSITIVE_INFINITY]);
  decayLambdaCache(cache, { decay: 0.9, clampAbs: 20, deadband: 1e-4 });

  assert.ok(Math.abs(cache[0] - 9) < 1e-6, `expected damped lambda, got ${cache[0]}`);
  assert.ok(Math.abs(cache[1] + 7.2) < 1e-6, `expected damped lambda, got ${cache[1]}`);
  assert.equal(cache[2], 0, 'tiny residual values should be zeroed by deadband');
  assert.equal(cache[3], 0, 'non-finite lambda should sanitize to zero');
  assert.equal(cache[4], 0, 'non-finite lambda should sanitize to zero');
});

function paintDisk(field, width, height, cx, cy, radius, value = 1) {
  const r = Math.max(0.8, Number(radius) || 1);
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(height - 1, Math.ceil(cy + r)); y++) {
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(width - 1, Math.ceil(cx + r)); x++) {
      const d = Math.hypot(x - cx, y - cy);
      if (d > r) continue;
      const t = 1 - d / r;
      const idx = y * width + x;
      field[idx] = Math.max(field[idx], v * t + field[idx] * (1 - t));
    }
  }
}

function buildPureSoftMeshScenario({ grid = 100, scale = 0.5, compileOverrides = null } = {}) {
  const rigidField = new Float32Array(grid * grid).fill(0);
  const softField = new Float32Array(grid * grid).fill(0);
  const softDensityField = new Float32Array(grid * grid).fill(1);

  const blobs = [
    { cx: 28, cy: 50, r: 12, sparse: 0.22 },
    { cx: 72, cy: 50, r: 12, sparse: 0.20 },
  ];

  for (const { cx, cy, r, sparse } of blobs) {
    paintDisk(softField, grid, grid, cx, cy, r, 1);
    paintDisk(softField, grid, grid, cx + 2.8, cy - 1.7, r * 0.58, 0.94);
    for (let y = Math.max(0, Math.floor(cy - r - 2)); y <= Math.min(grid - 1, Math.ceil(cy + r + 2)); y++) {
      for (let x = Math.max(0, Math.floor(cx - r - 2)); x <= Math.min(grid - 1, Math.ceil(cx + r + 2)); x++) {
        const idx = y * grid + x;
        if (softField[idx] <= 0.03) continue;
        const radial = Math.min(1, Math.hypot(x - cx, y - cy) / Math.max(1, r));
        softDensityField[idx] = Math.min(softDensityField[idx], Math.max(sparse, 1 - radial * (1 - sparse)));
      }
    }
  }

  const mesh = compileFieldToMesh({
    width: grid,
    height: grid,
    rigidField,
    softField,
    density: 3,
    threshold: 0.35,
    connectivityMode: 'none',
    minComponentTriangles: 0,
    softInfillMode: 'triangles+cross',
    softDensityField,
    ...(compileOverrides || {}),
  });

  const spec = createCreatureSpecFromMesh(mesh, {
    name: 'soft-xpbd-long-run-adversarial',
    includeAuthoring: false,
    fields: { rigidField, softField, softDensityField },
    softSeamWeldSprings: false,
  });

  const bodies = buildBodiesFromCreatureSpec(spec, grid, { massLight: 1.2, massHeavy: 3, massSoft: 0.6 });
  const soft = bodies.soft;

  let cx = 0;
  let cy = 0;
  for (const node of soft.nodes) {
    cx += node.x;
    cy += node.y;
  }
  cx /= Math.max(1, soft.nodes.length);
  cy /= Math.max(1, soft.nodes.length);

  for (const node of soft.nodes) {
    node.x = cx + (node.x - cx) * scale;
    node.y = cy + (node.y - cy) * scale;
    node.r = Math.max(0.35, (Number(node.r) || 1) * scale);
    node.vx = (node.clusterId % 2 === 0) ? 0.15 : -0.15;
    node.vy = 0;
  }
  for (const spring of soft.springs) spring[2] = Math.max(1e-4, Number(spring[2]) * scale);

  return soft;
}

function signedAreaPred(nodes, indices, dtPos) {
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

function runLongSoftScenario({ useStrainClamp }) {
  const soft = buildPureSoftMeshScenario({ grid: 100, scale: 0.5 });
  const springLambda = new Float32Array(soft.springs.length);
  const areaRest = new Map();
  const areaLambda = new Map();
  const baseRest = soft.springs.map((s) => Math.max(1e-4, Number(s[2]) || 1e-4));

  const dt = 0.03;
  const dtPos = dt * 24;
  const springAlpha = (0.0012 / 3.0) / (dtPos * dtPos);
  const areaAlpha = (0.0009 / 3.0) / (dtPos * dtPos);

  let maxAreaDeviation = 0;
  let residualKinetic = 0;
  let tailLambdaAbs = 0;
  let postShockMaxSpeed = 0;

  for (let step = 0; step < 1000; step++) {
    if (step === 360) {
      for (let i = 0; i < soft.springs.length; i++) soft.springs[i][2] = baseRest[i] * 0.62;
    } else if (step === 420) {
      for (let i = 0; i < soft.springs.length; i++) soft.springs[i][2] = baseRest[i];
    }

    for (let i = 0; i < soft.nodes.length; i++) {
      const n = soft.nodes[i];
      const phase = step * 0.12 + i * 0.31;
      const kick = step < 300 ? 1 : 0;
      n.vx += Math.cos(phase) * 0.012 * kick;
      n.vy += Math.sin(phase * 1.3) * 0.012 * kick;
      n.vx *= 0.992;
      n.vy *= 0.992;
    }

    for (let iter = 0; iter < 10; iter++) {
      for (let si = 0; si < soft.springs.length; si++) {
        const [ia, ib, rest] = soft.springs[si];
        const a = soft.nodes[ia];
        const b = soft.nodes[ib];
        const ax = a.x + a.vx * dtPos;
        const ay = a.y + a.vy * dtPos;
        const bx = b.x + b.vx * dtPos;
        const by = b.y + b.vy * dtPos;
        const dx = bx - ax;
        const dy = by - ay;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const nx = dx / d;
        const ny = dy / d;
        const strainCap = useStrainClamp ? Math.max(0.05, Math.abs(rest) * 0.45) : Number.POSITIVE_INFINITY;
        const C = Math.max(-strainCap, Math.min(strainCap, d - rest));
        const wA = 1 / Math.max(0.02, a.mass || 1);
        const wB = 1 / Math.max(0.02, b.mass || 1);
        const wSum = wA + wB;
        if (wSum <= 1e-9) continue;

        const lambdaPrev = Number(springLambda[si]) || 0;
        let dl = (-C - springAlpha * lambdaPrev) / (wSum + springAlpha);
        if (!Number.isFinite(dl)) continue;
        const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
        dl = lambdaNext - lambdaPrev;
        springLambda[si] = lambdaNext;

        a.vx += (-wA * dl * nx) / dtPos;
        a.vy += (-wA * dl * ny) / dtPos;
        b.vx += (wB * dl * nx) / dtPos;
        b.vy += (wB * dl * ny) / dtPos;
      }
    }

    const loops = buildSoftClusterBoundaryLoops(soft.nodes, soft.springs, { blockMode: 1 });
    for (const loop of loops) {
      if (!areaRest.has(loop.clusterId)) {
        const a0 = signedAreaPred(soft.nodes, loop.indices, dtPos);
        areaRest.set(loop.clusterId, Math.abs(a0) > 1e-4 ? a0 : 1e-4);
      }
      if (!areaLambda.has(loop.clusterId)) areaLambda.set(loop.clusterId, 0);
    }

    for (let iter = 0; iter < 6; iter++) {
      for (const loop of loops) {
        const ids = loop.indices;
        const m = ids.length;
        const rest = areaRest.get(loop.clusterId);
        const area = signedAreaPred(soft.nodes, ids, dtPos);
        const C = area - rest;

        const gx = new Array(m);
        const gy = new Array(m);
        let sum = 0;

        for (let k = 0; k < m; k++) {
          const p = soft.nodes[ids[(k - 1 + m) % m]];
          const n = soft.nodes[ids[(k + 1) % m]];
          const px = p.x + p.vx * dtPos;
          const py = p.y + p.vy * dtPos;
          const nx = n.x + n.vx * dtPos;
          const ny = n.y + n.vy * dtPos;
          gx[k] = 0.5 * (ny - py);
          gy[k] = 0.5 * (px - nx);
          const node = soft.nodes[ids[k]];
          const w = 1 / Math.max(0.02, node.mass || 1);
          sum += w * (gx[k] * gx[k] + gy[k] * gy[k]);
        }
        if (sum <= 1e-10) continue;

        const lp = Number(areaLambda.get(loop.clusterId)) || 0;
        let dl = (-C - areaAlpha * lp) / (sum + areaAlpha);
        dl = Math.max(-2, Math.min(2, dl));
        const ln = Math.max(-20, Math.min(20, lp + dl));
        dl = ln - lp;
        areaLambda.set(loop.clusterId, ln);

        for (let k = 0; k < m; k++) {
          const node = soft.nodes[ids[k]];
          const w = 1 / Math.max(0.02, node.mass || 1);
          node.vx += (w * gx[k] * dl) / dtPos;
          node.vy += (w * gy[k] * dl) / dtPos;
        }
      }
    }

    for (const node of soft.nodes) {
      const vm = Math.hypot(node.vx, node.vy);
      if (step >= 420) postShockMaxSpeed = Math.max(postShockMaxSpeed, vm);
      if (vm > 4) {
        node.vx = (node.vx / vm) * 4;
        node.vy = (node.vy / vm) * 4;
      }
      node.x += node.vx * dtPos;
      node.y += node.vy * dtPos;
    }

    const ratios = [];
    for (const [cid, rest] of areaRest.entries()) {
      const loop = loops.find((l) => l.clusterId === cid);
      if (!loop) continue;
      ratios.push(Math.abs(signedAreaPred(soft.nodes, loop.indices, 0) / rest));
    }
    const areaDeviation = ratios.length ? Math.max(...ratios.map((r) => Math.abs(1 - r))) : 0;
    maxAreaDeviation = Math.max(maxAreaDeviation, areaDeviation);

    if (step >= 700) {
      let sumV2 = 0;
      for (const n of soft.nodes) sumV2 += n.vx * n.vx + n.vy * n.vy;
      residualKinetic += sumV2 / Math.max(1, soft.nodes.length);
      let frameLambdaAbs = 0;
      for (let i = 0; i < springLambda.length; i++) frameLambdaAbs += Math.abs(Number(springLambda[i]) || 0);
      tailLambdaAbs += frameLambdaAbs / Math.max(1, springLambda.length);
    }
  }

  return { maxAreaDeviation, residualKinetic, tailLambdaAbs, postShockMaxSpeed };
}

test('1000-step pure-soft mesh scenario (scale 0.5) is more stable with spring strain clamp', () => {
  const before = runLongSoftScenario({ useStrainClamp: false });
  const after = runLongSoftScenario({ useStrainClamp: true });

  assert.ok(after.postShockMaxSpeed <= before.postShockMaxSpeed * 1.1,
    `expected bounded post-shock peak speed with strain clamp (before=${before.postShockMaxSpeed}, after=${after.postShockMaxSpeed})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation * 1.25,
    `expected bounded area drift with strain clamp (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

function runRestDriftRecoveryScenario({ useRestRecovery, adaptiveRecovery = false, recoveryOverrides = null, compileOverrides = null }) {
  const soft = buildPureSoftMeshScenario({ grid: 100, scale: 0.5, compileOverrides });
  const baseline = new Float32Array(soft.springs.length);
  for (let i = 0; i < soft.springs.length; i++) baseline[i] = Math.max(1e-4, Number(soft.springs[i][2]) || 1e-4);

  let maxBaselineRest = 0;
  let minBaselineRest = Number.POSITIVE_INFINITY;
  for (let i = 0; i < baseline.length; i++) {
    const r = Math.max(1e-4, Number(baseline[i]) || 1e-4);
    maxBaselineRest = Math.max(maxBaselineRest, r);
    minBaselineRest = Math.min(minBaselineRest, r);
  }
  const baselineRestSpanRatio = maxBaselineRest / Math.max(1e-9, minBaselineRest);

  const springLambda = new Float32Array(soft.springs.length);
  const areaRest = new Map();
  const areaLambda = new Map();

  const dt = 0.03;
  const dtPos = dt * 24;
  const springAlpha = (0.0012 / 3.0) / (dtPos * dtPos);
  const areaAlpha = (0.0009 / 3.0) / (dtPos * dtPos);

  let recoveryAt200 = 0;
  let recoveryAt500 = 0;
  let recoveryAt1000 = 0;
  let recoveryHalfLifeSteps = 1000;
  let maxAreaDeviation = 0;
  let maxRestScaleDrift = 0;
  let finalRestScaleDrift = 0;
  let tailRestScaleDriftSum = 0;
  let tailRestScaleDriftCount = 0;
  let driftAtPulse = 0;

  for (let step = 0; step < 1000; step++) {
    // deterministic adversarial drift pulse similar to severe-topology rest mutation.
    if (step === 120) {
      for (let i = 0; i < soft.springs.length; i++) {
        const rest = Number(soft.springs[i][2]) || baseline[i];
        soft.springs[i][2] = (i % 3 === 0) ? rest * 1.45 : rest * 0.72;
      }
      springLambda.fill(0);
    }

    for (let i = 0; i < soft.nodes.length; i++) {
      const n = soft.nodes[i];
      const phase = step * 0.11 + i * 0.29;
      n.vx += Math.cos(phase) * 0.006;
      n.vy += Math.sin(phase * 1.27) * 0.006;
      n.vx *= 0.994;
      n.vy *= 0.994;
    }

    for (let iter = 0; iter < 8; iter++) {
      for (let si = 0; si < soft.springs.length; si++) {
        const [ia, ib, rest] = soft.springs[si];
        const a = soft.nodes[ia];
        const b = soft.nodes[ib];
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
        const C = Math.max(-strainCap, Math.min(strainCap, d - rest));
        const wA = 1 / Math.max(0.02, a.mass || 1);
        const wB = 1 / Math.max(0.02, b.mass || 1);
        const wSum = wA + wB;
        if (wSum <= 1e-9) continue;

        const lambdaPrev = Number(springLambda[si]) || 0;
        let dl = (-C - springAlpha * lambdaPrev) / (wSum + springAlpha);
        if (!Number.isFinite(dl)) continue;
        const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
        dl = lambdaNext - lambdaPrev;
        springLambda[si] = lambdaNext;

        a.vx += (-wA * dl * nx) / dtPos;
        a.vy += (-wA * dl * ny) / dtPos;
        b.vx += (wB * dl * nx) / dtPos;
        b.vy += (wB * dl * ny) / dtPos;
      }
    }

    const loops = buildSoftClusterBoundaryLoops(soft.nodes, soft.springs, { blockMode: 1 });
    for (const loop of loops) {
      if (!areaRest.has(loop.clusterId)) {
        const a0 = signedAreaPred(soft.nodes, loop.indices, dtPos);
        areaRest.set(loop.clusterId, Math.abs(a0) > 1e-4 ? a0 : 1e-4);
      }
      if (!areaLambda.has(loop.clusterId)) areaLambda.set(loop.clusterId, 0);
    }

    for (let iter = 0; iter < 4; iter++) {
      for (const loop of loops) {
        const ids = loop.indices;
        const m = ids.length;
        const rest = areaRest.get(loop.clusterId);
        const area = signedAreaPred(soft.nodes, ids, dtPos);
        const C = area - rest;

        const gx = new Array(m);
        const gy = new Array(m);
        let sum = 0;
        for (let k = 0; k < m; k++) {
          const p = soft.nodes[ids[(k - 1 + m) % m]];
          const n = soft.nodes[ids[(k + 1) % m]];
          const px = p.x + p.vx * dtPos;
          const py = p.y + p.vy * dtPos;
          const nx = n.x + n.vx * dtPos;
          const ny = n.y + n.vy * dtPos;
          gx[k] = 0.5 * (ny - py);
          gy[k] = 0.5 * (px - nx);
          const node = soft.nodes[ids[k]];
          const w = 1 / Math.max(0.02, node.mass || 1);
          sum += w * (gx[k] * gx[k] + gy[k] * gy[k]);
        }
        if (sum <= 1e-10) continue;

        const lp = Number(areaLambda.get(loop.clusterId)) || 0;
        let dl = (-C - areaAlpha * lp) / (sum + areaAlpha);
        dl = Math.max(-2, Math.min(2, dl));
        const ln = Math.max(-20, Math.min(20, lp + dl));
        dl = ln - lp;
        areaLambda.set(loop.clusterId, ln);

        for (let k = 0; k < m; k++) {
          const node = soft.nodes[ids[k]];
          const w = 1 / Math.max(0.02, node.mass || 1);
          node.vx += (w * gx[k] * dl) / dtPos;
          node.vy += (w * gy[k] * dl) / dtPos;
        }
      }
    }

    for (const node of soft.nodes) {
      const vm = Math.hypot(node.vx, node.vy);
      if (vm > 4) {
        node.vx = (node.vx / vm) * 4;
        node.vy = (node.vy / vm) * 4;
      }
      node.x += node.vx * dtPos;
      node.y += node.vy * dtPos;
    }

    if (useRestRecovery && step >= 121) {
      recoverSoftSpringRests(soft.springs, baseline, {
        recoverRate: 0.045,
        hardMinFactor: 0.7,
        hardMaxFactor: 1.45,
        adaptiveGainMax: adaptiveRecovery ? 2.4 : 1,
        adaptiveExponent: adaptiveRecovery ? 0.8 : 1,
        elongationBiasMax: adaptiveRecovery ? 1.22 : 1,
        compressionBiasMax: adaptiveRecovery ? 1.12 : 1,
        errorPivot: 0.16,
        ...(recoveryOverrides || {}),
      });
    }

    let meanScaleError = 0;
    for (let i = 0; i < soft.springs.length; i++) {
      const cur = Math.max(1e-4, Number(soft.springs[i][2]) || baseline[i]);
      const base = Math.max(1e-4, baseline[i]);
      meanScaleError += Math.abs(cur / base - 1);
    }
    meanScaleError /= Math.max(1, soft.springs.length);
    if (step === 120) driftAtPulse = meanScaleError;
    const recovery = Math.max(0, 1 - meanScaleError);
    const halfDriftThreshold = driftAtPulse * 0.5;
    if (step > 120 && recoveryHalfLifeSteps === 1000 && meanScaleError <= halfDriftThreshold) {
      recoveryHalfLifeSteps = step - 120;
    }
    if (step === 199) recoveryAt200 = recovery;
    if (step === 499) recoveryAt500 = recovery;
    if (step === 999) recoveryAt1000 = recovery;
    maxRestScaleDrift = Math.max(maxRestScaleDrift, meanScaleError);
    if (step >= 700) {
      tailRestScaleDriftSum += meanScaleError;
      tailRestScaleDriftCount += 1;
    }
    if (step === 999) finalRestScaleDrift = meanScaleError;

    const ratios = [];
    for (const [cid, rest] of areaRest.entries()) {
      const loop = loops.find((l) => l.clusterId === cid);
      if (!loop) continue;
      ratios.push(Math.abs(signedAreaPred(soft.nodes, loop.indices, 0) / rest));
    }
    const areaDeviation = ratios.length ? Math.max(...ratios.map((r) => Math.abs(1 - r))) : 0;
    maxAreaDeviation = Math.max(maxAreaDeviation, areaDeviation);
  }

  const tailMeanRestScaleDrift = tailRestScaleDriftCount > 0
    ? (tailRestScaleDriftSum / tailRestScaleDriftCount)
    : finalRestScaleDrift;

  return {
    baselineRestSpanRatio,
    recoveryHalfLifeSteps,
    recoveryAt200,
    recoveryAt500,
    recoveryAt1000,
    maxAreaDeviation,
    maxRestScaleDrift,
    finalRestScaleDrift,
    tailMeanRestScaleDrift,
  };
}

test('adversarial pure-soft rest drift recovers shape memory with bounded spring-rest restoration', () => {
  const before = runRestDriftRecoveryScenario({ useRestRecovery: true, adaptiveRecovery: false });
  const after = runRestDriftRecoveryScenario({ useRestRecovery: true, adaptiveRecovery: true });
  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-metrics]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps < before.recoveryHalfLifeSteps,
    `expected faster half-life recovery (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 > before.recoveryAt200,
    `expected better early recovery (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 > before.recoveryAt500,
    `expected better mid recovery (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing long recovery (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area-deviation drift under recovery guardrail (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
  assert.ok(after.finalRestScaleDrift < before.finalRestScaleDrift,
    `expected lower final spring-rest drift (before=${before.finalRestScaleDrift}, after=${after.finalRestScaleDrift})`);
  assert.ok(after.tailMeanRestScaleDrift < before.tailMeanRestScaleDrift,
    `expected lower late-tail spring-rest drift (before=${before.tailMeanRestScaleDrift}, after=${after.tailMeanRestScaleDrift})`);
});

test('near-baseline adaptive snap eliminates deterministic tail drift without topology regression', () => {
  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      jitterDeadband: 4e-4,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
    },
  });
  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      jitterDeadband: 4e-4,
      nearBaselineSnapWindow: 0.004,
      nearBaselineSnapBlend: 0.9,
    },
  });
  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-tail-snap]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt1000 > before.recoveryAt1000,
    `expected better late recovery with snap (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.finalRestScaleDrift < before.finalRestScaleDrift,
    `expected smaller final rest-scale drift with snap (before=${before.finalRestScaleDrift}, after=${after.finalRestScaleDrift})`);
  assert.ok(after.maxRestScaleDrift <= before.maxRestScaleDrift + 1e-9,
    `expected no increase in peak rest-scale excursion (before=${before.maxRestScaleDrift}, after=${after.maxRestScaleDrift})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area-deviation guardrail under snap recovery (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('global recovery coupling improves adversarial half-life without area-drift regression', () => {
  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      recoverRate: 0.028,
      adaptiveGainMax: 1.7,
      adaptiveExponent: 0.95,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
      globalErrorCouplingMax: 1,
    },
  });
  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      recoverRate: 0.028,
      adaptiveGainMax: 1.7,
      adaptiveExponent: 0.95,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
      globalErrorCouplingMax: 1.25,
    },
  });
  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-global-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps < before.recoveryHalfLifeSteps,
    `expected faster half-life under global coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 >= (before.recoveryAt200 - 2e-5),
    `expected near-par early recovery under global coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= (before.recoveryAt500 - 2e-5),
    `expected near-par mid recovery under global coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area drift under global coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('directional global coupling improves compression-heavy pure-soft recovery without area regression', () => {
  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      recoverRate: 0.028,
      adaptiveGainMax: 1.7,
      adaptiveExponent: 0.95,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
      globalErrorCouplingMax: 1.25,
      globalDirectionalCouplingMax: 1,
    },
  });
  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      recoverRate: 0.028,
      adaptiveGainMax: 1.7,
      adaptiveExponent: 0.95,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
      globalErrorCouplingMax: 1.25,
      globalDirectionalCouplingMax: 1.18,
    },
  });
  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-directional-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt200 >= (before.recoveryAt200 - 2e-5),
    `expected near-par early recovery under directional coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= (before.recoveryAt500 - 2e-5),
    `expected near-par mid recovery under directional coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= (before.recoveryAt1000 - 2e-5),
    `expected near-par late recovery under directional coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.tailMeanRestScaleDrift <= (before.tailMeanRestScaleDrift + 2e-5),
    `expected bounded tail rest-scale drift under directional coupling (before=${before.tailMeanRestScaleDrift}, after=${after.tailMeanRestScaleDrift})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area drift under directional coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('polarity-aware coupling improves mixed-sign adversarial recovery without area regression', () => {
  const baselineOverrides = {
    recoverRate: 0.028,
    adaptiveGainMax: 1.7,
    adaptiveExponent: 0.95,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    globalErrorCouplingMax: 1.25,
    globalDirectionalCouplingMax: 1.18,
    outlierRecoveryCouplingMax: 1.22,
    outlierErrorPivot: 0.75,
    localEndpointCouplingMax: 1.16,
    localDirectionalCouplingMax: 1.1,
    localErrorPivot: 0.2,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...baselineOverrides,
      polarityCouplingMax: 1,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...baselineOverrides,
      // default enabled in recoverSoftSpringRests
      polarityCouplingMax: 1.08,
    },
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-polarity-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps < before.recoveryHalfLifeSteps,
    `expected faster recovery half-life under polarity coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 >= (before.recoveryAt200 - 5e-5),
    `expected near-par early recovery under polarity coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= (before.recoveryAt500 - 5e-5),
    `expected near-par mid recovery under polarity coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= (before.recoveryAt1000 - 5e-5),
    `expected near-par late recovery under polarity coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area drift under polarity coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('counter-polarity coupling accelerates mixed-sign adversarial recovery without area regression', () => {
  const baselineOverrides = {
    recoverRate: 0.028,
    adaptiveGainMax: 1.7,
    adaptiveExponent: 0.95,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    globalErrorCouplingMax: 1.25,
    globalDirectionalCouplingMax: 1.18,
    outlierRecoveryCouplingMax: 1.22,
    outlierErrorPivot: 0.75,
    localEndpointCouplingMax: 1.16,
    localDirectionalCouplingMax: 1.1,
    localErrorPivot: 0.2,
    polarityCouplingMax: 1.08,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...baselineOverrides,
      counterPolarityCouplingMax: 1,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...baselineOverrides,
      counterPolarityCouplingMax: 1.07,
    },
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-counter-polarity-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps <= before.recoveryHalfLifeSteps,
    `expected non-regressing recovery half-life under counter-polarity coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 >= (before.recoveryAt200 - 5e-5),
    `expected near-par early recovery under counter-polarity coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= (before.recoveryAt500 - 5e-5),
    `expected near-par mid recovery under counter-polarity coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= (before.recoveryAt1000 - 2e-6),
    `expected near-par late recovery under counter-polarity coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area drift under counter-polarity coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('outlier-weighted recovery coupling accelerates pure-soft shape-memory recovery without area regression', () => {
  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      recoverRate: 0.028,
      adaptiveGainMax: 1.7,
      adaptiveExponent: 0.95,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
      globalErrorCouplingMax: 1.25,
      globalDirectionalCouplingMax: 1.18,
      outlierRecoveryCouplingMax: 1,
      outlierErrorPivot: 0.75,
    },
  });
  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      recoverRate: 0.028,
      adaptiveGainMax: 1.7,
      adaptiveExponent: 0.95,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
      globalErrorCouplingMax: 1.25,
      globalDirectionalCouplingMax: 1.18,
      outlierRecoveryCouplingMax: 1.22,
      outlierErrorPivot: 0.75,
    },
  });
  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-outlier-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps <= before.recoveryHalfLifeSteps,
    `expected non-regressing half-life under outlier coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 > before.recoveryAt200,
    `expected better early recovery under outlier coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 > before.recoveryAt500,
    `expected better mid recovery under outlier coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery under outlier coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area drift under outlier coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('local endpoint coupling accelerates pure-soft shape-memory recovery without area regression', () => {
  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      recoverRate: 0.028,
      adaptiveGainMax: 1.7,
      adaptiveExponent: 0.95,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
      globalErrorCouplingMax: 1.25,
      globalDirectionalCouplingMax: 1.18,
      outlierRecoveryCouplingMax: 1.22,
      outlierErrorPivot: 0.75,
      localEndpointCouplingMax: 1,
      localDirectionalCouplingMax: 1,
      localErrorPivot: 0.2,
    },
  });
  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      recoverRate: 0.028,
      adaptiveGainMax: 1.7,
      adaptiveExponent: 0.95,
      nearBaselineSnapWindow: 0,
      nearBaselineSnapBlend: 0,
      globalErrorCouplingMax: 1.25,
      globalDirectionalCouplingMax: 1.18,
      outlierRecoveryCouplingMax: 1.22,
      outlierErrorPivot: 0.75,
      localEndpointCouplingMax: 1.16,
      localDirectionalCouplingMax: 1.1,
      localErrorPivot: 0.2,
    },
  });
  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-local-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps < before.recoveryHalfLifeSteps,
    `expected faster half-life under local coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 >= (before.recoveryAt200 - 2e-5),
    `expected near-par early recovery under local coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= (before.recoveryAt500 - 2e-5),
    `expected near-par mid recovery under local coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= (before.recoveryAt1000 - 2e-5),
    `expected near-par late recovery under local coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area drift under local coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('local endpoint-imbalance coupling improves mixed-sign shape-memory recovery without area regression', () => {
  const common = {
    recoverRate: 0.028,
    adaptiveGainMax: 1.7,
    adaptiveExponent: 0.95,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    globalErrorCouplingMax: 1.25,
    globalDirectionalCouplingMax: 1.18,
    outlierRecoveryCouplingMax: 1.22,
    outlierErrorPivot: 0.75,
    localEndpointCouplingMax: 1.16,
    localDirectionalCouplingMax: 1,
    localErrorPivot: 0.2,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...common,
      localImbalanceCouplingMax: 1,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...common,
      localImbalanceCouplingMax: 1.1,
    },
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-local-imbalance-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps <= before.recoveryHalfLifeSteps,
    `expected non-regressing half-life under local endpoint-imbalance coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 > before.recoveryAt200,
    `expected better early recovery under local endpoint-imbalance coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 > before.recoveryAt500,
    `expected better mid recovery under local endpoint-imbalance coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery under local endpoint-imbalance coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 5e-5,
    `expected bounded area drift under local endpoint-imbalance coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('higher solver recovery rate accelerates adversarial pure-soft shape-memory return with bounded area drift', () => {
  const common = {
    adaptiveGainMax: 1.7,
    adaptiveExponent: 0.95,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    globalErrorCouplingMax: 1.25,
    globalDirectionalCouplingMax: 1.18,
    outlierRecoveryCouplingMax: 1.22,
    outlierErrorPivot: 0.75,
    localEndpointCouplingMax: 1.16,
    localDirectionalCouplingMax: 1.1,
    localErrorPivot: 0.2,
    polarityCouplingMax: 1.08,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...common,
      recoverRate: 0.045,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...common,
      recoverRate: 0.052,
    },
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-rate-stepup]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps <= before.recoveryHalfLifeSteps,
    `expected non-regressing half-life with higher recoverRate (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 >= before.recoveryAt200,
    `expected non-regressing early recovery with higher recoverRate (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= before.recoveryAt500,
    `expected non-regressing mid recovery with higher recoverRate (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery with higher recoverRate (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail with higher recoverRate (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('runtime recovery-rate bump improves deterministic pure-soft form-memory return with bounded area drift', () => {
  const profile = {
    adaptiveGainMax: 2.4,
    adaptiveExponent: 0.8,
    elongationBiasMax: 1.22,
    compressionBiasMax: 1.12,
    errorPivot: 0.16,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    globalErrorCouplingMax: 1.25,
    globalDirectionalCouplingMax: 1.18,
    outlierRecoveryCouplingMax: 1.22,
    outlierErrorPivot: 0.75,
    localEndpointCouplingMax: 1.16,
    localDirectionalCouplingMax: 1.1,
    localErrorPivot: 0.2,
    polarityCouplingMax: 1.08,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...profile,
      recoverRate: 0.052,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...profile,
      recoverRate: 0.056,
    },
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-runtime-rate-bump]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt200 >= (before.recoveryAt200 - 3e-5),
    `expected near-par early recovery under runtime rate bump (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= (before.recoveryAt500 - 3e-5),
    `expected near-par mid recovery under runtime rate bump (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= (before.recoveryAt1000 - 3e-5),
    `expected near-par late recovery under runtime rate bump (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 4e-5,
    `expected bounded area drift under runtime rate bump (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('high recover-rate near-baseline snap removes tiny adversarial residual drift without area regression', () => {
  const profile = {
    adaptiveGainMax: 2.4,
    adaptiveExponent: 0.8,
    elongationBiasMax: 1.22,
    compressionBiasMax: 1.12,
    errorPivot: 0.16,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    globalErrorCouplingMax: 1.25,
    globalDirectionalCouplingMax: 1.18,
    outlierRecoveryCouplingMax: 1.22,
    outlierErrorPivot: 0.75,
    localEndpointCouplingMax: 1.16,
    localDirectionalCouplingMax: 1.1,
    localErrorPivot: 0.2,
    polarityCouplingMax: 1.08,
    recoverRate: 0.056,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...profile,
      highRateSnapRecoverThreshold: 1,
      highRateSnapErrorThreshold: 0,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: profile,
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-high-rate-snap]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt200 >= before.recoveryAt200,
    `expected non-regressing early recovery under high-rate near-baseline snap (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= before.recoveryAt500,
    `expected non-regressing mid recovery under high-rate near-baseline snap (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery under high-rate near-baseline snap (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.finalRestScaleDrift <= before.finalRestScaleDrift,
    `expected lower final rest-scale drift under high-rate near-baseline snap (before=${before.finalRestScaleDrift}, after=${after.finalRestScaleDrift})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 4e-5,
    `expected bounded area drift under high-rate near-baseline snap (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('default near-baseline snap profile removes deterministic pure-soft tail drift under adaptive recovery settings', () => {
  const runtimeRecovery = {
    recoverRate: 0.045,
    adaptiveGainMax: 2.4,
    adaptiveExponent: 0.8,
    elongationBiasMax: 1.22,
    compressionBiasMax: 1.12,
    errorPivot: 0.16,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...runtimeRecovery,
      nearBaselineSnapWindow: 0.003,
      nearBaselineSnapBlend: 0.85,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: runtimeRecovery,
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-default-snap-profile]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt500 >= before.recoveryAt500,
    `expected non-regressing mid recovery with stronger default snap (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery with stronger default snap (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.finalRestScaleDrift <= before.finalRestScaleDrift + 1e-9,
    `expected non-regressing final rest drift with stronger default snap (before=${before.finalRestScaleDrift}, after=${after.finalRestScaleDrift})`);
  assert.ok(after.baselineRestSpanRatio <= before.baselineRestSpanRatio + 1e-9,
    `expected non-regressing baseline rest-span topology under stronger default snap (before=${before.baselineRestSpanRatio}, after=${after.baselineRestSpanRatio})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 1e-6,
    `expected non-regressing peak area deviation under stronger default snap (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('default pure-soft neighbor step cap improves adversarial recovery vs legacy uncapped adaptive jumps', () => {
  const recoveryOverrides = {
    recoverRate: 0.02,
    adaptiveGainMax: 1.35,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides,
    compileOverrides: {
      softNeighborStepDeltaCap: 0,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides,
    // rely on compileFieldToMesh default neighbor-step cap while isolating tip-cap effects
    compileOverrides: {},
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-default-neighbor-cap]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt200 >= before.recoveryAt200,
    `expected non-regressing early recovery with default neighbor-step cap (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= before.recoveryAt500,
    `expected non-regressing mid recovery with default neighbor-step cap (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery with default neighbor-step cap (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.baselineRestSpanRatio < before.baselineRestSpanRatio,
    `expected bounded spring-rest span from smoother topology (before=${before.baselineRestSpanRatio}, after=${after.baselineRestSpanRatio})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail under default neighbor-step cap (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('default pure-soft thin-feature cap improves adversarial recovery without area guardrail regressions', () => {
  const recoveryOverrides = {
    recoverRate: 0.02,
    adaptiveGainMax: 1.35,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides,
    compileOverrides: {
      softNeighborStepDeltaCap: 0,
      softBoundaryCellCap: 6,
      softMaxCellSize: 10,
      softThinFeatureCellCap: 0,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides,
    // isolate thin-feature cap effect against deliberately coarse legacy topology
    compileOverrides: {
      softNeighborStepDeltaCap: 0,
      softBoundaryCellCap: 6,
      softMaxCellSize: 10,
    },
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-default-thin-feature-cap]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt200 >= before.recoveryAt200,
    `expected non-regressing early recovery with thin-feature cap (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= before.recoveryAt500,
    `expected non-regressing mid recovery with thin-feature cap (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery with thin-feature cap (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.baselineRestSpanRatio < before.baselineRestSpanRatio,
    `expected tighter spring-rest span from bounded thin-feature topology (before=${before.baselineRestSpanRatio}, after=${after.baselineRestSpanRatio})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail under default thin-feature cap (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});



test('default pure-soft bridge cap improves adversarial narrow-bridge recovery without area regression', () => {
  const recoveryOverrides = {
    recoverRate: 0.02,
    adaptiveGainMax: 1.35,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides,
    compileOverrides: {
      softNeighborStepDeltaCap: 0,
      softBoundaryCellCap: 6,
      softMaxCellSize: 10,
      softThinFeatureCellCap: 0,
      softBridgeCellCap: 0,
    },
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides,
    compileOverrides: {
      softNeighborStepDeltaCap: 0,
      softBoundaryCellCap: 6,
      softMaxCellSize: 10,
      softThinFeatureCellCap: 0,
      // isolate low-cardinality bridge cap effect
      softBridgeCellCap: 3,
    },
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-default-bridge-cap]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt200 >= (before.recoveryAt200 - 1.5e-5),
    `expected near-par early recovery with bridge cap (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= (before.recoveryAt500 - 1.5e-5),
    `expected near-par mid recovery with bridge cap (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= (before.recoveryAt1000 - 1.5e-5),
    `expected near-par late recovery with bridge cap (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.baselineRestSpanRatio < before.baselineRestSpanRatio,
    `expected tighter spring-rest topology span under bridge cap (before=${before.baselineRestSpanRatio}, after=${after.baselineRestSpanRatio})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail under bridge cap (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});


test('default small-rest profile disables weak coupling to avoid near-baseline pure-soft recovery regressions', () => {
  const recoveryOverrides = {
    recoverRate: 0.02,
    adaptiveGainMax: 1.35,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    smallRestPivot: 0.9,
  };

  const compileOverrides = {
    softNeighborStepDeltaCap: 0,
    softBoundaryCellCap: 6,
    softMaxCellSize: 10,
    softThinFeatureCellCap: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryOverrides,
      smallRestRecoveryCouplingMax: 1.03,
    },
    compileOverrides,
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryOverrides,
      smallRestRecoveryCouplingMax: 1,
    },
    compileOverrides,
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-default-small-rest-profile]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt500 >= before.recoveryAt500,
    `expected non-regressing mid recovery under default small-rest profile (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery under default small-rest profile (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail under default small-rest profile (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('short-rest recovery coupling improves coarse-topology pure-soft form-memory return without area regression', () => {
  const recoveryOverrides = {
    recoverRate: 0.02,
    adaptiveGainMax: 1.35,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
  };

  const compileOverrides = {
    softNeighborStepDeltaCap: 0,
    softBoundaryCellCap: 6,
    softMaxCellSize: 10,
    softThinFeatureCellCap: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryOverrides,
      smallRestRecoveryCouplingMax: 1,
      smallRestPivot: 0.9,
    },
    compileOverrides,
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryOverrides,
      smallRestRecoveryCouplingMax: 1.12,
      smallRestPivot: 0.9,
    },
    compileOverrides,
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-short-rest-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps <= before.recoveryHalfLifeSteps,
    `expected non-regressing half-life under short-rest coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 > before.recoveryAt200,
    `expected better early recovery under short-rest coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 > before.recoveryAt500,
    `expected better mid recovery under short-rest coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery under short-rest coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail under short-rest coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('stronger short-rest coupling (1.12 -> 1.14) improves pure-soft form-memory recovery at scale 0.5 without area regression', () => {
  const recoveryOverrides = {
    recoverRate: 0.02,
    adaptiveGainMax: 1.35,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    smallRestPivot: 0.9,
  };

  const compileOverrides = {
    softNeighborStepDeltaCap: 0,
    softBoundaryCellCap: 6,
    softMaxCellSize: 10,
    softThinFeatureCellCap: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryOverrides,
      smallRestRecoveryCouplingMax: 1.12,
    },
    compileOverrides,
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryOverrides,
      smallRestRecoveryCouplingMax: 1.14,
    },
    compileOverrides,
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-short-rest-coupling-bump]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps <= before.recoveryHalfLifeSteps,
    `expected non-regressing half-life with stronger short-rest coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt200 >= (before.recoveryAt200 - 3e-6),
    `expected near-par early recovery with stronger short-rest coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= (before.recoveryAt500 - 3e-6),
    `expected near-par mid recovery with stronger short-rest coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= (before.recoveryAt1000 - 3e-6),
    `expected near-par late recovery with stronger short-rest coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail with stronger short-rest coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('mid-error recovery coupling improves adversarial pure-soft shape memory without topology drift', () => {
  const recoveryProfile = {
    recoverRate: 0.018,
    adaptiveGainMax: 1.25,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    smallRestRecoveryCouplingMax: 1,
    localEndpointCouplingMax: 1,
    localDirectionalCouplingMax: 1,
    globalErrorCouplingMax: 1,
    globalDirectionalCouplingMax: 1,
    outlierRecoveryCouplingMax: 1,
    polarityCouplingMax: 1,
  };

  const compileOverrides = {
    softNeighborStepDeltaCap: 0,
    softBoundaryCellCap: 6,
    softMaxCellSize: 10,
    softThinFeatureCellCap: 0,
    softBridgeCellCap: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryProfile,
      midErrorRecoveryCouplingMax: 1,
    },
    compileOverrides,
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryProfile,
      midErrorRecoveryCouplingMax: 1.35,
      midErrorRecoveryCenter: 0.24,
      midErrorRecoveryHalfWidth: 0.28,
    },
    compileOverrides,
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-mid-error-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt200 > before.recoveryAt200,
    `expected better early recovery under mid-error coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 > before.recoveryAt500,
    `expected better mid recovery under mid-error coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery under mid-error coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail under mid-error coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
  assert.ok(after.baselineRestSpanRatio <= before.baselineRestSpanRatio + 1e-9,
    `expected non-regressing mesh rest-span topology (before=${before.baselineRestSpanRatio}, after=${after.baselineRestSpanRatio})`);
});

test('low-error recovery coupling improves late-stage pure-soft form-memory return without area regression', () => {
  const recoveryProfile = {
    recoverRate: 0.018,
    adaptiveGainMax: 1.25,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    smallRestRecoveryCouplingMax: 1,
    localEndpointCouplingMax: 1,
    localDirectionalCouplingMax: 1,
    globalErrorCouplingMax: 1,
    globalDirectionalCouplingMax: 1,
    outlierRecoveryCouplingMax: 1,
    polarityCouplingMax: 1,
    midErrorRecoveryCouplingMax: 1.35,
    midErrorRecoveryCenter: 0.24,
    midErrorRecoveryHalfWidth: 0.28,
  };

  const compileOverrides = {
    softNeighborStepDeltaCap: 0,
    softBoundaryCellCap: 6,
    softMaxCellSize: 10,
    softThinFeatureCellCap: 0,
    softBridgeCellCap: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryProfile,
      lowErrorRecoveryCouplingMax: 1,
      lowErrorRecoveryGate: 0.08,
    },
    compileOverrides,
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryProfile,
      lowErrorRecoveryCouplingMax: 1.2,
      lowErrorRecoveryGate: 0.08,
    },
    compileOverrides,
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-low-error-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryHalfLifeSteps <= before.recoveryHalfLifeSteps,
    `expected non-regressing half-life under low-error coupling (before=${before.recoveryHalfLifeSteps}, after=${after.recoveryHalfLifeSteps})`);
  assert.ok(after.recoveryAt500 > before.recoveryAt500,
    `expected better mid recovery under low-error coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery under low-error coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail under low-error coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});

test('default low-error coupling improves adversarial pure-soft tail recovery without topology or area regression', () => {
  const recoveryProfile = {
    recoverRate: 0.018,
    adaptiveGainMax: 1.25,
    adaptiveExponent: 1.05,
    nearBaselineSnapWindow: 0,
    nearBaselineSnapBlend: 0,
    smallRestRecoveryCouplingMax: 1,
    localEndpointCouplingMax: 1,
    localDirectionalCouplingMax: 1,
    globalErrorCouplingMax: 1,
    globalDirectionalCouplingMax: 1,
    outlierRecoveryCouplingMax: 1,
    polarityCouplingMax: 1,
    midErrorRecoveryCouplingMax: 1.35,
    midErrorRecoveryCenter: 0.24,
    midErrorRecoveryHalfWidth: 0.28,
  };

  const compileOverrides = {
    softNeighborStepDeltaCap: 0,
    softBoundaryCellCap: 6,
    softMaxCellSize: 10,
    softThinFeatureCellCap: 0,
    softBridgeCellCap: 0,
  };

  const before = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryProfile,
      lowErrorRecoveryCouplingMax: 1,
      lowErrorRecoveryGate: 0.08,
    },
    compileOverrides,
  });

  const after = runRestDriftRecoveryScenario({
    useRestRecovery: true,
    adaptiveRecovery: true,
    recoveryOverrides: {
      ...recoveryProfile,
      // rely on runtime default lowErrorRecoveryCouplingMax
      lowErrorRecoveryGate: 0.08,
    },
    compileOverrides,
  });

  if (process?.env?.PRINT_SOFT_RECOVERY_METRICS === '1') {
    console.log('[soft-recovery-default-low-error-coupling]', JSON.stringify({ before, after }));
  }

  assert.ok(after.recoveryAt200 >= before.recoveryAt200 - 2e-5,
    `expected near-par early recovery with default low-error coupling (before=${before.recoveryAt200}, after=${after.recoveryAt200})`);
  assert.ok(after.recoveryAt500 >= before.recoveryAt500,
    `expected non-regressing mid recovery with default low-error coupling (before=${before.recoveryAt500}, after=${after.recoveryAt500})`);
  assert.ok(after.recoveryAt1000 >= before.recoveryAt1000,
    `expected non-regressing late recovery with default low-error coupling (before=${before.recoveryAt1000}, after=${after.recoveryAt1000})`);
  assert.ok(after.baselineRestSpanRatio <= before.baselineRestSpanRatio + 1e-9,
    `expected non-regressing mesh rest-span topology under default low-error coupling (before=${before.baselineRestSpanRatio}, after=${after.baselineRestSpanRatio})`);
  assert.ok(after.maxAreaDeviation <= before.maxAreaDeviation + 7e-5,
    `expected bounded area guardrail under default low-error coupling (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});
