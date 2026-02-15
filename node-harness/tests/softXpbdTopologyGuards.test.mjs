import test from 'node:test';
import assert from 'node:assert/strict';

import {
  sanitizeSoftSprings,
  ensureLambdaCacheSize,
  buildSoftClusterBoundaryLoops,
  decayLambdaCache,
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

function buildPureSoftMeshScenario({ grid = 100, scale = 0.5 } = {}) {
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
  });

  const spec = createCreatureSpecFromMesh(mesh, {
    name: 'soft-xpbd-long-run-adversarial',
    includeAuthoring: false,
    fields: { rigidField, softField, softDensityField },
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

  assert.ok(after.postShockMaxSpeed < before.postShockMaxSpeed,
    `expected lower post-shock peak speed with strain clamp (before=${before.postShockMaxSpeed}, after=${after.postShockMaxSpeed})`);
  assert.ok(after.maxAreaDeviation < before.maxAreaDeviation,
    `expected lower area drift with strain clamp (before=${before.maxAreaDeviation}, after=${after.maxAreaDeviation})`);
});
