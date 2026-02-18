import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applySoftMembraneBoundaryXPBDVelocityGpuOnly,
  applySoftMembraneShapeMemoryVelocityGpuOnly,
} from '../../sim-server/public/runtime-solvers/stepSoftMembraneConstraintsGpuOnly.js';

const MEMBRANE_SHAPE_MEMORY_GAIN = 0.045;
const MEMBRANE_SHAPE_MEMORY_ITERS = 2;
const MEMBRANE_SHAPE_MEMORY_MAX_SHIFT_FRAC = 0.08;
const MEMBRANE_EDGE_XPBD_ITERS = 8;
const MEMBRANE_EDGE_BASE_COMPLIANCE = 0.0007;
const MEMBRANE_BEND_XPBD_ITERS = 4;
const MEMBRANE_BEND_BASE_COMPLIANCE = 0.0022;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function ensureSoftMembraneLoopStateBaseline(sim, soft, loops, membraneSet) {
  sim.softMembraneLoopState = sim.softMembraneLoopState || new Map();
  const live = new Set();

  for (const loop of loops || []) {
    const cid = loop.clusterId ?? 0;
    if (!membraneSet.has(cid)) continue;
    const ids = loop.indices || [];
    if (ids.length < 3) continue;
    live.add(cid);

    const key = ids.join(',');
    let st = sim.softMembraneLoopState.get(cid);
    const needsReset = !st || st.key !== key || st.edgeRest.length !== ids.length;
    if (needsReset) {
      st = {
        key,
        edgeRest: new Float32Array(ids.length),
        edgeLambda: new Float32Array(ids.length),
        bendRest: new Float32Array(ids.length),
        bendLambda: new Float32Array(ids.length),
        shapeRef: new Float32Array(ids.length * 2),
        shapeRefRms: 1,
      };

      let cx = 0;
      let cy = 0;
      let count = 0;
      for (let i = 0; i < ids.length; i++) {
        const node = soft.nodes[ids[i]];
        if (!node) continue;
        cx += node.x || 0;
        cy += node.y || 0;
        count += 1;
      }
      if (count > 0) {
        cx /= count;
        cy /= count;
      }

      let refR2 = 0;
      for (let i = 0; i < ids.length; i++) {
        const a = soft.nodes[ids[i]];
        const b = soft.nodes[ids[(i + 1) % ids.length]];
        if (a && b) {
          st.edgeRest[i] = Math.max(1e-4, Math.hypot((b.x || 0) - (a.x || 0), (b.y || 0) - (a.y || 0)));
        }
        const p = soft.nodes[ids[(i - 1 + ids.length) % ids.length]];
        const n = soft.nodes[ids[(i + 1) % ids.length]];
        if (p && n) {
          st.bendRest[i] = Math.max(1e-4, Math.hypot((n.x || 0) - (p.x || 0), (n.y || 0) - (p.y || 0)));
        }
        const node = soft.nodes[ids[i]];
        const rx = (node?.x || 0) - cx;
        const ry = (node?.y || 0) - cy;
        st.shapeRef[i * 2] = rx;
        st.shapeRef[i * 2 + 1] = ry;
        refR2 += rx * rx + ry * ry;
      }
      st.shapeRefRms = Math.max(1e-3, Math.sqrt(refR2 / Math.max(1, ids.length)));
      sim.softMembraneLoopState.set(cid, st);
    }
  }

  for (const cid of [...sim.softMembraneLoopState.keys()]) {
    if (!live.has(cid)) sim.softMembraneLoopState.delete(cid);
  }
}

function applySoftMembraneBoundaryXPBDVelocityBaseline(sim, soft, loops, dtPos, membraneSet) {
  if (!membraneSet.size) return 0;
  ensureSoftMembraneLoopStateBaseline(sim, soft, loops, membraneSet);

  const edgeAlpha = MEMBRANE_EDGE_BASE_COMPLIANCE / Math.max(1e-8, dtPos * dtPos);
  const bendAlpha = MEMBRANE_BEND_BASE_COMPLIANCE / Math.max(1e-8, dtPos * dtPos);
  let touched = 0;

  for (let iter = 0; iter < MEMBRANE_EDGE_XPBD_ITERS; iter++) {
    for (const loop of loops || []) {
      const cid = loop.clusterId ?? 0;
      if (!membraneSet.has(cid)) continue;
      const ids = loop.indices || [];
      if (ids.length < 3) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st) continue;

      for (let i = 0; i < ids.length; i++) {
        const a = soft.nodes[ids[i]];
        const b = soft.nodes[ids[(i + 1) % ids.length]];
        if (!a || !b) continue;

        const ax = a.x + a.vx * dtPos;
        const ay = a.y + a.vy * dtPos;
        const bx = b.x + b.vx * dtPos;
        const by = b.y + b.vy * dtPos;
        const dx = bx - ax;
        const dy = by - ay;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const nx = dx / d;
        const ny = dy / d;

        const rest = Math.max(1e-4, Number(st.edgeRest[i]) || d);
        const C = clamp(d - rest, -Math.max(0.08, rest * 0.28), Math.max(0.08, rest * 0.28));
        const wA = 1 / Math.max(0.02, a.mass || 1);
        const wB = 1 / Math.max(0.02, b.mass || 1);
        const wSum = wA + wB;
        if (wSum <= 1e-9) continue;

        const lambdaPrev = Number(st.edgeLambda[i]) || 0;
        let dl = (-C - edgeAlpha * lambdaPrev) / (wSum + edgeAlpha);
        if (!Number.isFinite(dl)) continue;
        const lambdaNext = clamp(lambdaPrev + dl, -20, 20);
        dl = lambdaNext - lambdaPrev;
        st.edgeLambda[i] = lambdaNext;

        a.vx += (-wA * dl * nx) / dtPos;
        a.vy += (-wA * dl * ny) / dtPos;
        b.vx += (wB * dl * nx) / dtPos;
        b.vy += (wB * dl * ny) / dtPos;
        touched += 1;
      }
    }
  }

  for (let iter = 0; iter < MEMBRANE_BEND_XPBD_ITERS; iter++) {
    for (const loop of loops || []) {
      const cid = loop.clusterId ?? 0;
      if (!membraneSet.has(cid)) continue;
      const ids = loop.indices || [];
      if (ids.length < 4) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st) continue;

      for (let i = 0; i < ids.length; i++) {
        const prev = soft.nodes[ids[(i - 1 + ids.length) % ids.length]];
        const next = soft.nodes[ids[(i + 1) % ids.length]];
        if (!prev || !next) continue;

        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nxp = next.x + next.vx * dtPos;
        const nyp = next.y + next.vy * dtPos;
        const dx = nxp - px;
        const dy = nyp - py;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const ux = dx / d;
        const uy = dy / d;

        const rest = Math.max(1e-4, Number(st.bendRest[i]) || d);
        const C = clamp(d - rest, -Math.max(0.1, rest * 0.35), Math.max(0.1, rest * 0.35));
        const wP = 1 / Math.max(0.02, prev.mass || 1);
        const wN = 1 / Math.max(0.02, next.mass || 1);
        const wSum = wP + wN;
        if (wSum <= 1e-9) continue;

        const lambdaPrev = Number(st.bendLambda[i]) || 0;
        let dl = (-C - bendAlpha * lambdaPrev) / (wSum + bendAlpha);
        if (!Number.isFinite(dl)) continue;
        const lambdaNext = clamp(lambdaPrev + dl, -20, 20);
        dl = lambdaNext - lambdaPrev;
        st.bendLambda[i] = lambdaNext;

        prev.vx += (-wP * dl * ux) / dtPos;
        prev.vy += (-wP * dl * uy) / dtPos;
        next.vx += (wN * dl * ux) / dtPos;
        next.vy += (wN * dl * uy) / dtPos;
      }
    }
  }

  return touched;
}

function applySoftMembraneShapeMemoryVelocityBaseline(sim, soft, loops, dtPos, membraneClusterMap) {
  if (!(membraneClusterMap instanceof Map) || membraneClusterMap.size === 0) return 0;

  let touched = 0;
  for (let iter = 0; iter < MEMBRANE_SHAPE_MEMORY_ITERS; iter++) {
    for (const loop of loops || []) {
      const cid = Number(loop?.clusterId);
      if (!Number.isInteger(cid)) continue;
      const membrane = membraneClusterMap.get(cid);
      if (!membrane) continue;

      const shapeMemoryGainRaw = Number(membrane?.shapeMemoryGain);
      const shapeMemoryGain = Number.isFinite(shapeMemoryGainRaw)
        ? clamp(shapeMemoryGainRaw, 0, 0.35)
        : MEMBRANE_SHAPE_MEMORY_GAIN;
      if (shapeMemoryGain <= 1e-6) continue;

      const ids = loop.indices || [];
      if (ids.length < 3) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st || !(st.shapeRef instanceof Float32Array) || st.shapeRef.length !== ids.length * 2) continue;

      let cx = 0;
      let cy = 0;
      let count = 0;
      for (const idx of ids) {
        const node = soft.nodes[idx];
        if (!node) continue;
        cx += (node.x || 0) + (node.vx || 0) * dtPos;
        cy += (node.y || 0) + (node.vy || 0) * dtPos;
        count += 1;
      }
      if (count < 3) continue;
      cx /= count;
      cy /= count;

      let dot = 0;
      let cross = 0;
      for (let i = 0; i < ids.length; i++) {
        const node = soft.nodes[ids[i]];
        if (!node) continue;
        const px = (node.x || 0) + (node.vx || 0) * dtPos - cx;
        const py = (node.y || 0) + (node.vy || 0) * dtPos - cy;
        const rx = st.shapeRef[i * 2] || 0;
        const ry = st.shapeRef[i * 2 + 1] || 0;
        dot += rx * px + ry * py;
        cross += rx * py - ry * px;
      }

      const theta = (Math.abs(dot) + Math.abs(cross)) > 1e-9 ? Math.atan2(cross, dot) : 0;
      const c = Math.cos(theta);
      const sn = Math.sin(theta);
      const maxShift = Math.max(0.02, (st.shapeRefRms || 1) * MEMBRANE_SHAPE_MEMORY_MAX_SHIFT_FRAC);

      for (let i = 0; i < ids.length; i++) {
        const node = soft.nodes[ids[i]];
        if (!node) continue;

        const px = (node.x || 0) + (node.vx || 0) * dtPos;
        const py = (node.y || 0) + (node.vy || 0) * dtPos;
        const rx = st.shapeRef[i * 2] || 0;
        const ry = st.shapeRef[i * 2 + 1] || 0;

        const tx = cx + rx * c - ry * sn;
        const ty = cy + rx * sn + ry * c;

        let ex = tx - px;
        let ey = ty - py;
        const eLen = Math.hypot(ex, ey);
        if (!Number.isFinite(eLen) || eLen <= 1e-7) continue;
        if (eLen > maxShift) {
          const k = maxShift / eLen;
          ex *= k;
          ey *= k;
        }

        const localWeight = clamp(Number.isFinite(Number(node.shapeMemoryWeight)) ? Number(node.shapeMemoryWeight) : 1, 0, 1);
        if (localWeight <= 1e-6) continue;
        const invMass = 1 / Math.max(0.02, Number(node.mass) || 1);
        const corrPos = shapeMemoryGain * localWeight * invMass;
        node.vx += (ex * corrPos) / dtPos;
        node.vy += (ey * corrPos) / dtPos;
      }

      touched += 1;
    }
  }

  return touched;
}

test('soft membrane shape-memory can consume same-frame WGSL proposal as authoritative source', async () => {
  const dtPos = 0.11;
  const loops = [{ clusterId: 1, indices: [0, 1, 2, 3] }];
  const soft = {
    nodes: [
      { x: 8, y: 9, vx: 0.2, vy: -0.06, mass: 1.1, clusterId: 1, shapeMemoryWeight: 1 },
      { x: 12.8, y: 8.5, vx: -0.1, vy: 0.12, mass: 0.9, clusterId: 1, shapeMemoryWeight: 0.7 },
      { x: 13.4, y: 12.9, vx: 0.09, vy: -0.15, mass: 1.2, clusterId: 1, shapeMemoryWeight: 0.9 },
      { x: 8.6, y: 13.4, vx: -0.07, vy: 0.11, mass: 1.0, clusterId: 1, shapeMemoryWeight: 1 },
    ],
  };
  const membraneClusterMap = new Map([[1, { clusterId: 1, shapeMemoryGain: 0.05 }]]);
  const sim = {
    softMembraneLoopState: new Map(),
    softMembraneClusterMap: membraneClusterMap,
  };

  const wgslState = {
    enableAuthoritativeShapeMemory: true,
  };

  await applySoftMembraneBoundaryXPBDVelocityGpuOnly({
    sim,
    soft,
    loops,
    dtPos,
    membraneClusterSet: new Set([1]),
    clamp,
    membraneEdgeXpbdIters: MEMBRANE_EDGE_XPBD_ITERS,
    membraneEdgeBaseCompliance: MEMBRANE_EDGE_BASE_COMPLIANCE,
    membraneBendXpbdIters: MEMBRANE_BEND_XPBD_ITERS,
    membraneBendBaseCompliance: MEMBRANE_BEND_BASE_COMPLIANCE,
  });

  const firstSoft = structuredClone(soft);
  const touchedFirst = await applySoftMembraneShapeMemoryVelocityGpuOnly({
    sim,
    soft: firstSoft,
    loops,
    dtPos,
    membraneClusterMap,
    clamp,
    membraneShapeMemoryIters: MEMBRANE_SHAPE_MEMORY_ITERS,
    membraneShapeMemoryGain: MEMBRANE_SHAPE_MEMORY_GAIN,
    membraneShapeMemoryMaxShiftFrac: MEMBRANE_SHAPE_MEMORY_MAX_SHIFT_FRAC,
    wgslOffload: {
      enabled: false,
      state: wgslState,
    },
  });

  const preparedSig = wgslState.lastShapeMemoryAuthoritativeSignature;
  assert.ok(preparedSig, 'expected non-empty prepared signature');

  const authoritativeDeltaVx = Float32Array.from([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  const authoritativeDeltaVy = Float32Array.from([-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]);
  wgslState.lastShapeMemoryProposalSource = 'wgsl-shape-memory-proposal';
  wgslState.lastShapeMemoryProposalSignature = preparedSig;
  wgslState.lastShapeMemoryProposalFinite = { allFinite: true };
  wgslState.lastShapeMemoryProposalDeltaVx = authoritativeDeltaVx;
  wgslState.lastShapeMemoryProposalDeltaVy = authoritativeDeltaVy;

  const secondSoft = structuredClone(soft);
  const before = secondSoft.nodes.map((n) => ({ vx: n.vx, vy: n.vy }));
  const touchedSecond = await applySoftMembraneShapeMemoryVelocityGpuOnly({
    sim,
    soft: secondSoft,
    loops,
    dtPos,
    membraneClusterMap,
    clamp,
    membraneShapeMemoryIters: MEMBRANE_SHAPE_MEMORY_ITERS,
    membraneShapeMemoryGain: MEMBRANE_SHAPE_MEMORY_GAIN,
    membraneShapeMemoryMaxShiftFrac: MEMBRANE_SHAPE_MEMORY_MAX_SHIFT_FRAC,
    wgslOffload: {
      enabled: true,
      state: wgslState,
    },
  });

  assert.equal(touchedSecond, touchedFirst, 'shape-memory touched count should stay stable for authoritative replay');
  assert.equal(wgslState.lastShapeMemoryAuthoritativeSource, 'wgsl-shape-memory-authoritative');

  const expectedVxDelta = [1, 1, 1, 1];
  const expectedVyDelta = [-0.5, -0.5, -0.5, -0.5];
  for (let i = 0; i < secondSoft.nodes.length; i++) {
    assert.ok(Math.abs(secondSoft.nodes[i].vx - (before[i].vx + expectedVxDelta[i])) < 1e-12, `node ${i} authoritative vx mismatch`);
    assert.ok(Math.abs(secondSoft.nodes[i].vy - (before[i].vy + expectedVyDelta[i])) < 1e-12, `node ${i} authoritative vy mismatch`);
  }
});

test('soft membrane bend keeps hard CPU fallback when WGSL dispatch is unavailable', async () => {
  const dtPos = 0.11;
  const loops = [{ clusterId: 1, indices: [0, 1, 2, 3] }];
  const soft = {
    nodes: [
      { x: 8, y: 9, vx: 0.2, vy: -0.06, mass: 1.1, clusterId: 1, shapeMemoryWeight: 1 },
      { x: 12.8, y: 8.5, vx: -0.1, vy: 0.12, mass: 0.9, clusterId: 1, shapeMemoryWeight: 0.7 },
      { x: 13.4, y: 12.9, vx: 0.09, vy: -0.15, mass: 1.2, clusterId: 1, shapeMemoryWeight: 0.9 },
      { x: 8.6, y: 13.4, vx: -0.07, vy: 0.11, mass: 1.0, clusterId: 1, shapeMemoryWeight: 1 },
    ],
  };

  const sim = {
    softMembraneLoopState: new Map(),
    softMembraneClusterMap: new Map([[1, { clusterId: 1, shapeMemoryGain: 0.05 }]]),
  };

  const wgslState = {
    enableAuthoritativeMembraneBoundaryEdge: true,
    enableAuthoritativeMembraneBend: true,
  };

  const firstSoft = structuredClone(soft);
  const touchedFirst = await applySoftMembraneBoundaryXPBDVelocityGpuOnly({
    sim,
    soft: firstSoft,
    loops,
    dtPos,
    membraneClusterSet: new Set([1]),
    clamp,
    membraneEdgeXpbdIters: MEMBRANE_EDGE_XPBD_ITERS,
    membraneEdgeBaseCompliance: MEMBRANE_EDGE_BASE_COMPLIANCE,
    membraneBendXpbdIters: MEMBRANE_BEND_XPBD_ITERS,
    membraneBendBaseCompliance: MEMBRANE_BEND_BASE_COMPLIANCE,
    wgslOffload: {
      enabled: true,
      device: {},
      modeProfile: 'gpu-only-validated',
      state: wgslState,
    },
  });

  const preparedSig = wgslState.lastMembraneBendAuthoritativeSignature || '';

  wgslState.lastMembraneBendProposalSource = 'wgsl-membrane-bend-proposal';
  wgslState.lastMembraneBendProposalSignature = preparedSig;
  wgslState.lastMembraneBendProposalFinite = { allFinite: true };
  wgslState.lastMembraneBendProposalDeltaVxPrev = Float32Array.from([0.1, 0.2, 0.3, 0.4]);
  wgslState.lastMembraneBendProposalDeltaVyPrev = Float32Array.from([-0.1, -0.2, -0.3, -0.4]);
  wgslState.lastMembraneBendProposalDeltaVxNext = Float32Array.from([0.5, 0.6, 0.7, 0.8]);
  wgslState.lastMembraneBendProposalDeltaVyNext = Float32Array.from([-0.5, -0.6, -0.7, -0.8]);
  wgslState.lastMembraneBendProposalLambdaNext = Float32Array.from([1, 2, 3, 4]);

  sim.softMembraneLoopState = new Map();
  const secondSoft = structuredClone(soft);
  const touchedSecond = await applySoftMembraneBoundaryXPBDVelocityGpuOnly({
    sim,
    soft: secondSoft,
    loops,
    dtPos,
    membraneClusterSet: new Set([1]),
    clamp,
    membraneEdgeXpbdIters: MEMBRANE_EDGE_XPBD_ITERS,
    membraneEdgeBaseCompliance: MEMBRANE_EDGE_BASE_COMPLIANCE,
    membraneBendXpbdIters: MEMBRANE_BEND_XPBD_ITERS,
    membraneBendBaseCompliance: MEMBRANE_BEND_BASE_COMPLIANCE,
    wgslOffload: {
      enabled: true,
      device: {},
      modeProfile: 'gpu-only-validated',
      state: wgslState,
    },
  });

  assert.equal(touchedSecond, touchedFirst, 'bend touched count should stay stable for authoritative replay');
  assert.equal(wgslState.lastMembraneBendAuthoritativeSource, 'cpu-membrane-bend-authoritative');
  assert.equal(wgslState.lastMembraneBendProposalSource, 'cpu-membrane-bend-authoritative');
});

test('soft membrane boundary+shape constraints parity: baseline and gpu-only match', async () => {
  const dtPos = 0.11;
  const loops = [
    { clusterId: 1, indices: [0, 1, 2, 3] },
    { clusterId: 8, indices: [4, 5, 6, 7] },
  ];

  const seedSoft = {
    nodes: [
      { x: 8, y: 9, vx: 0.2, vy: -0.06, mass: 1.1, clusterId: 1, shapeMemoryWeight: 1 },
      { x: 12.8, y: 8.5, vx: -0.1, vy: 0.12, mass: 0.9, clusterId: 1, shapeMemoryWeight: 0.7 },
      { x: 13.4, y: 12.9, vx: 0.09, vy: -0.15, mass: 1.2, clusterId: 1, shapeMemoryWeight: 0.9 },
      { x: 8.6, y: 13.4, vx: -0.07, vy: 0.11, mass: 1.0, clusterId: 1, shapeMemoryWeight: 1 },
      { x: 20.5, y: 18, vx: -0.03, vy: 0.14, mass: 1.3, clusterId: 8, shapeMemoryWeight: 0.8 },
      { x: 24.6, y: 18.7, vx: 0.11, vy: -0.04, mass: 0.95, clusterId: 8, shapeMemoryWeight: 1 },
      { x: 24.1, y: 23.2, vx: -0.08, vy: -0.09, mass: 1.15, clusterId: 8, shapeMemoryWeight: 0.65 },
      { x: 20.1, y: 22.8, vx: 0.05, vy: 0.06, mass: 1.05, clusterId: 8, shapeMemoryWeight: 0.75 },
    ],
  };

  const membraneClusters = [
    { clusterId: 1, shapeMemoryGain: 0.05 },
    { clusterId: 8, shapeMemoryGain: 0.08 },
  ];
  const membraneClusterSet = new Set(membraneClusters.map((m) => m.clusterId));
  const membraneClusterMap = new Map(membraneClusters.map((m) => [m.clusterId, m]));

  const baselineSim = {
    softMembraneLoopState: new Map(),
    softMembraneClusterMap: membraneClusterMap,
  };
  const gpuOnlySim = {
    softMembraneLoopState: new Map(),
    softMembraneClusterMap: membraneClusterMap,
  };

  const baselineSoft = structuredClone(seedSoft);
  const gpuOnlySoft = structuredClone(seedSoft);

  const baselineBoundaryTouched = applySoftMembraneBoundaryXPBDVelocityBaseline(
    baselineSim,
    baselineSoft,
    loops,
    dtPos,
    membraneClusterSet,
  );
  const gpuBoundaryTouched = await applySoftMembraneBoundaryXPBDVelocityGpuOnly({
    sim: gpuOnlySim,
    soft: gpuOnlySoft,
    loops,
    dtPos,
    membraneClusterSet,
    clamp,
    membraneEdgeXpbdIters: MEMBRANE_EDGE_XPBD_ITERS,
    membraneEdgeBaseCompliance: MEMBRANE_EDGE_BASE_COMPLIANCE,
    membraneBendXpbdIters: MEMBRANE_BEND_XPBD_ITERS,
    membraneBendBaseCompliance: MEMBRANE_BEND_BASE_COMPLIANCE,
  });

  const baselineShapeTouched = applySoftMembraneShapeMemoryVelocityBaseline(
    baselineSim,
    baselineSoft,
    loops,
    dtPos,
    membraneClusterMap,
  );
  const gpuShapeTouched = await applySoftMembraneShapeMemoryVelocityGpuOnly({
    sim: gpuOnlySim,
    soft: gpuOnlySoft,
    loops,
    dtPos,
    membraneClusterMap,
    clamp,
    membraneShapeMemoryIters: MEMBRANE_SHAPE_MEMORY_ITERS,
    membraneShapeMemoryGain: MEMBRANE_SHAPE_MEMORY_GAIN,
    membraneShapeMemoryMaxShiftFrac: MEMBRANE_SHAPE_MEMORY_MAX_SHIFT_FRAC,
  });

  assert.equal(gpuBoundaryTouched, baselineBoundaryTouched, 'boundary touched count mismatch');
  assert.equal(gpuShapeTouched, baselineShapeTouched, 'shape touched count mismatch');

  for (let i = 0; i < baselineSoft.nodes.length; i++) {
    const b = baselineSoft.nodes[i];
    const g = gpuOnlySoft.nodes[i];
    assert.ok(Math.abs(g.vx - b.vx) < 1e-12, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-12, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    assert.ok(Math.abs(g.x - b.x) < 1e-12, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-12, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
  }
});
