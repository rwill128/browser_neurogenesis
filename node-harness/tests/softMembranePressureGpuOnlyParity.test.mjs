import test from 'node:test';
import assert from 'node:assert/strict';

import { applySoftMembraneCellPressureGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftMembranePressureGpuOnly.js';

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function signedAreaCurrent(nodes, indices) {
  if (!Array.isArray(indices) || indices.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    if (!a || !b) continue;
    sum += a.x * b.y - a.y * b.x;
  }
  return 0.5 * sum;
}

function applySoftMembraneCellPressureBaseline({
  sim,
  soft,
  loops,
  dtPos,
  membraneCellBasePressureGain = 0.08,
  membraneCellBaseRadialDamping = 0.06,
}) {
  const membranes = sim?.bodies?.softMembraneClusters;
  if (!Array.isArray(membranes) || membranes.length === 0) return 0;
  if (!(sim.softMembraneAreaBaseline instanceof Map)) sim.softMembraneAreaBaseline = new Map();

  const loopByCluster = new Map();
  for (const loop of loops || []) {
    loopByCluster.set(loop.clusterId ?? 0, loop);
  }

  let touched = 0;
  for (const membrane of membranes) {
    const cid = Number(membrane?.clusterId);
    if (!Number.isInteger(cid)) continue;
    const loop = loopByCluster.get(cid);
    if (!loop || !Array.isArray(loop.indices) || loop.indices.length < 3) continue;

    const areaNow = Math.abs(signedAreaCurrent(soft.nodes, loop.indices));
    if (!Number.isFinite(areaNow) || areaNow < 1e-6) continue;

    const seededBase = Math.max(1e-4, Number(membrane?.restArea) || areaNow);
    if (!sim.softMembraneAreaBaseline.has(cid)) {
      sim.softMembraneAreaBaseline.set(cid, seededBase);
    }
    const areaBase = Math.max(1e-4, Number(sim.softMembraneAreaBaseline.get(cid)) || seededBase);
    const err = clamp((areaBase - areaNow) / areaBase, -0.65, 0.65);
    if (Math.abs(err) < 1e-4) continue;

    const pressureGain = Math.max(0.005, Number(membrane?.pressureGain) || membraneCellBasePressureGain);
    const radialDamping = clamp(Number(membrane?.radialDamping) || membraneCellBaseRadialDamping, 0, 0.2);
    const gain = pressureGain * (1 + Math.min(1.4, Math.abs(err) * 2.2));

    let cx = 0;
    let cy = 0;
    for (const ni of loop.indices) {
      const node = soft.nodes[ni];
      if (!node) continue;
      cx += node.x;
      cy += node.y;
    }
    cx /= loop.indices.length;
    cy /= loop.indices.length;

    for (let k = 0; k < loop.indices.length; k++) {
      const iPrev = loop.indices[(k - 1 + loop.indices.length) % loop.indices.length];
      const iCurr = loop.indices[k];
      const iNext = loop.indices[(k + 1) % loop.indices.length];
      const prev = soft.nodes[iPrev];
      const curr = soft.nodes[iCurr];
      const next = soft.nodes[iNext];
      if (!prev || !curr || !next) continue;

      let nx = (curr.y - prev.y) + (next.y - curr.y);
      let ny = -((curr.x - prev.x) + (next.x - curr.x));
      let nLen = Math.hypot(nx, ny);
      if (!Number.isFinite(nLen) || nLen < 1e-6) {
        nx = curr.x - cx;
        ny = curr.y - cy;
        nLen = Math.hypot(nx, ny);
      }
      if (!Number.isFinite(nLen) || nLen < 1e-6) continue;
      nx /= nLen;
      ny /= nLen;

      const invMass = 1 / Math.max(0.02, curr.mass || 1);
      const impulse = err * gain * dtPos * invMass;
      curr.vx += nx * impulse;
      curr.vy += ny * impulse;

      if (radialDamping > 0) {
        const rv = curr.vx * nx + curr.vy * ny;
        curr.vx -= nx * rv * radialDamping;
        curr.vy -= ny * rv * radialDamping;
      }
    }

    touched += 1;
  }

  return touched;
}

test('soft membrane pressure gpu-only path matches baseline while publishing WGSL prep metadata', () => {
  const loops = [
    { clusterId: 0, indices: [0, 1, 2, 3] },
    { clusterId: 1, indices: [4, 5, 6] },
  ];

  const seedNodes = [
    { x: 0.0, y: 0.0, vx: 0.02, vy: -0.01, mass: 1.1 },
    { x: 1.2, y: 0.0, vx: -0.01, vy: 0.04, mass: 1.0 },
    { x: 1.2, y: 1.0, vx: -0.03, vy: -0.02, mass: 0.95 },
    { x: 0.0, y: 1.0, vx: 0.01, vy: 0.03, mass: 1.05 },
    { x: 2.0, y: 0.1, vx: -0.04, vy: 0.01, mass: 1.2 },
    { x: 3.1, y: 0.4, vx: 0.02, vy: -0.03, mass: 1.15 },
    { x: 2.6, y: 1.2, vx: 0.01, vy: 0.05, mass: 1.0 },
  ];

  const seedMembranes = [
    { clusterId: 0, restArea: 1.42, pressureGain: 0.09, radialDamping: 0.04 },
    { clusterId: 1, restArea: 0.58, pressureGain: 0.07, radialDamping: 0.03 },
  ];

  const baselineSim = {
    bodies: { softMembraneClusters: structuredClone(seedMembranes) },
    softMembraneAreaBaseline: new Map([[0, 1.42], [1, 0.58]]),
  };
  const gpuSim = {
    bodies: { softMembraneClusters: structuredClone(seedMembranes) },
    softMembraneAreaBaseline: new Map([[0, 1.42], [1, 0.58]]),
  };
  const baselineSoft = { nodes: structuredClone(seedNodes) };
  const gpuSoft = { nodes: structuredClone(seedNodes) };
  const wgslState = {};

  const baselineTouched = applySoftMembraneCellPressureBaseline({
    sim: baselineSim,
    soft: baselineSoft,
    loops,
    dtPos: 0.83,
  });

  const gpuTouched = applySoftMembraneCellPressureGpuOnly({
    sim: gpuSim,
    soft: gpuSoft,
    loops,
    dtPos: 0.83,
    signedAreaCurrent,
    clamp,
    wgslOffload: {
      enabled: true,
      state: wgslState,
    },
  });

  assert.equal(gpuTouched, baselineTouched);
  assert.deepEqual(gpuSoft, baselineSoft, 'gpu-only membrane pressure should preserve baseline node velocity semantics');
  assert.deepEqual([...gpuSim.softMembraneAreaBaseline.entries()], [...baselineSim.softMembraneAreaBaseline.entries()]);

  assert.equal(wgslState.lastMode, 'cpu-prepared');
  assert.equal(wgslState.lastPreparedMembraneCount, seedMembranes.length);
  assert.equal(wgslState.preparedLayout?.membraneOffsets instanceof Uint32Array, true);
  assert.equal(wgslState.preparedLayout?.loopIndices instanceof Uint32Array, true);
  assert.equal(wgslState.preparedLayout?.loopMembraneIndex instanceof Uint32Array, true);
  assert.equal(wgslState.preparedLayout?.nodeX instanceof Float32Array, true);
  assert.equal(wgslState.preparedLayout?.nodeMass instanceof Float32Array, true);
  assert.equal(wgslState.lastVelocityProposalSource, 'cpu-membrane-authoritative');
  assert.ok((wgslState.lastPreparedLayoutBytes || 0) > 0);
});

test('soft membrane pressure gpu-only route uses WGSL proposal as authoritative when deterministic signature matches', () => {
  const loops = [
    { clusterId: 0, indices: [0, 1, 2, 3] },
  ];

  const seedNodes = [
    { x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, mass: 1.0 },
    { x: 1.0, y: 0.0, vx: 0.0, vy: 0.0, mass: 1.0 },
    { x: 1.0, y: 1.0, vx: 0.0, vy: 0.0, mass: 1.0 },
    { x: 0.0, y: 1.0, vx: 0.0, vy: 0.0, mass: 1.0 },
  ];
  const membranes = [{ clusterId: 0, restArea: 1.4, pressureGain: 0.09, radialDamping: 0.0 }];

  const sim = {
    frame: 42,
    bodies: { softMembraneClusters: structuredClone(membranes) },
    softMembraneAreaBaseline: new Map([[0, 1.4]]),
  };
  const soft = { nodes: structuredClone(seedNodes) };
  const wgslState = {};

  // First pass prepares deterministic signature/layout.
  applySoftMembraneCellPressureGpuOnly({
    sim,
    soft: { nodes: structuredClone(seedNodes) },
    loops,
    dtPos: 0.5,
    signedAreaCurrent,
    clamp,
    wgslOffload: { enabled: true, state: wgslState },
  });

  const signature = wgslState.lastPreparedProposalSignature;
  const contributionCount = wgslState.preparedPlan.indexCount;
  wgslState.lastVelocityProposalSignature = signature;
  wgslState.lastVelocityProposalFinite = { allFinite: true };
  wgslState.lastVelocityProposalDeltaVx = new Float32Array(contributionCount).fill(0.02);
  wgslState.lastVelocityProposalDeltaVy = new Float32Array(contributionCount).fill(-0.01);

  applySoftMembraneCellPressureGpuOnly({
    sim,
    soft,
    loops,
    dtPos: 0.5,
    signedAreaCurrent,
    clamp,
    wgslOffload: { enabled: true, state: wgslState },
  });

  for (const node of soft.nodes) {
    assert.equal(Number.isFinite(node.vx), true);
    assert.equal(Number.isFinite(node.vy), true);
    assert.ok(Math.abs(node.vx) > 1e-5 || Math.abs(node.vy) > 1e-5);
  }
  assert.equal(wgslState.lastVelocityProposalSource, 'wgsl-pressure-authoritative');
  assert.equal(wgslState.lastMode, 'wgsl-pressure-authoritative');
  assert.equal(wgslState.lastAuthoritativeProposalSignature, signature);
  assert.equal(wgslState.lastAuthoritativeProposalFrame, 42);
});
