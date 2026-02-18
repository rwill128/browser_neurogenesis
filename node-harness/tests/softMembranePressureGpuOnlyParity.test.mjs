import test from 'node:test';
import assert from 'node:assert/strict';
import { applySoftMembraneCellPressureGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftMembranePressureGpuOnly.js';

const MEMBRANE_CELL_BASE_PRESSURE_GAIN = 0.08;
const MEMBRANE_CELL_BASE_RADIAL_DAMPING = 0.06;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function signedAreaCurrent(nodes, indices) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    s += a.x * b.y - b.x * a.y;
  }
  return 0.5 * s;
}

function applySoftMembraneCellPressureBaseline(sim, soft, loops, dtPos) {
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

    const pressureGain = Math.max(0.005, Number(membrane?.pressureGain) || MEMBRANE_CELL_BASE_PRESSURE_GAIN);
    const radialDamping = clamp(Number(membrane?.radialDamping) || MEMBRANE_CELL_BASE_RADIAL_DAMPING, 0, 0.2);
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

test('soft membrane cell-pressure parity: baseline and gpu-only module produce matching velocity updates', () => {
  const dtPos = 0.12;
  const seedSoft = {
    nodes: [
      { x: 8, y: 8, vx: 0.2, vy: -0.1, mass: 1.0, clusterId: 2 },
      { x: 14, y: 8.5, vx: -0.1, vy: 0.22, mass: 0.8, clusterId: 2 },
      { x: 15, y: 13.5, vx: 0.05, vy: -0.2, mass: 1.3, clusterId: 2 },
      { x: 9, y: 14, vx: -0.18, vy: 0.07, mass: 1.1, clusterId: 2 },
      { x: 20, y: 20, vx: 0.1, vy: 0.05, mass: 1.2, clusterId: 5 },
      { x: 24, y: 20, vx: -0.08, vy: -0.04, mass: 1.0, clusterId: 5 },
      { x: 22, y: 24, vx: 0.03, vy: -0.09, mass: 0.9, clusterId: 5 },
    ],
  };
  const loops = [
    { clusterId: 2, indices: [0, 1, 2, 3] },
    { clusterId: 5, indices: [4, 5, 6] },
    { clusterId: 77, indices: [0, 1] },
  ];

  const baselineSim = {
    bodies: {
      softMembraneClusters: [
        { clusterId: 2, restArea: 42.5, pressureGain: 0.09, radialDamping: 0.04 },
        { clusterId: 5, restArea: 11.2, pressureGain: 0.07, radialDamping: 0.05 },
      ],
    },
    softMembraneAreaBaseline: new Map([[2, 42.5], [5, 11.2]]),
  };
  const gpuOnlySim = structuredClone(baselineSim);
  gpuOnlySim.softMembraneAreaBaseline = new Map([[2, 42.5], [5, 11.2]]);

  const baselineSoft = structuredClone(seedSoft);
  const gpuOnlySoft = structuredClone(seedSoft);

  const baselineTouched = applySoftMembraneCellPressureBaseline(baselineSim, baselineSoft, loops, dtPos);
  const gpuOnlyTouched = applySoftMembraneCellPressureGpuOnly({
    sim: gpuOnlySim,
    soft: gpuOnlySoft,
    loops,
    dtPos,
    signedAreaCurrent,
    clamp,
    membraneCellBasePressureGain: MEMBRANE_CELL_BASE_PRESSURE_GAIN,
    membraneCellBaseRadialDamping: MEMBRANE_CELL_BASE_RADIAL_DAMPING,
  });

  assert.equal(gpuOnlyTouched, baselineTouched, 'touched membrane cluster count should match baseline');
  assert.deepEqual(
    Array.from(gpuOnlySim.softMembraneAreaBaseline.entries()),
    Array.from(baselineSim.softMembraneAreaBaseline.entries()),
    'membrane area baselines should stay aligned',
  );

  assert.equal(gpuOnlySoft.nodes.length, baselineSoft.nodes.length);
  for (let i = 0; i < baselineSoft.nodes.length; i++) {
    const b = baselineSoft.nodes[i];
    const g = gpuOnlySoft.nodes[i];
    assert.ok(Math.abs(g.vx - b.vx) < 1e-12, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-12, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    assert.ok(Math.abs(g.x - b.x) < 1e-12, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-12, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
  }
});
