import test from 'node:test';
import assert from 'node:assert/strict';

import { applyBodyFluidInjectionGpuOnly } from '../../sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js';

function sampleFieldBilinear(field, n, x, y) {
  const cx = Math.max(0, Math.min(n - 1.001, Number(x) || 0));
  const cy = Math.max(0, Math.min(n - 1.001, Number(y) || 0));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(n - 1, x0 + 1);
  const y1 = Math.min(n - 1, y0 + 1);
  const sx = cx - x0;
  const sy = cy - y0;

  const i00 = y0 * n + x0;
  const i10 = y0 * n + x1;
  const i01 = y1 * n + x0;
  const i11 = y1 * n + x1;

  const a = Number(field[i00]) * (1 - sx) + Number(field[i10]) * sx;
  const b = Number(field[i01]) * (1 - sx) + Number(field[i11]) * sx;
  return a * (1 - sy) + b * sy;
}

function makeSoftClusterKinematics(nodes) {
  const m = new Map();
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const entry = m.get(cid) || { x: 0, y: 0, vx: 0, vy: 0, n: 0, omega: 0 };
    entry.x += Number(node.x) || 0;
    entry.y += Number(node.y) || 0;
    entry.vx += Number(node.vx) || 0;
    entry.vy += Number(node.vy) || 0;
    entry.n += 1;
    m.set(cid, entry);
  }
  for (const [cid, entry] of m) {
    const inv = 1 / Math.max(1, entry.n);
    m.set(cid, {
      x: entry.x * inv,
      y: entry.y * inv,
      vx: entry.vx * inv,
      vy: entry.vy * inv,
      omega: 0,
    });
  }
  return m;
}

test('body fluid injection gpu-only keeps cpu outputs stable while publishing wgsl prep layout metadata', () => {
  const n = 16;
  const size = n * n;

  const seedVx = new Float32Array(size);
  const seedVy = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    seedVx[i] = (i % 7) * 0.03;
    seedVy[i] = ((i + 3) % 5) * -0.02;
  }

  const bodies = {
    rigid: [
      { x: 5.2, y: 4.7, vx: 0.9, vy: -0.2, mass: 1.5, r: 2.8, edgeMomentumCoupling: [1, 0.8, 0.9] },
      { x: 10.1, y: 11.4, vx: -0.4, vy: 0.7, mass: 2.2, r: 3.1, edgeMomentumCoupling: [0.7, 0.6, 0.75] },
    ],
  };
  const soft = {
    nodes: [
      { x: 7.0, y: 9.0, vx: 0.2, vy: -0.1, mass: 0.6, clusterId: 0 },
      { x: 7.8, y: 9.4, vx: 0.1, vy: 0.3, mass: 0.7, clusterId: 0 },
      { x: 11.1, y: 5.5, vx: -0.3, vy: 0.2, mass: 0.65, clusterId: 1 },
    ],
  };

  const makeRun = (withPrep) => {
    const vxField = new Float32Array(seedVx);
    const vyField = new Float32Array(seedVy);
    const currVx = new Float32Array(size);
    const currVy = new Float32Array(size);
    const offload = withPrep ? { enabled: true, state: {} } : undefined;

    const result = applyBodyFluidInjectionGpuOnly({
      sim: { frame: 42 },
      bodies,
      soft,
      n,
      vxField,
      vyField,
      feedbackK: 0.01,
      swimGain: 0.8,
      bodyFeedbackCurrVx: currVx,
      bodyFeedbackCurrVy: currVy,
      rigidEdgeMomentumScale: (rb) => (Array.isArray(rb.edgeMomentumCoupling) ? rb.edgeMomentumCoupling[0] : 1),
      softNodeMomentumScale: (i) => (i % 2 === 0 ? 0.85 : 0.65),
      sampleFieldBilinear,
      computeSoftClusterKinematics: makeSoftClusterKinematics,
      softClusterFluidInjectBlend: 0.55,
      fluidCouplingComponentLimit: 12,
      wgslOffload: offload,
    });

    return { vxField, vyField, currVx, currVy, result, offload };
  };

  const plain = makeRun(false);
  const prepared = makeRun(true);

  assert.deepEqual(Array.from(prepared.vxField), Array.from(plain.vxField), 'wgsl prep mode should preserve vx field output');
  assert.deepEqual(Array.from(prepared.vyField), Array.from(plain.vyField), 'wgsl prep mode should preserve vy field output');
  assert.deepEqual(Array.from(prepared.currVx), Array.from(plain.currVx), 'wgsl prep mode should preserve feedback vx output');
  assert.deepEqual(Array.from(prepared.currVy), Array.from(plain.currVy), 'wgsl prep mode should preserve feedback vy output');
  assert.equal(prepared.result.injectedMomentum, plain.result.injectedMomentum, 'injected momentum should match in prep mode');

  const state = { ...(prepared?.offload?.state || {}) };
  assert.equal(state.lastMode, 'cpu-prepared');
  assert.equal(state.lastPreparedRigidCount, bodies.rigid.length);
  assert.equal(state.lastPreparedSoftCount, soft.nodes.length);
  assert.equal(state.lastPreparedPointCount, bodies.rigid.length + soft.nodes.length);
  assert.ok((state.lastPreparedLayoutBytes || 0) > 0, 'expected prepared layout byte footprint to be tracked');
  assert.equal(state.preparedLayout?.pointX instanceof Float32Array, true);
  assert.equal(state.preparedLayout?.pointVx instanceof Float32Array, true);
  assert.equal(state.preparedLayout?.pointX?.length, bodies.rigid.length + soft.nodes.length);
});
