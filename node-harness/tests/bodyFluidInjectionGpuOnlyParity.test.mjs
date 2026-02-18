import test from 'node:test';
import assert from 'node:assert/strict';

import { applyBodyFluidInjectionGpuOnly } from '../../sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js';

function buildFixture() {
  const n = 6;
  const vxField = new Float32Array(n * n);
  const vyField = new Float32Array(n * n);
  const bodyFeedbackCurrVx = new Float32Array(n * n);
  const bodyFeedbackCurrVy = new Float32Array(n * n);

  const rigid = [{ x: 2.1, y: 2.4, vx: 0.5, vy: -0.2, mass: 1.2, r: 0.9 }];
  const softNodes = [
    { x: 3.0, y: 2.0, vx: -0.1, vy: 0.2, mass: 0.8, clusterId: 1 },
    { x: 3.4, y: 2.3, vx: 0.2, vy: -0.3, mass: 1.0, clusterId: 1 },
  ];

  const soft = { nodes: softNodes };
  const bodies = { rigid };
  const sim = { frame: 12 };

  const sampleFieldBilinear = (field, size, x, y) => {
    const xi = Math.max(0, Math.min(size - 1, Math.round(x)));
    const yi = Math.max(0, Math.min(size - 1, Math.round(y)));
    return Number(field[yi * size + xi]) || 0;
  };

  const computeSoftClusterKinematics = (nodes) => {
    const m = new Map();
    for (const node of nodes) {
      const cid = node.clusterId ?? 0;
      if (!m.has(cid)) m.set(cid, { x: 0, y: 0, vx: 0, vy: 0, omega: 0, c: 0 });
      const entry = m.get(cid);
      entry.x += node.x;
      entry.y += node.y;
      entry.vx += node.vx;
      entry.vy += node.vy;
      entry.c += 1;
    }
    for (const entry of m.values()) {
      const c = Math.max(1, entry.c);
      entry.x /= c;
      entry.y /= c;
      entry.vx /= c;
      entry.vy /= c;
    }
    return m;
  };

  return {
    sim,
    bodies,
    soft,
    n,
    vxField,
    vyField,
    bodyFeedbackCurrVx,
    bodyFeedbackCurrVy,
    sampleFieldBilinear,
    computeSoftClusterKinematics,
  };
}

test('body-fluid injection parity: gpu-only CPU-authoritative path matches with/without WGSL offload wiring', () => {
  const base = buildFixture();
  const withOffload = buildFixture();

  const commonArgs = {
    feedbackK: 0.6,
    swimGain: 0.3,
    rigidEdgeMomentumScale: () => 0.9,
    softNodeMomentumScale: () => 0.75,
    softClusterFluidInjectBlend: 0.55,
    fluidCouplingComponentLimit: 8,
  };

  const baselineResult = applyBodyFluidInjectionGpuOnly({
    ...base,
    ...commonArgs,
    wgslOffload: null,
  });

  const stubOffload = {
    enabled: true,
    // missing device intentionally -> no WGSL execution, CPU path must remain identical.
    state: {},
  };

  const offloadResult = applyBodyFluidInjectionGpuOnly({
    ...withOffload,
    ...commonArgs,
    wgslOffload: stubOffload,
  });

  assert.equal(offloadResult.injectedMomentum, baselineResult.injectedMomentum);
  assert.deepEqual(Array.from(withOffload.vxField), Array.from(base.vxField));
  assert.deepEqual(Array.from(withOffload.vyField), Array.from(base.vyField));
  assert.deepEqual(Array.from(withOffload.bodyFeedbackCurrVx), Array.from(base.bodyFeedbackCurrVx));
  assert.deepEqual(Array.from(withOffload.bodyFeedbackCurrVy), Array.from(base.bodyFeedbackCurrVy));
  assert.equal(stubOffload.state.lastPreparedPointCount, 3);
  assert.equal(stubOffload.state.lastMode, 'cpu-prepared');
});
