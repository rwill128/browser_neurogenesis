import test from 'node:test';
import assert from 'node:assert/strict';
import { applyBodyFluidInjectionGpuOnly } from '../../sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js';

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function clampFluidComponent(v, limitRaw) {
  const limit = Number.isFinite(Number(limitRaw)) ? clamp(Number(limitRaw), 0.25, 48) : 12;
  if (!Number.isFinite(v)) return 0;
  return clamp(v, -limit, limit);
}

function sampleFieldBilinear(field, n, x, y) {
  const ix = clamp(Math.round(x), 0, n - 1);
  const iy = clamp(Math.round(y), 0, n - 1);
  return field[iy * n + ix] || 0;
}

function computeSoftClusterKinematics(nodes) {
  const map = new Map();
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const st = map.get(cid) || { mass: 0, x: 0, y: 0, vx: 0, vy: 0, inertia: 0, omegaNum: 0 };
    const m = Math.max(0.02, Number(node.mass) || 1);
    st.mass += m;
    st.x += node.x * m;
    st.y += node.y * m;
    st.vx += node.vx * m;
    st.vy += node.vy * m;
    map.set(cid, st);
  }
  for (const st of map.values()) {
    const inv = st.mass > 0 ? (1 / st.mass) : 0;
    st.x *= inv; st.y *= inv; st.vx *= inv; st.vy *= inv;
  }
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const st = map.get(cid);
    const m = Math.max(0.02, Number(node.mass) || 1);
    const rx = node.x - st.x;
    const ry = node.y - st.y;
    st.inertia += m * (rx * rx + ry * ry);
    st.omegaNum += m * (rx * (node.vy - st.vy) - ry * (node.vx - st.vx));
  }
  for (const st of map.values()) {
    const inertia = Math.max(1e-4, st.inertia);
    st.inertia = inertia;
    st.omega = st.omegaNum / inertia;
  }
  return map;
}

function rigidEdgeMomentumScale(rb) {
  const arr = Array.isArray(rb?.edgeMomentumCoupling)
    ? rb.edgeMomentumCoupling
    : (Array.isArray(rb?.edgeMomentumTransfer) ? rb.edgeMomentumTransfer : null);
  if (!arr || arr.length === 0) return 1;
  let sum = 0; let c = 0;
  for (const v of arr) {
    const value = Number(v);
    if (!Number.isFinite(value)) continue;
    sum += clamp(value, 0, 1);
    c += 1;
  }
  return c > 0 ? (sum / c) : 1;
}

function deepClone(v) {
  return JSON.parse(JSON.stringify(v));
}

function runBaselineBodyFluidInjection(args) {
  const {
    sim, bodies, s, n, vxField, vyField, feedbackK, swimGain, bodyFeedbackCurrVx,
    bodyFeedbackCurrVy, softNodeMomentumScale,
  } = args;

  let injectedMomentum = 0;
  let softClusterForInjection;

  const injectPoint = (px, py, pvx, pvy, localFluidX, localFluidY, mass, rad = 3.0, swimInjectX = 0, swimInjectY = 0, momentumScale = 1) => {
    if (!Number.isFinite(px) || !Number.isFinite(py)) return;
    const radius = Math.max(0.4, Number.isFinite(rad) ? rad : 3.0);
    const minX = Math.max(0, Math.floor(px - radius));
    const maxX = Math.min(n - 1, Math.ceil(px + radius));
    const minY = Math.max(0, Math.floor(py - radius));
    const maxY = Math.min(n - 1, Math.ceil(py + radius));
    const couplingLimit = Number(sim?.controls?.fluidCouplingComponentLimit) || 12;
    const relXRaw = pvx - localFluidX + swimInjectX;
    const relYRaw = pvy - localFluidY + swimInjectY;
    const relX = clampFluidComponent(relXRaw, couplingLimit);
    const relY = clampFluidComponent(relYRaw, couplingLimit);
    const scaleRaw = feedbackK * Math.max(0.1, Number.isFinite(mass) ? mass : 0.1) * clamp(Number(momentumScale), 0, 1);
    const scale = Number.isFinite(scaleRaw) ? clamp(scaleRaw, 0, couplingLimit) : 0;
    if (scale <= 0) return;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - px;
        const dy = y - py;
        const d = Math.hypot(dx, dy);
        if (d > radius) continue;
        const w = 1 - d / radius;
        const idx = y * n + x;
        const jx = clampFluidComponent(relX * scale * w, couplingLimit);
        const jy = clampFluidComponent(relY * scale * w, couplingLimit);
        const nextVx = Number(vxField[idx]) + jx;
        const nextVy = Number(vyField[idx]) + jy;
        vxField[idx] = clampFluidComponent(nextVx, couplingLimit);
        vyField[idx] = clampFluidComponent(nextVy, couplingLimit);
        bodyFeedbackCurrVx[idx] += jx;
        bodyFeedbackCurrVy[idx] += jy;
        injectedMomentum += Math.hypot(jx, jy);
      }
    }
  };

  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const fx = sampleFieldBilinear(vxField, n, b.x, b.y);
    const fy = sampleFieldBilinear(vyField, n, b.x, b.y);
    const swimPhase = sim.frame * 0.08 + bi * 2.1;
    injectPoint(
      b.x, b.y, b.vx, b.vy, fx, fy, b.mass, b.r * 0.8,
      swimGain * Math.cos(swimPhase) * 0.015,
      swimGain * Math.sin(swimPhase) * 0.012,
      rigidEdgeMomentumScale(b),
    );
  }

  softClusterForInjection = computeSoftClusterKinematics(s.nodes);
  for (let i = 0; i < s.nodes.length; i++) {
    const node = s.nodes[i];
    const fx = sampleFieldBilinear(vxField, n, node.x, node.y);
    const fy = sampleFieldBilinear(vyField, n, node.x, node.y);
    const cid = node.clusterId ?? 0;
    const c = softClusterForInjection.get(cid);
    const cx = Number.isFinite(Number(c?.x)) ? Number(c.x) : node.x;
    const cy = Number.isFinite(Number(c?.y)) ? Number(c.y) : node.y;
    const cvx = Number.isFinite(Number(c?.vx)) ? Number(c.vx) : node.vx;
    const cvy = Number.isFinite(Number(c?.vy)) ? Number(c.vy) : node.vy;
    const omega = Number.isFinite(Number(c?.omega)) ? Number(c.omega) : 0;
    const rx = node.x - cx;
    const ry = node.y - cy;
    const rigidLikeVx = cvx - omega * ry;
    const rigidLikeVy = cvy + omega * rx;
    const blend = 0.55;
    const injectVx = node.vx * (1 - blend) + rigidLikeVx * blend;
    const injectVy = node.vy * (1 - blend) + rigidLikeVy * blend;
    const swimPhase = sim.frame * 0.12 + i * 1.57;
    injectPoint(
      node.x, node.y, injectVx, injectVy, fx, fy, node.mass, 2.2,
      swimGain * Math.cos(swimPhase) * 0.01,
      swimGain * Math.sin(swimPhase) * 0.01,
      softNodeMomentumScale(i),
    );
  }

  return { injectedMomentum, softClusterForInjection };
}

test('applyBodyFluidInjectionGpuOnly matches baseline injection momentum and field updates', () => {
  const n = 24;
  const cells = n * n;
  const baseState = {
    sim: { frame: 37, controls: { fluidCouplingComponentLimit: 9.5 } },
    bodies: {
      rigid: [
        { x: 8.2, y: 5.8, vx: 1.4, vy: -0.7, mass: 1.5, r: 2.8, edgeMomentumCoupling: [1, 0.8, 0.9] },
        { x: 13.1, y: 14.7, vx: -0.9, vy: 1.2, mass: 2.2, r: 3.1, edgeMomentumTransfer: [0.35, 0.55] },
      ],
    },
    s: {
      nodes: [
        { x: 7.5, y: 7.1, vx: 0.8, vy: -0.2, mass: 0.9, clusterId: 0 },
        { x: 9.1, y: 8.4, vx: -0.3, vy: 0.6, mass: 1.1, clusterId: 0 },
        { x: 14.3, y: 13.7, vx: 1.0, vy: -0.4, mass: 1.0, clusterId: 1 },
        { x: 16.0, y: 15.2, vx: -0.7, vy: 0.9, mass: 0.8, clusterId: 1 },
      ],
    },
    vxField: Float32Array.from({ length: cells }, (_, i) => Math.sin(i * 0.07) * 0.6),
    vyField: Float32Array.from({ length: cells }, (_, i) => Math.cos(i * 0.05) * 0.5),
    bodyFeedbackCurrVx: new Float32Array(cells),
    bodyFeedbackCurrVy: new Float32Array(cells),
  };

  const feedbackK = 0.014;
  const swimGain = 0.33;
  const softNodeMomentumScale = (idx) => (idx % 2 === 0 ? 0.85 : 0.6);

  const baseline = deepClone({
    sim: baseState.sim,
    bodies: baseState.bodies,
    s: baseState.s,
  });
  const baselineVx = new Float32Array(baseState.vxField);
  const baselineVy = new Float32Array(baseState.vyField);
  const baselineFeedbackX = new Float32Array(baseState.bodyFeedbackCurrVx);
  const baselineFeedbackY = new Float32Array(baseState.bodyFeedbackCurrVy);

  const gpuOnly = deepClone({
    sim: baseState.sim,
    bodies: baseState.bodies,
    s: baseState.s,
  });
  const gpuVx = new Float32Array(baseState.vxField);
  const gpuVy = new Float32Array(baseState.vyField);
  const gpuFeedbackX = new Float32Array(baseState.bodyFeedbackCurrVx);
  const gpuFeedbackY = new Float32Array(baseState.bodyFeedbackCurrVy);

  const baselineResult = runBaselineBodyFluidInjection({
    sim: baseline.sim,
    bodies: baseline.bodies,
    s: baseline.s,
    n,
    vxField: baselineVx,
    vyField: baselineVy,
    feedbackK,
    swimGain,
    bodyFeedbackCurrVx: baselineFeedbackX,
    bodyFeedbackCurrVy: baselineFeedbackY,
    softNodeMomentumScale,
  });

  const gpuOnlyResult = applyBodyFluidInjectionGpuOnly({
    sim: gpuOnly.sim,
    bodies: gpuOnly.bodies,
    soft: gpuOnly.s,
    n,
    vxField: gpuVx,
    vyField: gpuVy,
    feedbackK,
    swimGain,
    bodyFeedbackCurrVx: gpuFeedbackX,
    bodyFeedbackCurrVy: gpuFeedbackY,
    rigidEdgeMomentumScale,
    softNodeMomentumScale,
    sampleFieldBilinear,
    computeSoftClusterKinematics,
    softClusterFluidInjectBlend: 0.55,
    fluidCouplingComponentLimit: gpuOnly.sim.controls.fluidCouplingComponentLimit,
  });

  assert.ok(Math.abs(gpuOnlyResult.injectedMomentum - baselineResult.injectedMomentum) < 1e-9);

  for (let i = 0; i < cells; i++) {
    assert.ok(Math.abs(gpuVx[i] - baselineVx[i]) < 1e-9, `vx mismatch @ ${i}`);
    assert.ok(Math.abs(gpuVy[i] - baselineVy[i]) < 1e-9, `vy mismatch @ ${i}`);
    assert.ok(Math.abs(gpuFeedbackX[i] - baselineFeedbackX[i]) < 1e-9, `feedback vx mismatch @ ${i}`);
    assert.ok(Math.abs(gpuFeedbackY[i] - baselineFeedbackY[i]) < 1e-9, `feedback vy mismatch @ ${i}`);
  }

  const baselineClusters = baselineResult.softClusterForInjection;
  const gpuClusters = gpuOnlyResult.softClusterForInjection;
  assert.equal(gpuClusters.size, baselineClusters.size);
  for (const [cid, c0] of baselineClusters.entries()) {
    const c1 = gpuClusters.get(cid);
    assert.ok(c1, `missing cluster ${cid}`);
    assert.ok(Math.abs(c1.x - c0.x) < 1e-9);
    assert.ok(Math.abs(c1.y - c0.y) < 1e-9);
    assert.ok(Math.abs(c1.vx - c0.vx) < 1e-9);
    assert.ok(Math.abs(c1.vy - c0.vy) < 1e-9);
    assert.ok(Math.abs(c1.omega - c0.omega) < 1e-9);
  }
});
