import test from 'node:test';
import assert from 'node:assert/strict';

import { stepRigidBodiesGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidGpuOnly.js';

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function baselineRigidStep(args) {
  const {
    sim,
    bodies,
    vxField,
    vyField,
    n,
    dt,
    dtNorm,
    dragK,
    swimGain,
    localHoneyDrag,
    viscosityMotionResponse,
    obstacleMask,
    bodyFeedbackPrevVx,
    bodyFeedbackPrevVy,
    selfFeedbackSuppression,
    rigidVerticesWorld,
    sampleFluidForBodyCoupling,
    applyBounceBoundary,
  } = args;

  let rigidCarryTransfer = 0;
  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const arr = Array.isArray(b?.edgeMomentumCoupling) ? b.edgeMomentumCoupling : (Array.isArray(b?.edgeMomentumTransfer) ? b.edgeMomentumTransfer : null);
    const edgeMomentumScale = (() => {
      if (!arr || arr.length === 0) return 1;
      let sum = 0; let c = 0;
      for (const v of arr) {
        const value = Number(v);
        if (Number.isFinite(value)) {
          sum += clamp(value, 0, 1);
          c += 1;
        }
      }
      return c > 0 ? (sum / c) : 1;
    })();

    const invMass = 1 / Math.max(0.05, b.mass);
    const invInertia = 1 / Math.max(0.05, b.inertia || 1);
    const sampleVerts = rigidVerticesWorld(b);
    const sampleCount = Math.max(1, sampleVerts.length);
    let forceX = 0;
    let forceY = 0;
    let torque = 0;

    for (let si = 0; si < sampleCount; si++) {
      const sx = sampleVerts[si].x;
      const sy = sampleVerts[si].y;
      const rx = sx - b.x;
      const ry = sy - b.y;
      const fx = sampleFluidForBodyCoupling(vxField, n, sx, sy, rx, ry, obstacleMask, bodyFeedbackPrevVx, selfFeedbackSuppression);
      const fy = sampleFluidForBodyCoupling(vyField, n, sx, sy, rx, ry, obstacleMask, bodyFeedbackPrevVy, selfFeedbackSuppression);
      const localVx = b.vx + (-(b.omega || 0) * ry);
      const localVy = b.vy + ((b.omega || 0) * rx);
      const relX = fx - localVx;
      const relY = fy - localVy;
      const honey = localHoneyDrag(sx, sy);
      const fpx = relX * dragK * honey * edgeMomentumScale;
      const fpy = relY * dragK * honey * edgeMomentumScale;
      forceX += fpx;
      forceY += fpy;
      torque += rx * fpy - ry * fpx;
    }

    forceX /= sampleCount;
    forceY /= sampleCount;
    torque /= sampleCount;

    const ax = forceX * invMass;
    const ay = forceY * invMass;
    const alpha = torque * invInertia;

    const swimPhase = sim.frame * 0.08 + bi * 2.1;
    const swimX = swimGain * Math.cos(swimPhase) * 0.012 * invMass;
    const swimY = swimGain * Math.sin(swimPhase * 1.6) * 0.009 * invMass;
    const swimTorque = swimGain * Math.sin(swimPhase * 1.1) * 0.0025;

    b.vx += ax * dt * 60 + swimX * dtNorm;
    b.vy += ay * dt * 60 + swimY * dtNorm;
    b.omega = (b.omega || 0) + alpha * dt * 60 + swimTorque * dtNorm;

    const centerHoney = localHoneyDrag(b.x, b.y);
    const rigidVisc = viscosityMotionResponse(centerHoney, 3.2);
    b.vx *= rigidVisc.damp;
    b.vy *= rigidVisc.damp;
    b.omega *= Math.max(0.72, 0.99 - 0.01 * centerHoney);

    const bMax = rigidVisc.vmax;
    const bMag = Math.hypot(b.vx, b.vy);
    if (bMag > bMax) {
      b.vx = (b.vx / bMag) * bMax;
      b.vy = (b.vy / bMag) * bMax;
    }
    b.omega = Math.max(-0.25, Math.min(0.25, b.omega));

    rigidCarryTransfer += Math.hypot(ax, ay);
    b.x = b.x + b.vx * dt * 22;
    b.y = b.y + b.vy * dt * 28;
    b.theta = (b.theta || 0) + b.omega * dt * 60;
    applyBounceBoundary(b, n, 0.84);
  }

  return rigidCarryTransfer;
}

function makeArgs() {
  const n = 32;
  const cells = n * n;
  const mkField = (xScale, yScale) => {
    const f = new Float32Array(cells);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        f[y * n + x] = Math.sin((x + 1) * xScale) * 0.4 + Math.cos((y + 1) * yScale) * 0.3;
      }
    }
    return f;
  };

  return {
    sim: { frame: 37 },
    bodies: {
      rigid: [
        { x: 10.3, y: 9.2, vx: 0.7, vy: -0.2, omega: 0.03, theta: 0.1, mass: 1.4, inertia: 1.1, r: 1.7, edgeMomentumCoupling: [1, 0.6, 0.2] },
        { x: 16.1, y: 12.8, vx: -0.5, vy: 0.4, omega: -0.02, theta: -0.3, mass: 2.1, inertia: 1.9, r: 2.2, edgeMomentumTransfer: [0.9, 0.4, 0.7] },
      ],
    },
    vxField: mkField(0.2, 0.09),
    vyField: mkField(0.13, 0.15),
    n,
    dt: 0.012,
    dtNorm: 0.85,
    dragK: 1.15,
    swimGain: 0.6,
    localHoneyDrag: (x, y) => 1 + ((Math.sin(x * 0.21) + Math.cos(y * 0.17)) * 0.5 + 0.5) * 2.5,
    viscosityMotionResponse: (honey, vmaxBase) => ({
      damp: Math.max(0.72, 1.0 - 0.018 * honey),
      vmax: Math.max(0.4, vmaxBase / (1 + 0.28 * honey)),
    }),
    obstacleMask: new Float32Array(cells),
    bodyFeedbackPrevVx: new Float32Array(cells),
    bodyFeedbackPrevVy: new Float32Array(cells),
    selfFeedbackSuppression: 0.82,
    rigidVerticesWorld: (b) => [
      { x: b.x - 0.8, y: b.y - 0.4 },
      { x: b.x + 0.9, y: b.y - 0.2 },
      { x: b.x + 0.6, y: b.y + 0.7 },
      { x: b.x - 0.7, y: b.y + 0.5 },
    ],
    sampleFluidForBodyCoupling: (field, nGrid, sx, sy, rx, ry, _obs, _feedback, suppression) => {
      const xi = Math.max(0, Math.min(nGrid - 1, Math.round(sx)));
      const yi = Math.max(0, Math.min(nGrid - 1, Math.round(sy)));
      const base = field[yi * nGrid + xi] || 0;
      return base - suppression * 0.04 * (rx + ry);
    },
    applyBounceBoundary: (b, nGrid, e) => {
      if (b.x < 0) { b.x = 0; b.vx = Math.abs(b.vx) * e; }
      if (b.y < 0) { b.y = 0; b.vy = Math.abs(b.vy) * e; }
      if (b.x > nGrid - 1) { b.x = nGrid - 1; b.vx = -Math.abs(b.vx) * e; }
      if (b.y > nGrid - 1) { b.y = nGrid - 1; b.vy = -Math.abs(b.vy) * e; }
    },
  };
}

test('gpu-only rigid solver path matches baseline rigid integration/coupling step outcomes', async () => {
  const baselineArgs = makeArgs();
  const gpuArgs = makeArgs();

  const baselineCarry = baselineRigidStep(baselineArgs);
  const gpuCarry = await stepRigidBodiesGpuOnly(gpuArgs);

  assert.equal(gpuArgs.bodies.rigid.length, baselineArgs.bodies.rigid.length);
  for (let i = 0; i < baselineArgs.bodies.rigid.length; i++) {
    assert.deepEqual(gpuArgs.bodies.rigid[i], baselineArgs.bodies.rigid[i], `rigid body ${i} should match baseline stepping`);
  }
  assert.equal(gpuCarry, baselineCarry);
});
