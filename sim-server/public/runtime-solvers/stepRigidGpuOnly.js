function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function rigidEdgeMomentumScale(rb) {
  const arr = Array.isArray(rb?.edgeMomentumCoupling)
    ? rb.edgeMomentumCoupling
    : (Array.isArray(rb?.edgeMomentumTransfer) ? rb.edgeMomentumTransfer : null);
  if (!arr || arr.length === 0) return 1;
  let sum = 0;
  let c = 0;
  for (const v of arr) {
    const n = Number(v);
    if (Number.isFinite(n)) {
      sum += clamp(n, 0, 1);
      c += 1;
    }
  }
  return c > 0 ? (sum / c) : 1;
}

/**
 * GPU-only runtime rigid stepping path.
 * Isolated from baseline in gpu-lab.js so integration/coupling responsibility can
 * migrate incrementally while baseline remains the reference/default path.
 */
export function stepRigidBodiesGpuOnly({
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
}) {
  let rigidCarryTransfer = 0;

  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const edgeMomentumScale = rigidEdgeMomentumScale(b);
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
      const fx = sampleFluidForBodyCoupling(
        vxField,
        n,
        sx,
        sy,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVx,
        selfFeedbackSuppression,
      );
      const fy = sampleFluidForBodyCoupling(
        vyField,
        n,
        sx,
        sy,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVy,
        selfFeedbackSuppression,
      );
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
