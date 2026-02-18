function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function clampComponent(v, limit) {
  if (!Number.isFinite(v)) return 0;
  return clamp(v, -limit, limit);
}

export function applyBodyFluidInjectionGpuOnly({
  sim,
  bodies,
  soft,
  n,
  vxField,
  vyField,
  feedbackK,
  swimGain,
  bodyFeedbackCurrVx,
  bodyFeedbackCurrVy,
  rigidEdgeMomentumScale,
  softNodeMomentumScale,
  sampleFieldBilinear,
  computeSoftClusterKinematics,
  softClusterFluidInjectBlend,
  fluidCouplingComponentLimit,
}) {
  const couplingLimitRaw = Number(fluidCouplingComponentLimit);
  const couplingLimit = Number.isFinite(couplingLimitRaw)
    ? clamp(couplingLimitRaw, 0.25, 48)
    : 12;

  let injectedMomentum = 0;

  const injectPoint = (
    px,
    py,
    pvx,
    pvy,
    localFluidX,
    localFluidY,
    mass,
    rad = 3.0,
    swimInjectX = 0,
    swimInjectY = 0,
    momentumScale = 1,
  ) => {
    if (!Number.isFinite(px) || !Number.isFinite(py)) return;
    const radius = Math.max(0.4, Number.isFinite(rad) ? rad : 3.0);
    const minX = Math.max(0, Math.floor(px - radius));
    const maxX = Math.min(n - 1, Math.ceil(px + radius));
    const minY = Math.max(0, Math.floor(py - radius));
    const maxY = Math.min(n - 1, Math.ceil(py + radius));

    const relXRaw = pvx - localFluidX + swimInjectX;
    const relYRaw = pvy - localFluidY + swimInjectY;
    const relX = clampComponent(relXRaw, couplingLimit);
    const relY = clampComponent(relYRaw, couplingLimit);
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
        const jx = clampComponent(relX * scale * w, couplingLimit);
        const jy = clampComponent(relY * scale * w, couplingLimit);

        vxField[idx] = clampComponent(Number(vxField[idx]) + jx, couplingLimit);
        vyField[idx] = clampComponent(Number(vyField[idx]) + jy, couplingLimit);
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
      b.x,
      b.y,
      b.vx,
      b.vy,
      fx,
      fy,
      b.mass,
      b.r * 0.8,
      swimGain * Math.cos(swimPhase) * 0.015,
      swimGain * Math.sin(swimPhase) * 0.012,
      rigidEdgeMomentumScale(b),
    );
  }

  const softClusterForInjection = computeSoftClusterKinematics(soft.nodes);
  for (let i = 0; i < soft.nodes.length; i++) {
    const node = soft.nodes[i];
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
    const blend = Number.isFinite(Number(softClusterFluidInjectBlend))
      ? clamp(Number(softClusterFluidInjectBlend), 0, 1)
      : 0.55;
    const injectVx = node.vx * (1 - blend) + rigidLikeVx * blend;
    const injectVy = node.vy * (1 - blend) + rigidLikeVy * blend;
    const swimPhase = sim.frame * 0.12 + i * 1.57;

    injectPoint(
      node.x,
      node.y,
      injectVx,
      injectVy,
      fx,
      fy,
      node.mass,
      2.2,
      swimGain * Math.cos(swimPhase) * 0.01,
      swimGain * Math.sin(swimPhase) * 0.01,
      softNodeMomentumScale(i),
    );
  }

  return {
    injectedMomentum,
    softClusterForInjection,
  };
}
