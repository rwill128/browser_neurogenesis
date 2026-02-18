function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function clampComponent(v, limit) {
  if (!Number.isFinite(v)) return 0;
  return clamp(v, -limit, limit);
}

function buildBodyFluidInjectionWgslPrep({
  sim,
  bodies,
  soft,
  n,
  vxField,
  vyField,
  swimGain,
  rigidEdgeMomentumScale,
  softNodeMomentumScale,
  sampleFieldBilinear,
  computeSoftClusterKinematics,
  softClusterFluidInjectBlend,
}) {
  const rigidCount = Array.isArray(bodies?.rigid) ? bodies.rigid.length : 0;
  const softNodeCount = Array.isArray(soft?.nodes) ? soft.nodes.length : 0;
  const pointCount = rigidCount + softNodeCount;

  const pointX = new Float32Array(pointCount);
  const pointY = new Float32Array(pointCount);
  const pointVx = new Float32Array(pointCount);
  const pointVy = new Float32Array(pointCount);
  const localFluidX = new Float32Array(pointCount);
  const localFluidY = new Float32Array(pointCount);
  const pointMass = new Float32Array(pointCount);
  const pointRadius = new Float32Array(pointCount);
  const swimInjectX = new Float32Array(pointCount);
  const swimInjectY = new Float32Array(pointCount);
  const pointMomentumScale = new Float32Array(pointCount);

  let pointIndex = 0;

  for (let bi = 0; bi < rigidCount; bi++) {
    const b = bodies.rigid[bi];
    const fx = sampleFieldBilinear(vxField, n, b.x, b.y);
    const fy = sampleFieldBilinear(vyField, n, b.x, b.y);
    const swimPhase = sim.frame * 0.08 + bi * 2.1;

    pointX[pointIndex] = Number(b.x) || 0;
    pointY[pointIndex] = Number(b.y) || 0;
    pointVx[pointIndex] = Number(b.vx) || 0;
    pointVy[pointIndex] = Number(b.vy) || 0;
    localFluidX[pointIndex] = Number(fx) || 0;
    localFluidY[pointIndex] = Number(fy) || 0;
    pointMass[pointIndex] = Number(b.mass) || 0;
    pointRadius[pointIndex] = (Number(b.r) || 0) * 0.8;
    swimInjectX[pointIndex] = swimGain * Math.cos(swimPhase) * 0.015;
    swimInjectY[pointIndex] = swimGain * Math.sin(swimPhase) * 0.012;
    pointMomentumScale[pointIndex] = rigidEdgeMomentumScale(b);
    pointIndex += 1;
  }

  const softClusterForInjection = computeSoftClusterKinematics(soft.nodes);
  for (let i = 0; i < softNodeCount; i++) {
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

    pointX[pointIndex] = Number(node.x) || 0;
    pointY[pointIndex] = Number(node.y) || 0;
    pointVx[pointIndex] = Number(injectVx) || 0;
    pointVy[pointIndex] = Number(injectVy) || 0;
    localFluidX[pointIndex] = Number(fx) || 0;
    localFluidY[pointIndex] = Number(fy) || 0;
    pointMass[pointIndex] = Number(node.mass) || 0;
    pointRadius[pointIndex] = 2.2;
    swimInjectX[pointIndex] = swimGain * Math.cos(swimPhase) * 0.01;
    swimInjectY[pointIndex] = swimGain * Math.sin(swimPhase) * 0.01;
    pointMomentumScale[pointIndex] = softNodeMomentumScale(i);
    pointIndex += 1;
  }

  const layout = {
    pointX,
    pointY,
    pointVx,
    pointVy,
    localFluidX,
    localFluidY,
    pointMass,
    pointRadius,
    swimInjectX,
    swimInjectY,
    pointMomentumScale,
  };

  layout.byteLength = Object.values(layout).reduce((sum, arr) => sum + (arr?.byteLength || 0), 0);

  return {
    plan: {
      rigidCount,
      softNodeCount,
      pointCount,
      frame: Number(sim?.frame) || 0,
    },
    layout,
    softClusterForInjection,
  };
}

function buildBodyFluidInjectionGatherLayout({
  n,
  layout,
  feedbackK,
  fluidCouplingComponentLimit,
}) {
  const pointCount = Number(layout?.pointX?.length) || 0;
  const cellCount = Math.max(0, (Number(n) || 0) * (Number(n) || 0));
  const couplingLimitRaw = Number(fluidCouplingComponentLimit);
  const couplingLimit = Number.isFinite(couplingLimitRaw)
    ? clamp(couplingLimitRaw, 0.25, 48)
    : 12;

  const pointRelX = new Float32Array(pointCount);
  const pointRelY = new Float32Array(pointCount);
  const pointScale = new Float32Array(pointCount);
  const pointRadius = new Float32Array(pointCount);

  const perCell = Array.from({ length: cellCount }, () => []);

  for (let i = 0; i < pointCount; i++) {
    const px = Number(layout.pointX[i]) || 0;
    const py = Number(layout.pointY[i]) || 0;
    const pvx = Number(layout.pointVx[i]) || 0;
    const pvy = Number(layout.pointVy[i]) || 0;
    const localFluidX = Number(layout.localFluidX[i]) || 0;
    const localFluidY = Number(layout.localFluidY[i]) || 0;
    const swimInjectX = Number(layout.swimInjectX[i]) || 0;
    const swimInjectY = Number(layout.swimInjectY[i]) || 0;
    const mass = Number(layout.pointMass[i]) || 0;
    const momentumScale = clamp(Number(layout.pointMomentumScale[i]), 0, 1);
    const radius = Math.max(0.4, Number(layout.pointRadius[i]) || 0.4);

    const relX = clampComponent(pvx - localFluidX + swimInjectX, couplingLimit);
    const relY = clampComponent(pvy - localFluidY + swimInjectY, couplingLimit);
    const scaleRaw = feedbackK * Math.max(0.1, mass) * momentumScale;
    const scale = Number.isFinite(scaleRaw) ? clamp(scaleRaw, 0, couplingLimit) : 0;

    pointRelX[i] = relX;
    pointRelY[i] = relY;
    pointScale[i] = scale;
    pointRadius[i] = radius;

    if (scale <= 0 || !Number.isFinite(px) || !Number.isFinite(py)) continue;

    const minX = Math.max(0, Math.floor(px - radius));
    const maxX = Math.min(n - 1, Math.ceil(px + radius));
    const minY = Math.max(0, Math.floor(py - radius));
    const maxY = Math.min(n - 1, Math.ceil(py + radius));

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - px;
        const dy = y - py;
        const d = Math.hypot(dx, dy);
        if (d > radius) continue;
        const w = 1 - d / radius;
        const idx = y * n + x;
        perCell[idx].push({ pointIndex: i, weight: w });
      }
    }
  }

  const cellOffsets = new Uint32Array(cellCount + 1);
  let totalContrib = 0;
  for (let i = 0; i < cellCount; i++) {
    cellOffsets[i] = totalContrib;
    totalContrib += perCell[i].length;
  }
  cellOffsets[cellCount] = totalContrib;

  const contribPointIndex = new Uint32Array(totalContrib);
  const contribWeight = new Float32Array(totalContrib);
  let write = 0;
  for (let cell = 0; cell < cellCount; cell++) {
    const bucket = perCell[cell];
    for (let j = 0; j < bucket.length; j++) {
      contribPointIndex[write] = bucket[j].pointIndex;
      contribWeight[write] = bucket[j].weight;
      write += 1;
    }
  }

  return {
    pointRelX,
    pointRelY,
    pointScale,
    pointRadius,
    cellOffsets,
    contribPointIndex,
    contribWeight,
    contributionCount: totalContrib,
    byteLength:
      pointRelX.byteLength + pointRelY.byteLength + pointScale.byteLength + pointRadius.byteLength
      + cellOffsets.byteLength + contribPointIndex.byteLength + contribWeight.byteLength,
  };
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
  wgslOffload,
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

  const { plan, layout, softClusterForInjection } = buildBodyFluidInjectionWgslPrep({
    sim,
    bodies,
    soft,
    n,
    vxField,
    vyField,
    swimGain,
    rigidEdgeMomentumScale,
    softNodeMomentumScale,
    sampleFieldBilinear,
    computeSoftClusterKinematics,
    softClusterFluidInjectBlend,
  });

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const gatherLayout = buildBodyFluidInjectionGatherLayout({
      n,
      layout,
      feedbackK,
      fluidCouplingComponentLimit: couplingLimit,
    });

    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.preparedGatherLayout = gatherLayout;
    wgslOffload.state.lastPreparedRigidCount = plan.rigidCount;
    wgslOffload.state.lastPreparedSoftCount = plan.softNodeCount;
    wgslOffload.state.lastPreparedPointCount = plan.pointCount;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastPreparedGatherBytes = gatherLayout.byteLength;
    wgslOffload.state.lastPreparedGatherContributionCount = gatherLayout.contributionCount;
    // Blocker for immediate WGSL dispatch: this stage writes overlapping splat
    // circles into shared float fields. Gather layout now exists for a deterministic
    // per-cell reduction pass, but the reduction shader itself is still pending.
    wgslOffload.state.lastMode = 'cpu-prepared';
  }

  const count = plan.pointCount;
  for (let i = 0; i < count; i++) {
    injectPoint(
      layout.pointX[i],
      layout.pointY[i],
      layout.pointVx[i],
      layout.pointVy[i],
      layout.localFluidX[i],
      layout.localFluidY[i],
      layout.pointMass[i],
      layout.pointRadius[i],
      layout.swimInjectX[i],
      layout.swimInjectY[i],
      layout.pointMomentumScale[i],
    );
  }

  return {
    injectedMomentum,
    softClusterForInjection,
  };
}
