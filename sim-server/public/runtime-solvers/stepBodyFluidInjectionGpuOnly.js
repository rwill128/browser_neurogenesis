const bodyFluidInjectionGatherProposalWgsl = /* wgsl */`
struct Params {
  cell_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  coupling_limit: f32,
  _pad3: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> point_rel_x: array<f32>;
@group(0) @binding(2) var<storage, read> point_rel_y: array<f32>;
@group(0) @binding(3) var<storage, read> point_scale: array<f32>;
@group(0) @binding(4) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> contrib_point_index: array<u32>;
@group(0) @binding(6) var<storage, read> contrib_weight: array<f32>;
@group(0) @binding(7) var<storage, read_write> cell_delta_vx: array<f32>;
@group(0) @binding(8) var<storage, read_write> cell_delta_vy: array<f32>;

fn clamp_component(v: f32, limit: f32) -> f32 {
  return clamp(v, -limit, limit);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x;
  if (cell >= params.cell_count) {
    return;
  }

  let coupling_limit = max(0.001, params.coupling_limit);
  let start = cell_offsets[cell];
  let stop = cell_offsets[cell + 1u];

  var sum_x = 0.0;
  var sum_y = 0.0;
  for (var i = start; i < stop; i = i + 1u) {
    let pi = contrib_point_index[i];
    let w = contrib_weight[i];
    let scale = point_scale[pi];
    let jx = clamp_component(point_rel_x[pi] * scale * w, coupling_limit);
    let jy = clamp_component(point_rel_y[pi] * scale * w, coupling_limit);
    sum_x = sum_x + jx;
    sum_y = sum_y + jy;
  }

  cell_delta_vx[cell] = clamp_component(sum_x, coupling_limit);
  cell_delta_vy[cell] = clamp_component(sum_y, coupling_limit);
}
`;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function clampComponent(v, limit) {
  if (!Number.isFinite(v)) return 0;
  return clamp(v, -limit, limit);
}

function canUseWgslOffload(offload) {
  return Boolean(
    offload
      && offload.enabled === true
      && offload.device
      && typeof offload.device.createBuffer === 'function'
      && typeof offload.device.createCommandEncoder === 'function'
      && offload.state,
  );
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

function computeBodyFluidInjectionCellDeltasFromGatherLayout({ gatherLayout, couplingLimit, n }) {
  const cellCount = Math.max(0, (Number(n) || 0) * (Number(n) || 0));
  const cellDeltaVx = new Float32Array(cellCount);
  const cellDeltaVy = new Float32Array(cellCount);

  if (!gatherLayout || cellCount <= 0) {
    return { cellDeltaVx, cellDeltaVy };
  }

  for (let cell = 0; cell < cellCount; cell++) {
    const start = Number(gatherLayout.cellOffsets[cell]) || 0;
    const stop = Number(gatherLayout.cellOffsets[cell + 1]) || start;
    let sumX = 0;
    let sumY = 0;

    for (let i = start; i < stop; i++) {
      const pointIndex = Number(gatherLayout.contribPointIndex[i]) || 0;
      const weight = Number(gatherLayout.contribWeight[i]) || 0;
      const scale = Number(gatherLayout.pointScale[pointIndex]) || 0;
      const relX = Number(gatherLayout.pointRelX[pointIndex]) || 0;
      const relY = Number(gatherLayout.pointRelY[pointIndex]) || 0;
      const jx = clampComponent(relX * scale * weight, couplingLimit);
      const jy = clampComponent(relY * scale * weight, couplingLimit);
      sumX += jx;
      sumY += jy;
    }

    cellDeltaVx[cell] = clampComponent(sumX, couplingLimit);
    cellDeltaVy[cell] = clampComponent(sumY, couplingLimit);
  }

  return { cellDeltaVx, cellDeltaVy };
}

function applyBodyFluidInjectionCellDeltas({
  n,
  cellDeltaVx,
  cellDeltaVy,
  couplingLimit,
  vxField,
  vyField,
  bodyFeedbackCurrVx,
  bodyFeedbackCurrVy,
}) {
  let injectedMomentum = 0;
  const cellCount = Math.max(0, (Number(n) || 0) * (Number(n) || 0));
  for (let cell = 0; cell < cellCount; cell++) {
    const jx = clampComponent(Number(cellDeltaVx[cell]) || 0, couplingLimit);
    const jy = clampComponent(Number(cellDeltaVy[cell]) || 0, couplingLimit);
    if (jx === 0 && jy === 0) continue;

    vxField[cell] = clampComponent(Number(vxField[cell]) + jx, couplingLimit);
    vyField[cell] = clampComponent(Number(vyField[cell]) + jy, couplingLimit);
    bodyFeedbackCurrVx[cell] += jx;
    bodyFeedbackCurrVy[cell] += jy;
    injectedMomentum += Math.hypot(jx, jy);
  }

  return injectedMomentum;
}

function ensureWgslGatherState({ offload, gatherLayout }) {
  const device = offload.device;
  const state = offload.state;
  const pointCapacity = Math.max(1, Number(gatherLayout?.pointRelX?.length) || 0);
  const cellCapacity = Math.max(1, (Number(gatherLayout?.cellOffsets?.length) || 1) - 1);
  const contribCapacity = Math.max(1, Number(gatherLayout?.contribWeight?.length) || 0);

  if (!state.gatherProposalParams) {
    state.gatherProposalParams = device.createBuffer({
      size: 32,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.gatherProposalBindGroup = null;
  }

  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  if ((state.gatherProposalPointCapacity || 0) < pointCapacity) {
    const bytes = pointCapacity * 4;
    state.gatherProposalPointRelX?.destroy?.();
    state.gatherProposalPointRelY?.destroy?.();
    state.gatherProposalPointScale?.destroy?.();
    state.gatherProposalPointRelX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.gatherProposalPointRelY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.gatherProposalPointScale = device.createBuffer({ size: bytes, usage: storageUsage });
    state.gatherProposalPointCapacity = pointCapacity;
    state.gatherProposalBindGroup = null;
  }

  if ((state.gatherProposalCellCapacity || 0) < cellCapacity) {
    const cellBytes = cellCapacity * 4;
    const offsetBytes = (cellCapacity + 1) * 4;
    state.gatherProposalCellOffsets?.destroy?.();
    state.gatherProposalDeltaVx?.destroy?.();
    state.gatherProposalDeltaVy?.destroy?.();
    state.gatherProposalCellOffsets = device.createBuffer({ size: offsetBytes, usage: storageUsage });
    state.gatherProposalDeltaVx = device.createBuffer({ size: cellBytes, usage: storageUsage });
    state.gatherProposalDeltaVy = device.createBuffer({ size: cellBytes, usage: storageUsage });
    state.gatherProposalCellCapacity = cellCapacity;
    state.gatherProposalBindGroup = null;
  }

  if ((state.gatherProposalContributionCapacity || 0) < contribCapacity) {
    const bytes = contribCapacity * 4;
    state.gatherProposalContribPointIndex?.destroy?.();
    state.gatherProposalContribWeight?.destroy?.();
    state.gatherProposalContribPointIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.gatherProposalContribWeight = device.createBuffer({ size: bytes, usage: storageUsage });
    state.gatherProposalContributionCapacity = contribCapacity;
    state.gatherProposalBindGroup = null;
  }

  if (!state.gatherProposalPipeline) {
    const module = device.createShaderModule({ code: bodyFluidInjectionGatherProposalWgsl });
    state.gatherProposalPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.gatherProposalBindGroup = null;
  }

  if (!state.gatherProposalBindGroup) {
    state.gatherProposalBindGroup = device.createBindGroup({
      layout: state.gatherProposalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.gatherProposalParams } },
        { binding: 1, resource: { buffer: state.gatherProposalPointRelX } },
        { binding: 2, resource: { buffer: state.gatherProposalPointRelY } },
        { binding: 3, resource: { buffer: state.gatherProposalPointScale } },
        { binding: 4, resource: { buffer: state.gatherProposalCellOffsets } },
        { binding: 5, resource: { buffer: state.gatherProposalContribPointIndex } },
        { binding: 6, resource: { buffer: state.gatherProposalContribWeight } },
        { binding: 7, resource: { buffer: state.gatherProposalDeltaVx } },
        { binding: 8, resource: { buffer: state.gatherProposalDeltaVy } },
      ],
    });
  }

  return state;
}

function dispatchBodyFluidInjectionGatherProposal({ offload, gatherLayout, couplingLimit, n }) {
  if (!canUseWgslOffload(offload)) return false;

  const state = ensureWgslGatherState({ offload, gatherLayout });
  const device = offload.device;
  const cellCount = Math.max(0, Number(n) || 0) * Math.max(0, Number(n) || 0);
  if (cellCount <= 0) return false;

  const paramsBytes = new ArrayBuffer(32);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  paramsU32[0] = cellCount >>> 0;
  paramsF32[4] = Math.max(0.001, Number(couplingLimit) || 0.001);

  device.queue.writeBuffer(state.gatherProposalParams, 0, paramsBytes);
  device.queue.writeBuffer(state.gatherProposalPointRelX, 0, gatherLayout.pointRelX);
  device.queue.writeBuffer(state.gatherProposalPointRelY, 0, gatherLayout.pointRelY);
  device.queue.writeBuffer(state.gatherProposalPointScale, 0, gatherLayout.pointScale);
  device.queue.writeBuffer(state.gatherProposalCellOffsets, 0, gatherLayout.cellOffsets);
  if (gatherLayout.contribPointIndex.length > 0) {
    device.queue.writeBuffer(state.gatherProposalContribPointIndex, 0, gatherLayout.contribPointIndex);
    device.queue.writeBuffer(state.gatherProposalContribWeight, 0, gatherLayout.contribWeight);
  }

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.gatherProposalPipeline);
  pass.setBindGroup(0, state.gatherProposalBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(cellCount / 64)));
  pass.end();
  device.queue.submit([encoder.finish()]);

  state.lastGatherProposalCellCount = cellCount;
  state.lastGatherProposalContributionCount = gatherLayout.contributionCount;
  state.lastGatherProposalDispatchCount = Math.max(1, Math.ceil(cellCount / 64));
  return true;
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

  const gatherLayout = buildBodyFluidInjectionGatherLayout({
    n,
    layout,
    feedbackK,
    fluidCouplingComponentLimit: couplingLimit,
  });

  const cpuGatherDelta = computeBodyFluidInjectionCellDeltasFromGatherLayout({
    gatherLayout,
    couplingLimit,
    n,
  });

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.preparedGatherLayout = gatherLayout;
    wgslOffload.state.lastPreparedRigidCount = plan.rigidCount;
    wgslOffload.state.lastPreparedSoftCount = plan.softNodeCount;
    wgslOffload.state.lastPreparedPointCount = plan.pointCount;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastPreparedGatherBytes = gatherLayout.byteLength;
    wgslOffload.state.lastPreparedGatherContributionCount = gatherLayout.contributionCount;
    wgslOffload.state.lastCpuGatherDeltaVx = cpuGatherDelta.cellDeltaVx;
    wgslOffload.state.lastCpuGatherDeltaVy = cpuGatherDelta.cellDeltaVy;

    let wgslGatherRan = false;
    try {
      wgslGatherRan = dispatchBodyFluidInjectionGatherProposal({
        offload: wgslOffload,
        gatherLayout,
        couplingLimit,
        n,
      });
      wgslOffload.state.lastError = null;
    } catch (err) {
      wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
      wgslGatherRan = false;
    }

    // CPU gather apply remains authoritative until solver-owned fluid fields
    // migrate from JS arrays to GPU storage buffers (or staged readback).
    wgslOffload.state.lastMode = wgslGatherRan ? 'wgsl-gather-proposal' : 'cpu-gather-authoritative';
  }

  injectedMomentum = applyBodyFluidInjectionCellDeltas({
    n,
    cellDeltaVx: cpuGatherDelta.cellDeltaVx,
    cellDeltaVy: cpuGatherDelta.cellDeltaVy,
    couplingLimit,
    vxField,
    vyField,
    bodyFeedbackCurrVx,
    bodyFeedbackCurrVy,
  });

  return {
    injectedMomentum,
    softClusterForInjection,
  };
}
