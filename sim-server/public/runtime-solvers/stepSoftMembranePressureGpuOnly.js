const WGSL_WORKGROUP_SIZE = 64;

const softMembranePressureAreaProbeWgsl = /* wgsl */`
struct Params {
  membrane_count: u32,
  _pad0: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> membrane_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> loop_indices: array<u32>;
@group(0) @binding(3) var<storage, read> node_x: array<f32>;
@group(0) @binding(4) var<storage, read> node_y: array<f32>;
@group(0) @binding(5) var<storage, read> area_base: array<f32>;
@group(0) @binding(6) var<storage, read_write> area_now_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> area_err_out: array<f32>;

fn abs_area(start: u32, stop: u32) -> f32 {
  var accum = 0.0;
  let count = stop - start;
  if (count < 3u) {
    return 0.0;
  }

  for (var i = start; i < stop; i = i + 1u) {
    let next = select(i + 1u, start, (i + 1u) >= stop);
    let ia = loop_indices[i];
    let ib = loop_indices[next];
    accum = accum + (node_x[ia] * node_y[ib] - node_y[ia] * node_x[ib]);
  }

  return abs(0.5 * accum);
}

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mi = gid.x;
  if (mi >= params.membrane_count) {
    return;
  }

  let start = membrane_offsets[mi];
  let stop = membrane_offsets[mi + 1u];
  let now = abs_area(start, stop);
  let base = max(1e-4, area_base[mi]);
  let err = clamp((base - now) / base, -0.65, 0.65);

  area_now_out[mi] = now;
  area_err_out[mi] = err;
}
`;

const softMembranePressureVelocityProposalWgsl = /* wgsl */`
struct Params {
  contribution_count: u32,
  dt_pos: f32,
  _pad0: vec2<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> membrane_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> loop_indices: array<u32>;
@group(0) @binding(3) var<storage, read> loop_membrane_index: array<u32>;
@group(0) @binding(4) var<storage, read> node_x: array<f32>;
@group(0) @binding(5) var<storage, read> node_y: array<f32>;
@group(0) @binding(6) var<storage, read> node_mass: array<f32>;
@group(0) @binding(7) var<storage, read> area_err: array<f32>;
@group(0) @binding(8) var<storage, read> pressure_gain: array<f32>;
@group(0) @binding(9) var<storage, read_write> delta_vx_out: array<f32>;
@group(0) @binding(10) var<storage, read_write> delta_vy_out: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let li = gid.x;
  if (li >= params.contribution_count) {
    return;
  }

  let mi = loop_membrane_index[li];
  let start = membrane_offsets[mi];
  let stop = membrane_offsets[mi + 1u];
  let count = stop - start;
  if (count < 3u) {
    delta_vx_out[li] = 0.0;
    delta_vy_out[li] = 0.0;
    return;
  }

  let local = li - start;
  let prev_local = select(local - 1u, count - 1u, local == 0u);
  let next_local = select(local + 1u, 0u, (local + 1u) >= count);

  let prev_idx = loop_indices[start + prev_local];
  let curr_idx = loop_indices[li];
  let next_idx = loop_indices[start + next_local];

  let prev_x = node_x[prev_idx];
  let prev_y = node_y[prev_idx];
  let curr_x = node_x[curr_idx];
  let curr_y = node_y[curr_idx];
  let next_x = node_x[next_idx];
  let next_y = node_y[next_idx];

  var nx = (curr_y - prev_y) + (next_y - curr_y);
  var ny = -((curr_x - prev_x) + (next_x - curr_x));
  let n_len = max(1e-6, sqrt(nx * nx + ny * ny));
  nx = nx / n_len;
  ny = ny / n_len;

  let err = area_err[mi];
  let gain = max(0.005, pressure_gain[mi]) * (1.0 + min(1.4, abs(err) * 2.2));
  let inv_mass = 1.0 / max(0.02, node_mass[curr_idx]);
  let impulse = err * gain * params.dt_pos * inv_mass;

  delta_vx_out[li] = nx * impulse;
  delta_vy_out[li] = ny * impulse;
}
`;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function canUseWgslOffload(offload) {
  return Boolean(
    offload
      && offload.enabled === true
      && offload.device
      && typeof offload.device.createComputePipelineAsync === 'function'
      && typeof offload.device.createBuffer === 'function'
      && typeof offload.device.createCommandEncoder === 'function'
      && offload.state,
  );
}

function buildSoftMembranePressureWgslPrep({ membranes, loopByCluster, softNodes, softMembraneAreaBaseline, membraneCellBasePressureGain = 0.08 }) {
  const valid = [];
  for (const membrane of membranes || []) {
    const cid = Number(membrane?.clusterId);
    if (!Number.isInteger(cid)) continue;
    const loop = loopByCluster.get(cid);
    if (!loop || !Array.isArray(loop.indices) || loop.indices.length < 3) continue;

    const seededBase = Math.max(1e-4, Number(membrane?.restArea) || 0);
    const areaBase = Math.max(1e-4, Number(softMembraneAreaBaseline.get(cid)) || seededBase || 1e-4);
    const pressureGain = Math.max(0.005, Number(membrane?.pressureGain) || membraneCellBasePressureGain);
    valid.push({ cid, loopIndices: loop.indices, areaBase, pressureGain });
  }

  const membraneCount = valid.length;
  const membraneOffsets = new Uint32Array(membraneCount + 1);
  let indexCount = 0;
  for (let i = 0; i < membraneCount; i++) {
    membraneOffsets[i] = indexCount;
    indexCount += valid[i].loopIndices.length;
  }
  membraneOffsets[membraneCount] = indexCount;

  const loopIndices = new Uint32Array(indexCount);
  const loopMembraneIndex = new Uint32Array(indexCount);
  const areaBase = new Float32Array(membraneCount);
  const pressureGain = new Float32Array(membraneCount);
  const clusterId = new Int32Array(membraneCount);

  let write = 0;
  for (let i = 0; i < membraneCount; i++) {
    const entry = valid[i];
    areaBase[i] = entry.areaBase;
    pressureGain[i] = entry.pressureGain;
    clusterId[i] = entry.cid;
    for (let k = 0; k < entry.loopIndices.length; k++) {
      loopIndices[write] = entry.loopIndices[k] >>> 0;
      loopMembraneIndex[write] = i;
      write += 1;
    }
  }

  const nodeCount = Array.isArray(softNodes) ? softNodes.length : 0;
  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  const nodeMass = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    nodeX[i] = Number(softNodes[i]?.x) || 0;
    nodeY[i] = Number(softNodes[i]?.y) || 0;
    nodeMass[i] = Math.max(0.02, Number(softNodes[i]?.mass) || 1);
  }

  const layout = {
    membraneOffsets,
    loopIndices,
    loopMembraneIndex,
    nodeX,
    nodeY,
    nodeMass,
    areaBase,
    pressureGain,
    clusterId,
  };
  layout.byteLength = membraneOffsets.byteLength
    + loopIndices.byteLength
    + loopMembraneIndex.byteLength
    + nodeX.byteLength
    + nodeY.byteLength
    + nodeMass.byteLength
    + areaBase.byteLength
    + pressureGain.byteLength
    + clusterId.byteLength;

  return {
    plan: { membraneCount, nodeCount, indexCount },
    layout,
  };
}

function buildCpuMembranePressureVelocityDeltaProposal(prep, dtPos) {
  const contributionCount = prep.plan.indexCount;
  const deltaVx = new Float32Array(contributionCount);
  const deltaVy = new Float32Array(contributionCount);
  const areaErr = prep.areaErr;
  if (!(areaErr instanceof Float32Array) || areaErr.length === 0) {
    return { deltaVx, deltaVy };
  }

  for (let li = 0; li < contributionCount; li++) {
    const mi = prep.layout.loopMembraneIndex[li] >>> 0;
    const start = prep.layout.membraneOffsets[mi] >>> 0;
    const stop = prep.layout.membraneOffsets[mi + 1] >>> 0;
    const count = stop - start;
    if (count < 3) continue;

    const local = li - start;
    const prevLocal = local === 0 ? count - 1 : local - 1;
    const nextLocal = (local + 1) >= count ? 0 : local + 1;

    const prevIdx = prep.layout.loopIndices[start + prevLocal] >>> 0;
    const currIdx = prep.layout.loopIndices[li] >>> 0;
    const nextIdx = prep.layout.loopIndices[start + nextLocal] >>> 0;

    const prevX = prep.layout.nodeX[prevIdx] || 0;
    const prevY = prep.layout.nodeY[prevIdx] || 0;
    const currX = prep.layout.nodeX[currIdx] || 0;
    const currY = prep.layout.nodeY[currIdx] || 0;
    const nextX = prep.layout.nodeX[nextIdx] || 0;
    const nextY = prep.layout.nodeY[nextIdx] || 0;

    let nx = (currY - prevY) + (nextY - currY);
    let ny = -((currX - prevX) + (nextX - currX));
    const nLen = Math.max(1e-6, Math.hypot(nx, ny));
    nx /= nLen;
    ny /= nLen;

    const err = areaErr[mi] || 0;
    const gain = Math.max(0.005, prep.layout.pressureGain[mi] || 0.08) * (1 + Math.min(1.4, Math.abs(err) * 2.2));
    const invMass = 1 / Math.max(0.02, prep.layout.nodeMass[currIdx] || 1);
    const impulse = err * gain * dtPos * invMass;

    deltaVx[li] = nx * impulse;
    deltaVy[li] = ny * impulse;
  }

  return { deltaVx, deltaVy };
}

function computeProposalParityStats(expected, actual) {
  const count = Math.min(expected.length, actual.length);
  if (count <= 0) {
    return { count: 0, absMean: 0, absMax: 0 };
  }
  let absSum = 0;
  let absMax = 0;
  for (let i = 0; i < count; i++) {
    const d = Math.abs((actual[i] || 0) - (expected[i] || 0));
    absSum += d;
    if (d > absMax) absMax = d;
  }
  return {
    count,
    absMean: absSum / count,
    absMax,
  };
}

function computeMembraneProposalSignature(prep, dtPos) {
  const plan = prep?.plan || {};
  const layout = prep?.layout || {};
  const indexCount = Number(plan.indexCount) || 0;
  const membraneCount = Number(plan.membraneCount) || 0;
  const nodeCount = Number(plan.nodeCount) || 0;
  const dt = Number.isFinite(Number(dtPos)) ? Number(dtPos).toFixed(8) : '0.00000000';
  const firstCluster = (layout.clusterId instanceof Int32Array && layout.clusterId.length > 0)
    ? Number(layout.clusterId[0])
    : -1;
  const lastCluster = (layout.clusterId instanceof Int32Array && layout.clusterId.length > 0)
    ? Number(layout.clusterId[layout.clusterId.length - 1])
    : -1;
  const firstIndex = (layout.loopIndices instanceof Uint32Array && layout.loopIndices.length > 0)
    ? Number(layout.loopIndices[0])
    : -1;
  const lastIndex = (layout.loopIndices instanceof Uint32Array && layout.loopIndices.length > 0)
    ? Number(layout.loopIndices[layout.loopIndices.length - 1])
    : -1;
  return `${membraneCount}|${indexCount}|${nodeCount}|${firstCluster}|${lastCluster}|${firstIndex}|${lastIndex}|${dt}`;
}

async function ensureSoftMembranePressureProbeState(offload, prep) {
  const device = offload.device;
  const state = offload.state;
  const membraneCount = Math.max(1, prep.plan.membraneCount);
  const nodeCount = Math.max(1, prep.plan.nodeCount);
  const indexCount = Math.max(1, prep.plan.indexCount);

  if (!state.areaProbePipeline) {
    const module = device.createShaderModule({ code: softMembranePressureAreaProbeWgsl });
    state.areaProbePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.areaProbeBindGroup = null;
  }

  if (!state.velocityProposalPipeline) {
    const module = device.createShaderModule({ code: softMembranePressureVelocityProposalWgsl });
    state.velocityProposalPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.velocityProposalBindGroup = null;
  }

  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;

  if (!state.areaProbeParams) {
    state.areaProbeParams = device.createBuffer({
      size: 16,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.areaProbeBindGroup = null;
  }

  if (!state.velocityProposalParams) {
    state.velocityProposalParams = device.createBuffer({
      size: 16,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.velocityProposalBindGroup = null;
  }

  if ((state.areaProbeMembraneCapacity || 0) < membraneCount) {
    const bytes = membraneCount * 4;
    state.areaProbeOffsets?.destroy?.();
    state.areaProbeAreaBase?.destroy?.();
    state.areaProbeAreaNow?.destroy?.();
    state.areaProbeAreaErr?.destroy?.();
    state.areaProbeAreaNowReadback?.destroy?.();
    state.areaProbeAreaErrReadback?.destroy?.();
    state.velocityProposalPressureGain?.destroy?.();

    state.areaProbeOffsets = device.createBuffer({ size: (membraneCount + 1) * 4, usage: storageUsage });
    state.areaProbeAreaBase = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeAreaNow = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaProbeAreaErr = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaProbeAreaNowReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaProbeAreaErrReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.velocityProposalPressureGain = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeMembraneCapacity = membraneCount;
    state.areaProbeBindGroup = null;
    state.velocityProposalBindGroup = null;
  }

  if ((state.areaProbeNodeCapacity || 0) < nodeCount) {
    const bytes = nodeCount * 4;
    state.areaProbeNodeX?.destroy?.();
    state.areaProbeNodeY?.destroy?.();
    state.velocityProposalNodeMass?.destroy?.();
    state.areaProbeNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityProposalNodeMass = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeCapacity = nodeCount;
    state.areaProbeBindGroup = null;
    state.velocityProposalBindGroup = null;
  }

  if ((state.areaProbeIndexCapacity || 0) < indexCount) {
    const bytes = indexCount * 4;
    state.areaProbeLoopIndices?.destroy?.();
    state.velocityProposalLoopMembraneIndex?.destroy?.();
    state.velocityProposalDeltaVx?.destroy?.();
    state.velocityProposalDeltaVy?.destroy?.();
    state.velocityProposalDeltaVxReadback?.destroy?.();
    state.velocityProposalDeltaVyReadback?.destroy?.();
    state.areaProbeLoopIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityProposalLoopMembraneIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityProposalDeltaVx = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.velocityProposalDeltaVy = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.velocityProposalDeltaVxReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.velocityProposalDeltaVyReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaProbeIndexCapacity = indexCount;
    state.areaProbeBindGroup = null;
    state.velocityProposalBindGroup = null;
  }

  if (!state.areaProbeBindGroup) {
    state.areaProbeBindGroup = device.createBindGroup({
      layout: state.areaProbePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.areaProbeParams } },
        { binding: 1, resource: { buffer: state.areaProbeOffsets } },
        { binding: 2, resource: { buffer: state.areaProbeLoopIndices } },
        { binding: 3, resource: { buffer: state.areaProbeNodeX } },
        { binding: 4, resource: { buffer: state.areaProbeNodeY } },
        { binding: 5, resource: { buffer: state.areaProbeAreaBase } },
        { binding: 6, resource: { buffer: state.areaProbeAreaNow } },
        { binding: 7, resource: { buffer: state.areaProbeAreaErr } },
      ],
    });
  }

  if (!state.velocityProposalBindGroup) {
    state.velocityProposalBindGroup = device.createBindGroup({
      layout: state.velocityProposalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.velocityProposalParams } },
        { binding: 1, resource: { buffer: state.areaProbeOffsets } },
        { binding: 2, resource: { buffer: state.areaProbeLoopIndices } },
        { binding: 3, resource: { buffer: state.velocityProposalLoopMembraneIndex } },
        { binding: 4, resource: { buffer: state.areaProbeNodeX } },
        { binding: 5, resource: { buffer: state.areaProbeNodeY } },
        { binding: 6, resource: { buffer: state.velocityProposalNodeMass } },
        { binding: 7, resource: { buffer: state.areaProbeAreaErr } },
        { binding: 8, resource: { buffer: state.velocityProposalPressureGain } },
        { binding: 9, resource: { buffer: state.velocityProposalDeltaVx } },
        { binding: 10, resource: { buffer: state.velocityProposalDeltaVy } },
      ],
    });
  }

  return state;
}

async function dispatchSoftMembranePressureAreaProbe(offload, prep) {
  const device = offload.device;
  const state = await ensureSoftMembranePressureProbeState(offload, prep);
  const membraneCount = prep.plan.membraneCount;
  if (membraneCount <= 0) return false;

  const paramsBytes = new ArrayBuffer(16);
  new Uint32Array(paramsBytes)[0] = membraneCount >>> 0;

  device.queue.writeBuffer(state.areaProbeParams, 0, paramsBytes);
  device.queue.writeBuffer(state.areaProbeOffsets, 0, prep.layout.membraneOffsets);
  device.queue.writeBuffer(state.areaProbeAreaBase, 0, prep.layout.areaBase);
  device.queue.writeBuffer(state.velocityProposalPressureGain, 0, prep.layout.pressureGain);
  if (prep.layout.loopIndices.length > 0) {
    device.queue.writeBuffer(state.areaProbeLoopIndices, 0, prep.layout.loopIndices);
    device.queue.writeBuffer(state.velocityProposalLoopMembraneIndex, 0, prep.layout.loopMembraneIndex);
  }
  if (prep.layout.nodeX.length > 0) {
    device.queue.writeBuffer(state.areaProbeNodeX, 0, prep.layout.nodeX);
    device.queue.writeBuffer(state.areaProbeNodeY, 0, prep.layout.nodeY);
    device.queue.writeBuffer(state.velocityProposalNodeMass, 0, prep.layout.nodeMass);
  }

  const bytes = membraneCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.areaProbePipeline);
  pass.setBindGroup(0, state.areaProbeBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(membraneCount / WGSL_WORKGROUP_SIZE)));
  pass.end();

  encoder.copyBufferToBuffer(state.areaProbeAreaNow, 0, state.areaProbeAreaNowReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.areaProbeAreaErr, 0, state.areaProbeAreaErrReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.areaProbeAreaNowReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const nowMapped = state.areaProbeAreaNowReadback.getMappedRange(0, bytes);
  const areaNow = new Float32Array(nowMapped.slice(0));
  state.areaProbeAreaNowReadback.unmap();

  await state.areaProbeAreaErrReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const errMapped = state.areaProbeAreaErrReadback.getMappedRange(0, bytes);
  const areaErr = new Float32Array(errMapped.slice(0));
  state.areaProbeAreaErrReadback.unmap();

  prep.areaErr = areaErr;
  state.lastAreaProbeDispatchCount = Math.max(1, Math.ceil(membraneCount / WGSL_WORKGROUP_SIZE));
  state.lastAreaProbeAreaNow = areaNow;
  state.lastAreaProbeAreaErr = areaErr;
  state.lastAreaProbeMembraneCount = membraneCount;
  return true;
}

async function dispatchSoftMembranePressureVelocityDeltaProposal(offload, prep, dtPos, proposalSignature) {
  const device = offload.device;
  const state = await ensureSoftMembranePressureProbeState(offload, prep);
  const contributionCount = prep.plan.indexCount;
  if (contributionCount <= 0 || !(prep.areaErr instanceof Float32Array) || prep.areaErr.length <= 0) return false;

  const paramsBytes = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  paramsU32[0] = contributionCount >>> 0;
  paramsF32[1] = Number.isFinite(dtPos) ? dtPos : 0;

  device.queue.writeBuffer(state.velocityProposalParams, 0, paramsBytes);

  const bytes = contributionCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.velocityProposalPipeline);
  pass.setBindGroup(0, state.velocityProposalBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(contributionCount / WGSL_WORKGROUP_SIZE)));
  pass.end();

  encoder.copyBufferToBuffer(state.velocityProposalDeltaVx, 0, state.velocityProposalDeltaVxReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.velocityProposalDeltaVy, 0, state.velocityProposalDeltaVyReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.velocityProposalDeltaVxReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const deltaVxMapped = state.velocityProposalDeltaVxReadback.getMappedRange(0, bytes);
  const deltaVx = new Float32Array(deltaVxMapped.slice(0));
  state.velocityProposalDeltaVxReadback.unmap();

  await state.velocityProposalDeltaVyReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const deltaVyMapped = state.velocityProposalDeltaVyReadback.getMappedRange(0, bytes);
  const deltaVy = new Float32Array(deltaVyMapped.slice(0));
  state.velocityProposalDeltaVyReadback.unmap();

  const cpuExpected = buildCpuMembranePressureVelocityDeltaProposal(prep, dtPos);
  state.lastVelocityProposalDispatchCount = Math.max(1, Math.ceil(contributionCount / WGSL_WORKGROUP_SIZE));
  state.lastVelocityProposalContributionCount = contributionCount;
  state.lastVelocityProposalDeltaVx = deltaVx;
  state.lastVelocityProposalDeltaVy = deltaVy;
  state.lastVelocityProposalExpectedDeltaVx = cpuExpected.deltaVx;
  state.lastVelocityProposalExpectedDeltaVy = cpuExpected.deltaVy;
  state.lastVelocityProposalParity = {
    vx: computeProposalParityStats(cpuExpected.deltaVx, deltaVx),
    vy: computeProposalParityStats(cpuExpected.deltaVy, deltaVy),
    source: 'wgsl-pressure-velocity-proposal',
  };
  state.lastVelocityProposalSource = 'wgsl-pressure-velocity-proposal';
  state.lastVelocityProposalSignature = String(proposalSignature || '');
  return true;
}

export function applySoftMembraneCellPressureGpuOnly({
  sim,
  soft,
  loops,
  dtPos,
  signedAreaCurrent,
  clamp,
  membraneCellBasePressureGain = 0.08,
  membraneCellBaseRadialDamping = 0.06,
  wgslOffload,
}) {
  const membranes = sim?.bodies?.softMembraneClusters;
  if (!Array.isArray(membranes) || membranes.length === 0) return 0;
  if (!(sim.softMembraneAreaBaseline instanceof Map)) sim.softMembraneAreaBaseline = new Map();

  const loopByCluster = new Map();
  for (const loop of loops || []) {
    loopByCluster.set(loop.clusterId ?? 0, loop);
  }

  let authoritativeProposal = null;
  let authoritativeMembraneIndexByCluster = null;
  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const prep = buildSoftMembranePressureWgslPrep({
      membranes,
      loopByCluster,
      softNodes: soft?.nodes,
      softMembraneAreaBaseline: sim.softMembraneAreaBaseline,
      membraneCellBasePressureGain,
    });
    const proposalSignature = computeMembraneProposalSignature(prep, dtPos);
    wgslOffload.state.preparedPlan = prep.plan;
    wgslOffload.state.preparedLayout = prep.layout;
    wgslOffload.state.lastPreparedMembraneCount = prep.plan.membraneCount;
    wgslOffload.state.lastPreparedLoopIndexCount = prep.plan.indexCount;
    wgslOffload.state.lastPreparedLayoutBytes = prep.layout.byteLength;
    wgslOffload.state.lastPreparedProposalSignature = proposalSignature;
    wgslOffload.state.lastMode = 'cpu-prepared';
    wgslOffload.state.lastVelocityProposalSource = 'cpu-membrane-authoritative';

    const canUseAuthoritativeWgsl = prep.plan.indexCount > 0
      && wgslOffload.state.lastVelocityProposalSignature === proposalSignature
      && wgslOffload.state.lastVelocityProposalDeltaVx instanceof Float32Array
      && wgslOffload.state.lastVelocityProposalDeltaVy instanceof Float32Array
      && wgslOffload.state.lastVelocityProposalDeltaVx.length === prep.plan.indexCount
      && wgslOffload.state.lastVelocityProposalDeltaVy.length === prep.plan.indexCount;

    if (canUseAuthoritativeWgsl) {
      authoritativeProposal = {
        prep,
        deltaVx: wgslOffload.state.lastVelocityProposalDeltaVx,
        deltaVy: wgslOffload.state.lastVelocityProposalDeltaVy,
      };
      authoritativeMembraneIndexByCluster = new Map();
      for (let i = 0; i < prep.layout.clusterId.length; i++) {
        authoritativeMembraneIndexByCluster.set(Number(prep.layout.clusterId[i]), i);
      }
      wgslOffload.state.lastMode = 'wgsl-pressure-authoritative';
      wgslOffload.state.lastVelocityProposalSource = 'wgsl-pressure-authoritative';
      wgslOffload.state.lastAuthoritativeProposalSignature = proposalSignature;
      wgslOffload.state.lastAuthoritativeProposalFrame = Number(sim?.frame) || 0;
    }

    if (canUseWgslOffload(wgslOffload) && prep.plan.membraneCount > 0) {
      const serializedDispatch = (wgslOffload.state.pendingWgslAreaProbePromise || Promise.resolve())
        .catch(() => {})
        .then(async () => {
          const probeRan = await dispatchSoftMembranePressureAreaProbe(wgslOffload, prep);
          const proposalRan = probeRan
            ? await dispatchSoftMembranePressureVelocityDeltaProposal(wgslOffload, prep, dtPos, proposalSignature)
            : false;
          return { probeRan, proposalRan };
        })
        .then(({ probeRan, proposalRan }) => {
          wgslOffload.state.lastError = null;
          wgslOffload.state.lastMode = proposalRan
            ? 'wgsl-velocity-proposal'
            : probeRan
              ? 'wgsl-area-probe'
              : 'cpu-membrane-authoritative';
          if (!proposalRan && !authoritativeProposal) {
            wgslOffload.state.lastVelocityProposalSource = 'cpu-membrane-authoritative';
          }
        })
        .catch((err) => {
          wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
          if (!authoritativeProposal) {
            wgslOffload.state.lastMode = 'cpu-membrane-authoritative';
            wgslOffload.state.lastVelocityProposalSource = 'cpu-membrane-authoritative';
          }
        });
      wgslOffload.state.pendingWgslAreaProbePromise = serializedDispatch;
    }
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

      // Outward normal for CCW loop via right-normal accumulation.
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

      let proposalVx = 0;
      let proposalVy = 0;
      let hasAuthoritativeProposal = false;
      if (authoritativeProposal && authoritativeMembraneIndexByCluster) {
        const proposalMembraneIndex = authoritativeMembraneIndexByCluster.get(cid);
        if (Number.isInteger(proposalMembraneIndex)) {
          const start = authoritativeProposal.prep.layout.membraneOffsets[proposalMembraneIndex] >>> 0;
          const li = start + k;
          if (li >= 0 && li < authoritativeProposal.deltaVx.length) {
            proposalVx = authoritativeProposal.deltaVx[li] || 0;
            proposalVy = authoritativeProposal.deltaVy[li] || 0;
            hasAuthoritativeProposal = true;
          }
        }
      }

      if (hasAuthoritativeProposal) {
        curr.vx += proposalVx;
        curr.vy += proposalVy;
      } else {
        const invMass = 1 / Math.max(0.02, curr.mass || 1);
        const impulse = err * gain * dtPos * invMass;
        curr.vx += nx * impulse;
        curr.vy += ny * impulse;
      }

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
