/**
 * GPU-only runtime soft spring XPBD pass.
 * Keeps spring-constraint stepping isolated for the gpu-only runtime path while
 * preserving baseline/default behavior and call contracts.
 */
const WGSL_WORKGROUP_SIZE = 64;

const softSpringStretchProbeWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  springCount: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> springNodeA: array<u32>;
@group(0) @binding(4) var<storage, read> springNodeB: array<u32>;
@group(0) @binding(5) var<storage, read> springRest: array<f32>;
@group(0) @binding(6) var<storage, read_write> stretchOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let si = gid.x;
  if (si >= params.springCount) { return; }

  let ia = springNodeA[si];
  let ib = springNodeB[si];
  if (ia >= params.nodeCount || ib >= params.nodeCount) {
    stretchOut[si] = 0.0;
    return;
  }

  let dx = nodeX[ib] - nodeX[ia];
  let dy = nodeY[ib] - nodeY[ia];
  let d = max(length(vec2<f32>(dx, dy)), 1e-6);
  let rest = max(abs(springRest[si]), 1e-6);
  stretchOut[si] = (d - rest) / rest;
}
`;

const softSpringLambdaProposalWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  springCount: u32,
  alpha: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> springNodeA: array<u32>;
@group(0) @binding(4) var<storage, read> springNodeB: array<u32>;
@group(0) @binding(5) var<storage, read> springRest: array<f32>;
@group(0) @binding(6) var<storage, read> springInvMassA: array<f32>;
@group(0) @binding(7) var<storage, read> springInvMassB: array<f32>;
@group(0) @binding(8) var<storage, read> lambdaPrev: array<f32>;
@group(0) @binding(9) var<storage, read_write> deltaLambdaOut: array<f32>;
@group(0) @binding(10) var<storage, read_write> lambdaNextOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let si = gid.x;
  if (si >= params.springCount) { return; }

  let ia = springNodeA[si];
  let ib = springNodeB[si];
  if (ia >= params.nodeCount || ib >= params.nodeCount) {
    deltaLambdaOut[si] = 0.0;
    lambdaNextOut[si] = 0.0;
    return;
  }

  let ax = nodeX[ia];
  let ay = nodeY[ia];
  let bx = nodeX[ib];
  let by = nodeY[ib];
  let dx = bx - ax;
  let dy = by - ay;
  let d = max(length(vec2<f32>(dx, dy)), 1e-6);
  let restRaw = springRest[si];
  let strainCap = max(0.05, abs(restRaw) * 0.45);
  let c = clamp(d - restRaw, -strainCap, strainCap);

  let wSum = springInvMassA[si] + springInvMassB[si];
  if (wSum <= 1e-9) {
    deltaLambdaOut[si] = 0.0;
    lambdaNextOut[si] = lambdaPrev[si];
    return;
  }

  let lambdaPrevValue = lambdaPrev[si];
  let dlRaw = (-c - params.alpha * lambdaPrevValue) / (wSum + params.alpha);
  let lambdaNext = clamp(lambdaPrevValue + dlRaw, -20.0, 20.0);
  let dl = lambdaNext - lambdaPrevValue;

  deltaLambdaOut[si] = dl;
  lambdaNextOut[si] = lambdaNext;
}
`;

const softSpringVelocityDeltaProposalWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  springCount: u32,
  endpointCount: u32,
  dtPos: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodePredX: array<f32>;
@group(0) @binding(2) var<storage, read> nodePredY: array<f32>;
@group(0) @binding(3) var<storage, read> springNodeA: array<u32>;
@group(0) @binding(4) var<storage, read> springNodeB: array<u32>;
@group(0) @binding(5) var<storage, read> springInvMassA: array<f32>;
@group(0) @binding(6) var<storage, read> springInvMassB: array<f32>;
@group(0) @binding(7) var<storage, read> endpointNodeIndices: array<u32>;
@group(0) @binding(8) var<storage, read> endpointSpringIndices: array<u32>;
@group(0) @binding(9) var<storage, read> endpointSigns: array<i32>;
@group(0) @binding(10) var<storage, read> deltaLambdaByColor: array<f32>;
@group(0) @binding(11) var<storage, read_write> endpointDeltaVXOut: array<f32>;
@group(0) @binding(12) var<storage, read_write> endpointDeltaVYOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ei = gid.x;
  if (ei >= params.endpointCount) { return; }

  let ni = endpointNodeIndices[ei];
  let si = endpointSpringIndices[ei];
  if (ni >= params.nodeCount || si >= params.springCount || params.dtPos <= 1e-8) {
    endpointDeltaVXOut[ei] = 0.0;
    endpointDeltaVYOut[ei] = 0.0;
    return;
  }

  let ia = springNodeA[si];
  let ib = springNodeB[si];
  if (ia >= params.nodeCount || ib >= params.nodeCount) {
    endpointDeltaVXOut[ei] = 0.0;
    endpointDeltaVYOut[ei] = 0.0;
    return;
  }

  let dx = nodePredX[ib] - nodePredX[ia];
  let dy = nodePredY[ib] - nodePredY[ia];
  let d = max(length(vec2<f32>(dx, dy)), 1e-6);
  let nx = dx / d;
  let ny = dy / d;

  let sign = select(1.0, -1.0, endpointSigns[ei] < 0);
  let w = select(springInvMassB[si], springInvMassA[si], endpointSigns[ei] < 0);
  let dl = deltaLambdaByColor[si];
  let scale = (sign * w * dl) / params.dtPos;

  endpointDeltaVXOut[ei] = nx * scale;
  endpointDeltaVYOut[ei] = ny * scale;
}
`;

const softSpringVelocityNodeReductionWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  endpointCount: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> endpointNodeIndices: array<u32>;
@group(0) @binding(2) var<storage, read> endpointDeltaVX: array<f32>;
@group(0) @binding(3) var<storage, read> endpointDeltaVY: array<f32>;
@group(0) @binding(4) var<storage, read_write> nodeDeltaVXOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> nodeDeltaVYOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> nodeContributionCountOut: array<u32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ni = gid.x;
  if (ni >= params.nodeCount) { return; }

  var sumX = 0.0;
  var sumY = 0.0;
  var count = 0u;
  for (var ei = 0u; ei < params.endpointCount; ei = ei + 1u) {
    if (endpointNodeIndices[ei] == ni) {
      sumX = sumX + endpointDeltaVX[ei];
      sumY = sumY + endpointDeltaVY[ei];
      count = count + 1u;
    }
  }

  nodeDeltaVXOut[ni] = sumX;
  nodeDeltaVYOut[ni] = sumY;
  nodeContributionCountOut[ni] = count;
}
`;

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined') return false;
  if (typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function ensureProbeBuffers(offload, nodeCount, springCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.probeNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.probeNodeCapacity ? state.probeNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.probeNodeX?.destroy?.();
    state.probeNodeY?.destroy?.();
    state.probeNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeNodeCapacity = capacity;
    state.probeBindGroup = null;
  }

  const requiredSpringCapacity = Math.max(1, springCount);
  if ((state.probeSpringCapacity || 0) < requiredSpringCapacity) {
    const capacity = Math.max(requiredSpringCapacity, state.probeSpringCapacity ? state.probeSpringCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.probeSpringNodeA?.destroy?.();
    state.probeSpringNodeB?.destroy?.();
    state.probeSpringRest?.destroy?.();
    state.probeStretchOut?.destroy?.();
    state.probeStretchReadback?.destroy?.();
    state.probeSpringNodeA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeSpringNodeB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeSpringRest = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeStretchOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.probeStretchReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.probeSpringCapacity = capacity;
    state.probeBindGroup = null;
  }

  if (!state.probeParams) {
    state.probeParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.probeBindGroup = null;
  }

  return state;
}

function ensureLambdaProposalBuffers(offload, nodeCount, springCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.lambdaNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.lambdaNodeCapacity ? state.lambdaNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.lambdaNodePredX?.destroy?.();
    state.lambdaNodePredY?.destroy?.();
    state.lambdaNodePredX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.lambdaNodePredY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.lambdaNodeCapacity = capacity;
    state.lambdaBindGroup = null;
  }

  const requiredSpringCapacity = Math.max(1, springCount);
  if ((state.lambdaSpringCapacity || 0) < requiredSpringCapacity) {
    const capacity = Math.max(requiredSpringCapacity, state.lambdaSpringCapacity ? state.lambdaSpringCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.lambdaSpringNodeA?.destroy?.();
    state.lambdaSpringNodeB?.destroy?.();
    state.lambdaSpringRest?.destroy?.();
    state.lambdaSpringInvMassA?.destroy?.();
    state.lambdaSpringInvMassB?.destroy?.();
    state.lambdaPrev?.destroy?.();
    state.deltaLambdaOut?.destroy?.();
    state.lambdaNextOut?.destroy?.();
    state.deltaLambdaReadback?.destroy?.();
    state.lambdaNextReadback?.destroy?.();
    state.lambdaSpringNodeA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.lambdaSpringNodeB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.lambdaSpringRest = device.createBuffer({ size: bytes, usage: storageUsage });
    state.lambdaSpringInvMassA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.lambdaSpringInvMassB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.lambdaPrev = device.createBuffer({ size: bytes, usage: storageUsage });
    state.deltaLambdaOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.lambdaNextOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.deltaLambdaReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.lambdaNextReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.lambdaSpringCapacity = capacity;
    state.lambdaBindGroup = null;
  }

  if (!state.lambdaParams) {
    state.lambdaParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.lambdaBindGroup = null;
  }

  return state;
}


function ensureVelocityDeltaProposalBuffers(offload, nodeCount, springCount, endpointCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.velocityNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.velocityNodeCapacity ? state.velocityNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.velocityNodePredX?.destroy?.();
    state.velocityNodePredY?.destroy?.();
    state.velocityNodeDeltaVXOut?.destroy?.();
    state.velocityNodeDeltaVYOut?.destroy?.();
    state.velocityNodeContributionCountOut?.destroy?.();
    state.velocityNodeDeltaVXReadback?.destroy?.();
    state.velocityNodeDeltaVYReadback?.destroy?.();
    state.velocityNodeContributionCountReadback?.destroy?.();
    state.velocityNodePredX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityNodePredY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityNodeDeltaVXOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.velocityNodeDeltaVYOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.velocityNodeContributionCountOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.velocityNodeDeltaVXReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.velocityNodeDeltaVYReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.velocityNodeContributionCountReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.velocityNodeCapacity = capacity;
    state.velocityBindGroup = null;
    state.velocityReductionBindGroup = null;
  }

  const requiredSpringCapacity = Math.max(1, springCount);
  if ((state.velocitySpringCapacity || 0) < requiredSpringCapacity) {
    const capacity = Math.max(requiredSpringCapacity, state.velocitySpringCapacity ? state.velocitySpringCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.velocitySpringNodeA?.destroy?.();
    state.velocitySpringNodeB?.destroy?.();
    state.velocitySpringInvMassA?.destroy?.();
    state.velocitySpringInvMassB?.destroy?.();
    state.velocityDeltaLambdaByColor?.destroy?.();
    state.velocitySpringNodeA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocitySpringNodeB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocitySpringInvMassA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocitySpringInvMassB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityDeltaLambdaByColor = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocitySpringCapacity = capacity;
    state.velocityBindGroup = null;
  }

  const requiredEndpointCapacity = Math.max(1, endpointCount);
  if ((state.velocityEndpointCapacity || 0) < requiredEndpointCapacity) {
    const capacity = Math.max(requiredEndpointCapacity, state.velocityEndpointCapacity ? state.velocityEndpointCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.velocityEndpointNodeIndices?.destroy?.();
    state.velocityEndpointSpringIndices?.destroy?.();
    state.velocityEndpointSigns?.destroy?.();
    state.velocityEndpointDeltaVXOut?.destroy?.();
    state.velocityEndpointDeltaVYOut?.destroy?.();
    state.velocityEndpointDeltaVXReadback?.destroy?.();
    state.velocityEndpointDeltaVYReadback?.destroy?.();
    state.velocityEndpointNodeIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityEndpointSpringIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityEndpointSigns = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityEndpointDeltaVXOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.velocityEndpointDeltaVYOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.velocityEndpointDeltaVXReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.velocityEndpointDeltaVYReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.velocityEndpointCapacity = capacity;
    state.velocityBindGroup = null;
  }

  if (!state.velocityParams) {
    state.velocityParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.velocityBindGroup = null;
  }
  if (!state.velocityReductionParams) {
    state.velocityReductionParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.velocityReductionBindGroup = null;
  }

  return state;
}

async function dispatchSoftSpringWgslVelocityDeltaProposal({
  soft,
  offload,
  layout,
  dtPos,
  deltaLambdaByColor,
  includeEndpointTelemetry = true,
}) {
  if (!canUseWgslOffload(offload)) return null;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springCount = Number(layout?.springNodeAByColor?.length) || 0;
  const endpointCount = Number(layout?.endpointNodeIndicesByColor?.length) || 0;
  if (nodes.length <= 0 || springCount <= 0 || endpointCount <= 0) return null;

  const state = ensureVelocityDeltaProposalBuffers(offload, nodes.length, springCount, endpointCount);
  const device = offload.device;

  if (!state.velocityPipeline) {
    const module = device.createShaderModule({ code: softSpringVelocityDeltaProposalWgsl });
    state.velocityPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.velocityBindGroup = null;
  }

  if (!state.velocityBindGroup) {
    state.velocityBindGroup = device.createBindGroup({
      layout: state.velocityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.velocityParams } },
        { binding: 1, resource: { buffer: state.velocityNodePredX } },
        { binding: 2, resource: { buffer: state.velocityNodePredY } },
        { binding: 3, resource: { buffer: state.velocitySpringNodeA } },
        { binding: 4, resource: { buffer: state.velocitySpringNodeB } },
        { binding: 5, resource: { buffer: state.velocitySpringInvMassA } },
        { binding: 6, resource: { buffer: state.velocitySpringInvMassB } },
        { binding: 7, resource: { buffer: state.velocityEndpointNodeIndices } },
        { binding: 8, resource: { buffer: state.velocityEndpointSpringIndices } },
        { binding: 9, resource: { buffer: state.velocityEndpointSigns } },
        { binding: 10, resource: { buffer: state.velocityDeltaLambdaByColor } },
        { binding: 11, resource: { buffer: state.velocityEndpointDeltaVXOut } },
        { binding: 12, resource: { buffer: state.velocityEndpointDeltaVYOut } },
      ],
    });
  }

  const safeDtPos = Math.max(1e-8, Number(dtPos) || 0);
  const nodePredX = new Float32Array(nodes.length);
  const nodePredY = new Float32Array(nodes.length);
  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni] || {};
    nodePredX[ni] = (Number(node.x) || 0) + (Number(node.vx) || 0) * safeDtPos;
    nodePredY[ni] = (Number(node.y) || 0) + (Number(node.vy) || 0) * safeDtPos;
  }

  const paramsBytes = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  paramsU32[0] = nodes.length >>> 0;
  paramsU32[1] = springCount >>> 0;
  paramsU32[2] = endpointCount >>> 0;
  paramsF32[3] = safeDtPos;

  device.queue.writeBuffer(state.velocityParams, 0, paramsBytes);
  device.queue.writeBuffer(state.velocityNodePredX, 0, nodePredX);
  device.queue.writeBuffer(state.velocityNodePredY, 0, nodePredY);
  device.queue.writeBuffer(state.velocitySpringNodeA, 0, layout.springNodeAByColor);
  device.queue.writeBuffer(state.velocitySpringNodeB, 0, layout.springNodeBByColor);
  device.queue.writeBuffer(state.velocitySpringInvMassA, 0, layout.springInvMassAByColor);
  device.queue.writeBuffer(state.velocitySpringInvMassB, 0, layout.springInvMassBByColor);
  device.queue.writeBuffer(state.velocityEndpointNodeIndices, 0, layout.endpointNodeIndicesByColor);
  device.queue.writeBuffer(state.velocityEndpointSpringIndices, 0, layout.endpointSpringIndicesByColor);
  device.queue.writeBuffer(state.velocityEndpointSigns, 0, layout.endpointSignsI32ByColor);
  device.queue.writeBuffer(state.velocityDeltaLambdaByColor, 0, deltaLambdaByColor);

  if (!state.velocityReductionPipeline) {
    const module = device.createShaderModule({ code: softSpringVelocityNodeReductionWgsl });
    state.velocityReductionPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.velocityReductionBindGroup = null;
  }

  if (!state.velocityReductionBindGroup) {
    state.velocityReductionBindGroup = device.createBindGroup({
      layout: state.velocityReductionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.velocityReductionParams } },
        { binding: 1, resource: { buffer: state.velocityEndpointNodeIndices } },
        { binding: 2, resource: { buffer: state.velocityEndpointDeltaVXOut } },
        { binding: 3, resource: { buffer: state.velocityEndpointDeltaVYOut } },
        { binding: 4, resource: { buffer: state.velocityNodeDeltaVXOut } },
        { binding: 5, resource: { buffer: state.velocityNodeDeltaVYOut } },
        { binding: 6, resource: { buffer: state.velocityNodeContributionCountOut } },
      ],
    });
  }

  const endpointBytes = endpointCount * 4;
  const endpointDispatchCount = Math.ceil(endpointCount / WGSL_WORKGROUP_SIZE);
  const nodeDispatchCount = Math.ceil(nodes.length / WGSL_WORKGROUP_SIZE);
  const reductionParams = new Uint32Array(4);
  reductionParams[0] = nodes.length >>> 0;
  reductionParams[1] = endpointCount >>> 0;
  device.queue.writeBuffer(state.velocityReductionParams, 0, reductionParams);

  const encoder = device.createCommandEncoder();
  const endpointPass = encoder.beginComputePass();
  endpointPass.setPipeline(state.velocityPipeline);
  endpointPass.setBindGroup(0, state.velocityBindGroup);
  endpointPass.dispatchWorkgroups(endpointDispatchCount);
  endpointPass.end();

  const nodeReductionPass = encoder.beginComputePass();
  nodeReductionPass.setPipeline(state.velocityReductionPipeline);
  nodeReductionPass.setBindGroup(0, state.velocityReductionBindGroup);
  nodeReductionPass.dispatchWorkgroups(nodeDispatchCount);
  nodeReductionPass.end();

  if (includeEndpointTelemetry) {
    encoder.copyBufferToBuffer(state.velocityEndpointDeltaVXOut, 0, state.velocityEndpointDeltaVXReadback, 0, endpointBytes);
    encoder.copyBufferToBuffer(state.velocityEndpointDeltaVYOut, 0, state.velocityEndpointDeltaVYReadback, 0, endpointBytes);
  }

  const nodeBytes = nodes.length * 4;
  encoder.copyBufferToBuffer(state.velocityNodeDeltaVXOut, 0, state.velocityNodeDeltaVXReadback, 0, nodeBytes);
  encoder.copyBufferToBuffer(state.velocityNodeDeltaVYOut, 0, state.velocityNodeDeltaVYReadback, 0, nodeBytes);
  encoder.copyBufferToBuffer(state.velocityNodeContributionCountOut, 0, state.velocityNodeContributionCountReadback, 0, nodeBytes);
  device.queue.submit([encoder.finish()]);

  let endpointDeltaVX = null;
  let endpointDeltaVY = null;
  if (includeEndpointTelemetry) {
    await state.velocityEndpointDeltaVXReadback.mapAsync(globalThis.GPUMapMode.READ, 0, endpointBytes);
    const mappedVX = state.velocityEndpointDeltaVXReadback.getMappedRange(0, endpointBytes);
    endpointDeltaVX = new Float32Array(mappedVX.slice(0));
    state.velocityEndpointDeltaVXReadback.unmap();

    await state.velocityEndpointDeltaVYReadback.mapAsync(globalThis.GPUMapMode.READ, 0, endpointBytes);
    const mappedVY = state.velocityEndpointDeltaVYReadback.getMappedRange(0, endpointBytes);
    endpointDeltaVY = new Float32Array(mappedVY.slice(0));
    state.velocityEndpointDeltaVYReadback.unmap();
  }

  await state.velocityNodeDeltaVXReadback.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes);
  const mappedNodeVX = state.velocityNodeDeltaVXReadback.getMappedRange(0, nodeBytes);
  const nodeDeltaVx = new Float32Array(mappedNodeVX.slice(0));
  state.velocityNodeDeltaVXReadback.unmap();

  await state.velocityNodeDeltaVYReadback.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes);
  const mappedNodeVY = state.velocityNodeDeltaVYReadback.getMappedRange(0, nodeBytes);
  const nodeDeltaVy = new Float32Array(mappedNodeVY.slice(0));
  state.velocityNodeDeltaVYReadback.unmap();

  await state.velocityNodeContributionCountReadback.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes);
  const mappedNodeCounts = state.velocityNodeContributionCountReadback.getMappedRange(0, nodeBytes);
  const nodeContributionCount = new Uint32Array(mappedNodeCounts.slice(0));
  state.velocityNodeContributionCountReadback.unmap();

  return {
    dispatchCount: endpointDispatchCount,
    reductionDispatchCount: nodeDispatchCount,
    endpointDeltaVX,
    endpointDeltaVY,
    nodeDeltaVx,
    nodeDeltaVy,
    nodeContributionCount,
  };
}

async function dispatchSoftSpringWgslProbe({ soft, offload, layout }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springCount = Number(layout?.springRestByColor?.length) || 0;
  if (nodes.length <= 0 || springCount <= 0) return false;

  const state = ensureProbeBuffers(offload, nodes.length, springCount);
  const device = offload.device;

  if (!state.probePipeline) {
    const module = device.createShaderModule({ code: softSpringStretchProbeWgsl });
    state.probePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.probeBindGroup = null;
  }

  if (!state.probeBindGroup) {
    state.probeBindGroup = device.createBindGroup({
      layout: state.probePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.probeParams } },
        { binding: 1, resource: { buffer: state.probeNodeX } },
        { binding: 2, resource: { buffer: state.probeNodeY } },
        { binding: 3, resource: { buffer: state.probeSpringNodeA } },
        { binding: 4, resource: { buffer: state.probeSpringNodeB } },
        { binding: 5, resource: { buffer: state.probeSpringRest } },
        { binding: 6, resource: { buffer: state.probeStretchOut } },
      ],
    });
  }

  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  for (let ni = 0; ni < nodes.length; ni++) {
    nodeX[ni] = Number(nodes[ni]?.x) || 0;
    nodeY[ni] = Number(nodes[ni]?.y) || 0;
  }

  const params = new Uint32Array(4);
  params[0] = nodes.length >>> 0;
  params[1] = springCount >>> 0;

  device.queue.writeBuffer(state.probeParams, 0, params);
  device.queue.writeBuffer(state.probeNodeX, 0, nodeX);
  device.queue.writeBuffer(state.probeNodeY, 0, nodeY);
  device.queue.writeBuffer(state.probeSpringNodeA, 0, layout.springNodeAByColor);
  device.queue.writeBuffer(state.probeSpringNodeB, 0, layout.springNodeBByColor);
  device.queue.writeBuffer(state.probeSpringRest, 0, layout.springRestByColor);

  const dispatchCount = Math.ceil(springCount / WGSL_WORKGROUP_SIZE);
  const bytes = springCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.probePipeline);
  pass.setBindGroup(0, state.probeBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  encoder.copyBufferToBuffer(state.probeStretchOut, 0, state.probeStretchReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.probeStretchReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = state.probeStretchReadback.getMappedRange(0, bytes);
  const stretchByColor = new Float32Array(mapped.slice(0));
  state.probeStretchReadback.unmap();

  let absMax = 0;
  let absSum = 0;
  for (let i = 0; i < stretchByColor.length; i++) {
    const abs = Math.abs(stretchByColor[i]);
    if (abs > absMax) absMax = abs;
    absSum += abs;
  }

  state.lastProbeNodeCount = nodes.length;
  state.lastProbeSpringCount = springCount;
  state.lastProbeDispatch = dispatchCount;
  state.lastProbeAbsMean = stretchByColor.length > 0 ? absSum / stretchByColor.length : 0;
  state.lastProbeAbsMax = absMax;
  state.lastProbeStretchByColor = stretchByColor;
  state.lastProbeColorBucketCount = Math.max(0, (layout?.springColorOffsets?.length || 1) - 1);
  return true;
}

async function dispatchSoftSpringWgslLambdaProposal({ soft, offload, layout, dtPos, alpha, lambdaCache }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springCount = Number(layout?.springRestByColor?.length) || 0;
  if (nodes.length <= 0 || springCount <= 0) return false;

  const state = ensureLambdaProposalBuffers(offload, nodes.length, springCount);
  const device = offload.device;

  if (!state.lambdaPipeline) {
    const module = device.createShaderModule({ code: softSpringLambdaProposalWgsl });
    state.lambdaPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.lambdaBindGroup = null;
  }

  if (!state.lambdaBindGroup) {
    state.lambdaBindGroup = device.createBindGroup({
      layout: state.lambdaPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.lambdaParams } },
        { binding: 1, resource: { buffer: state.lambdaNodePredX } },
        { binding: 2, resource: { buffer: state.lambdaNodePredY } },
        { binding: 3, resource: { buffer: state.lambdaSpringNodeA } },
        { binding: 4, resource: { buffer: state.lambdaSpringNodeB } },
        { binding: 5, resource: { buffer: state.lambdaSpringRest } },
        { binding: 6, resource: { buffer: state.lambdaSpringInvMassA } },
        { binding: 7, resource: { buffer: state.lambdaSpringInvMassB } },
        { binding: 8, resource: { buffer: state.lambdaPrev } },
        { binding: 9, resource: { buffer: state.deltaLambdaOut } },
        { binding: 10, resource: { buffer: state.lambdaNextOut } },
      ],
    });
  }

  const nodePredX = new Float32Array(nodes.length);
  const nodePredY = new Float32Array(nodes.length);
  const invSpringOrder = layout?.springColorOrderedIndices instanceof Uint32Array
    ? layout.springColorOrderedIndices
    : new Uint32Array(0);
  const lambdaPrevByColor = new Float32Array(springCount);

  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni] || {};
    nodePredX[ni] = (Number(node.x) || 0) + (Number(node.vx) || 0) * dtPos;
    nodePredY[ni] = (Number(node.y) || 0) + (Number(node.vy) || 0) * dtPos;
  }

  for (let oi = 0; oi < springCount; oi++) {
    const ai = invSpringOrder[oi];
    lambdaPrevByColor[oi] = Number(lambdaCache?.[ai]) || 0;
  }

  const paramsBytes = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  paramsU32[0] = nodes.length >>> 0;
  paramsU32[1] = springCount >>> 0;
  paramsF32[2] = Number.isFinite(alpha) ? alpha : 0;

  device.queue.writeBuffer(state.lambdaParams, 0, paramsBytes);
  device.queue.writeBuffer(state.lambdaNodePredX, 0, nodePredX);
  device.queue.writeBuffer(state.lambdaNodePredY, 0, nodePredY);
  device.queue.writeBuffer(state.lambdaSpringNodeA, 0, layout.springNodeAByColor);
  device.queue.writeBuffer(state.lambdaSpringNodeB, 0, layout.springNodeBByColor);
  device.queue.writeBuffer(state.lambdaSpringRest, 0, layout.springRestByColor);
  device.queue.writeBuffer(state.lambdaSpringInvMassA, 0, layout.springInvMassAByColor);
  device.queue.writeBuffer(state.lambdaSpringInvMassB, 0, layout.springInvMassBByColor);
  device.queue.writeBuffer(state.lambdaPrev, 0, lambdaPrevByColor);

  const dispatchCount = Math.ceil(springCount / WGSL_WORKGROUP_SIZE);
  const bytes = springCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.lambdaPipeline);
  pass.setBindGroup(0, state.lambdaBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  encoder.copyBufferToBuffer(state.deltaLambdaOut, 0, state.deltaLambdaReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.lambdaNextOut, 0, state.lambdaNextReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.deltaLambdaReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mappedDelta = state.deltaLambdaReadback.getMappedRange(0, bytes);
  const deltaByColor = new Float32Array(mappedDelta.slice(0));
  state.deltaLambdaReadback.unmap();

  await state.lambdaNextReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mappedNext = state.lambdaNextReadback.getMappedRange(0, bytes);
  const lambdaNextByColor = new Float32Array(mappedNext.slice(0));
  state.lambdaNextReadback.unmap();

  const deltaBySpring = new Float32Array(springCount);
  const lambdaNextBySpring = new Float32Array(springCount);
  for (let oi = 0; oi < springCount; oi++) {
    const ai = invSpringOrder[oi];
    if (!Number.isInteger(ai) || ai < 0 || ai >= springCount) continue;
    deltaBySpring[ai] = deltaByColor[oi];
    lambdaNextBySpring[ai] = lambdaNextByColor[oi];
  }

  let maxAbsDelta = 0;
  let sumAbsDelta = 0;
  for (let i = 0; i < deltaByColor.length; i++) {
    const abs = Math.abs(deltaByColor[i]);
    if (abs > maxAbsDelta) maxAbsDelta = abs;
    sumAbsDelta += abs;
  }

  state.lastProposalSpringCount = springCount;
  state.lastProposalDispatch = dispatchCount;
  state.lastProposalAbsDeltaMean = deltaByColor.length > 0 ? sumAbsDelta / deltaByColor.length : 0;
  state.lastProposalAbsDeltaMax = maxAbsDelta;
  state.lastProposalDeltaLambdaByColor = deltaByColor;
  state.lastProposalLambdaNextByColor = lambdaNextByColor;
  state.lastProposalDeltaLambdaBySpring = deltaBySpring;
  state.lastProposalLambdaNextBySpring = lambdaNextBySpring;
  return true;
}

export function buildSoftSpringXpbdWgslPlan({
  soft,
  skipClusterSet = null,
} = {}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springs = Array.isArray(soft?.springs) ? soft.springs : [];

  const activeSpringIndices = [];
  const activeNodeAIndices = [];
  const activeNodeBIndices = [];
  const endpointNodeIndicesRaw = [];
  const endpointSpringIndicesRaw = [];
  const endpointSignsRaw = [];

  for (let si = 0; si < springs.length; si++) {
    const spring = springs[si];
    const i = Number(spring?.[0]);
    const j = Number(spring?.[1]);
    const rest = Number(spring?.[2]);
    if (!Number.isInteger(i) || !Number.isInteger(j)) continue;
    if (i < 0 || i >= nodes.length || j < 0 || j >= nodes.length) continue;
    if (!Number.isFinite(rest)) continue;

    const a = nodes[i];
    const b = nodes[j];
    if (!a || !b) continue;
    if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

    const activeIndex = activeSpringIndices.length;
    activeSpringIndices.push(si);
    activeNodeAIndices.push(i);
    activeNodeBIndices.push(j);

    endpointNodeIndicesRaw.push(i, j);
    endpointSpringIndicesRaw.push(activeIndex, activeIndex);
    endpointSignsRaw.push(-1, 1);
  }

  const endpointCount = endpointNodeIndicesRaw.length;
  const nodeEndpointCounts = new Uint32Array(nodes.length);
  for (let ei = 0; ei < endpointCount; ei++) {
    nodeEndpointCounts[endpointNodeIndicesRaw[ei]] += 1;
  }

  const nodeEndpointOffsets = new Uint32Array(nodes.length + 1);
  for (let ni = 0; ni < nodes.length; ni++) {
    nodeEndpointOffsets[ni + 1] = nodeEndpointOffsets[ni] + nodeEndpointCounts[ni];
  }

  const endpointNodeIndices = new Uint32Array(endpointCount);
  const endpointSpringIndices = new Uint32Array(endpointCount);
  const endpointSigns = new Int8Array(endpointCount);
  const cursor = nodeEndpointOffsets.slice(0, nodes.length);

  for (let ei = 0; ei < endpointCount; ei++) {
    const ni = endpointNodeIndicesRaw[ei];
    const dst = cursor[ni]++;
    endpointNodeIndices[dst] = ni;
    endpointSpringIndices[dst] = endpointSpringIndicesRaw[ei];
    endpointSigns[dst] = endpointSignsRaw[ei];
  }

  // Deterministic conflict-free spring coloring (no shared nodes per color).
  // This is a direct unblocker for WGSL spring solves: each color can dispatch
  // in parallel without atomics while preserving Gauss-Seidel ownership order.
  const nodeUsedColors = Array.from({ length: nodes.length }, () => new Set());
  const springColors = new Uint32Array(activeSpringIndices.length);
  let colorCount = 0;
  for (let ai = 0; ai < activeSpringIndices.length; ai++) {
    const i = activeNodeAIndices[ai];
    const j = activeNodeBIndices[ai];
    let color = 0;
    while (nodeUsedColors[i].has(color) || nodeUsedColors[j].has(color)) color++;
    nodeUsedColors[i].add(color);
    nodeUsedColors[j].add(color);
    springColors[ai] = color;
    if (color + 1 > colorCount) colorCount = color + 1;
  }

  const springColorCounts = new Uint32Array(colorCount);
  for (let ai = 0; ai < springColors.length; ai++) springColorCounts[springColors[ai]] += 1;
  const springColorOffsets = new Uint32Array(colorCount + 1);
  for (let ci = 0; ci < colorCount; ci++) springColorOffsets[ci + 1] = springColorOffsets[ci] + springColorCounts[ci];
  const springColorOrderedIndices = new Uint32Array(activeSpringIndices.length);
  const springColorCursor = springColorOffsets.slice(0, colorCount);
  for (let ai = 0; ai < springColors.length; ai++) {
    const ci = springColors[ai];
    const dst = springColorCursor[ci]++;
    springColorOrderedIndices[dst] = ai;
  }

  return {
    nodeCount: nodes.length,
    springCount: springs.length,
    activeSpringCount: activeSpringIndices.length,
    endpointCount,
    activeSpringIndices: Uint32Array.from(activeSpringIndices),
    endpointNodeIndices,
    endpointSpringIndices,
    endpointSigns,
    nodeEndpointOffsets,
    springColors,
    springColorOffsets,
    springColorOrderedIndices,
  };
}

export function buildSoftSpringXpbdWgslLayout({
  soft,
  plan,
} = {}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const activeSpringCount = Number(plan?.activeSpringCount) || 0;
  const activeSpringIndices = plan?.activeSpringIndices instanceof Uint32Array
    ? plan.activeSpringIndices
    : new Uint32Array(0);
  const springColorOrderedIndices = plan?.springColorOrderedIndices instanceof Uint32Array
    ? plan.springColorOrderedIndices
    : new Uint32Array(0);

  const springNodeA = new Uint32Array(activeSpringCount);
  const springNodeB = new Uint32Array(activeSpringCount);
  const springRest = new Float32Array(activeSpringCount);
  const springInvMassA = new Float32Array(activeSpringCount);
  const springInvMassB = new Float32Array(activeSpringCount);

  for (let ai = 0; ai < activeSpringCount; ai++) {
    const si = activeSpringIndices[ai];
    const spring = soft?.springs?.[si];
    const i = Number(spring?.[0]);
    const j = Number(spring?.[1]);
    const rest = Number(spring?.[2]);
    const a = nodes[i];
    const b = nodes[j];

    springNodeA[ai] = Number.isInteger(i) && i >= 0 ? i : 0;
    springNodeB[ai] = Number.isInteger(j) && j >= 0 ? j : 0;
    springRest[ai] = Number.isFinite(rest) ? rest : 0;
    springInvMassA[ai] = 1 / Math.max(0.02, Number(a?.mass) || 1);
    springInvMassB[ai] = 1 / Math.max(0.02, Number(b?.mass) || 1);
  }

  const springNodeAByColor = new Uint32Array(springColorOrderedIndices.length);
  const springNodeBByColor = new Uint32Array(springColorOrderedIndices.length);
  const springRestByColor = new Float32Array(springColorOrderedIndices.length);
  const springInvMassAByColor = new Float32Array(springColorOrderedIndices.length);
  const springInvMassBByColor = new Float32Array(springColorOrderedIndices.length);
  const colorCount = Math.max(0, (plan?.springColorOffsets?.length || 1) - 1);
  const colorEndpointOffsets = new Uint32Array(colorCount + 1);
  for (let oi = 0; oi < springColorOrderedIndices.length; oi++) {
    const ai = springColorOrderedIndices[oi];
    springNodeAByColor[oi] = springNodeA[ai];
    springNodeBByColor[oi] = springNodeB[ai];
    springRestByColor[oi] = springRest[ai];
    springInvMassAByColor[oi] = springInvMassA[ai];
    springInvMassBByColor[oi] = springInvMassB[ai];
  }

  // Deterministic no-atomic reduction ownership for next WGSL stage: endpoints
  // are grouped by spring-color batch, then stably ordered by color dispatch
  // order. This lets a future WGSL node-delta reduction consume each color in
  // isolation while preserving current Gauss-Seidel equivalence.
  for (let ci = 0; ci < colorCount; ci++) {
    const springStart = plan.springColorOffsets[ci] || 0;
    const springEnd = plan.springColorOffsets[ci + 1] || springStart;
    colorEndpointOffsets[ci + 1] = colorEndpointOffsets[ci] + Math.max(0, (springEnd - springStart) * 2);
  }
  const endpointCountByColor = colorEndpointOffsets[colorEndpointOffsets.length - 1] || 0;
  const endpointNodeIndicesByColor = new Uint32Array(endpointCountByColor);
  const endpointSpringIndicesByColor = new Uint32Array(endpointCountByColor);
  const endpointSignsI32ByColor = new Int32Array(endpointCountByColor);
  for (let ci = 0; ci < colorCount; ci++) {
    const springStart = plan.springColorOffsets[ci] || 0;
    const springEnd = plan.springColorOffsets[ci + 1] || springStart;
    let dst = colorEndpointOffsets[ci];
    for (let oi = springStart; oi < springEnd; oi++) {
      const ai = springColorOrderedIndices[oi];
      endpointNodeIndicesByColor[dst] = springNodeA[ai];
      endpointSpringIndicesByColor[dst] = ai;
      endpointSignsI32ByColor[dst] = -1;
      dst += 1;
      endpointNodeIndicesByColor[dst] = springNodeB[ai];
      endpointSpringIndicesByColor[dst] = ai;
      endpointSignsI32ByColor[dst] = 1;
      dst += 1;
    }
  }

  // WebGPU storage buffers are naturally 4-byte addressed; widen signs now so
  // upcoming WGSL reduction kernels can consume endpoint direction directly.
  const endpointSignsI32 = new Int32Array(plan?.endpointCount || 0);
  for (let ei = 0; ei < endpointSignsI32.length; ei++) {
    endpointSignsI32[ei] = Number(plan?.endpointSigns?.[ei]) < 0 ? -1 : 1;
  }

  return {
    nodeEndpointOffsets: plan?.nodeEndpointOffsets || new Uint32Array(0),
    endpointSpringIndices: plan?.endpointSpringIndices || new Uint32Array(0),
    endpointSignsI32,
    springNodeA,
    springNodeB,
    springRest,
    springInvMassA,
    springInvMassB,
    springColorOffsets: plan?.springColorOffsets || new Uint32Array(0),
    springColorOrderedIndices,
    springNodeAByColor,
    springNodeBByColor,
    springRestByColor,
    springInvMassAByColor,
    springInvMassBByColor,
    colorEndpointOffsets,
    endpointNodeIndicesByColor,
    endpointSpringIndicesByColor,
    endpointSignsI32ByColor,
    byteLength:
      (plan?.nodeEndpointOffsets?.byteLength || 0)
      + (plan?.endpointSpringIndices?.byteLength || 0)
      + endpointSignsI32.byteLength
      + springNodeA.byteLength
      + springNodeB.byteLength
      + springRest.byteLength
      + springInvMassA.byteLength
      + springInvMassB.byteLength
      + (plan?.springColorOffsets?.byteLength || 0)
      + springColorOrderedIndices.byteLength
      + springNodeAByColor.byteLength
      + springNodeBByColor.byteLength
      + springRestByColor.byteLength
      + springInvMassAByColor.byteLength
      + springInvMassBByColor.byteLength
      + colorEndpointOffsets.byteLength
      + endpointNodeIndicesByColor.byteLength
      + endpointSpringIndicesByColor.byteLength
      + endpointSignsI32ByColor.byteLength,
  };
}


export function reduceSoftSpringVelocityDeltasDeterministic({
  soft,
  dtPos,
  layout,
  deltaLambdaByColor,
} = {}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const safeDtPos = Math.max(1e-8, Number(dtPos) || 0);
  const endpointNodeIndices = layout?.endpointNodeIndicesByColor instanceof Uint32Array
    ? layout.endpointNodeIndicesByColor
    : new Uint32Array(0);
  const endpointSpringIndices = layout?.endpointSpringIndicesByColor instanceof Uint32Array
    ? layout.endpointSpringIndicesByColor
    : new Uint32Array(0);
  const endpointSigns = layout?.endpointSignsI32ByColor instanceof Int32Array
    ? layout.endpointSignsI32ByColor
    : new Int32Array(0);
  const springNodeA = layout?.springNodeAByColor instanceof Uint32Array
    ? layout.springNodeAByColor
    : new Uint32Array(0);
  const springNodeB = layout?.springNodeBByColor instanceof Uint32Array
    ? layout.springNodeBByColor
    : new Uint32Array(0);
  const springInvMassA = layout?.springInvMassAByColor instanceof Float32Array
    ? layout.springInvMassAByColor
    : new Float32Array(0);
  const springInvMassB = layout?.springInvMassBByColor instanceof Float32Array
    ? layout.springInvMassBByColor
    : new Float32Array(0);
  const dlByColor = deltaLambdaByColor instanceof Float32Array
    ? deltaLambdaByColor
    : new Float32Array(0);

  const nodePredX = new Float32Array(nodes.length);
  const nodePredY = new Float32Array(nodes.length);
  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni] || {};
    nodePredX[ni] = (Number(node.x) || 0) + (Number(node.vx) || 0) * safeDtPos;
    nodePredY[ni] = (Number(node.y) || 0) + (Number(node.vy) || 0) * safeDtPos;
  }

  const springNx = new Float32Array(springNodeA.length);
  const springNy = new Float32Array(springNodeA.length);
  for (let si = 0; si < springNodeA.length; si++) {
    const ia = springNodeA[si];
    const ib = springNodeB[si];
    if (ia >= nodePredX.length || ib >= nodePredX.length) continue;
    const dx = nodePredX[ib] - nodePredX[ia];
    const dy = nodePredY[ib] - nodePredY[ia];
    const d = Math.max(1e-6, Math.hypot(dx, dy));
    springNx[si] = dx / d;
    springNy[si] = dy / d;
  }

  const deltaVxByNode = new Float32Array(nodes.length);
  const deltaVyByNode = new Float32Array(nodes.length);
  const contributionCountByNode = new Uint32Array(nodes.length);

  const endpointCount = Math.min(endpointNodeIndices.length, endpointSpringIndices.length, endpointSigns.length);
  for (let ei = 0; ei < endpointCount; ei++) {
    const ni = endpointNodeIndices[ei];
    const si = endpointSpringIndices[ei];
    if (ni >= deltaVxByNode.length || si >= dlByColor.length) continue;

    const sign = endpointSigns[ei] < 0 ? -1 : 1;
    const w = sign < 0 ? springInvMassA[si] : springInvMassB[si];
    const dl = dlByColor[si];
    if (!Number.isFinite(dl) || !Number.isFinite(w)) continue;

    const scale = (sign * w * dl) / safeDtPos;
    deltaVxByNode[ni] += springNx[si] * scale;
    deltaVyByNode[ni] += springNy[si] * scale;
    contributionCountByNode[ni] += 1;
  }

  return {
    deltaVxByNode,
    deltaVyByNode,
    contributionCountByNode,
  };
}

function computeVelocityDeltaParityStats(proposed = new Float32Array(0), expected = new Float32Array(0)) {
  const count = Math.min(proposed.length, expected.length);
  if (count <= 0) {
    return {
      comparedCount: 0,
      absMean: 0,
      absMax: 0,
      l2: 0,
      valid: false,
    };
  }

  let absSum = 0;
  let absMax = 0;
  let l2Sum = 0;
  for (let i = 0; i < count; i++) {
    const pv = Number(proposed[i]) || 0;
    const ev = Number(expected[i]) || 0;
    const diff = pv - ev;
    const abs = Math.abs(diff);
    absSum += abs;
    absMax = Math.max(absMax, abs);
    l2Sum += diff * diff;
  }

  return {
    comparedCount: count,
    absMean: absSum / count,
    absMax,
    l2: Math.sqrt(l2Sum),
    valid: true,
  };
}

function computeContributionCountParityStats(proposed = new Uint32Array(0), expected = new Uint32Array(0)) {
  const count = Math.min(proposed.length, expected.length);
  if (count <= 0) {
    return {
      comparedCount: 0,
      mismatchCount: 0,
      mismatchRatio: 0,
      maxAbs: 0,
      valid: false,
    };
  }

  let mismatchCount = 0;
  let maxAbs = 0;
  for (let i = 0; i < count; i++) {
    const pv = (Number(proposed[i]) || 0) >>> 0;
    const ev = (Number(expected[i]) || 0) >>> 0;
    const abs = Math.abs(pv - ev);
    if (abs > 0) mismatchCount += 1;
    if (abs > maxAbs) maxAbs = abs;
  }

  return {
    comparedCount: count,
    mismatchCount,
    mismatchRatio: mismatchCount / count,
    maxAbs,
    valid: true,
  };
}

function computeSoftSpringVelocityProposalSignature({ soft, layout, lambdaCache, dtPos, alpha }) {
  const nodeCount = Array.isArray(soft?.nodes) ? soft.nodes.length : 0;
  const springCount = layout?.springNodeA?.length || 0;
  let hash = 2166136261;

  const mix = (value) => {
    const v = Number.isFinite(value) ? Number(value) : 0;
    const scaled = Math.trunc(v * 1e6) | 0;
    hash ^= (scaled >>> 0);
    hash = Math.imul(hash, 16777619) >>> 0;
  };

  mix(nodeCount);
  mix(springCount);
  mix(dtPos);
  mix(alpha);

  for (let ni = 0; ni < nodeCount; ni++) {
    const node = soft.nodes[ni] || {};
    mix((Number(node.x) || 0) + (Number(node.vx) || 0) * dtPos);
    mix((Number(node.y) || 0) + (Number(node.vy) || 0) * dtPos);
  }

  const nodeA = layout?.springNodeA instanceof Uint32Array ? layout.springNodeA : new Uint32Array(0);
  const nodeB = layout?.springNodeB instanceof Uint32Array ? layout.springNodeB : new Uint32Array(0);
  const rest = layout?.springRest instanceof Float32Array ? layout.springRest : new Float32Array(0);
  const invMassA = layout?.springInvMassA instanceof Float32Array ? layout.springInvMassA : new Float32Array(0);
  const invMassB = layout?.springInvMassB instanceof Float32Array ? layout.springInvMassB : new Float32Array(0);

  for (let si = 0; si < springCount; si++) {
    mix(nodeA[si] || 0);
    mix(nodeB[si] || 0);
    mix(rest[si] || 0);
    mix(invMassA[si] || 0);
    mix(invMassB[si] || 0);
    mix(lambdaCache?.[si] || 0);
  }

  return `spr-v1-${hash.toString(16)}-${nodeCount}-${springCount}`;
}

function isGpuOnlyFastMode(wgslOffload) {
  return String(wgslOffload?.modeProfile || '').trim().toLowerCase() === 'gpu-only-fast';
}

function checkFiniteFloat32Array(values) {
  if (!(values instanceof Float32Array)) return { allFinite: false, nonFiniteCount: 0, comparedCount: 0 };
  let nonFiniteCount = 0;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) nonFiniteCount += 1;
  }
  return {
    allFinite: nonFiniteCount === 0,
    nonFiniteCount,
    comparedCount: values.length,
  };
}

function applySoftSpringWgslAuthoritativeProposal({
  soft,
  lambdaCache,
  nodeDeltaVx,
  nodeDeltaVy,
  lambdaNextBySpring,
}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const nodeCount = Math.min(nodes.length, nodeDeltaVx.length, nodeDeltaVy.length);
  for (let ni = 0; ni < nodeCount; ni++) {
    const node = nodes[ni];
    if (!node) continue;
    node.vx = (Number(node.vx) || 0) + (Number(nodeDeltaVx[ni]) || 0);
    node.vy = (Number(node.vy) || 0) + (Number(nodeDeltaVy[ni]) || 0);
  }

  const springCount = Math.min(lambdaNextBySpring.length, lambdaCache?.length || 0);
  for (let si = 0; si < springCount; si++) {
    lambdaCache[si] = Number(lambdaNextBySpring[si]) || 0;
  }
}

export function applySoftSpringsXPBDVelocityGpuOnly({
  soft,
  dtPos,
  stiffnessScale,
  lambdaCache,
  softXpbdIters,
  softXpbdBaseCompliance,
  clamp,
  skipClusterSet = null,
  wgslOffload,
}) {
  if (!soft?.nodes?.length || !soft?.springs?.length) return;
  if (!Array.isArray(lambdaCache) && !(lambdaCache instanceof Float32Array)) {
    throw new Error('gpu-only soft spring XPBD pass requires lambdaCache array-like');
  }
  if (typeof clamp !== 'function') {
    throw new Error('gpu-only soft spring XPBD pass requires clamp callback');
  }

  const alpha = (softXpbdBaseCompliance / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);
  let xpbdIterStart = 0;

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    // Unblocker for upcoming WGSL XPBD stage: prepare deterministic CSR endpoint
    // ownership and spring SoA buffers now so the compute stage can run spring
    // solve + node reduction without atomics changing ownership semantics.
    const plan = buildSoftSpringXpbdWgslPlan({ soft, skipClusterSet });
    const layout = buildSoftSpringXpbdWgslLayout({ soft, plan });
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.lastPreparedSpringCount = plan.activeSpringCount;
    wgslOffload.state.lastPreparedEndpointCount = plan.endpointCount;
    wgslOffload.state.lastPreparedColorCount = Math.max(0, (plan.springColorOffsets?.length || 1) - 1);
    wgslOffload.state.lastPreparedColorEndpointCount = layout.endpointSpringIndicesByColor.length;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared';

    const proposalSignature = computeSoftSpringVelocityProposalSignature({
      soft,
      layout,
      lambdaCache,
      dtPos,
      alpha,
    });
    wgslOffload.state.lastPreparedProposalSignature = proposalSignature;

    // Concrete WGSL soft-spring stage: consume prior deterministic WGSL velocity
    // reduction as authoritative node/lambda update when the current frame input
    // signature matches and parity remains within tolerance.
    const fastMode = isGpuOnlyFastMode(wgslOffload);
    const cachedProposalReady = wgslOffload.state.enableAuthoritativeVelocityDelta === true
      && wgslOffload.state.lastVelocityDeltaProposalSignature === proposalSignature
      && (
        wgslOffload.state.lastVelocityDeltaProposalSource === 'wgsl-node-reduction'
        || wgslOffload.state.lastVelocityDeltaProposalSource === 'wgsl-node-reduction-fast'
      )
      && (
        fastMode
          ? wgslOffload.state.lastVelocityDeltaFinite?.allFinite === true
          : (
            wgslOffload.state.lastVelocityDeltaParity?.maxAbs <= 1e-5
            && (
              wgslOffload.state.lastContributionCountParity?.mismatchCount === 0
              || wgslOffload.state.lastVelocityDeltaParity?.maxAbs <= 1e-9
            )
            && wgslOffload.state.lastContributionCountParity?.comparedCount === soft.nodes.length
          )
      )
      && wgslOffload.state.lastVelocityDeltaProposalNodeVxByColor instanceof Float32Array
      && wgslOffload.state.lastVelocityDeltaProposalNodeVyByColor instanceof Float32Array
      && wgslOffload.state.lastVelocityDeltaProposalNodeContributionCount instanceof Uint32Array
      && wgslOffload.state.lastProposalLambdaNextBySpring instanceof Float32Array
      && wgslOffload.state.lastVelocityDeltaProposalNodeVxByColor.length === soft.nodes.length
      && wgslOffload.state.lastVelocityDeltaProposalNodeVyByColor.length === soft.nodes.length
      && wgslOffload.state.lastVelocityDeltaProposalNodeContributionCount.length === soft.nodes.length
      && wgslOffload.state.lastProposalLambdaNextBySpring.length === soft.springs.length;

    if (cachedProposalReady) {
      applySoftSpringWgslAuthoritativeProposal({
        soft,
        lambdaCache,
        nodeDeltaVx: wgslOffload.state.lastVelocityDeltaProposalNodeVxByColor,
        nodeDeltaVy: wgslOffload.state.lastVelocityDeltaProposalNodeVyByColor,
        lambdaNextBySpring: wgslOffload.state.lastProposalLambdaNextBySpring,
      });
      xpbdIterStart = 1;
      wgslOffload.state.lastAuthoritativeProposalSignature = proposalSignature;
      wgslOffload.state.lastAuthoritativeProposalSource = 'wgsl-node-reduction';
      wgslOffload.state.lastMode = 'wgsl-velocity-authoritative';
      wgslOffload.state.lastError = null;
    }

    // Continue probing/proposal dispatch while CPU remains authoritative so the
    // next matching frame can promote deterministic WGSL deltas safely.
    if (!cachedProposalReady && canUseWgslOffload(wgslOffload)) {
      if (wgslOffload.state.wgslInFlight) {
        wgslOffload.state.wgslSkippedWhileBusy = (wgslOffload.state.wgslSkippedWhileBusy || 0) + 1;
      } else {
        const runId = (wgslOffload.state.wgslRunId || 0) + 1;
        wgslOffload.state.wgslRunId = runId;
        wgslOffload.state.wgslInFlight = true;
        void Promise.all([
          dispatchSoftSpringWgslProbe({ soft, offload: wgslOffload, layout }),
          dispatchSoftSpringWgslLambdaProposal({
            soft,
            offload: wgslOffload,
            layout,
            dtPos,
            alpha,
            lambdaCache,
          }),
        ])
          .then(async ([probeRan, proposalRan]) => {
            if (proposalRan) {
              const velocityProposal = await dispatchSoftSpringWgslVelocityDeltaProposal({
                soft,
                offload: wgslOffload,
                layout,
                dtPos,
                deltaLambdaByColor: wgslOffload.state.lastProposalDeltaLambdaByColor,
                includeEndpointTelemetry: !fastMode,
              });
              let proposalReduction = null;
              if (!fastMode) {
                proposalReduction = reduceSoftSpringVelocityDeltasDeterministic({
                  soft,
                  dtPos,
                  layout,
                  deltaLambdaByColor: wgslOffload.state.lastProposalDeltaLambdaByColor,
                });
                wgslOffload.state.lastVelocityDeltaExpectedNodeVxByColor = proposalReduction.deltaVxByNode;
                wgslOffload.state.lastVelocityDeltaExpectedNodeVyByColor = proposalReduction.deltaVyByNode;
                wgslOffload.state.lastVelocityDeltaExpectedNodeContributionCount = proposalReduction.contributionCountByNode;
              } else {
                wgslOffload.state.lastVelocityDeltaExpectedNodeVxByColor = null;
                wgslOffload.state.lastVelocityDeltaExpectedNodeVyByColor = null;
                wgslOffload.state.lastVelocityDeltaExpectedNodeContributionCount = null;
              }

              if (velocityProposal) {
                wgslOffload.state.lastVelocityDeltaProposalDispatch = velocityProposal.dispatchCount;
                wgslOffload.state.lastVelocityDeltaReductionDispatch = velocityProposal.reductionDispatchCount;
                wgslOffload.state.lastVelocityDeltaProposalEndpointVxByColor = velocityProposal.endpointDeltaVX;
                wgslOffload.state.lastVelocityDeltaProposalEndpointVyByColor = velocityProposal.endpointDeltaVY;
                wgslOffload.state.lastVelocityDeltaProposalNodeVxByColor = velocityProposal.nodeDeltaVx;
                wgslOffload.state.lastVelocityDeltaProposalNodeVyByColor = velocityProposal.nodeDeltaVy;
                wgslOffload.state.lastVelocityDeltaProposalNodeContributionCount = velocityProposal.nodeContributionCount;
                wgslOffload.state.lastVelocityDeltaProposalSource = fastMode ? 'wgsl-node-reduction-fast' : 'wgsl-node-reduction';
              } else if (proposalReduction) {
                wgslOffload.state.lastVelocityDeltaProposalNodeVxByColor = proposalReduction.deltaVxByNode;
                wgslOffload.state.lastVelocityDeltaProposalNodeVyByColor = proposalReduction.deltaVyByNode;
                wgslOffload.state.lastVelocityDeltaProposalNodeContributionCount = proposalReduction.contributionCountByNode;
                wgslOffload.state.lastVelocityDeltaProposalSource = 'cpu-deterministic-reduction';
              }

              const finiteVx = checkFiniteFloat32Array(wgslOffload.state.lastVelocityDeltaProposalNodeVxByColor);
              const finiteVy = checkFiniteFloat32Array(wgslOffload.state.lastVelocityDeltaProposalNodeVyByColor);
              wgslOffload.state.lastVelocityDeltaFinite = {
                allFinite: finiteVx.allFinite && finiteVy.allFinite,
                nonFiniteCount: (finiteVx.nonFiniteCount || 0) + (finiteVy.nonFiniteCount || 0),
                comparedCount: Math.min(finiteVx.comparedCount || 0, finiteVy.comparedCount || 0),
              };

              if (fastMode) {
                wgslOffload.state.lastContributionCountParity = {
                  mismatchCount: 0,
                  comparedCount: soft.nodes.length,
                  source: 'skipped-fast-mode',
                };
                wgslOffload.state.lastVelocityDeltaParity = {
                  vx: null,
                  vy: null,
                  contributionCount: wgslOffload.state.lastContributionCountParity,
                  comparedNodeCount: soft.nodes.length,
                  maxAbs: 0,
                  meanAbs: 0,
                  source: wgslOffload.state.lastVelocityDeltaProposalSource,
                  mode: 'gpu-only-fast',
                  validation: 'skipped-cpu-parity',
                };
              } else if (proposalReduction) {
                const vxParity = computeVelocityDeltaParityStats(
                  wgslOffload.state.lastVelocityDeltaProposalNodeVxByColor,
                  proposalReduction.deltaVxByNode,
                );
                const vyParity = computeVelocityDeltaParityStats(
                  wgslOffload.state.lastVelocityDeltaProposalNodeVyByColor,
                  proposalReduction.deltaVyByNode,
                );
                const contributionCountParity = computeContributionCountParityStats(
                  wgslOffload.state.lastVelocityDeltaProposalNodeContributionCount,
                  proposalReduction.contributionCountByNode,
                );
                wgslOffload.state.lastContributionCountParity = contributionCountParity;
                wgslOffload.state.lastVelocityDeltaParity = {
                  vx: vxParity,
                  vy: vyParity,
                  contributionCount: contributionCountParity,
                  comparedNodeCount: Math.min(vxParity.comparedCount, vyParity.comparedCount, contributionCountParity.comparedCount),
                  maxAbs: Math.max(vxParity.absMax, vyParity.absMax),
                  meanAbs: (vxParity.absMean + vyParity.absMean) * 0.5,
                  source: wgslOffload.state.lastVelocityDeltaProposalSource,
                };
              }
              wgslOffload.state.lastVelocityDeltaProposalSignature = proposalSignature;
            }
            if (probeRan || proposalRan) {
              wgslOffload.state.lastError = null;
              wgslOffload.state.lastMode = proposalRan ? 'wgsl-velocity-proposal' : 'wgsl-probe';
            }
          })
          .catch((err) => {
            wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
            wgslOffload.state.lastMode = 'cpu-fallback';
          })
          .finally(() => {
            if (wgslOffload.state.wgslRunId === runId) {
              wgslOffload.state.wgslInFlight = false;
              wgslOffload.state.lastCompletedWgslRunId = runId;
            }
          });
      }
    }
  }

  for (let iter = xpbdIterStart; iter < softXpbdIters; iter++) {
    for (let si = 0; si < soft.springs.length; si++) {
      const [i, j, rest] = soft.springs[si];
      const a = soft.nodes[i];
      const b = soft.nodes[j];
      if (!a || !b) continue;
      if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

      const ax = a.x + a.vx * dtPos;
      const ay = a.y + a.vy * dtPos;
      const bx = b.x + b.vx * dtPos;
      const by = b.y + b.vy * dtPos;

      const dx = bx - ax;
      const dy = by - ay;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = dx / d;
      const ny = dy / d;
      const strainCap = Math.max(0.05, Math.abs(rest) * 0.45);
      const C = clamp(d - rest, -strainCap, strainCap);

      const wA = 1 / Math.max(0.02, a.mass || 1);
      const wB = 1 / Math.max(0.02, b.mass || 1);
      const wSum = wA + wB;
      if (wSum <= 1e-9) continue;

      const lambdaPrev = Number(lambdaCache[si]) || 0;
      let dl = (-C - alpha * lambdaPrev) / (wSum + alpha);
      if (!Number.isFinite(dl)) continue;
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      lambdaCache[si] = lambdaNext;

      const corrAx = -wA * dl * nx;
      const corrAy = -wA * dl * ny;
      const corrBx = wB * dl * nx;
      const corrBy = wB * dl * ny;

      a.vx += corrAx / dtPos;
      a.vy += corrAy / dtPos;
      b.vx += corrBx / dtPos;
      b.vy += corrBy / dtPos;
    }
  }
}
