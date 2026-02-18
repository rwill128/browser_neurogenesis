/**
 * GPU-only runtime soft area XPBD pass.
 * Isolates area-preservation constraint stepping for the gpu-only runtime path
 * while preserving baseline/default behavior and call contracts.
 */
const WGSL_WORKGROUP_SIZE = 64;

const softAreaClusterProbeWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  clusterCount: u32,
  dtPos: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> nodeVX: array<f32>;
@group(0) @binding(4) var<storage, read> nodeVY: array<f32>;
@group(0) @binding(5) var<storage, read> clusterOffsets: array<u32>;
@group(0) @binding(6) var<storage, read> clusterNodeIndices: array<u32>;
@group(0) @binding(7) var<storage, read_write> areaOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= params.clusterCount) { return; }

  let start = clusterOffsets[ci];
  let end = clusterOffsets[ci + 1u];
  if (end <= start + 1u) {
    areaOut[ci] = 0.0;
    return;
  }

  var twiceSignedArea = 0.0;
  for (var ei = start; ei < end; ei = ei + 1u) {
    let ni = clusterNodeIndices[ei];
    let nj = clusterNodeIndices[select(start, ei + 1u, (ei + 1u) < end)];
    if (ni >= params.nodeCount || nj >= params.nodeCount) { continue; }

    let ax = nodeX[ni] + nodeVX[ni] * params.dtPos;
    let ay = nodeY[ni] + nodeVY[ni] * params.dtPos;
    let bx = nodeX[nj] + nodeVX[nj] * params.dtPos;
    let by = nodeY[nj] + nodeVY[nj] * params.dtPos;
    twiceSignedArea += ax * by - bx * ay;
  }

  areaOut[ci] = 0.5 * twiceSignedArea;
}
`;

const softAreaLambdaProposalWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  clusterCount: u32,
  dtPos: f32,
  alpha: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> nodeVX: array<f32>;
@group(0) @binding(4) var<storage, read> nodeVY: array<f32>;
@group(0) @binding(5) var<storage, read> clusterOffsets: array<u32>;
@group(0) @binding(6) var<storage, read> clusterNodeIndices: array<u32>;
@group(0) @binding(7) var<storage, read> clusterNodeInvMass: array<f32>;
@group(0) @binding(8) var<storage, read> clusterRestArea: array<f32>;
@group(0) @binding(9) var<storage, read> clusterLambdaPrev: array<f32>;
@group(0) @binding(10) var<storage, read_write> deltaLambdaOut: array<f32>;
@group(0) @binding(11) var<storage, read_write> lambdaNextOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= params.clusterCount) { return; }

  let start = clusterOffsets[ci];
  let end = clusterOffsets[ci + 1u];
  if (end <= start + 1u) {
    deltaLambdaOut[ci] = 0.0;
    lambdaNextOut[ci] = clusterLambdaPrev[ci];
    return;
  }

  var twiceSignedArea = 0.0;
  var sumWGrad2 = 0.0;
  for (var ei = start; ei < end; ei = ei + 1u) {
    let ni = clusterNodeIndices[ei];
    let nj = clusterNodeIndices[select(start, ei + 1u, (ei + 1u) < end)];
    if (ni >= params.nodeCount || nj >= params.nodeCount) { continue; }

    let ax = nodeX[ni] + nodeVX[ni] * params.dtPos;
    let ay = nodeY[ni] + nodeVY[ni] * params.dtPos;
    let bx = nodeX[nj] + nodeVX[nj] * params.dtPos;
    let by = nodeY[nj] + nodeVY[nj] * params.dtPos;
    twiceSignedArea += ax * by - bx * ay;

    let prevEi = select(end - 1u, ei - 1u, ei > start);
    let pi = clusterNodeIndices[prevEi];
    let ni2 = clusterNodeIndices[select(start, ei + 1u, (ei + 1u) < end)];
    if (pi >= params.nodeCount || ni2 >= params.nodeCount) { continue; }

    let px = nodeX[pi] + nodeVX[pi] * params.dtPos;
    let py = nodeY[pi] + nodeVY[pi] * params.dtPos;
    let nx = nodeX[ni2] + nodeVX[ni2] * params.dtPos;
    let ny = nodeY[ni2] + nodeVY[ni2] * params.dtPos;
    let gx = 0.5 * (ny - py);
    let gy = 0.5 * (px - nx);
    let w = clusterNodeInvMass[ei];
    sumWGrad2 += w * (gx * gx + gy * gy);
  }

  if (sumWGrad2 <= 1e-10) {
    deltaLambdaOut[ci] = 0.0;
    lambdaNextOut[ci] = clusterLambdaPrev[ci];
    return;
  }

  let area = 0.5 * twiceSignedArea;
  let c = area - clusterRestArea[ci];
  let lambdaPrev = clusterLambdaPrev[ci];
  let dlRaw = (-c - params.alpha * lambdaPrev) / (sumWGrad2 + params.alpha);
  let dlClamped = clamp(dlRaw, -2.0, 2.0);
  let lambdaNext = clamp(lambdaPrev + dlClamped, -20.0, 20.0);
  let dl = lambdaNext - lambdaPrev;

  deltaLambdaOut[ci] = dl;
  lambdaNextOut[ci] = lambdaNext;
}
`;


const softAreaVelocityDeltaProposalWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  clusterCount: u32,
  endpointCount: u32,
  dtPos: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> nodeVX: array<f32>;
@group(0) @binding(4) var<storage, read> nodeVY: array<f32>;
@group(0) @binding(5) var<storage, read> clusterOffsets: array<u32>;
@group(0) @binding(6) var<storage, read> clusterNodeIndices: array<u32>;
@group(0) @binding(7) var<storage, read> clusterNodeInvMass: array<f32>;
@group(0) @binding(8) var<storage, read> endpointClusterIndex: array<u32>;
@group(0) @binding(9) var<storage, read> clusterDeltaLambda: array<f32>;
@group(0) @binding(10) var<storage, read_write> endpointDeltaVXOut: array<f32>;
@group(0) @binding(11) var<storage, read_write> endpointDeltaVYOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ei = gid.x;
  if (ei >= params.endpointCount) { return; }

  let ci = endpointClusterIndex[ei];
  if (ci >= params.clusterCount) {
    endpointDeltaVXOut[ei] = 0.0;
    endpointDeltaVYOut[ei] = 0.0;
    return;
  }

  let start = clusterOffsets[ci];
  let end = clusterOffsets[ci + 1u];
  if (end <= start + 1u || params.dtPos <= 1e-8) {
    endpointDeltaVXOut[ei] = 0.0;
    endpointDeltaVYOut[ei] = 0.0;
    return;
  }

  let ni = clusterNodeIndices[ei];
  if (ni >= params.nodeCount) {
    endpointDeltaVXOut[ei] = 0.0;
    endpointDeltaVYOut[ei] = 0.0;
    return;
  }

  let prevEi = select(end - 1u, ei - 1u, ei > start);
  let nextEi = select(start, ei + 1u, (ei + 1u) < end);
  let pi = clusterNodeIndices[prevEi];
  let ni2 = clusterNodeIndices[nextEi];
  if (pi >= params.nodeCount || ni2 >= params.nodeCount) {
    endpointDeltaVXOut[ei] = 0.0;
    endpointDeltaVYOut[ei] = 0.0;
    return;
  }

  let px = nodeX[pi] + nodeVX[pi] * params.dtPos;
  let py = nodeY[pi] + nodeVY[pi] * params.dtPos;
  let nx = nodeX[ni2] + nodeVX[ni2] * params.dtPos;
  let ny = nodeY[ni2] + nodeVY[ni2] * params.dtPos;
  let gx = 0.5 * (ny - py);
  let gy = 0.5 * (px - nx);

  let dl = clusterDeltaLambda[ci];
  let w = clusterNodeInvMass[ei];
  endpointDeltaVXOut[ei] = (w * gx * dl) / params.dtPos;
  endpointDeltaVYOut[ei] = (w * gy * dl) / params.dtPos;
}
`;

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined') return false;
  if (typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function ensureSoftAreaProbeBuffers(offload, nodeCount, endpointCount, clusterCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.areaProbeNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.areaProbeNodeCapacity ? state.areaProbeNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaProbeNodeX?.destroy?.();
    state.areaProbeNodeY?.destroy?.();
    state.areaProbeNodeVX?.destroy?.();
    state.areaProbeNodeVY?.destroy?.();
    state.areaProbeNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeVX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeVY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeCapacity = capacity;
    state.areaProbeBindGroup = null;
  }

  const requiredEndpointCapacity = Math.max(1, endpointCount);
  if ((state.areaProbeEndpointCapacity || 0) < requiredEndpointCapacity) {
    const capacity = Math.max(requiredEndpointCapacity, state.areaProbeEndpointCapacity ? state.areaProbeEndpointCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaProbeClusterNodeIndices?.destroy?.();
    state.areaProbeClusterNodeIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeEndpointCapacity = capacity;
    state.areaProbeBindGroup = null;
  }

  const requiredClusterCapacity = Math.max(1, clusterCount);
  if ((state.areaProbeClusterCapacity || 0) < requiredClusterCapacity) {
    const capacity = Math.max(requiredClusterCapacity, state.areaProbeClusterCapacity ? state.areaProbeClusterCapacity * 2 : 64);
    const bytes = capacity * 4;
    state.areaProbeClusterOffsets?.destroy?.();
    state.areaProbeAreaOut?.destroy?.();
    state.areaProbeAreaReadback?.destroy?.();
    state.areaProbeClusterOffsets = device.createBuffer({ size: (capacity + 1) * 4, usage: storageUsage });
    state.areaProbeAreaOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.areaProbeAreaReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.areaProbeClusterCapacity = capacity;
    state.areaProbeBindGroup = null;
  }

  if (!state.areaProbeParams) {
    state.areaProbeParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.areaProbeBindGroup = null;
  }

  return state;
}


function ensureSoftAreaLambdaProposalBuffers(offload, nodeCount, endpointCount, clusterCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.areaLambdaNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.areaLambdaNodeCapacity ? state.areaLambdaNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaLambdaNodeX?.destroy?.();
    state.areaLambdaNodeY?.destroy?.();
    state.areaLambdaNodeVX?.destroy?.();
    state.areaLambdaNodeVY?.destroy?.();
    state.areaLambdaNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaNodeVX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaNodeVY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaNodeCapacity = capacity;
    state.areaLambdaBindGroup = null;
  }

  const requiredEndpointCapacity = Math.max(1, endpointCount);
  if ((state.areaLambdaEndpointCapacity || 0) < requiredEndpointCapacity) {
    const capacity = Math.max(requiredEndpointCapacity, state.areaLambdaEndpointCapacity ? state.areaLambdaEndpointCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaLambdaClusterNodeIndices?.destroy?.();
    state.areaLambdaClusterNodeInvMass?.destroy?.();
    state.areaLambdaClusterNodeIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaClusterNodeInvMass = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaEndpointCapacity = capacity;
    state.areaLambdaBindGroup = null;
  }

  const requiredClusterCapacity = Math.max(1, clusterCount);
  if ((state.areaLambdaClusterCapacity || 0) < requiredClusterCapacity) {
    const capacity = Math.max(requiredClusterCapacity, state.areaLambdaClusterCapacity ? state.areaLambdaClusterCapacity * 2 : 64);
    const bytes = capacity * 4;
    state.areaLambdaClusterOffsets?.destroy?.();
    state.areaLambdaClusterRestArea?.destroy?.();
    state.areaLambdaClusterPrev?.destroy?.();
    state.areaLambdaDeltaOut?.destroy?.();
    state.areaLambdaNextOut?.destroy?.();
    state.areaLambdaDeltaReadback?.destroy?.();
    state.areaLambdaNextReadback?.destroy?.();
    state.areaLambdaClusterOffsets = device.createBuffer({ size: (capacity + 1) * 4, usage: storageUsage });
    state.areaLambdaClusterRestArea = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaClusterPrev = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaDeltaOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaLambdaNextOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaLambdaDeltaReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaLambdaNextReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaLambdaClusterCapacity = capacity;
    state.areaLambdaBindGroup = null;
  }

  if (!state.areaLambdaParams) {
    state.areaLambdaParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.areaLambdaBindGroup = null;
  }

  return state;
}


function ensureSoftAreaVelocityDeltaProposalBuffers(offload, nodeCount, endpointCount, clusterCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.areaVelocityNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.areaVelocityNodeCapacity ? state.areaVelocityNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaVelocityNodeX?.destroy?.();
    state.areaVelocityNodeY?.destroy?.();
    state.areaVelocityNodeVX?.destroy?.();
    state.areaVelocityNodeVY?.destroy?.();
    state.areaVelocityNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityNodeVX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityNodeVY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityNodeCapacity = capacity;
    state.areaVelocityBindGroup = null;
  }

  const requiredEndpointCapacity = Math.max(1, endpointCount);
  if ((state.areaVelocityEndpointCapacity || 0) < requiredEndpointCapacity) {
    const capacity = Math.max(requiredEndpointCapacity, state.areaVelocityEndpointCapacity ? state.areaVelocityEndpointCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaVelocityClusterNodeIndices?.destroy?.();
    state.areaVelocityClusterNodeInvMass?.destroy?.();
    state.areaVelocityEndpointClusterIndex?.destroy?.();
    state.areaVelocityDeltaVXOut?.destroy?.();
    state.areaVelocityDeltaVYOut?.destroy?.();
    state.areaVelocityDeltaVXReadback?.destroy?.();
    state.areaVelocityDeltaVYReadback?.destroy?.();
    state.areaVelocityClusterNodeIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityClusterNodeInvMass = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityEndpointClusterIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityDeltaVXOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaVelocityDeltaVYOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaVelocityDeltaVXReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaVelocityDeltaVYReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaVelocityEndpointCapacity = capacity;
    state.areaVelocityBindGroup = null;
  }

  const requiredClusterCapacity = Math.max(1, clusterCount);
  if ((state.areaVelocityClusterCapacity || 0) < requiredClusterCapacity) {
    const capacity = Math.max(requiredClusterCapacity, state.areaVelocityClusterCapacity ? state.areaVelocityClusterCapacity * 2 : 64);
    const bytes = capacity * 4;
    state.areaVelocityClusterOffsets?.destroy?.();
    state.areaVelocityClusterDeltaLambda?.destroy?.();
    state.areaVelocityClusterOffsets = device.createBuffer({ size: (capacity + 1) * 4, usage: storageUsage });
    state.areaVelocityClusterDeltaLambda = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityClusterCapacity = capacity;
    state.areaVelocityBindGroup = null;
  }

  if (!state.areaVelocityParams) {
    state.areaVelocityParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.areaVelocityBindGroup = null;
  }

  return state;
}

async function dispatchSoftAreaWgslVelocityDeltaProposal({ soft, offload, plan, dtPos, deltaByCluster }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const clusterCount = Number(plan?.clusterCount) || 0;
  const endpointCount = Number(plan?.endpointCount) || 0;
  if (nodes.length <= 0 || clusterCount <= 0 || endpointCount <= 0) return false;
  if (!(deltaByCluster instanceof Float32Array) || deltaByCluster.length < clusterCount) return false;

  const state = ensureSoftAreaVelocityDeltaProposalBuffers(offload, nodes.length, endpointCount, clusterCount);
  const device = offload.device;

  if (!state.areaVelocityPipeline) {
    const module = device.createShaderModule({ code: softAreaVelocityDeltaProposalWgsl });
    state.areaVelocityPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.areaVelocityBindGroup = null;
  }

  if (!state.areaVelocityBindGroup) {
    state.areaVelocityBindGroup = device.createBindGroup({
      layout: state.areaVelocityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.areaVelocityParams } },
        { binding: 1, resource: { buffer: state.areaVelocityNodeX } },
        { binding: 2, resource: { buffer: state.areaVelocityNodeY } },
        { binding: 3, resource: { buffer: state.areaVelocityNodeVX } },
        { binding: 4, resource: { buffer: state.areaVelocityNodeVY } },
        { binding: 5, resource: { buffer: state.areaVelocityClusterOffsets } },
        { binding: 6, resource: { buffer: state.areaVelocityClusterNodeIndices } },
        { binding: 7, resource: { buffer: state.areaVelocityClusterNodeInvMass } },
        { binding: 8, resource: { buffer: state.areaVelocityEndpointClusterIndex } },
        { binding: 9, resource: { buffer: state.areaVelocityClusterDeltaLambda } },
        { binding: 10, resource: { buffer: state.areaVelocityDeltaVXOut } },
        { binding: 11, resource: { buffer: state.areaVelocityDeltaVYOut } },
      ],
    });
  }

  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  const nodeVX = new Float32Array(nodes.length);
  const nodeVY = new Float32Array(nodes.length);
  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni] || {};
    nodeX[ni] = Number(node.x) || 0;
    nodeY[ni] = Number(node.y) || 0;
    nodeVX[ni] = Number(node.vx) || 0;
    nodeVY[ni] = Number(node.vy) || 0;
  }

  const paramsBytes = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  paramsU32[0] = nodes.length >>> 0;
  paramsU32[1] = clusterCount >>> 0;
  paramsU32[2] = endpointCount >>> 0;
  paramsF32[3] = Number.isFinite(dtPos) ? dtPos : 0;

  device.queue.writeBuffer(state.areaVelocityParams, 0, paramsBytes);
  device.queue.writeBuffer(state.areaVelocityNodeX, 0, nodeX);
  device.queue.writeBuffer(state.areaVelocityNodeY, 0, nodeY);
  device.queue.writeBuffer(state.areaVelocityNodeVX, 0, nodeVX);
  device.queue.writeBuffer(state.areaVelocityNodeVY, 0, nodeVY);
  device.queue.writeBuffer(state.areaVelocityClusterOffsets, 0, plan.clusterOffsets);
  device.queue.writeBuffer(state.areaVelocityClusterNodeIndices, 0, plan.clusterNodeIndices);
  device.queue.writeBuffer(state.areaVelocityClusterNodeInvMass, 0, plan.clusterNodeInvMass);
  device.queue.writeBuffer(state.areaVelocityEndpointClusterIndex, 0, plan.endpointClusterIndex);
  device.queue.writeBuffer(state.areaVelocityClusterDeltaLambda, 0, deltaByCluster);

  const dispatchCount = Math.ceil(endpointCount / WGSL_WORKGROUP_SIZE);
  const bytes = endpointCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.areaVelocityPipeline);
  pass.setBindGroup(0, state.areaVelocityBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  encoder.copyBufferToBuffer(state.areaVelocityDeltaVXOut, 0, state.areaVelocityDeltaVXReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.areaVelocityDeltaVYOut, 0, state.areaVelocityDeltaVYReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.areaVelocityDeltaVXReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mappedVX = state.areaVelocityDeltaVXReadback.getMappedRange(0, bytes);
  const deltaVxByEndpoint = new Float32Array(mappedVX.slice(0));
  state.areaVelocityDeltaVXReadback.unmap();

  await state.areaVelocityDeltaVYReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mappedVY = state.areaVelocityDeltaVYReadback.getMappedRange(0, bytes);
  const deltaVyByEndpoint = new Float32Array(mappedVY.slice(0));
  state.areaVelocityDeltaVYReadback.unmap();

  let maxAbsDelta = 0;
  let sumAbsDelta = 0;
  for (let ei = 0; ei < endpointCount; ei++) {
    const abs = Math.hypot(deltaVxByEndpoint[ei], deltaVyByEndpoint[ei]);
    if (abs > maxAbsDelta) maxAbsDelta = abs;
    sumAbsDelta += abs;
  }

  state.lastAreaVelocityProposalEndpointCount = endpointCount;
  state.lastAreaVelocityProposalDispatch = dispatchCount;
  state.lastAreaVelocityProposalDeltaVxByEndpoint = deltaVxByEndpoint;
  state.lastAreaVelocityProposalDeltaVyByEndpoint = deltaVyByEndpoint;
  state.lastAreaVelocityProposalAbsDeltaMean = endpointCount > 0 ? sumAbsDelta / endpointCount : 0;
  state.lastAreaVelocityProposalAbsDeltaMax = maxAbsDelta;
  return true;
}

async function dispatchSoftAreaWgslLambdaProposal({ soft, offload, plan, dtPos, alpha }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const clusterCount = Number(plan?.clusterCount) || 0;
  const endpointCount = Number(plan?.endpointCount) || 0;
  if (nodes.length <= 0 || clusterCount <= 0 || endpointCount <= 0) return false;

  const state = ensureSoftAreaLambdaProposalBuffers(offload, nodes.length, endpointCount, clusterCount);
  const device = offload.device;

  if (!state.areaLambdaPipeline) {
    const module = device.createShaderModule({ code: softAreaLambdaProposalWgsl });
    state.areaLambdaPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.areaLambdaBindGroup = null;
  }

  if (!state.areaLambdaBindGroup) {
    state.areaLambdaBindGroup = device.createBindGroup({
      layout: state.areaLambdaPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.areaLambdaParams } },
        { binding: 1, resource: { buffer: state.areaLambdaNodeX } },
        { binding: 2, resource: { buffer: state.areaLambdaNodeY } },
        { binding: 3, resource: { buffer: state.areaLambdaNodeVX } },
        { binding: 4, resource: { buffer: state.areaLambdaNodeVY } },
        { binding: 5, resource: { buffer: state.areaLambdaClusterOffsets } },
        { binding: 6, resource: { buffer: state.areaLambdaClusterNodeIndices } },
        { binding: 7, resource: { buffer: state.areaLambdaClusterNodeInvMass } },
        { binding: 8, resource: { buffer: state.areaLambdaClusterRestArea } },
        { binding: 9, resource: { buffer: state.areaLambdaClusterPrev } },
        { binding: 10, resource: { buffer: state.areaLambdaDeltaOut } },
        { binding: 11, resource: { buffer: state.areaLambdaNextOut } },
      ],
    });
  }

  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  const nodeVX = new Float32Array(nodes.length);
  const nodeVY = new Float32Array(nodes.length);
  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni] || {};
    nodeX[ni] = Number(node.x) || 0;
    nodeY[ni] = Number(node.y) || 0;
    nodeVX[ni] = Number(node.vx) || 0;
    nodeVY[ni] = Number(node.vy) || 0;
  }

  const paramsBytes = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  paramsU32[0] = nodes.length >>> 0;
  paramsU32[1] = clusterCount >>> 0;
  paramsF32[2] = Number.isFinite(dtPos) ? dtPos : 0;
  paramsF32[3] = Number.isFinite(alpha) ? alpha : 0;

  device.queue.writeBuffer(state.areaLambdaParams, 0, paramsBytes);
  device.queue.writeBuffer(state.areaLambdaNodeX, 0, nodeX);
  device.queue.writeBuffer(state.areaLambdaNodeY, 0, nodeY);
  device.queue.writeBuffer(state.areaLambdaNodeVX, 0, nodeVX);
  device.queue.writeBuffer(state.areaLambdaNodeVY, 0, nodeVY);
  device.queue.writeBuffer(state.areaLambdaClusterOffsets, 0, plan.clusterOffsets);
  device.queue.writeBuffer(state.areaLambdaClusterNodeIndices, 0, plan.clusterNodeIndices);
  device.queue.writeBuffer(state.areaLambdaClusterNodeInvMass, 0, plan.clusterNodeInvMass);
  device.queue.writeBuffer(state.areaLambdaClusterRestArea, 0, plan.clusterRestArea);
  device.queue.writeBuffer(state.areaLambdaClusterPrev, 0, plan.clusterLambda);

  const dispatchCount = Math.ceil(clusterCount / WGSL_WORKGROUP_SIZE);
  const bytes = clusterCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.areaLambdaPipeline);
  pass.setBindGroup(0, state.areaLambdaBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  encoder.copyBufferToBuffer(state.areaLambdaDeltaOut, 0, state.areaLambdaDeltaReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.areaLambdaNextOut, 0, state.areaLambdaNextReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.areaLambdaDeltaReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mappedDelta = state.areaLambdaDeltaReadback.getMappedRange(0, bytes);
  const deltaByCluster = new Float32Array(mappedDelta.slice(0));
  state.areaLambdaDeltaReadback.unmap();

  await state.areaLambdaNextReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mappedNext = state.areaLambdaNextReadback.getMappedRange(0, bytes);
  const nextByCluster = new Float32Array(mappedNext.slice(0));
  state.areaLambdaNextReadback.unmap();

  let maxAbsDelta = 0;
  let sumAbsDelta = 0;
  for (let ci = 0; ci < clusterCount; ci++) {
    const abs = Math.abs(deltaByCluster[ci]);
    if (abs > maxAbsDelta) maxAbsDelta = abs;
    sumAbsDelta += abs;
  }

  state.lastAreaProposalClusterCount = clusterCount;
  state.lastAreaProposalDispatch = dispatchCount;
  state.lastAreaProposalDeltaLambdaByCluster = deltaByCluster;
  state.lastAreaProposalLambdaNextByCluster = nextByCluster;
  state.lastAreaProposalAbsDeltaMean = clusterCount > 0 ? sumAbsDelta / clusterCount : 0;
  state.lastAreaProposalAbsDeltaMax = maxAbsDelta;
  return true;
}

async function dispatchSoftAreaWgslProbe({ sim, soft, offload, plan, dtPos }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const clusterCount = Number(plan?.clusterCount) || 0;
  const endpointCount = Number(plan?.endpointCount) || 0;
  if (nodes.length <= 0 || clusterCount <= 0 || endpointCount <= 0) return false;

  const state = ensureSoftAreaProbeBuffers(offload, nodes.length, endpointCount, clusterCount);
  const device = offload.device;

  if (!state.areaProbePipeline) {
    const module = device.createShaderModule({ code: softAreaClusterProbeWgsl });
    state.areaProbePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.areaProbeBindGroup = null;
  }

  if (!state.areaProbeBindGroup) {
    state.areaProbeBindGroup = device.createBindGroup({
      layout: state.areaProbePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.areaProbeParams } },
        { binding: 1, resource: { buffer: state.areaProbeNodeX } },
        { binding: 2, resource: { buffer: state.areaProbeNodeY } },
        { binding: 3, resource: { buffer: state.areaProbeNodeVX } },
        { binding: 4, resource: { buffer: state.areaProbeNodeVY } },
        { binding: 5, resource: { buffer: state.areaProbeClusterOffsets } },
        { binding: 6, resource: { buffer: state.areaProbeClusterNodeIndices } },
        { binding: 7, resource: { buffer: state.areaProbeAreaOut } },
      ],
    });
  }

  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  const nodeVX = new Float32Array(nodes.length);
  const nodeVY = new Float32Array(nodes.length);
  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni] || {};
    nodeX[ni] = Number(node.x) || 0;
    nodeY[ni] = Number(node.y) || 0;
    nodeVX[ni] = Number(node.vx) || 0;
    nodeVY[ni] = Number(node.vy) || 0;
  }

  const paramsBytes = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  paramsU32[0] = nodes.length >>> 0;
  paramsU32[1] = clusterCount >>> 0;
  paramsF32[2] = Number.isFinite(dtPos) ? dtPos : 0;

  device.queue.writeBuffer(state.areaProbeParams, 0, paramsBytes);
  device.queue.writeBuffer(state.areaProbeNodeX, 0, nodeX);
  device.queue.writeBuffer(state.areaProbeNodeY, 0, nodeY);
  device.queue.writeBuffer(state.areaProbeNodeVX, 0, nodeVX);
  device.queue.writeBuffer(state.areaProbeNodeVY, 0, nodeVY);
  device.queue.writeBuffer(state.areaProbeClusterOffsets, 0, plan.clusterOffsets);
  device.queue.writeBuffer(state.areaProbeClusterNodeIndices, 0, plan.clusterNodeIndices);

  const dispatchCount = Math.ceil(clusterCount / WGSL_WORKGROUP_SIZE);
  const bytes = clusterCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.areaProbePipeline);
  pass.setBindGroup(0, state.areaProbeBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  encoder.copyBufferToBuffer(state.areaProbeAreaOut, 0, state.areaProbeAreaReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.areaProbeAreaReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = state.areaProbeAreaReadback.getMappedRange(0, bytes);
  const areaByCluster = new Float32Array(mapped.slice(0));
  state.areaProbeAreaReadback.unmap();

  const restByCluster = plan.clusterRestArea instanceof Float32Array ? plan.clusterRestArea : new Float32Array(clusterCount);
  const clusterErrors = new Float32Array(clusterCount);
  let maxAbsError = 0;
  let sumAbsError = 0;
  for (let ci = 0; ci < clusterCount; ci++) {
    const rest = restByCluster[ci] || 0;
    const err = areaByCluster[ci] - rest;
    clusterErrors[ci] = err;
    const abs = Math.abs(err);
    if (abs > maxAbsError) maxAbsError = abs;
    sumAbsError += abs;
  }

  state.lastAreaProbeClusterCount = clusterCount;
  state.lastAreaProbeEndpointCount = endpointCount;
  state.lastAreaProbeDispatch = dispatchCount;
  state.lastAreaProbeAreaByCluster = areaByCluster;
  state.lastAreaProbeConstraintErrorByCluster = clusterErrors;
  state.lastAreaProbeAbsErrorMean = clusterCount > 0 ? sumAbsError / clusterCount : 0;
  state.lastAreaProbeAbsErrorMax = maxAbsError;
  state.lastAreaProbeLambdaSeedByCluster = plan.clusterLambda;
  state.lastAreaProbeRestAreaByCluster = restByCluster;
  state.lastAreaProbeSoftAreaLambdaSize = Number(sim?.softAreaLambda?.size) || 0;
  return true;
}

function signedAreaPredicted(nodes, indices, dtPos) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    const ax = a.x + a.vx * dtPos;
    const ay = a.y + a.vy * dtPos;
    const bx = b.x + b.vx * dtPos;
    const by = b.y + b.vy * dtPos;
    s += ax * by - bx * ay;
  }
  return 0.5 * s;
}

function buildSoftAreaXpbdWgslPlan({ sim, soft, loops }) {
  const loopList = Array.isArray(loops) ? loops : [];
  const nodes = soft?.nodes || [];
  const clusterOffsets = new Uint32Array(loopList.length + 1);
  let endpointCount = 0;
  for (let li = 0; li < loopList.length; li++) {
    const len = Math.max(0, Number(loopList[li]?.indices?.length) || 0);
    endpointCount += len;
    clusterOffsets[li + 1] = endpointCount;
  }

  const clusterNodeIndices = new Uint32Array(endpointCount);
  const clusterNodeInvMass = new Float32Array(endpointCount);
  const endpointClusterIndex = new Uint32Array(endpointCount);
  const clusterRestArea = new Float32Array(loopList.length);
  const clusterLambda = new Float32Array(loopList.length);

  let write = 0;
  for (let li = 0; li < loopList.length; li++) {
    const loop = loopList[li];
    const ids = Array.isArray(loop?.indices) ? loop.indices : [];
    const cid = loop?.clusterId;
    clusterRestArea[li] = Number(sim?.softAreaRest?.get?.(cid)) || 0;
    clusterLambda[li] = Number(sim?.softAreaLambda?.get?.(cid)) || 0;

    for (let k = 0; k < ids.length; k++) {
      const ni = Number(ids[k]) || 0;
      clusterNodeIndices[write] = ni;
      endpointClusterIndex[write] = li;
      const node = nodes[ni];
      clusterNodeInvMass[write] = 1 / Math.max(0.02, Number(node?.mass) || 1);
      write += 1;
    }
  }

  return {
    clusterCount: loopList.length,
    endpointCount,
    clusterOffsets,
    clusterNodeIndices,
    clusterNodeInvMass,
    endpointClusterIndex,
    clusterRestArea,
    clusterLambda,
  };
}

function buildSoftAreaXpbdWgslLayout(plan) {
  return {
    clusterOffsets: plan.clusterOffsets,
    clusterNodeIndices: plan.clusterNodeIndices,
    clusterNodeInvMass: plan.clusterNodeInvMass,
    endpointClusterIndex: plan.endpointClusterIndex,
    clusterRestArea: plan.clusterRestArea,
    clusterLambda: plan.clusterLambda,
    byteLength:
      plan.clusterOffsets.byteLength
      + plan.clusterNodeIndices.byteLength
      + plan.clusterNodeInvMass.byteLength
      + plan.endpointClusterIndex.byteLength
      + plan.clusterRestArea.byteLength
      + plan.clusterLambda.byteLength,
  };
}

export function applySoftAreaXPBDVelocityGpuOnly({
  sim,
  soft,
  loops,
  dtPos,
  stiffnessScale,
  softAreaXpbdIters,
  softAreaBaseCompliance,
  wgslOffload,
}) {
  if (!loops?.length) return;
  const alpha = (softAreaBaseCompliance / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    // WGSL prep ownership for area-XPBD offload: deterministic CSR loop layout
    // plus per-cluster rest/lambda seeds shared by probe + future solve kernels.
    const plan = buildSoftAreaXpbdWgslPlan({ sim, soft, loops });
    const layout = buildSoftAreaXpbdWgslLayout(plan);
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.lastPreparedClusterCount = plan.clusterCount;
    wgslOffload.state.lastPreparedEndpointCount = plan.endpointCount;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared';

    // Concrete WGSL area stages: per-cluster area probe + lambda proposal.
    // CPU remains authoritative for velocity updates until reduction kernels
    // land; these kernels validate deterministic area/lambda parity telemetry.
    if (canUseWgslOffload(wgslOffload)) {
      void (async () => {
        const [probeRan, proposalRan] = await Promise.all([
          dispatchSoftAreaWgslProbe({ sim, soft, offload: wgslOffload, plan, dtPos }),
          dispatchSoftAreaWgslLambdaProposal({ soft, offload: wgslOffload, plan, dtPos, alpha }),
        ]);

        let velocityProposalRan = false;
        if (proposalRan) {
          velocityProposalRan = await dispatchSoftAreaWgslVelocityDeltaProposal({
            soft,
            offload: wgslOffload,
            plan,
            dtPos,
            deltaByCluster: wgslOffload.state.lastAreaProposalDeltaLambdaByCluster,
          });
        }

        if (probeRan || proposalRan || velocityProposalRan) {
          wgslOffload.state.lastError = null;
          wgslOffload.state.lastMode = velocityProposalRan
            ? 'wgsl-velocity-proposal'
            : (proposalRan ? 'wgsl-proposal' : 'wgsl-probe');
        }
      })().catch((err) => {
        wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
        wgslOffload.state.lastMode = 'cpu-fallback';
      });
    }
  }

  for (let iter = 0; iter < softAreaXpbdIters; iter++) {
    for (const loop of loops) {
      const ids = loop.indices;
      const m = ids.length;
      if (m < 3) continue;
      const restArea = sim.softAreaRest.get(loop.clusterId);
      if (!Number.isFinite(restArea)) continue;

      const area = signedAreaPredicted(soft.nodes, ids, dtPos);
      const C = area - restArea;

      const gradX = new Array(m);
      const gradY = new Array(m);
      let sumWGrad2 = 0;

      for (let k = 0; k < m; k++) {
        const prev = soft.nodes[ids[(k - 1 + m) % m]];
        const next = soft.nodes[ids[(k + 1) % m]];
        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nx = next.x + next.vx * dtPos;
        const ny = next.y + next.vy * dtPos;
        const gx = 0.5 * (ny - py);
        const gy = 0.5 * (px - nx);
        gradX[k] = gx;
        gradY[k] = gy;

        const node = soft.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        sumWGrad2 += w * (gx * gx + gy * gy);
      }

      if (sumWGrad2 <= 1e-10) continue;

      const lambdaPrev = Number(sim.softAreaLambda.get(loop.clusterId)) || 0;
      let dl = (-C - alpha * lambdaPrev) / (sumWGrad2 + alpha);
      if (!Number.isFinite(dl)) continue;
      dl = Math.max(-2.0, Math.min(2.0, dl));
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      sim.softAreaLambda.set(loop.clusterId, lambdaNext);

      for (let k = 0; k < m; k++) {
        const node = soft.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        node.vx += (w * gradX[k] * dl) / dtPos;
        node.vy += (w * gradY[k] * dl) / dtPos;
      }
    }
  }
}
