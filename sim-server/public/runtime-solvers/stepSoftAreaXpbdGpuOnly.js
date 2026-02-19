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

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= params.clusterCount) { return; }

  let start = clusterOffsets[ci];
  let end = clusterOffsets[ci + 1u];
  if (end <= start + 1u) {
    deltaLambdaOut[ci] = 0.0;
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
@group(0) @binding(10) var<storage, read_write> endpointDeltaVOut: array<vec2<f32>>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ei = gid.x;
  if (ei >= params.endpointCount) { return; }

  let ci = endpointClusterIndex[ei];
  if (ci >= params.clusterCount) {
    endpointDeltaVOut[ei] = vec2<f32>(0.0, 0.0);
    return;
  }

  let start = clusterOffsets[ci];
  let end = clusterOffsets[ci + 1u];
  if (end <= start + 1u || params.dtPos <= 1e-8) {
    endpointDeltaVOut[ei] = vec2<f32>(0.0, 0.0);
    return;
  }

  let ni = clusterNodeIndices[ei];
  if (ni >= params.nodeCount) {
    endpointDeltaVOut[ei] = vec2<f32>(0.0, 0.0);
    return;
  }

  let prevEi = select(end - 1u, ei - 1u, ei > start);
  let nextEi = select(start, ei + 1u, (ei + 1u) < end);
  let pi = clusterNodeIndices[prevEi];
  let ni2 = clusterNodeIndices[nextEi];
  if (pi >= params.nodeCount || ni2 >= params.nodeCount) {
    endpointDeltaVOut[ei] = vec2<f32>(0.0, 0.0);
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
  endpointDeltaVOut[ei] = vec2<f32>((w * gx * dl) / params.dtPos, (w * gy * dl) / params.dtPos);
}
`;


const softAreaVelocityNodeReductionWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  endpointCount: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> endpointNodeIndices: array<u32>;
@group(0) @binding(2) var<storage, read> endpointDeltaV: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> nodeDeltaVXOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> nodeDeltaVYOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> nodeContributionCountOut: array<u32>;

@compute @workgroup_size(
${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ni = gid.x;
  if (ni >= params.nodeCount) { return; }

  var sumX = 0.0;
  var sumY = 0.0;
  var count = 0u;
  for (var ei = 0u; ei < params.endpointCount; ei = ei + 1u) {
    if (endpointNodeIndices[ei] == ni) {
      let dv = endpointDeltaV[ei];
      sumX = sumX + dv.x;
      sumY = sumY + dv.y;
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

function getGpuOnlyPipelineModeProfile(wgslOffload) {
  const modeProfile = String(wgslOffload?.modeProfile || '').trim().toLowerCase();
  if (modeProfile === 'gpu-only-fast') return 'gpu-only-fast';
  if (modeProfile === 'gpu-only-validated') return 'gpu-only-validated';
  return 'standard';
}

function isGpuOnlyFastMode(wgslOffload) {
  return getGpuOnlyPipelineModeProfile(wgslOffload) === 'gpu-only-fast';
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
    state.areaLambdaDeltaReadback?.destroy?.();
    state.areaLambdaClusterOffsets = device.createBuffer({ size: (capacity + 1) * 4, usage: storageUsage });
    state.areaLambdaClusterRestArea = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaClusterPrev = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaLambdaDeltaOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaLambdaDeltaReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
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
    const scalarBytes = capacity * 4;
    const vec2Bytes = capacity * 8;
    state.areaVelocityClusterNodeIndices?.destroy?.();
    state.areaVelocityClusterNodeInvMass?.destroy?.();
    state.areaVelocityEndpointClusterIndex?.destroy?.();
    state.areaVelocityDeltaVOut?.destroy?.();
    state.areaVelocityDeltaVReadback?.destroy?.();
    state.areaVelocityClusterNodeIndices = device.createBuffer({ size: scalarBytes, usage: storageUsage });
    state.areaVelocityClusterNodeInvMass = device.createBuffer({ size: scalarBytes, usage: storageUsage });
    state.areaVelocityEndpointClusterIndex = device.createBuffer({ size: scalarBytes, usage: storageUsage });
    state.areaVelocityDeltaVOut = device.createBuffer({ size: vec2Bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaVelocityDeltaVReadback = device.createBuffer({ size: vec2Bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
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



function ensureSoftAreaVelocityNodeReductionBuffers(offload, nodeCount, endpointCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.areaVelocityNodeReductionNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.areaVelocityNodeReductionNodeCapacity ? state.areaVelocityNodeReductionNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaVelocityNodeReductionDeltaVXOut?.destroy?.();
    state.areaVelocityNodeReductionDeltaVYOut?.destroy?.();
    state.areaVelocityNodeReductionContributionOut?.destroy?.();
    state.areaVelocityNodeReductionDeltaVXReadback?.destroy?.();
    state.areaVelocityNodeReductionDeltaVYReadback?.destroy?.();
    state.areaVelocityNodeReductionContributionReadback?.destroy?.();
    state.areaVelocityNodeReductionDeltaVXOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaVelocityNodeReductionDeltaVYOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaVelocityNodeReductionContributionOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaVelocityNodeReductionDeltaVXReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaVelocityNodeReductionDeltaVYReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaVelocityNodeReductionContributionReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaVelocityNodeReductionNodeCapacity = capacity;
    state.areaVelocityNodeReductionBindGroup = null;
  }

  const requiredEndpointCapacity = Math.max(1, endpointCount);
  if ((state.areaVelocityNodeReductionEndpointCapacity || 0) < requiredEndpointCapacity) {
    const capacity = Math.max(requiredEndpointCapacity, state.areaVelocityNodeReductionEndpointCapacity ? state.areaVelocityNodeReductionEndpointCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaVelocityNodeReductionEndpointNodeIndices?.destroy?.();
    state.areaVelocityNodeReductionEndpointNodeIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaVelocityNodeReductionEndpointCapacity = capacity;
    state.areaVelocityNodeReductionBindGroup = null;
  }

  if (!state.areaVelocityNodeReductionParams) {
    state.areaVelocityNodeReductionParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.areaVelocityNodeReductionBindGroup = null;
  }

  return state;
}

async function dispatchSoftAreaWgslVelocityNodeReduction({ soft, offload, plan }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const endpointCount = Number(plan?.endpointCount) || 0;
  if (nodes.length <= 0 || endpointCount <= 0) return false;

  const state = ensureSoftAreaVelocityNodeReductionBuffers(offload, nodes.length, endpointCount);
  const device = offload.device;

  if (!state.areaVelocityNodeReductionPipeline) {
    const module = device.createShaderModule({ code: softAreaVelocityNodeReductionWgsl });
    state.areaVelocityNodeReductionPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.areaVelocityNodeReductionBindGroup = null;
  }

  if (!state.areaVelocityNodeReductionBindGroup) {
    state.areaVelocityNodeReductionBindGroup = device.createBindGroup({
      layout: state.areaVelocityNodeReductionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.areaVelocityNodeReductionParams } },
        { binding: 1, resource: { buffer: state.areaVelocityNodeReductionEndpointNodeIndices } },
        { binding: 2, resource: { buffer: state.areaVelocityDeltaVOut } },
        { binding: 3, resource: { buffer: state.areaVelocityNodeReductionDeltaVXOut } },
        { binding: 4, resource: { buffer: state.areaVelocityNodeReductionDeltaVYOut } },
        { binding: 5, resource: { buffer: state.areaVelocityNodeReductionContributionOut } },
      ],
    });
  }

  const paramsBytes = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsBytes);
  paramsU32[0] = nodes.length >>> 0;
  paramsU32[1] = endpointCount >>> 0;

  device.queue.writeBuffer(state.areaVelocityNodeReductionParams, 0, paramsBytes);
  device.queue.writeBuffer(state.areaVelocityNodeReductionEndpointNodeIndices, 0, plan.clusterNodeIndices);

  const nodeBytes = nodes.length * 4;
  const dispatchCount = Math.ceil(nodes.length / WGSL_WORKGROUP_SIZE);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.areaVelocityNodeReductionPipeline);
  pass.setBindGroup(0, state.areaVelocityNodeReductionBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();

  encoder.copyBufferToBuffer(state.areaVelocityNodeReductionDeltaVXOut, 0, state.areaVelocityNodeReductionDeltaVXReadback, 0, nodeBytes);
  encoder.copyBufferToBuffer(state.areaVelocityNodeReductionDeltaVYOut, 0, state.areaVelocityNodeReductionDeltaVYReadback, 0, nodeBytes);
  encoder.copyBufferToBuffer(state.areaVelocityNodeReductionContributionOut, 0, state.areaVelocityNodeReductionContributionReadback, 0, nodeBytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    state.areaVelocityNodeReductionDeltaVXReadback.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes),
    state.areaVelocityNodeReductionDeltaVYReadback.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes),
    state.areaVelocityNodeReductionContributionReadback.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes),
  ]);

  const mappedVx = state.areaVelocityNodeReductionDeltaVXReadback.getMappedRange(0, nodeBytes);
  const deltaVxByNode = new Float32Array(mappedVx.slice(0));
  state.areaVelocityNodeReductionDeltaVXReadback.unmap();

  const mappedVy = state.areaVelocityNodeReductionDeltaVYReadback.getMappedRange(0, nodeBytes);
  const deltaVyByNode = new Float32Array(mappedVy.slice(0));
  state.areaVelocityNodeReductionDeltaVYReadback.unmap();

  const mappedContribution = state.areaVelocityNodeReductionContributionReadback.getMappedRange(0, nodeBytes);
  const contributionCountByNode = new Uint32Array(mappedContribution.slice(0));
  state.areaVelocityNodeReductionContributionReadback.unmap();

  let maxAbsDelta = 0;
  let sumAbsDelta = 0;
  for (let ni = 0; ni < nodes.length; ni++) {
    const abs = Math.hypot(deltaVxByNode[ni], deltaVyByNode[ni]);
    if (abs > maxAbsDelta) maxAbsDelta = abs;
    sumAbsDelta += abs;
  }

  state.lastAreaVelocityProposalNodeReductionDispatch = dispatchCount;
  state.lastAreaVelocityProposalNodeDeltaVx = deltaVxByNode;
  state.lastAreaVelocityProposalNodeDeltaVy = deltaVyByNode;
  state.lastAreaVelocityProposalNodeContributionCount = contributionCountByNode;
  state.lastAreaVelocityProposalNodeReductionAbsDeltaMean = nodes.length > 0 ? sumAbsDelta / nodes.length : 0;
  state.lastAreaVelocityProposalNodeReductionAbsDeltaMax = maxAbsDelta;
  return true;
}
async function dispatchSoftAreaWgslVelocityDeltaProposal({
  soft,
  offload,
  plan,
  dtPos,
  deltaByCluster,
  includeEndpointTelemetry = true,
}) {
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
        { binding: 10, resource: { buffer: state.areaVelocityDeltaVOut } },
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
  const packedBytes = endpointCount * 8;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.areaVelocityPipeline);
  pass.setBindGroup(0, state.areaVelocityBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  if (includeEndpointTelemetry) {
    encoder.copyBufferToBuffer(state.areaVelocityDeltaVOut, 0, state.areaVelocityDeltaVReadback, 0, packedBytes);
  }
  device.queue.submit([encoder.finish()]);

  state.lastAreaVelocityProposalEndpointCount = endpointCount;
  state.lastAreaVelocityProposalDispatch = dispatchCount;
  state.lastAreaVelocityProposalTelemetryMode = includeEndpointTelemetry ? 'full-readback' : 'reduced-readback';

  if (!includeEndpointTelemetry) {
    state.lastAreaVelocityProposalDeltaVxByEndpoint = null;
    state.lastAreaVelocityProposalDeltaVyByEndpoint = null;
    state.lastAreaVelocityProposalAbsDeltaMean = null;
    state.lastAreaVelocityProposalAbsDeltaMax = null;
    return true;
  }

  await state.areaVelocityDeltaVReadback.mapAsync(globalThis.GPUMapMode.READ, 0, packedBytes);
  const mappedPacked = state.areaVelocityDeltaVReadback.getMappedRange(0, packedBytes);
  const packedDelta = new Float32Array(mappedPacked.slice(0));
  state.areaVelocityDeltaVReadback.unmap();

  const deltaVxByEndpoint = new Float32Array(endpointCount);
  const deltaVyByEndpoint = new Float32Array(endpointCount);
  for (let ei = 0; ei < endpointCount; ei++) {
    const base = ei * 2;
    deltaVxByEndpoint[ei] = packedDelta[base] || 0;
    deltaVyByEndpoint[ei] = packedDelta[base + 1] || 0;
  }

  let maxAbsDelta = 0;
  let sumAbsDelta = 0;
  for (let ei = 0; ei < endpointCount; ei++) {
    const abs = Math.hypot(deltaVxByEndpoint[ei], deltaVyByEndpoint[ei]);
    if (abs > maxAbsDelta) maxAbsDelta = abs;
    sumAbsDelta += abs;
  }

  state.lastAreaVelocityProposalDeltaVxByEndpoint = deltaVxByEndpoint;
  state.lastAreaVelocityProposalDeltaVyByEndpoint = deltaVyByEndpoint;
  state.lastAreaVelocityProposalAbsDeltaMean = endpointCount > 0 ? sumAbsDelta / endpointCount : 0;
  state.lastAreaVelocityProposalAbsDeltaMax = maxAbsDelta;
  return true;
}

function reduceSoftAreaVelocityProposalToNodeDeltas({ soft, offload, plan }) {
  const state = offload?.state;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const endpointCount = Number(plan?.endpointCount) || 0;
  if (!state || nodes.length <= 0 || endpointCount <= 0) return false;

  const deltaVxByEndpoint = state.lastAreaVelocityProposalDeltaVxByEndpoint;
  const deltaVyByEndpoint = state.lastAreaVelocityProposalDeltaVyByEndpoint;
  if (!(deltaVxByEndpoint instanceof Float32Array) || !(deltaVyByEndpoint instanceof Float32Array)) return false;
  if (deltaVxByEndpoint.length < endpointCount || deltaVyByEndpoint.length < endpointCount) return false;

  const endpointNodeIndices = plan.clusterNodeIndices;
  if (!(endpointNodeIndices instanceof Uint32Array) || endpointNodeIndices.length < endpointCount) return false;

  const deltaVxByNode = new Float32Array(nodes.length);
  const deltaVyByNode = new Float32Array(nodes.length);
  const contributionCountByNode = new Uint32Array(nodes.length);

  let maxAbsDelta = 0;
  let sumAbsDelta = 0;
  for (let ei = 0; ei < endpointCount; ei++) {
    const ni = endpointNodeIndices[ei] >>> 0;
    if (ni >= nodes.length) continue;
    const dvx = deltaVxByEndpoint[ei] || 0;
    const dvy = deltaVyByEndpoint[ei] || 0;
    deltaVxByNode[ni] += dvx;
    deltaVyByNode[ni] += dvy;
    contributionCountByNode[ni] += 1;

    const abs = Math.hypot(dvx, dvy);
    if (abs > maxAbsDelta) maxAbsDelta = abs;
    sumAbsDelta += abs;
  }

  state.lastAreaVelocityProposalNodeDeltaVx = deltaVxByNode;
  state.lastAreaVelocityProposalNodeDeltaVy = deltaVyByNode;
  state.lastAreaVelocityProposalNodeContributionCount = contributionCountByNode;
  state.lastAreaVelocityProposalNodeReductionAbsDeltaMean = endpointCount > 0 ? sumAbsDelta / endpointCount : 0;
  state.lastAreaVelocityProposalNodeReductionAbsDeltaMax = maxAbsDelta;
  return true;
}

function buildSoftAreaVelocityNodeReference({ soft, plan, dtPos, deltaByCluster }) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const clusterCount = Number(plan?.clusterCount) || 0;
  const endpointCount = Number(plan?.endpointCount) || 0;
  if (nodes.length <= 0 || clusterCount <= 0 || endpointCount <= 0) return null;
  if (!(deltaByCluster instanceof Float32Array) || deltaByCluster.length < clusterCount) return null;

  const endpointDeltaVx = new Float32Array(endpointCount);
  const endpointDeltaVy = new Float32Array(endpointCount);
  const nodeDeltaVx = new Float32Array(nodes.length);
  const nodeDeltaVy = new Float32Array(nodes.length);
  const nodeContributionCount = new Uint32Array(nodes.length);
  if (!(Number.isFinite(dtPos) && dtPos > 1e-8)) {
    return { endpointDeltaVx, endpointDeltaVy, nodeDeltaVx, nodeDeltaVy, nodeContributionCount };
  }

  const invDtPos = 1 / dtPos;
  const offsets = plan.clusterOffsets;
  const nodeIndices = plan.clusterNodeIndices;
  const endpointCluster = plan.endpointClusterIndex;
  const invMass = plan.clusterNodeInvMass;

  for (let ei = 0; ei < endpointCount; ei++) {
    const ci = endpointCluster[ei] >>> 0;
    if (ci >= clusterCount) continue;
    const start = offsets[ci] >>> 0;
    const end = offsets[ci + 1] >>> 0;
    if (end <= start + 1 || ei < start || ei >= end) continue;

    const ni = nodeIndices[ei] >>> 0;
    const prevEi = ei > start ? (ei - 1) : (end - 1);
    const nextEi = (ei + 1) < end ? (ei + 1) : start;
    const pi = nodeIndices[prevEi] >>> 0;
    const ni2 = nodeIndices[nextEi] >>> 0;
    if (ni >= nodes.length || pi >= nodes.length || ni2 >= nodes.length) continue;

    const prev = nodes[pi] || {};
    const next = nodes[ni2] || {};
    const px = (Number(prev.x) || 0) + (Number(prev.vx) || 0) * dtPos;
    const py = (Number(prev.y) || 0) + (Number(prev.vy) || 0) * dtPos;
    const nx = (Number(next.x) || 0) + (Number(next.vx) || 0) * dtPos;
    const ny = (Number(next.y) || 0) + (Number(next.vy) || 0) * dtPos;
    const gx = 0.5 * (ny - py);
    const gy = 0.5 * (px - nx);

    const dl = deltaByCluster[ci] || 0;
    const w = invMass[ei] || 0;
    const dvx = w * gx * dl * invDtPos;
    const dvy = w * gy * dl * invDtPos;
    endpointDeltaVx[ei] = dvx;
    endpointDeltaVy[ei] = dvy;

    nodeDeltaVx[ni] += dvx;
    nodeDeltaVy[ni] += dvy;
    nodeContributionCount[ni] += 1;
  }

  return { endpointDeltaVx, endpointDeltaVy, nodeDeltaVx, nodeDeltaVy, nodeContributionCount };
}

function publishSoftAreaVelocityNodeParity({ offload, reference }) {
  const state = offload?.state;
  if (!state || !reference) return false;

  const actualVx = state.lastAreaVelocityProposalNodeDeltaVx;
  const actualVy = state.lastAreaVelocityProposalNodeDeltaVy;
  const actualCount = state.lastAreaVelocityProposalNodeContributionCount;
  if (!(actualVx instanceof Float32Array) || !(actualVy instanceof Float32Array) || !(actualCount instanceof Uint32Array)) return false;
  if (actualVx.length !== reference.nodeDeltaVx.length || actualVy.length !== reference.nodeDeltaVy.length || actualCount.length !== reference.nodeContributionCount.length) return false;

  let maxNodeDeltaError = 0;
  let sumNodeDeltaError = 0;
  let maxContributionError = 0;
  for (let ni = 0; ni < actualVx.length; ni++) {
    const dvxErr = Math.abs((actualVx[ni] || 0) - (reference.nodeDeltaVx[ni] || 0));
    const dvyErr = Math.abs((actualVy[ni] || 0) - (reference.nodeDeltaVy[ni] || 0));
    const pairErr = Math.max(dvxErr, dvyErr);
    if (pairErr > maxNodeDeltaError) maxNodeDeltaError = pairErr;
    sumNodeDeltaError += pairErr;

    const countErr = Math.abs((actualCount[ni] || 0) - (reference.nodeContributionCount[ni] || 0));
    if (countErr > maxContributionError) maxContributionError = countErr;
  }

  state.lastAreaVelocityProposalNodeParityReferenceDeltaVx = reference.nodeDeltaVx;
  state.lastAreaVelocityProposalNodeParityReferenceDeltaVy = reference.nodeDeltaVy;
  state.lastAreaVelocityProposalNodeParityReferenceContributionCount = reference.nodeContributionCount;
  state.lastAreaVelocityProposalNodeParityMaxError = maxNodeDeltaError;
  state.lastAreaVelocityProposalNodeParityMeanError = actualVx.length > 0 ? (sumNodeDeltaError / actualVx.length) : 0;
  state.lastAreaVelocityProposalNodeContributionParityMaxError = maxContributionError;
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
  device.queue.submit([encoder.finish()]);

  await state.areaLambdaDeltaReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mappedDelta = state.areaLambdaDeltaReadback.getMappedRange(0, bytes);
  const deltaByCluster = new Float32Array(mappedDelta.slice(0));
  state.areaLambdaDeltaReadback.unmap();

  const nextByCluster = new Float32Array(clusterCount);

  let maxAbsDelta = 0;
  let sumAbsDelta = 0;
  for (let ci = 0; ci < clusterCount; ci++) {
    const dl = Number(deltaByCluster[ci]) || 0;
    const abs = Math.abs(dl);
    if (abs > maxAbsDelta) maxAbsDelta = abs;
    sumAbsDelta += abs;
    const lambdaPrev = Number(plan.clusterLambda?.[ci]) || 0;
    nextByCluster[ci] = Math.max(-20, Math.min(20, lambdaPrev + dl));
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

function computeSoftAreaProposalSignature({ soft, plan, dtPos, alpha, softAreaXpbdIters }) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  let acc = `${nodes.length}|${Number(plan?.clusterCount) || 0}|${Number(plan?.endpointCount) || 0}|${Number(softAreaXpbdIters) || 0}|${Number(dtPos) || 0}|${Number(alpha) || 0}`;
  for (let i = 0; i < Math.min(nodes.length, 128); i++) {
    const node = nodes[i] || {};
    acc += `|${Number(node.x) || 0},${Number(node.y) || 0},${Number(node.vx) || 0},${Number(node.vy) || 0}`;
  }
  if (plan?.clusterLambda instanceof Float32Array) {
    for (let i = 0; i < plan.clusterLambda.length; i++) acc += `|${plan.clusterLambda[i] || 0}`;
  }
  return acc;
}

function applySoftAreaAuthoritativeCachedProposal({ sim, soft, loops, state }) {
  if (!(state?.lastAreaProposalLambdaNextByCluster instanceof Float32Array)) return false;
  if (!(state?.lastAreaVelocityProposalNodeDeltaVx instanceof Float32Array)) return false;
  if (!(state?.lastAreaVelocityProposalNodeDeltaVy instanceof Float32Array)) return false;

  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  if (state.lastAreaProposalLambdaNextByCluster.length !== loops.length) return false;
  if (state.lastAreaVelocityProposalNodeDeltaVx.length !== nodes.length) return false;
  if (state.lastAreaVelocityProposalNodeDeltaVy.length !== nodes.length) return false;

  const finiteX = checkFiniteFloat32Array(state.lastAreaVelocityProposalNodeDeltaVx);
  const finiteY = checkFiniteFloat32Array(state.lastAreaVelocityProposalNodeDeltaVy);
  if (finiteX.allFinite !== true || finiteY.allFinite !== true) {
    state.lastAreaVelocityProposalFinite = {
      allFinite: false,
      nonFiniteCount: (finiteX.nonFiniteCount || 0) + (finiteY.nonFiniteCount || 0),
      comparedCount: (finiteX.comparedCount || 0) + (finiteY.comparedCount || 0),
    };
    return false;
  }

  for (let ci = 0; ci < loops.length; ci++) {
    sim?.softAreaLambda?.set?.(loops[ci]?.clusterId, Number(state.lastAreaProposalLambdaNextByCluster[ci]) || 0);
  }

  for (let ni = 0; ni < nodes.length; ni++) {
    nodes[ni].vx += Number(state.lastAreaVelocityProposalNodeDeltaVx[ni]) || 0;
    nodes[ni].vy += Number(state.lastAreaVelocityProposalNodeDeltaVy[ni]) || 0;
  }

  state.lastAreaVelocityProposalFinite = {
    allFinite: true,
    nonFiniteCount: 0,
    comparedCount: nodes.length * 2,
  };
  return true;
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
    wgslOffload.state.lastPipelineModeProfile = getGpuOnlyPipelineModeProfile(wgslOffload);
    wgslOffload.state.lastMode = 'cpu-prepared';

    const proposalSignature = computeSoftAreaProposalSignature({
      soft,
      plan,
      dtPos,
      alpha,
      softAreaXpbdIters,
    });
    const proposalEpoch = (Number(wgslOffload.state.lastPreparedProposalEpoch) || 0) + 1;
    wgslOffload.state.lastPreparedProposalEpoch = proposalEpoch;
    wgslOffload.state.lastPreparedProposalSignature = proposalSignature;

    const fastModeAuthoritative = isGpuOnlyFastMode(wgslOffload);
    const canUseAuthoritativeReplay = wgslOffload?.authoritativeAreaXpbd === true || fastModeAuthoritative;
    const signatureMatched = wgslOffload.state.lastAreaVelocityProposalSignature === proposalSignature;
    const proposalEpochDelta = Math.max(0, proposalEpoch - (Number(wgslOffload.state.lastAreaVelocityProposalEpoch) || 0));
    const fastEpochReplayEligible = fastModeAuthoritative && proposalEpochDelta <= 2;
    const fastProposalMatch = fastModeAuthoritative ? (signatureMatched || fastEpochReplayEligible) : signatureMatched;
    if (
      canUseAuthoritativeReplay
      && fastProposalMatch
      && applySoftAreaAuthoritativeCachedProposal({
        sim,
        soft,
        loops,
        state: wgslOffload.state,
      })
    ) {
      const fastEpochReplayUsed = fastModeAuthoritative && !signatureMatched && fastEpochReplayEligible;
      wgslOffload.state.lastMode = fastModeAuthoritative
        ? 'wgsl-area-authoritative-fast'
        : 'wgsl-area-authoritative';
      wgslOffload.state.lastAuthoritativeProposalSignature = proposalSignature;
      wgslOffload.state.lastAuthoritativeProposalSource = fastEpochReplayUsed
        ? 'wgsl-area-proposal-fast-epoch-replay'
        : 'wgsl-area-proposal-signature-replay';
      wgslOffload.state.lastAuthoritativeProposalEpochDelta = proposalEpochDelta;
      wgslOffload.state.lastAuthoritativeProposalFrame = Number(sim?.frame) || 0;
      return;
    }
    if (canUseAuthoritativeReplay && fastProposalMatch && wgslOffload.state.lastAreaVelocityProposalFinite?.allFinite === false) {
      wgslOffload.state.lastMode = 'cpu-fallback';
      wgslOffload.state.lastError = 'non-finite-area-authoritative-replay';
    }

    // Concrete WGSL area stages: per-cluster area probe + lambda proposal.
    // CPU remains authoritative by default; optional authoritative replay can
    // consume a deterministic signature-matched WGSL proposal on the next frame.
    if (canUseWgslOffload(wgslOffload)) {
      const state = wgslOffload.state;
      const runWgslDispatch = async () => {
        const modeProfile = getGpuOnlyPipelineModeProfile(wgslOffload);
        const fastMode = modeProfile === 'gpu-only-fast';
        state.lastPipelineModeProfile = modeProfile;

        const probeRan = fastMode
          ? false
          : await dispatchSoftAreaWgslProbe({ sim, soft, offload: wgslOffload, plan, dtPos });
        state.lastAreaProbeSource = fastMode ? 'skipped-fast-mode' : 'wgsl-probe';

        const proposalRan = await dispatchSoftAreaWgslLambdaProposal({ soft, offload: wgslOffload, plan, dtPos, alpha });

        let velocityProposalRan = false;
        if (proposalRan) {
          velocityProposalRan = await dispatchSoftAreaWgslVelocityDeltaProposal({
            soft,
            offload: wgslOffload,
            plan,
            dtPos,
            deltaByCluster: state.lastAreaProposalDeltaLambdaByCluster,
            includeEndpointTelemetry: !fastMode,
          });
          if (velocityProposalRan) {
            const nodeReductionRan = await dispatchSoftAreaWgslVelocityNodeReduction({
              soft,
              offload: wgslOffload,
              plan,
            });

            state.lastAreaVelocityProposalSource = fastMode
              ? 'wgsl-node-reduction-fast'
              : 'wgsl-node-reduction';

            if (!nodeReductionRan) {
              if (fastMode) {
                const fallbackTelemetry = await dispatchSoftAreaWgslVelocityDeltaProposal({
                  soft,
                  offload: wgslOffload,
                  plan,
                  dtPos,
                  deltaByCluster: state.lastAreaProposalDeltaLambdaByCluster,
                  includeEndpointTelemetry: true,
                });
                if (fallbackTelemetry) {
                  reduceSoftAreaVelocityProposalToNodeDeltas({
                    soft,
                    offload: wgslOffload,
                    plan,
                  });
                  state.lastAreaVelocityProposalSource = 'cpu-deterministic-reduction-fallback';
                }
              } else {
                reduceSoftAreaVelocityProposalToNodeDeltas({
                  soft,
                  offload: wgslOffload,
                  plan,
                });
                state.lastAreaVelocityProposalSource = 'cpu-deterministic-reduction';
              }
            }

            const deltaFinite = checkFiniteFloat32Array(state.lastAreaVelocityProposalNodeDeltaVx);
            const deltaFiniteY = checkFiniteFloat32Array(state.lastAreaVelocityProposalNodeDeltaVy);
            state.lastAreaVelocityProposalFinite = {
              allFinite: deltaFinite.allFinite === true && deltaFiniteY.allFinite === true,
              nonFiniteCount: (deltaFinite.nonFiniteCount || 0) + (deltaFiniteY.nonFiniteCount || 0),
              comparedCount: (deltaFinite.comparedCount || 0) + (deltaFiniteY.comparedCount || 0),
            };

            if (fastMode) {
              state.lastAreaVelocityProposalParity = {
                source: state.lastAreaVelocityProposalSource,
                validation: 'skipped-cpu-parity',
                finite: state.lastAreaVelocityProposalFinite,
              };
            } else {
              const nodeReference = buildSoftAreaVelocityNodeReference({
                soft,
                plan,
                dtPos,
                deltaByCluster: state.lastAreaProposalDeltaLambdaByCluster,
              });
              publishSoftAreaVelocityNodeParity({
                offload: wgslOffload,
                reference: nodeReference,
              });
              state.lastAreaVelocityProposalParity = {
                source: state.lastAreaVelocityProposalSource,
                validation: 'cpu-parity',
                maxNodeDeltaError: state.lastAreaVelocityProposalNodeParityMaxError,
                maxContributionError: state.lastAreaVelocityProposalNodeContributionParityMaxError,
              };
            }

            if (state.lastAreaVelocityProposalFinite?.allFinite !== true) {
              state.lastMode = 'cpu-fallback';
              state.lastError = 'non-finite-area-velocity-proposal';
              return false;
            }
          }
        }

        if (probeRan || proposalRan || velocityProposalRan) {
          state.lastError = null;
          state.lastMode = velocityProposalRan
            ? 'wgsl-velocity-proposal'
            : (proposalRan ? 'wgsl-proposal' : 'wgsl-probe');
          if (proposalRan && velocityProposalRan) {
            state.lastAreaVelocityProposalSignature = proposalSignature;
            state.lastAreaVelocityProposalEpoch = proposalEpoch;
          }
        }
        return probeRan || proposalRan || velocityProposalRan;
      };

      const serializedDispatch = (state.pendingSoftAreaWgslDispatchPromise || Promise.resolve())
        .catch(() => {})
        .then(() => runWgslDispatch())
        .catch((err) => {
          state.lastError = String(err?.message || err || 'unknown-error');
          state.lastMode = 'cpu-fallback';
          return false;
        });
      state.pendingSoftAreaWgslDispatchPromise = serializedDispatch;
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
