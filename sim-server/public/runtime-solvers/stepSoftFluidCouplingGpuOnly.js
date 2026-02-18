/**
 * GPU-only runtime soft fluid-coupling path.
 * Isolates soft-node fluid carry, cluster load redistribution, and
 * cluster rigid-motion projection from baseline loops in gpu-lab.js.
 */

function hashU32ArrayFnv1a(arr) {
  let hash = 0x811c9dc5;
  const len = Number(arr?.length) || 0;
  for (let i = 0; i < len; i++) {
    hash ^= (Number(arr[i]) || 0) >>> 0;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function hashF32ArrayFnv1a(arr) {
  const len = Number(arr?.length) || 0;
  const scratch = new ArrayBuffer(4);
  const asF32 = new Float32Array(scratch);
  const asU32 = new Uint32Array(scratch);
  let hash = 0x811c9dc5;
  for (let i = 0; i < len; i++) {
    asF32[0] = Number(arr[i]) || 0;
    hash ^= asU32[0] >>> 0;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
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

function checkFiniteUint32Array(values) {
  if (!(values instanceof Uint32Array)) return { allFinite: false, nonFiniteCount: 0, comparedCount: 0 };
  let nonFiniteCount = 0;
  for (let i = 0; i < values.length; i++) {
    const value = Number(values[i]);
    if (!Number.isFinite(value)) nonFiniteCount += 1;
  }
  return {
    allFinite: nonFiniteCount === 0,
    nonFiniteCount,
    comparedCount: values.length,
  };
}

function buildSoftFluidCouplingWgslLayout({ nodes, softNodeMomentumScale, softMembraneClusterSet }) {
  const nodeCount = Number(nodes?.length) || 0;
  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  const nodeVx = new Float32Array(nodeCount);
  const nodeVy = new Float32Array(nodeCount);
  const nodeMass = new Float32Array(nodeCount);
  const nodeMomentum = new Float32Array(nodeCount);
  const nodeClusterId = new Int32Array(nodeCount);
  const nodeIsMembraneCluster = new Uint32Array(nodeCount);

  const clusterCounts = new Map();
  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i] || {};
    const cid = Number.isFinite(Number(node.clusterId)) ? Math.floor(Number(node.clusterId)) : 0;
    nodeX[i] = Number(node.x) || 0;
    nodeY[i] = Number(node.y) || 0;
    nodeVx[i] = Number(node.vx) || 0;
    nodeVy[i] = Number(node.vy) || 0;
    nodeMass[i] = Math.max(0.02, Number(node.mass) || 0.02);
    nodeMomentum[i] = Number(softNodeMomentumScale(i)) || 0;
    nodeClusterId[i] = cid;
    nodeIsMembraneCluster[i] = softMembraneClusterSet.has(cid) ? 1 : 0;
    clusterCounts.set(cid, (clusterCounts.get(cid) || 0) + 1);
  }

  const sortedClusterIds = Array.from(clusterCounts.keys()).sort((a, b) => a - b);
  const clusterCount = sortedClusterIds.length;
  const clusterIds = new Int32Array(clusterCount);
  const clusterNodeOffsets = new Uint32Array(clusterCount + 1);
  const clusterNodeCount = new Uint32Array(clusterCount);
  const nodeClusterSlot = new Uint32Array(nodeCount);
  const clusterNodeIndices = new Uint32Array(nodeCount);

  const clusterSlotById = new Map();
  let offset = 0;
  for (let i = 0; i < clusterCount; i++) {
    const cid = sortedClusterIds[i];
    const count = Number(clusterCounts.get(cid)) || 0;
    clusterSlotById.set(cid, i);
    clusterIds[i] = cid;
    clusterNodeOffsets[i] = offset;
    clusterNodeCount[i] = count;
    offset += count;
  }
  clusterNodeOffsets[clusterCount] = offset;

  const clusterCursor = new Uint32Array(clusterCount);
  for (let i = 0; i < clusterCount; i++) clusterCursor[i] = clusterNodeOffsets[i];
  for (let i = 0; i < nodeCount; i++) {
    const cid = nodeClusterId[i];
    const slot = clusterSlotById.get(cid) ?? 0;
    nodeClusterSlot[i] = slot >>> 0;
    const outIndex = clusterCursor[slot]++;
    clusterNodeIndices[outIndex] = i >>> 0;
  }

  const layout = {
    nodeX,
    nodeY,
    nodeVx,
    nodeVy,
    nodeMass,
    nodeMomentum,
    nodeClusterId,
    nodeClusterSlot,
    nodeIsMembraneCluster,
    clusterIds,
    clusterNodeOffsets,
    clusterNodeCount,
    clusterNodeIndices,
  };

  const signatureSeed = [nodeCount, clusterCount, offset];
  let signature = hashU32ArrayFnv1a(signatureSeed);
  signature ^= hashF32ArrayFnv1a(nodeX);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(nodeY);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(nodeClusterId);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(clusterNodeOffsets);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(clusterNodeIndices);
  signature >>>= 0;

  const byteLength = Object.values(layout).reduce((sum, arr) => sum + (arr?.byteLength || 0), 0);
  return {
    layout,
    nodeCount,
    clusterCount,
    signature,
    byteLength,
  };
}

function buildSoftFluidCouplingCpuSampleLayout({
  sim,
  nodes,
  n,
  vxField,
  vyField,
  obstacleMask,
  bodyFeedbackPrevVx,
  bodyFeedbackPrevVy,
  selfFeedbackSuppression,
  computeSoftCentroid,
  computeSoftClusterKinematics,
  sampleFluidForBodyCoupling,
  localHoneyDrag,
}) {
  const nodeCount = Number(nodes?.length) || 0;
  const fluidSampleVx = new Float32Array(nodeCount);
  const fluidSampleVy = new Float32Array(nodeCount);
  const clusterLocalVx = new Float32Array(nodeCount);
  const clusterLocalVy = new Float32Array(nodeCount);
  const sampleDeltaVx = new Float32Array(nodeCount);
  const sampleDeltaVy = new Float32Array(nodeCount);
  const clusterCenterX = new Float32Array(nodeCount);
  const clusterCenterY = new Float32Array(nodeCount);
  const localHoney = new Float32Array(nodeCount);

  const centroid = computeSoftCentroid(nodes);
  const clusterKinematics = computeSoftClusterKinematics(nodes);

  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i] || {};
    const cid = node.clusterId ?? 0;
    const clusterKin = clusterKinematics.get(cid);
    const clusterX = Number.isFinite(Number(clusterKin?.x)) ? Number(clusterKin.x) : centroid.x;
    const clusterY = Number.isFinite(Number(clusterKin?.y)) ? Number(clusterKin.y) : centroid.y;
    const vx = Number.isFinite(Number(clusterKin?.vx)) ? Number(clusterKin.vx) : 0;
    const vy = Number.isFinite(Number(clusterKin?.vy)) ? Number(clusterKin.vy) : 0;
    const omega = Number.isFinite(Number(clusterKin?.omega)) ? Number(clusterKin.omega) : 0;
    const rx = (Number(node.x) || 0) - clusterX;
    const ry = (Number(node.y) || 0) - clusterY;

    const fx = Number(sampleFluidForBodyCoupling(
      vxField,
      n,
      Number(node.x) || 0,
      Number(node.y) || 0,
      rx,
      ry,
      obstacleMask,
      bodyFeedbackPrevVx,
      selfFeedbackSuppression,
    )) || 0;
    const fy = Number(sampleFluidForBodyCoupling(
      vyField,
      n,
      Number(node.x) || 0,
      Number(node.y) || 0,
      rx,
      ry,
      obstacleMask,
      bodyFeedbackPrevVy,
      selfFeedbackSuppression,
    )) || 0;
    const localVx = vx - omega * ry;
    const localVy = vy + omega * rx;

    fluidSampleVx[i] = fx;
    fluidSampleVy[i] = fy;
    clusterLocalVx[i] = localVx;
    clusterLocalVy[i] = localVy;
    sampleDeltaVx[i] = fx - localVx;
    sampleDeltaVy[i] = fy - localVy;
    clusterCenterX[i] = clusterX;
    clusterCenterY[i] = clusterY;
    localHoney[i] = Number(localHoneyDrag?.(Number(node.x) || 0, Number(node.y) || 0)) || 0;
  }

  const layout = {
    fluidSampleVx,
    fluidSampleVy,
    clusterLocalVx,
    clusterLocalVy,
    sampleDeltaVx,
    sampleDeltaVy,
    clusterCenterX,
    clusterCenterY,
    localHoney,
  };
  const signatureSeed = [nodeCount, Number(sim?.frame) || 0, Number(n) || 0];
  let signature = hashU32ArrayFnv1a(signatureSeed);
  signature ^= hashF32ArrayFnv1a(fluidSampleVx);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(fluidSampleVy);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(sampleDeltaVx);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(sampleDeltaVy);
  signature >>>= 0;

  const byteLength = Object.values(layout).reduce((sum, arr) => sum + (arr?.byteLength || 0), 0);
  return {
    layout,
    nodeCount,
    signature,
    byteLength,
  };
}

function hasAuthoritativeWgslCarryProposal(wgslOffload, proposalSignature, nodeCount) {
  const state = wgslOffload?.state;
  if (!state) return false;
  return state.lastCarryProposalSignature === proposalSignature
    && state.lastCarryProposalForceX instanceof Float32Array
    && state.lastCarryProposalForceY instanceof Float32Array
    && state.lastCarryProposalCarryX instanceof Float32Array
    && state.lastCarryProposalCarryY instanceof Float32Array
    && state.lastCarryProposalLocalCarryX instanceof Float32Array
    && state.lastCarryProposalLocalCarryY instanceof Float32Array
    && state.lastCarryProposalForceX.length === nodeCount
    && state.lastCarryProposalForceY.length === nodeCount
    && state.lastCarryProposalCarryX.length === nodeCount
    && state.lastCarryProposalCarryY.length === nodeCount
    && state.lastCarryProposalLocalCarryX.length === nodeCount
    && state.lastCarryProposalLocalCarryY.length === nodeCount;
}

function hasAuthoritativeWgslClusterLoadProposal(wgslOffload, proposalSignature, clusterCount) {
  const state = wgslOffload?.state;
  if (!state) return false;
  return state.lastClusterLoadProposalSignature === proposalSignature
    && state.lastClusterLoadProposalForceX instanceof Float32Array
    && state.lastClusterLoadProposalForceY instanceof Float32Array
    && state.lastClusterLoadProposalTorque instanceof Float32Array
    && state.lastClusterLoadProposalCount instanceof Uint32Array
    && state.lastClusterLoadProposalForceX.length === clusterCount
    && state.lastClusterLoadProposalForceY.length === clusterCount
    && state.lastClusterLoadProposalTorque.length === clusterCount
    && state.lastClusterLoadProposalCount.length === clusterCount;
}

function buildCarryProposalFiniteSummary({ forceX, forceY, carryX, carryY, localCarryX, localCarryY }) {
  const forceXFinite = checkFiniteFloat32Array(forceX);
  const forceYFinite = checkFiniteFloat32Array(forceY);
  const carryXFinite = checkFiniteFloat32Array(carryX);
  const carryYFinite = checkFiniteFloat32Array(carryY);
  const localCarryXFinite = checkFiniteFloat32Array(localCarryX);
  const localCarryYFinite = checkFiniteFloat32Array(localCarryY);
  return {
    allFinite: forceXFinite.allFinite
      && forceYFinite.allFinite
      && carryXFinite.allFinite
      && carryYFinite.allFinite
      && localCarryXFinite.allFinite
      && localCarryYFinite.allFinite,
    forceX: forceXFinite,
    forceY: forceYFinite,
    carryX: carryXFinite,
    carryY: carryYFinite,
    localCarryX: localCarryXFinite,
    localCarryY: localCarryYFinite,
  };
}

const SOFT_FLUID_CARRY_PROPOSAL_WGSL = /* wgsl */`
struct Params {
  nodeCount: u32,
  _pad0: vec3<u32>,
  dragK: f32,
  nodeFlowCoupling: f32,
  localFlowShare: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sampleVx: array<f32>;
@group(0) @binding(2) var<storage, read> sampleVy: array<f32>;
@group(0) @binding(3) var<storage, read> nodeVx: array<f32>;
@group(0) @binding(4) var<storage, read> nodeVy: array<f32>;
@group(0) @binding(5) var<storage, read> sampleDeltaVx: array<f32>;
@group(0) @binding(6) var<storage, read> sampleDeltaVy: array<f32>;
@group(0) @binding(7) var<storage, read> mass: array<f32>;
@group(0) @binding(8) var<storage, read> momentum: array<f32>;
@group(0) @binding(9) var<storage, read> isMembraneCluster: array<u32>;
@group(0) @binding(10) var<storage, read> honey: array<f32>;
@group(0) @binding(11) var<storage, read_write> outForceCarryPacked: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> outLocalCarryPacked: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.nodeCount) {
    return;
  }

  let m = max(0.02, mass[i]);
  let invM = 1.0 / m;
  let membraneScale = select(1.0, 0.88, isMembraneCluster[i] > 0u);
  let flowCoupling = params.nodeFlowCoupling * membraneScale * momentum[i];
  let dragHoney = params.dragK * honey[i] * flowCoupling;

  let fx = sampleDeltaVx[i] * dragHoney * m;
  let fy = sampleDeltaVy[i] * dragHoney * m;
  let carryX = fx * invM;
  let carryY = fy * invM;
  outForceCarryPacked[i] = vec4<f32>(fx, fy, carryX, carryY);

  let localCarryX = (sampleVx[i] - nodeVx[i]) * dragHoney * invM * params.localFlowShare;
  let localCarryY = (sampleVy[i] - nodeVy[i]) * dragHoney * invM * params.localFlowShare;
  outLocalCarryPacked[i] = vec2<f32>(localCarryX, localCarryY);
}
`;

const SOFT_FLUID_CLUSTER_LOAD_REDUCTION_WGSL = /* wgsl */`
struct Params {
  clusterCount: u32,
  _pad0: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> clusterNodeOffsets: array<u32>;
@group(0) @binding(2) var<storage, read> clusterNodeIndices: array<u32>;
@group(0) @binding(3) var<storage, read> nodeX: array<f32>;
@group(0) @binding(4) var<storage, read> nodeY: array<f32>;
@group(0) @binding(5) var<storage, read> clusterCenterX: array<f32>;
@group(0) @binding(6) var<storage, read> clusterCenterY: array<f32>;
@group(0) @binding(7) var<storage, read> forceX: array<f32>;
@group(0) @binding(8) var<storage, read> forceY: array<f32>;
@group(0) @binding(9) var<storage, read_write> outClusterForceX: array<f32>;
@group(0) @binding(10) var<storage, read_write> outClusterForceY: array<f32>;
@group(0) @binding(11) var<storage, read_write> outClusterTorque: array<f32>;
@group(0) @binding(12) var<storage, read_write> outClusterCount: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= params.clusterCount) {
    return;
  }

  let begin = clusterNodeOffsets[ci];
  let end = clusterNodeOffsets[ci + 1u];
  var sumFx = 0.0;
  var sumFy = 0.0;
  var sumTorque = 0.0;
  var count = 0u;

  for (var idx = begin; idx < end; idx = idx + 1u) {
    let ni = clusterNodeIndices[idx];
    let fx = forceX[ni];
    let fy = forceY[ni];
    let rx = nodeX[ni] - clusterCenterX[ni];
    let ry = nodeY[ni] - clusterCenterY[ni];
    sumFx = sumFx + fx;
    sumFy = sumFy + fy;
    sumTorque = sumTorque + (rx * fy - ry * fx);
    count = count + 1u;
  }

  outClusterForceX[ci] = sumFx;
  outClusterForceY[ci] = sumFy;
  outClusterTorque[ci] = sumTorque;
  outClusterCount[ci] = count;
}
`;

function writeFloatArrayToBuffer(device, buffer, arr) {
  if (!device?.queue || !buffer || !(arr instanceof Float32Array)) return;
  device.queue.writeBuffer(buffer, 0, arr);
}

function writeUintArrayToBuffer(device, buffer, arr) {
  if (!device?.queue || !buffer || !(arr instanceof Uint32Array)) return;
  device.queue.writeBuffer(buffer, 0, arr);
}

function ensureSoftFluidCarryProposalResources(state, device, nodeCount) {
  if (!state || !device || !Number.isFinite(nodeCount) || nodeCount <= 0) return null;
  if (typeof GPUBufferUsage === 'undefined' || typeof GPUMapMode === 'undefined') return null;

  const capacity = Math.max(1, Number(nodeCount) | 0);
  const bytes = capacity * Float32Array.BYTES_PER_ELEMENT;
  const u32Bytes = capacity * Uint32Array.BYTES_PER_ELEMENT;
  const recreate = !state.carryProposalResources || (Number(state.carryProposalResources.capacity) || 0) < capacity;

  if (!recreate) return state.carryProposalResources;

  state.carryProposalResources = null;

  const createStorage = (label, byteSize) => device.createBuffer({
    label,
    size: Math.max(4, byteSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const createOutputStorage = (label, byteSize) => device.createBuffer({
    label,
    size: Math.max(4, byteSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const createReadback = (label, byteSize) => device.createBuffer({
    label,
    size: Math.max(4, byteSize),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const module = device.createShaderModule({
    label: 'soft-fluid-carry-proposal-wgsl',
    code: SOFT_FLUID_CARRY_PROPOSAL_WGSL,
  });
  const pipeline = device.createComputePipeline({
    label: 'soft-fluid-carry-proposal-pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  });

  const packedForceCarryBytes = capacity * 4 * Float32Array.BYTES_PER_ELEMENT;
  const packedLocalCarryBytes = capacity * 2 * Float32Array.BYTES_PER_ELEMENT;

  const resources = {
    capacity,
    paramsBuffer: device.createBuffer({
      label: 'soft-fluid-carry-proposal-params',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    sampleVxBuffer: createStorage('soft-fluid-carry-sample-vx', bytes),
    sampleVyBuffer: createStorage('soft-fluid-carry-sample-vy', bytes),
    nodeVxBuffer: createStorage('soft-fluid-carry-node-vx', bytes),
    nodeVyBuffer: createStorage('soft-fluid-carry-node-vy', bytes),
    sampleDeltaVxBuffer: createStorage('soft-fluid-carry-delta-vx', bytes),
    sampleDeltaVyBuffer: createStorage('soft-fluid-carry-delta-vy', bytes),
    massBuffer: createStorage('soft-fluid-carry-mass', bytes),
    momentumBuffer: createStorage('soft-fluid-carry-momentum', bytes),
    membraneBuffer: createStorage('soft-fluid-carry-membrane', u32Bytes),
    honeyBuffer: createStorage('soft-fluid-carry-honey', bytes),
    outForceCarryPackedBuffer: createOutputStorage('soft-fluid-carry-out-force-carry-packed', packedForceCarryBytes),
    outLocalCarryPackedBuffer: createOutputStorage('soft-fluid-carry-out-local-carry-packed', packedLocalCarryBytes),
    readForceCarryPackedBuffer: createReadback('soft-fluid-carry-read-force-carry-packed', packedForceCarryBytes),
    readLocalCarryPackedBuffer: createReadback('soft-fluid-carry-read-local-carry-packed', packedLocalCarryBytes),
    pipeline,
    bindGroup: null,
  };

  resources.bindGroup = device.createBindGroup({
    label: 'soft-fluid-carry-proposal-bind-group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: resources.paramsBuffer } },
      { binding: 1, resource: { buffer: resources.sampleVxBuffer } },
      { binding: 2, resource: { buffer: resources.sampleVyBuffer } },
      { binding: 3, resource: { buffer: resources.nodeVxBuffer } },
      { binding: 4, resource: { buffer: resources.nodeVyBuffer } },
      { binding: 5, resource: { buffer: resources.sampleDeltaVxBuffer } },
      { binding: 6, resource: { buffer: resources.sampleDeltaVyBuffer } },
      { binding: 7, resource: { buffer: resources.massBuffer } },
      { binding: 8, resource: { buffer: resources.momentumBuffer } },
      { binding: 9, resource: { buffer: resources.membraneBuffer } },
      { binding: 10, resource: { buffer: resources.honeyBuffer } },
      { binding: 11, resource: { buffer: resources.outForceCarryPackedBuffer } },
      { binding: 12, resource: { buffer: resources.outLocalCarryPackedBuffer } },
    ],
  });

  state.carryProposalResources = resources;
  return resources;
}

function dispatchSoftFluidCarryProposalWgsl({ wgslOffload, nodeCount, dragK, nodeFlowCoupling, localFlowShare, proposalSignature }) {
  const state = wgslOffload?.state;
  const device = wgslOffload?.device;
  const prep = state?.preparedLayout;
  const sample = state?.preparedSampleLayout;
  if (!state || !device || !prep || !sample || !Number.isFinite(nodeCount) || nodeCount <= 0) return false;
  if (state.pendingCarryProposalPromise) return false;

  const resources = ensureSoftFluidCarryProposalResources(state, device, nodeCount);
  if (!resources) return false;

  const paramsBuffer = new ArrayBuffer(32);
  const paramsU32 = new Uint32Array(paramsBuffer);
  const paramsF32 = new Float32Array(paramsBuffer);
  paramsU32[0] = (nodeCount >>> 0);
  paramsF32[4] = Number(dragK) || 0;
  paramsF32[5] = Number(nodeFlowCoupling) || 0;
  paramsF32[6] = Number(localFlowShare) || 0;

  writeFloatArrayToBuffer(device, resources.sampleVxBuffer, sample.fluidSampleVx);
  writeFloatArrayToBuffer(device, resources.sampleVyBuffer, sample.fluidSampleVy);
  writeFloatArrayToBuffer(device, resources.nodeVxBuffer, prep.nodeVx);
  writeFloatArrayToBuffer(device, resources.nodeVyBuffer, prep.nodeVy);
  writeFloatArrayToBuffer(device, resources.sampleDeltaVxBuffer, sample.sampleDeltaVx);
  writeFloatArrayToBuffer(device, resources.sampleDeltaVyBuffer, sample.sampleDeltaVy);
  writeFloatArrayToBuffer(device, resources.massBuffer, prep.nodeMass);
  writeFloatArrayToBuffer(device, resources.momentumBuffer, prep.nodeMomentum);
  writeUintArrayToBuffer(device, resources.membraneBuffer, prep.nodeIsMembraneCluster);
  writeFloatArrayToBuffer(device, resources.honeyBuffer, sample.localHoney);
  device.queue.writeBuffer(resources.paramsBuffer, 0, paramsBuffer);

  const encoder = device.createCommandEncoder({ label: 'soft-fluid-carry-proposal-encoder' });
  const pass = encoder.beginComputePass({ label: 'soft-fluid-carry-proposal-pass' });
  pass.setPipeline(resources.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(nodeCount / 64)));
  pass.end();

  const packedForceCarryBytes = nodeCount * 4 * Float32Array.BYTES_PER_ELEMENT;
  const packedLocalCarryBytes = nodeCount * 2 * Float32Array.BYTES_PER_ELEMENT;
  encoder.copyBufferToBuffer(resources.outForceCarryPackedBuffer, 0, resources.readForceCarryPackedBuffer, 0, packedForceCarryBytes);
  encoder.copyBufferToBuffer(resources.outLocalCarryPackedBuffer, 0, resources.readLocalCarryPackedBuffer, 0, packedLocalCarryBytes);

  device.queue.submit([encoder.finish()]);

  const readback = async () => {
    await Promise.all([
      resources.readForceCarryPackedBuffer.mapAsync(GPUMapMode.READ, 0, packedForceCarryBytes),
      resources.readLocalCarryPackedBuffer.mapAsync(GPUMapMode.READ, 0, packedLocalCarryBytes),
    ]);

    let forceCarryPacked = null;
    let localCarryPacked = null;
    try {
      forceCarryPacked = new Float32Array(resources.readForceCarryPackedBuffer.getMappedRange(0, packedForceCarryBytes).slice(0));
      localCarryPacked = new Float32Array(resources.readLocalCarryPackedBuffer.getMappedRange(0, packedLocalCarryBytes).slice(0));
    } finally {
      resources.readForceCarryPackedBuffer.unmap();
      resources.readLocalCarryPackedBuffer.unmap();
    }

    const forceX = new Float32Array(nodeCount);
    const forceY = new Float32Array(nodeCount);
    const carryX = new Float32Array(nodeCount);
    const carryY = new Float32Array(nodeCount);
    const localCarryX = new Float32Array(nodeCount);
    const localCarryY = new Float32Array(nodeCount);

    for (let i = 0; i < nodeCount; i++) {
      const base4 = i * 4;
      const base2 = i * 2;
      forceX[i] = Number(forceCarryPacked[base4]) || 0;
      forceY[i] = Number(forceCarryPacked[base4 + 1]) || 0;
      carryX[i] = Number(forceCarryPacked[base4 + 2]) || 0;
      carryY[i] = Number(forceCarryPacked[base4 + 3]) || 0;
      localCarryX[i] = Number(localCarryPacked[base2]) || 0;
      localCarryY[i] = Number(localCarryPacked[base2 + 1]) || 0;
    }

    state.lastCarryProposalForceX = forceX;
    state.lastCarryProposalForceY = forceY;
    state.lastCarryProposalCarryX = carryX;
    state.lastCarryProposalCarryY = carryY;
    state.lastCarryProposalLocalCarryX = localCarryX;
    state.lastCarryProposalLocalCarryY = localCarryY;
    state.lastCarryProposalFinite = buildCarryProposalFiniteSummary({
      forceX,
      forceY,
      carryX,
      carryY,
      localCarryX,
      localCarryY,
    });
    state.lastCarryProposalSignature = Number(proposalSignature) >>> 0;
    state.lastCarryProposalSource = 'wgsl-carry-proposal';
    state.lastMode = 'wgsl-carry-proposal';
    state.lastSourceRoute = 'wgsl-carry-proposal';
  };

  state.pendingCarryProposalPromise = readback()
    .catch((err) => {
      state.lastCarryProposalError = String(err?.message || err);
    })
    .finally(() => {
      state.pendingCarryProposalPromise = null;
    });

  return true;
}

function ensureSoftFluidClusterLoadReductionResources(state, device, nodeCount, clusterCount) {
  if (!state || !device || !Number.isFinite(nodeCount) || !Number.isFinite(clusterCount) || nodeCount <= 0 || clusterCount <= 0) return null;
  if (typeof GPUBufferUsage === 'undefined' || typeof GPUMapMode === 'undefined') return null;

  const nodeCapacity = Math.max(1, Number(nodeCount) | 0);
  const clusterCapacity = Math.max(1, Number(clusterCount) | 0);
  const nodeBytes = nodeCapacity * Float32Array.BYTES_PER_ELEMENT;
  const nodeU32Bytes = nodeCapacity * Uint32Array.BYTES_PER_ELEMENT;
  const clusterBytes = clusterCapacity * Float32Array.BYTES_PER_ELEMENT;
  const clusterU32Bytes = clusterCapacity * Uint32Array.BYTES_PER_ELEMENT;
  const offsetBytes = (clusterCapacity + 1) * Uint32Array.BYTES_PER_ELEMENT;
  const recreate = !state.clusterLoadReductionResources
    || (Number(state.clusterLoadReductionResources.nodeCapacity) || 0) < nodeCapacity
    || (Number(state.clusterLoadReductionResources.clusterCapacity) || 0) < clusterCapacity;

  if (!recreate) return state.clusterLoadReductionResources;

  const createStorage = (label, byteSize) => device.createBuffer({
    label,
    size: Math.max(4, byteSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const createOutStorage = (label, byteSize) => device.createBuffer({
    label,
    size: Math.max(4, byteSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const createReadback = (label, byteSize) => device.createBuffer({
    label,
    size: Math.max(4, byteSize),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const module = device.createShaderModule({
    label: 'soft-fluid-cluster-load-reduction-wgsl',
    code: SOFT_FLUID_CLUSTER_LOAD_REDUCTION_WGSL,
  });
  const pipeline = device.createComputePipeline({
    label: 'soft-fluid-cluster-load-reduction-pipeline',
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });

  const resources = {
    nodeCapacity,
    clusterCapacity,
    paramsBuffer: device.createBuffer({
      label: 'soft-fluid-cluster-load-reduction-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    clusterNodeOffsetsBuffer: createStorage('soft-fluid-cluster-node-offsets', offsetBytes),
    clusterNodeIndicesBuffer: createStorage('soft-fluid-cluster-node-indices', nodeU32Bytes),
    nodeXBuffer: createStorage('soft-fluid-cluster-node-x', nodeBytes),
    nodeYBuffer: createStorage('soft-fluid-cluster-node-y', nodeBytes),
    clusterCenterXBuffer: createStorage('soft-fluid-cluster-center-x', nodeBytes),
    clusterCenterYBuffer: createStorage('soft-fluid-cluster-center-y', nodeBytes),
    forceXBuffer: createStorage('soft-fluid-cluster-force-x', nodeBytes),
    forceYBuffer: createStorage('soft-fluid-cluster-force-y', nodeBytes),
    outClusterForceXBuffer: createOutStorage('soft-fluid-cluster-out-force-x', clusterBytes),
    outClusterForceYBuffer: createOutStorage('soft-fluid-cluster-out-force-y', clusterBytes),
    outClusterTorqueBuffer: createOutStorage('soft-fluid-cluster-out-torque', clusterBytes),
    outClusterCountBuffer: createOutStorage('soft-fluid-cluster-out-count', clusterU32Bytes),
    readClusterForceXBuffer: createReadback('soft-fluid-cluster-read-force-x', clusterBytes),
    readClusterForceYBuffer: createReadback('soft-fluid-cluster-read-force-y', clusterBytes),
    readClusterTorqueBuffer: createReadback('soft-fluid-cluster-read-torque', clusterBytes),
    readClusterCountBuffer: createReadback('soft-fluid-cluster-read-count', clusterU32Bytes),
    pipeline,
    bindGroup: null,
  };

  resources.bindGroup = device.createBindGroup({
    label: 'soft-fluid-cluster-load-reduction-bind-group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: resources.paramsBuffer } },
      { binding: 1, resource: { buffer: resources.clusterNodeOffsetsBuffer } },
      { binding: 2, resource: { buffer: resources.clusterNodeIndicesBuffer } },
      { binding: 3, resource: { buffer: resources.nodeXBuffer } },
      { binding: 4, resource: { buffer: resources.nodeYBuffer } },
      { binding: 5, resource: { buffer: resources.clusterCenterXBuffer } },
      { binding: 6, resource: { buffer: resources.clusterCenterYBuffer } },
      { binding: 7, resource: { buffer: resources.forceXBuffer } },
      { binding: 8, resource: { buffer: resources.forceYBuffer } },
      { binding: 9, resource: { buffer: resources.outClusterForceXBuffer } },
      { binding: 10, resource: { buffer: resources.outClusterForceYBuffer } },
      { binding: 11, resource: { buffer: resources.outClusterTorqueBuffer } },
      { binding: 12, resource: { buffer: resources.outClusterCountBuffer } },
    ],
  });

  state.clusterLoadReductionResources = resources;
  return resources;
}

function dispatchSoftFluidClusterLoadReductionWgsl({ wgslOffload, proposalSignature, forceX, forceY }) {
  const state = wgslOffload?.state;
  const device = wgslOffload?.device;
  const prep = state?.preparedLayout;
  const sample = state?.preparedSampleLayout;
  const nodeCount = Number(prep?.nodeX?.length) || 0;
  const clusterCount = Number(prep?.clusterIds?.length) || 0;
  if (!state || !device || !prep || !sample || !(forceX instanceof Float32Array) || !(forceY instanceof Float32Array)) return false;
  if (forceX.length !== nodeCount || forceY.length !== nodeCount || nodeCount <= 0 || clusterCount <= 0) return false;
  if (state.pendingClusterLoadProposalPromise) return false;

  const resources = ensureSoftFluidClusterLoadReductionResources(state, device, nodeCount, clusterCount);
  if (!resources) return false;

  const params = new Uint32Array(4);
  params[0] = clusterCount >>> 0;
  device.queue.writeBuffer(resources.paramsBuffer, 0, params);
  writeUintArrayToBuffer(device, resources.clusterNodeOffsetsBuffer, prep.clusterNodeOffsets);
  writeUintArrayToBuffer(device, resources.clusterNodeIndicesBuffer, prep.clusterNodeIndices);
  writeFloatArrayToBuffer(device, resources.nodeXBuffer, prep.nodeX);
  writeFloatArrayToBuffer(device, resources.nodeYBuffer, prep.nodeY);
  writeFloatArrayToBuffer(device, resources.clusterCenterXBuffer, sample.clusterCenterX);
  writeFloatArrayToBuffer(device, resources.clusterCenterYBuffer, sample.clusterCenterY);
  writeFloatArrayToBuffer(device, resources.forceXBuffer, forceX);
  writeFloatArrayToBuffer(device, resources.forceYBuffer, forceY);

  const encoder = device.createCommandEncoder({ label: 'soft-fluid-cluster-load-reduction-encoder' });
  const pass = encoder.beginComputePass({ label: 'soft-fluid-cluster-load-reduction-pass' });
  pass.setPipeline(resources.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(clusterCount / 64)));
  pass.end();

  encoder.copyBufferToBuffer(resources.outClusterForceXBuffer, 0, resources.readClusterForceXBuffer, 0, clusterCount * 4);
  encoder.copyBufferToBuffer(resources.outClusterForceYBuffer, 0, resources.readClusterForceYBuffer, 0, clusterCount * 4);
  encoder.copyBufferToBuffer(resources.outClusterTorqueBuffer, 0, resources.readClusterTorqueBuffer, 0, clusterCount * 4);
  encoder.copyBufferToBuffer(resources.outClusterCountBuffer, 0, resources.readClusterCountBuffer, 0, clusterCount * 4);
  device.queue.submit([encoder.finish()]);

  state.pendingClusterLoadProposalPromise = Promise.all([
    resources.readClusterForceXBuffer.mapAsync(GPUMapMode.READ),
    resources.readClusterForceYBuffer.mapAsync(GPUMapMode.READ),
    resources.readClusterTorqueBuffer.mapAsync(GPUMapMode.READ),
    resources.readClusterCountBuffer.mapAsync(GPUMapMode.READ),
  ]).then(() => {
    try {
      state.lastClusterLoadProposalForceX = new Float32Array(resources.readClusterForceXBuffer.getMappedRange().slice(0));
      state.lastClusterLoadProposalForceY = new Float32Array(resources.readClusterForceYBuffer.getMappedRange().slice(0));
      state.lastClusterLoadProposalTorque = new Float32Array(resources.readClusterTorqueBuffer.getMappedRange().slice(0));
      state.lastClusterLoadProposalCount = new Uint32Array(resources.readClusterCountBuffer.getMappedRange().slice(0));
      state.lastClusterLoadProposalSignature = Number(proposalSignature) >>> 0;
      state.lastClusterLoadProposalSource = 'wgsl-cluster-load-proposal';
      state.lastSourceRoute = 'wgsl-cluster-load-proposal';
      state.lastMode = 'wgsl-cluster-load-proposal';
    } finally {
      resources.readClusterForceXBuffer.unmap();
      resources.readClusterForceYBuffer.unmap();
      resources.readClusterTorqueBuffer.unmap();
      resources.readClusterCountBuffer.unmap();
    }
  }).catch((err) => {
    state.lastClusterLoadProposalError = String(err?.message || err);
  }).finally(() => {
    state.pendingClusterLoadProposalPromise = null;
  });

  return true;
}

export function applySoftFluidCouplingGpuOnly({
  sim,
  soft,
  n,
  dt,
  dtNorm,
  vxField,
  vyField,
  dragK,
  swimGain,
  localHoneyDrag,
  viscosityMotionResponse,
  obstacleMask,
  bodyFeedbackPrevVx,
  bodyFeedbackPrevVy,
  selfFeedbackSuppression,
  softMembraneClusterSet,
  softNodeMomentumScale,
  constants,
  computeSoftCentroid,
  computeSoftClusterKinematics,
  projectNodesTowardClusterRigidMotion,
  sampleFluidForBodyCoupling,
  wgslOffload,
}) {
  const {
    SOFT_NODE_FLOW_COUPLING,
    SOFT_NODE_LOCAL_FLOW_SHARE,
    SOFT_CLUSTER_TUG_COUPLING,
    SOFT_CLUSTER_RELATIVE_DRAG,
    softClusterFluidTorqueCoupling,
    SOFT_CLUSTER_LINEAR_PROJECTION,
    softClusterAngularProjection,
  } = constants;

  const nodes = soft?.nodes || [];
  const modeProfile = getGpuOnlyPipelineModeProfile(wgslOffload);
  const fastMode = isGpuOnlyFastMode(wgslOffload);
  let carryProposalSignature = 0;
  let clusterLoadProposalSignature = 0;

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    wgslOffload.state.lastPipelineModeProfile = modeProfile;
    const prep = buildSoftFluidCouplingWgslLayout({
      nodes,
      softNodeMomentumScale,
      softMembraneClusterSet,
    });
    const samplePrep = buildSoftFluidCouplingCpuSampleLayout({
      sim,
      nodes,
      n,
      vxField,
      vyField,
      obstacleMask,
      bodyFeedbackPrevVx,
      bodyFeedbackPrevVy,
      selfFeedbackSuppression,
      computeSoftCentroid,
      computeSoftClusterKinematics,
      sampleFluidForBodyCoupling,
      localHoneyDrag,
    });
    wgslOffload.state.preparedLayout = prep.layout;
    wgslOffload.state.preparedSampleLayout = samplePrep.layout;
    wgslOffload.state.lastPreparedNodeCount = prep.nodeCount;
    wgslOffload.state.lastPreparedClusterCount = prep.clusterCount;
    wgslOffload.state.lastPreparedLayoutBytes = prep.byteLength;
    wgslOffload.state.lastPreparedLayoutSignature = prep.signature;
    wgslOffload.state.lastPreparedOwnershipCount = prep.layout.clusterNodeIndices.length;
    carryProposalSignature = (Math.imul(prep.signature ^ samplePrep.signature, 0x01000193) ^ ((nodes.length || 0) >>> 0)) >>> 0;
    clusterLoadProposalSignature = (Math.imul(carryProposalSignature ^ prep.signature, 0x01000193) ^ (prep.clusterCount >>> 0)) >>> 0;
    wgslOffload.state.lastPreparedSampleLayoutBytes = samplePrep.byteLength;
    wgslOffload.state.lastPreparedSampleLayoutSignature = samplePrep.signature;
    wgslOffload.state.lastPreparedSampleNodeCount = samplePrep.nodeCount;
    wgslOffload.state.lastPreparedProposalSignature = carryProposalSignature;
    wgslOffload.state.lastPreparedClusterLoadProposalSignature = clusterLoadProposalSignature;
    wgslOffload.state.lastSourceRoute = 'cpu-sampled-layout+cluster-ownership';
    wgslOffload.state.lastMode = 'cpu-prepared';
    wgslOffload.state.lastError = null;

    const wgslProposalDispatched = dispatchSoftFluidCarryProposalWgsl({
      wgslOffload,
      nodeCount: nodes.length,
      dragK,
      nodeFlowCoupling: SOFT_NODE_FLOW_COUPLING,
      localFlowShare: SOFT_NODE_LOCAL_FLOW_SHARE,
      proposalSignature: carryProposalSignature,
    });
    wgslOffload.state.lastCarryProposalDispatched = wgslProposalDispatched;
    if (wgslProposalDispatched) {
      wgslOffload.state.lastMode = 'wgsl-carry-proposal-dispatched';
      wgslOffload.state.lastSourceRoute = 'wgsl-carry-proposal-dispatched';
    }
  }

  const softCentroid = computeSoftCentroid(nodes);
  let softClusterKinematics = computeSoftClusterKinematics(nodes);
  const clusterCarryMap = new Map();
  const clusterFluidLoadMap = new Map();
  const authoritativeCarryProposal = hasAuthoritativeWgslCarryProposal(
    wgslOffload,
    carryProposalSignature,
    nodes.length,
  )
    ? {
      forceX: wgslOffload.state.lastCarryProposalForceX,
      forceY: wgslOffload.state.lastCarryProposalForceY,
      carryX: wgslOffload.state.lastCarryProposalCarryX,
      carryY: wgslOffload.state.lastCarryProposalCarryY,
      localCarryX: wgslOffload.state.lastCarryProposalLocalCarryX,
      localCarryY: wgslOffload.state.lastCarryProposalLocalCarryY,
    }
    : null;
  const authoritativeCarryFinite = authoritativeCarryProposal
    ? buildCarryProposalFiniteSummary(authoritativeCarryProposal)
    : { allFinite: false };
  const canUseAuthoritativeCarryProposal = authoritativeCarryProposal && (fastMode ? authoritativeCarryFinite.allFinite === true : true);
  const useFastAuthoritativeCarryShortcut = fastMode && canUseAuthoritativeCarryProposal;
  let carrySource = canUseAuthoritativeCarryProposal ? 'wgsl-carry-authoritative' : 'cpu-carry-authoritative';

  const clusterIds = wgslOffload?.state?.preparedLayout?.clusterIds;
  const proposalForceX = wgslOffload?.state?.lastClusterLoadProposalForceX;
  const proposalForceY = wgslOffload?.state?.lastClusterLoadProposalForceY;
  const proposalTorque = wgslOffload?.state?.lastClusterLoadProposalTorque;
  const proposalCount = wgslOffload?.state?.lastClusterLoadProposalCount;
  const hasClusterProposal = clusterIds instanceof Int32Array
    && proposalForceX instanceof Float32Array
    && proposalForceY instanceof Float32Array
    && proposalTorque instanceof Float32Array
    && proposalCount instanceof Uint32Array
    && wgslOffload?.state?.lastClusterLoadProposalSignature === clusterLoadProposalSignature
    && proposalForceX.length === clusterIds.length
    && proposalForceY.length === clusterIds.length
    && proposalTorque.length === clusterIds.length
    && proposalCount.length === clusterIds.length;
  const proposalForceXFinite = checkFiniteFloat32Array(proposalForceX);
  const proposalForceYFinite = checkFiniteFloat32Array(proposalForceY);
  const proposalTorqueFinite = checkFiniteFloat32Array(proposalTorque);
  const proposalCountFinite = checkFiniteUint32Array(proposalCount);
  const clusterLoadFinite = hasClusterProposal
    && proposalForceXFinite.allFinite
    && proposalForceYFinite.allFinite
    && proposalTorqueFinite.allFinite
    && proposalCountFinite.allFinite;

  const canUseAuthoritativeClusterLoadPreloop = fastMode
    && wgslOffload?.authoritativeClusterLoad === true
    && clusterLoadFinite
    && clusterIds instanceof Int32Array
    && hasAuthoritativeWgslClusterLoadProposal(wgslOffload, clusterLoadProposalSignature, clusterIds.length);

  const ensureClusterLoad = (cid) => {
    if (!clusterFluidLoadMap.has(cid)) {
      clusterFluidLoadMap.set(cid, { forceX: 0, forceY: 0, torque: 0, count: 0 });
    }
    return clusterFluidLoadMap.get(cid);
  };

  let softCarryTransfer = 0;
  const cpuProposalForceX = useFastAuthoritativeCarryShortcut ? null : new Float32Array(nodes.length);
  const cpuProposalForceY = useFastAuthoritativeCarryShortcut ? null : new Float32Array(nodes.length);
  const cpuProposalCarryX = useFastAuthoritativeCarryShortcut ? null : new Float32Array(nodes.length);
  const cpuProposalCarryY = useFastAuthoritativeCarryShortcut ? null : new Float32Array(nodes.length);
  const cpuProposalLocalCarryX = useFastAuthoritativeCarryShortcut ? null : new Float32Array(nodes.length);
  const cpuProposalLocalCarryY = useFastAuthoritativeCarryShortcut ? null : new Float32Array(nodes.length);

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const mass = Math.max(0.02, node.mass);
    const invMass = 1 / mass;
    const cx = node.x - softCentroid.x;
    const cy = node.y - softCentroid.y;
    const activeSwimPhase = sim.frame * 0.12 + i * 1.57;
    const activeSwimAmp = 0.008 * (1 + 0.2 * Math.sin(sim.frame * 0.05 + i));
    const swimX = swimGain * (-cy * activeSwimAmp + Math.cos(activeSwimPhase) * 0.004) * invMass;
    const swimY = swimGain * (cx * activeSwimAmp + Math.sin(activeSwimPhase) * 0.004) * invMass;
    const honey = localHoneyDrag(node.x, node.y);
    const cid = node.clusterId ?? 0;
    const isMembraneCluster = softMembraneClusterSet.has(cid);
    const nodeMomentum = softNodeMomentumScale(i);
    const flowCouplingBase = isMembraneCluster ? (SOFT_NODE_FLOW_COUPLING * 0.88) : SOFT_NODE_FLOW_COUPLING;
    const flowCoupling = flowCouplingBase * nodeMomentum;

    const clusterKin = softClusterKinematics.get(cid);
    const clusterX = Number.isFinite(Number(clusterKin?.x)) ? Number(clusterKin.x) : softCentroid.x;
    const clusterY = Number.isFinite(Number(clusterKin?.y)) ? Number(clusterKin.y) : softCentroid.y;
    const clusterVx = Number.isFinite(Number(clusterKin?.vx)) ? Number(clusterKin.vx) : 0;
    const clusterVy = Number.isFinite(Number(clusterKin?.vy)) ? Number(clusterKin.vy) : 0;
    const clusterOmega = Number.isFinite(Number(clusterKin?.omega)) ? Number(clusterKin.omega) : 0;

    const rx = node.x - clusterX;
    const ry = node.y - clusterY;
    let cpuForceX = 0;
    let cpuForceY = 0;
    let cpuCarryX = 0;
    let cpuCarryY = 0;
    let cpuLocalCarryX = 0;
    let cpuLocalCarryY = 0;

    if (!useFastAuthoritativeCarryShortcut) {
      const fx = sampleFluidForBodyCoupling(
        vxField,
        n,
        node.x,
        node.y,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVx,
        selfFeedbackSuppression,
      );
      const fy = sampleFluidForBodyCoupling(
        vyField,
        n,
        node.x,
        node.y,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVy,
        selfFeedbackSuppression,
      );
      const clusterLocalVx = clusterVx - clusterOmega * ry;
      const clusterLocalVy = clusterVy + clusterOmega * rx;

      cpuForceX = (fx - clusterLocalVx) * dragK * honey * flowCoupling * mass;
      cpuForceY = (fy - clusterLocalVy) * dragK * honey * flowCoupling * mass;
      cpuCarryX = cpuForceX * invMass;
      cpuCarryY = cpuForceY * invMass;

      cpuLocalCarryX = (fx - node.vx) * dragK * honey * invMass * flowCoupling * SOFT_NODE_LOCAL_FLOW_SHARE;
      cpuLocalCarryY = (fy - node.vy) * dragK * honey * invMass * flowCoupling * SOFT_NODE_LOCAL_FLOW_SHARE;

      cpuProposalForceX[i] = cpuForceX;
      cpuProposalForceY[i] = cpuForceY;
      cpuProposalCarryX[i] = cpuCarryX;
      cpuProposalCarryY[i] = cpuCarryY;
      cpuProposalLocalCarryX[i] = cpuLocalCarryX;
      cpuProposalLocalCarryY[i] = cpuLocalCarryY;
    }

    const forceX = canUseAuthoritativeCarryProposal ? authoritativeCarryProposal.forceX[i] : cpuForceX;
    const forceY = canUseAuthoritativeCarryProposal ? authoritativeCarryProposal.forceY[i] : cpuForceY;
    const carryX = canUseAuthoritativeCarryProposal ? authoritativeCarryProposal.carryX[i] : cpuCarryX;
    const carryY = canUseAuthoritativeCarryProposal ? authoritativeCarryProposal.carryY[i] : cpuCarryY;
    const localCarryX = canUseAuthoritativeCarryProposal ? authoritativeCarryProposal.localCarryX[i] : cpuLocalCarryX;
    const localCarryY = canUseAuthoritativeCarryProposal ? authoritativeCarryProposal.localCarryY[i] : cpuLocalCarryY;

    node.vx += localCarryX * dt * 60 + swimX * dtNorm;
    node.vy += localCarryY * dt * 60 + swimY * dtNorm;

    if (!canUseAuthoritativeClusterLoadPreloop) {
      const load = ensureClusterLoad(cid);
      load.forceX += forceX;
      load.forceY += forceY;
      load.torque += rx * forceY - ry * forceX;
      load.count += 1;
    }

    const st = clusterCarryMap.get(cid) || { sumX: 0, sumY: 0, count: 0, maxX: 0, maxY: 0, maxMag: 0 };
    st.sumX += carryX;
    st.sumY += carryY;
    st.count += 1;
    const cmag = Math.hypot(carryX, carryY);
    if (cmag > st.maxMag) {
      st.maxMag = cmag;
      st.maxX = carryX;
      st.maxY = carryY;
    }
    clusterCarryMap.set(cid, st);

    const softVisc = viscosityMotionResponse(honey, 2.8);
    node.vx *= softVisc.damp;
    node.vy *= softVisc.damp;
    const nMax = softVisc.vmax;
    const nMag = Math.hypot(node.vx, node.vy);
    if (nMag > nMax) {
      node.vx = (node.vx / nMag) * nMax;
      node.vy = (node.vy / nMag) * nMax;
    }
    softCarryTransfer += Math.hypot(carryX, carryY);
  }

  if (wgslOffload?.state && clusterLoadProposalSignature !== 0) {
    const clusterLoadForceX = useFastAuthoritativeCarryShortcut ? authoritativeCarryProposal?.forceX : cpuProposalForceX;
    const clusterLoadForceY = useFastAuthoritativeCarryShortcut ? authoritativeCarryProposal?.forceY : cpuProposalForceY;
    const wgslClusterLoadDispatched = dispatchSoftFluidClusterLoadReductionWgsl({
      wgslOffload,
      proposalSignature: clusterLoadProposalSignature,
      forceX: clusterLoadForceX,
      forceY: clusterLoadForceY,
    });
    wgslOffload.state.lastClusterLoadProposalDispatched = wgslClusterLoadDispatched;
  }

  // Cluster-load proposal + finite checks were computed before the node loop so
  // fast-mode can skip CPU cluster-load accumulation when authoritative WGSL
  // cluster reduction is already available for this signature.

  let clusterLoadMismatchCount = 0;
  if (!fastMode && hasClusterProposal) {
    for (let i = 0; i < clusterIds.length; i++) {
      const cid = clusterIds[i];
      const cpuLoad = clusterFluidLoadMap.get(cid) || { forceX: 0, forceY: 0, torque: 0, count: 0 };
      if (Math.abs((proposalForceX[i] || 0) - (cpuLoad.forceX || 0)) > 1e-5
        || Math.abs((proposalForceY[i] || 0) - (cpuLoad.forceY || 0)) > 1e-5
        || Math.abs((proposalTorque[i] || 0) - (cpuLoad.torque || 0)) > 1e-5
        || Math.abs((proposalCount[i] || 0) - (cpuLoad.count || 0)) > 0) {
        clusterLoadMismatchCount += 1;
      }
    }
  }

  const canUseAuthoritativeClusterLoad = wgslOffload?.authoritativeClusterLoad === true
    && clusterLoadFinite
    && clusterIds instanceof Int32Array
    && hasAuthoritativeWgslClusterLoadProposal(wgslOffload, clusterLoadProposalSignature, clusterIds.length)
    && (fastMode ? true : clusterLoadMismatchCount === 0);
  const clusterLoadSource = canUseAuthoritativeClusterLoad
    ? 'wgsl-cluster-load-authoritative'
    : 'cpu-cluster-load-authoritative';

  const clusterAccelMap = new Map();
  if (canUseAuthoritativeClusterLoad) {
    for (let i = 0; i < clusterIds.length; i++) {
      const cid = clusterIds[i];
      const clusterKin = softClusterKinematics.get(cid);
      const count = Number(proposalCount[i]) || 0;
      const clusterMass = Math.max(0.02, Number(clusterKin?.mass) || (Math.max(1, count) * sim.controls.massSoft));
      const clusterInertia = Math.max(1e-4, Number(clusterKin?.inertia) || 1e-4);
      clusterAccelMap.set(cid, {
        x: Number.isFinite(Number(clusterKin?.x)) ? Number(clusterKin.x) : softCentroid.x,
        y: Number.isFinite(Number(clusterKin?.y)) ? Number(clusterKin.y) : softCentroid.y,
        ax: (proposalForceX[i] || 0) / clusterMass,
        ay: (proposalForceY[i] || 0) / clusterMass,
        alpha: (proposalTorque[i] || 0) / clusterInertia,
      });
    }
  } else {
    for (const [cid, load] of clusterFluidLoadMap.entries()) {
      const clusterKin = softClusterKinematics.get(cid);
      const clusterMass = Math.max(0.02, Number(clusterKin?.mass) || (Math.max(1, load.count) * sim.controls.massSoft));
      const clusterInertia = Math.max(1e-4, Number(clusterKin?.inertia) || 1e-4);
      clusterAccelMap.set(cid, {
        x: Number.isFinite(Number(clusterKin?.x)) ? Number(clusterKin.x) : softCentroid.x,
        y: Number.isFinite(Number(clusterKin?.y)) ? Number(clusterKin.y) : softCentroid.y,
        ax: load.forceX / clusterMass,
        ay: load.forceY / clusterMass,
        alpha: load.torque / clusterInertia,
      });
    }
  }

  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const acc = clusterAccelMap.get(cid);
    if (!acc) continue;
    const rx = node.x - acc.x;
    const ry = node.y - acc.y;
    const flowShare = softMembraneClusterSet.has(cid)
      ? (softClusterFluidTorqueCoupling * 0.7)
      : softClusterFluidTorqueCoupling;
    node.vx += (acc.ax - acc.alpha * ry) * flowShare * dt * 60;
    node.vy += (acc.ay + acc.alpha * rx) * flowShare * dt * 60;
  }

  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const st = clusterCarryMap.get(cid);
    if (!st || st.count <= 0) continue;
    const meanX = st.sumX / st.count;
    const meanY = st.sumY / st.count;
    const pullX = meanX * 0.6 + st.maxX * 0.4;
    const pullY = meanY * 0.6 + st.maxY * 0.4;
    const tugCoupling = softMembraneClusterSet.has(cid) ? (SOFT_CLUSTER_TUG_COUPLING * 0.45) : (SOFT_CLUSTER_TUG_COUPLING * 0.72);
    node.vx += pullX * tugCoupling * dt * 60;
    node.vy += pullY * tugCoupling * dt * 60;
  }

  const clusterVelMap = new Map();
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const st = clusterVelMap.get(cid) || { sumVx: 0, sumVy: 0, count: 0 };
    st.sumVx += node.vx || 0;
    st.sumVy += node.vy || 0;
    st.count += 1;
    clusterVelMap.set(cid, st);
  }
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const st = clusterVelMap.get(cid);
    if (!st || st.count <= 0) continue;
    const meanVx = st.sumVx / st.count;
    const meanVy = st.sumVy / st.count;
    const relDamp = softMembraneClusterSet.has(cid) ? (SOFT_CLUSTER_RELATIVE_DRAG * 0.65) : SOFT_CLUSTER_RELATIVE_DRAG;
    node.vx -= (node.vx - meanVx) * relDamp * dtNorm;
    node.vy -= (node.vy - meanVy) * relDamp * dtNorm;
  }

  softClusterKinematics = computeSoftClusterKinematics(nodes);
  projectNodesTowardClusterRigidMotion(nodes, softClusterKinematics, {
    linearGain: SOFT_CLUSTER_LINEAR_PROJECTION * dtNorm,
    angularGain: softClusterAngularProjection * dtNorm,
    membraneClusterSet: softMembraneClusterSet,
    membraneGainScale: 0.72,
  });

  if (wgslOffload?.state) {
    wgslOffload.state.lastPipelineModeProfile = modeProfile;
    wgslOffload.state.lastCarryProposalFinite = authoritativeCarryFinite;
    wgslOffload.state.lastCarryFastShortcutUsed = useFastAuthoritativeCarryShortcut;
    wgslOffload.state.lastClusterLoadCpuAccumulationSkipped = canUseAuthoritativeClusterLoadPreloop;
    if (!fastMode) {
      wgslOffload.state.lastCpuCarryProposalForceX = cpuProposalForceX;
      wgslOffload.state.lastCpuCarryProposalForceY = cpuProposalForceY;
      wgslOffload.state.lastCpuCarryProposalCarryX = cpuProposalCarryX;
      wgslOffload.state.lastCpuCarryProposalCarryY = cpuProposalCarryY;
      wgslOffload.state.lastCpuCarryProposalLocalCarryX = cpuProposalLocalCarryX;
      wgslOffload.state.lastCpuCarryProposalLocalCarryY = cpuProposalLocalCarryY;
      if (clusterIds instanceof Int32Array) {
        const cpuClusterForceX = new Float32Array(clusterIds.length);
        const cpuClusterForceY = new Float32Array(clusterIds.length);
        const cpuClusterTorque = new Float32Array(clusterIds.length);
        const cpuClusterCount = new Uint32Array(clusterIds.length);
        for (let i = 0; i < clusterIds.length; i++) {
          const cpuLoad = clusterFluidLoadMap.get(clusterIds[i]) || { forceX: 0, forceY: 0, torque: 0, count: 0 };
          cpuClusterForceX[i] = Number(cpuLoad.forceX) || 0;
          cpuClusterForceY[i] = Number(cpuLoad.forceY) || 0;
          cpuClusterTorque[i] = Number(cpuLoad.torque) || 0;
          cpuClusterCount[i] = (Number(cpuLoad.count) || 0) >>> 0;
        }
        wgslOffload.state.lastCpuClusterLoadForceX = cpuClusterForceX;
        wgslOffload.state.lastCpuClusterLoadForceY = cpuClusterForceY;
        wgslOffload.state.lastCpuClusterLoadTorque = cpuClusterTorque;
        wgslOffload.state.lastCpuClusterLoadCount = cpuClusterCount;
      }
    }
    wgslOffload.state.lastClusterLoadParity = fastMode
      ? {
        source: 'wgsl-cluster-load-proposal-finite-fast',
        mismatchCount: null,
        clusterCount: Number(clusterIds?.length) || 0,
        signature: clusterLoadProposalSignature >>> 0,
        allFinite: clusterLoadFinite,
        finite: {
          forceX: proposalForceXFinite,
          forceY: proposalForceYFinite,
          torque: proposalTorqueFinite,
          count: proposalCountFinite,
        },
      }
      : {
        source: 'wgsl-cluster-load-proposal-vs-cpu',
        mismatchCount: clusterLoadMismatchCount,
        clusterCount: Number(clusterIds?.length) || 0,
        signature: clusterLoadProposalSignature >>> 0,
        allFinite: clusterLoadFinite,
      };
    wgslOffload.state.lastAuthoritativeCarrySource = carrySource;
    wgslOffload.state.lastAuthoritativeCarrySignature = carryProposalSignature;
    wgslOffload.state.lastAuthoritativeClusterLoadSource = clusterLoadSource;
    wgslOffload.state.lastAuthoritativeClusterLoadSignature = clusterLoadProposalSignature;
    wgslOffload.state.lastSourceRoute = `${carrySource}+${clusterLoadSource}`;
    wgslOffload.state.lastMode = carrySource === 'wgsl-carry-authoritative' || clusterLoadSource === 'wgsl-cluster-load-authoritative'
      ? (fastMode ? 'wgsl-carry-or-cluster-authoritative-fast' : 'wgsl-carry-or-cluster-authoritative')
      : (fastMode ? 'cpu-carry+cluster-authoritative-fast-fallback' : 'cpu-carry+cluster-authoritative');
  }

  return {
    softCarryTransfer,
    softCentroid,
    softClusterKinematics,
  };
}
