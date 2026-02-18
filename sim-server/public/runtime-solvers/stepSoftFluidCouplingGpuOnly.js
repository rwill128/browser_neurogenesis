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
    localHoney[i] = Number(localHoneyDrag?.(Number(node.x) || 0, Number(node.y) || 0)) || 0;
  }

  const layout = {
    fluidSampleVx,
    fluidSampleVy,
    clusterLocalVx,
    clusterLocalVy,
    sampleDeltaVx,
    sampleDeltaVy,
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
@group(0) @binding(11) var<storage, read_write> outForceX: array<f32>;
@group(0) @binding(12) var<storage, read_write> outForceY: array<f32>;
@group(0) @binding(13) var<storage, read_write> outCarryX: array<f32>;
@group(0) @binding(14) var<storage, read_write> outCarryY: array<f32>;
@group(0) @binding(15) var<storage, read_write> outLocalCarryX: array<f32>;
@group(0) @binding(16) var<storage, read_write> outLocalCarryY: array<f32>;

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
  outForceX[i] = fx;
  outForceY[i] = fy;
  outCarryX[i] = fx * invM;
  outCarryY[i] = fy * invM;

  outLocalCarryX[i] = (sampleVx[i] - nodeVx[i]) * dragHoney * invM * params.localFlowShare;
  outLocalCarryY[i] = (sampleVy[i] - nodeVy[i]) * dragHoney * invM * params.localFlowShare;
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
  const createOutputStorage = (label) => device.createBuffer({
    label,
    size: Math.max(4, bytes),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const createReadback = (label) => device.createBuffer({
    label,
    size: Math.max(4, bytes),
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
    outForceXBuffer: createOutputStorage('soft-fluid-carry-out-force-x'),
    outForceYBuffer: createOutputStorage('soft-fluid-carry-out-force-y'),
    outCarryXBuffer: createOutputStorage('soft-fluid-carry-out-carry-x'),
    outCarryYBuffer: createOutputStorage('soft-fluid-carry-out-carry-y'),
    outLocalCarryXBuffer: createOutputStorage('soft-fluid-carry-out-local-carry-x'),
    outLocalCarryYBuffer: createOutputStorage('soft-fluid-carry-out-local-carry-y'),
    readForceXBuffer: createReadback('soft-fluid-carry-read-force-x'),
    readForceYBuffer: createReadback('soft-fluid-carry-read-force-y'),
    readCarryXBuffer: createReadback('soft-fluid-carry-read-carry-x'),
    readCarryYBuffer: createReadback('soft-fluid-carry-read-carry-y'),
    readLocalCarryXBuffer: createReadback('soft-fluid-carry-read-local-carry-x'),
    readLocalCarryYBuffer: createReadback('soft-fluid-carry-read-local-carry-y'),
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
      { binding: 11, resource: { buffer: resources.outForceXBuffer } },
      { binding: 12, resource: { buffer: resources.outForceYBuffer } },
      { binding: 13, resource: { buffer: resources.outCarryXBuffer } },
      { binding: 14, resource: { buffer: resources.outCarryYBuffer } },
      { binding: 15, resource: { buffer: resources.outLocalCarryXBuffer } },
      { binding: 16, resource: { buffer: resources.outLocalCarryYBuffer } },
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

  encoder.copyBufferToBuffer(resources.outForceXBuffer, 0, resources.readForceXBuffer, 0, nodeCount * 4);
  encoder.copyBufferToBuffer(resources.outForceYBuffer, 0, resources.readForceYBuffer, 0, nodeCount * 4);
  encoder.copyBufferToBuffer(resources.outCarryXBuffer, 0, resources.readCarryXBuffer, 0, nodeCount * 4);
  encoder.copyBufferToBuffer(resources.outCarryYBuffer, 0, resources.readCarryYBuffer, 0, nodeCount * 4);
  encoder.copyBufferToBuffer(resources.outLocalCarryXBuffer, 0, resources.readLocalCarryXBuffer, 0, nodeCount * 4);
  encoder.copyBufferToBuffer(resources.outLocalCarryYBuffer, 0, resources.readLocalCarryYBuffer, 0, nodeCount * 4);

  device.queue.submit([encoder.finish()]);

  const readback = async () => {
    const mapRead = async (buffer) => {
      await buffer.mapAsync(GPUMapMode.READ);
      try {
        return new Float32Array(buffer.getMappedRange().slice(0));
      } finally {
        buffer.unmap();
      }
    };

    const [forceX, forceY, carryX, carryY, localCarryX, localCarryY] = await Promise.all([
      mapRead(resources.readForceXBuffer),
      mapRead(resources.readForceYBuffer),
      mapRead(resources.readCarryXBuffer),
      mapRead(resources.readCarryYBuffer),
      mapRead(resources.readLocalCarryXBuffer),
      mapRead(resources.readLocalCarryYBuffer),
    ]);

    state.lastCarryProposalForceX = forceX;
    state.lastCarryProposalForceY = forceY;
    state.lastCarryProposalCarryX = carryX;
    state.lastCarryProposalCarryY = carryY;
    state.lastCarryProposalLocalCarryX = localCarryX;
    state.lastCarryProposalLocalCarryY = localCarryY;
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
  let carryProposalSignature = 0;

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
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
    wgslOffload.state.lastPreparedSampleLayoutBytes = samplePrep.byteLength;
    wgslOffload.state.lastPreparedSampleLayoutSignature = samplePrep.signature;
    wgslOffload.state.lastPreparedSampleNodeCount = samplePrep.nodeCount;
    wgslOffload.state.lastPreparedProposalSignature = carryProposalSignature;
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
  let carrySource = authoritativeCarryProposal ? 'wgsl-carry-authoritative' : 'cpu-carry-authoritative';
  const ensureClusterLoad = (cid) => {
    if (!clusterFluidLoadMap.has(cid)) {
      clusterFluidLoadMap.set(cid, { forceX: 0, forceY: 0, torque: 0, count: 0 });
    }
    return clusterFluidLoadMap.get(cid);
  };

  let softCarryTransfer = 0;
  const cpuProposalForceX = new Float32Array(nodes.length);
  const cpuProposalForceY = new Float32Array(nodes.length);
  const cpuProposalCarryX = new Float32Array(nodes.length);
  const cpuProposalCarryY = new Float32Array(nodes.length);
  const cpuProposalLocalCarryX = new Float32Array(nodes.length);
  const cpuProposalLocalCarryY = new Float32Array(nodes.length);

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

    const cpuForceX = (fx - clusterLocalVx) * dragK * honey * flowCoupling * mass;
    const cpuForceY = (fy - clusterLocalVy) * dragK * honey * flowCoupling * mass;
    const cpuCarryX = cpuForceX * invMass;
    const cpuCarryY = cpuForceY * invMass;

    const cpuLocalCarryX = (fx - node.vx) * dragK * honey * invMass * flowCoupling * SOFT_NODE_LOCAL_FLOW_SHARE;
    const cpuLocalCarryY = (fy - node.vy) * dragK * honey * invMass * flowCoupling * SOFT_NODE_LOCAL_FLOW_SHARE;

    cpuProposalForceX[i] = cpuForceX;
    cpuProposalForceY[i] = cpuForceY;
    cpuProposalCarryX[i] = cpuCarryX;
    cpuProposalCarryY[i] = cpuCarryY;
    cpuProposalLocalCarryX[i] = cpuLocalCarryX;
    cpuProposalLocalCarryY[i] = cpuLocalCarryY;

    const forceX = authoritativeCarryProposal ? authoritativeCarryProposal.forceX[i] : cpuForceX;
    const forceY = authoritativeCarryProposal ? authoritativeCarryProposal.forceY[i] : cpuForceY;
    const carryX = authoritativeCarryProposal ? authoritativeCarryProposal.carryX[i] : cpuCarryX;
    const carryY = authoritativeCarryProposal ? authoritativeCarryProposal.carryY[i] : cpuCarryY;
    const localCarryX = authoritativeCarryProposal ? authoritativeCarryProposal.localCarryX[i] : cpuLocalCarryX;
    const localCarryY = authoritativeCarryProposal ? authoritativeCarryProposal.localCarryY[i] : cpuLocalCarryY;

    node.vx += localCarryX * dt * 60 + swimX * dtNorm;
    node.vy += localCarryY * dt * 60 + swimY * dtNorm;

    const load = ensureClusterLoad(cid);
    load.forceX += forceX;
    load.forceY += forceY;
    load.torque += rx * forceY - ry * forceX;
    load.count += 1;

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

  const clusterAccelMap = new Map();
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
    wgslOffload.state.lastCpuCarryProposalForceX = cpuProposalForceX;
    wgslOffload.state.lastCpuCarryProposalForceY = cpuProposalForceY;
    wgslOffload.state.lastCpuCarryProposalCarryX = cpuProposalCarryX;
    wgslOffload.state.lastCpuCarryProposalCarryY = cpuProposalCarryY;
    wgslOffload.state.lastCpuCarryProposalLocalCarryX = cpuProposalLocalCarryX;
    wgslOffload.state.lastCpuCarryProposalLocalCarryY = cpuProposalLocalCarryY;
    wgslOffload.state.lastAuthoritativeCarrySource = carrySource;
    wgslOffload.state.lastAuthoritativeCarrySignature = carryProposalSignature;
    wgslOffload.state.lastSourceRoute = carrySource;
    wgslOffload.state.lastMode = carrySource === 'wgsl-carry-authoritative'
      ? 'wgsl-carry-authoritative'
      : 'cpu-carry-authoritative';
  }

  return {
    softCarryTransfer,
    softCentroid,
    softClusterKinematics,
  };
}
