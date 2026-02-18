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

const SOFT_DEFORMATION_WGSL = /* wgsl */`
struct Params {
  springCount:u32,
  clusterCount:u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> springNodeA: array<u32>;
@group(0) @binding(4) var<storage, read> springNodeB: array<u32>;
@group(0) @binding(5) var<storage, read> springRest: array<f32>;
@group(0) @binding(6) var<storage, read> springClusterIndex: array<u32>;
@group(0) @binding(7) var<storage, read_write> clusterStretchMaxQ: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> clusterStretchMinQ: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> clusterSpringCount: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> clusterNonFiniteMask: array<atomic<u32>>;

fn quantizeRatio(v:f32)->i32 {
  let clamped = clamp(v, 0.0, 2048.0);
  return i32(clamped * 1048576.0 + 0.5);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let si = gid.x;
  if (si >= params.springCount) { return; }

  let ci = springClusterIndex[si];
  if (ci >= params.clusterCount) { return; }

  let ia = springNodeA[si];
  let ib = springNodeB[si];
  let dx = nodeX[ib] - nodeX[ia];
  let dy = nodeY[ib] - nodeY[ia];
  let len = sqrt(dx * dx + dy * dy);
  let rest = max(1e-4, springRest[si]);
  let ratio = len / rest;

  atomicAdd(&clusterSpringCount[ci], 1u);

  if (!(isFinite(ratio))) {
    atomicOr(&clusterNonFiniteMask[ci], 1u);
    return;
  }

  let q = quantizeRatio(max(1e-6, ratio));
  atomicMax(&clusterStretchMaxQ[ci], q);
  atomicMin(&clusterStretchMinQ[ci], q);
}
`;

function clampToNodeIndex(raw, nodeCount) {
  if (!Number.isFinite(raw)) return 0;
  if (nodeCount <= 0) return 0;
  return Math.max(0, Math.min(nodeCount - 1, raw | 0));
}

function createGpuBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, data.byteLength),
    usage,
    mappedAtCreation: true,
  });
  const ctor = data.constructor;
  new ctor(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

async function readGpuArray(device, srcBuffer, ArrayType, byteLength) {
  const size = Math.max(4, byteLength);
  const readback = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(srcBuffer, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = new ArrayType(readback.getMappedRange(0, size).slice(0));
  readback.unmap();
  readback.destroy();
  return out;
}

async function runSoftDeformationSpringMetricsWgsl({ device, prep, state }) {
  if (!device || !prep?.layout) return null;
  const springCount = Number(prep.plan?.springCount) || 0;
  const clusterCount = Number(prep.plan?.clusterCount) || 0;
  if (springCount <= 0 || clusterCount <= 0) return null;

  const pipeline = state.pipeline ||= await device.createComputePipelineAsync({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: SOFT_DEFORMATION_WGSL }),
      entryPoint: 'main',
    },
  });

  const params = new Uint32Array([springCount >>> 0, clusterCount >>> 0]);
  const maxInit = new Int32Array(clusterCount);
  const minInit = new Int32Array(clusterCount);
  minInit.fill(0x7fffffff);
  const countInit = new Uint32Array(clusterCount);
  const nonFiniteInit = new Uint32Array(clusterCount);

  const paramsBuffer = createGpuBuffer(device, params, GPUBufferUsage.UNIFORM);
  const nodeXBuffer = createGpuBuffer(device, prep.layout.nodeX, GPUBufferUsage.STORAGE);
  const nodeYBuffer = createGpuBuffer(device, prep.layout.nodeY, GPUBufferUsage.STORAGE);
  const springNodeABuffer = createGpuBuffer(device, prep.layout.springNodeA, GPUBufferUsage.STORAGE);
  const springNodeBBuffer = createGpuBuffer(device, prep.layout.springNodeB, GPUBufferUsage.STORAGE);
  const springRestBuffer = createGpuBuffer(device, prep.layout.springRest, GPUBufferUsage.STORAGE);
  const springClusterIndexBuffer = createGpuBuffer(device, prep.layout.springClusterIndex, GPUBufferUsage.STORAGE);

  const maxBuffer = createGpuBuffer(device, maxInit, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const minBuffer = createGpuBuffer(device, minInit, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const countBuffer = createGpuBuffer(device, countInit, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const nonFiniteBuffer = createGpuBuffer(device, nonFiniteInit, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: nodeXBuffer } },
      { binding: 2, resource: { buffer: nodeYBuffer } },
      { binding: 3, resource: { buffer: springNodeABuffer } },
      { binding: 4, resource: { buffer: springNodeBBuffer } },
      { binding: 5, resource: { buffer: springRestBuffer } },
      { binding: 6, resource: { buffer: springClusterIndexBuffer } },
      { binding: 7, resource: { buffer: maxBuffer } },
      { binding: 8, resource: { buffer: minBuffer } },
      { binding: 9, resource: { buffer: countBuffer } },
      { binding: 10, resource: { buffer: nonFiniteBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(springCount / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);

  const [stretchMaxQ, stretchMinQ, clusterSpringCount, clusterNonFiniteMask] = await Promise.all([
    readGpuArray(device, maxBuffer, Int32Array, maxInit.byteLength),
    readGpuArray(device, minBuffer, Int32Array, minInit.byteLength),
    readGpuArray(device, countBuffer, Uint32Array, countInit.byteLength),
    readGpuArray(device, nonFiniteBuffer, Uint32Array, nonFiniteInit.byteLength),
  ]);

  [paramsBuffer, nodeXBuffer, nodeYBuffer, springNodeABuffer, springNodeBBuffer, springRestBuffer,
    springClusterIndexBuffer, maxBuffer, minBuffer, countBuffer, nonFiniteBuffer].forEach((b) => b.destroy());

  return { stretchMaxQ, stretchMinQ, clusterSpringCount, clusterNonFiniteMask };
}

export function buildSoftDeformationInterventionsWgslPrep({ sim, soft, softClusterLoops }) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springs = Array.isArray(soft?.springs) ? soft.springs : [];
  const loops = Array.isArray(softClusterLoops) ? softClusterLoops : [];
  const nodeCount = nodes.length;
  const loopCount = loops.length;

  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  const nodeInvMass = new Float32Array(nodeCount);
  const nodeClusterId = new Int32Array(nodeCount);
  const clusterIdSet = new Set();
  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i] || {};
    nodeX[i] = Number(node.x) || 0;
    nodeY[i] = Number(node.y) || 0;
    const mass = Number(node.mass);
    nodeInvMass[i] = Number.isFinite(mass) && mass > 0 ? (1 / mass) : 0;
    nodeClusterId[i] = Number(node.clusterId) | 0;
    clusterIdSet.add(nodeClusterId[i]);
  }

  const loopOffsets = new Uint32Array(loopCount + 1);
  const loopClusterId = new Int32Array(loopCount);
  let loopNodeRefCount = 0;
  for (let li = 0; li < loopCount; li++) {
    loopOffsets[li] = loopNodeRefCount;
    const loop = loops[li];
    const indices = Array.isArray(loop)
      ? loop
      : (Array.isArray(loop?.indices) ? loop.indices : []);
    loopNodeRefCount += indices.length;
    loopClusterId[li] = Number(loop?.clusterId) | 0;
    clusterIdSet.add(loopClusterId[li]);
  }
  loopOffsets[loopCount] = loopNodeRefCount;

  const loopNodeIndex = new Uint32Array(loopNodeRefCount);
  let write = 0;
  for (let li = 0; li < loopCount; li++) {
    const loop = loops[li];
    const indices = Array.isArray(loop)
      ? loop
      : (Array.isArray(loop?.indices) ? loop.indices : []);
    for (let j = 0; j < indices.length; j++) {
      const idx = clampToNodeIndex(Number(indices[j]), nodeCount);
      loopNodeIndex[write] = idx >>> 0;
      write += 1;
    }
  }

  const springCount = springs.length;
  const springNodeA = new Uint32Array(springCount);
  const springNodeB = new Uint32Array(springCount);
  const springRest = new Float32Array(springCount);
  const springClusterId = new Int32Array(springCount);
  for (let si = 0; si < springCount; si++) {
    const spring = Array.isArray(springs[si]) ? springs[si] : [];
    const a = clampToNodeIndex(Number(spring[0]), nodeCount);
    const b = clampToNodeIndex(Number(spring[1]), nodeCount);
    springNodeA[si] = a >>> 0;
    springNodeB[si] = b >>> 0;
    springRest[si] = Math.max(1e-4, Number(spring[2]) || 1e-4);
    const ca = Number(nodes[a]?.clusterId);
    const cb = Number(nodes[b]?.clusterId);
    springClusterId[si] = Number.isFinite(ca) ? (ca | 0) : (Number.isFinite(cb) ? (cb | 0) : 0);
    clusterIdSet.add(springClusterId[si]);
  }

  const clusterIds = Int32Array.from([...clusterIdSet].sort((a, b) => a - b));
  const clusterIndexById = new Map();
  for (let i = 0; i < clusterIds.length; i++) clusterIndexById.set(clusterIds[i], i);
  const springClusterIndex = new Uint32Array(springCount);
  for (let si = 0; si < springCount; si++) {
    const idx = clusterIndexById.get(springClusterId[si]);
    springClusterIndex[si] = Number.isFinite(idx) ? (idx >>> 0) : 0;
  }

  let signature = 0x811c9dc5;
  signature ^= (Number(sim?.frame) || 0) >>> 0;
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= nodeCount >>> 0;
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= loopCount >>> 0;
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(nodeX);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(nodeY);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(nodeInvMass);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(new Uint32Array(nodeClusterId.buffer));
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(loopOffsets);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(loopNodeIndex);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(new Uint32Array(loopClusterId.buffer));
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(springNodeA);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(springNodeB);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(springRest);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(new Uint32Array(springClusterId.buffer));
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(new Uint32Array(clusterIds.buffer));
  signature >>>= 0;

  const layout = {
    nodeX,
    nodeY,
    nodeInvMass,
    nodeClusterId,
    loopOffsets,
    loopNodeIndex,
    loopClusterId,
    springNodeA,
    springNodeB,
    springRest,
    springClusterId,
    springClusterIndex,
    clusterIds,
  };
  layout.byteLength = nodeX.byteLength
    + nodeY.byteLength
    + nodeInvMass.byteLength
    + nodeClusterId.byteLength
    + loopOffsets.byteLength
    + loopNodeIndex.byteLength
    + loopClusterId.byteLength
    + springNodeA.byteLength
    + springNodeB.byteLength
    + springRest.byteLength
    + springClusterId.byteLength
    + springClusterIndex.byteLength
    + clusterIds.byteLength;

  return {
    plan: {
      frame: Number(sim?.frame) || 0,
      nodeCount,
      loopCount,
      loopNodeRefCount,
      springCount,
      clusterCount: clusterIds.length,
    },
    layout,
    signature,
  };
}

function applyAuthoritativeWgslSpringMetrics(deform, prep, metrics) {
  if (!deform || !prep?.layout?.clusterIds || !metrics) return false;
  const byCluster = new Map();
  for (const c of deform.clusters || []) byCluster.set(Number(c?.clusterId) | 0, c);

  let applied = 0;
  for (let i = 0; i < prep.layout.clusterIds.length; i++) {
    const cid = prep.layout.clusterIds[i] | 0;
    const cluster = byCluster.get(cid);
    if (!cluster) continue;
    const springCount = Number(metrics.clusterSpringCount?.[i]) || 0;
    if (springCount <= 0) continue;
    const nonFinite = (Number(metrics.clusterNonFiniteMask?.[i]) || 0) !== 0;
    const maxQ = Number(metrics.stretchMaxQ?.[i]);
    const minQ = Number(metrics.stretchMinQ?.[i]);
    if (nonFinite || !Number.isFinite(maxQ) || !Number.isFinite(minQ) || minQ <= 0 || minQ >= 0x7fffffff) {
      cluster.severe = true;
      cluster.severeCollapse = true;
      cluster.warning = true;
      continue;
    }
    cluster.springs = springCount;
    cluster.stretchMax = Math.max(1e-6, maxQ / 1048576);
    cluster.stretchMin = Math.max(1e-6, minQ / 1048576);
    applied += 1;
  }

  return applied > 0;
}

function refreshSoftDeformationClassificationFromMetrics({
  deform,
  membraneClusterSet,
  thresholds,
}) {
  if (!deform || !Array.isArray(deform.clusters)) return deform;

  const membraneSet = membraneClusterSet instanceof Set ? membraneClusterSet : new Set();
  const t = thresholds || {};
  const warnStretch = Number.isFinite(Number(t.warnStretch)) ? Number(t.warnStretch) : 1.55;
  const severeStretch = Number.isFinite(Number(t.severeStretch)) ? Number(t.severeStretch) : 2.4;
  const warnAreaMin = Number.isFinite(Number(t.warnAreaMin)) ? Number(t.warnAreaMin) : 0.62;
  const warnAreaMax = Number.isFinite(Number(t.warnAreaMax)) ? Number(t.warnAreaMax) : 1.55;
  const severeAreaMin = Number.isFinite(Number(t.severeAreaMin)) ? Number(t.severeAreaMin) : 0.35;
  const severeAreaMax = Number.isFinite(Number(t.severeAreaMax)) ? Number(t.severeAreaMax) : 2.35;
  const warnPoseRms = Number.isFinite(Number(t.warnPoseRms)) ? Number(t.warnPoseRms) : 0.16;
  const warnPoseMax = Number.isFinite(Number(t.warnPoseMax)) ? Number(t.warnPoseMax) : 0.3;
  const severePoseRms = Number.isFinite(Number(t.severePoseRms)) ? Number(t.severePoseRms) : 0.35;
  const severePoseMax = Number.isFinite(Number(t.severePoseMax)) ? Number(t.severePoseMax) : 0.65;

  const warningClusters = [];
  const severeClusters = [];
  const severeCollapseClusters = [];
  let worstStretch = 1;
  let worstAreaRatio = 1;
  let worstPoseError = 0;

  for (const c of deform.clusters) {
    if (!c) continue;
    const legacyWarn = c.stretchMax >= warnStretch
      || c.areaRatio <= warnAreaMin
      || c.areaRatio >= warnAreaMax;
    const legacySevere = c.stretchMax >= severeStretch
      || c.stretchMin <= 0.12
      || c.areaRatio <= severeAreaMin
      || c.areaRatio >= severeAreaMax;
    const poseAvailable = Number.isFinite(c.poseErrorRms) && Number.isFinite(c.poseErrorMax);
    const poseWarn = c.poseErrorRms >= warnPoseRms || c.poseErrorMax >= warnPoseMax;
    const poseSevere = c.poseErrorRms >= severePoseRms || c.poseErrorMax >= severePoseMax;
    const isMembraneCluster = membraneSet.has(c.clusterId);

    c.warning = c.warning || legacyWarn || (poseAvailable && poseWarn);
    c.severeCollapse = c.severeCollapse || (!isMembraneCluster && legacySevere);
    c.severe = c.severe || c.severeCollapse || (poseAvailable && poseSevere);

    if (c.warning) warningClusters.push(c.clusterId);
    if (c.severe) severeClusters.push(c.clusterId);
    if (c.severeCollapse) severeCollapseClusters.push(c.clusterId);
    worstStretch = Math.max(worstStretch, Number(c.stretchMax) || 1);
    const areaRatio = Number(c.areaRatio) || 1;
    worstAreaRatio = Math.max(worstAreaRatio, Math.max(areaRatio, areaRatio > 0 ? 1 / areaRatio : 1));
    worstPoseError = Math.max(worstPoseError, Number(c.poseErrorRms) || 0);
  }

  deform.warningClusters = warningClusters;
  deform.severeClusters = severeClusters;
  deform.severeCollapseClusters = severeCollapseClusters;
  deform.warningSet = new Set(warningClusters);
  deform.severeSet = new Set(severeClusters);
  deform.severeCollapseSet = new Set(severeCollapseClusters);
  deform.warningCount = warningClusters.length;
  deform.severeCount = severeClusters.length;
  deform.severeCollapseCount = severeCollapseClusters.length;
  deform.worstStretch = worstStretch;
  deform.worstAreaRatio = worstAreaRatio;
  deform.worstPoseError = worstPoseError;
  return deform;
}

export async function applySoftDeformationInterventionsGpuOnly({
  sim,
  soft,
  softClusterLoops,
  severeInterventionsOn,
  buildSoftDeformationState,
  stabilizeSeverelyDeformedSoftClusters,
  membraneClusterSet,
  deformationThresholds,
  wgslOffload,
}) {
  if (!sim || !soft || typeof buildSoftDeformationState !== 'function') {
    throw new Error('applySoftDeformationInterventionsGpuOnly requires sim, soft, and buildSoftDeformationState');
  }

  const prep = buildSoftDeformationInterventionsWgslPrep({ sim, soft, softClusterLoops });
  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    wgslOffload.state.preparedSoftDeformationPlan = prep.plan;
    wgslOffload.state.preparedSoftDeformationLayout = prep.layout;
    wgslOffload.state.preparedSoftDeformationSignature = prep.signature;
    wgslOffload.state.lastPreparedSoftDeformationNodeCount = prep.plan.nodeCount;
    wgslOffload.state.lastPreparedSoftDeformationLoopCount = prep.plan.loopCount;
    wgslOffload.state.lastPreparedSoftDeformationLoopNodeRefCount = prep.plan.loopNodeRefCount;
    wgslOffload.state.lastPreparedSoftDeformationSpringCount = prep.plan.springCount;
    wgslOffload.state.lastPreparedSoftDeformationClusterCount = prep.plan.clusterCount;
    wgslOffload.state.lastPreparedSoftDeformationLayoutBytes = prep.layout.byteLength;
    wgslOffload.state.lastPreparedSoftDeformationFrame = prep.plan.frame;
    wgslOffload.state.lastSourceRoute = 'cpu-soft-deformation-authoritative';
    wgslOffload.state.lastMode = 'cpu-soft-deformation-prepared';
  }

  let deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  let wgslApplied = false;
  if (wgslOffload?.enabled === true && wgslOffload?.device && wgslOffload?.state) {
    try {
      const metrics = await runSoftDeformationSpringMetricsWgsl({
        device: wgslOffload.device,
        prep,
        state: wgslOffload.state,
      });
      wgslApplied = applyAuthoritativeWgslSpringMetrics(deform, prep, metrics);
      if (wgslApplied) {
        refreshSoftDeformationClassificationFromMetrics({
          deform,
          membraneClusterSet,
          thresholds: deformationThresholds,
        });
        wgslOffload.state.lastSoftDeformationWgslMetricsSource = 'wgsl-soft-deformation-spring-metrics-authoritative';
      }
    } catch (err) {
      wgslOffload.state.lastSoftDeformationWgslMetricsSource = 'cpu-soft-deformation-metrics-fallback';
      wgslOffload.state.lastSoftDeformationWgslError = String(err?.message || err || 'unknown-error');
    }
  }

  if (severeInterventionsOn && deform?.severeCollapseCount > 0) {
    stabilizeSeverelyDeformedSoftClusters?.(sim, soft, softClusterLoops, deform);
    deform = buildSoftDeformationState(sim, soft, softClusterLoops);
    if (wgslApplied) {
      // Keep the post-stabilization deformation state aligned with authoritative
      // WGSL spring metrics when they are available and finite.
      try {
        const postPrep = buildSoftDeformationInterventionsWgslPrep({ sim, soft, softClusterLoops });
        const metrics = await runSoftDeformationSpringMetricsWgsl({
          device: wgslOffload.device,
          prep: postPrep,
          state: wgslOffload.state,
        });
        const postApplied = applyAuthoritativeWgslSpringMetrics(deform, postPrep, metrics);
        if (postApplied) {
          refreshSoftDeformationClassificationFromMetrics({
            deform,
            membraneClusterSet,
            thresholds: deformationThresholds,
          });
        }
      } catch {
        // fall back silently to CPU-derived metrics for post-stabilization state
      }
    }
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    wgslOffload.state.lastSoftDeformationSevereCollapseCount = Number(deform?.severeCollapseCount) || 0;
    wgslOffload.state.lastSourceRoute = wgslApplied
      ? 'wgsl-soft-deformation-spring-metrics-authoritative'
      : 'cpu-soft-deformation-authoritative';
    wgslOffload.state.lastMode = wgslApplied
      ? 'wgsl-soft-deformation-spring-metrics-authoritative'
      : 'cpu-soft-deformation-authoritative';
  }

  return deform;
}
