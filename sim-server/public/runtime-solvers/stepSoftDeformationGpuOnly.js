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

export function buildSoftDeformationInterventionsWgslPrep({ sim, soft, softClusterLoops }) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const loops = Array.isArray(softClusterLoops) ? softClusterLoops : [];
  const nodeCount = nodes.length;
  const loopCount = loops.length;

  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  const nodeInvMass = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i] || {};
    nodeX[i] = Number(node.x) || 0;
    nodeY[i] = Number(node.y) || 0;
    const mass = Number(node.mass);
    nodeInvMass[i] = Number.isFinite(mass) && mass > 0 ? (1 / mass) : 0;
  }

  const loopOffsets = new Uint32Array(loopCount + 1);
  let loopNodeRefCount = 0;
  for (let li = 0; li < loopCount; li++) {
    loopOffsets[li] = loopNodeRefCount;
    const loop = Array.isArray(loops[li]) ? loops[li] : [];
    loopNodeRefCount += loop.length;
  }
  loopOffsets[loopCount] = loopNodeRefCount;

  const loopNodeIndex = new Uint32Array(loopNodeRefCount);
  let write = 0;
  for (let li = 0; li < loopCount; li++) {
    const loop = Array.isArray(loops[li]) ? loops[li] : [];
    for (let j = 0; j < loop.length; j++) {
      const raw = Number(loop[j]);
      const idx = Number.isFinite(raw) ? Math.max(0, Math.min(nodeCount - 1, raw | 0)) : 0;
      loopNodeIndex[write] = idx >>> 0;
      write += 1;
    }
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
  signature ^= hashU32ArrayFnv1a(loopOffsets);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(loopNodeIndex);
  signature >>>= 0;

  const layout = { nodeX, nodeY, nodeInvMass, loopOffsets, loopNodeIndex };
  layout.byteLength = nodeX.byteLength
    + nodeY.byteLength
    + nodeInvMass.byteLength
    + loopOffsets.byteLength
    + loopNodeIndex.byteLength;

  return {
    plan: {
      frame: Number(sim?.frame) || 0,
      nodeCount,
      loopCount,
      loopNodeRefCount,
    },
    layout,
    signature,
  };
}

export function applySoftDeformationInterventionsGpuOnly({
  sim,
  soft,
  softClusterLoops,
  severeInterventionsOn,
  buildSoftDeformationState,
  stabilizeSeverelyDeformedSoftClusters,
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
    wgslOffload.state.lastPreparedSoftDeformationLayoutBytes = prep.layout.byteLength;
    wgslOffload.state.lastPreparedSoftDeformationFrame = prep.plan.frame;
    wgslOffload.state.lastSourceRoute = 'cpu-soft-deformation-authoritative';
    wgslOffload.state.lastMode = 'cpu-soft-deformation-prepared';
  }

  let deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  if (severeInterventionsOn && deform?.severeCollapseCount > 0) {
    stabilizeSeverelyDeformedSoftClusters?.(sim, soft, softClusterLoops, deform);
    deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    wgslOffload.state.lastSoftDeformationSevereCollapseCount = Number(deform?.severeCollapseCount) || 0;
    wgslOffload.state.lastMode = 'cpu-soft-deformation-authoritative';
  }

  return deform;
}
