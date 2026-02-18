/**
 * GPU-only runtime rigid↔soft hybrid attachment constraint pass.
 *
 * Isolates weld-like rigid-soft hybrid spring stepping behind the gpu-only
 * runtime solver path while preserving baseline/default behavior contracts.
 */

const WGSL_WORKGROUP_SIZE = 64;

const hybridAttachmentErrorProbeWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  attachmentCount: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> nodeIndex: array<u32>;
@group(0) @binding(4) var<storage, read> anchorAX: array<f32>;
@group(0) @binding(5) var<storage, read> anchorAY: array<f32>;
@group(0) @binding(6) var<storage, read> anchorBX: array<f32>;
@group(0) @binding(7) var<storage, read> anchorBY: array<f32>;
@group(0) @binding(8) var<storage, read> restA: array<f32>;
@group(0) @binding(9) var<storage, read> restB: array<f32>;
@group(0) @binding(10) var<storage, read_write> maxErrorOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ai = gid.x;
  if (ai >= params.attachmentCount) { return; }

  let ni = nodeIndex[ai];
  if (ni >= params.nodeCount) {
    maxErrorOut[ai] = 0.0;
    return;
  }

  let px = nodeX[ni];
  let py = nodeY[ni];

  let dax = px - anchorAX[ai];
  let day = py - anchorAY[ai];
  let dbx = px - anchorBX[ai];
  let dby = py - anchorBY[ai];

  let distA = max(length(vec2<f32>(dax, day)), 1e-6);
  let distB = max(length(vec2<f32>(dbx, dby)), 1e-6);

  let errA = abs(distA - max(restA[ai], 0.0));
  let errB = abs(distB - max(restB[ai], 0.0));
  maxErrorOut[ai] = max(errA, errB);
}
`;



const hybridAttachmentVelocityProposalWgsl = /* wgsl */`
struct VelocityParams {
  nodeCount: u32,
  attachmentCount: u32,
  _pad0: u32,
  _pad1: u32,
  nodeErrorGain: f32,
  nodeImpulseScale: f32,
  dtNorm: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: VelocityParams;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> nodeIndex: array<u32>;
@group(0) @binding(4) var<storage, read> anchorAX: array<f32>;
@group(0) @binding(5) var<storage, read> anchorAY: array<f32>;
@group(0) @binding(6) var<storage, read> anchorBX: array<f32>;
@group(0) @binding(7) var<storage, read> anchorBY: array<f32>;
@group(0) @binding(8) var<storage, read> restA: array<f32>;
@group(0) @binding(9) var<storage, read> restB: array<f32>;
@group(0) @binding(10) var<storage, read_write> deltaVxOut: array<f32>;
@group(0) @binding(11) var<storage, read_write> deltaVyOut: array<f32>;

fn accumulate(anchorX: f32, anchorY: f32, restLen: f32, px: f32, py: f32, inoutDx: ptr<function, f32>, inoutDy: ptr<function, f32>) {
  let dx = px - anchorX;
  let dy = py - anchorY;
  let d = max(length(vec2<f32>(dx, dy)), 1e-6);
  let err = (d - max(restLen, 0.0)) * params.nodeErrorGain;
  let nx = dx / d;
  let ny = dy / d;
  *inoutDx = *inoutDx - nx * err * params.nodeImpulseScale * params.dtNorm;
  *inoutDy = *inoutDy - ny * err * params.nodeImpulseScale * params.dtNorm;
}

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ai = gid.x;
  if (ai >= params.attachmentCount) { return; }

  let ni = nodeIndex[ai];
  if (ni >= params.nodeCount) {
    deltaVxOut[ai] = 0.0;
    deltaVyOut[ai] = 0.0;
    return;
  }

  let px = nodeX[ni];
  let py = nodeY[ni];

  var deltaVx = 0.0;
  var deltaVy = 0.0;
  accumulate(anchorAX[ai], anchorAY[ai], restA[ai], px, py, &deltaVx, &deltaVy);
  accumulate(anchorBX[ai], anchorBY[ai], restB[ai], px, py, &deltaVx, &deltaVy);

  deltaVxOut[ai] = deltaVx;
  deltaVyOut[ai] = deltaVy;
}
`;
function clampFinite(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.state) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined') return false;
  if (typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function ensureHybridAttachmentProbeBuffers(offload, nodeCount, attachmentCount) {
  const state = offload.state;
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;

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

  const requiredAttachmentCapacity = Math.max(1, attachmentCount);
  if ((state.probeAttachmentCapacity || 0) < requiredAttachmentCapacity) {
    const capacity = Math.max(requiredAttachmentCapacity, state.probeAttachmentCapacity ? state.probeAttachmentCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.probeNodeIndex?.destroy?.();
    state.probeAnchorAX?.destroy?.();
    state.probeAnchorAY?.destroy?.();
    state.probeAnchorBX?.destroy?.();
    state.probeAnchorBY?.destroy?.();
    state.probeRestA?.destroy?.();
    state.probeRestB?.destroy?.();
    state.probeMaxErrorOut?.destroy?.();
    state.probeMaxErrorReadback?.destroy?.();
    state.probeNodeIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeAnchorAX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeAnchorAY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeAnchorBX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeAnchorBY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeRestA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeRestB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeMaxErrorOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.probeMaxErrorReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.probeAttachmentCapacity = capacity;
    state.probeBindGroup = null;
  }

  if (!state.probeParams) {
    state.probeParams = device.createBuffer({
      size: 16,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.probeBindGroup = null;
  }

  return state;
}

async function dispatchHybridAttachmentErrorProbe(offload, prep, soft) {
  const state = offload.state;
  const device = offload.device;
  const nodeCount = prep.plan.softNodeCount >>> 0;
  const attachmentCount = prep.plan.attachmentCount >>> 0;
  if (attachmentCount === 0 || nodeCount === 0) return;

  ensureHybridAttachmentProbeBuffers(offload, nodeCount, attachmentCount);

  if (!state.probePipeline) {
    state.probeShaderModule = device.createShaderModule({ code: hybridAttachmentErrorProbeWgsl });
    state.probePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: state.probeShaderModule,
        entryPoint: 'main',
      },
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
        { binding: 3, resource: { buffer: state.probeNodeIndex } },
        { binding: 4, resource: { buffer: state.probeAnchorAX } },
        { binding: 5, resource: { buffer: state.probeAnchorAY } },
        { binding: 6, resource: { buffer: state.probeAnchorBX } },
        { binding: 7, resource: { buffer: state.probeAnchorBY } },
        { binding: 8, resource: { buffer: state.probeRestA } },
        { binding: 9, resource: { buffer: state.probeRestB } },
        { binding: 10, resource: { buffer: state.probeMaxErrorOut } },
      ],
    });
  }

  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const n = soft.nodes[i];
    nodeX[i] = clampFinite(n?.x);
    nodeY[i] = clampFinite(n?.y);
  }

  const params = new Uint32Array(4);
  params[0] = nodeCount;
  params[1] = attachmentCount;

  device.queue.writeBuffer(state.probeParams, 0, params.buffer, params.byteOffset, params.byteLength);
  device.queue.writeBuffer(state.probeNodeX, 0, nodeX.buffer, nodeX.byteOffset, nodeX.byteLength);
  device.queue.writeBuffer(state.probeNodeY, 0, nodeY.buffer, nodeY.byteOffset, nodeY.byteLength);
  device.queue.writeBuffer(state.probeNodeIndex, 0, prep.layout.nodeIndex.buffer, prep.layout.nodeIndex.byteOffset, prep.layout.nodeIndex.byteLength);
  device.queue.writeBuffer(state.probeAnchorAX, 0, prep.layout.anchorAX.buffer, prep.layout.anchorAX.byteOffset, prep.layout.anchorAX.byteLength);
  device.queue.writeBuffer(state.probeAnchorAY, 0, prep.layout.anchorAY.buffer, prep.layout.anchorAY.byteOffset, prep.layout.anchorAY.byteLength);
  device.queue.writeBuffer(state.probeAnchorBX, 0, prep.layout.anchorBX.buffer, prep.layout.anchorBX.byteOffset, prep.layout.anchorBX.byteLength);
  device.queue.writeBuffer(state.probeAnchorBY, 0, prep.layout.anchorBY.buffer, prep.layout.anchorBY.byteOffset, prep.layout.anchorBY.byteLength);
  device.queue.writeBuffer(state.probeRestA, 0, prep.layout.restA.buffer, prep.layout.restA.byteOffset, prep.layout.restA.byteLength);
  device.queue.writeBuffer(state.probeRestB, 0, prep.layout.restB.buffer, prep.layout.restB.byteOffset, prep.layout.restB.byteLength);

  const bytes = attachmentCount * 4;
  const encoder = device.createCommandEncoder({ label: 'hybrid-attachment-probe-encoder' });
  const pass = encoder.beginComputePass({ label: 'hybrid-attachment-probe-pass' });
  pass.setPipeline(state.probePipeline);
  pass.setBindGroup(0, state.probeBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(attachmentCount / WGSL_WORKGROUP_SIZE)));
  pass.end();
  encoder.copyBufferToBuffer(state.probeMaxErrorOut, 0, state.probeMaxErrorReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.probeMaxErrorReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = state.probeMaxErrorReadback.getMappedRange(0, bytes);
  const errors = new Float32Array(mapped.slice(0));
  state.probeMaxErrorReadback.unmap();

  let maxError = 0;
  let sumError = 0;
  for (let i = 0; i < attachmentCount; i++) {
    const value = clampFinite(errors[i]);
    if (value > maxError) maxError = value;
    sumError += value;
  }
  state.lastProbeMaxAttachmentError = maxError;
  state.lastProbeAvgAttachmentError = attachmentCount > 0 ? sumError / attachmentCount : 0;
  state.lastProbeAttachmentCount = attachmentCount;
  state.lastProbeMode = 'wgsl-error-probe';
}



function buildCpuHybridAttachmentVelocityDelta({ layout, soft, nodeErrorGain, nodeImpulseScale, dtNorm }) {
  const attachmentCount = layout?.nodeIndex?.length || 0;
  const cellDeltaVx = new Float32Array(attachmentCount);
  const cellDeltaVy = new Float32Array(attachmentCount);
  for (let i = 0; i < attachmentCount; i++) {
    const node = soft?.nodes?.[layout.nodeIndex[i]];
    if (!node) continue;

    const px = clampFinite(node.x);
    const py = clampFinite(node.y);
    const runPair = (anchorX, anchorY, restLen) => {
      const dx = px - anchorX;
      const dy = py - anchorY;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const err = (d - Math.max(0, restLen)) * nodeErrorGain;
      const nx = dx / d;
      const ny = dy / d;
      cellDeltaVx[i] -= nx * err * nodeImpulseScale * dtNorm;
      cellDeltaVy[i] -= ny * err * nodeImpulseScale * dtNorm;
    };

    runPair(layout.anchorAX[i], layout.anchorAY[i], layout.restA[i]);
    runPair(layout.anchorBX[i], layout.anchorBY[i], layout.restB[i]);
  }
  return { cellDeltaVx, cellDeltaVy };
}

function computeHybridVelocityDeltaParity(cpuDelta, wgslDelta) {
  const count = Math.min(cpuDelta?.cellDeltaVx?.length || 0, wgslDelta?.cellDeltaVx?.length || 0);
  let maxAbsError = 0;
  let sumAbsError = 0;
  for (let i = 0; i < count; i++) {
    const errX = Math.abs(clampFinite(wgslDelta.cellDeltaVx[i]) - clampFinite(cpuDelta.cellDeltaVx[i]));
    const errY = Math.abs(clampFinite(wgslDelta.cellDeltaVy[i]) - clampFinite(cpuDelta.cellDeltaVy[i]));
    const err = Math.max(errX, errY);
    if (err > maxAbsError) maxAbsError = err;
    sumAbsError += err;
  }
  return {
    comparedAttachmentCount: count,
    maxAbsError,
    avgAbsError: count > 0 ? sumAbsError / count : 0,
  };
}

function ensureHybridAttachmentVelocityProposalBuffers(offload, nodeCount, attachmentCount) {
  const state = offload.state;
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.velocityNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.velocityNodeCapacity ? state.velocityNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.velocityNodeX?.destroy?.();
    state.velocityNodeY?.destroy?.();
    state.velocityNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityNodeCapacity = capacity;
    state.velocityBindGroup = null;
  }

  const requiredAttachmentCapacity = Math.max(1, attachmentCount);
  if ((state.velocityAttachmentCapacity || 0) < requiredAttachmentCapacity) {
    const capacity = Math.max(requiredAttachmentCapacity, state.velocityAttachmentCapacity ? state.velocityAttachmentCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.velocityNodeIndex?.destroy?.();
    state.velocityAnchorAX?.destroy?.();
    state.velocityAnchorAY?.destroy?.();
    state.velocityAnchorBX?.destroy?.();
    state.velocityAnchorBY?.destroy?.();
    state.velocityRestA?.destroy?.();
    state.velocityRestB?.destroy?.();
    state.velocityDeltaVxOut?.destroy?.();
    state.velocityDeltaVyOut?.destroy?.();
    state.velocityDeltaVxReadback?.destroy?.();
    state.velocityDeltaVyReadback?.destroy?.();

    state.velocityNodeIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityAnchorAX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityAnchorAY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityAnchorBX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityAnchorBY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityRestA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityRestB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.velocityDeltaVxOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.velocityDeltaVyOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.velocityDeltaVxReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.velocityDeltaVyReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });

    state.velocityAttachmentCapacity = capacity;
    state.velocityBindGroup = null;
  }

  if (!state.velocityParams) {
    state.velocityParams = device.createBuffer({
      size: 32,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.velocityBindGroup = null;
  }

  return state;
}

async function dispatchHybridAttachmentVelocityDeltaProposal(offload, prep, soft, paramsConfig) {
  const state = offload.state;
  const device = offload.device;
  const nodeCount = prep.plan.softNodeCount >>> 0;
  const attachmentCount = prep.plan.attachmentCount >>> 0;
  if (attachmentCount === 0 || nodeCount === 0) return;

  const nodeErrorGain = clampFinite(paramsConfig?.nodeErrorGain, 0.74);
  const nodeImpulseScale = clampFinite(paramsConfig?.nodeImpulseScale, 0.052);
  const dtNorm = clampFinite(paramsConfig?.dtNorm, 0);

  ensureHybridAttachmentVelocityProposalBuffers(offload, nodeCount, attachmentCount);

  if (!state.velocityPipeline) {
    state.velocityShaderModule = device.createShaderModule({ code: hybridAttachmentVelocityProposalWgsl });
    state.velocityPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: state.velocityShaderModule,
        entryPoint: 'main',
      },
    });
    state.velocityBindGroup = null;
  }

  if (!state.velocityBindGroup) {
    state.velocityBindGroup = device.createBindGroup({
      layout: state.velocityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.velocityParams } },
        { binding: 1, resource: { buffer: state.velocityNodeX } },
        { binding: 2, resource: { buffer: state.velocityNodeY } },
        { binding: 3, resource: { buffer: state.velocityNodeIndex } },
        { binding: 4, resource: { buffer: state.velocityAnchorAX } },
        { binding: 5, resource: { buffer: state.velocityAnchorAY } },
        { binding: 6, resource: { buffer: state.velocityAnchorBX } },
        { binding: 7, resource: { buffer: state.velocityAnchorBY } },
        { binding: 8, resource: { buffer: state.velocityRestA } },
        { binding: 9, resource: { buffer: state.velocityRestB } },
        { binding: 10, resource: { buffer: state.velocityDeltaVxOut } },
        { binding: 11, resource: { buffer: state.velocityDeltaVyOut } },
      ],
    });
  }

  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const n = soft.nodes[i];
    nodeX[i] = clampFinite(n?.x);
    nodeY[i] = clampFinite(n?.y);
  }

  const paramPack = new ArrayBuffer(32);
  const paramsU32 = new Uint32Array(paramPack, 0, 4);
  const paramsF32 = new Float32Array(paramPack, 16, 4);
  paramsU32[0] = nodeCount;
  paramsU32[1] = attachmentCount;
  paramsF32[0] = nodeErrorGain;
  paramsF32[1] = nodeImpulseScale;
  paramsF32[2] = dtNorm;

  device.queue.writeBuffer(state.velocityParams, 0, paramPack, 0, paramPack.byteLength);
  device.queue.writeBuffer(state.velocityNodeX, 0, nodeX.buffer, nodeX.byteOffset, nodeX.byteLength);
  device.queue.writeBuffer(state.velocityNodeY, 0, nodeY.buffer, nodeY.byteOffset, nodeY.byteLength);
  device.queue.writeBuffer(state.velocityNodeIndex, 0, prep.layout.nodeIndex.buffer, prep.layout.nodeIndex.byteOffset, prep.layout.nodeIndex.byteLength);
  device.queue.writeBuffer(state.velocityAnchorAX, 0, prep.layout.anchorAX.buffer, prep.layout.anchorAX.byteOffset, prep.layout.anchorAX.byteLength);
  device.queue.writeBuffer(state.velocityAnchorAY, 0, prep.layout.anchorAY.buffer, prep.layout.anchorAY.byteOffset, prep.layout.anchorAY.byteLength);
  device.queue.writeBuffer(state.velocityAnchorBX, 0, prep.layout.anchorBX.buffer, prep.layout.anchorBX.byteOffset, prep.layout.anchorBX.byteLength);
  device.queue.writeBuffer(state.velocityAnchorBY, 0, prep.layout.anchorBY.buffer, prep.layout.anchorBY.byteOffset, prep.layout.anchorBY.byteLength);
  device.queue.writeBuffer(state.velocityRestA, 0, prep.layout.restA.buffer, prep.layout.restA.byteOffset, prep.layout.restA.byteLength);
  device.queue.writeBuffer(state.velocityRestB, 0, prep.layout.restB.buffer, prep.layout.restB.byteOffset, prep.layout.restB.byteLength);

  const bytes = attachmentCount * 4;
  const encoder = device.createCommandEncoder({ label: 'hybrid-attachment-velocity-proposal-encoder' });
  const pass = encoder.beginComputePass({ label: 'hybrid-attachment-velocity-proposal-pass' });
  pass.setPipeline(state.velocityPipeline);
  pass.setBindGroup(0, state.velocityBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(attachmentCount / WGSL_WORKGROUP_SIZE)));
  pass.end();
  encoder.copyBufferToBuffer(state.velocityDeltaVxOut, 0, state.velocityDeltaVxReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.velocityDeltaVyOut, 0, state.velocityDeltaVyReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    state.velocityDeltaVxReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.velocityDeltaVyReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
  ]);

  const mappedDeltaVx = state.velocityDeltaVxReadback.getMappedRange(0, bytes);
  const mappedDeltaVy = state.velocityDeltaVyReadback.getMappedRange(0, bytes);
  const deltaVx = new Float32Array(mappedDeltaVx.slice(0));
  const deltaVy = new Float32Array(mappedDeltaVy.slice(0));
  state.velocityDeltaVxReadback.unmap();
  state.velocityDeltaVyReadback.unmap();

  const cpuDelta = buildCpuHybridAttachmentVelocityDelta({
    layout: prep.layout,
    soft,
    nodeErrorGain,
    nodeImpulseScale,
    dtNorm,
  });
  const wgslDelta = { cellDeltaVx: deltaVx, cellDeltaVy: deltaVy };
  state.lastVelocityDeltaCpuReference = cpuDelta;
  state.lastVelocityDeltaTelemetry = wgslDelta;
  state.lastVelocityDeltaParity = {
    source: 'wgsl-attachment-proposal',
    ...computeHybridVelocityDeltaParity(cpuDelta, wgslDelta),
  };
  state.lastVelocityDeltaProposalSource = 'wgsl-attachment-proposal';
}

export function buildHybridAttachmentWgslPrep({ rigidBodies, soft, hybrid, rigidVertexWorld }) {
  const attachments = Array.isArray(hybrid) ? hybrid : [];
  const count = attachments.length;

  const nodeIndex = new Uint32Array(count);
  const rigidIndex = new Uint32Array(count);
  const anchorAX = new Float32Array(count);
  const anchorAY = new Float32Array(count);
  const anchorBX = new Float32Array(count);
  const anchorBY = new Float32Array(count);
  const restA = new Float32Array(count);
  const restB = new Float32Array(count);

  let validCount = 0;
  for (let i = 0; i < count; i++) {
    const h = attachments[i];
    const rb = rigidBodies[h?.rigidIndex];
    const node = soft?.nodes?.[h?.nodeIndex];
    if (!rb || !node) continue;

    const va = rigidVertexWorld(rb, h.vertexA);
    const vb = rigidVertexWorld(rb, h.vertexB);

    nodeIndex[i] = h.nodeIndex >>> 0;
    rigidIndex[i] = h.rigidIndex >>> 0;
    anchorAX[i] = clampFinite(va?.x);
    anchorAY[i] = clampFinite(va?.y);
    anchorBX[i] = clampFinite(vb?.x);
    anchorBY[i] = clampFinite(vb?.y);
    restA[i] = Math.max(0, clampFinite(h.restA));
    restB[i] = Math.max(0, clampFinite(h.restB));
    validCount += 1;
  }

  const layout = {
    nodeIndex,
    rigidIndex,
    anchorAX,
    anchorAY,
    anchorBX,
    anchorBY,
    restA,
    restB,
  };
  layout.byteLength = Object.values(layout).reduce((sum, arr) => sum + (arr?.byteLength || 0), 0);

  return {
    plan: {
      attachmentCount: count,
      validAttachmentCount: validCount,
      rigidBodyCount: Array.isArray(rigidBodies) ? rigidBodies.length : 0,
      softNodeCount: Array.isArray(soft?.nodes) ? soft.nodes.length : 0,
    },
    layout,
  };
}

export function applyHybridAttachmentConstraintsGpuOnly({
  rigidBodies,
  soft,
  hybrid,
  rigidVertexWorld,
  dtNorm,
  iterations = 5,
  nodeErrorGain = 0.74,
  nodeImpulseScale = 0.052,
  rigidImpulseScale = 0.0075,
  rigidAngularScale = 0.00075,
  wgslOffload,
}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return;
  if (!soft || !Array.isArray(soft.nodes) || soft.nodes.length === 0) return;
  if (!Array.isArray(hybrid) || hybrid.length === 0) return;
  if (typeof rigidVertexWorld !== 'function') {
    throw new Error('gpu-only hybrid constraint pass requires rigidVertexWorld callback');
  }

  const safeDtNorm = Number.isFinite(dtNorm) ? dtNorm : 0;
  const totalIterations = Math.max(0, Number(iterations) | 0);
  if (totalIterations <= 0 || safeDtNorm === 0) return;

  let prep = null;
  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    prep = buildHybridAttachmentWgslPrep({ rigidBodies, soft, hybrid, rigidVertexWorld });
    wgslOffload.state.preparedPlan = prep.plan;
    wgslOffload.state.preparedLayout = prep.layout;
    wgslOffload.state.lastPreparedAttachmentCount = prep.plan.attachmentCount;
    wgslOffload.state.lastPreparedValidAttachmentCount = prep.plan.validAttachmentCount;
    wgslOffload.state.lastPreparedRigidBodyCount = prep.plan.rigidBodyCount;
    wgslOffload.state.lastPreparedSoftNodeCount = prep.plan.softNodeCount;
    wgslOffload.state.lastPreparedLayoutBytes = prep.layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared';
    wgslOffload.state.lastError = null;

    if (canUseWgslOffload(wgslOffload)) {
      const serializedDispatch = (wgslOffload.state.pendingWgslProbePromise || Promise.resolve())
        .catch(() => {})
        .then(async () => {
          await dispatchHybridAttachmentErrorProbe(wgslOffload, prep, soft);
          await dispatchHybridAttachmentVelocityDeltaProposal(wgslOffload, prep, soft, {
            nodeErrorGain,
            nodeImpulseScale,
            dtNorm: safeDtNorm,
          });
        });
      wgslOffload.state.pendingWgslProbePromise = serializedDispatch;
      serializedDispatch
        .then(() => {
          wgslOffload.state.lastMode = 'wgsl-velocity-proposal';
          wgslOffload.state.lastError = null;
        })
        .catch((error) => {
          wgslOffload.state.lastMode = 'cpu-fallback';
          wgslOffload.state.lastError = error?.message || String(error);
        });
    }
  }

  for (let iter = 0; iter < totalIterations; iter++) {
    for (const h of hybrid) {
      const rb = rigidBodies[h?.rigidIndex];
      const node = soft.nodes[h?.nodeIndex];
      if (!rb || !node) continue;

      const va = rigidVertexWorld(rb, h.vertexA);
      const vb = rigidVertexWorld(rb, h.vertexB);
      const pairs = [[va, h.restA], [vb, h.restB]];
      for (const [anchor, rest] of pairs) {
        const dx = node.x - anchor.x;
        const dy = node.y - anchor.y;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const err = (d - rest) * nodeErrorGain;
        const nx = dx / d;
        const ny = dy / d;

        node.vx -= nx * err * nodeImpulseScale * safeDtNorm;
        node.vy -= ny * err * nodeImpulseScale * safeDtNorm;

        rb.vx += nx * err * rigidImpulseScale * safeDtNorm;
        rb.vy += ny * err * rigidImpulseScale * safeDtNorm;
        rb.omega = (rb.omega || 0) + (nx * ny) * err * rigidAngularScale * safeDtNorm;
      }
    }
  }
}
