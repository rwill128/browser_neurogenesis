/**
 * GPU-only runtime soft integration path.
 * Isolated from baseline in gpu-lab.js so soft-body stepping responsibility
 * can migrate incrementally while baseline remains the default/reference path.
 *
 * This module now supports optional WebGPU/WGSL offload for the velocity-cap +
 * position-integration stage. Boundary bounce remains on CPU to preserve the
 * existing contract and avoid baseline-path coupling.
 */

const WGSL_WORKGROUP_SIZE = 64;

function addTimingSample(state, key, ms) {
  if (!state || !key) return;
  const timing = state.lastTiming && typeof state.lastTiming === 'object'
    ? state.lastTiming
    : (state.lastTiming = {});
  timing[key] = (Number(timing[key]) || 0) + Math.max(0, Number(ms) || 0);
}

const integrateSoftNodesWgsl = /* wgsl */`
struct Params {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  dt: f32,
  scale: f32,
  cap: f32,
  _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> vx: array<f32>;
@group(0) @binding(4) var<storage, read_write> vy: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  var vxv = vx[i];
  var vyv = vy[i];
  let speed = sqrt(vxv * vxv + vyv * vyv);
  if (speed > params.cap) {
    let inv = params.cap / max(speed, 1e-9);
    vxv = vxv * inv;
    vyv = vyv * inv;
  }

  x[i] = x[i] + vxv * params.dt * params.scale;
  y[i] = y[i] + vyv * params.dt * params.scale;
  vx[i] = vxv;
  vy[i] = vyv;
}
`;

function hashF32Fnv1a(arr) {
  const buf = new ArrayBuffer(4);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  let hash = 0x811c9dc5;
  for (let i = 0; i < arr.length; i++) {
    f[0] = Number(arr[i]) || 0;
    hash ^= u[0] >>> 0;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function computeSoftIntegrateProposalSignature({ x, y, vx, vy, dt, softIntegrationScale, hybridNodeVCap, n }) {
  let hash = 0x811c9dc5;
  hash ^= (x.length >>> 0);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32Fnv1a(x);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32Fnv1a(y);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32Fnv1a(vx);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32Fnv1a(vy);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  const scalars = new Float32Array([Number(dt) || 0, Number(softIntegrationScale) || 0, Number(hybridNodeVCap) || 0, Number(n) || 0]);
  hash ^= hashF32Fnv1a(scalars);
  return hash >>> 0;
}

function buildCpuReferenceIntegration({ x, y, vx, vy, dt, softIntegrationScale, hybridNodeVCap, n, applyBounceBoundary, nodes }) {
  const outX = new Float32Array(x);
  const outY = new Float32Array(y);
  const outVx = new Float32Array(vx);
  const outVy = new Float32Array(vy);

  for (let i = 0; i < outX.length; i++) {
    let vxv = outVx[i];
    let vyv = outVy[i];
    const speed = Math.hypot(vxv, vyv);
    if (speed > hybridNodeVCap) {
      const inv = hybridNodeVCap / Math.max(speed, 1e-9);
      vxv *= inv;
      vyv *= inv;
    }

    const tmpNode = {
      x: outX[i] + vxv * dt * softIntegrationScale,
      y: outY[i] + vyv * dt * softIntegrationScale,
      vx: vxv,
      vy: vyv,
      r: Number(nodes?.[i]?.r) || 0,
    };

    applyBounceBoundary(tmpNode, n, 0.78);
    outX[i] = Number(tmpNode.x) || 0;
    outY[i] = Number(tmpNode.y) || 0;
    outVx[i] = Number(tmpNode.vx) || 0;
    outVy[i] = Number(tmpNode.vy) || 0;
  }

  return { outX, outY, outVx, outVy };
}

function computeSoftIntegrateParity(expected, actual, epsilon = 1e-5) {
  let mismatchCount = 0;
  let maxAbsErr = 0;
  for (let i = 0; i < expected.length; i++) {
    const err = Math.abs((actual[i] || 0) - (expected[i] || 0));
    if (err > epsilon) mismatchCount += 1;
    if (err > maxAbsErr) maxAbsErr = err;
  }
  return { mismatchCount, maxAbsErr };
}

function integrateSoftBodiesCpu({ soft, n, dt, softIntegrationScale, hybridNodeVCap, applyBounceBoundary }) {
  for (const node of (soft?.nodes || [])) {
    const vmag = Math.hypot(node.vx, node.vy);
    if (vmag > hybridNodeVCap) {
      node.vx = (node.vx / vmag) * hybridNodeVCap;
      node.vy = (node.vy / vmag) * hybridNodeVCap;
    }
    node.x = node.x + node.vx * dt * softIntegrationScale;
    node.y = node.y + node.vy * dt * softIntegrationScale;
    applyBounceBoundary(node, n, 0.78);
  }
}

function normalizeGpuOnlyPipelineMode(offload) {
  const mode = String(offload?.modeProfile || '').trim().toLowerCase();
  if (mode === 'gpu-only-fast') return 'gpu-only-fast';
  if (mode === 'gpu-only-validated') return 'gpu-only-validated';
  if (mode === 'standard') return 'standard';
  return 'gpu-only-validated';
}

function isGpuOnlyFastMode(offload) {
  return normalizeGpuOnlyPipelineMode(offload) === 'gpu-only-fast';
}

function checkFiniteFloat32Array(values) {
  if (!(values instanceof Float32Array)) {
    return { allFinite: false, nonFiniteCount: 0, comparedCount: 0 };
  }
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

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined' || typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function createBuffer(device, size, usage) {
  return device.createBuffer({ size, usage });
}

async function ensureWgslState(offload, count) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;

  if (!state.pipeline) {
    const module = device.createShaderModule({ code: integrateSoftNodesWgsl });
    state.pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  const requiredCapacity = Math.max(1, count);
  if ((state.capacity || 0) < requiredCapacity) {
    const capacity = Math.max(requiredCapacity, state.capacity ? state.capacity * 2 : 256);
    const bytes = capacity * 4;
    const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.COPY_SRC;
    const readbackUsage = globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ;

    state.x?.destroy?.();
    state.y?.destroy?.();
    state.vx?.destroy?.();
    state.vy?.destroy?.();
    state.readX?.destroy?.();
    state.readY?.destroy?.();
    state.readVx?.destroy?.();
    state.readVy?.destroy?.();

    state.x = createBuffer(device, bytes, storageUsage);
    state.y = createBuffer(device, bytes, storageUsage);
    state.vx = createBuffer(device, bytes, storageUsage);
    state.vy = createBuffer(device, bytes, storageUsage);

    state.readX = createBuffer(device, bytes, readbackUsage);
    state.readY = createBuffer(device, bytes, readbackUsage);
    state.readVx = createBuffer(device, bytes, readbackUsage);
    state.readVy = createBuffer(device, bytes, readbackUsage);

    state.capacity = capacity;
    state.bindGroup = null;
  }

  if (!state.params) {
    state.params = createBuffer(
      device,
      32,
      globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    );
  }

  if (!state.bindGroup) {
    state.bindGroup = device.createBindGroup({
      layout: state.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.params } },
        { binding: 1, resource: { buffer: state.x } },
        { binding: 2, resource: { buffer: state.y } },
        { binding: 3, resource: { buffer: state.vx } },
        { binding: 4, resource: { buffer: state.vy } },
      ],
    });
  }

  return state;
}

async function integrateSoftBodiesWgsl({ soft, n, dt, softIntegrationScale, hybridNodeVCap, applyBounceBoundary, offload }) {
  const timingState = offload?.state || null;
  const fastMode = isGpuOnlyFastMode(offload);
  const nodes = soft?.nodes || [];
  const count = nodes.length;
  if (count === 0) return;

  const state = await ensureWgslState(offload, count);
  const device = offload.device;
  const bytes = count * 4;

  const prepArrayStartMs = performance.now();
  const x = new Float32Array(count);
  const y = new Float32Array(count);
  const vx = new Float32Array(count);
  const vy = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const node = nodes[i];
    x[i] = Number(node.x) || 0;
    y[i] = Number(node.y) || 0;
    vx[i] = Number(node.vx) || 0;
    vy[i] = Number(node.vy) || 0;
  }
  addTimingSample(timingState, 'prep.inputArraysMs', performance.now() - prepArrayStartMs);

  const paramsBuffer = new ArrayBuffer(32);
  const paramsView = new DataView(paramsBuffer);
  paramsView.setUint32(0, count, true);
  paramsView.setFloat32(16, dt, true);
  paramsView.setFloat32(20, softIntegrationScale, true);
  paramsView.setFloat32(24, hybridNodeVCap, true);

  device.queue.writeBuffer(state.params, 0, paramsBuffer);
  device.queue.writeBuffer(state.x, 0, x);
  device.queue.writeBuffer(state.y, 0, y);
  device.queue.writeBuffer(state.vx, 0, vx);
  device.queue.writeBuffer(state.vy, 0, vy);

  const wgslDispatchStartMs = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / WGSL_WORKGROUP_SIZE));
  pass.end();

  encoder.copyBufferToBuffer(state.x, 0, state.readX, 0, bytes);
  encoder.copyBufferToBuffer(state.y, 0, state.readY, 0, bytes);
  encoder.copyBufferToBuffer(state.vx, 0, state.readVx, 0, bytes);
  encoder.copyBufferToBuffer(state.vy, 0, state.readVy, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    state.readX.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.readY.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.readVx.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.readVy.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
  ]);

  const outX = new Float32Array(state.readX.getMappedRange(0, bytes).slice(0));
  const outY = new Float32Array(state.readY.getMappedRange(0, bytes).slice(0));
  const outVx = new Float32Array(state.readVx.getMappedRange(0, bytes).slice(0));
  const outVy = new Float32Array(state.readVy.getMappedRange(0, bytes).slice(0));

  const proposalSignature = computeSoftIntegrateProposalSignature({ x, y, vx, vy, dt, softIntegrationScale, hybridNodeVCap, n });
  const finiteX = checkFiniteFloat32Array(outX);
  const finiteY = checkFiniteFloat32Array(outY);
  const finiteVx = checkFiniteFloat32Array(outVx);
  const finiteVy = checkFiniteFloat32Array(outVy);

  state.readX.unmap();
  state.readY.unmap();
  state.readVx.unmap();
  state.readVy.unmap();
  addTimingSample(timingState, 'wgsl.dispatchAndReadbackMs', performance.now() - wgslDispatchStartMs);

  state.lastProposalSignature = proposalSignature >>> 0;
  state.lastFinite = { x: finiteX, y: finiteY, vx: finiteVx, vy: finiteVy };
  state.lastSourceRoute = fastMode ? 'wgsl-integrate-proposal-fast' : 'wgsl-integrate-proposal';

  if (!finiteX.allFinite || !finiteY.allFinite || !finiteVx.allFinite || !finiteVy.allFinite) {
    state.lastParity = {
      source: fastMode ? 'wgsl-integrate-proposal-fast' : 'wgsl-integrate-proposal',
      proposalSignature: proposalSignature >>> 0,
      validation: 'non-finite',
      finite: state.lastFinite,
    };
    throw new Error('wgsl-integrate produced non-finite output');
  }

  let cpuRef = null;
  if (!fastMode) {
    cpuRef = buildCpuReferenceIntegration({
      x,
      y,
      vx,
      vy,
      dt,
      softIntegrationScale,
      hybridNodeVCap,
      n,
      applyBounceBoundary,
      nodes,
    });
  }

  const applyOutputStartMs = performance.now();
  for (let i = 0; i < count; i++) {
    const node = nodes[i];
    node.x = outX[i];
    node.y = outY[i];
    node.vx = outVx[i];
    node.vy = outVy[i];
    applyBounceBoundary(node, n, 0.78);
  }
  addTimingSample(timingState, 'cpu.applyOutputsMs', performance.now() - applyOutputStartMs);

  if (fastMode) {
    state.lastParity = {
      source: 'wgsl-integrate-proposal-fast',
      proposalSignature: proposalSignature >>> 0,
      validation: 'skipped-cpu-parity',
      finite: state.lastFinite,
    };
    return;
  }
  const parityX = computeSoftIntegrateParity(cpuRef.outX, outX);
  const parityY = computeSoftIntegrateParity(cpuRef.outY, outY);
  const parityVx = computeSoftIntegrateParity(cpuRef.outVx, outVx);
  const parityVy = computeSoftIntegrateParity(cpuRef.outVy, outVy);
  state.lastParity = {
    source: 'wgsl-integrate-proposal-vs-cpu',
    proposalSignature: proposalSignature >>> 0,
    mismatchCount: parityX.mismatchCount + parityY.mismatchCount + parityVx.mismatchCount + parityVy.mismatchCount,
    maxAbsErr: Math.max(parityX.maxAbsErr, parityY.maxAbsErr, parityVx.maxAbsErr, parityVy.maxAbsErr),
    x: parityX,
    y: parityY,
    vx: parityVx,
    vy: parityVy,
    finite: state.lastFinite,
  };
}

export async function integrateSoftBodiesGpuOnly({
  soft,
  n,
  dt,
  softIntegrationScale,
  hybridNodeVCap,
  applyBounceBoundary,
  wgslOffload,
}) {
  const timingState = wgslOffload?.state || null;
  if (timingState) timingState.lastTiming = {};
  const stageStartMs = performance.now();
  const pipelineMode = normalizeGpuOnlyPipelineMode(wgslOffload);

  if (pipelineMode === 'standard') {
    const cpuIntegrateStartMs = performance.now();
    integrateSoftBodiesCpu({ soft, n, dt, softIntegrationScale, hybridNodeVCap, applyBounceBoundary });
    addTimingSample(timingState, 'cpu.integrateMs', performance.now() - cpuIntegrateStartMs);
    addTimingSample(timingState, 'totalMs', performance.now() - stageStartMs);
    if (wgslOffload?.state) {
      wgslOffload.state.lastError = null;
      wgslOffload.state.lastMode = 'cpu-standard';
      wgslOffload.state.lastSourceRoute = 'cpu-standard-authoritative';
      wgslOffload.state.lastParity = {
        source: 'cpu-standard-authoritative',
        validation: 'baseline-reference',
      };
    }
    return { mode: 'cpu-standard', reason: 'pipeline-standard' };
  }

  if (!canUseWgslOffload(wgslOffload)) {
    const cpuIntegrateStartMs = performance.now();
    integrateSoftBodiesCpu({ soft, n, dt, softIntegrationScale, hybridNodeVCap, applyBounceBoundary });
    addTimingSample(timingState, 'cpu.integrateMs', performance.now() - cpuIntegrateStartMs);
    addTimingSample(timingState, 'totalMs', performance.now() - stageStartMs);
    return { mode: 'cpu', reason: 'wgsl-unavailable' };
  }

  try {
    await integrateSoftBodiesWgsl({
      soft,
      n,
      dt,
      softIntegrationScale,
      hybridNodeVCap,
      applyBounceBoundary,
      offload: wgslOffload,
    });
    if (wgslOffload?.state) {
      wgslOffload.state.lastError = null;
      wgslOffload.state.lastMode = isGpuOnlyFastMode(wgslOffload) ? 'wgsl-fast' : 'wgsl-validated';
      wgslOffload.state.lastSourceRoute = isGpuOnlyFastMode(wgslOffload)
        ? 'wgsl-integrate-authoritative-fast'
        : 'wgsl-integrate-authoritative';
    }
    addTimingSample(timingState, 'totalMs', performance.now() - stageStartMs);
    return { mode: 'wgsl', reason: 'ok' };
  } catch (err) {
    // Keep runtime behavior stable if the optional WGSL path fails in-session.
    const cpuFallbackStartMs = performance.now();
    integrateSoftBodiesCpu({ soft, n, dt, softIntegrationScale, hybridNodeVCap, applyBounceBoundary });
    addTimingSample(timingState, 'cpu.fallbackIntegrateMs', performance.now() - cpuFallbackStartMs);
    addTimingSample(timingState, 'totalMs', performance.now() - stageStartMs);
    if (wgslOffload?.state) {
      wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
      wgslOffload.state.lastMode = 'cpu-fallback';
      wgslOffload.state.lastSourceRoute = 'cpu-fallback-authoritative';
    }
    return { mode: 'cpu-fallback', reason: String(err?.message || err || 'unknown-error') };
  }
}
