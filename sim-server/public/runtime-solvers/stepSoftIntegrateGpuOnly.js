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
  const nodes = soft?.nodes || [];
  const count = nodes.length;
  if (count === 0) return;

  const state = await ensureWgslState(offload, count);
  const device = offload.device;
  const bytes = count * 4;

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

  state.readX.unmap();
  state.readY.unmap();
  state.readVx.unmap();
  state.readVy.unmap();

  for (let i = 0; i < count; i++) {
    const node = nodes[i];
    node.x = outX[i];
    node.y = outY[i];
    node.vx = outVx[i];
    node.vy = outVy[i];
    applyBounceBoundary(node, n, 0.78);
  }
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
  if (!canUseWgslOffload(wgslOffload)) {
    integrateSoftBodiesCpu({ soft, n, dt, softIntegrationScale, hybridNodeVCap, applyBounceBoundary });
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
      wgslOffload.state.lastMode = 'wgsl';
    }
    return { mode: 'wgsl', reason: 'ok' };
  } catch (err) {
    // Keep runtime behavior stable if the optional WGSL path fails in-session.
    integrateSoftBodiesCpu({ soft, n, dt, softIntegrationScale, hybridNodeVCap, applyBounceBoundary });
    if (wgslOffload?.state) {
      wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
      wgslOffload.state.lastMode = 'cpu-fallback';
    }
    return { mode: 'cpu-fallback', reason: String(err?.message || err || 'unknown-error') };
  }
}
