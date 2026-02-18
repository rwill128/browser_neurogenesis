/**
 * GPU-only runtime rigid post-integration stabilization.
 * Owns rigid linear/angular velocity capping before collision passes so this
 * stepping responsibility lives in isolated gpu-only runtime modules.
 *
 * Optional WGSL offload is isolated here for the clamp-only stage; baseline
 * remains untouched/default and callers can safely fall back to CPU.
 */

const WGSL_WORKGROUP_SIZE = 64;

const clampRigidVelocitiesWgsl = /* wgsl */`
struct Params {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  vCap: f32,
  wCap: f32,
  _pad3: f32,
  _pad4: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> vx: array<f32>;
@group(0) @binding(2) var<storage, read_write> vy: array<f32>;
@group(0) @binding(3) var<storage, read_write> omega: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  var vxv = vx[i];
  var vyv = vy[i];
  let speed = sqrt(vxv * vxv + vyv * vyv);
  if (speed > params.vCap) {
    let inv = params.vCap / max(speed, 1e-9);
    vxv = vxv * inv;
    vyv = vyv * inv;
  }

  var w = omega[i];
  w = max(-params.wCap, min(params.wCap, w));

  vx[i] = vxv;
  vy[i] = vyv;
  omega[i] = w;
}
`;

function cpuClampRigidPostIntegrate({ rigidBodies, velocityCap, omegaCap }) {
  const vCap = Math.max(0, Number(velocityCap) || 0);
  const wCap = Math.max(0, Number(omegaCap) || 0);

  for (const rb of rigidBodies) {
    const vmag = Math.hypot(rb.vx, rb.vy);
    if (vmag > vCap) {
      rb.vx = (rb.vx / vmag) * vCap;
      rb.vy = (rb.vy / vmag) * vCap;
    }

    rb.omega = Math.max(-wCap, Math.min(wCap, rb.omega || 0));
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
    const module = device.createShaderModule({ code: clampRigidVelocitiesWgsl });
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

    state.vx?.destroy?.();
    state.vy?.destroy?.();
    state.omega?.destroy?.();
    state.readVx?.destroy?.();
    state.readVy?.destroy?.();
    state.readOmega?.destroy?.();

    state.vx = createBuffer(device, bytes, storageUsage);
    state.vy = createBuffer(device, bytes, storageUsage);
    state.omega = createBuffer(device, bytes, storageUsage);

    state.readVx = createBuffer(device, bytes, readbackUsage);
    state.readVy = createBuffer(device, bytes, readbackUsage);
    state.readOmega = createBuffer(device, bytes, readbackUsage);

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
        { binding: 1, resource: { buffer: state.vx } },
        { binding: 2, resource: { buffer: state.vy } },
        { binding: 3, resource: { buffer: state.omega } },
      ],
    });
  }

  return state;
}

async function wgslClampRigidPostIntegrate({ rigidBodies, velocityCap, omegaCap, offload }) {
  const count = rigidBodies.length;
  if (count === 0) return;

  const state = await ensureWgslState(offload, count);
  const device = offload.device;
  const bytes = count * 4;

  const vx = new Float32Array(count);
  const vy = new Float32Array(count);
  const omega = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const rb = rigidBodies[i];
    vx[i] = Number(rb.vx);
    vy[i] = Number(rb.vy);
    omega[i] = Number(rb.omega) || 0;
  }

  const paramsBuffer = new ArrayBuffer(32);
  const paramsView = new DataView(paramsBuffer);
  paramsView.setUint32(0, count, true);
  paramsView.setFloat32(16, Math.max(0, Number(velocityCap) || 0), true);
  paramsView.setFloat32(20, Math.max(0, Number(omegaCap) || 0), true);

  device.queue.writeBuffer(state.params, 0, paramsBuffer);
  device.queue.writeBuffer(state.vx, 0, vx);
  device.queue.writeBuffer(state.vy, 0, vy);
  device.queue.writeBuffer(state.omega, 0, omega);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / WGSL_WORKGROUP_SIZE));
  pass.end();

  encoder.copyBufferToBuffer(state.vx, 0, state.readVx, 0, bytes);
  encoder.copyBufferToBuffer(state.vy, 0, state.readVy, 0, bytes);
  encoder.copyBufferToBuffer(state.omega, 0, state.readOmega, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    state.readVx.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.readVy.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.readOmega.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
  ]);

  const outVx = new Float32Array(state.readVx.getMappedRange(0, bytes).slice(0));
  const outVy = new Float32Array(state.readVy.getMappedRange(0, bytes).slice(0));
  const outOmega = new Float32Array(state.readOmega.getMappedRange(0, bytes).slice(0));

  state.readVx.unmap();
  state.readVy.unmap();
  state.readOmega.unmap();

  for (let i = 0; i < count; i++) {
    const rb = rigidBodies[i];
    rb.vx = outVx[i];
    rb.vy = outVy[i];
    rb.omega = outOmega[i];
  }
}

export async function stabilizeRigidPostIntegrateGpuOnly({
  rigidBodies,
  velocityCap = 4.0,
  omegaCap = 0.22,
  wgslOffload,
} = {}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) {
    return { mode: 'cpu', reason: 'empty' };
  }

  if (!canUseWgslOffload(wgslOffload)) {
    cpuClampRigidPostIntegrate({ rigidBodies, velocityCap, omegaCap });
    return { mode: 'cpu', reason: 'wgsl-unavailable' };
  }

  try {
    await wgslClampRigidPostIntegrate({ rigidBodies, velocityCap, omegaCap, offload: wgslOffload });
    if (wgslOffload?.state) {
      wgslOffload.state.lastError = null;
      wgslOffload.state.lastMode = 'wgsl';
    }
    return { mode: 'wgsl', reason: 'ok' };
  } catch (err) {
    cpuClampRigidPostIntegrate({ rigidBodies, velocityCap, omegaCap });
    if (wgslOffload?.state) {
      wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
      wgslOffload.state.lastMode = 'cpu-fallback';
    }
    return { mode: 'cpu-fallback', reason: String(err?.message || err || 'unknown-error') };
  }
}
