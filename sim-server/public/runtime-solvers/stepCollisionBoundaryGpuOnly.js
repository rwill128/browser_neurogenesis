const WGSL_WORKGROUP_SIZE = 64;

const bounceBoundaryWgsl = /* wgsl */`
struct Params {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  n: f32,
  damping: f32,
  hasOmega: f32,
  _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> posX: array<f32>;
@group(0) @binding(2) var<storage, read_write> posY: array<f32>;
@group(0) @binding(3) var<storage, read_write> velX: array<f32>;
@group(0) @binding(4) var<storage, read_write> velY: array<f32>;
@group(0) @binding(5) var<storage, read_write> omega: array<f32>;
@group(0) @binding(6) var<storage, read> radius: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  var x = posX[i];
  var y = posY[i];
  var vx = velX[i];
  var vy = velY[i];
  var w = omega[i];

  let r = radius[i];
  let minX = r;
  let maxX = params.n - r;
  let minY = r;
  let maxY = params.n - r;

  if (x < minX) {
    x = minX;
    if (vx < 0.0) { vx = -vx * params.damping; }
    if (params.hasOmega > 0.5) { w = w * 0.9; }
  } else if (x > maxX) {
    x = maxX;
    if (vx > 0.0) { vx = -vx * params.damping; }
    if (params.hasOmega > 0.5) { w = w * 0.9; }
  }

  if (y < minY) {
    y = minY;
    if (vy < 0.0) { vy = -vy * params.damping; }
    if (params.hasOmega > 0.5) { w = w * 0.9; }
  } else if (y > maxY) {
    y = maxY;
    if (vy > 0.0) { vy = -vy * params.damping; }
    if (params.hasOmega > 0.5) { w = w * 0.9; }
  }

  posX[i] = x;
  posY[i] = y;
  velX[i] = vx;
  velY[i] = vy;
  omega[i] = w;
}
`;

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined' || typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function createBuffer(device, size, usage) {
  return device.createBuffer({ size, usage });
}

function safeUnmapBuffer(buffer) {
  if (!buffer || typeof buffer.unmap !== 'function') return;
  try {
    buffer.unmap();
  } catch {
    // ignore invalid-state unmap attempts while recovering readback lifecycle
  }
}

function isFiniteBoundaryEntry(entry) {
  if (!entry) return false;
  if (!Number.isFinite(Number(entry.x)) || !Number.isFinite(Number(entry.y))) return false;
  if (!Number.isFinite(Number(entry.vx)) || !Number.isFinite(Number(entry.vy))) return false;
  if (!Number.isFinite(Number(entry.r))) return false;
  return true;
}

function splitFiniteBoundaryEntries(entries) {
  const finite = [];
  const nonFinite = [];
  for (const entry of entries) {
    if (isFiniteBoundaryEntry(entry)) finite.push(entry);
    else nonFinite.push(entry);
  }
  return { finite, nonFinite };
}

async function ensureWgslState(offload, count) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;

  if (!state.pipeline) {
    const module = device.createShaderModule({ code: bounceBoundaryWgsl });
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

    state.posX?.destroy?.();
    state.posY?.destroy?.();
    state.velX?.destroy?.();
    state.velY?.destroy?.();
    state.omega?.destroy?.();
    state.radius?.destroy?.();
    state.readPosX?.destroy?.();
    state.readPosY?.destroy?.();
    state.readVelX?.destroy?.();
    state.readVelY?.destroy?.();
    state.readOmega?.destroy?.();

    state.posX = createBuffer(device, bytes, storageUsage);
    state.posY = createBuffer(device, bytes, storageUsage);
    state.velX = createBuffer(device, bytes, storageUsage);
    state.velY = createBuffer(device, bytes, storageUsage);
    state.omega = createBuffer(device, bytes, storageUsage);
    state.radius = createBuffer(device, bytes, globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST);

    state.readPosX = createBuffer(device, bytes, readbackUsage);
    state.readPosY = createBuffer(device, bytes, readbackUsage);
    state.readVelX = createBuffer(device, bytes, readbackUsage);
    state.readVelY = createBuffer(device, bytes, readbackUsage);
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
        { binding: 1, resource: { buffer: state.posX } },
        { binding: 2, resource: { buffer: state.posY } },
        { binding: 3, resource: { buffer: state.velX } },
        { binding: 4, resource: { buffer: state.velY } },
        { binding: 5, resource: { buffer: state.omega } },
        { binding: 6, resource: { buffer: state.radius } },
      ],
    });
  }

  return state;
}

async function runBoundaryWgsl(entries, { n, damping, hasOmega, offload }) {
  const count = entries.length;
  if (count === 0) return;

  const state = await ensureWgslState(offload, count);
  const device = offload.device;
  const bytes = count * 4;

  const x = new Float32Array(count);
  const y = new Float32Array(count);
  const vx = new Float32Array(count);
  const vy = new Float32Array(count);
  const omega = new Float32Array(count);
  const radius = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const b = entries[i];
    x[i] = Number(b.x);
    y[i] = Number(b.y);
    vx[i] = Number(b.vx);
    vy[i] = Number(b.vy);
    omega[i] = Number(b.omega) || 0;
    radius[i] = Number(b.r);
  }

  const paramsBuffer = new ArrayBuffer(32);
  const paramsView = new DataView(paramsBuffer);
  paramsView.setUint32(0, count, true);
  paramsView.setFloat32(16, Number(n) || 0, true);
  paramsView.setFloat32(20, Number(damping) || 0, true);
  paramsView.setFloat32(24, hasOmega ? 1 : 0, true);

  device.queue.writeBuffer(state.params, 0, paramsBuffer);
  device.queue.writeBuffer(state.posX, 0, x);
  device.queue.writeBuffer(state.posY, 0, y);
  device.queue.writeBuffer(state.velX, 0, vx);
  device.queue.writeBuffer(state.velY, 0, vy);
  device.queue.writeBuffer(state.omega, 0, omega);
  device.queue.writeBuffer(state.radius, 0, radius);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / WGSL_WORKGROUP_SIZE));
  pass.end();

  encoder.copyBufferToBuffer(state.posX, 0, state.readPosX, 0, bytes);
  encoder.copyBufferToBuffer(state.posY, 0, state.readPosY, 0, bytes);
  encoder.copyBufferToBuffer(state.velX, 0, state.readVelX, 0, bytes);
  encoder.copyBufferToBuffer(state.velY, 0, state.readVelY, 0, bytes);
  encoder.copyBufferToBuffer(state.omega, 0, state.readOmega, 0, bytes);
  device.queue.submit([encoder.finish()]);

  safeUnmapBuffer(state.readPosX);
  safeUnmapBuffer(state.readPosY);
  safeUnmapBuffer(state.readVelX);
  safeUnmapBuffer(state.readVelY);
  safeUnmapBuffer(state.readOmega);

  try {
    const mapResults = await Promise.allSettled([
      state.readPosX.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
      state.readPosY.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
      state.readVelX.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
      state.readVelY.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
      state.readOmega.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    ]);
    const failedMap = mapResults.find((result) => result.status === 'rejected');
    if (failedMap) {
      throw failedMap.reason || new Error('boundary-readback-map-failed');
    }

    const outX = new Float32Array(state.readPosX.getMappedRange(0, bytes).slice(0));
    const outY = new Float32Array(state.readPosY.getMappedRange(0, bytes).slice(0));
    const outVx = new Float32Array(state.readVelX.getMappedRange(0, bytes).slice(0));
    const outVy = new Float32Array(state.readVelY.getMappedRange(0, bytes).slice(0));
    const outOmega = new Float32Array(state.readOmega.getMappedRange(0, bytes).slice(0));

    for (let i = 0; i < count; i++) {
      const b = entries[i];
      b.x = outX[i];
      b.y = outY[i];
      b.vx = outVx[i];
      b.vy = outVy[i];
      if (hasOmega && typeof b.omega === 'number') b.omega = outOmega[i];
    }
  } finally {
    safeUnmapBuffer(state.readPosX);
    safeUnmapBuffer(state.readPosY);
    safeUnmapBuffer(state.readVelX);
    safeUnmapBuffer(state.readVelY);
    safeUnmapBuffer(state.readOmega);
  }
}

function applyBoundaryCpu(entries, n, damping, applyBounceBoundary) {
  for (const b of entries) {
    if (!b) continue;
    applyBounceBoundary(b, n, damping);
  }
}

function scheduleSerializedBoundaryDispatch(wgslOffload, task) {
  const state = wgslOffload?.state;
  if (!state || typeof task !== 'function') return task();

  const chainedTask = (state.pendingCollisionBoundaryDispatchPromise || Promise.resolve())
    .catch(() => {})
    .then(task);

  state.pendingCollisionBoundaryDispatchPromise = chainedTask.finally(() => {
    if (state.pendingCollisionBoundaryDispatchPromise === chainedTask) {
      state.pendingCollisionBoundaryDispatchPromise = null;
    }
  });

  return state.pendingCollisionBoundaryDispatchPromise;
}

export async function applyCollisionBoundaryPassGpuOnly({
  rigidBodies,
  soft,
  n,
  rigidBounce = 0.84,
  softBounce = 0.78,
  applyBounceBoundary,
  wgslOffload,
}) {
  const rigidList = Array.isArray(rigidBodies) ? rigidBodies : [];
  const softNodes = Array.isArray(soft?.nodes) ? soft.nodes : [];

  if (typeof applyBounceBoundary !== 'function') {
    throw new Error('gpu-only collision boundary pass requires applyBounceBoundary callback');
  }

  const { finite: finiteRigid, nonFinite: nonFiniteRigid } = splitFiniteBoundaryEntries(rigidList);
  const { finite: finiteSoft, nonFinite: nonFiniteSoft } = splitFiniteBoundaryEntries(softNodes);
  const finiteEntryCount = finiteRigid.length + finiteSoft.length;
  const nonFiniteEntryCount = nonFiniteRigid.length + nonFiniteSoft.length;

  const wgslAvailable = canUseWgslOffload(wgslOffload) && finiteEntryCount > 0;

  if (!wgslAvailable) {
    applyBoundaryCpu(rigidList, n, rigidBounce, applyBounceBoundary);
    applyBoundaryCpu(softNodes, n, softBounce, applyBounceBoundary);
    return {
      mode: 'cpu',
      reason: finiteEntryCount === 0 ? 'non-finite-boundary-abi' : 'wgsl-unavailable',
    };
  }

  return scheduleSerializedBoundaryDispatch(wgslOffload, async () => {
    try {
      await runBoundaryWgsl(finiteRigid, {
        n,
        damping: rigidBounce,
        hasOmega: true,
        offload: wgslOffload,
      });
      await runBoundaryWgsl(finiteSoft, {
        n,
        damping: softBounce,
        hasOmega: false,
        offload: wgslOffload,
      });

      if (nonFiniteEntryCount > 0) {
        applyBoundaryCpu(nonFiniteRigid, n, rigidBounce, applyBounceBoundary);
        applyBoundaryCpu(nonFiniteSoft, n, softBounce, applyBounceBoundary);
      }

      if (wgslOffload?.state) {
        wgslOffload.state.lastError = null;
        wgslOffload.state.lastMode = nonFiniteEntryCount > 0 ? 'wgsl-partial' : 'wgsl';
        wgslOffload.state.lastFiniteEntryCount = finiteEntryCount;
        wgslOffload.state.lastNonFiniteEntryCount = nonFiniteEntryCount;
      }
      return {
        mode: nonFiniteEntryCount > 0 ? 'wgsl-partial' : 'wgsl',
        reason: nonFiniteEntryCount > 0 ? 'ok-with-cpu-nonfinite' : 'ok',
      };
    } catch (err) {
      applyBoundaryCpu(rigidList, n, rigidBounce, applyBounceBoundary);
      applyBoundaryCpu(softNodes, n, softBounce, applyBounceBoundary);
      if (wgslOffload?.state) {
        wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
        wgslOffload.state.lastMode = 'cpu-fallback';
        wgslOffload.state.lastFiniteEntryCount = finiteEntryCount;
        wgslOffload.state.lastNonFiniteEntryCount = nonFiniteEntryCount;
      }
      return { mode: 'cpu-fallback', reason: String(err?.message || err || 'unknown-error') };
    }
  });
}
