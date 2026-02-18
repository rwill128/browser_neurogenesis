/**
 * GPU-only runtime rigid-rigid collision pass.
 * Keeps collision stepping isolated for the gpu-only runtime path while
 * preserving baseline/default behavior and call contracts.
 */

const WGSL_WORKGROUP_SIZE = 64;

const RIGID_COLLISION_WGSL = /* wgsl */`
struct Params {
  pair_count: u32,
  slop: f32,
  _pad0: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pair_a_index: array<u32>;
@group(0) @binding(2) var<storage, read> pair_b_index: array<u32>;
@group(0) @binding(3) var<storage, read> body_x: array<f32>;
@group(0) @binding(4) var<storage, read> body_y: array<f32>;
@group(0) @binding(5) var<storage, read> body_vx: array<f32>;
@group(0) @binding(6) var<storage, read> body_vy: array<f32>;
@group(0) @binding(7) var<storage, read> body_r: array<f32>;
@group(0) @binding(8) var<storage, read> body_mass: array<f32>;
@group(0) @binding(9) var<storage, read_write> out_delta_vx_a: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_delta_vy_a: array<f32>;
@group(0) @binding(11) var<storage, read_write> out_delta_vx_b: array<f32>;
@group(0) @binding(12) var<storage, read_write> out_delta_vy_b: array<f32>;
@group(0) @binding(13) var<storage, read_write> out_contact_mask: array<u32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= params.pair_count) {
    return;
  }

  let ai = pair_a_index[pi];
  let bi = pair_b_index[pi];

  let ax = body_x[ai];
  let ay = body_y[ai];
  let bx = body_x[bi];
  let by = body_y[bi];
  let avx = body_vx[ai];
  let avy = body_vy[ai];
  let bvx = body_vx[bi];
  let bvy = body_vy[bi];
  let ar = max(0.05, body_r[ai]);
  let br = max(0.05, body_r[bi]);

  let dx = bx - ax;
  let dy = by - ay;
  let dist2 = dx * dx + dy * dy;
  let minDist = max(0.0, ar + br - params.slop);
  let minDist2 = minDist * minDist;

  var nx = 1.0;
  var ny = 0.0;
  var dist = sqrt(max(dist2, 1e-12));
  if (dist > 1e-6) {
    nx = dx / dist;
    ny = dy / dist;
  }

  if (dist2 > minDist2) {
    out_delta_vx_a[pi] = 0.0;
    out_delta_vy_a[pi] = 0.0;
    out_delta_vx_b[pi] = 0.0;
    out_delta_vy_b[pi] = 0.0;
    out_contact_mask[pi] = 0u;
    return;
  }

  let invMassA = 1.0 / max(1e-6, body_mass[ai]);
  let invMassB = 1.0 / max(1e-6, body_mass[bi]);
  let rvx = bvx - avx;
  let rvy = bvy - avy;
  let relN = rvx * nx + rvy * ny;
  let e = 0.12;
  let j = -(1.0 + e) * relN / max(1e-6, invMassA + invMassB);

  out_delta_vx_a[pi] = -j * nx * invMassA;
  out_delta_vy_a[pi] = -j * ny * invMassA;
  out_delta_vx_b[pi] = j * nx * invMassB;
  out_delta_vy_b[pi] = j * ny * invMassB;
  out_contact_mask[pi] = 1u;
}
`;

function canUseWgslOffload(offload) {
  return Boolean(
    offload
      && offload.enabled === true
      && offload.authoritativeRigidCollision === true
      && offload.device
      && offload.state
      && typeof offload.device.createComputePipelineAsync === 'function'
      && typeof offload.device.createBuffer === 'function'
      && typeof offload.device.createCommandEncoder === 'function'
      && globalThis.GPUBufferUsage
      && globalThis.GPUMapMode,
  );
}

function pairCountForBodyCount(count) {
  const n = Math.max(0, Number(count) || 0);
  return (n * (n - 1)) / 2;
}

function buildPairLayout(bodyCount) {
  const pairCount = pairCountForBodyCount(bodyCount);
  const pairAIndex = new Uint32Array(pairCount);
  const pairBIndex = new Uint32Array(pairCount);
  let p = 0;
  for (let i = 0; i < bodyCount; i++) {
    for (let j = i + 1; j < bodyCount; j++) {
      pairAIndex[p] = i;
      pairBIndex[p] = j;
      p += 1;
    }
  }
  return { pairCount, pairAIndex, pairBIndex };
}

async function ensureRigidCollisionWgslState(offload, bodyCount) {
  const device = offload.device;
  const state = offload.state;

  if (!state.rigidCollisionPipeline) {
    const module = device.createShaderModule({ code: RIGID_COLLISION_WGSL });
    state.rigidCollisionPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.rigidCollisionBindGroup = null;
  }

  if (!state.rigidCollisionParams) {
    state.rigidCollisionParams = device.createBuffer({
      size: 16,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.rigidCollisionBindGroup = null;
  }

  const pairCapacity = Math.max(1, pairCountForBodyCount(bodyCount));
  const bodyCapacity = Math.max(1, bodyCount);

  if ((state.rigidCollisionPairCapacity || 0) < pairCapacity) {
    const pairBytes = pairCapacity * 4;
    state.rigidCollisionPairA?.destroy?.();
    state.rigidCollisionPairB?.destroy?.();
    state.rigidCollisionDeltaVxA?.destroy?.();
    state.rigidCollisionDeltaVyA?.destroy?.();
    state.rigidCollisionDeltaVxB?.destroy?.();
    state.rigidCollisionDeltaVyB?.destroy?.();
    state.rigidCollisionContactMask?.destroy?.();
    state.rigidCollisionDeltaVxAReadback?.destroy?.();
    state.rigidCollisionDeltaVyAReadback?.destroy?.();
    state.rigidCollisionDeltaVxBReadback?.destroy?.();
    state.rigidCollisionDeltaVyBReadback?.destroy?.();
    state.rigidCollisionContactMaskReadback?.destroy?.();

    state.rigidCollisionPairA = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST });
    state.rigidCollisionPairB = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST });
    state.rigidCollisionDeltaVxA = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.rigidCollisionDeltaVyA = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.rigidCollisionDeltaVxB = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.rigidCollisionDeltaVyB = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.rigidCollisionContactMask = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });

    state.rigidCollisionDeltaVxAReadback = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.rigidCollisionDeltaVyAReadback = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.rigidCollisionDeltaVxBReadback = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.rigidCollisionDeltaVyBReadback = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.rigidCollisionContactMaskReadback = device.createBuffer({ size: pairBytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });

    state.rigidCollisionPairCapacity = pairCapacity;
    state.rigidCollisionBindGroup = null;
  }

  if ((state.rigidCollisionBodyCapacity || 0) < bodyCapacity) {
    const bodyBytes = bodyCapacity * 4;
    state.rigidCollisionBodyX?.destroy?.();
    state.rigidCollisionBodyY?.destroy?.();
    state.rigidCollisionBodyVx?.destroy?.();
    state.rigidCollisionBodyVy?.destroy?.();
    state.rigidCollisionBodyR?.destroy?.();
    state.rigidCollisionBodyMass?.destroy?.();

    const usage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
    state.rigidCollisionBodyX = device.createBuffer({ size: bodyBytes, usage });
    state.rigidCollisionBodyY = device.createBuffer({ size: bodyBytes, usage });
    state.rigidCollisionBodyVx = device.createBuffer({ size: bodyBytes, usage });
    state.rigidCollisionBodyVy = device.createBuffer({ size: bodyBytes, usage });
    state.rigidCollisionBodyR = device.createBuffer({ size: bodyBytes, usage });
    state.rigidCollisionBodyMass = device.createBuffer({ size: bodyBytes, usage });

    state.rigidCollisionBodyCapacity = bodyCapacity;
    state.rigidCollisionBindGroup = null;
  }

  if (!state.rigidCollisionBindGroup) {
    state.rigidCollisionBindGroup = device.createBindGroup({
      layout: state.rigidCollisionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.rigidCollisionParams } },
        { binding: 1, resource: { buffer: state.rigidCollisionPairA } },
        { binding: 2, resource: { buffer: state.rigidCollisionPairB } },
        { binding: 3, resource: { buffer: state.rigidCollisionBodyX } },
        { binding: 4, resource: { buffer: state.rigidCollisionBodyY } },
        { binding: 5, resource: { buffer: state.rigidCollisionBodyVx } },
        { binding: 6, resource: { buffer: state.rigidCollisionBodyVy } },
        { binding: 7, resource: { buffer: state.rigidCollisionBodyR } },
        { binding: 8, resource: { buffer: state.rigidCollisionBodyMass } },
        { binding: 9, resource: { buffer: state.rigidCollisionDeltaVxA } },
        { binding: 10, resource: { buffer: state.rigidCollisionDeltaVyA } },
        { binding: 11, resource: { buffer: state.rigidCollisionDeltaVxB } },
        { binding: 12, resource: { buffer: state.rigidCollisionDeltaVyB } },
        { binding: 13, resource: { buffer: state.rigidCollisionContactMask } },
      ],
    });
  }

  return state;
}

async function readF32(readback, bytes) {
  await readback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = readback.getMappedRange(0, bytes);
  const values = new Float32Array(mapped.slice(0));
  readback.unmap();
  return values;
}

async function readU32(readback, bytes) {
  await readback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = readback.getMappedRange(0, bytes);
  const values = new Uint32Array(mapped.slice(0));
  readback.unmap();
  return values;
}

async function dispatchRigidCollisionWgsl(offload, rigidBodies, slop) {
  const state = await ensureRigidCollisionWgslState(offload, rigidBodies.length);
  const { pairCount, pairAIndex, pairBIndex } = buildPairLayout(rigidBodies.length);
  if (pairCount <= 0) return null;

  const bodyX = new Float32Array(rigidBodies.length);
  const bodyY = new Float32Array(rigidBodies.length);
  const bodyVx = new Float32Array(rigidBodies.length);
  const bodyVy = new Float32Array(rigidBodies.length);
  const bodyR = new Float32Array(rigidBodies.length);
  const bodyMass = new Float32Array(rigidBodies.length);
  for (let i = 0; i < rigidBodies.length; i++) {
    const rb = rigidBodies[i] || {};
    bodyX[i] = Number(rb.x) || 0;
    bodyY[i] = Number(rb.y) || 0;
    bodyVx[i] = Number(rb.vx) || 0;
    bodyVy[i] = Number(rb.vy) || 0;
    bodyR[i] = Math.max(0.05, Number(rb.r) || 1);
    bodyMass[i] = Math.max(1e-6, Number(rb.mass) || 1);
  }

  const params = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(params);
  const paramsF32 = new Float32Array(params);
  paramsU32[0] = pairCount >>> 0;
  paramsF32[1] = Number(slop) || 0;

  const pairBytes = pairCount * 4;
  offload.device.queue.writeBuffer(state.rigidCollisionParams, 0, params);
  offload.device.queue.writeBuffer(state.rigidCollisionPairA, 0, pairAIndex);
  offload.device.queue.writeBuffer(state.rigidCollisionPairB, 0, pairBIndex);
  offload.device.queue.writeBuffer(state.rigidCollisionBodyX, 0, bodyX);
  offload.device.queue.writeBuffer(state.rigidCollisionBodyY, 0, bodyY);
  offload.device.queue.writeBuffer(state.rigidCollisionBodyVx, 0, bodyVx);
  offload.device.queue.writeBuffer(state.rigidCollisionBodyVy, 0, bodyVy);
  offload.device.queue.writeBuffer(state.rigidCollisionBodyR, 0, bodyR);
  offload.device.queue.writeBuffer(state.rigidCollisionBodyMass, 0, bodyMass);

  const encoder = offload.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.rigidCollisionPipeline);
  pass.setBindGroup(0, state.rigidCollisionBindGroup);
  const dispatchCount = Math.max(1, Math.ceil(pairCount / WGSL_WORKGROUP_SIZE));
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();

  encoder.copyBufferToBuffer(state.rigidCollisionDeltaVxA, 0, state.rigidCollisionDeltaVxAReadback, 0, pairBytes);
  encoder.copyBufferToBuffer(state.rigidCollisionDeltaVyA, 0, state.rigidCollisionDeltaVyAReadback, 0, pairBytes);
  encoder.copyBufferToBuffer(state.rigidCollisionDeltaVxB, 0, state.rigidCollisionDeltaVxBReadback, 0, pairBytes);
  encoder.copyBufferToBuffer(state.rigidCollisionDeltaVyB, 0, state.rigidCollisionDeltaVyBReadback, 0, pairBytes);
  encoder.copyBufferToBuffer(state.rigidCollisionContactMask, 0, state.rigidCollisionContactMaskReadback, 0, pairBytes);
  offload.device.queue.submit([encoder.finish()]);

  const deltaVxA = await readF32(state.rigidCollisionDeltaVxAReadback, pairBytes);
  const deltaVyA = await readF32(state.rigidCollisionDeltaVyAReadback, pairBytes);
  const deltaVxB = await readF32(state.rigidCollisionDeltaVxBReadback, pairBytes);
  const deltaVyB = await readF32(state.rigidCollisionDeltaVyBReadback, pairBytes);
  const contactMask = await readU32(state.rigidCollisionContactMaskReadback, pairBytes);

  state.lastRigidCollisionProposalSource = 'wgsl-rigid-collision-authoritative';
  state.lastRigidCollisionDispatchCount = dispatchCount;
  state.lastRigidCollisionPairCount = pairCount;

  return { pairAIndex, pairBIndex, deltaVxA, deltaVyA, deltaVxB, deltaVyB, contactMask, pairCount };
}

function canApplyAuthoritativeProposal(proposal) {
  if (!proposal || proposal.pairCount <= 0) return false;
  for (let i = 0; i < proposal.pairCount; i++) {
    if (!Number.isFinite(proposal.deltaVxA[i])
      || !Number.isFinite(proposal.deltaVyA[i])
      || !Number.isFinite(proposal.deltaVxB[i])
      || !Number.isFinite(proposal.deltaVyB[i])) {
      return false;
    }
  }
  return true;
}

function applyAuthoritativeProposal({ rigidBodies, proposal, contacts, iter, phase }) {
  for (let i = 0; i < proposal.pairCount; i++) {
    if ((proposal.contactMask[i] >>> 0) === 0) continue;
    const ai = proposal.pairAIndex[i] >>> 0;
    const bi = proposal.pairBIndex[i] >>> 0;
    const a = rigidBodies[ai];
    const b = rigidBodies[bi];
    if (!a || !b) continue;

    a.vx += proposal.deltaVxA[i] || 0;
    a.vy += proposal.deltaVyA[i] || 0;
    b.vx += proposal.deltaVxB[i] || 0;
    b.vy += proposal.deltaVyB[i] || 0;

    if (Array.isArray(contacts)) {
      contacts.push({
        phase,
        iter,
        aIndex: ai,
        bIndex: bi,
        source: 'wgsl-rigid-collision-authoritative',
      });
    }
  }
}

export async function resolveRigidRigidCollisionPassGpuOnly({
  rigidBodies,
  slop,
  contacts,
  iter,
  phase,
  resolveRigidVsRigidPolygonCollision,
  wgslOffload,
}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length < 2) return;
  if (typeof resolveRigidVsRigidPolygonCollision !== 'function') {
    throw new Error('gpu-only rigid collision pass requires resolveRigidVsRigidPolygonCollision callback');
  }

  if (canUseWgslOffload(wgslOffload)) {
    try {
      const proposal = await dispatchRigidCollisionWgsl(wgslOffload, rigidBodies, slop);
      if (proposal && canApplyAuthoritativeProposal(proposal)) {
        applyAuthoritativeProposal({ rigidBodies, proposal, contacts, iter, phase });
        wgslOffload.state.lastRigidCollisionAuthoritativeSource = 'wgsl-rigid-collision-authoritative';
        wgslOffload.state.lastSourceRoute = 'wgsl-rigid-collision-authoritative';
        wgslOffload.state.lastMode = 'wgsl-rigid-collision-authoritative';
        return;
      }
      wgslOffload.state.lastRigidCollisionAuthoritativeSource = 'cpu-rigid-collision-authoritative-nonfinite';
    } catch (err) {
      wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
      wgslOffload.state.lastRigidCollisionAuthoritativeSource = 'cpu-rigid-collision-authoritative-fallback';
    }
    wgslOffload.state.lastSourceRoute = 'cpu-rigid-collision-authoritative';
    wgslOffload.state.lastMode = 'cpu-rigid-collision-authoritative';
  }

  for (let i = 0; i < rigidBodies.length; i++) {
    for (let j = i + 1; j < rigidBodies.length; j++) {
      resolveRigidVsRigidPolygonCollision(rigidBodies[i], rigidBodies[j], slop, {
        contacts,
        aIndex: i,
        bIndex: j,
        iter,
        phase,
      });
    }
  }
}
