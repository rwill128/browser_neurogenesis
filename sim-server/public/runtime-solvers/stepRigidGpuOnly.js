function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

const WGSL_WORKGROUP_SIZE = 64;
const RIGID_LAYOUT_STRIDE_FLOATS = 14;

const rigidStepProposalWgsl = /* wgsl */ `
struct Params {
  count : f32,
  dt : f32,
  dtNorm : f32,
  boundaryN : f32,
};

@group(0) @binding(0) var<storage, read> layout : array<f32>;
@group(0) @binding(1) var<storage, read_write> vxOut : array<f32>;
@group(0) @binding(2) var<storage, read_write> vyOut : array<f32>;
@group(0) @binding(3) var<storage, read_write> omegaOut : array<f32>;
@group(0) @binding(4) var<storage, read_write> xOut : array<f32>;
@group(0) @binding(5) var<storage, read_write> yOut : array<f32>;
@group(0) @binding(6) var<storage, read_write> thetaOut : array<f32>;
@group(0) @binding(7) var<storage, read_write> carryOut : array<f32>;
@group(0) @binding(8) var<uniform> params : Params;

fn finiteOrZero(v : f32) -> f32 {
  if (v == v && abs(v) < 1e20) {
    return v;
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (f32(i) >= params.count) {
    return;
  }

  let base = i * ${RIGID_LAYOUT_STRIDE_FLOATS}u;
  var vx = finiteOrZero(layout[base + 0u]);
  var vy = finiteOrZero(layout[base + 1u]);
  var omega = finiteOrZero(layout[base + 2u]);
  var x = finiteOrZero(layout[base + 3u]);
  var y = finiteOrZero(layout[base + 4u]);
  var theta = finiteOrZero(layout[base + 5u]);
  let ax = finiteOrZero(layout[base + 6u]);
  let ay = finiteOrZero(layout[base + 7u]);
  let alpha = finiteOrZero(layout[base + 8u]);
  let swimX = finiteOrZero(layout[base + 9u]);
  let swimY = finiteOrZero(layout[base + 10u]);
  let swimTorque = finiteOrZero(layout[base + 11u]);
  let damp = max(0.0, finiteOrZero(layout[base + 12u]));
  let vmax = max(0.0, finiteOrZero(layout[base + 13u]));

  vx = vx + ax * params.dt * 60.0 + swimX * params.dtNorm;
  vy = vy + ay * params.dt * 60.0 + swimY * params.dtNorm;
  omega = omega + alpha * params.dt * 60.0 + swimTorque * params.dtNorm;

  let centerHoney = clamp((1.0 - damp) / 0.018, 0.0, 100.0);
  vx = vx * damp;
  vy = vy * damp;
  omega = omega * max(0.72, 0.99 - 0.01 * centerHoney);

  let vMag = sqrt(vx * vx + vy * vy);
  if (vMag > vmax && vmax > 1e-8) {
    let invMag = 1.0 / max(vMag, 1e-9);
    vx = vx * invMag * vmax;
    vy = vy * invMag * vmax;
  }
  omega = clamp(omega, -0.25, 0.25);

  x = x + vx * params.dt * 22.0;
  y = y + vy * params.dt * 28.0;
  theta = theta + omega * params.dt * 60.0;

  let e = 0.84;
  let edge = max(0.0, params.boundaryN - 1.0);
  if (x < 0.0) { x = 0.0; vx = abs(vx) * e; }
  if (y < 0.0) { y = 0.0; vy = abs(vy) * e; }
  if (x > edge) { x = edge; vx = -abs(vx) * e; }
  if (y > edge) { y = edge; vy = -abs(vy) * e; }

  vxOut[i] = vx;
  vyOut[i] = vy;
  omegaOut[i] = omega;
  xOut[i] = x;
  yOut[i] = y;
  thetaOut[i] = theta;
  carryOut[i] = sqrt(ax * ax + ay * ay);
}
`;

function checkFiniteFloat32Array(values) {
  if (!(values instanceof Float32Array)) return false;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) return false;
  }
  return true;
}

function ensureRigidStepProposalPipeline(offload) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device) return null;
  if (!state.rigidStepProposalPipelinePromise) {
    state.rigidStepProposalPipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: rigidStepProposalWgsl }),
        entryPoint: 'main',
      },
    }).catch((err) => {
      state.rigidStepProposalPipelinePromise = null;
      throw err;
    });
  }
  return state.rigidStepProposalPipelinePromise;
}

function getGpuOnlyPipelineModeProfile(offload) {
  const modeProfile = String(offload?.modeProfile || '').trim().toLowerCase();
  if (modeProfile === 'gpu-only-fast') return 'gpu-only-fast';
  if (modeProfile === 'gpu-only-validated') return 'gpu-only-validated';
  return 'standard';
}

async function dispatchRigidStepProposal(offload, layout, dt, dtNorm, boundaryN, signature) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device || !(layout instanceof Float32Array) || layout.length === 0) return false;
  const count = Math.floor(layout.length / RIGID_LAYOUT_STRIDE_FLOATS);
  if (count <= 0) return false;
  const pipeline = await ensureRigidStepProposalPipeline(offload);
  if (!pipeline) return false;

  const layoutBuffer = device.createBuffer({ size: layout.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBytes = count * Float32Array.BYTES_PER_ELEMENT;
  const makeOut = () => device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const vxBuffer = makeOut();
  const vyBuffer = makeOut();
  const omegaBuffer = makeOut();
  const xBuffer = makeOut();
  const yBuffer = makeOut();
  const thetaBuffer = makeOut();
  const carryBuffer = makeOut();
  const makeRead = () => device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readVx = makeRead();
  const readVy = makeRead();
  const readOmega = makeRead();
  const readX = makeRead();
  const readY = makeRead();
  const readTheta = makeRead();
  const readCarry = makeRead();
  const paramBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const params = new Float32Array([count, dt, dtNorm, boundaryN]);

  device.queue.writeBuffer(layoutBuffer, 0, layout);
  device.queue.writeBuffer(paramBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: layoutBuffer } },
      { binding: 1, resource: { buffer: vxBuffer } },
      { binding: 2, resource: { buffer: vyBuffer } },
      { binding: 3, resource: { buffer: omegaBuffer } },
      { binding: 4, resource: { buffer: xBuffer } },
      { binding: 5, resource: { buffer: yBuffer } },
      { binding: 6, resource: { buffer: thetaBuffer } },
      { binding: 7, resource: { buffer: carryBuffer } },
      { binding: 8, resource: { buffer: paramBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / WGSL_WORKGROUP_SIZE)));
  pass.end();

  encoder.copyBufferToBuffer(vxBuffer, 0, readVx, 0, outBytes);
  encoder.copyBufferToBuffer(vyBuffer, 0, readVy, 0, outBytes);
  encoder.copyBufferToBuffer(omegaBuffer, 0, readOmega, 0, outBytes);
  encoder.copyBufferToBuffer(xBuffer, 0, readX, 0, outBytes);
  encoder.copyBufferToBuffer(yBuffer, 0, readY, 0, outBytes);
  encoder.copyBufferToBuffer(thetaBuffer, 0, readTheta, 0, outBytes);
  encoder.copyBufferToBuffer(carryBuffer, 0, readCarry, 0, outBytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([readVx, readVy, readOmega, readX, readY, readTheta, readCarry].map((b) => b.mapAsync(GPUMapMode.READ)));
  const vx = new Float32Array(readVx.getMappedRange().slice(0));
  const vy = new Float32Array(readVy.getMappedRange().slice(0));
  const omega = new Float32Array(readOmega.getMappedRange().slice(0));
  const x = new Float32Array(readX.getMappedRange().slice(0));
  const y = new Float32Array(readY.getMappedRange().slice(0));
  const theta = new Float32Array(readTheta.getMappedRange().slice(0));
  const carry = new Float32Array(readCarry.getMappedRange().slice(0));
  [readVx, readVy, readOmega, readX, readY, readTheta, readCarry].forEach((b) => b.unmap());

  const allFinite = checkFiniteFloat32Array(vx)
    && checkFiniteFloat32Array(vy)
    && checkFiniteFloat32Array(omega)
    && checkFiniteFloat32Array(x)
    && checkFiniteFloat32Array(y)
    && checkFiniteFloat32Array(theta)
    && checkFiniteFloat32Array(carry);

  state.lastRigidStepProposalSignature = String(signature || '');
  state.lastRigidStepProposalSource = allFinite ? 'wgsl-rigid-step-proposal' : 'cpu-rigid-step-authoritative-nonfinite';
  state.lastRigidStepProposalFinite = { allFinite };
  state.lastMode = allFinite ? 'wgsl-rigid-step-proposal' : 'cpu-rigid-step-authoritative';

  if (allFinite) {
    state.lastRigidStepProposalVx = vx;
    state.lastRigidStepProposalVy = vy;
    state.lastRigidStepProposalOmega = omega;
    state.lastRigidStepProposalX = x;
    state.lastRigidStepProposalY = y;
    state.lastRigidStepProposalTheta = theta;
    state.lastRigidStepProposalCarry = carry;
  }

  [layoutBuffer, vxBuffer, vyBuffer, omegaBuffer, xBuffer, yBuffer, thetaBuffer, carryBuffer, readVx, readVy, readOmega, readX, readY, readTheta, readCarry, paramBuffer].forEach((b) => b.destroy());
  return allFinite;
}

function rigidEdgeMomentumScale(rb) {
  const arr = Array.isArray(rb?.edgeMomentumCoupling)
    ? rb.edgeMomentumCoupling
    : (Array.isArray(rb?.edgeMomentumTransfer) ? rb.edgeMomentumTransfer : null);
  if (!arr || arr.length === 0) return 1;
  let sum = 0;
  let c = 0;
  for (const v of arr) {
    const n = Number(v);
    if (Number.isFinite(n)) {
      sum += clamp(n, 0, 1);
      c += 1;
    }
  }
  return c > 0 ? (sum / c) : 1;
}

/**
 * GPU-only runtime rigid stepping path.
 * Isolated from baseline in gpu-lab.js so integration/coupling responsibility can
 * migrate incrementally while baseline remains the reference/default path.
 */
export function stepRigidBodiesGpuOnly({
  sim,
  bodies,
  vxField,
  vyField,
  n,
  dt,
  dtNorm,
  dragK,
  swimGain,
  localHoneyDrag,
  viscosityMotionResponse,
  obstacleMask,
  bodyFeedbackPrevVx,
  bodyFeedbackPrevVy,
  selfFeedbackSuppression,
  rigidVerticesWorld,
  sampleFluidForBodyCoupling,
  applyBounceBoundary,
  wgslOffload,
}) {
  let rigidCarryTransfer = 0;
  const runWgslProbe = wgslOffload?.enabled === true
    && wgslOffload?.state
    && wgslOffload?.device
    && getGpuOnlyPipelineModeProfile(wgslOffload) !== 'standard';
  const rigidProposalLayout = [];
  let rigidProposalSignatureAccumulator = 0;

  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const edgeMomentumScale = rigidEdgeMomentumScale(b);
    const invMass = 1 / Math.max(0.05, b.mass);
    const invInertia = 1 / Math.max(0.05, b.inertia || 1);
    const sampleVerts = rigidVerticesWorld(b);
    const sampleCount = Math.max(1, sampleVerts.length);
    let forceX = 0;
    let forceY = 0;
    let torque = 0;

    for (let si = 0; si < sampleCount; si++) {
      const sx = sampleVerts[si].x;
      const sy = sampleVerts[si].y;
      const rx = sx - b.x;
      const ry = sy - b.y;
      const fx = sampleFluidForBodyCoupling(
        vxField,
        n,
        sx,
        sy,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVx,
        selfFeedbackSuppression,
      );
      const fy = sampleFluidForBodyCoupling(
        vyField,
        n,
        sx,
        sy,
        rx,
        ry,
        obstacleMask,
        bodyFeedbackPrevVy,
        selfFeedbackSuppression,
      );
      const localVx = b.vx + (-(b.omega || 0) * ry);
      const localVy = b.vy + ((b.omega || 0) * rx);
      const relX = fx - localVx;
      const relY = fy - localVy;
      const honey = localHoneyDrag(sx, sy);
      const fpx = relX * dragK * honey * edgeMomentumScale;
      const fpy = relY * dragK * honey * edgeMomentumScale;
      forceX += fpx;
      forceY += fpy;
      torque += rx * fpy - ry * fpx;
    }

    forceX /= sampleCount;
    forceY /= sampleCount;
    torque /= sampleCount;

    const ax = forceX * invMass;
    const ay = forceY * invMass;
    const alpha = torque * invInertia;

    const swimPhase = sim.frame * 0.08 + bi * 2.1;
    const swimX = swimGain * Math.cos(swimPhase) * 0.012 * invMass;
    const swimY = swimGain * Math.sin(swimPhase * 1.6) * 0.009 * invMass;
    const swimTorque = swimGain * Math.sin(swimPhase * 1.1) * 0.0025;

    b.vx += ax * dt * 60 + swimX * dtNorm;
    b.vy += ay * dt * 60 + swimY * dtNorm;
    b.omega = (b.omega || 0) + alpha * dt * 60 + swimTorque * dtNorm;

    const centerHoney = localHoneyDrag(b.x, b.y);
    const rigidVisc = viscosityMotionResponse(centerHoney, 3.2);

    if (runWgslProbe) {
      rigidProposalLayout.push(
        b.vx,
        b.vy,
        b.omega || 0,
        b.x,
        b.y,
        b.theta || 0,
        ax,
        ay,
        alpha,
        swimX,
        swimY,
        swimTorque,
        rigidVisc.damp,
        rigidVisc.vmax,
      );
      rigidProposalSignatureAccumulator += Math.fround(ax) * 0.31 + Math.fround(ay) * 0.19 + Math.fround(alpha) * 0.11;
    }

    b.vx *= rigidVisc.damp;
    b.vy *= rigidVisc.damp;
    b.omega *= Math.max(0.72, 0.99 - 0.01 * centerHoney);

    const bMax = rigidVisc.vmax;
    const bMag = Math.hypot(b.vx, b.vy);
    if (bMag > bMax) {
      b.vx = (b.vx / bMag) * bMax;
      b.vy = (b.vy / bMag) * bMax;
    }
    b.omega = Math.max(-0.25, Math.min(0.25, b.omega));

    rigidCarryTransfer += Math.hypot(ax, ay);
    b.x = b.x + b.vx * dt * 22;
    b.y = b.y + b.vy * dt * 28;
    b.theta = (b.theta || 0) + b.omega * dt * 60;
    applyBounceBoundary(b, n, 0.84);
  }

  if (runWgslProbe && rigidProposalLayout.length > 0) {
    const layout = Float32Array.from(rigidProposalLayout);
    const signature = `${bodies.rigid.length}|${Math.fround(dt)}|${Math.fround(dtNorm)}|${Math.fround(rigidProposalSignatureAccumulator)}`;
    wgslOffload.state.lastRigidStepProposalLayoutBytes = layout.byteLength;
    wgslOffload.state.lastRigidStepProposalSignaturePrepared = signature;
    if (!wgslOffload.state.lastRigidStepProposalSource) {
      wgslOffload.state.lastRigidStepProposalSource = 'cpu-rigid-step-authoritative';
    }
    if (!wgslOffload.state.lastMode) {
      wgslOffload.state.lastMode = 'cpu-rigid-step-authoritative';
    }
    const serializedDispatch = (wgslOffload.state.pendingWgslRigidStepProposalPromise || Promise.resolve())
      .catch(() => {})
      .then(() => dispatchRigidStepProposal(wgslOffload, layout, dt, dtNorm, n, signature))
      .catch((err) => {
        wgslOffload.state.lastRigidStepProposalError = String(err?.message || err || 'unknown-error');
        wgslOffload.state.lastRigidStepProposalSource = 'cpu-rigid-step-authoritative';
        wgslOffload.state.lastMode = 'cpu-rigid-step-authoritative';
      });
    wgslOffload.state.pendingWgslRigidStepProposalPromise = serializedDispatch;
  }

  return rigidCarryTransfer;
}
