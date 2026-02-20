const WASM_URL = '/wasm/rigid-step-baseline.wasm?v=20260219a';
const DEFAULT_HEAP_BASE = 65536;

let runtimePromise = null;

function nowMs() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

async function instantiateWasm() {
  const imports = {};
  if (typeof WebAssembly.instantiateStreaming === 'function') {
    try {
      const streamed = await WebAssembly.instantiateStreaming(fetch(WASM_URL), imports);
      return streamed.instance || streamed;
    } catch {
      // fall back for non-wasm content type
    }
  }
  const res = await fetch(WASM_URL);
  if (!res.ok) throw new Error(`failed to fetch wasm (${res.status})`);
  const bytes = await res.arrayBuffer();
  const loaded = await WebAssembly.instantiate(bytes, imports);
  return loaded.instance || loaded;
}

async function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      const instance = await instantiateWasm();
      const exports = instance?.exports || {};
      const memory = exports.memory;
      if (!(memory instanceof WebAssembly.Memory)) {
        throw new Error('rsb wasm: missing exported memory');
      }
      if (
        typeof exports.rsb_alloc !== 'function'
        || typeof exports.rsb_reset_heap !== 'function'
        || typeof exports.rsb_step_bodies !== 'function'
      ) {
        throw new Error('rsb wasm: missing required exports');
      }
      const heapBase = typeof exports.rsb_heap_base === 'function'
        ? (Number(exports.rsb_heap_base()) || DEFAULT_HEAP_BASE)
        : DEFAULT_HEAP_BASE;
      return { instance, exports, memory, heapBase };
    })();
  }
  return runtimePromise;
}

function ensureMemoryCapacity(runtime, requiredBytes) {
  const pageSize = 64 * 1024;
  const current = runtime.memory.buffer.byteLength;
  if (current >= requiredBytes) return;
  const missing = requiredBytes - current;
  runtime.memory.grow(Math.max(1, Math.ceil(missing / pageSize)));
}

function resolveRigidEdgeVelocityMode(rb, edgeIndex) {
  const velocityModeRaw = Array.isArray(rb?.edgeVelocityMode)
    ? Number(rb.edgeVelocityMode[edgeIndex])
    : Number.NaN;
  if (velocityModeRaw === 0) return 0;
  const bodyModeRaw = Array.isArray(rb?.edgeBodyMode)
    ? Number(rb.edgeBodyMode[edgeIndex])
    : Number.NaN;
  return bodyModeRaw === 0 ? 0 : 1;
}

function rigidEdgeMomentumScale(rb, allowPassEdgeFlowPush = false) {
  const arr = Array.isArray(rb?.edgeMomentumCoupling)
    ? rb.edgeMomentumCoupling
    : (Array.isArray(rb?.edgeMomentumTransfer) ? rb.edgeMomentumTransfer : null);
  const edgeCount = Math.max(
    Number(arr?.length) || 0,
    Array.isArray(rb?.edgeVelocityMode) ? rb.edgeVelocityMode.length : 0,
    Array.isArray(rb?.edgeBodyMode) ? rb.edgeBodyMode.length : 0,
    Array.isArray(rb?.verticesLocal) ? rb.verticesLocal.length : 0,
    Math.max(0, Number(rb?.sides) || 0),
  );
  if (edgeCount <= 0) return 1;

  let sum = 0;
  for (let ei = 0; ei < edgeCount; ei++) {
    const raw = Array.isArray(arr) ? Number(arr[ei]) : Number.NaN;
    let momentum = Number.isFinite(raw) ? clamp(raw, 0, 1) : 1;
    if (!allowPassEdgeFlowPush && resolveRigidEdgeVelocityMode(rb, ei) === 0) {
      momentum = 0;
    }
    sum += momentum;
  }
  return clamp(sum / Math.max(1, edgeCount), 0, 1);
}

function finiteOr(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function buildRigidStepState({
  sim,
  bodies,
  vxField,
  vyField,
  n,
  dt,
  dtNorm,
  swimGain,
  obstacleMask,
  bodyFeedbackPrevVx,
  bodyFeedbackPrevVy,
  selfFeedbackSuppression,
  rigidVerticesWorld,
  sampleFluidForBodyCoupling,
  localHoneyDrag,
  allowPassEdgeFlowPush,
}) {
  const rigidBodies = Array.isArray(bodies?.rigid) ? bodies.rigid : [];
  const bodyCount = rigidBodies.length;

  const bodyVx = new Float32Array(bodyCount);
  const bodyVy = new Float32Array(bodyCount);
  const bodyOmega = new Float32Array(bodyCount);
  const bodyX = new Float32Array(bodyCount);
  const bodyY = new Float32Array(bodyCount);
  const bodyTheta = new Float32Array(bodyCount);
  const bodyMass = new Float32Array(bodyCount);
  const bodyInertia = new Float32Array(bodyCount);
  const bodyEdgeMomentum = new Float32Array(bodyCount);
  const bodyCenterHoney = new Float32Array(bodyCount);
  const bodySwimX = new Float32Array(bodyCount);
  const bodySwimY = new Float32Array(bodyCount);
  const bodySwimTorque = new Float32Array(bodyCount);
  const sampleOffsets = new Int32Array(bodyCount + 1);

  const sampleRx = [];
  const sampleRy = [];
  const sampleFx = [];
  const sampleFy = [];
  const sampleHoney = [];

  for (let bi = 0; bi < bodyCount; bi++) {
    const b = rigidBodies[bi];
    sampleOffsets[bi] = sampleRx.length;

    const mass = Math.max(0.05, finiteOr(b?.mass, 1));
    const inertia = Math.max(0.05, finiteOr(b?.inertia, 1));
    const invMass = 1 / mass;
    const swimPhase = finiteOr(sim?.frame, 0) * 0.08 + bi * 2.1;

    bodyVx[bi] = finiteOr(b?.vx, 0);
    bodyVy[bi] = finiteOr(b?.vy, 0);
    bodyOmega[bi] = finiteOr(b?.omega, 0);
    bodyX[bi] = finiteOr(b?.x, 0);
    bodyY[bi] = finiteOr(b?.y, 0);
    bodyTheta[bi] = finiteOr(b?.theta, 0);
    bodyMass[bi] = mass;
    bodyInertia[bi] = inertia;
    bodyEdgeMomentum[bi] = rigidEdgeMomentumScale(b, allowPassEdgeFlowPush);
    bodyCenterHoney[bi] = finiteOr(localHoneyDrag(bodyX[bi], bodyY[bi]), 1);
    bodySwimX[bi] = swimGain * Math.cos(swimPhase) * 0.012 * invMass;
    bodySwimY[bi] = swimGain * Math.sin(swimPhase * 1.6) * 0.009 * invMass;
    bodySwimTorque[bi] = swimGain * Math.sin(swimPhase * 1.1) * 0.0025;

    const sampleVerts = rigidVerticesWorld(b);
    for (let si = 0; si < sampleVerts.length; si++) {
      const sx = finiteOr(sampleVerts[si]?.x, bodyX[bi]);
      const sy = finiteOr(sampleVerts[si]?.y, bodyY[bi]);
      const rx = sx - bodyX[bi];
      const ry = sy - bodyY[bi];
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
      sampleRx.push(finiteOr(rx, 0));
      sampleRy.push(finiteOr(ry, 0));
      sampleFx.push(finiteOr(fx, 0));
      sampleFy.push(finiteOr(fy, 0));
      sampleHoney.push(finiteOr(localHoneyDrag(sx, sy), 1));
    }
  }
  sampleOffsets[bodyCount] = sampleRx.length;

  return {
    bodyCount,
    bodyVx,
    bodyVy,
    bodyOmega,
    bodyX,
    bodyY,
    bodyTheta,
    bodyMass,
    bodyInertia,
    bodyEdgeMomentum,
    bodyCenterHoney,
    bodySwimX,
    bodySwimY,
    bodySwimTorque,
    sampleOffsets,
    sampleRx: Float32Array.from(sampleRx),
    sampleRy: Float32Array.from(sampleRy),
    sampleFx: Float32Array.from(sampleFx),
    sampleFy: Float32Array.from(sampleFy),
    sampleHoney: Float32Array.from(sampleHoney),
    dt: finiteOr(dt, 0.01),
    dtNorm: finiteOr(dtNorm, 1),
  };
}

function stepWithRuntime(runtime, args) {
  const totalStartMs = nowMs();
  const marshalStartMs = nowMs();
  const state = buildRigidStepState(args);
  const jsMarshalMs = nowMs() - marshalStartMs;

  const rigidBodies = Array.isArray(args?.bodies?.rigid) ? args.bodies.rigid : [];
  if (state.bodyCount === 0) {
    return {
      ok: true,
      rigidCarryTransfer: 0,
      processedBodies: 0,
      sampleCount: 0,
      jsMarshalMs,
      backendComputeMs: 0,
      totalMs: jsMarshalMs,
    };
  }

  const bytes = [
    state.bodyVx,
    state.bodyVy,
    state.bodyOmega,
    state.bodyX,
    state.bodyY,
    state.bodyTheta,
    state.bodyMass,
    state.bodyInertia,
    state.bodyEdgeMomentum,
    state.bodyCenterHoney,
    state.bodySwimX,
    state.bodySwimY,
    state.bodySwimTorque,
    state.sampleOffsets,
    state.sampleRx,
    state.sampleRy,
    state.sampleFx,
    state.sampleFy,
    state.sampleHoney,
  ].reduce((sum, arr) => sum + arr.byteLength, 0);

  const bytesStats = 3 * Float32Array.BYTES_PER_ELEMENT;
  ensureMemoryCapacity(runtime, runtime.heapBase + bytes + bytesStats + 64);
  runtime.exports.rsb_reset_heap();

  const alloc = (size) => Number(runtime.exports.rsb_alloc(size));

  const bodyVxPtr = alloc(state.bodyVx.byteLength);
  const bodyVyPtr = alloc(state.bodyVy.byteLength);
  const bodyOmegaPtr = alloc(state.bodyOmega.byteLength);
  const bodyXPtr = alloc(state.bodyX.byteLength);
  const bodyYPtr = alloc(state.bodyY.byteLength);
  const bodyThetaPtr = alloc(state.bodyTheta.byteLength);
  const bodyMassPtr = alloc(state.bodyMass.byteLength);
  const bodyInertiaPtr = alloc(state.bodyInertia.byteLength);
  const bodyEdgeMomentumPtr = alloc(state.bodyEdgeMomentum.byteLength);
  const bodyCenterHoneyPtr = alloc(state.bodyCenterHoney.byteLength);
  const bodySwimXPtr = alloc(state.bodySwimX.byteLength);
  const bodySwimYPtr = alloc(state.bodySwimY.byteLength);
  const bodySwimTorquePtr = alloc(state.bodySwimTorque.byteLength);

  const sampleOffsetsPtr = alloc(state.sampleOffsets.byteLength);
  const sampleRxPtr = alloc(state.sampleRx.byteLength);
  const sampleRyPtr = alloc(state.sampleRy.byteLength);
  const sampleFxPtr = alloc(state.sampleFx.byteLength);
  const sampleFyPtr = alloc(state.sampleFy.byteLength);
  const sampleHoneyPtr = alloc(state.sampleHoney.byteLength);

  const statsPtr = alloc(bytesStats);

  const memF32 = new Float32Array(runtime.memory.buffer);
  const memI32 = new Int32Array(runtime.memory.buffer);

  memF32.set(state.bodyVx, bodyVxPtr >> 2);
  memF32.set(state.bodyVy, bodyVyPtr >> 2);
  memF32.set(state.bodyOmega, bodyOmegaPtr >> 2);
  memF32.set(state.bodyX, bodyXPtr >> 2);
  memF32.set(state.bodyY, bodyYPtr >> 2);
  memF32.set(state.bodyTheta, bodyThetaPtr >> 2);
  memF32.set(state.bodyMass, bodyMassPtr >> 2);
  memF32.set(state.bodyInertia, bodyInertiaPtr >> 2);
  memF32.set(state.bodyEdgeMomentum, bodyEdgeMomentumPtr >> 2);
  memF32.set(state.bodyCenterHoney, bodyCenterHoneyPtr >> 2);
  memF32.set(state.bodySwimX, bodySwimXPtr >> 2);
  memF32.set(state.bodySwimY, bodySwimYPtr >> 2);
  memF32.set(state.bodySwimTorque, bodySwimTorquePtr >> 2);

  memI32.set(state.sampleOffsets, sampleOffsetsPtr >> 2);
  memF32.set(state.sampleRx, sampleRxPtr >> 2);
  memF32.set(state.sampleRy, sampleRyPtr >> 2);
  memF32.set(state.sampleFx, sampleFxPtr >> 2);
  memF32.set(state.sampleFy, sampleFyPtr >> 2);
  memF32.set(state.sampleHoney, sampleHoneyPtr >> 2);

  const backendStartMs = nowMs();
  const processedBodies = Number(runtime.exports.rsb_step_bodies(
    state.bodyCount,
    bodyVxPtr,
    bodyVyPtr,
    bodyOmegaPtr,
    bodyXPtr,
    bodyYPtr,
    bodyThetaPtr,
    bodyMassPtr,
    bodyInertiaPtr,
    bodyEdgeMomentumPtr,
    bodyCenterHoneyPtr,
    bodySwimXPtr,
    bodySwimYPtr,
    bodySwimTorquePtr,
    sampleOffsetsPtr,
    sampleRxPtr,
    sampleRyPtr,
    sampleFxPtr,
    sampleFyPtr,
    sampleHoneyPtr,
    state.dt,
    state.dtNorm,
    finiteOr(args?.dragK, 0),
    finiteOr(args?.worldSize, finiteOr(args?.n, 0)),
    statsPtr,
  )) | 0;
  const backendComputeMs = nowMs() - backendStartMs;

  const outVx = memF32.subarray(bodyVxPtr >> 2, (bodyVxPtr >> 2) + state.bodyCount);
  const outVy = memF32.subarray(bodyVyPtr >> 2, (bodyVyPtr >> 2) + state.bodyCount);
  const outOmega = memF32.subarray(bodyOmegaPtr >> 2, (bodyOmegaPtr >> 2) + state.bodyCount);
  const outX = memF32.subarray(bodyXPtr >> 2, (bodyXPtr >> 2) + state.bodyCount);
  const outY = memF32.subarray(bodyYPtr >> 2, (bodyYPtr >> 2) + state.bodyCount);
  const outTheta = memF32.subarray(bodyThetaPtr >> 2, (bodyThetaPtr >> 2) + state.bodyCount);

  for (let i = 0; i < state.bodyCount; i++) {
    const b = rigidBodies[i];
    if (!b) continue;
    b.vx = finiteOr(outVx[i], 0);
    b.vy = finiteOr(outVy[i], 0);
    b.omega = finiteOr(outOmega[i], 0);
    b.x = finiteOr(outX[i], 0);
    b.y = finiteOr(outY[i], 0);
    b.theta = finiteOr(outTheta[i], 0);
  }

  const statsIndex = statsPtr >> 2;
  const rigidCarryTransfer = finiteOr(memF32[statsIndex], 0);
  const sampleCount = Math.max(0, Math.round(finiteOr(memF32[statsIndex + 1], 0)));

  return {
    ok: true,
    rigidCarryTransfer,
    processedBodies,
    sampleCount,
    jsMarshalMs,
    backendComputeMs,
    totalMs: nowMs() - totalStartMs,
  };
}

export async function loadRigidStepBaselineBackendWasm() {
  const runtime = await getRuntime();
  return {
    label: 'wasm-rigid-step-v1',
    stepBodies: (args) => stepWithRuntime(runtime, args),
  };
}
