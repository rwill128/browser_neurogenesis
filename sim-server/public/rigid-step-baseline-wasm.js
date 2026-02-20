const WASM_URL = '/wasm/rigid-step-baseline.wasm?v=20260219c';
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
      return {
        instance,
        exports,
        memory,
        heapBase,
        topology: {
          signature: null,
          bodyCount: 0,
          sampleCount: 0,
          counts: new Int32Array(0),
        },
        pendingSync: null,
        frameBuffers: {
          bodyCapacity: 0,
          sampleCapacity: 0,
          bodyVx: new Float32Array(0),
          bodyVy: new Float32Array(0),
          bodyOmega: new Float32Array(0),
          bodyX: new Float32Array(0),
          bodyY: new Float32Array(0),
          bodyTheta: new Float32Array(0),
          bodyMass: new Float32Array(0),
          bodyInertia: new Float32Array(0),
          bodyEdgeMomentum: new Float32Array(0),
          bodyCenterHoney: new Float32Array(0),
          bodySwimX: new Float32Array(0),
          bodySwimY: new Float32Array(0),
          bodySwimTorque: new Float32Array(0),
          sampleOffsets: new Int32Array(0),
          sampleRx: new Float32Array(0),
          sampleRy: new Float32Array(0),
          sampleFx: new Float32Array(0),
          sampleFy: new Float32Array(0),
          sampleHoney: new Float32Array(0),
        },
        wasmLayout: null,
      };
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

function getRigidBodySampleCount(body) {
  if (Array.isArray(body?.verticesLocal) && body.verticesLocal.length >= 3) {
    return body.verticesLocal.length;
  }
  return Math.max(3, Number(body?.sides) || 3);
}

function ensureTopologyCounts(runtime, bodyCount) {
  const topology = runtime.topology;
  if ((topology.counts?.length || 0) >= bodyCount) return topology.counts;
  const nextLen = Math.max(bodyCount, Math.ceil((topology.counts?.length || 0) * 1.5), 64);
  topology.counts = new Int32Array(nextLen);
  return topology.counts;
}

function ensureFrameBuffers(runtime, bodyCount, sampleCount) {
  const fb = runtime.frameBuffers;
  const needBody = Math.max(1, bodyCount);
  const needSample = Math.max(1, sampleCount);
  const growBody = fb.bodyCapacity < needBody;
  const growSample = fb.sampleCapacity < needSample;

  if (!growBody && !growSample) return fb;

  const nextBodyCapacity = growBody
    ? Math.max(needBody, Math.ceil(fb.bodyCapacity * 1.5), 64)
    : fb.bodyCapacity;
  const nextSampleCapacity = growSample
    ? Math.max(needSample, Math.ceil(fb.sampleCapacity * 1.5), 256)
    : fb.sampleCapacity;

  runtime.frameBuffers = {
    bodyCapacity: nextBodyCapacity,
    sampleCapacity: nextSampleCapacity,
    bodyVx: new Float32Array(nextBodyCapacity),
    bodyVy: new Float32Array(nextBodyCapacity),
    bodyOmega: new Float32Array(nextBodyCapacity),
    bodyX: new Float32Array(nextBodyCapacity),
    bodyY: new Float32Array(nextBodyCapacity),
    bodyTheta: new Float32Array(nextBodyCapacity),
    bodyMass: new Float32Array(nextBodyCapacity),
    bodyInertia: new Float32Array(nextBodyCapacity),
    bodyEdgeMomentum: new Float32Array(nextBodyCapacity),
    bodyCenterHoney: new Float32Array(nextBodyCapacity),
    bodySwimX: new Float32Array(nextBodyCapacity),
    bodySwimY: new Float32Array(nextBodyCapacity),
    bodySwimTorque: new Float32Array(nextBodyCapacity),
    sampleOffsets: new Int32Array(nextBodyCapacity + 1),
    sampleRx: new Float32Array(nextSampleCapacity),
    sampleRy: new Float32Array(nextSampleCapacity),
    sampleFx: new Float32Array(nextSampleCapacity),
    sampleFy: new Float32Array(nextSampleCapacity),
    sampleHoney: new Float32Array(nextSampleCapacity),
  };

  return runtime.frameBuffers;
}

function buildRigidTopology(runtime, rigidBodies) {
  const bodyCount = rigidBodies.length;
  const counts = ensureTopologyCounts(runtime, bodyCount);

  let sampleCount = 0;
  let hash = (bodyCount * 2654435761) >>> 0;
  for (let i = 0; i < bodyCount; i++) {
    const c = getRigidBodySampleCount(rigidBodies[i]);
    counts[i] = c;
    sampleCount += c;
    hash = (Math.imul(hash ^ (c + i + 1), 2246822519) + 3266489917) >>> 0;
  }

  const signature = `${bodyCount}:${sampleCount}:${hash}`;
  return { bodyCount, sampleCount, signature, counts };
}

function ensureSampleOffsets(runtime, frameBuffers, topology) {
  const topoState = runtime.topology;
  const cacheHit = topoState.signature === topology.signature
    && topoState.bodyCount === topology.bodyCount
    && topoState.sampleCount === topology.sampleCount;

  if (!cacheHit) {
    let offset = 0;
    for (let i = 0; i < topology.bodyCount; i++) {
      frameBuffers.sampleOffsets[i] = offset;
      offset += topology.counts[i] | 0;
    }
    frameBuffers.sampleOffsets[topology.bodyCount] = offset;

    topoState.signature = topology.signature;
    topoState.bodyCount = topology.bodyCount;
    topoState.sampleCount = topology.sampleCount;
  }

  return cacheHit;
}

function buildRigidStepState(runtime, {
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
  sampleFluidForBodyCoupling,
  localHoneyDrag,
  allowPassEdgeFlowPush,
}) {
  const rigidBodies = Array.isArray(bodies?.rigid) ? bodies.rigid : [];
  const topology = buildRigidTopology(runtime, rigidBodies);
  const frameBuffers = ensureFrameBuffers(runtime, topology.bodyCount, topology.sampleCount);
  const topologyCacheHit = ensureSampleOffsets(runtime, frameBuffers, topology);

  const bodyCount = topology.bodyCount;
  const sampleCount = topology.sampleCount;

  for (let bi = 0; bi < bodyCount; bi++) {
    const b = rigidBodies[bi];

    const mass = Math.max(0.05, finiteOr(b?.mass, 1));
    const inertia = Math.max(0.05, finiteOr(b?.inertia, 1));
    const invMass = 1 / mass;
    const swimPhase = finiteOr(sim?.frame, 0) * 0.08 + bi * 2.1;

    const bx = finiteOr(b?.x, 0);
    const by = finiteOr(b?.y, 0);
    const theta = finiteOr(b?.theta, 0);

    frameBuffers.bodyVx[bi] = finiteOr(b?.vx, 0);
    frameBuffers.bodyVy[bi] = finiteOr(b?.vy, 0);
    frameBuffers.bodyOmega[bi] = finiteOr(b?.omega, 0);
    frameBuffers.bodyX[bi] = bx;
    frameBuffers.bodyY[bi] = by;
    frameBuffers.bodyTheta[bi] = theta;
    frameBuffers.bodyMass[bi] = mass;
    frameBuffers.bodyInertia[bi] = inertia;
    frameBuffers.bodyEdgeMomentum[bi] = rigidEdgeMomentumScale(b, allowPassEdgeFlowPush);
    frameBuffers.bodyCenterHoney[bi] = finiteOr(localHoneyDrag(bx, by), 1);
    frameBuffers.bodySwimX[bi] = swimGain * Math.cos(swimPhase) * 0.012 * invMass;
    frameBuffers.bodySwimY[bi] = swimGain * Math.sin(swimPhase * 1.6) * 0.009 * invMass;
    frameBuffers.bodySwimTorque[bi] = swimGain * Math.sin(swimPhase * 1.1) * 0.0025;

    let write = frameBuffers.sampleOffsets[bi] | 0;

    if (Array.isArray(b?.verticesLocal) && b.verticesLocal.length >= 3) {
      const verts = b.verticesLocal;
      const c = Math.cos(theta);
      const s = Math.sin(theta);
      for (let vi = 0; vi < verts.length; vi++) {
        const lv = verts[vi] || {};
        const lx = finiteOr(lv.x, 0);
        const ly = finiteOr(lv.y, 0);
        const sx = bx + lx * c - ly * s;
        const sy = by + lx * s + ly * c;
        const rx = sx - bx;
        const ry = sy - by;

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

        frameBuffers.sampleRx[write] = finiteOr(rx, 0);
        frameBuffers.sampleRy[write] = finiteOr(ry, 0);
        frameBuffers.sampleFx[write] = finiteOr(fx, 0);
        frameBuffers.sampleFy[write] = finiteOr(fy, 0);
        frameBuffers.sampleHoney[write] = finiteOr(localHoneyDrag(sx, sy), 1);
        write += 1;
      }
    } else {
      const sides = Math.max(3, Number(b?.sides) || 3);
      const r = finiteOr(b?.r, 0);
      const rot = theta + (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
      for (let vi = 0; vi < sides; vi++) {
        const a = rot + (vi / sides) * Math.PI * 2;
        const sx = bx + Math.cos(a) * r;
        const sy = by + Math.sin(a) * r;
        const rx = sx - bx;
        const ry = sy - by;

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

        frameBuffers.sampleRx[write] = finiteOr(rx, 0);
        frameBuffers.sampleRy[write] = finiteOr(ry, 0);
        frameBuffers.sampleFx[write] = finiteOr(fx, 0);
        frameBuffers.sampleFy[write] = finiteOr(fy, 0);
        frameBuffers.sampleHoney[write] = finiteOr(localHoneyDrag(sx, sy), 1);
        write += 1;
      }
    }
  }

  return {
    rigidBodies,
    bodyCount,
    sampleCount,
    bodyVx: frameBuffers.bodyVx,
    bodyVy: frameBuffers.bodyVy,
    bodyOmega: frameBuffers.bodyOmega,
    bodyX: frameBuffers.bodyX,
    bodyY: frameBuffers.bodyY,
    bodyTheta: frameBuffers.bodyTheta,
    bodyMass: frameBuffers.bodyMass,
    bodyInertia: frameBuffers.bodyInertia,
    bodyEdgeMomentum: frameBuffers.bodyEdgeMomentum,
    bodyCenterHoney: frameBuffers.bodyCenterHoney,
    bodySwimX: frameBuffers.bodySwimX,
    bodySwimY: frameBuffers.bodySwimY,
    bodySwimTorque: frameBuffers.bodySwimTorque,
    sampleOffsets: frameBuffers.sampleOffsets,
    sampleRx: frameBuffers.sampleRx,
    sampleRy: frameBuffers.sampleRy,
    sampleFx: frameBuffers.sampleFx,
    sampleFy: frameBuffers.sampleFy,
    sampleHoney: frameBuffers.sampleHoney,
    dt: finiteOr(dt, 0.01),
    dtNorm: finiteOr(dtNorm, 1),
    topologyCacheHit,
  };
}

function ensureWasmLayout(runtime, bodyCount, sampleCount) {
  const prev = runtime.wasmLayout;
  const needBody = Math.max(1, bodyCount);
  const needSample = Math.max(1, sampleCount);

  if (prev && prev.bodyCapacity >= needBody && prev.sampleCapacity >= needSample) {
    return { layout: prev, wasmBufferReuse: 'reused' };
  }

  const bodyCapacity = Math.max(
    needBody,
    prev ? Math.ceil(prev.bodyCapacity * 1.5) : 0,
    64,
  );
  const sampleCapacity = Math.max(
    needSample,
    prev ? Math.ceil(prev.sampleCapacity * 1.5) : 0,
    256,
  );

  const bodyBytes = bodyCapacity * Float32Array.BYTES_PER_ELEMENT;
  const sampleBytes = sampleCapacity * Float32Array.BYTES_PER_ELEMENT;
  const sampleOffsetBytes = (bodyCapacity + 1) * Int32Array.BYTES_PER_ELEMENT;
  const bytesStats = 3 * Float32Array.BYTES_PER_ELEMENT;

  const bytesTotal =
    (bodyBytes * 13)
    + sampleOffsetBytes
    + (sampleBytes * 5)
    + bytesStats
    + 128;

  ensureMemoryCapacity(runtime, runtime.heapBase + bytesTotal);
  runtime.exports.rsb_reset_heap();
  const alloc = (size) => Number(runtime.exports.rsb_alloc(size));

  const layout = {
    bodyCapacity,
    sampleCapacity,
    bodyVxPtr: alloc(bodyBytes),
    bodyVyPtr: alloc(bodyBytes),
    bodyOmegaPtr: alloc(bodyBytes),
    bodyXPtr: alloc(bodyBytes),
    bodyYPtr: alloc(bodyBytes),
    bodyThetaPtr: alloc(bodyBytes),
    bodyMassPtr: alloc(bodyBytes),
    bodyInertiaPtr: alloc(bodyBytes),
    bodyEdgeMomentumPtr: alloc(bodyBytes),
    bodyCenterHoneyPtr: alloc(bodyBytes),
    bodySwimXPtr: alloc(bodyBytes),
    bodySwimYPtr: alloc(bodyBytes),
    bodySwimTorquePtr: alloc(bodyBytes),
    sampleOffsetsPtr: alloc(sampleOffsetBytes),
    sampleRxPtr: alloc(sampleBytes),
    sampleRyPtr: alloc(sampleBytes),
    sampleFxPtr: alloc(sampleBytes),
    sampleFyPtr: alloc(sampleBytes),
    sampleHoneyPtr: alloc(sampleBytes),
    statsPtr: alloc(bytesStats),
  };

  runtime.wasmLayout = layout;
  return { layout, wasmBufferReuse: prev ? 'resized' : 'init' };
}

function syncBodiesToJsWithRuntime(runtime, rigidBodies) {
  const pending = runtime?.pendingSync;
  if (!pending) {
    return {
      ok: true,
      syncedBodies: 0,
      syncCount: 0,
      syncBytes: 0,
      syncReason: null,
      syncMs: 0,
      pending: false,
    };
  }

  const syncStartMs = nowMs();
  const bodyCount = Math.max(0, Number(pending.bodyCount) || 0);
  const layout = pending.layout;
  const memF32 = new Float32Array(runtime.memory.buffer);

  const outVx = memF32.subarray(layout.bodyVxPtr >> 2, (layout.bodyVxPtr >> 2) + bodyCount);
  const outVy = memF32.subarray(layout.bodyVyPtr >> 2, (layout.bodyVyPtr >> 2) + bodyCount);
  const outOmega = memF32.subarray(layout.bodyOmegaPtr >> 2, (layout.bodyOmegaPtr >> 2) + bodyCount);
  const outX = memF32.subarray(layout.bodyXPtr >> 2, (layout.bodyXPtr >> 2) + bodyCount);
  const outY = memF32.subarray(layout.bodyYPtr >> 2, (layout.bodyYPtr >> 2) + bodyCount);
  const outTheta = memF32.subarray(layout.bodyThetaPtr >> 2, (layout.bodyThetaPtr >> 2) + bodyCount);

  for (let i = 0; i < bodyCount; i++) {
    const b = rigidBodies?.[i];
    if (!b) continue;
    b.vx = finiteOr(outVx[i], 0);
    b.vy = finiteOr(outVy[i], 0);
    b.omega = finiteOr(outOmega[i], 0);
    b.x = finiteOr(outX[i], 0);
    b.y = finiteOr(outY[i], 0);
    b.theta = finiteOr(outTheta[i], 0);
  }

  const syncMs = nowMs() - syncStartMs;
  runtime.pendingSync = null;
  return {
    ok: true,
    syncedBodies: bodyCount,
    syncCount: 1,
    syncBytes: bodyCount * 6 * Float32Array.BYTES_PER_ELEMENT,
    syncReason: String(pending.syncReason || 'collision-boundary'),
    syncMs,
    pending: false,
  };
}

function getRigidStateViewWithRuntime(runtime) {
  const pending = runtime?.pendingSync;
  if (!pending?.layout) return null;
  const layout = pending.layout;
  const bodyCount = Math.max(0, Number(pending.bodyCount) || 0);
  const memF32 = new Float32Array(runtime.memory.buffer);
  return {
    bodyCount,
    x: memF32.subarray(layout.bodyXPtr >> 2, (layout.bodyXPtr >> 2) + bodyCount),
    y: memF32.subarray(layout.bodyYPtr >> 2, (layout.bodyYPtr >> 2) + bodyCount),
    theta: memF32.subarray(layout.bodyThetaPtr >> 2, (layout.bodyThetaPtr >> 2) + bodyCount),
    vx: memF32.subarray(layout.bodyVxPtr >> 2, (layout.bodyVxPtr >> 2) + bodyCount),
    vy: memF32.subarray(layout.bodyVyPtr >> 2, (layout.bodyVyPtr >> 2) + bodyCount),
    omega: memF32.subarray(layout.bodyOmegaPtr >> 2, (layout.bodyOmegaPtr >> 2) + bodyCount),
    source: 'wasm-pending',
  };
}

function stepWithRuntime(runtime, args) {
  const totalStartMs = nowMs();
  const marshalStartMs = nowMs();
  const state = buildRigidStepState(runtime, args);
  const jsMarshalMs = nowMs() - marshalStartMs;

  if (state.bodyCount === 0) {
    return {
      ok: true,
      rigidCarryTransfer: 0,
      processedBodies: 0,
      sampleCount: 0,
      jsMarshalMs,
      backendComputeMs: 0,
      totalMs: jsMarshalMs,
      topologyCacheHit: state.topologyCacheHit,
      wasmBufferReuse: 'reused',
      includesPostIntegrate: true,
      syncCount: 0,
      syncBytes: 0,
      syncReason: null,
    };
  }

  const { layout, wasmBufferReuse } = ensureWasmLayout(runtime, state.bodyCount, state.sampleCount);
  const memF32 = new Float32Array(runtime.memory.buffer);
  const memI32 = new Int32Array(runtime.memory.buffer);

  const bodyCount = state.bodyCount;
  const sampleCount = state.sampleCount;

  memF32.set(state.bodyVx.subarray(0, bodyCount), layout.bodyVxPtr >> 2);
  memF32.set(state.bodyVy.subarray(0, bodyCount), layout.bodyVyPtr >> 2);
  memF32.set(state.bodyOmega.subarray(0, bodyCount), layout.bodyOmegaPtr >> 2);
  memF32.set(state.bodyX.subarray(0, bodyCount), layout.bodyXPtr >> 2);
  memF32.set(state.bodyY.subarray(0, bodyCount), layout.bodyYPtr >> 2);
  memF32.set(state.bodyTheta.subarray(0, bodyCount), layout.bodyThetaPtr >> 2);
  memF32.set(state.bodyMass.subarray(0, bodyCount), layout.bodyMassPtr >> 2);
  memF32.set(state.bodyInertia.subarray(0, bodyCount), layout.bodyInertiaPtr >> 2);
  memF32.set(state.bodyEdgeMomentum.subarray(0, bodyCount), layout.bodyEdgeMomentumPtr >> 2);
  memF32.set(state.bodyCenterHoney.subarray(0, bodyCount), layout.bodyCenterHoneyPtr >> 2);
  memF32.set(state.bodySwimX.subarray(0, bodyCount), layout.bodySwimXPtr >> 2);
  memF32.set(state.bodySwimY.subarray(0, bodyCount), layout.bodySwimYPtr >> 2);
  memF32.set(state.bodySwimTorque.subarray(0, bodyCount), layout.bodySwimTorquePtr >> 2);

  memI32.set(state.sampleOffsets.subarray(0, bodyCount + 1), layout.sampleOffsetsPtr >> 2);
  memF32.set(state.sampleRx.subarray(0, sampleCount), layout.sampleRxPtr >> 2);
  memF32.set(state.sampleRy.subarray(0, sampleCount), layout.sampleRyPtr >> 2);
  memF32.set(state.sampleFx.subarray(0, sampleCount), layout.sampleFxPtr >> 2);
  memF32.set(state.sampleFy.subarray(0, sampleCount), layout.sampleFyPtr >> 2);
  memF32.set(state.sampleHoney.subarray(0, sampleCount), layout.sampleHoneyPtr >> 2);

  const copyInBytes = (
    (bodyCount * 13 * Float32Array.BYTES_PER_ELEMENT)
    + ((bodyCount + 1) * Int32Array.BYTES_PER_ELEMENT)
    + (sampleCount * 5 * Float32Array.BYTES_PER_ELEMENT)
  );

  const postVelocityCap = finiteOr(args?.postVelocityCap, 4.0);
  const postOmegaCap = finiteOr(args?.postOmegaCap, 0.22);

  const backendStartMs = nowMs();
  const processedBodies = Number(runtime.exports.rsb_step_bodies(
    bodyCount,
    layout.bodyVxPtr,
    layout.bodyVyPtr,
    layout.bodyOmegaPtr,
    layout.bodyXPtr,
    layout.bodyYPtr,
    layout.bodyThetaPtr,
    layout.bodyMassPtr,
    layout.bodyInertiaPtr,
    layout.bodyEdgeMomentumPtr,
    layout.bodyCenterHoneyPtr,
    layout.bodySwimXPtr,
    layout.bodySwimYPtr,
    layout.bodySwimTorquePtr,
    layout.sampleOffsetsPtr,
    layout.sampleRxPtr,
    layout.sampleRyPtr,
    layout.sampleFxPtr,
    layout.sampleFyPtr,
    layout.sampleHoneyPtr,
    state.dt,
    state.dtNorm,
    finiteOr(args?.dragK, 0),
    finiteOr(args?.worldSize, finiteOr(args?.n, 0)),
    postVelocityCap,
    postOmegaCap,
    layout.statsPtr,
  )) | 0;
  const backendComputeMs = nowMs() - backendStartMs;

  const statsIndex = layout.statsPtr >> 2;
  const rigidCarryTransfer = finiteOr(memF32[statsIndex], 0);
  const outSampleCount = Math.max(0, Math.round(finiteOr(memF32[statsIndex + 1], 0)));

  const copyOutBytes = bodyCount * 6 * Float32Array.BYTES_PER_ELEMENT;
  const deferSyncToJs = args?.deferSyncToJs === true;

  if (deferSyncToJs) {
    runtime.pendingSync = {
      layout,
      bodyCount,
      syncReason: String(args?.syncReason || 'collision-boundary'),
    };
  } else {
    runtime.pendingSync = null;
    const outVx = memF32.subarray(layout.bodyVxPtr >> 2, (layout.bodyVxPtr >> 2) + bodyCount);
    const outVy = memF32.subarray(layout.bodyVyPtr >> 2, (layout.bodyVyPtr >> 2) + bodyCount);
    const outOmega = memF32.subarray(layout.bodyOmegaPtr >> 2, (layout.bodyOmegaPtr >> 2) + bodyCount);
    const outX = memF32.subarray(layout.bodyXPtr >> 2, (layout.bodyXPtr >> 2) + bodyCount);
    const outY = memF32.subarray(layout.bodyYPtr >> 2, (layout.bodyYPtr >> 2) + bodyCount);
    const outTheta = memF32.subarray(layout.bodyThetaPtr >> 2, (layout.bodyThetaPtr >> 2) + bodyCount);

    for (let i = 0; i < bodyCount; i++) {
      const b = state.rigidBodies[i];
      if (!b) continue;
      b.vx = finiteOr(outVx[i], 0);
      b.vy = finiteOr(outVy[i], 0);
      b.omega = finiteOr(outOmega[i], 0);
      b.x = finiteOr(outX[i], 0);
      b.y = finiteOr(outY[i], 0);
      b.theta = finiteOr(outTheta[i], 0);
    }
  }

  return {
    ok: true,
    rigidCarryTransfer,
    processedBodies,
    sampleCount: outSampleCount,
    jsMarshalMs,
    backendComputeMs,
    totalMs: nowMs() - totalStartMs,
    topologyCacheHit: state.topologyCacheHit,
    wasmBufferReuse,
    includesPostIntegrate: true,
    syncCount: deferSyncToJs ? 0 : 1,
    syncBytes: deferSyncToJs ? 0 : copyOutBytes,
    syncReason: deferSyncToJs ? null : 'post-step-immediate',
    copyInBytes,
    pendingSync: deferSyncToJs,
  };
}

export async function loadRigidStepBaselineBackendWasm() {
  const runtime = await getRuntime();
  return {
    label: 'wasm-rigid-step-v3',
    stepBodies: (args) => stepWithRuntime(runtime, args),
    getRigidStateView: () => getRigidStateViewWithRuntime(runtime),
    syncBodiesToJs: ({ bodies, reason } = {}) => {
      if (runtime?.pendingSync && reason) {
        runtime.pendingSync.syncReason = String(reason);
      }
      const rigidBodies = Array.isArray(bodies?.rigid)
        ? bodies.rigid
        : (Array.isArray(bodies) ? bodies : []);
      return syncBodiesToJsWithRuntime(runtime, rigidBodies);
    },
  };
}
