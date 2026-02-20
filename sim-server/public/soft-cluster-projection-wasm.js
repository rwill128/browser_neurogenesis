const WASM_URL = '/wasm/soft-cluster-projection.wasm?v=20260219a';
const DEFAULT_HEAP_BASE = 65536;

let runtimePromise = null;

function nowMs() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

async function instantiateWasm() {
  const imports = {};
  if (typeof WebAssembly.instantiateStreaming === 'function') {
    try {
      const streamed = await WebAssembly.instantiateStreaming(fetch(WASM_URL), imports);
      return streamed.instance || streamed;
    } catch {
      // Fall back when server MIME is not wasm.
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
        throw new Error('scp wasm: missing exported memory');
      }
      if (
        typeof exports.scp_alloc !== 'function'
        || typeof exports.scp_reset_heap !== 'function'
        || typeof exports.scp_project_nodes !== 'function'
      ) {
        throw new Error('scp wasm: missing required exports');
      }
      const heapBase = typeof exports.scp_heap_base === 'function'
        ? (Number(exports.scp_heap_base()) || DEFAULT_HEAP_BASE)
        : DEFAULT_HEAP_BASE;
      return { instance, exports, memory, heapBase };
    })();
  }
  return runtimePromise;
}

function ensureMemoryCapacity(runtime, requiredBytes) {
  const memory = runtime.memory;
  const pageSize = 64 * 1024;
  const current = memory.buffer.byteLength;
  if (current >= requiredBytes) return;
  const missing = requiredBytes - current;
  memory.grow(Math.max(1, Math.ceil(missing / pageSize)));
}

function finiteOr(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function buildProjectionState(nodes, clusterKinematics, membraneClusterSet) {
  const clusterEntries = Array.from(clusterKinematics.values())
    .map((st) => ({
      cid: Number.isFinite(Number(st?.clusterId)) ? Number(st.clusterId) : 0,
      x: finiteOr(st?.x, 0),
      y: finiteOr(st?.y, 0),
      vx: finiteOr(st?.vx, 0),
      vy: finiteOr(st?.vy, 0),
      omega: finiteOr(st?.omega, 0),
      membrane: membraneClusterSet instanceof Set && membraneClusterSet.has(Number(st?.clusterId)) ? 1 : 0,
    }))
    .sort((a, b) => a.cid - b.cid);

  const clusterCount = clusterEntries.length;
  const clusterIdToIndex = new Map();

  const clusterX = new Float32Array(clusterCount);
  const clusterY = new Float32Array(clusterCount);
  const clusterVx = new Float32Array(clusterCount);
  const clusterVy = new Float32Array(clusterCount);
  const clusterOmega = new Float32Array(clusterCount);
  const clusterMembrane = new Int32Array(clusterCount);

  for (let i = 0; i < clusterCount; i++) {
    const e = clusterEntries[i];
    clusterIdToIndex.set(e.cid, i);
    clusterX[i] = e.x;
    clusterY[i] = e.y;
    clusterVx[i] = e.vx;
    clusterVy[i] = e.vy;
    clusterOmega[i] = e.omega;
    clusterMembrane[i] = e.membrane;
  }

  const nodeCount = Array.isArray(nodes) ? nodes.length : 0;
  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  const nodeVx = new Float32Array(nodeCount);
  const nodeVy = new Float32Array(nodeCount);
  const nodeClusterIndex = new Int32Array(nodeCount);

  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i];
    const cid = Number.isFinite(Number(node?.clusterId)) ? Number(node.clusterId) : 0;
    const cidx = clusterIdToIndex.has(cid) ? clusterIdToIndex.get(cid) : -1;

    nodeX[i] = finiteOr(node?.x, 0);
    nodeY[i] = finiteOr(node?.y, 0);
    nodeVx[i] = finiteOr(node?.vx, 0);
    nodeVy[i] = finiteOr(node?.vy, 0);
    nodeClusterIndex[i] = Number.isFinite(Number(cidx)) ? Number(cidx) : -1;
  }

  return {
    nodeCount,
    nodeX,
    nodeY,
    nodeVx,
    nodeVy,
    nodeClusterIndex,
    clusterCount,
    clusterX,
    clusterY,
    clusterVx,
    clusterVy,
    clusterOmega,
    clusterMembrane,
  };
}

function projectWithRuntime(runtime, {
  nodes,
  clusterKinematics,
  linearGain,
  angularGain,
  membraneClusterSet,
  membraneGainScale,
}) {
  const totalStartMs = nowMs();
  const marshalStartMs = nowMs();
  const state = buildProjectionState(nodes, clusterKinematics, membraneClusterSet);
  const marshalMs = nowMs() - marshalStartMs;

  if (!Array.isArray(nodes) || state.nodeCount === 0 || state.clusterCount === 0) {
    return {
      ok: true,
      projectedNodes: 0,
      meanDelta: 0,
      jsMarshalMs: marshalMs,
      backendComputeMs: 0,
      totalMs: marshalMs,
    };
  }

  const bytesNodeX = state.nodeX.byteLength;
  const bytesNodeY = state.nodeY.byteLength;
  const bytesNodeVx = state.nodeVx.byteLength;
  const bytesNodeVy = state.nodeVy.byteLength;
  const bytesNodeCluster = state.nodeClusterIndex.byteLength;

  const bytesClusterX = state.clusterX.byteLength;
  const bytesClusterY = state.clusterY.byteLength;
  const bytesClusterVx = state.clusterVx.byteLength;
  const bytesClusterVy = state.clusterVy.byteLength;
  const bytesClusterOmega = state.clusterOmega.byteLength;
  const bytesClusterMembrane = state.clusterMembrane.byteLength;

  const bytesStats = 4;
  const totalBytes = bytesNodeX + bytesNodeY + bytesNodeVx + bytesNodeVy + bytesNodeCluster
    + bytesClusterX + bytesClusterY + bytesClusterVx + bytesClusterVy + bytesClusterOmega + bytesClusterMembrane
    + bytesStats + 64;

  ensureMemoryCapacity(runtime, runtime.heapBase + totalBytes);
  runtime.exports.scp_reset_heap();

  const nodeXPtr = Number(runtime.exports.scp_alloc(bytesNodeX));
  const nodeYPtr = Number(runtime.exports.scp_alloc(bytesNodeY));
  const nodeVxPtr = Number(runtime.exports.scp_alloc(bytesNodeVx));
  const nodeVyPtr = Number(runtime.exports.scp_alloc(bytesNodeVy));
  const nodeClusterPtr = Number(runtime.exports.scp_alloc(bytesNodeCluster));

  const clusterXPtr = Number(runtime.exports.scp_alloc(bytesClusterX));
  const clusterYPtr = Number(runtime.exports.scp_alloc(bytesClusterY));
  const clusterVxPtr = Number(runtime.exports.scp_alloc(bytesClusterVx));
  const clusterVyPtr = Number(runtime.exports.scp_alloc(bytesClusterVy));
  const clusterOmegaPtr = Number(runtime.exports.scp_alloc(bytesClusterOmega));
  const clusterMembranePtr = Number(runtime.exports.scp_alloc(bytesClusterMembrane));

  const statsPtr = Number(runtime.exports.scp_alloc(bytesStats));

  const memF32 = new Float32Array(runtime.memory.buffer);
  const memI32 = new Int32Array(runtime.memory.buffer);

  memF32.set(state.nodeX, nodeXPtr >> 2);
  memF32.set(state.nodeY, nodeYPtr >> 2);
  memF32.set(state.nodeVx, nodeVxPtr >> 2);
  memF32.set(state.nodeVy, nodeVyPtr >> 2);
  memI32.set(state.nodeClusterIndex, nodeClusterPtr >> 2);

  memF32.set(state.clusterX, clusterXPtr >> 2);
  memF32.set(state.clusterY, clusterYPtr >> 2);
  memF32.set(state.clusterVx, clusterVxPtr >> 2);
  memF32.set(state.clusterVy, clusterVyPtr >> 2);
  memF32.set(state.clusterOmega, clusterOmegaPtr >> 2);
  memI32.set(state.clusterMembrane, clusterMembranePtr >> 2);

  const computeStartMs = nowMs();
  const projectedNodes = Number(runtime.exports.scp_project_nodes(
    state.nodeCount,
    nodeXPtr,
    nodeYPtr,
    nodeVxPtr,
    nodeVyPtr,
    nodeClusterPtr,
    state.clusterCount,
    clusterXPtr,
    clusterYPtr,
    clusterVxPtr,
    clusterVyPtr,
    clusterOmegaPtr,
    clusterMembranePtr,
    Number(linearGain) || 0,
    Number(angularGain) || 0,
    Number(membraneGainScale) || 1,
    statsPtr,
  )) | 0;
  const backendComputeMs = nowMs() - computeStartMs;

  const outVx = memF32.subarray(nodeVxPtr >> 2, (nodeVxPtr >> 2) + state.nodeCount);
  const outVy = memF32.subarray(nodeVyPtr >> 2, (nodeVyPtr >> 2) + state.nodeCount);
  for (let i = 0; i < state.nodeCount; i++) {
    const node = nodes[i];
    if (!node) continue;
    node.vx = Number(outVx[i]) || 0;
    node.vy = Number(outVy[i]) || 0;
  }

  const sumDelta = Number(memF32[statsPtr >> 2]) || 0;
  const totalMs = nowMs() - totalStartMs;
  return {
    ok: true,
    projectedNodes,
    meanDelta: projectedNodes > 0 ? (sumDelta / projectedNodes) : 0,
    jsMarshalMs: marshalMs,
    backendComputeMs,
    totalMs,
  };
}

export async function loadSoftClusterProjectionBackendWasm() {
  const runtime = await getRuntime();
  return {
    label: 'wasm-scp-v1',
    project: (args) => projectWithRuntime(runtime, args),
  };
}
