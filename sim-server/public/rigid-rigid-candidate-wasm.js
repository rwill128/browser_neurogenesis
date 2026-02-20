const WASM_URL = '/wasm/rigid-rigid-candidates.wasm?v=20260219a';
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
      // Fall through to arrayBuffer path (useful when content-type is not wasm).
    }
  }

  const res = await fetch(WASM_URL);
  if (!res.ok) {
    throw new Error(`failed to fetch wasm (${res.status})`);
  }
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
        throw new Error('rr wasm: missing exported memory');
      }
      if (typeof exports.rr_alloc !== 'function' || typeof exports.rr_reset_heap !== 'function' || typeof exports.rr_build_pairs !== 'function') {
        throw new Error('rr wasm: missing required exports');
      }
      const heapBase = typeof exports.rr_heap_base === 'function'
        ? (Number(exports.rr_heap_base()) || DEFAULT_HEAP_BASE)
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
  const pages = Math.ceil(missing / pageSize);
  memory.grow(Math.max(1, pages));
}

function buildNeighborStreams(state) {
  const bodyCount = Number(state?.bodyCount) | 0;
  const bodyCellKeys = Array.isArray(state?.bodyCellKeys) ? state.bodyCellKeys : [];
  const cells = state?.cells;

  const offsets = new Int32Array(Math.max(0, bodyCount) + 1);
  const neighbors = [];

  for (let i = 0; i < bodyCount; i++) {
    offsets[i] = neighbors.length;
    const cellKeys = bodyCellKeys[i];
    if (!Array.isArray(cellKeys) || cellKeys.length === 0) continue;
    for (let k = 0; k < cellKeys.length; k++) {
      const ids = cells?.get(cellKeys[k]) || [];
      for (let idx = 0; idx < ids.length; idx++) {
        neighbors.push(Number(ids[idx]) | 0);
      }
    }
  }
  offsets[bodyCount] = neighbors.length;

  return {
    bodyCount,
    offsets,
    neighbors: Int32Array.from(neighbors),
  };
}

function buildCandidatesWithRuntime(runtime, state) {
  const buildStartMs = nowMs();
  const { bodyCount, offsets, neighbors } = buildNeighborStreams(state);

  if (bodyCount <= 1) {
    return {
      pairs: [],
      stats: {
        emitAttempts: 0,
        duplicatesRejected: 0,
        dedupeMs: 0,
        sortMs: 0,
        totalBuildMs: 0,
      },
    };
  }

  const maxPairs = Math.max(0, Number(state?.bruteForcePairs) | 0);
  const bytesOffsets = offsets.byteLength;
  const bytesNeighbors = neighbors.byteLength;
  const bytesSeen = bodyCount * 4;
  const bytesPairs = maxPairs * 2 * 4;
  const bytesStats = 3 * 4;
  const bytesTotal = bytesOffsets + bytesNeighbors + bytesSeen + bytesPairs + bytesStats + 32;

  ensureMemoryCapacity(runtime, runtime.heapBase + bytesTotal);
  runtime.exports.rr_reset_heap();

  const offsetsPtr = Number(runtime.exports.rr_alloc(bytesOffsets));
  const neighborsPtr = Number(runtime.exports.rr_alloc(bytesNeighbors));
  const seenPtr = Number(runtime.exports.rr_alloc(bytesSeen));
  const pairsPtr = Number(runtime.exports.rr_alloc(bytesPairs));
  const statsPtr = Number(runtime.exports.rr_alloc(bytesStats));

  const memI32 = new Int32Array(runtime.memory.buffer);
  memI32.set(offsets, offsetsPtr >> 2);
  memI32.set(neighbors, neighborsPtr >> 2);

  const pairCount = Number(runtime.exports.rr_build_pairs(
    bodyCount,
    offsetsPtr,
    neighborsPtr,
    seenPtr,
    pairsPtr,
    statsPtr,
  )) | 0;

  const statsIdx = statsPtr >> 2;
  const emitAttempts = Number(memI32[statsIdx]) || 0;
  const duplicatesRejected = Number(memI32[statsIdx + 1]) || 0;

  const pairs = new Array(Math.max(0, pairCount));
  const pairBase = pairsPtr >> 2;
  for (let i = 0; i < pairCount; i++) {
    const o = pairBase + i * 2;
    pairs[i] = [Number(memI32[o]) | 0, Number(memI32[o + 1]) | 0];
  }

  const totalBuildMs = nowMs() - buildStartMs;
  return {
    pairs,
    stats: {
      emitAttempts,
      duplicatesRejected,
      dedupeMs: totalBuildMs,
      sortMs: 0,
      totalBuildMs,
    },
  };
}

export async function loadRigidRigidCandidateBackendWasm() {
  const runtime = await getRuntime();
  return {
    label: 'wasm-rr-v1',
    buildCandidates: (state) => buildCandidatesWithRuntime(runtime, state),
  };
}
