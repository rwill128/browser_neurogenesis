const WASM_URL = '/wasm/rigid-rigid-candidates.wasm?v=20260219b';
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
      // Fall through when MIME type blocks streaming instantiate.
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
        throw new Error('rr wasm: missing exported memory');
      }
      if (
        typeof exports.rr_alloc !== 'function'
        || typeof exports.rr_reset_heap !== 'function'
        || typeof exports.rr_build_pairs_from_cells !== 'function'
      ) {
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

function buildCellMembershipStateFromMaps(state) {
  const bodyCount = Number(state?.bodyCount) | 0;
  const cells = state?.cells;
  const bodyCellKeys = Array.isArray(state?.bodyCellKeys) ? state.bodyCellKeys : [];

  const cellKeys = Array.from(cells?.keys?.() || []);
  const keyToIndex = new Map();
  for (let ci = 0; ci < cellKeys.length; ci++) {
    keyToIndex.set(cellKeys[ci], ci);
  }

  const cellOffsets = new Int32Array(cellKeys.length + 1);
  let cellBodiesTotal = 0;
  for (let ci = 0; ci < cellKeys.length; ci++) {
    cellOffsets[ci] = cellBodiesTotal;
    const ids = cells?.get(cellKeys[ci]) || [];
    cellBodiesTotal += ids.length;
  }
  cellOffsets[cellKeys.length] = cellBodiesTotal;

  const cellBodyIds = new Int32Array(cellBodiesTotal);
  let cellWrite = 0;
  for (let ci = 0; ci < cellKeys.length; ci++) {
    const ids = cells?.get(cellKeys[ci]) || [];
    for (let k = 0; k < ids.length; k++) {
      cellBodyIds[cellWrite++] = Number(ids[k]) | 0;
    }
  }

  const bodyCellOffsets = new Int32Array(bodyCount + 1);
  let bodyCellsTotal = 0;
  for (let i = 0; i < bodyCount; i++) {
    bodyCellOffsets[i] = bodyCellsTotal;
    const keys = bodyCellKeys[i] || [];
    bodyCellsTotal += keys.length;
  }
  bodyCellOffsets[bodyCount] = bodyCellsTotal;

  const bodyCellIndices = new Int32Array(bodyCellsTotal);
  let bodyWrite = 0;
  for (let i = 0; i < bodyCount; i++) {
    const keys = bodyCellKeys[i] || [];
    for (let k = 0; k < keys.length; k++) {
      const ci = keyToIndex.get(keys[k]);
      bodyCellIndices[bodyWrite++] = Number.isFinite(Number(ci)) ? (Number(ci) | 0) : -1;
    }
  }

  return {
    bodyCount,
    cellOffsets,
    cellBodyIds,
    bodyCellOffsets,
    bodyCellIndices,
  };
}

function getCellMembershipState(state) {
  const bodyCount = Number(state?.bodyCount) | 0;
  const prepared = state?.wasmCandidateState;
  if (
    prepared
    && prepared.cellOffsets instanceof Int32Array
    && prepared.cellBodyIds instanceof Int32Array
    && prepared.bodyCellOffsets instanceof Int32Array
    && prepared.bodyCellIndices instanceof Int32Array
  ) {
    return {
      bodyCount,
      cellOffsets: prepared.cellOffsets,
      cellBodyIds: prepared.cellBodyIds,
      bodyCellOffsets: prepared.bodyCellOffsets,
      bodyCellIndices: prepared.bodyCellIndices,
      marshaledFrom: 'state-prepared',
    };
  }

  return {
    ...buildCellMembershipStateFromMaps(state),
    marshaledFrom: 'js-fallback-build',
  };
}

function buildCandidatesWithRuntime(runtime, state) {
  const totalStartMs = nowMs();
  const marshalStartMs = nowMs();
  const {
    bodyCount,
    cellOffsets,
    cellBodyIds,
    bodyCellOffsets,
    bodyCellIndices,
    marshaledFrom,
  } = getCellMembershipState(state);
  const marshalMs = nowMs() - marshalStartMs;

  if (bodyCount <= 1) {
    return {
      pairs: [],
      stats: {
        emitAttempts: 0,
        duplicatesRejected: 0,
        dedupeMs: 0,
        sortMs: 0,
        totalBuildMs: marshalMs,
        jsMarshalMs: marshalMs,
        backendComputeMs: 0,
        backendMarshalPath: marshaledFrom,
      },
    };
  }

  const maxPairs = Math.max(0, Number(state?.bruteForcePairs) | 0);
  const bytesBodyCellOffsets = bodyCellOffsets.byteLength;
  const bytesBodyCellIndices = bodyCellIndices.byteLength;
  const bytesCellOffsets = cellOffsets.byteLength;
  const bytesCellBodyIds = cellBodyIds.byteLength;
  const bytesSeen = bodyCount * 4;
  const bytesPairs = maxPairs * 2 * 4;
  const bytesStats = 3 * 4;
  const bytesTotal = bytesBodyCellOffsets + bytesBodyCellIndices + bytesCellOffsets + bytesCellBodyIds + bytesSeen + bytesPairs + bytesStats + 64;

  ensureMemoryCapacity(runtime, runtime.heapBase + bytesTotal);
  runtime.exports.rr_reset_heap();

  const bodyCellOffsetsPtr = Number(runtime.exports.rr_alloc(bytesBodyCellOffsets));
  const bodyCellIndicesPtr = Number(runtime.exports.rr_alloc(bytesBodyCellIndices));
  const cellOffsetsPtr = Number(runtime.exports.rr_alloc(bytesCellOffsets));
  const cellBodyIdsPtr = Number(runtime.exports.rr_alloc(bytesCellBodyIds));
  const seenPtr = Number(runtime.exports.rr_alloc(bytesSeen));
  const pairsPtr = Number(runtime.exports.rr_alloc(bytesPairs));
  const statsPtr = Number(runtime.exports.rr_alloc(bytesStats));

  const memI32 = new Int32Array(runtime.memory.buffer);
  memI32.set(bodyCellOffsets, bodyCellOffsetsPtr >> 2);
  memI32.set(bodyCellIndices, bodyCellIndicesPtr >> 2);
  memI32.set(cellOffsets, cellOffsetsPtr >> 2);
  memI32.set(cellBodyIds, cellBodyIdsPtr >> 2);

  const wasmStartMs = nowMs();
  const pairCount = Number(runtime.exports.rr_build_pairs_from_cells(
    bodyCount,
    bodyCellOffsetsPtr,
    bodyCellIndicesPtr,
    cellOffsetsPtr,
    cellBodyIdsPtr,
    seenPtr,
    pairsPtr,
    statsPtr,
  )) | 0;
  const backendComputeMs = nowMs() - wasmStartMs;

  const statsIdx = statsPtr >> 2;
  const emitAttempts = Number(memI32[statsIdx]) || 0;
  const duplicatesRejected = Number(memI32[statsIdx + 1]) || 0;

  const pairs = new Array(Math.max(0, pairCount));
  const pairBase = pairsPtr >> 2;
  for (let i = 0; i < pairCount; i++) {
    const o = pairBase + i * 2;
    pairs[i] = [Number(memI32[o]) | 0, Number(memI32[o + 1]) | 0];
  }

  const totalBuildMs = nowMs() - totalStartMs;
  return {
    pairs,
    stats: {
      emitAttempts,
      duplicatesRejected,
      dedupeMs: backendComputeMs,
      sortMs: 0,
      totalBuildMs,
      jsMarshalMs: marshalMs,
      backendComputeMs,
      backendMarshalPath: marshaledFrom,
    },
  };
}

export async function loadRigidRigidCandidateBackendWasm() {
  const runtime = await getRuntime();
  return {
    label: 'wasm-rr-v2',
    prepareWasmCandidateState: true,
    buildCandidates: (state) => buildCandidatesWithRuntime(runtime, state),
  };
}
