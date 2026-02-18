import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applySoftSpringsXPBDVelocityGpuOnly,
  buildSoftSpringXpbdWgslPlan,
  buildSoftSpringXpbdWgslLayout,
  reduceSoftSpringVelocityDeltasDeterministic,
} from '../../sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js';

const SOFT_XPBD_ITERS = 10;
const SOFT_XPBD_BASE_COMPLIANCE = 0.0012;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function createSoftSpringMockWgslDevice({ mapDelayMs = 0 } = {}) {
  const storage = new WeakMap();

  function ensure(buffer, size) {
    const current = storage.get(buffer);
    if (!current || current.byteLength < size) {
      storage.set(buffer, new ArrayBuffer(size));
    }
    return storage.get(buffer);
  }

  function readU32(buffer, count) {
    return new Uint32Array(ensure(buffer, count * 4).slice(0, count * 4));
  }

  function readF32(buffer, count) {
    return new Float32Array(ensure(buffer, count * 4).slice(0, count * 4));
  }

  function writeF32(buffer, values) {
    const data = ensure(buffer, values.byteLength);
    new Uint8Array(data).set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  }

  return {
    createShaderModule() { return {}; },
    async createComputePipelineAsync() {
      return {
        getBindGroupLayout() { return {}; },
      };
    },
    createBuffer({ size }) {
      const buf = {
        size,
        destroy() {},
        mapAsync: async () => {
          if (mapDelayMs > 0) {
            await new Promise((resolve) => setTimeout(resolve, mapDelayMs));
          }
        },
        getMappedRange: (offset = 0, len = size) => ensure(buf, size).slice(offset, offset + len),
        unmap() {},
      };
      ensure(buf, size);
      return buf;
    },
    createBindGroup({ entries }) {
      const map = new Map(entries.map((entry) => [entry.binding, entry.resource.buffer]));
      return { __buffers: map };
    },
    createCommandEncoder() {
      const ops = [];
      return {
        beginComputePass() {
          const pass = { bindGroup: null };
          return {
            setPipeline() {},
            setBindGroup(_index, bindGroup) { pass.bindGroup = bindGroup; },
            dispatchWorkgroups() {},
            end() { ops.push({ type: 'compute', pass }); },
          };
        },
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push({ type: 'copy', src, srcOffset, dst, dstOffset, size });
        },
        finish() { return ops; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset || 0, data.byteLength || data.length);
        const target = ensure(buffer, offset + bytes.byteLength);
        new Uint8Array(target).set(bytes, offset);
      },
      submit(commandsList) {
        for (const commands of commandsList) {
          for (const cmd of commands) {
            if (cmd.type === 'compute') {
              const buffers = cmd.pass.bindGroup.__buffers;
              const paramsBuf = ensure(buffers.get(0), 16);
              const paramsU32 = new Uint32Array(paramsBuf);
              const paramsF32 = new Float32Array(paramsBuf);
              const nodeCount = paramsU32[0] || 0;
              const springCount = paramsU32[1] || 0;

              if (buffers.has(12)) {
                const endpointCount = paramsU32[2] || 0;
                const dtPos = Math.max(1e-8, paramsF32[3] || 0);
                const nodePredX = readF32(buffers.get(1), nodeCount);
                const nodePredY = readF32(buffers.get(2), nodeCount);
                const springA = readU32(buffers.get(3), springCount);
                const springB = readU32(buffers.get(4), springCount);
                const invMassA = readF32(buffers.get(5), springCount);
                const invMassB = readF32(buffers.get(6), springCount);
                const endpointNodes = readU32(buffers.get(7), endpointCount);
                const endpointSprings = readU32(buffers.get(8), endpointCount);
                const endpointSigns = new Int32Array(ensure(buffers.get(9), endpointCount * 4).slice(0, endpointCount * 4));
                const deltaLambda = readF32(buffers.get(10), springCount);

                const outVX = new Float32Array(endpointCount);
                const outVY = new Float32Array(endpointCount);
                for (let ei = 0; ei < endpointCount; ei++) {
                  const ni = endpointNodes[ei];
                  const si = endpointSprings[ei];
                  if (ni >= nodeCount || si >= springCount || dtPos <= 1e-8) continue;
                  const ia = springA[si];
                  const ib = springB[si];
                  if (ia >= nodeCount || ib >= nodeCount) continue;

                  const dx = nodePredX[ib] - nodePredX[ia];
                  const dy = nodePredY[ib] - nodePredY[ia];
                  const d = Math.max(1e-6, Math.hypot(dx, dy));
                  const nx = dx / d;
                  const ny = dy / d;
                  const sign = endpointSigns[ei] < 0 ? -1 : 1;
                  const w = sign < 0 ? invMassA[si] : invMassB[si];
                  const scale = (sign * w * deltaLambda[si]) / dtPos;
                  outVX[ei] = nx * scale;
                  outVY[ei] = ny * scale;
                }
                writeF32(buffers.get(11), outVX);
                writeF32(buffers.get(12), outVY);
              } else if (buffers.has(6)) {
                const endpointCount = paramsU32[1] || 0;
                const endpointNodes = readU32(buffers.get(1), endpointCount);
                const endpointVX = readF32(buffers.get(2), endpointCount);
                const endpointVY = readF32(buffers.get(3), endpointCount);
                const outVX = new Float32Array(nodeCount);
                const outVY = new Float32Array(nodeCount);
                const outCount = new Uint32Array(nodeCount);
                for (let ni = 0; ni < nodeCount; ni++) {
                  let sumX = 0;
                  let sumY = 0;
                  let count = 0;
                  for (let ei = 0; ei < endpointCount; ei++) {
                    if (endpointNodes[ei] !== ni) continue;
                    sumX += endpointVX[ei] || 0;
                    sumY += endpointVY[ei] || 0;
                    count += 1;
                  }
                  outVX[ni] = sumX;
                  outVY[ni] = sumY;
                  outCount[ni] = count >>> 0;
                }
                writeF32(buffers.get(4), outVX);
                writeF32(buffers.get(5), outVY);
                const outCountBytes = ensure(buffers.get(6), outCount.byteLength);
                new Uint8Array(outCountBytes).set(new Uint8Array(outCount.buffer));
              } else if (buffers.has(10)) {
                const nodeX = readF32(buffers.get(1), nodeCount);
                const nodeY = readF32(buffers.get(2), nodeCount);
                const springA = readU32(buffers.get(3), springCount);
                const springB = readU32(buffers.get(4), springCount);
                const springRest = readF32(buffers.get(5), springCount);
                const invMassA = readF32(buffers.get(6), springCount);
                const invMassB = readF32(buffers.get(7), springCount);
                const lambdaPrev = readF32(buffers.get(8), springCount);
                const alpha = paramsF32[2] || 0;

                const deltaOut = new Float32Array(springCount);
                const nextOut = new Float32Array(springCount);

                for (let i = 0; i < springCount; i++) {
                  const ia = springA[i];
                  const ib = springB[i];
                  if (ia >= nodeCount || ib >= nodeCount) continue;

                  const dx = nodeX[ib] - nodeX[ia];
                  const dy = nodeY[ib] - nodeY[ia];
                  const d = Math.max(1e-6, Math.hypot(dx, dy));
                  const rest = springRest[i];
                  const strainCap = Math.max(0.05, Math.abs(rest) * 0.45);
                  const C = clamp(d - rest, -strainCap, strainCap);
                  const wSum = invMassA[i] + invMassB[i];
                  if (wSum <= 1e-9) {
                    nextOut[i] = lambdaPrev[i];
                    continue;
                  }
                  const prev = lambdaPrev[i];
                  const dlRaw = (-C - alpha * prev) / (wSum + alpha);
                  const next = Math.max(-20, Math.min(20, prev + dlRaw));
                  deltaOut[i] = next - prev;
                  nextOut[i] = next;
                }

                writeF32(buffers.get(9), deltaOut);
                writeF32(buffers.get(10), nextOut);
              } else {
                const nodeX = readF32(buffers.get(1), nodeCount);
                const nodeY = readF32(buffers.get(2), nodeCount);
                const springA = readU32(buffers.get(3), springCount);
                const springB = readU32(buffers.get(4), springCount);
                const springRest = readF32(buffers.get(5), springCount);
                const stretch = new Float32Array(springCount);
                for (let i = 0; i < springCount; i++) {
                  const ia = springA[i];
                  const ib = springB[i];
                  if (ia >= nodeCount || ib >= nodeCount) continue;
                  const dx = nodeX[ib] - nodeX[ia];
                  const dy = nodeY[ib] - nodeY[ia];
                  const d = Math.max(1e-6, Math.hypot(dx, dy));
                  const rest = Math.max(Math.abs(springRest[i]), 1e-6);
                  stretch[i] = (d - rest) / rest;
                }
                writeF32(buffers.get(6), stretch);
              }
            } else if (cmd.type === 'copy') {
              const src = ensure(cmd.src, cmd.srcOffset + cmd.size);
              const dst = ensure(cmd.dst, cmd.dstOffset + cmd.size);
              new Uint8Array(dst).set(new Uint8Array(src, cmd.srcOffset, cmd.size), cmd.dstOffset);
            }
          }
        }
      },
    },
  };
}

function applySoftSpringsXPBDVelocityBaseline(soft, dtPos, stiffnessScale, lambdaCache, { skipClusterSet = null } = {}) {
  if (!soft?.nodes?.length || !soft?.springs?.length) return;
  const alpha = (SOFT_XPBD_BASE_COMPLIANCE / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < SOFT_XPBD_ITERS; iter++) {
    for (let si = 0; si < soft.springs.length; si++) {
      const [i, j, rest] = soft.springs[si];
      const a = soft.nodes[i];
      const b = soft.nodes[j];
      if (!a || !b) continue;
      if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

      const ax = a.x + a.vx * dtPos;
      const ay = a.y + a.vy * dtPos;
      const bx = b.x + b.vx * dtPos;
      const by = b.y + b.vy * dtPos;

      const dx = bx - ax;
      const dy = by - ay;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = dx / d;
      const ny = dy / d;
      const strainCap = Math.max(0.05, Math.abs(rest) * 0.45);
      const C = clamp(d - rest, -strainCap, strainCap);

      const wA = 1 / Math.max(0.02, a.mass || 1);
      const wB = 1 / Math.max(0.02, b.mass || 1);
      const wSum = wA + wB;
      if (wSum <= 1e-9) continue;

      const lambdaPrev = Number(lambdaCache[si]) || 0;
      let dl = (-C - alpha * lambdaPrev) / (wSum + alpha);
      if (!Number.isFinite(dl)) continue;
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      lambdaCache[si] = lambdaNext;

      const corrAx = -wA * dl * nx;
      const corrAy = -wA * dl * ny;
      const corrBx = wB * dl * nx;
      const corrBy = wB * dl * ny;

      a.vx += corrAx / dtPos;
      a.vy += corrAy / dtPos;
      b.vx += corrBx / dtPos;
      b.vy += corrBy / dtPos;
    }
  }
}

test('soft spring XPBD parity: baseline stepping and gpu-only module produce matching soft states', () => {
  const dtPos = 0.18;
  const stiffnessScale = 3.0;
  const softSeed = {
    nodes: [
      { x: 20, y: 25, vx: 0.2, vy: -0.1, mass: 1.1, clusterId: 1 },
      { x: 30, y: 21, vx: -0.3, vy: 0.4, mass: 0.8, clusterId: 1 },
      { x: 39, y: 28, vx: 0.5, vy: 0.2, mass: 1.4, clusterId: 2 },
      { x: 48, y: 20, vx: -0.4, vy: -0.35, mass: 1.0, clusterId: 3 },
    ],
    springs: [
      [0, 1, 11.2],
      [1, 2, 10.8],
      [2, 3, 12.1],
      [0, 2, 19.4],
    ],
  };
  const skipClusterSet = new Set([3]);

  const baseline = structuredClone(softSeed);
  const gpuOnly = structuredClone(softSeed);
  const baselineLambda = new Float32Array(softSeed.springs.length);
  const gpuLambda = new Float32Array(softSeed.springs.length);

  applySoftSpringsXPBDVelocityBaseline(baseline, dtPos, stiffnessScale, baselineLambda, { skipClusterSet });
  applySoftSpringsXPBDVelocityGpuOnly({
    soft: gpuOnly,
    dtPos,
    stiffnessScale,
    lambdaCache: gpuLambda,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    skipClusterSet,
  });

  assert.deepEqual(Array.from(gpuLambda), Array.from(baselineLambda), 'lambda cache should match baseline stepping exactly');
  assert.equal(gpuOnly.nodes.length, baseline.nodes.length);
  for (let i = 0; i < baseline.nodes.length; i++) {
    const b = baseline.nodes[i];
    const g = gpuOnly.nodes[i];
    assert.ok(Math.abs(g.vx - b.vx) < 1e-12, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-12, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    assert.ok(Math.abs(g.x - b.x) < 1e-12, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-12, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
  }
});


test('soft spring XPBD WGSL plan builder emits deterministic CSR endpoint ownership for active springs', () => {
  const soft = {
    nodes: [
      { clusterId: 0 },
      { clusterId: 1 },
      { clusterId: 2 },
      { clusterId: 3 },
    ],
    springs: [
      [0, 1, 10],
      [1, 2, 9],
      [2, 3, 8],
      [0, 99, 7],
      [3, 0, Number.NaN],
    ],
  };

  const plan = buildSoftSpringXpbdWgslPlan({
    soft,
    skipClusterSet: new Set([3]),
  });

  assert.equal(plan.nodeCount, 4);
  assert.equal(plan.springCount, 5);
  assert.equal(plan.activeSpringCount, 2, 'should include only valid, non-skipped springs');
  assert.equal(plan.endpointCount, 4, 'each active spring should contribute two endpoints');
  assert.deepEqual(Array.from(plan.activeSpringIndices), [0, 1]);

  assert.deepEqual(
    Array.from(plan.nodeEndpointOffsets),
    [0, 1, 3, 4, 4],
    'CSR offsets should group endpoint ownership by node index',
  );
  assert.deepEqual(
    Array.from(plan.endpointNodeIndices),
    [0, 1, 1, 2],
    'endpoints should be stably grouped by node index',
  );
  assert.deepEqual(
    Array.from(plan.endpointSpringIndices),
    [0, 0, 1, 1],
    'endpoint spring indices should map grouped endpoints back to active-spring rows',
  );
  assert.deepEqual(
    Array.from(plan.endpointSigns),
    [-1, 1, -1, 1],
    'endpoint signs should preserve i/j ownership direction for gather-reduce kernels',
  );
});



test('soft spring XPBD WGSL plan builder emits deterministic conflict-free spring color batches', () => {
  const soft = {
    nodes: [{}, {}, {}, {}],
    springs: [
      [0, 1, 1],
      [1, 2, 1],
      [2, 3, 1],
      [0, 3, 1],
      [0, 2, 1],
    ],
  };

  const plan = buildSoftSpringXpbdWgslPlan({ soft });

  assert.deepEqual(Array.from(plan.springColors), [0, 1, 0, 1, 2]);
  assert.deepEqual(Array.from(plan.springColorOffsets), [0, 2, 4, 5]);
  assert.deepEqual(Array.from(plan.springColorOrderedIndices), [0, 2, 1, 3, 4]);

  for (let ci = 0; ci < plan.springColorOffsets.length - 1; ci++) {
    const usedNodes = new Set();
    const start = plan.springColorOffsets[ci];
    const end = plan.springColorOffsets[ci + 1];
    for (let oi = start; oi < end; oi++) {
      const ai = plan.springColorOrderedIndices[oi];
      const si = plan.activeSpringIndices[ai];
      const [a, b] = soft.springs[si];
      assert.equal(usedNodes.has(a), false, `color ${ci} reuses node ${a}`);
      assert.equal(usedNodes.has(b), false, `color ${ci} reuses node ${b}`);
      usedNodes.add(a);
      usedNodes.add(b);
    }
  }
});

test('soft spring XPBD WGSL layout builder emits deterministic spring SoA + endpoint sign buffers', () => {
  const soft = {
    nodes: [
      { mass: 1.25, clusterId: 0 },
      { mass: 0.5, clusterId: 1 },
      { mass: 2.0, clusterId: 2 },
      { mass: 1.0, clusterId: 3 },
    ],
    springs: [
      [0, 1, 4.5],
      [1, 2, 3.25],
      [2, 3, 5.75],
    ],
  };

  const plan = buildSoftSpringXpbdWgslPlan({
    soft,
    skipClusterSet: new Set([3]),
  });
  const layout = buildSoftSpringXpbdWgslLayout({ soft, plan });

  assert.deepEqual(Array.from(layout.springNodeA), [0, 1]);
  assert.deepEqual(Array.from(layout.springNodeB), [1, 2]);
  assert.deepEqual(Array.from(layout.springRest), [4.5, 3.25]);
  assert.deepEqual(
    Array.from(layout.springInvMassA).map((v) => Number(v.toFixed(6))),
    [0.8, 2],
  );
  assert.deepEqual(
    Array.from(layout.springInvMassB).map((v) => Number(v.toFixed(6))),
    [2, 0.5],
  );
  assert.deepEqual(Array.from(layout.endpointSignsI32), [-1, 1, -1, 1]);
  assert.deepEqual(Array.from(layout.springColorOffsets), [0, 1, 2]);
  assert.deepEqual(Array.from(layout.springColorOrderedIndices), [0, 1]);
  assert.deepEqual(Array.from(layout.springNodeAByColor), [0, 1]);
  assert.deepEqual(Array.from(layout.springNodeBByColor), [1, 2]);
  assert.deepEqual(Array.from(layout.colorEndpointOffsets), [0, 2, 4]);
  assert.deepEqual(Array.from(layout.endpointNodeIndicesByColor), [0, 1, 1, 2]);
  assert.deepEqual(Array.from(layout.endpointSpringIndicesByColor), [0, 0, 1, 1]);
  assert.deepEqual(Array.from(layout.endpointSignsI32ByColor), [-1, 1, -1, 1]);
  assert.equal(layout.byteLength > 0, true);
});

test('soft spring XPBD stores WGSL-prep state while preserving cpu parity outputs', () => {
  const dtPos = 0.18;
  const stiffnessScale = 3.0;
  const seed = {
    nodes: [
      { x: 20, y: 25, vx: 0.2, vy: -0.1, mass: 1.1, clusterId: 1 },
      { x: 30, y: 21, vx: -0.3, vy: 0.4, mass: 0.8, clusterId: 1 },
      { x: 39, y: 28, vx: 0.5, vy: 0.2, mass: 1.4, clusterId: 2 },
    ],
    springs: [
      [0, 1, 11.2],
      [1, 2, 10.8],
    ],
  };

  const softA = structuredClone(seed);
  const softB = structuredClone(seed);
  const lambdaA = new Float32Array(seed.springs.length);
  const lambdaB = new Float32Array(seed.springs.length);

  applySoftSpringsXPBDVelocityGpuOnly({
    soft: softA,
    dtPos,
    stiffnessScale,
    lambdaCache: lambdaA,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
  });

  const wgslState = {};
  applySoftSpringsXPBDVelocityGpuOnly({
    soft: softB,
    dtPos,
    stiffnessScale,
    lambdaCache: lambdaB,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: {
      enabled: true,
      modeProfile: 'gpu-only-validated',
      state: wgslState,
    },
  });

  assert.deepEqual(Array.from(lambdaB), Array.from(lambdaA), 'wgsl prep mode should preserve cpu lambda outputs');
  assert.deepEqual(softB.nodes, softA.nodes, 'wgsl prep mode should preserve cpu node outputs');
  assert.equal(wgslState.lastMode, 'cpu-prepared');
  assert.equal(wgslState.lastPreparedSpringCount, 2);
  assert.equal(wgslState.lastPreparedEndpointCount, 4);
  assert.equal(wgslState.lastPreparedColorCount, 2);
  assert.equal(wgslState.lastPreparedColorEndpointCount, 4);
  assert.equal(wgslState.preparedPlan?.nodeEndpointOffsets?.length, seed.nodes.length + 1);
  assert.equal(wgslState.preparedLayout?.springNodeA?.length, 2);
  assert.equal(wgslState.preparedLayout?.endpointSignsI32?.length, 4);
  assert.equal(wgslState.preparedPlan?.springColorOffsets?.length, 3);
  assert.equal(wgslState.preparedLayout?.springNodeAByColor?.length, 2);
  assert.equal(wgslState.preparedLayout?.colorEndpointOffsets?.length, 3);
  assert.equal(wgslState.preparedLayout?.endpointSpringIndicesByColor?.length, 4);
  assert.equal(wgslState.lastPreparedLayoutBytes, wgslState.preparedLayout?.byteLength);
});

test('soft spring XPBD WGSL proposal stage runs on gpu-only path while CPU remains authoritative', async () => {
  globalThis.GPUBufferUsage = {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    COPY_SRC: 1 << 2,
    MAP_READ: 1 << 3,
    UNIFORM: 1 << 4,
  };
  globalThis.GPUMapMode = { READ: 1 };

  const dtPos = 0.18;
  const stiffnessScale = 3.0;
  const seed = {
    nodes: [
      { x: 20, y: 25, vx: 0.2, vy: -0.1, mass: 1.1, clusterId: 1 },
      { x: 30, y: 21, vx: -0.3, vy: 0.4, mass: 0.8, clusterId: 1 },
      { x: 39, y: 28, vx: 0.5, vy: 0.2, mass: 1.4, clusterId: 2 },
    ],
    springs: [
      [0, 1, 11.2],
      [1, 2, 10.8],
    ],
  };

  const baseline = structuredClone(seed);
  const gpuOnly = structuredClone(seed);
  const baselineLambda = new Float32Array(seed.springs.length);
  const gpuLambda = new Float32Array(seed.springs.length);
  const wgslState = {};

  applySoftSpringsXPBDVelocityBaseline(baseline, dtPos, stiffnessScale, baselineLambda);
  applySoftSpringsXPBDVelocityGpuOnly({
    soft: gpuOnly,
    dtPos,
    stiffnessScale,
    lambdaCache: gpuLambda,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: {
      enabled: true,
      modeProfile: 'gpu-only-validated',
      device: createSoftSpringMockWgslDevice(),
      state: wgslState,
    },
  });

  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(Array.from(gpuLambda), Array.from(baselineLambda));
  assert.deepEqual(gpuOnly.nodes, baseline.nodes);
  assert.equal(wgslState.lastMode, 'wgsl-velocity-proposal-validated');
  assert.equal(wgslState.lastError, null);
  assert.equal(wgslState.lastProposalSpringCount, 2);
  assert.equal(wgslState.lastProposalDeltaLambdaByColor instanceof Float32Array, true);
  assert.equal(wgslState.lastProposalLambdaNextByColor instanceof Float32Array, true);
  assert.equal(wgslState.lastProposalDeltaLambdaBySpring instanceof Float32Array, true);
  assert.equal(wgslState.lastProposalLambdaNextBySpring instanceof Float32Array, true);
  assert.equal(wgslState.lastProposalDeltaLambdaByColor.length, 2);
  assert.equal(wgslState.lastProposalLambdaNextByColor.length, 2);
  assert.equal(wgslState.lastProposalDeltaLambdaBySpring.length, 2);
  assert.equal(wgslState.lastProposalLambdaNextBySpring.length, 2);
  assert.equal(wgslState.lastVelocityDeltaProposalEndpointVxByColor instanceof Float32Array, true);
  assert.equal(wgslState.lastVelocityDeltaProposalEndpointVyByColor instanceof Float32Array, true);
  assert.equal(wgslState.lastVelocityDeltaProposalNodeVxByColor instanceof Float32Array, true);
  assert.equal(wgslState.lastVelocityDeltaProposalNodeVyByColor instanceof Float32Array, true);
  assert.equal(wgslState.lastVelocityDeltaProposalNodeContributionCount instanceof Uint32Array, true);
  assert.equal(wgslState.lastVelocityDeltaExpectedNodeVxByColor instanceof Float32Array, true);
  assert.equal(wgslState.lastVelocityDeltaExpectedNodeVyByColor instanceof Float32Array, true);
  assert.equal(wgslState.lastVelocityDeltaExpectedNodeContributionCount instanceof Uint32Array, true);
  assert.equal(wgslState.lastVelocityDeltaProposalSource, 'wgsl-node-reduction');
  assert.equal(Number.isInteger(wgslState.lastVelocityDeltaReductionDispatch), true);
  assert.equal(wgslState.lastVelocityDeltaReductionDispatch > 0, true);
  assert.equal(wgslState.lastContributionCountParity?.comparedCount, seed.nodes.length);
  assert.equal(Number.isInteger(wgslState.lastContributionCountParity?.mismatchCount), true);
  assert.equal(wgslState.lastVelocityDeltaParity?.source, 'wgsl-node-reduction');
  assert.equal(wgslState.lastVelocityDeltaParity?.comparedNodeCount, seed.nodes.length);
  assert.equal(Number.isInteger(wgslState.lastVelocityDeltaParity?.contributionCount?.mismatchCount), true);
  assert.ok((wgslState.lastVelocityDeltaParity?.maxAbs ?? 1) < 1e-6);
  assert.ok((wgslState.lastVelocityDeltaParity?.meanAbs ?? 1) < 1e-6);

  const invOrder = wgslState.preparedPlan.springColorOrderedIndices;
  const remappedDelta = new Float32Array(invOrder.length);
  const remappedNext = new Float32Array(invOrder.length);
  for (let oi = 0; oi < invOrder.length; oi++) {
    remappedDelta[invOrder[oi]] = wgslState.lastProposalDeltaLambdaByColor[oi];
    remappedNext[invOrder[oi]] = wgslState.lastProposalLambdaNextByColor[oi];
  }
  assert.deepEqual(
    Array.from(wgslState.lastProposalDeltaLambdaBySpring),
    Array.from(remappedDelta),
    'expected proposal telemetry remapped back to original spring ownership order',
  );
  assert.deepEqual(
    Array.from(wgslState.lastProposalLambdaNextBySpring),
    Array.from(remappedNext),
    'expected lambda-next telemetry remapped back to original spring ownership order',
  );
  assert.equal(wgslState.lastProbeSpringCount, 2);
});

test('soft spring XPBD WGSL authoritative replay applies cached node/lambda proposal when signature matches', async () => {
  globalThis.GPUBufferUsage = {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    COPY_SRC: 1 << 2,
    MAP_READ: 1 << 3,
    UNIFORM: 1 << 4,
  };
  globalThis.GPUMapMode = { READ: 1 };

  const dtPos = 0.18;
  const stiffnessScale = 3.0;
  const seed = {
    nodes: [
      { x: 20, y: 25, vx: 0.2, vy: -0.1, mass: 1.1, clusterId: 1 },
      { x: 30, y: 21, vx: -0.3, vy: 0.4, mass: 0.8, clusterId: 1 },
      { x: 39, y: 28, vx: 0.5, vy: 0.2, mass: 1.4, clusterId: 2 },
    ],
    springs: [
      [0, 1, 11.2],
      [1, 2, 10.8],
    ],
  };

  const wgslState = {};
  const offload = {
    enabled: true,
    modeProfile: 'gpu-only-validated',
    device: createSoftSpringMockWgslDevice(),
    state: wgslState,
  };

  applySoftSpringsXPBDVelocityGpuOnly({
    soft: structuredClone(seed),
    dtPos,
    stiffnessScale,
    lambdaCache: new Float32Array(seed.springs.length),
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: offload,
  });

  await new Promise((resolve) => setTimeout(resolve, 0));

  const replaySoft = structuredClone(seed);
  const replayLambda = new Float32Array(seed.springs.length);
  wgslState.enableAuthoritativeVelocityDelta = true;

  applySoftSpringsXPBDVelocityGpuOnly({
    soft: replaySoft,
    dtPos,
    stiffnessScale,
    lambdaCache: replayLambda,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: offload,
  });

  assert.equal(replayLambda.some((v) => Math.abs(v || 0) > 1e-8), true);
  assert.equal(replaySoft.nodes.some((n) => Math.abs(n?.vx || 0) > 1e-8 || Math.abs(n?.vy || 0) > 1e-8), true);
  assert.equal(wgslState.lastMode, 'wgsl-velocity-authoritative-validated');
  assert.equal(wgslState.lastAuthoritativeProposalSource, 'wgsl-node-reduction');
  assert.equal(wgslState.lastContributionCountParity?.comparedCount, seed.nodes.length);
  assert.equal(typeof wgslState.lastAuthoritativeProposalSignature, 'string');
});


test('soft spring XPBD gpu-only fast authoritative replay skips residual cpu XPBD iterations after finite-checked WGSL proposal', async () => {
  globalThis.GPUBufferUsage = {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    COPY_SRC: 1 << 2,
    MAP_READ: 1 << 3,
    UNIFORM: 1 << 4,
  };
  globalThis.GPUMapMode = { READ: 1 };

  const dtPos = 0.18;
  const stiffnessScale = 3.0;
  const seed = {
    nodes: [
      { x: 20, y: 25, vx: 0.2, vy: -0.1, mass: 1.1, clusterId: 1 },
      { x: 30, y: 21, vx: -0.3, vy: 0.4, mass: 0.8, clusterId: 1 },
      { x: 39, y: 28, vx: 0.5, vy: 0.2, mass: 1.4, clusterId: 2 },
    ],
    springs: [
      [0, 1, 11.2],
      [1, 2, 10.8],
    ],
  };

  const wgslState = {};
  const offload = {
    enabled: true,
    modeProfile: 'gpu-only-fast',
    device: createSoftSpringMockWgslDevice(),
    state: wgslState,
  };

  applySoftSpringsXPBDVelocityGpuOnly({
    soft: structuredClone(seed),
    dtPos,
    stiffnessScale,
    lambdaCache: new Float32Array(seed.springs.length),
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: offload,
  });

  await new Promise((resolve) => setTimeout(resolve, 0));

  const replaySoft = structuredClone(seed);
  const replayLambda = new Float32Array(seed.springs.length);
  const expectedNodeDeltaVx = new Float32Array(wgslState.lastVelocityDeltaProposalNodeVxByColor || []);
  const expectedNodeDeltaVy = new Float32Array(wgslState.lastVelocityDeltaProposalNodeVyByColor || []);
  const expectedLambda = new Float32Array(wgslState.lastProposalLambdaNextBySpring || []);
  wgslState.enableAuthoritativeVelocityDelta = true;

  applySoftSpringsXPBDVelocityGpuOnly({
    soft: replaySoft,
    dtPos,
    stiffnessScale,
    lambdaCache: replayLambda,
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: offload,
  });

  assert.equal(wgslState.lastMode, 'wgsl-velocity-authoritative-fast');
  assert.equal(wgslState.lastAuthoritativeProposalSource, 'wgsl-node-reduction-fast');
  assert.equal(wgslState.lastAuthoritativeCpuIterStart, SOFT_XPBD_ITERS);
  assert.equal(wgslState.lastAuthoritativeResidualCpuIters, 0);

  for (let i = 0; i < replaySoft.nodes.length; i++) {
    assert.ok(Math.abs(replaySoft.nodes[i].vx - (seed.nodes[i].vx + (expectedNodeDeltaVx[i] || 0))) < 1e-9);
    assert.ok(Math.abs(replaySoft.nodes[i].vy - (seed.nodes[i].vy + (expectedNodeDeltaVy[i] || 0))) < 1e-9);
  }
  assert.deepEqual(Array.from(replayLambda), Array.from(expectedLambda));
});


test('soft spring XPBD WGSL proposal skips overlapping dispatch while prior readback is in flight', async () => {
  globalThis.GPUBufferUsage = {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    COPY_SRC: 1 << 2,
    MAP_READ: 1 << 3,
    UNIFORM: 1 << 4,
  };
  globalThis.GPUMapMode = { READ: 1 };

  const dtPos = 0.18;
  const stiffnessScale = 3.0;
  const seed = {
    nodes: [
      { x: 20, y: 25, vx: 0.2, vy: -0.1, mass: 1.1, clusterId: 1 },
      { x: 30, y: 21, vx: -0.3, vy: 0.4, mass: 0.8, clusterId: 1 },
      { x: 39, y: 28, vx: 0.5, vy: 0.2, mass: 1.4, clusterId: 2 },
    ],
    springs: [
      [0, 1, 11.2],
      [1, 2, 10.8],
    ],
  };

  const wgslState = {};
  const sharedOffload = {
    enabled: true,
    modeProfile: 'gpu-only-validated',
    device: createSoftSpringMockWgslDevice({ mapDelayMs: 5 }),
    state: wgslState,
  };

  applySoftSpringsXPBDVelocityGpuOnly({
    soft: structuredClone(seed),
    dtPos,
    stiffnessScale,
    lambdaCache: new Float32Array(seed.springs.length),
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: sharedOffload,
  });

  applySoftSpringsXPBDVelocityGpuOnly({
    soft: structuredClone(seed),
    dtPos,
    stiffnessScale,
    lambdaCache: new Float32Array(seed.springs.length),
    softXpbdIters: SOFT_XPBD_ITERS,
    softXpbdBaseCompliance: SOFT_XPBD_BASE_COMPLIANCE,
    clamp,
    wgslOffload: sharedOffload,
  });

  await new Promise((resolve) => setTimeout(resolve, 80));

  assert.equal(wgslState.wgslSkippedWhileBusy, 1);
  assert.equal(wgslState.wgslInFlight, false);
  assert.equal(wgslState.lastCompletedWgslRunId, 1);
  assert.equal(wgslState.lastMode, 'wgsl-velocity-proposal-validated');
  assert.equal(wgslState.lastError, null);
});


test('reduceSoftSpringVelocityDeltasDeterministic deterministic endpoint ownership reduction matches spring correction math', () => {
  const dtPos = 0.2;
  const soft = {
    nodes: [
      { x: 0, y: 0, vx: 0.1, vy: 0.0, mass: 2 },
      { x: 1, y: 0, vx: -0.05, vy: 0.0, mass: 4 },
    ],
    springs: [[0, 1, 1]],
  };
  const plan = buildSoftSpringXpbdWgslPlan({ soft });
  const layout = buildSoftSpringXpbdWgslLayout({ soft, plan });
  const deltaLambdaByColor = new Float32Array([0.3]);

  const reduced = reduceSoftSpringVelocityDeltasDeterministic({
    soft,
    dtPos,
    layout,
    deltaLambdaByColor,
  });

  assert.equal(reduced.deltaVxByNode.length, 2);
  assert.equal(reduced.deltaVyByNode.length, 2);
  const wA = 1 / 2;
  const wB = 1 / 4;
  const nx = 1;
  const expectedA = (-wA * deltaLambdaByColor[0] * nx) / dtPos;
  const expectedB = (wB * deltaLambdaByColor[0] * nx) / dtPos;
  assert.ok(Math.abs(reduced.deltaVxByNode[0] - expectedA) < 1e-6);
  assert.ok(Math.abs(reduced.deltaVxByNode[1] - expectedB) < 1e-6);
  assert.ok(Math.abs(reduced.deltaVyByNode[0]) < 1e-6);
  assert.ok(Math.abs(reduced.deltaVyByNode[1]) < 1e-6);
  assert.deepEqual(Array.from(reduced.contributionCountByNode), [1, 1]);
});
