import test from 'node:test';
import assert from 'node:assert/strict';
import { applySoftAreaXPBDVelocityGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftAreaXpbdGpuOnly.js';

const SOFT_AREA_XPBD_ITERS = 6;
const SOFT_AREA_BASE_COMPLIANCE = 0.0009;

function signedAreaPredicted(nodes, indices, dtPos) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    const ax = a.x + a.vx * dtPos;
    const ay = a.y + a.vy * dtPos;
    const bx = b.x + b.vx * dtPos;
    const by = b.y + b.vy * dtPos;
    s += ax * by - bx * ay;
  }
  return 0.5 * s;
}

function applySoftAreaXPBDVelocityBaseline(sim, soft, loops, dtPos, stiffnessScale) {
  if (!loops.length) return;
  const alpha = (SOFT_AREA_BASE_COMPLIANCE / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < SOFT_AREA_XPBD_ITERS; iter++) {
    for (const loop of loops) {
      const ids = loop.indices;
      const m = ids.length;
      if (m < 3) continue;
      const restArea = sim.softAreaRest.get(loop.clusterId);
      if (!Number.isFinite(restArea)) continue;

      const area = signedAreaPredicted(soft.nodes, ids, dtPos);
      const C = area - restArea;

      const gradX = new Array(m);
      const gradY = new Array(m);
      let sumWGrad2 = 0;

      for (let k = 0; k < m; k++) {
        const prev = soft.nodes[ids[(k - 1 + m) % m]];
        const next = soft.nodes[ids[(k + 1) % m]];
        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nx = next.x + next.vx * dtPos;
        const ny = next.y + next.vy * dtPos;
        const gx = 0.5 * (ny - py);
        const gy = 0.5 * (px - nx);
        gradX[k] = gx;
        gradY[k] = gy;

        const node = soft.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        sumWGrad2 += w * (gx * gx + gy * gy);
      }

      if (sumWGrad2 <= 1e-10) continue;

      const lambdaPrev = Number(sim.softAreaLambda.get(loop.clusterId)) || 0;
      let dl = (-C - alpha * lambdaPrev) / (sumWGrad2 + alpha);
      if (!Number.isFinite(dl)) continue;
      dl = Math.max(-2.0, Math.min(2.0, dl));
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      sim.softAreaLambda.set(loop.clusterId, lambdaNext);

      for (let k = 0; k < m; k++) {
        const node = soft.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        node.vx += (w * gradX[k] * dl) / dtPos;
        node.vy += (w * gradY[k] * dl) / dtPos;
      }
    }
  }
}

function createSoftAreaMockWgslDevice() {
  const storage = new WeakMap();

  function ensure(buffer, size) {
    const current = storage.get(buffer);
    if (!current || current.byteLength < size) storage.set(buffer, new ArrayBuffer(size));
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
      return { getBindGroupLayout() { return {}; } };
    },
    createBuffer({ size }) {
      const buf = {
        size,
        destroy() {},
        mapAsync: async () => {},
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
              const clusterCount = paramsU32[1] || 0;
              const dtPos = paramsF32[2] || 0;

              const nodeX = readF32(buffers.get(1), nodeCount);
              const nodeY = readF32(buffers.get(2), nodeCount);
              const nodeVX = readF32(buffers.get(3), nodeCount);
              const nodeVY = readF32(buffers.get(4), nodeCount);
              const offsets = readU32(buffers.get(5), clusterCount + 1);
              const endpointCount = offsets[clusterCount] || 0;
              const nodeIdx = readU32(buffers.get(6), endpointCount);
              const endpointCountParam = paramsU32[2] || 0;

              if (endpointCountParam > 0 && buffers.has(11) && buffers.has(10) && buffers.has(9) && buffers.has(8)) {
                const invMass = readF32(buffers.get(7), endpointCount);
                const endpointCluster = readU32(buffers.get(8), endpointCount);
                const deltaLambda = readF32(buffers.get(9), clusterCount);
                const outVX = new Float32Array(endpointCount);
                const outVY = new Float32Array(endpointCount);
                const invDt = dtPos > 1e-8 ? (1 / dtPos) : 0;

                for (let ei = 0; ei < endpointCount; ei++) {
                  const ci = endpointCluster[ei];
                  if (ci >= clusterCount || invDt === 0) continue;
                  const start = offsets[ci];
                  const end = offsets[ci + 1];
                  if (end <= start + 1 || ei < start || ei >= end) continue;
                  const dl = deltaLambda[ci] || 0;
                  if (!dl) continue;
                  const pi = nodeIdx[ei > start ? (ei - 1) : (end - 1)];
                  const ni2 = nodeIdx[(ei + 1) < end ? (ei + 1) : start];
                  if (pi >= nodeCount || ni2 >= nodeCount) continue;
                  const px = nodeX[pi] + nodeVX[pi] * dtPos;
                  const py = nodeY[pi] + nodeVY[pi] * dtPos;
                  const nx = nodeX[ni2] + nodeVX[ni2] * dtPos;
                  const ny = nodeY[ni2] + nodeVY[ni2] * dtPos;
                  const gx = 0.5 * (ny - py);
                  const gy = 0.5 * (px - nx);
                  outVX[ei] = invMass[ei] * gx * dl * invDt;
                  outVY[ei] = invMass[ei] * gy * dl * invDt;
                }

                writeF32(buffers.get(10), outVX);
                writeF32(buffers.get(11), outVY);
              } else if (buffers.has(11)) {
                const invMass = readF32(buffers.get(7), endpointCount);
                const restArea = readF32(buffers.get(8), clusterCount);
                const lambdaPrev = readF32(buffers.get(9), clusterCount);
                const alpha = paramsF32[3] || 0;
                const deltaOut = new Float32Array(clusterCount);
                const nextOut = new Float32Array(clusterCount);

                for (let ci = 0; ci < clusterCount; ci++) {
                  const start = offsets[ci];
                  const end = offsets[ci + 1];
                  if (end <= start + 1) {
                    nextOut[ci] = lambdaPrev[ci];
                    continue;
                  }
                  let twiceArea = 0;
                  let sumWGrad2 = 0;
                  for (let ei = start; ei < end; ei++) {
                    const ni = nodeIdx[ei];
                    const nj = nodeIdx[(ei + 1) < end ? (ei + 1) : start];
                    if (ni >= nodeCount || nj >= nodeCount) continue;
                    const ax = nodeX[ni] + nodeVX[ni] * dtPos;
                    const ay = nodeY[ni] + nodeVY[ni] * dtPos;
                    const bx = nodeX[nj] + nodeVX[nj] * dtPos;
                    const by = nodeY[nj] + nodeVY[nj] * dtPos;
                    twiceArea += ax * by - bx * ay;

                    const pi = nodeIdx[ei > start ? (ei - 1) : (end - 1)];
                    if (pi >= nodeCount) continue;
                    const px = nodeX[pi] + nodeVX[pi] * dtPos;
                    const py = nodeY[pi] + nodeVY[pi] * dtPos;
                    const nx = bx;
                    const ny = by;
                    const gx = 0.5 * (ny - py);
                    const gy = 0.5 * (px - nx);
                    sumWGrad2 += invMass[ei] * (gx * gx + gy * gy);
                  }

                  if (sumWGrad2 <= 1e-10) {
                    nextOut[ci] = lambdaPrev[ci];
                    continue;
                  }

                  const area = 0.5 * twiceArea;
                  const C = area - restArea[ci];
                  const prev = lambdaPrev[ci];
                  let dl = (-C - alpha * prev) / (sumWGrad2 + alpha);
                  dl = Math.max(-2.0, Math.min(2.0, dl));
                  const next = Math.max(-20, Math.min(20, prev + dl));
                  deltaOut[ci] = next - prev;
                  nextOut[ci] = next;
                }

                writeF32(buffers.get(10), deltaOut);
                writeF32(buffers.get(11), nextOut);
              } else {
                const out = new Float32Array(clusterCount);
                for (let ci = 0; ci < clusterCount; ci++) {
                  const start = offsets[ci];
                  const end = offsets[ci + 1];
                  if (end <= start + 1) continue;
                  let twiceArea = 0;
                  for (let ei = start; ei < end; ei++) {
                    const ni = nodeIdx[ei];
                    const nj = nodeIdx[(ei + 1) < end ? (ei + 1) : start];
                    if (ni >= nodeCount || nj >= nodeCount) continue;
                    const ax = nodeX[ni] + nodeVX[ni] * dtPos;
                    const ay = nodeY[ni] + nodeVY[ni] * dtPos;
                    const bx = nodeX[nj] + nodeVX[nj] * dtPos;
                    const by = nodeY[nj] + nodeVY[nj] * dtPos;
                    twiceArea += ax * by - bx * ay;
                  }
                  out[ci] = 0.5 * twiceArea;
                }
                writeF32(buffers.get(7), out);
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

test('soft area XPBD parity: baseline stepping and gpu-only module produce matching soft states', () => {
  const dtPos = 0.16;
  const stiffnessScale = 3.4;
  const softSeed = {
    nodes: [
      { x: 14, y: 18, vx: 0.3, vy: -0.2, mass: 1.0, clusterId: 7 },
      { x: 22, y: 17, vx: -0.2, vy: 0.1, mass: 0.9, clusterId: 7 },
      { x: 26, y: 24, vx: 0.4, vy: 0.3, mass: 1.2, clusterId: 7 },
      { x: 19, y: 29, vx: -0.3, vy: -0.1, mass: 1.4, clusterId: 7 },
      { x: 34, y: 31, vx: 0.1, vy: -0.25, mass: 0.95, clusterId: 11 },
      { x: 41, y: 30, vx: -0.35, vy: 0.22, mass: 1.05, clusterId: 11 },
      { x: 39, y: 38, vx: 0.18, vy: 0.33, mass: 1.3, clusterId: 11 },
    ],
  };
  const loops = [
    { clusterId: 7, indices: [0, 1, 2, 3] },
    { clusterId: 11, indices: [4, 5, 6] },
    { clusterId: 99, indices: [0, 2] },
  ];

  const baselineSoft = structuredClone(softSeed);
  const gpuOnlySoft = structuredClone(softSeed);
  const baselineSim = {
    softAreaRest: new Map([[7, 44.2], [11, 29.4]]),
    softAreaLambda: new Map([[7, 0.07], [11, -0.04]]),
  };
  const gpuOnlySim = {
    softAreaRest: new Map([[7, 44.2], [11, 29.4]]),
    softAreaLambda: new Map([[7, 0.07], [11, -0.04]]),
  };
  const wgslOffload = { enabled: true, state: {} };

  applySoftAreaXPBDVelocityBaseline(baselineSim, baselineSoft, loops, dtPos, stiffnessScale);
  applySoftAreaXPBDVelocityGpuOnly({
    sim: gpuOnlySim,
    soft: gpuOnlySoft,
    loops,
    dtPos,
    stiffnessScale,
    softAreaXpbdIters: SOFT_AREA_XPBD_ITERS,
    softAreaBaseCompliance: SOFT_AREA_BASE_COMPLIANCE,
    wgslOffload,
  });

  assert.deepEqual(
    Array.from(gpuOnlySim.softAreaLambda.entries()),
    Array.from(baselineSim.softAreaLambda.entries()),
    'cluster lambda map should match baseline area stepping exactly',
  );

  assert.equal(gpuOnlySoft.nodes.length, baselineSoft.nodes.length);

  assert.equal(wgslOffload.state.lastMode, 'cpu-prepared', 'expected gpu-only area pass to publish WGSL-prepared layout mode');
  assert.equal(wgslOffload.state.lastPreparedClusterCount, 3, 'expected prepared cluster count to include all loops');
  assert.equal(wgslOffload.state.lastPreparedEndpointCount, 9, 'expected prepared endpoint count to match flattened loop endpoints');
  assert.equal(wgslOffload.state.preparedLayout?.endpointClusterIndex instanceof Uint32Array, true, 'expected prepared WGSL layout to expose deterministic endpoint->cluster ownership');
  assert.equal(wgslOffload.state.preparedLayout?.endpointClusterIndex?.length, 9, 'expected endpoint ownership lookup length to match flattened endpoints');
  assert.ok((wgslOffload.state.lastPreparedLayoutBytes || 0) > 0, 'expected prepared WGSL layout byte footprint to be tracked');

  for (let i = 0; i < baselineSoft.nodes.length; i++) {
    const b = baselineSoft.nodes[i];
    const g = gpuOnlySoft.nodes[i];
    assert.ok(Math.abs(g.vx - b.vx) < 1e-12, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-12, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    assert.ok(Math.abs(g.x - b.x) < 1e-12, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-12, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
  }
});

test('soft area XPBD WGSL proposal stage runs on gpu-only path while CPU remains authoritative', async () => {
  globalThis.GPUBufferUsage = {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    COPY_SRC: 1 << 2,
    MAP_READ: 1 << 3,
    UNIFORM: 1 << 4,
  };
  globalThis.GPUMapMode = { READ: 1 };

  const dtPos = 0.16;
  const stiffnessScale = 3.4;
  const softSeed = {
    nodes: [
      { x: 14, y: 18, vx: 0.3, vy: -0.2, mass: 1.0, clusterId: 7 },
      { x: 22, y: 17, vx: -0.2, vy: 0.1, mass: 0.9, clusterId: 7 },
      { x: 26, y: 24, vx: 0.4, vy: 0.3, mass: 1.2, clusterId: 7 },
      { x: 19, y: 29, vx: -0.3, vy: -0.1, mass: 1.4, clusterId: 7 },
    ],
  };
  const loops = [{ clusterId: 7, indices: [0, 1, 2, 3] }];

  const baselineSoft = structuredClone(softSeed);
  const gpuOnlySoft = structuredClone(softSeed);
  const baselineSim = {
    softAreaRest: new Map([[7, 44.2]]),
    softAreaLambda: new Map([[7, 0.07]]),
  };
  const gpuOnlySim = {
    softAreaRest: new Map([[7, 44.2]]),
    softAreaLambda: new Map([[7, 0.07]]),
  };
  const wgslState = {};

  applySoftAreaXPBDVelocityBaseline(baselineSim, baselineSoft, loops, dtPos, stiffnessScale);
  applySoftAreaXPBDVelocityGpuOnly({
    sim: gpuOnlySim,
    soft: gpuOnlySoft,
    loops,
    dtPos,
    stiffnessScale,
    softAreaXpbdIters: SOFT_AREA_XPBD_ITERS,
    softAreaBaseCompliance: SOFT_AREA_BASE_COMPLIANCE,
    wgslOffload: {
      enabled: true,
      device: createSoftAreaMockWgslDevice(),
      state: wgslState,
    },
  });

  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(Array.from(gpuOnlySim.softAreaLambda.entries()), Array.from(baselineSim.softAreaLambda.entries()));
  assert.deepEqual(gpuOnlySoft.nodes, baselineSoft.nodes);
  assert.equal(wgslState.lastMode, 'wgsl-velocity-proposal');
  assert.equal(wgslState.lastError, null);
  assert.equal(wgslState.lastAreaProbeClusterCount, 1);
  assert.equal(wgslState.lastAreaProposalClusterCount, 1);
  assert.equal(wgslState.lastAreaProposalDeltaLambdaByCluster instanceof Float32Array, true);
  assert.equal(wgslState.lastAreaProposalLambdaNextByCluster instanceof Float32Array, true);
  assert.equal(wgslState.lastAreaProposalDeltaLambdaByCluster.length, 1);
  assert.equal(wgslState.lastAreaProposalLambdaNextByCluster.length, 1);
  assert.equal(wgslState.lastAreaVelocityProposalEndpointCount, 4);
  assert.equal(wgslState.lastAreaVelocityProposalDeltaVxByEndpoint instanceof Float32Array, true);
  assert.equal(wgslState.lastAreaVelocityProposalDeltaVyByEndpoint instanceof Float32Array, true);
  assert.equal(wgslState.lastAreaVelocityProposalDeltaVxByEndpoint.length, 4);
  assert.equal(wgslState.lastAreaVelocityProposalDeltaVyByEndpoint.length, 4);
  assert.equal(wgslState.lastAreaVelocityProposalNodeDeltaVx instanceof Float32Array, true);
  assert.equal(wgslState.lastAreaVelocityProposalNodeDeltaVy instanceof Float32Array, true);
  assert.equal(wgslState.lastAreaVelocityProposalNodeContributionCount instanceof Uint32Array, true);
  assert.equal(wgslState.lastAreaVelocityProposalNodeDeltaVx.length, gpuOnlySoft.nodes.length);
  assert.equal(wgslState.lastAreaVelocityProposalNodeDeltaVy.length, gpuOnlySoft.nodes.length);
  assert.equal(wgslState.lastAreaVelocityProposalNodeContributionCount.length, gpuOnlySoft.nodes.length);

  const reducedSumVx = wgslState.lastAreaVelocityProposalNodeDeltaVx.reduce((sum, v) => sum + v, 0);
  const reducedSumVy = wgslState.lastAreaVelocityProposalNodeDeltaVy.reduce((sum, v) => sum + v, 0);
  const endpointSumVx = wgslState.lastAreaVelocityProposalDeltaVxByEndpoint.reduce((sum, v) => sum + v, 0);
  const endpointSumVy = wgslState.lastAreaVelocityProposalDeltaVyByEndpoint.reduce((sum, v) => sum + v, 0);
  assert.ok(Math.abs(reducedSumVx - endpointSumVx) < 1e-6, 'node-reduced vx should conserve endpoint proposal sum');
  assert.ok(Math.abs(reducedSumVy - endpointSumVy) < 1e-6, 'node-reduced vy should conserve endpoint proposal sum');
  assert.deepEqual(Array.from(wgslState.lastAreaVelocityProposalNodeContributionCount), [1, 1, 1, 1]);
});
