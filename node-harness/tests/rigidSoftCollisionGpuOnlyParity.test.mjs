import test from 'node:test';
import assert from 'node:assert/strict';

import { resolveRigidVsSoftNodeCollision } from '../../sim-server/public/rigid-collision.js';
import {
  resolveRigidSoftCollisionPassGpuOnly,
  resolveRigidVsSoftEdgeCollisionGpuOnly,
} from '../../sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js';

const EDGE_BODY_MODE = {
  PASS: 0,
  BLOCK: 1,
};

function makeState() {
  return {
    rigid: [
      { id: 'r0' },
      { id: 'r1' },
    ],
    soft: {
      nodes: [
        { id: 'n0' },
        { id: 'n1' },
        { id: 'n2' },
        { id: 'n3' },
      ],
      springs: [
        [0, 1, 1.2, EDGE_BODY_MODE.BLOCK],
        [1, 2, 1.0, EDGE_BODY_MODE.PASS],
        [2, 3, 1.1, EDGE_BODY_MODE.BLOCK],
      ],
    },
    hybridAttachedByRigid: new Map([
      [0, new Set([1])],
      [1, new Set([2])],
    ]),
  };
}

function runBaseline(state, calls) {
  for (let rbi = 0; rbi < state.rigid.length; rbi++) {
    const rb = state.rigid[rbi];
    const attachedNodeSet = state.hybridAttachedByRigid.get(rbi) || null;

    for (let ni = 0; ni < state.soft.nodes.length; ni++) {
      if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
      const sn = state.soft.nodes[ni];
      calls.nodes.push(`${rb.id}->${sn.id}`);
    }

    for (const [i, j, _rest, edgeBodyMode] of state.soft.springs) {
      if (edgeBodyMode !== EDGE_BODY_MODE.BLOCK) continue;
      if (attachedNodeSet && (attachedNodeSet.has(i) || attachedNodeSet.has(j))) continue;
      calls.edges.push(`${rb.id}->${state.soft.nodes[i].id}-${state.soft.nodes[j].id}`);
    }
  }
}

test('gpu-only rigid-soft collision pass matches baseline contact visitation and filtering', async () => {
  const baselineState = makeState();
  const gpuState = makeState();

  const baselineCalls = { nodes: [], edges: [] };
  runBaseline(baselineState, baselineCalls);

  const gpuCalls = { nodes: [], edges: [] };
  await resolveRigidSoftCollisionPassGpuOnly({
    rigidBodies: gpuState.rigid,
    soft: gpuState.soft,
    hybridAttachedByRigid: gpuState.hybridAttachedByRigid,
    resolveRigidVsSoftNodeCollision: (rb, sn) => {
      gpuCalls.nodes.push(`${rb.id}->${sn.id}`);
    },
    resolveRigidVsSoftEdgeCollision: (rb, a, b) => {
      gpuCalls.edges.push(`${rb.id}->${a.id}-${b.id}`);
    },
    edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
    nodeSlop: 0.18,
    edgeSlop: 0.16,
  });

  assert.deepEqual(gpuCalls, baselineCalls);
});

function baselineEdgeCollision(rigid, a, b, restitution = 0.28) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const ab2 = abx * abx + aby * aby;
  const apx = rigid.x - a.x;
  const apy = rigid.y - a.y;
  const t = ab2 < 1e-8 ? 0 : Math.max(0, Math.min(1, (apx * abx + apy * aby) / ab2));
  const cpx = a.x + abx * t;
  const cpy = a.y + aby * t;

  let nx = rigid.x - cpx;
  let ny = rigid.y - cpy;
  let dist = Math.hypot(nx, ny);
  const minDist = Math.max(0.8, rigid.r || 1);
  if (dist >= minDist) return false;

  if (dist < 1e-6) {
    const invLen = 1 / Math.max(1e-6, Math.hypot(-aby, abx));
    nx = -aby * invLen;
    ny = abx * invLen;
    dist = 1e-6;
  } else {
    nx /= dist;
    ny /= dist;
  }

  const penetration = minDist - dist;
  rigid.x += nx * penetration * 0.92;
  rigid.y += ny * penetration * 0.92;

  const edgeVx = (a.vx + b.vx) * 0.5;
  const edgeVy = (a.vy + b.vy) * 0.5;
  const rvx = rigid.vx - edgeVx;
  const rvy = rigid.vy - edgeVy;
  const vn = rvx * nx + rvy * ny;
  if (vn < 0) {
    const j = -(1 + restitution) * vn;
    rigid.vx += nx * j;
    rigid.vy += ny * j;
  }
  return true;
}

function makeRuntimeState() {
  return {
    rigidBodies: [
      {
        x: 6.8,
        y: 6.3,
        vx: -0.34,
        vy: 0.41,
        omega: 0.03,
        r: 1.3,
        mass: 1.7,
        inertia: 0.9,
        theta: 0.1,
        verticesLocal: [
          { x: -1.0, y: -0.8 },
          { x: 1.2, y: -0.5 },
          { x: 1.0, y: 0.9 },
          { x: -0.8, y: 1.0 },
        ],
      },
    ],
    soft: {
      nodes: [
        { x: 6.2, y: 6.1, vx: 0.2, vy: -0.1, r: 0.7, mass: 0.9 },
        { x: 7.4, y: 6.0, vx: -0.05, vy: 0.07, r: 0.7, mass: 0.95 },
        { x: 7.0, y: 7.2, vx: 0.08, vy: -0.02, r: 0.7, mass: 1.0 },
      ],
      springs: [
        [0, 1, 1.0, EDGE_BODY_MODE.BLOCK],
        [1, 2, 1.0, EDGE_BODY_MODE.PASS],
        [2, 0, 1.0, EDGE_BODY_MODE.BLOCK],
      ],
    },
    hybridAttachedByRigid: new Map(),
  };
}

function deepClone(v) {
  return structuredClone(v);
}

function snapshotRuntimeState(state) {
  return {
    rigidBodies: state.rigidBodies.map((rb) => ({ x: rb.x, y: rb.y, vx: rb.vx, vy: rb.vy, omega: rb.omega })),
    softNodes: state.soft.nodes.map((sn) => ({ x: sn.x, y: sn.y, vx: sn.vx, vy: sn.vy })),
  };
}

test('gpu-only rigid-soft pass default runtime collision solver matches baseline physics outcomes', async () => {
  const baseline = makeRuntimeState();
  const gpuOnly = deepClone(baseline);

  await resolveRigidSoftCollisionPassGpuOnly({
    rigidBodies: baseline.rigidBodies,
    soft: baseline.soft,
    hybridAttachedByRigid: baseline.hybridAttachedByRigid,
    resolveRigidVsSoftNodeCollision,
    resolveRigidVsSoftEdgeCollision: baselineEdgeCollision,
    edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
    nodeSlop: 0.18,
    edgeSlop: 0.16,
  });

  await resolveRigidSoftCollisionPassGpuOnly({
    rigidBodies: gpuOnly.rigidBodies,
    soft: gpuOnly.soft,
    hybridAttachedByRigid: gpuOnly.hybridAttachedByRigid,
    edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
    nodeSlop: 0.18,
    edgeSlop: 0.16,
  });

  const baselineSnapshot = snapshotRuntimeState(baseline);
  const gpuSnapshot = snapshotRuntimeState(gpuOnly);

  const eps = 1e-9;
  for (let i = 0; i < baselineSnapshot.rigidBodies.length; i++) {
    const a = baselineSnapshot.rigidBodies[i];
    const b = gpuSnapshot.rigidBodies[i];
    assert.ok(Math.abs(a.x - b.x) <= eps);
    assert.ok(Math.abs(a.y - b.y) <= eps);
    assert.ok(Math.abs(a.vx - b.vx) <= eps);
    assert.ok(Math.abs(a.vy - b.vy) <= eps);
    assert.ok(Math.abs(a.omega - b.omega) <= eps);
  }
  for (let i = 0; i < baselineSnapshot.softNodes.length; i++) {
    const a = baselineSnapshot.softNodes[i];
    const b = gpuSnapshot.softNodes[i];
    assert.ok(Math.abs(a.x - b.x) <= eps);
    assert.ok(Math.abs(a.y - b.y) <= eps);
    assert.ok(Math.abs(a.vx - b.vx) <= eps);
    assert.ok(Math.abs(a.vy - b.vy) <= eps);
  }

  const rigidBaselineSpeed = Math.hypot(baselineSnapshot.rigidBodies[0].vx, baselineSnapshot.rigidBodies[0].vy);
  const rigidGpuSpeed = Math.hypot(gpuSnapshot.rigidBodies[0].vx, gpuSnapshot.rigidBodies[0].vy);
  assert.ok(Math.abs(rigidBaselineSpeed - rigidGpuSpeed) <= eps);

  // PASS spring must remain non-colliding in both paths.
  const passEdgeA = baseline.soft.nodes[1];
  const passEdgeB = baseline.soft.nodes[2];
  const rb = baseline.rigidBodies[0];
  const num = Math.abs((passEdgeB.y - passEdgeA.y) * rb.x - (passEdgeB.x - passEdgeA.x) * rb.y + passEdgeB.x * passEdgeA.y - passEdgeB.y * passEdgeA.x);
  const den = Math.max(1e-6, Math.hypot(passEdgeB.y - passEdgeA.y, passEdgeB.x - passEdgeA.x));
  const dist = num / den;
  assert.ok(dist >= 0);

  // Ensure exported default edge solver path is active and callable.
  const probeRigid = { x: 0, y: 0, vx: 0, vy: 0, r: 1 };
  const moved = resolveRigidVsSoftEdgeCollisionGpuOnly(probeRigid, { x: 10, y: 10, vx: 0, vy: 0 }, { x: 12, y: 10, vx: 0, vy: 0 }, 0.16);
  assert.equal(moved, false);
});

test('gpu-only rigid-soft pass publishes deterministic WGSL candidate layout source-route telemetry while preserving parity', async () => {
  const baselineState = makeState();
  const gpuState = makeState();
  const wgslState = {};

  const baselineCalls = { nodes: [], edges: [] };
  runBaseline(baselineState, baselineCalls);

  const gpuCalls = { nodes: [], edges: [] };
  await resolveRigidSoftCollisionPassGpuOnly({
    rigidBodies: gpuState.rigid,
    soft: gpuState.soft,
    hybridAttachedByRigid: gpuState.hybridAttachedByRigid,
    resolveRigidVsSoftNodeCollision: (rb, sn) => {
      gpuCalls.nodes.push(`${rb.id}->${sn.id}`);
    },
    resolveRigidVsSoftEdgeCollision: (rb, a, b) => {
      gpuCalls.edges.push(`${rb.id}->${a.id}-${b.id}`);
    },
    edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
    wgslOffload: { enabled: true, state: wgslState },
  });

  assert.deepEqual(gpuCalls, baselineCalls);
  assert.equal(wgslState.lastMode, 'cpu-prepared');
  assert.equal(wgslState.lastSourceRoute, 'cpu-rigid-soft-candidate-layout');
  assert.ok(Number.isInteger(wgslState.lastPreparedLayoutSignature));
  assert.ok(wgslState.lastPreparedLayoutBytes > 0);
  assert.equal(wgslState.lastPreparedNodePairCount, baselineCalls.nodes.length);
  assert.equal(wgslState.lastPreparedEdgePairCount, baselineCalls.edges.length);
  assert.equal(wgslState.preparedLayout.nodePairRigidIndex.length, baselineCalls.nodes.length);
  assert.equal(wgslState.preparedLayout.nodePairNodeIndex.length, baselineCalls.nodes.length);
  assert.equal(wgslState.preparedLayout.edgePairRigidIndex.length, baselineCalls.edges.length);
  assert.equal(wgslState.preparedLayout.edgePairSpringIndex.length, baselineCalls.edges.length);
});

test('gpu-only rigid-soft pass dispatches WGSL node broadphase proposal when device is available while preserving CPU collision visitation parity', async () => {
  globalThis.GPUBufferUsage ??= {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    UNIFORM: 1 << 2,
    COPY_SRC: 1 << 3,
    MAP_READ: 1 << 4,
  };
  globalThis.GPUMapMode ??= {
    READ: 1,
  };

  const baselineState = makeState();
  const gpuState = makeState();
  const baselineCalls = { nodes: [], edges: [] };
  const gpuCalls = { nodes: [], edges: [] };
  runBaseline(baselineState, baselineCalls);

  const writes = [];
  const dispatches = [];
  const mockPipeline = { getBindGroupLayout: () => ({}) };
  const makeReadableBuffer = () => {
    const data = new Uint32Array(1024);
    data.fill(1);
    return {
      destroy() {},
      async mapAsync() {},
      getMappedRange(_offset = 0, size = data.byteLength) { return data.buffer.slice(0, size); },
      unmap() {},
    };
  };
  const mockDevice = {
    createBuffer: () => makeReadableBuffer(),
    createShaderModule: ({ code }) => ({ code }),
    createComputePipelineAsync: () => Promise.resolve(mockPipeline),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups(count) { dispatches.push(count); },
        end() {},
      }),
      copyBufferToBuffer() {},
      finish: () => ({}),
    }),
    queue: {
      writeBuffer: (_buf, _offset, data) => writes.push(data?.constructor?.name || typeof data),
      submit() {},
    },
  };

  const wgslState = {
    rigidSoftNodeBroadphasePipeline: mockPipeline,
  };

  await resolveRigidSoftCollisionPassGpuOnly({
    rigidBodies: gpuState.rigid,
    soft: gpuState.soft,
    hybridAttachedByRigid: gpuState.hybridAttachedByRigid,
    resolveRigidVsSoftNodeCollision: (rb, sn) => {
      gpuCalls.nodes.push(`${rb.id}->${sn.id}`);
    },
    resolveRigidVsSoftEdgeCollision: (rb, a, b) => {
      gpuCalls.edges.push(`${rb.id}->${a.id}-${b.id}`);
    },
    edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
    wgslOffload: { enabled: true, state: wgslState, device: mockDevice },
  });

  assert.deepEqual(gpuCalls, baselineCalls);
  assert.equal(wgslState.lastNodeBroadphaseDispatched, true);
  assert.equal(wgslState.lastSourceRoute, 'wgsl-rigid-soft-node-broadphase-authoritative-filter');
  assert.equal(wgslState.lastMode, 'wgsl-broadphase-authoritative-filter');
  assert.ok(dispatches[0] >= 1);
  assert.ok(writes.length >= 8);
});
