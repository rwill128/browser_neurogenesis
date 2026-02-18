import test from 'node:test';
import assert from 'node:assert/strict';

import { resolveRigidVsRigidPolygonCollision } from '../../sim-server/public/rigid-collision.js';
import { resolveRigidRigidCollisionPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidCollisionGpuOnly.js';

function makeRigidBodies() {
  return [
    {
      x: 10.2,
      y: 10.4,
      vx: 0.38,
      vy: -0.17,
      omega: 0.06,
      theta: 0.12,
      mass: 1.5,
      inertia: 0.9,
      verticesLocal: [
        { x: -1.0, y: -0.6 },
        { x: 1.0, y: -0.5 },
        { x: 0.9, y: 0.8 },
        { x: -0.8, y: 0.9 },
      ],
    },
    {
      x: 10.9,
      y: 10.7,
      vx: -0.24,
      vy: 0.19,
      omega: -0.04,
      theta: -0.07,
      mass: 1.8,
      inertia: 1.1,
      verticesLocal: [
        { x: -0.9, y: -0.7 },
        { x: 0.8, y: -0.7 },
        { x: 0.9, y: 0.6 },
        { x: -0.7, y: 0.8 },
      ],
    },
    {
      x: 11.3,
      y: 10.1,
      vx: 0.11,
      vy: 0.21,
      omega: 0.03,
      theta: 0.18,
      mass: 1.2,
      inertia: 0.8,
      verticesLocal: [
        { x: -0.6, y: -0.6 },
        { x: 0.7, y: -0.5 },
        { x: 0.6, y: 0.7 },
        { x: -0.7, y: 0.6 },
      ],
    },
  ];
}

function runBaselinePass(rigidBodies, { iter, phase }) {
  const contacts = [];
  for (let i = 0; i < rigidBodies.length; i++) {
    for (let j = i + 1; j < rigidBodies.length; j++) {
      resolveRigidVsRigidPolygonCollision(rigidBodies[i], rigidBodies[j], 0.32, {
        contacts,
        aIndex: i,
        bIndex: j,
        iter,
        phase,
      });
    }
  }
  return contacts;
}

test('gpu-only rigid collision pass matches baseline rigid-rigid solve behavior', async () => {
  const baselineRigid = makeRigidBodies();
  const gpuRigid = makeRigidBodies();

  const baselineContacts = runBaselinePass(baselineRigid, { iter: 1, phase: 'pre-soft' });
  const gpuContacts = [];
  await resolveRigidRigidCollisionPassGpuOnly({
    rigidBodies: gpuRigid,
    slop: 0.32,
    contacts: gpuContacts,
    iter: 1,
    phase: 'pre-soft',
    resolveRigidVsRigidPolygonCollision,
  });

  assert.deepEqual(gpuRigid, baselineRigid, 'gpu-only rigid collision pass should mutate bodies identically to baseline rigid-rigid loop');
  assert.deepEqual(gpuContacts, baselineContacts, 'gpu-only rigid collision pass should report matching contact debug payload');
});

test('gpu-only rigid collision pass can apply authoritative WGSL pair impulses with explicit CPU fallback route preserved', async () => {
  globalThis.GPUBufferUsage ??= {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    UNIFORM: 1 << 2,
    COPY_SRC: 1 << 3,
    MAP_READ: 1 << 4,
  };
  globalThis.GPUMapMode ??= { READ: 1 };

  const rigidBodies = [
    { x: 1, y: 1, vx: 0.2, vy: 0.0, mass: 2, r: 1 },
    { x: 2, y: 1, vx: -0.1, vy: 0.0, mass: 1, r: 1 },
  ];

  const readbackValues = [
    new Float32Array([0.12]).buffer,
    new Float32Array([-0.08]).buffer,
    new Float32Array([-0.24]).buffer,
    new Float32Array([0.16]).buffer,
    new Uint32Array([1]).buffer,
  ];
  let readbackCursor = 0;
  const dispatches = [];

  function makeBuffer(usage) {
    if ((usage & globalThis.GPUBufferUsage.MAP_READ) !== 0) {
      const payload = readbackValues[readbackCursor++] || new Uint8Array(4).buffer;
      return {
        destroy() {},
        async mapAsync() {},
        getMappedRange() { return payload; },
        unmap() {},
      };
    }
    return { destroy() {} };
  }

  const mockPipeline = { getBindGroupLayout: () => ({}) };
  const device = {
    createBuffer: ({ usage }) => makeBuffer(usage),
    createShaderModule: ({ code }) => ({ code }),
    createComputePipelineAsync: async () => mockPipeline,
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
      writeBuffer() {},
      submit() {},
    },
  };

  let cpuCalls = 0;
  const wgslState = {};
  const contacts = [];

  await resolveRigidRigidCollisionPassGpuOnly({
    rigidBodies,
    slop: 0.32,
    contacts,
    iter: 2,
    phase: 'post-soft',
    resolveRigidVsRigidPolygonCollision: () => {
      cpuCalls += 1;
    },
    wgslOffload: {
      enabled: true,
      authoritativeRigidCollision: true,
      device,
      state: wgslState,
    },
  });

  assert.equal(cpuCalls, 0, 'expected WGSL authoritative rigid collision to bypass CPU resolver when finite');
  assert.equal(dispatches.length, 1);
  assert.equal(wgslState.lastRigidCollisionAuthoritativeSource, 'wgsl-rigid-collision-authoritative');
  assert.equal(wgslState.lastSourceRoute, 'wgsl-rigid-collision-authoritative');
  assert.equal(wgslState.lastMode, 'wgsl-rigid-collision-authoritative');
  assert.equal(contacts.length, 1);
  assert.equal(contacts[0].source, 'wgsl-rigid-collision-authoritative');
  assert.ok(Math.abs(rigidBodies[0].vx - 0.32) < 1e-6);
  assert.ok(Math.abs(rigidBodies[0].vy - (-0.08)) < 1e-6);
  assert.ok(Math.abs(rigidBodies[1].vx - (-0.34)) < 1e-6);
  assert.ok(Math.abs(rigidBodies[1].vy - 0.16) < 1e-6);
});
