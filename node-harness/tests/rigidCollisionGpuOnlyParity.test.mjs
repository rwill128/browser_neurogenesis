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

test('gpu-only rigid collision pass matches baseline rigid-rigid solve behavior', () => {
  const baselineRigid = makeRigidBodies();
  const gpuRigid = makeRigidBodies();

  const baselineContacts = runBaselinePass(baselineRigid, { iter: 1, phase: 'pre-soft' });
  const gpuContacts = [];
  resolveRigidRigidCollisionPassGpuOnly({
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
