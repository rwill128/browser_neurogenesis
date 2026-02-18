import test from 'node:test';
import assert from 'node:assert/strict';

import {
  computeSoftClusterKinematics,
  projectNodesTowardClusterRigidMotion,
} from '../../sim-server/public/soft-cluster-kinematics.js';
import {
  buildSoftClusterKinematicsWgslPrep,
  computeSoftClusterKinematicsFromMassMomentsGpuOnly,
  computeSoftClusterKinematicsGpuOnly,
  computeSoftClusterKinematicsPrepSignature,
} from '../../sim-server/public/runtime-solvers/stepSoftClusterKinematicsGpuOnly.js';

test('computeSoftClusterKinematics returns mass-weighted COM, inertia, and omega', () => {
  const nodes = [
    { x: -1, y: 0, vx: 0, vy: -2, mass: 1, clusterId: 0 },
    { x: 1, y: 0, vx: 0, vy: 2, mass: 1, clusterId: 0 },
  ];

  const map = computeSoftClusterKinematics(nodes);
  const c = map.get(0);
  assert.ok(c, 'cluster 0 should exist');

  assert.equal(c.mass, 2);
  assert.equal(c.x, 0);
  assert.equal(c.y, 0);
  assert.equal(c.vx, 0);
  assert.equal(c.vy, 0);
  assert.equal(c.inertia, 2);
  assert.equal(c.angularMomentum, 4);
  assert.equal(c.omega, 2);
});

test('projectNodesTowardClusterRigidMotion applies explicit angular field', () => {
  const nodes = [
    { x: 1, y: 0, vx: 0, vy: 0, clusterId: 7 },
    { x: 0, y: 1, vx: 0, vy: 0, clusterId: 7 },
  ];

  const cluster = new Map([[7, {
    clusterId: 7,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    omega: 1,
  }]]);

  const out = projectNodesTowardClusterRigidMotion(nodes, cluster, {
    linearGain: 0,
    angularGain: 1,
  });

  assert.equal(out.projectedNodes, 2);
  assert.ok(out.meanDelta > 0, 'projection should adjust at least one velocity');

  // v = omega x r => (-omega*y, omega*x)
  assert.equal(nodes[0].vx, 0);
  assert.equal(nodes[0].vy, 1);
  assert.equal(nodes[1].vx, -1);
  assert.equal(nodes[1].vy, 0);
});

test('projectNodesTowardClusterRigidMotion honors membrane gain scale', () => {
  const nodes = [
    { x: 1, y: 0, vx: 0, vy: 0, clusterId: 3 },
  ];

  const cluster = new Map([[3, {
    clusterId: 3,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    omega: 2,
  }]]);

  projectNodesTowardClusterRigidMotion(nodes, cluster, {
    linearGain: 0,
    angularGain: 1,
    membraneClusterSet: new Set([3]),
    membraneGainScale: 0.5,
  });

  // Full target vy would be +2; with 0.5 scale and starting at 0, expect halfway.
  assert.equal(nodes[0].vx, 0);
  assert.equal(nodes[0].vy, 1);
});

test('buildSoftClusterKinematicsWgslPrep emits deterministic CSR layout for next WGSL reduction stage', () => {
  const nodes = [
    { x: 2, y: 4, vx: 1, vy: 0, mass: 2, clusterId: 5 },
    { x: 1, y: 3, vx: 0, vy: 1, mass: 1, clusterId: 2 },
    { x: 3, y: 5, vx: -1, vy: 0, mass: 3, clusterId: 5 },
    { x: Number.NaN, y: 9, vx: 0, vy: 0, mass: 1, clusterId: 9 },
  ];

  const prepA = buildSoftClusterKinematicsWgslPrep(nodes);
  const prepB = buildSoftClusterKinematicsWgslPrep(nodes);

  assert.deepEqual([...prepA.layout.clusterOriginalId], [2, 5]);
  assert.deepEqual([...prepA.layout.clusterOffsets], [0, 1, 3]);
  assert.deepEqual([...prepA.layout.nodeIndex], [1, 0, 2]);
  assert.ok(prepA.layout.byteLength > 0);
  assert.equal(prepA.plan.nodeCount, 3);
  assert.equal(prepA.plan.clusterCount, 2);

  assert.equal(prepA.signature, prepB.signature);
  assert.equal(prepA.signature, computeSoftClusterKinematicsPrepSignature(prepA.layout));
});

test('computeSoftClusterKinematicsFromMassMomentsGpuOnly reconstructs authoritative cluster state from WGSL mass moments', () => {
  const nodes = [
    { x: -1, y: 0, vx: 0, vy: -2, mass: 1, clusterId: 0 },
    { x: 1, y: 0, vx: 0, vy: 2, mass: 1, clusterId: 0 },
    { x: 3, y: 2, vx: 1, vy: 0.5, mass: 2, clusterId: 4 },
  ];

  const prep = buildSoftClusterKinematicsWgslPrep(nodes);
  const probe = {
    mass: new Float32Array(prep.plan.clusterCount),
    xMass: new Float32Array(prep.plan.clusterCount),
    yMass: new Float32Array(prep.plan.clusterCount),
    vxMass: new Float32Array(prep.plan.clusterCount),
    vyMass: new Float32Array(prep.plan.clusterCount),
  };

  for (let ci = 0; ci < prep.plan.clusterCount; ci++) {
    const start = prep.layout.clusterOffsets[ci];
    const stop = prep.layout.clusterOffsets[ci + 1];
    for (let i = start; i < stop; i++) {
      const m = Math.max(0.02, prep.layout.nodeMass[i]);
      probe.mass[ci] += m;
      probe.xMass[ci] += prep.layout.nodeX[i] * m;
      probe.yMass[ci] += prep.layout.nodeY[i] * m;
      probe.vxMass[ci] += prep.layout.nodeVx[i] * m;
      probe.vyMass[ci] += prep.layout.nodeVy[i] * m;
    }
  }

  const reconstructed = computeSoftClusterKinematicsFromMassMomentsGpuOnly(nodes, prep.layout, probe);
  const cpu = computeSoftClusterKinematicsGpuOnly(nodes);

  assert.equal(reconstructed.size, cpu.size);
  for (const [cid, ref] of cpu.entries()) {
    const got = reconstructed.get(cid);
    assert.ok(got, `missing cluster ${cid}`);
    assert.ok(Math.abs(got.mass - ref.mass) < 1e-6);
    assert.ok(Math.abs(got.x - ref.x) < 1e-6);
    assert.ok(Math.abs(got.y - ref.y) < 1e-6);
    assert.ok(Math.abs(got.vx - ref.vx) < 1e-6);
    assert.ok(Math.abs(got.vy - ref.vy) < 1e-6);
    assert.ok(Math.abs(got.omega - ref.omega) < 1e-6);
  }
});
