import test from 'node:test';
import assert from 'node:assert/strict';

import {
  applySoftDeformationInterventionsGpuOnly,
  buildSoftDeformationInterventionsWgslPrep,
} from '../../sim-server/public/runtime-solvers/stepSoftDeformationGpuOnly.js';

function runBaseline({ sim, soft, softClusterLoops, severeInterventionsOn, buildSoftDeformationState, stabilizeSeverelyDeformedSoftClusters }) {
  let deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  if (severeInterventionsOn && deform.severeCollapseCount > 0) {
    stabilizeSeverelyDeformedSoftClusters(sim, soft, softClusterLoops, deform);
    deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  }
  return deform;
}

test('soft deformation intervention parity: gpu-only orchestration matches baseline severe-collapse restabilization flow', async () => {
  const makeHarness = () => {
    const sim = { frame: 12 };
    const soft = { nodes: [{ x: 1, y: 2 }, { x: 5, y: 6 }] };
    const softClusterLoops = [[0, 1]];
    const callLog = [];

    let callIndex = 0;
    const buildSoftDeformationState = () => {
      callIndex += 1;
      callLog.push(`build:${callIndex}`);
      if (callIndex === 1) {
        return {
          severeCollapseCount: 1,
          severeClusters: [3],
          warningCount: 2,
          clusters: [{ clusterId: 3, poseErrorRms: 0.9 }],
        };
      }
      return {
        severeCollapseCount: 0,
        severeClusters: [],
        warningCount: 1,
        clusters: [{ clusterId: 3, poseErrorRms: 0.4 }],
      };
    };

    const stabilizeSeverelyDeformedSoftClusters = (simArg, softArg, loopsArg, deformArg) => {
      callLog.push(`stabilize:${deformArg.severeCollapseCount}`);
      assert.equal(simArg, sim);
      assert.equal(softArg, soft);
      assert.equal(loopsArg, softClusterLoops);
    };

    return {
      sim,
      soft,
      softClusterLoops,
      callLog,
      buildSoftDeformationState,
      stabilizeSeverelyDeformedSoftClusters,
    };
  };

  const baselineHarness = makeHarness();
  const baseline = runBaseline({
    ...baselineHarness,
    severeInterventionsOn: true,
  });

  const gpuHarness = makeHarness();
  const gpuOnly = await applySoftDeformationInterventionsGpuOnly({
    ...gpuHarness,
    severeInterventionsOn: true,
  });

  assert.deepEqual(gpuOnly, baseline);
  assert.deepEqual(gpuHarness.callLog, baselineHarness.callLog);
});

test('soft deformation intervention parity: gpu-only path skips severe stabilization when toggle is off', async () => {
  const sim = { frame: 33 };
  const soft = { nodes: [{ x: 0, y: 0 }] };
  const softClusterLoops = [];
  const callLog = [];

  const buildSoftDeformationState = () => {
    callLog.push('build');
    return {
      severeCollapseCount: 2,
      severeClusters: [1],
      warningCount: 0,
      clusters: [{ clusterId: 1 }],
    };
  };

  const stabilizeSeverelyDeformedSoftClusters = () => {
    callLog.push('stabilize');
  };

  const result = await applySoftDeformationInterventionsGpuOnly({
    sim,
    soft,
    softClusterLoops,
    severeInterventionsOn: false,
    buildSoftDeformationState,
    stabilizeSeverelyDeformedSoftClusters,
  });

  assert.equal(result.severeCollapseCount, 2);
  assert.deepEqual(callLog, ['build']);
});

test('soft deformation wgsl prep + authoritative source routing visibility', async () => {
  const sim = { frame: 91 };
  const soft = {
    nodes: [
      { x: 1.25, y: 2.5, mass: 2, clusterId: 7 },
      { x: -3.5, y: 4.75, mass: 0.5, clusterId: 7 },
      { x: 8, y: -1.5, mass: 0, clusterId: 9 },
    ],
    springs: [
      [0, 1, 1.75],
      [1, 2, 2.5],
    ],
  };
  const softClusterLoops = [{ clusterId: 7, indices: [0, 1, 2] }, { clusterId: 9, indices: [2, 99] }];

  const prepA = buildSoftDeformationInterventionsWgslPrep({ sim, soft, softClusterLoops });
  const prepB = buildSoftDeformationInterventionsWgslPrep({ sim, soft, softClusterLoops });

  assert.equal(prepA.signature, prepB.signature);
  assert.deepEqual(Array.from(prepA.layout.loopOffsets), [0, 3, 5]);
  assert.deepEqual(Array.from(prepA.layout.loopNodeIndex), [0, 1, 2, 2, 2]);
  assert.deepEqual(Array.from(prepA.layout.loopClusterId), [7, 9]);
  assert.deepEqual(Array.from(prepA.layout.nodeInvMass).map((v) => Number(v.toFixed(6))), [0.5, 2, 0]);
  assert.deepEqual(Array.from(prepA.layout.nodeClusterId), [7, 7, 9]);
  assert.deepEqual(Array.from(prepA.layout.springNodeA), [0, 1]);
  assert.deepEqual(Array.from(prepA.layout.springNodeB), [1, 2]);
  assert.deepEqual(Array.from(prepA.layout.springRest).map((v) => Number(v.toFixed(6))), [1.75, 2.5]);
  assert.deepEqual(Array.from(prepA.layout.springClusterId), [7, 7]);
  assert.deepEqual(Array.from(prepA.layout.clusterIds), [7, 9]);
  assert.deepEqual(Array.from(prepA.layout.springClusterIndex), [0, 0]);

  const wgslOffload = { enabled: true, state: {} };
  await applySoftDeformationInterventionsGpuOnly({
    sim,
    soft,
    softClusterLoops,
    severeInterventionsOn: false,
    buildSoftDeformationState: () => ({ severeCollapseCount: 0 }),
    stabilizeSeverelyDeformedSoftClusters: () => {},
    wgslOffload,
  });

  assert.equal(wgslOffload.state.lastPreparedSoftDeformationNodeCount, 3);
  assert.equal(wgslOffload.state.lastPreparedSoftDeformationLoopCount, 2);
  assert.equal(wgslOffload.state.lastPreparedSoftDeformationSpringCount, 2);
  assert.equal(wgslOffload.state.lastPreparedSoftDeformationClusterCount, 2);
  assert.equal(wgslOffload.state.lastPreparedSoftDeformationFrame, 91);
  assert.equal(wgslOffload.state.lastSourceRoute, 'cpu-soft-deformation-authoritative');
  assert.equal(wgslOffload.state.lastMode, 'cpu-soft-deformation-authoritative');
  assert.equal(
    wgslOffload.state.preparedSoftDeformationSignature,
    prepA.signature,
  );
});
