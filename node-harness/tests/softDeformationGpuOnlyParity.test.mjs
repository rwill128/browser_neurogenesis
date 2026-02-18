import test from 'node:test';
import assert from 'node:assert/strict';

import { applySoftDeformationInterventionsGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftDeformationGpuOnly.js';

function runBaseline({ sim, soft, softClusterLoops, severeInterventionsOn, buildSoftDeformationState, stabilizeSeverelyDeformedSoftClusters }) {
  let deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  if (severeInterventionsOn && deform.severeCollapseCount > 0) {
    stabilizeSeverelyDeformedSoftClusters(sim, soft, softClusterLoops, deform);
    deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  }
  return deform;
}

test('soft deformation intervention parity: gpu-only orchestration matches baseline severe-collapse restabilization flow', () => {
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
  const gpuOnly = applySoftDeformationInterventionsGpuOnly({
    ...gpuHarness,
    severeInterventionsOn: true,
  });

  assert.deepEqual(gpuOnly, baseline);
  assert.deepEqual(gpuHarness.callLog, baselineHarness.callLog);
});

test('soft deformation intervention parity: gpu-only path skips severe stabilization when toggle is off', () => {
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

  const result = applySoftDeformationInterventionsGpuOnly({
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
