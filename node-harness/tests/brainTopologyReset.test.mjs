import test from 'node:test';
import assert from 'node:assert/strict';

import { Brain } from '../../js/classes/Brain.js';
import { NodeType } from '../../js/classes/constants.js';

test('brain resize preserves overlapping weights and resets stale RL buffer on topology change', () => {
  const brainPoint = {
    nodeType: NodeType.NEURON,
    neuronData: {
      isBrain: true,
      hiddenLayerSize: 5
    }
  };

  const softBody = {
    massPoints: [brainPoint],
    springs: [],
    creatureEnergy: 42,

    // Node-type cache counters consumed by Brain sizing logic.
    numEmitterNodes: 1,
    numSwimmerNodes: 0,
    numEaterNodes: 0,
    numPredatorNodes: 0,
    numEyeNodes: 0,
    numJetNodes: 0,
    numPotentialGrabberNodes: 0,
    numAttractorNodes: 0,
    numRepulsorNodes: 0,

    // Continuity counters updated on topology changes.
    rlBufferResetsDueToTopology: 0,
    nnTopologyVersion: 0
  };

  const brain = new Brain(softBody);
  const nd = brainPoint.neuronData;

  // Install sentinels and stale buffer that should be preserved/reset selectively.
  nd.weightsIH[0][0] = 123.456;
  nd.weightsHO[0][0] = -77.25;
  nd.experienceBuffer = [{ state: [1, 2, 3], reward: 0.1, actionDetails: [] }];
  nd.framesSinceLastTrain = 7;

  // Trigger topology change by increasing emitter outputs.
  softBody.numEmitterNodes = 2;
  brain.initialize();

  assert.equal(nd.outputVectorSize, 16, '2 emitters should produce 16 outputs (8 each)');
  assert.equal(nd.weightsIH[0][0], 123.456, 'input-hidden overlap should be preserved');
  assert.equal(nd.weightsHO[0][0], -77.25, 'hidden-output overlap should be preserved');

  assert.deepEqual(nd.experienceBuffer, [], 'stale on-policy buffer should be flushed on topology change');
  assert.equal(nd.framesSinceLastTrain, 0, 'training frame counter should reset after flush');
  assert.equal(softBody.rlBufferResetsDueToTopology, 1, 'continuity telemetry should count reset');
  assert.equal(softBody.nnTopologyVersion, 1, 'topology version should bump on resize');
});
