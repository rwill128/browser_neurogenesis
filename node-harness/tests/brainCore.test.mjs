import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { Brain } from '../../js/classes/Brain.js';
import { NodeType } from '../../js/classes/constants.js';

function createSoftBodyStub(overrides = {}) {
  return {
    massPoints: [],
    springs: [],
    creatureEnergy: 50,

    numEmitterNodes: 0,
    numSwimmerNodes: 0,
    numEaterNodes: 0,
    numPredatorNodes: 0,
    numEyeNodes: 0,
    numJetNodes: 0,
    numPotentialGrabberNodes: 0,
    numAttractorNodes: 0,
    numRepulsorNodes: 0,

    rlBufferResetsDueToTopology: 0,
    nnTopologyVersion: 0,

    _applyFallbackBehaviors() {},
    ...overrides
  };
}

test('Brain prefers a pre-designated brain neuron when present', () => {
  const firstNeuron = { nodeType: NodeType.NEURON, neuronData: null };
  const designatedBrain = { nodeType: NodeType.NEURON, neuronData: { isBrain: true, hiddenLayerSize: 8 } };

  const softBody = createSoftBodyStub({
    massPoints: [firstNeuron, designatedBrain]
  });

  const brain = new Brain(softBody);

  assert.equal(brain.brainNode, designatedBrain);
  assert.equal(designatedBrain.neuronData.isBrain, true);
});


test('Brain assigns first neuron as brain when none is designated and clears others', () => {
  const firstNeuron = { nodeType: NodeType.NEURON, neuronData: null };
  const secondNeuron = { nodeType: NodeType.NEURON, neuronData: { isBrain: true, hiddenLayerSize: 8 } };
  secondNeuron.neuronData.isBrain = false;

  const softBody = createSoftBodyStub({
    massPoints: [firstNeuron, secondNeuron]
  });

  const brain = new Brain(softBody);

  assert.equal(brain.brainNode, firstNeuron);
  assert.equal(firstNeuron.neuronData.isBrain, true);
  assert.equal(secondNeuron.neuronData.isBrain, false);
});

test('Brain computes input/output vector sizes from node counters and spring count', () => {
  const brainPoint = {
    nodeType: NodeType.NEURON,
    neuronData: { isBrain: true, hiddenLayerSize: 10 }
  };

  const softBody = createSoftBodyStub({
    massPoints: [brainPoint],
    springs: [{}, {}, {}, {}],
    numEmitterNodes: 2,
    numSwimmerNodes: 3,
    numEaterNodes: 1,
    numPredatorNodes: 1,
    numEyeNodes: 2,
    numJetNodes: 2,
    numPotentialGrabberNodes: 1,
    numAttractorNodes: 1,
    numRepulsorNodes: 1
  });

  new Brain(softBody);

  const nd = brainPoint.neuronData;
  const expectedInput = config.NEURAL_INPUT_SIZE_BASE +
    (softBody.numEyeNodes * config.NEURAL_INPUTS_PER_EYE) +
    (softBody.numSwimmerNodes * config.NEURAL_INPUTS_PER_FLUID_SENSOR) +
    (softBody.numJetNodes * config.NEURAL_INPUTS_PER_FLUID_SENSOR) +
    (softBody.springs.length * config.NEURAL_INPUTS_PER_SPRING_SENSOR);

  const expectedOutput =
    (softBody.numEmitterNodes * config.NEURAL_OUTPUTS_PER_EMITTER) +
    (softBody.numSwimmerNodes * config.NEURAL_OUTPUTS_PER_SWIMMER) +
    (softBody.numEaterNodes * config.NEURAL_OUTPUTS_PER_EATER) +
    (softBody.numPredatorNodes * config.NEURAL_OUTPUTS_PER_PREDATOR) +
    (softBody.numJetNodes * config.NEURAL_OUTPUTS_PER_JET) +
    (softBody.numPotentialGrabberNodes * config.NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) +
    (softBody.numAttractorNodes * config.NEURAL_OUTPUTS_PER_ATTRACTOR) +
    (softBody.numRepulsorNodes * config.NEURAL_OUTPUTS_PER_REPULSOR);

  assert.equal(nd.inputVectorSize, expectedInput);
  assert.equal(nd.outputVectorSize, expectedOutput);
});

test('Brain.process falls back when no brain node or invalid NN tensors are present', () => {
  let fallbackCalls = 0;
  const noBrainBody = createSoftBodyStub({
    massPoints: [],
    _applyFallbackBehaviors() { fallbackCalls += 1; }
  });

  const noBrain = new Brain(noBrainBody);
  noBrain.process(0.016, null, null, null);

  const invalidPoint = {
    nodeType: NodeType.NEURON,
    pos: { x: 0, y: 0 },
    neuronData: { isBrain: true, hiddenLayerSize: 6 }
  };
  const invalidBody = createSoftBodyStub({
    massPoints: [invalidPoint],
    _applyFallbackBehaviors() { fallbackCalls += 1; }
  });

  const invalidBrain = new Brain(invalidBody);
  delete invalidPoint.neuronData.weightsIH;
  invalidBrain.process(0.016, null, null, null);

  assert.equal(fallbackCalls, 2);
});

test('Brain triggers policy update when training frame interval elapses', () => {
  const brainPoint = {
    nodeType: NodeType.NEURON,
    pos: { x: 0, y: 0 },
    neuronData: { isBrain: true, hiddenLayerSize: 6 }
  };

  const softBody = createSoftBodyStub({ massPoints: [brainPoint] });
  const brain = new Brain(softBody);

  const nd = brainPoint.neuronData;
  nd.framesSinceLastTrain = config.TRAINING_INTERVAL_FRAMES - 1;

  let updateCalls = 0;
  brain.updateBrainPolicy = () => {
    updateCalls += 1;
  };

  brain._triggerBrainPolicyUpdateIfNeeded();
  assert.equal(updateCalls, 1);
});
