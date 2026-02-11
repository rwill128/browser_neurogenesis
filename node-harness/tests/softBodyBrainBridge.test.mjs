import test from 'node:test';
import assert from 'node:assert/strict';

import { SoftBody } from '../../js/classes/SoftBody.js';
import { Brain } from '../../js/classes/Brain.js';
import { NodeType } from '../../js/classes/constants.js';
import config from '../../js/config.js';

test('SoftBody.initializeBrain delegates to existing brain instance', () => {
  const body = new SoftBody(1, 120, 140, null, false);

  let initializeCalls = 0;
  const stubBrain = {
    initialize() {
      initializeCalls += 1;
    }
  };

  body.brain = stubBrain;
  body.initializeBrain();

  assert.equal(initializeCalls, 1);
  assert.equal(body.brain, stubBrain);
});

test('SoftBody.initializeBrain creates Brain when missing and marks a neuron brain node', () => {
  const body = new SoftBody(2, 200, 240, null, false);

  body.brain = null;
  body.massPoints.forEach((p) => {
    if (p.neuronData) p.neuronData.isBrain = false;
  });

  const forcedNeuron = body.massPoints[0];
  forcedNeuron.nodeType = NodeType.NEURON;
  forcedNeuron.neuronData = null;

  body.initializeBrain();

  assert.ok(body.brain instanceof Brain);
  assert.equal(body.brain.brainNode, forcedNeuron);
  assert.equal(body.brain.brainNode.neuronData.isBrain, true);
});

test('SoftBody.calculateDiscountedRewards returns reward-to-go values', () => {
  const body = new SoftBody(3, 320, 260, null, false);

  const rewards = [1, 0, 2];
  const discounted = body.calculateDiscountedRewards(rewards, 0.5);

  assert.deepEqual(discounted, [1.5, 1, 2]);
});

test('SoftBody._updateBlueprintRadiusFromCurrentPhenotype expands but does not shrink radius', () => {
  const body = new SoftBody(4, 260, 180, null, false);

  body.blueprintRadius = 0;
  body._updateBlueprintRadiusFromCurrentPhenotype();
  const grownRadius = body.blueprintRadius;

  assert.ok(grownRadius > 0);

  body.blueprintRadius = grownRadius + 25;
  body._updateBlueprintRadiusFromCurrentPhenotype();

  assert.equal(body.blueprintRadius, grownRadius + 25);
});

test('SoftBody.getAverageDamping uses config rigid damping fallback for rigid-only springs', () => {
  const body = new SoftBody(5, 160, 160, null, false);

  body.springs = [
    { isRigid: true, dampingFactor: 999 },
    { isRigid: true, dampingFactor: 333 }
  ];

  assert.equal(body.getAverageDamping(), config.RIGID_SPRING_DAMPING);
});
