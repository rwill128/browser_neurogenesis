import test from 'node:test';
import assert from 'node:assert/strict';

import { SoftBody } from '../../js/classes/SoftBody.js';
import { NodeType } from '../../js/classes/constants.js';
import config from '../../js/config.js';

test('SoftBody activation interval gene sanitizer clamps to config bounds', () => {
  const body = new SoftBody(901, 120, 120, null, false);

  const min = Math.max(1, Math.floor(config.ACTUATION_INTERVAL_GENE_MIN));
  const max = Math.max(min, Math.floor(config.ACTUATION_INTERVAL_GENE_MAX));

  assert.equal(body._sanitizeActivationIntervalGene(-999), min);
  assert.equal(body._sanitizeActivationIntervalGene(999), max);
  assert.equal(body._sanitizeActivationIntervalGene(NaN), min);
});

test('point actuation cooldowns are tracked independently per channel', () => {
  const body = new SoftBody(902, 140, 140, null, false);

  const point = {
    nodeType: NodeType.EMITTER,
    activationIntervalGene: 2,
    actuationCooldownByChannel: {}
  };

  // First access per channel should evaluate.
  assert.equal(body._shouldEvaluatePointActuation(point, 'node'), true);
  assert.equal(body._shouldEvaluatePointActuation(point, 'grabber'), true);

  // Node channel is now cooling down, while grabber has its own independent timer.
  assert.equal(body._shouldEvaluatePointActuation(point, 'node'), false);
});

test('growth genome carries activation-interval bias controls for future growth additions', () => {
  const body = new SoftBody(903, 180, 180, null, false);
  const genome = body._createRandomGrowthGenome();

  assert.equal(Number.isFinite(Number(genome.nodeActivationIntervalBias)), true);
  assert.equal(Number.isFinite(Number(genome.edgeActivationIntervalBias)), true);
  assert.equal(Number.isFinite(Number(genome.activationIntervalJitter)), true);

  const mutated = body._mutateGrowthGenomeFromParent(genome).genome;
  assert.ok(mutated.nodeActivationIntervalBias >= -3 && mutated.nodeActivationIntervalBias <= 3);
  assert.ok(mutated.edgeActivationIntervalBias >= -3 && mutated.edgeActivationIntervalBias <= 3);
  assert.ok(mutated.activationIntervalJitter >= 0 && mutated.activationIntervalJitter <= 3);
});

test('exported blueprints preserve activation interval genes for points and springs', () => {
  const body = new SoftBody(904, 180, 180, null, false);
  const blueprint = body.exportBlueprint();

  assert.ok(Array.isArray(blueprint.blueprintPoints));
  assert.ok(Array.isArray(blueprint.blueprintSprings));
  assert.ok(blueprint.blueprintPoints.length > 0);
  assert.ok(blueprint.blueprintSprings.length > 0);

  for (const p of blueprint.blueprintPoints) {
    assert.equal(Number.isFinite(Number(p.activationIntervalGene)), true);
  }

  for (const s of blueprint.blueprintSprings) {
    assert.equal(Number.isFinite(Number(s.activationIntervalGene)), true);
  }
});
