import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { getScenario } from '../scenarios.mjs';
import { RealWorld } from '../realWorld.mjs';

/**
 * End-to-end save/load regression on a live-stepping real world.
 *
 * This specifically exercises growth-active runs before save, then verifies
 * the snapshot can be loaded without topology-count mismatches.
 */
test('runtime can step, save, and load after growth-active steps', () => {
  const backup = {
    GROWTH_ENABLED: config.GROWTH_ENABLED,
    GROWTH_BASE_CHANCE_MIN: config.GROWTH_BASE_CHANCE_MIN,
    GROWTH_BASE_CHANCE_MAX: config.GROWTH_BASE_CHANCE_MAX,
    GROWTH_MIN_ENERGY_RATIO_MIN: config.GROWTH_MIN_ENERGY_RATIO_MIN,
    GROWTH_MIN_ENERGY_RATIO_MAX: config.GROWTH_MIN_ENERGY_RATIO_MAX,
    GROWTH_COOLDOWN_MIN: config.GROWTH_COOLDOWN_MIN,
    GROWTH_COOLDOWN_MAX: config.GROWTH_COOLDOWN_MAX,
    GROWTH_ENERGY_COST_SCALAR: config.GROWTH_ENERGY_COST_SCALAR,
    GROWTH_MAX_POINTS_PER_CREATURE: config.GROWTH_MAX_POINTS_PER_CREATURE
  };

  try {
    config.GROWTH_ENABLED = true;
    config.GROWTH_BASE_CHANCE_MIN = 0.4;
    config.GROWTH_BASE_CHANCE_MAX = 0.6;
    config.GROWTH_MIN_ENERGY_RATIO_MIN = 0.15;
    config.GROWTH_MIN_ENERGY_RATIO_MAX = 0.25;
    config.GROWTH_COOLDOWN_MIN = 1;
    config.GROWTH_COOLDOWN_MAX = 4;
    config.GROWTH_ENERGY_COST_SCALAR = 0.2;
    config.GROWTH_MAX_POINTS_PER_CREATURE = 80;

    const scenario = getScenario('baseline');
    const world = new RealWorld(scenario, 2026);

    const maxSteps = 120;
    let saveStep = -1;
    for (let i = 0; i < maxSteps; i++) {
      world.step(scenario.dt);

      const hasGrownTopology = world.softBodyPopulation.some((b) =>
        Array.isArray(b.massPoints) &&
        Array.isArray(b.blueprintPoints) &&
        b.massPoints.length > b.blueprintPoints.length
      );

      if (hasGrownTopology) {
        saveStep = i + 1;
        break;
      }

      if (world.softBodyPopulation.length === 0) {
        break;
      }
    }

    assert.ok(saveStep > 0, 'expected at least one creature to grow before save');

    const snapshot = world.saveStateSnapshot({
      source: 'runtime-save-load-test',
      step: saveStep
    });

    assert.ok(snapshot.world.softBodies.length > 0, 'expected at least one saved body');

    const divergent = snapshot.world.softBodies.find((b) => {
      const reproCount = b?.blueprint?.blueprintPoints?.length || 0;
      const phenotypeCount = b?.phenotypeBlueprint?.blueprintPoints?.length || 0;
      return phenotypeCount > reproCount;
    });
    assert.ok(divergent, 'expected at least one snapshot body with phenotype > reproductive blueprint count');

    const restored = new RealWorld(scenario, 9090);
    assert.doesNotThrow(() => {
      restored.loadStateSnapshot(snapshot);
    });

    assert.equal(restored.softBodyPopulation.length, world.softBodyPopulation.length);

    const savedTotalPoints = snapshot.world.softBodies.reduce((sum, b) => sum + (b.massPoints?.length || 0), 0);
    const restoredTotalPoints = restored.softBodyPopulation.reduce((sum, b) => sum + (b.massPoints?.length || 0), 0);
    assert.equal(restoredTotalPoints, savedTotalPoints);

    const restoredDivergent = restored.softBodyPopulation.find((b) => b.id === divergent.id);
    assert.ok(restoredDivergent, 'expected divergent body id to survive load');
    assert.equal(restoredDivergent.massPoints.length, divergent.massPoints.length);
    assert.equal(restoredDivergent.blueprintPoints.length, divergent.blueprint.blueprintPoints.length);
    assert.notEqual(restoredDivergent.blueprintPoints.length, restoredDivergent.massPoints.length);
  } finally {
    Object.assign(config, backup);
  }
});
