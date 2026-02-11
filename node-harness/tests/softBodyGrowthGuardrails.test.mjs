import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { runtimeState } from '../../js/engine/runtimeState.js';

test('soft-body growth is suppressed when population exceeds hard growth limit', () => {
  const cfgBackup = {
    GROWTH_ENABLED: config.GROWTH_ENABLED,
    CREATURE_POPULATION_FLOOR: config.CREATURE_POPULATION_FLOOR,
    CREATURE_POPULATION_CEILING: config.CREATURE_POPULATION_CEILING,
    GROWTH_POP_SOFT_LIMIT_MULTIPLIER: config.GROWTH_POP_SOFT_LIMIT_MULTIPLIER,
    GROWTH_POP_HARD_LIMIT_MULTIPLIER: config.GROWTH_POP_HARD_LIMIT_MULTIPLIER,
    GROWTH_BASE_CHANCE_MIN: config.GROWTH_BASE_CHANCE_MIN,
    GROWTH_BASE_CHANCE_MAX: config.GROWTH_BASE_CHANCE_MAX
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation,
    mutationStats: runtimeState.mutationStats
  };

  try {
    config.GROWTH_ENABLED = true;
    config.CREATURE_POPULATION_FLOOR = 10;
    config.CREATURE_POPULATION_CEILING = 100;
    config.GROWTH_POP_SOFT_LIMIT_MULTIPLIER = 2;
    config.GROWTH_POP_HARD_LIMIT_MULTIPLIER = 4;
    config.GROWTH_BASE_CHANCE_MIN = 1;
    config.GROWTH_BASE_CHANCE_MAX = 1;

    runtimeState.mutationStats = {};

    const body = new SoftBody(1, 100, 100, null, false);
    body.creatureEnergy = body.currentMaxEnergy;
    body.growthCooldownRemaining = 0;

    // Exceed hard limit: floor(10) * hard(4) = 40 -> use 45.
    runtimeState.softBodyPopulation = Array.from({ length: 45 }, () => body);

    const before = body.growthSuppressedByPopulation;
    const didGrow = body._attemptGrowthStep(1 / 60);

    assert.equal(didGrow, false);
    assert.equal(body.growthSuppressedByPopulation, before + 1);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
    runtimeState.mutationStats = runtimeBackup.mutationStats;
  }
});
