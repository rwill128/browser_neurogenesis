import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { runtimeState } from '../../js/engine/runtimeState.js';

function withMockedRandom(value, fn) {
  const original = Math.random;
  Math.random = () => value;
  try {
    return fn();
  } finally {
    Math.random = original;
  }
}

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

test('soft-body growth increments max-point suppression telemetry at cap', () => {
  const cfgBackup = {
    GROWTH_ENABLED: config.GROWTH_ENABLED,
    GROWTH_MAX_POINTS_PER_CREATURE: config.GROWTH_MAX_POINTS_PER_CREATURE
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  try {
    config.GROWTH_ENABLED = true;

    const body = new SoftBody(2, 100, 100, null, false);
    body.creatureEnergy = body.currentMaxEnergy;
    body.growthCooldownRemaining = 0;

    config.GROWTH_MAX_POINTS_PER_CREATURE = body.massPoints.length;
    runtimeState.softBodyPopulation = [body];

    const before = body.growthSuppressedByMaxPoints;
    const didGrow = body._attemptGrowthStep(1 / 60);

    assert.equal(didGrow, false);
    assert.equal(body.growthSuppressedByMaxPoints, before + 1);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
  }
});

test('soft-body growth increments chance-roll suppression telemetry when probability roll misses', () => {
  const cfgBackup = {
    GROWTH_ENABLED: config.GROWTH_ENABLED,
    GROWTH_BASE_CHANCE_MIN: config.GROWTH_BASE_CHANCE_MIN,
    GROWTH_BASE_CHANCE_MAX: config.GROWTH_BASE_CHANCE_MAX,
    GROWTH_MIN_ENERGY_RATIO_MIN: config.GROWTH_MIN_ENERGY_RATIO_MIN,
    GROWTH_MIN_ENERGY_RATIO_MAX: config.GROWTH_MIN_ENERGY_RATIO_MAX
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  try {
    config.GROWTH_ENABLED = true;
    config.GROWTH_BASE_CHANCE_MIN = 0.2;
    config.GROWTH_BASE_CHANCE_MAX = 0.2;
    config.GROWTH_MIN_ENERGY_RATIO_MIN = 0;
    config.GROWTH_MIN_ENERGY_RATIO_MAX = 0;

    const body = new SoftBody(3, 100, 100, null, false);
    body.creatureEnergy = body.currentMaxEnergy;
    body.growthCooldownRemaining = 0;

    runtimeState.softBodyPopulation = [body];

    const before = body.growthSuppressedByChanceRoll;
    const didGrow = withMockedRandom(0.99, () => body._attemptGrowthStep(1 / 60));

    assert.equal(didGrow, false);
    assert.equal(body.growthSuppressedByChanceRoll, before + 1);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
  }
});

test('soft-body growth increments placement suppression telemetry when no candidate placement succeeds', () => {
  const cfgBackup = {
    GROWTH_ENABLED: config.GROWTH_ENABLED,
    GROWTH_BASE_CHANCE_MIN: config.GROWTH_BASE_CHANCE_MIN,
    GROWTH_BASE_CHANCE_MAX: config.GROWTH_BASE_CHANCE_MAX,
    GROWTH_MIN_ENERGY_RATIO_MIN: config.GROWTH_MIN_ENERGY_RATIO_MIN,
    GROWTH_MIN_ENERGY_RATIO_MAX: config.GROWTH_MIN_ENERGY_RATIO_MAX,
    IS_WORLD_WRAPPING: config.IS_WORLD_WRAPPING,
    WORLD_WIDTH: config.WORLD_WIDTH,
    WORLD_HEIGHT: config.WORLD_HEIGHT
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  try {
    config.GROWTH_ENABLED = true;
    config.GROWTH_BASE_CHANCE_MIN = 1;
    config.GROWTH_BASE_CHANCE_MAX = 1;
    config.GROWTH_MIN_ENERGY_RATIO_MIN = 0;
    config.GROWTH_MIN_ENERGY_RATIO_MAX = 0;
    config.IS_WORLD_WRAPPING = false;
    config.WORLD_WIDTH = 1;
    config.WORLD_HEIGHT = 1;

    const body = new SoftBody(4, 0.5, 0.5, null, false);
    body.creatureEnergy = body.currentMaxEnergy;
    body.growthCooldownRemaining = 0;

    runtimeState.softBodyPopulation = [body];

    const before = body.growthSuppressedByPlacement;
    const didGrow = withMockedRandom(0, () => body._attemptGrowthStep(1 / 60));

    assert.equal(didGrow, false);
    assert.equal(body.growthSuppressedByPlacement, before + 1);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
  }
});

test('dye mismatch can reduce growth chance and increment dye suppression telemetry', () => {
  const cfgBackup = {
    GROWTH_ENABLED: config.GROWTH_ENABLED,
    GROWTH_BASE_CHANCE_MIN: config.GROWTH_BASE_CHANCE_MIN,
    GROWTH_BASE_CHANCE_MAX: config.GROWTH_BASE_CHANCE_MAX,
    GROWTH_MIN_ENERGY_RATIO_MIN: config.GROWTH_MIN_ENERGY_RATIO_MIN,
    GROWTH_MIN_ENERGY_RATIO_MAX: config.GROWTH_MIN_ENERGY_RATIO_MAX,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED,
    DYE_GROWTH_EFFECT_WEIGHT: config.DYE_GROWTH_EFFECT_WEIGHT,
    DYE_EFFECT_MIN_SCALE: config.DYE_EFFECT_MIN_SCALE,
    DYE_EFFECT_MAX_SCALE: config.DYE_EFFECT_MAX_SCALE
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation,
    fluidField: runtimeState.fluidField
  };

  try {
    config.GROWTH_ENABLED = true;
    config.GROWTH_BASE_CHANCE_MIN = 1;
    config.GROWTH_BASE_CHANCE_MAX = 1;
    config.GROWTH_MIN_ENERGY_RATIO_MIN = 0;
    config.GROWTH_MIN_ENERGY_RATIO_MAX = 0;
    config.DYE_ECOLOGY_ENABLED = true;
    config.DYE_GROWTH_EFFECT_WEIGHT = 1.5;
    config.DYE_EFFECT_MIN_SCALE = 0.01;
    config.DYE_EFFECT_MAX_SCALE = 2.0;

    const body = new SoftBody(55, 100, 100, null, false);
    body.creatureEnergy = body.currentMaxEnergy;
    body.growthCooldownRemaining = 0;
    body.dyePreferredHue = 0.0; // red
    body.dyeHueTolerance = 0.02;
    body.dyeResponseGain = 1.5;
    body.dyeResponseSign = 1;

    runtimeState.fluidField = {
      scaleX: 1,
      scaleY: 1,
      IX: () => 0,
      densityR: new Float32Array([0]),
      densityG: new Float32Array([255]),
      densityB: new Float32Array([255])
    };
    runtimeState.softBodyPopulation = [body];

    const beforeChance = body.growthSuppressedByChanceRoll;
    const beforeDye = body.growthSuppressedByDye;
    const didGrow = withMockedRandom(0.9, () => body._attemptGrowthStep(1 / 60));

    assert.equal(didGrow, false);
    assert.equal(body.growthSuppressedByChanceRoll, beforeChance + 1);
    assert.equal(body.growthSuppressedByDye > beforeDye, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
    runtimeState.fluidField = runtimeBackup.fluidField;
  }
});
