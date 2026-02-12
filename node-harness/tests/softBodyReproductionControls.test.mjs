import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { runtimeState } from '../../js/engine/runtimeState.js';

function withPatchedMathRandom(value, fn) {
  const original = Math.random;
  Math.random = () => value;
  try {
    return fn();
  } finally {
    Math.random = original;
  }
}

test('reproduction is resource-gated when local nutrient/light are below minima', () => {
  const cfgBackup = {
    canCreaturesReproduceGlobally: config.canCreaturesReproduceGlobally,
    REPRO_RESOURCE_MIN_NUTRIENT: config.REPRO_RESOURCE_MIN_NUTRIENT,
    REPRO_RESOURCE_MIN_LIGHT: config.REPRO_RESOURCE_MIN_LIGHT
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation,
    fluidField: runtimeState.fluidField,
    mutationStats: runtimeState.mutationStats
  };

  try {
    config.canCreaturesReproduceGlobally = true;
    config.REPRO_RESOURCE_MIN_NUTRIENT = 0.6;
    config.REPRO_RESOURCE_MIN_LIGHT = 0.4;

    runtimeState.mutationStats = {};

    const body = new SoftBody(1, 100, 100, null, false);
    body.canReproduce = true;
    body.creatureEnergy = body.currentMaxEnergy;
    body.failedReproductionCooldown = 0;
    body.numOffspring = 1;

    body.nutrientField = new Float32Array([0.1]);
    body.lightField = new Float32Array([0.1]);
    runtimeState.fluidField = { scaleX: 1, scaleY: 1, IX: () => 0 };
    runtimeState.softBodyPopulation = [body];

    const offspring = body.reproduce();
    assert.equal(Array.isArray(offspring), true);
    assert.equal(offspring.length, 0);
    assert.equal(body.reproductionSuppressedByResources, 1);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
    runtimeState.fluidField = runtimeBackup.fluidField;
    runtimeState.mutationStats = runtimeBackup.mutationStats;
  }
});

test('density-dependent fertility can suppress reproduction probabilistically', () => {
  const cfgBackup = {
    canCreaturesReproduceGlobally: config.canCreaturesReproduceGlobally,
    REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER: config.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER,
    REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER: config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER,
    REPRO_FERTILITY_GLOBAL_MIN_SCALE: config.REPRO_FERTILITY_GLOBAL_MIN_SCALE,
    REPRO_LOCAL_DENSITY_RADIUS: config.REPRO_LOCAL_DENSITY_RADIUS,
    REPRO_FERTILITY_LOCAL_SOFT_NEIGHBORS: config.REPRO_FERTILITY_LOCAL_SOFT_NEIGHBORS,
    REPRO_FERTILITY_LOCAL_HARD_NEIGHBORS: config.REPRO_FERTILITY_LOCAL_HARD_NEIGHBORS,
    REPRO_FERTILITY_LOCAL_MIN_SCALE: config.REPRO_FERTILITY_LOCAL_MIN_SCALE,
    REPRO_MIN_FERTILITY_SCALE: config.REPRO_MIN_FERTILITY_SCALE,
    CREATURE_POPULATION_FLOOR: config.CREATURE_POPULATION_FLOOR,
    CREATURE_POPULATION_CEILING: config.CREATURE_POPULATION_CEILING,
    REPRO_RESOURCE_MIN_NUTRIENT: config.REPRO_RESOURCE_MIN_NUTRIENT,
    REPRO_RESOURCE_MIN_LIGHT: config.REPRO_RESOURCE_MIN_LIGHT
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation,
    fluidField: runtimeState.fluidField,
    mutationStats: runtimeState.mutationStats
  };

  try {
    config.canCreaturesReproduceGlobally = true;
    config.CREATURE_POPULATION_FLOOR = 10;
    config.CREATURE_POPULATION_CEILING = 1000;
    config.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER = 1.2;
    config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER = 1.4;
    config.REPRO_FERTILITY_GLOBAL_MIN_SCALE = 0.05;
    config.REPRO_LOCAL_DENSITY_RADIUS = 1000;
    config.REPRO_FERTILITY_LOCAL_SOFT_NEIGHBORS = 2;
    config.REPRO_FERTILITY_LOCAL_HARD_NEIGHBORS = 3;
    config.REPRO_FERTILITY_LOCAL_MIN_SCALE = 0.05;
    config.REPRO_MIN_FERTILITY_SCALE = 0.05;
    config.REPRO_RESOURCE_MIN_NUTRIENT = 0.1;
    config.REPRO_RESOURCE_MIN_LIGHT = 0.1;

    runtimeState.mutationStats = {};

    const body = new SoftBody(2, 100, 100, null, false);
    body.canReproduce = true;
    body.creatureEnergy = body.currentMaxEnergy;
    body.failedReproductionCooldown = 0;
    body.numOffspring = 1;

    body.nutrientField = new Float32Array([1]);
    body.lightField = new Float32Array([1]);
    runtimeState.fluidField = { scaleX: 1, scaleY: 1, IX: () => 0 };

    const crowd = [];
    for (let i = 0; i < 30; i++) {
      crowd.push({ isUnstable: false, getAveragePosition: () => ({ x: 100, y: 100 }) });
    }
    runtimeState.softBodyPopulation = [body, ...crowd];

    const offspring = withPatchedMathRandom(0.99, () => body.reproduce());
    assert.equal(offspring.length, 0);
    assert.equal(body.reproductionSuppressedByFertilityRoll > 0, true);
    assert.equal(body.reproductionSuppressedByDensity > 0, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
    runtimeState.fluidField = runtimeBackup.fluidField;
    runtimeState.mutationStats = runtimeBackup.mutationStats;
  }
});

test('dye mismatch can suppress reproduction fertility scale', () => {
  const cfgBackup = {
    canCreaturesReproduceGlobally: config.canCreaturesReproduceGlobally,
    REPRO_MIN_FERTILITY_SCALE: config.REPRO_MIN_FERTILITY_SCALE,
    REPRO_RESOURCE_MIN_NUTRIENT: config.REPRO_RESOURCE_MIN_NUTRIENT,
    REPRO_RESOURCE_MIN_LIGHT: config.REPRO_RESOURCE_MIN_LIGHT,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED,
    DYE_REPRO_EFFECT_WEIGHT: config.DYE_REPRO_EFFECT_WEIGHT,
    DYE_EFFECT_MIN_SCALE: config.DYE_EFFECT_MIN_SCALE,
    DYE_EFFECT_MAX_SCALE: config.DYE_EFFECT_MAX_SCALE
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation,
    fluidField: runtimeState.fluidField,
    mutationStats: runtimeState.mutationStats
  };

  try {
    config.canCreaturesReproduceGlobally = true;
    config.REPRO_MIN_FERTILITY_SCALE = 0;
    config.REPRO_RESOURCE_MIN_NUTRIENT = 0.1;
    config.REPRO_RESOURCE_MIN_LIGHT = 0.1;
    config.DYE_ECOLOGY_ENABLED = true;
    config.DYE_REPRO_EFFECT_WEIGHT = 1.5;
    config.DYE_EFFECT_MIN_SCALE = 0.01;
    config.DYE_EFFECT_MAX_SCALE = 2.0;

    runtimeState.mutationStats = {};

    const body = new SoftBody(22, 100, 100, null, false);
    body.canReproduce = true;
    body.creatureEnergy = body.currentMaxEnergy;
    body.failedReproductionCooldown = 0;
    body.numOffspring = 1;
    body.dyePreferredHue = 0.0; // red
    body.dyeHueTolerance = 0.02;
    body.dyeResponseGain = 1.5;
    body.dyeResponseSign = 1;

    body.nutrientField = new Float32Array([1]);
    body.lightField = new Float32Array([1]);
    runtimeState.fluidField = {
      scaleX: 1,
      scaleY: 1,
      IX: () => 0,
      densityR: new Float32Array([0]),
      densityG: new Float32Array([255]),
      densityB: new Float32Array([255])
    };
    runtimeState.softBodyPopulation = [body];

    const offspring = withPatchedMathRandom(0.9, () => body.reproduce());
    assert.equal(offspring.length, 0);
    assert.equal(body.reproductionSuppressedByDye > 0, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
    runtimeState.fluidField = runtimeBackup.fluidField;
    runtimeState.mutationStats = runtimeBackup.mutationStats;
  }
});
