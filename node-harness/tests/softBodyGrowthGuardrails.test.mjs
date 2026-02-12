import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { NodeType, MovementType } from '../../js/classes/constants.js';
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

test('triangulated growth attaches a new node to both endpoints of an existing edge', () => {
  const cfgBackup = {
    GROWTH_ENABLED: config.GROWTH_ENABLED,
    GROWTH_TRIANGULATED_PRIMITIVES_ENABLED: config.GROWTH_TRIANGULATED_PRIMITIVES_ENABLED,
    GROWTH_BASE_CHANCE_MIN: config.GROWTH_BASE_CHANCE_MIN,
    GROWTH_BASE_CHANCE_MAX: config.GROWTH_BASE_CHANCE_MAX,
    GROWTH_MIN_ENERGY_RATIO_MIN: config.GROWTH_MIN_ENERGY_RATIO_MIN,
    GROWTH_MIN_ENERGY_RATIO_MAX: config.GROWTH_MIN_ENERGY_RATIO_MAX,
    GROWTH_PLACEMENT_ATTEMPTS_PER_NODE: config.GROWTH_PLACEMENT_ATTEMPTS_PER_NODE,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  try {
    config.GROWTH_ENABLED = true;
    config.GROWTH_TRIANGULATED_PRIMITIVES_ENABLED = true;
    config.GROWTH_BASE_CHANCE_MIN = 1;
    config.GROWTH_BASE_CHANCE_MAX = 1;
    config.GROWTH_MIN_ENERGY_RATIO_MIN = 0;
    config.GROWTH_MIN_ENERGY_RATIO_MAX = 0;
    config.GROWTH_PLACEMENT_ATTEMPTS_PER_NODE = Math.max(12, Number(config.GROWTH_PLACEMENT_ATTEMPTS_PER_NODE) || 12);
    config.DYE_ECOLOGY_ENABLED = false;

    const body = new SoftBody(77, 300, 240, null, false);
    const a = body.massPoints[0];
    const b = new SoftBody(78, 310, 240, null, false).massPoints[0];

    a.nodeType = NodeType.SWIMMER;
    b.nodeType = NodeType.SWIMMER;
    a.movementType = MovementType.NEUTRAL;
    b.movementType = MovementType.NEUTRAL;
    a.radius = 1.2;
    b.radius = 1.2;
    a.pos.x = 300; a.pos.y = 240; a.prevPos.x = 300; a.prevPos.y = 240;
    b.pos.x = 310; b.pos.y = 240; b.prevPos.x = 310; b.prevPos.y = 240;

    body.massPoints = [a, b];
    body.springs = [new Spring(a, b, 500, 5, 10, false)];
    body.creatureEnergy = body.currentMaxEnergy;
    body.growthCooldownRemaining = 0;
    body.growthGenome = {
      growthChancePerTick: 1,
      minEnergyRatioToGrow: 0,
      growthCooldownTicks: 1,
      nodesPerGrowthWeights: [{ count: 1, weight: 1 }],
      newNodeTypeWeights: [{ nodeType: NodeType.SWIMMER, weight: 1 }],
      anchorNodeTypeWeights: [{ nodeType: NodeType.SWIMMER, weight: 1 }],
      distanceRangeWeights: [{ key: 'near', min: 1, max: 100, weight: 1 }],
      edgeTypeWeights: [{ type: 'soft', weight: 1 }],
      edgeStiffnessScale: 1,
      edgeDampingScale: 1,
      nodeActivationIntervalBias: 0,
      edgeActivationIntervalBias: 0,
      activationIntervalJitter: 0
    };

    const prePoints = body.massPoints.slice();
    const prePointCount = prePoints.length;
    const preSpringCount = body.springs.length;
    runtimeState.softBodyPopulation = [body];

    const didGrow = withMockedRandom(0, () => body._attemptGrowthStep(1 / 60));
    assert.equal(didGrow, true);
    assert.equal(body.massPoints.length, prePointCount + 1);

    const newPoint = body.massPoints[body.massPoints.length - 1];
    const newPointSprings = body.springs.filter((s) => s.p1 === newPoint || s.p2 === newPoint);
    assert.equal(newPointSprings.length, 2);

    const anchorPoints = newPointSprings.map((s) => (s.p1 === newPoint ? s.p2 : s.p1));
    assert.equal(anchorPoints.includes(a), true);
    assert.equal(anchorPoints.includes(b), true);

    const sharedSpring = body.springs.find(
      (s) => (s.p1 === a && s.p2 === b) || (s.p1 === b && s.p2 === a)
    );
    assert.ok(sharedSpring);

    const r0 = Number(newPointSprings[0].restLength);
    const r1 = Number(newPointSprings[1].restLength);
    assert.ok(Number.isFinite(r0) && Number.isFinite(r1));
    assert.ok(Math.abs(r0 - r1) < 1e-6);
    assert.ok((r0 + r1) + 1e-9 >= Number(sharedSpring.restLength));
    assert.equal(body.springs.length, preSpringCount + 2);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
  }
});
