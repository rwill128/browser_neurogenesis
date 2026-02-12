import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { NodeType, MovementType } from '../../js/classes/constants.js';
import { runtimeState } from '../../js/engine/runtimeState.js';

function makeGrid(cols, rows) {
  return Array.from({ length: cols * rows }, () => []);
}

function gridIndex(x, y, cell, cols, rows) {
  const gx = Math.max(0, Math.min(cols - 1, Math.floor(x / cell)));
  const gy = Math.max(0, Math.min(rows - 1, Math.floor(y / cell)));
  return gx + gy * cols;
}

test('eater nodes consume nearby particles and increase creature energy', () => {
  const cfgBackup = {
    GRID_CELL_SIZE: config.GRID_CELL_SIZE,
    GRID_COLS: config.GRID_COLS,
    GRID_ROWS: config.GRID_ROWS,
    ENERGY_PER_PARTICLE: config.ENERGY_PER_PARTICLE,
    EATING_RADIUS_MULTIPLIER_BASE: config.EATING_RADIUS_MULTIPLIER_BASE,
    EATING_RADIUS_MULTIPLIER_MAX_BONUS: config.EATING_RADIUS_MULTIPLIER_MAX_BONUS,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation,
    fluidField: runtimeState.fluidField,
    mutationStats: runtimeState.mutationStats
  };

  try {
    config.GRID_CELL_SIZE = 20;
    config.GRID_COLS = 8;
    config.GRID_ROWS = 8;
    config.ENERGY_PER_PARTICLE = 30;
    config.EATING_RADIUS_MULTIPLIER_BASE = 1.0;
    config.EATING_RADIUS_MULTIPLIER_MAX_BONUS = 0;
    config.DYE_ECOLOGY_ENABLED = false;

    runtimeState.softBodyPopulation = [];
    runtimeState.fluidField = null;
    runtimeState.mutationStats = {};

    const body = new SoftBody(101, 50, 50, null, false);
    const eater = body.massPoints[0];
    eater.nodeType = NodeType.EATER;
    eater.currentExertionLevel = 1;
    eater.radius = 8;
    eater.isGrabbing = false;
    eater.movementType = MovementType.NEUTRAL;

    body.massPoints = [eater];
    body.springs = [];
    body.currentMaxEnergy = 500;
    body.creatureEnergy = 100;
    body.energyGainedFromEating = 0;

    const spatialGrid = makeGrid(config.GRID_COLS, config.GRID_ROWS);
    const particle = {
      pos: { x: eater.pos.x + 3, y: eater.pos.y },
      life: 1,
      isEaten: false
    };

    const cellIdx = gridIndex(particle.pos.x, particle.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
    spatialGrid[cellIdx].push({ type: 'particle', particleRef: particle });

    body.setSpatialGrid(spatialGrid);
    body.setParticles([particle]);
    body.setNutrientField(null);

    body._finalizeUpdateAndCheckStability(1 / 60);

    assert.equal(particle.isEaten, true);
    assert.equal(particle.life, 0);
    assert.equal(body.energyGainedFromEating > 0, true);
    assert.equal(body.creatureEnergy > 100, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
    runtimeState.fluidField = runtimeBackup.fluidField;
    runtimeState.mutationStats = runtimeBackup.mutationStats;
  }
});

test('predator nodes sap nearby foreign body energy and gain predation energy', () => {
  const cfgBackup = {
    GRID_CELL_SIZE: config.GRID_CELL_SIZE,
    GRID_COLS: config.GRID_COLS,
    GRID_ROWS: config.GRID_ROWS,
    PREDATION_RADIUS_MULTIPLIER_BASE: config.PREDATION_RADIUS_MULTIPLIER_BASE,
    PREDATION_RADIUS_MULTIPLIER_MAX_BONUS: config.PREDATION_RADIUS_MULTIPLIER_MAX_BONUS,
    ENERGY_SAPPED_PER_PREDATION_BASE: config.ENERGY_SAPPED_PER_PREDATION_BASE,
    ENERGY_SAPPED_PER_PREDATION_MAX_BONUS: config.ENERGY_SAPPED_PER_PREDATION_MAX_BONUS,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation,
    fluidField: runtimeState.fluidField,
    mutationStats: runtimeState.mutationStats
  };

  try {
    config.GRID_CELL_SIZE = 20;
    config.GRID_COLS = 8;
    config.GRID_ROWS = 8;
    config.PREDATION_RADIUS_MULTIPLIER_BASE = 1.0;
    config.PREDATION_RADIUS_MULTIPLIER_MAX_BONUS = 0;
    config.ENERGY_SAPPED_PER_PREDATION_BASE = 12;
    config.ENERGY_SAPPED_PER_PREDATION_MAX_BONUS = 0;
    config.DYE_ECOLOGY_ENABLED = false;

    runtimeState.fluidField = null;
    runtimeState.mutationStats = {};

    const predatorBody = new SoftBody(201, 60, 60, null, false);
    const predatorPoint = predatorBody.massPoints[0];
    predatorPoint.nodeType = NodeType.PREDATOR;
    predatorPoint.currentExertionLevel = 1;
    predatorPoint.radius = 10;
    predatorPoint.isGrabbing = false;
    predatorPoint.movementType = MovementType.NEUTRAL;
    predatorPoint.pos.x = 60;
    predatorPoint.pos.y = 60;
    predatorPoint.prevPos.x = 60;
    predatorPoint.prevPos.y = 60;

    predatorBody.massPoints = [predatorPoint];
    predatorBody.springs = [];
    predatorBody.currentMaxEnergy = 500;
    predatorBody.creatureEnergy = 100;
    predatorBody.energyGainedFromPredation = 0;

    const preyBody = new SoftBody(202, 63, 60, null, false);
    const preyPoint = preyBody.massPoints[0];
    preyPoint.nodeType = NodeType.EATER;
    preyPoint.radius = 8;
    preyPoint.isGrabbing = false;
    preyPoint.movementType = MovementType.NEUTRAL;
    preyPoint.pos.x = 63;
    preyPoint.pos.y = 60;
    preyPoint.prevPos.x = 63;
    preyPoint.prevPos.y = 60;
    preyBody.massPoints = [preyPoint];
    preyBody.springs = [];
    preyBody.currentMaxEnergy = 500;
    preyBody.creatureEnergy = 120;

    const spatialGrid = makeGrid(config.GRID_COLS, config.GRID_ROWS);
    const idx = gridIndex(predatorPoint.pos.x, predatorPoint.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
    spatialGrid[idx].push({ type: 'softbody_point', pointRef: predatorPoint, bodyRef: predatorBody });
    spatialGrid[idx].push({ type: 'softbody_point', pointRef: preyPoint, bodyRef: preyBody });

    predatorBody.setSpatialGrid(spatialGrid);
    preyBody.setSpatialGrid(spatialGrid);
    runtimeState.softBodyPopulation = [predatorBody, preyBody];

    predatorBody._finalizeUpdateAndCheckStability(1 / 60);

    assert.equal(predatorBody.energyGainedFromPredation > 0, true);
    assert.equal(predatorBody.creatureEnergy > 100, true);
    assert.equal(preyBody.creatureEnergy < 120, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
    runtimeState.fluidField = runtimeBackup.fluidField;
    runtimeState.mutationStats = runtimeBackup.mutationStats;
  }
});

function createMockFluidField(size = 16) {
  const calls = {
    addDensity: [],
    addVelocity: []
  };
  return {
    scaleX: 1,
    scaleY: 1,
    size,
    Vx: new Float32Array(size).fill(0),
    Vy: new Float32Array(size).fill(0),
    IX: () => 0,
    addDensity(x, y, r, g, b, s) {
      calls.addDensity.push({ x, y, r, g, b, s });
    },
    addVelocity(x, y, vx, vy) {
      calls.addVelocity.push({ x, y, vx, vy });
    },
    calls
  };
}

test('different predator nodes can each sap the same prey body once per tick', () => {
  const cfgBackup = {
    GRID_CELL_SIZE: config.GRID_CELL_SIZE,
    GRID_COLS: config.GRID_COLS,
    GRID_ROWS: config.GRID_ROWS,
    PREDATION_RADIUS_MULTIPLIER_BASE: config.PREDATION_RADIUS_MULTIPLIER_BASE,
    PREDATION_RADIUS_MULTIPLIER_MAX_BONUS: config.PREDATION_RADIUS_MULTIPLIER_MAX_BONUS,
    ENERGY_SAPPED_PER_PREDATION_BASE: config.ENERGY_SAPPED_PER_PREDATION_BASE,
    ENERGY_SAPPED_PER_PREDATION_MAX_BONUS: config.ENERGY_SAPPED_PER_PREDATION_MAX_BONUS,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };
  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  try {
    config.GRID_CELL_SIZE = 20;
    config.GRID_COLS = 8;
    config.GRID_ROWS = 8;
    config.PREDATION_RADIUS_MULTIPLIER_BASE = 1.0;
    config.PREDATION_RADIUS_MULTIPLIER_MAX_BONUS = 0;
    config.ENERGY_SAPPED_PER_PREDATION_BASE = 10;
    config.ENERGY_SAPPED_PER_PREDATION_MAX_BONUS = 0;
    config.DYE_ECOLOGY_ENABLED = false;

    const predator = new SoftBody(301, 60, 60, null, false);
    const pA = predator.massPoints[0];
    const pB = new SoftBody(302, 62, 60, null, false).massPoints[0];
    pA.nodeType = NodeType.PREDATOR;
    pB.nodeType = NodeType.PREDATOR;
    pA.currentExertionLevel = 1;
    pB.currentExertionLevel = 1;
    pA.movementType = MovementType.NEUTRAL;
    pB.movementType = MovementType.NEUTRAL;
    pA.isGrabbing = false;
    pB.isGrabbing = false;
    pA.radius = 10;
    pB.radius = 10;
    pA.pos.x = 60;
    pA.pos.y = 60;
    pA.prevPos.x = 60;
    pA.prevPos.y = 60;
    pB.pos.x = 61;
    pB.pos.y = 60;
    pB.prevPos.x = 61;
    pB.prevPos.y = 60;

    predator.massPoints = [pA, pB];
    predator.springs = [];
    predator.currentMaxEnergy = 500;
    predator.creatureEnergy = 100;

    const prey = new SoftBody(303, 63, 60, null, false);
    const preyPoint = prey.massPoints[0];
    preyPoint.nodeType = NodeType.EATER;
    preyPoint.radius = 8;
    preyPoint.movementType = MovementType.NEUTRAL;
    preyPoint.isGrabbing = false;
    preyPoint.pos.x = 63;
    preyPoint.pos.y = 60;
    preyPoint.prevPos.x = 63;
    preyPoint.prevPos.y = 60;
    prey.massPoints = [preyPoint];
    prey.springs = [];
    prey.creatureEnergy = 100;

    const grid = makeGrid(config.GRID_COLS, config.GRID_ROWS);
    const idx = gridIndex(pA.pos.x, pA.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
    grid[idx].push({ type: 'softbody_point', pointRef: pA, bodyRef: predator });
    grid[idx].push({ type: 'softbody_point', pointRef: pB, bodyRef: predator });
    grid[idx].push({ type: 'softbody_point', pointRef: preyPoint, bodyRef: prey });

    predator.setSpatialGrid(grid);
    prey.setSpatialGrid(grid);
    runtimeState.softBodyPopulation = [predator, prey];

    const beforePrey = prey.creatureEnergy;
    predator._finalizeUpdateAndCheckStability(1 / 60);

    const lost = beforePrey - prey.creatureEnergy;
    assert.equal(lost > config.ENERGY_SAPPED_PER_PREDATION_BASE, true);
    assert.equal(lost <= (config.ENERGY_SAPPED_PER_PREDATION_BASE * 2) + 1e-9, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
  }
});

test('eater gain is scaled by local nutrient value when nutrient field is available', () => {
  const cfgBackup = {
    GRID_CELL_SIZE: config.GRID_CELL_SIZE,
    GRID_COLS: config.GRID_COLS,
    GRID_ROWS: config.GRID_ROWS,
    ENERGY_PER_PARTICLE: config.ENERGY_PER_PARTICLE,
    EATING_RADIUS_MULTIPLIER_BASE: config.EATING_RADIUS_MULTIPLIER_BASE,
    EATING_RADIUS_MULTIPLIER_MAX_BONUS: config.EATING_RADIUS_MULTIPLIER_MAX_BONUS,
    MIN_NUTRIENT_VALUE: config.MIN_NUTRIENT_VALUE,
    globalNutrientMultiplier: config.globalNutrientMultiplier,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };
  const runtimeBackup = {
    fluidField: runtimeState.fluidField,
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  try {
    config.GRID_CELL_SIZE = 20;
    config.GRID_COLS = 8;
    config.GRID_ROWS = 8;
    config.ENERGY_PER_PARTICLE = 20;
    config.EATING_RADIUS_MULTIPLIER_BASE = 1.0;
    config.EATING_RADIUS_MULTIPLIER_MAX_BONUS = 0;
    config.MIN_NUTRIENT_VALUE = 0.1;
    config.globalNutrientMultiplier = 1.5;
    config.DYE_ECOLOGY_ENABLED = false;

    runtimeState.fluidField = {
      scaleX: 1,
      scaleY: 1,
      IX: () => 0
    };

    const body = new SoftBody(401, 50, 50, null, false);
    const eater = body.massPoints[0];
    eater.nodeType = NodeType.EATER;
    eater.currentExertionLevel = 1;
    eater.radius = 8;
    eater.movementType = MovementType.NEUTRAL;
    eater.isGrabbing = false;

    body.massPoints = [eater];
    body.springs = [];
    body.currentMaxEnergy = 1000;
    body.creatureEnergy = 100;

    const nutrientField = new Float32Array([2.0]);
    body.setNutrientField(nutrientField);

    const particle = { pos: { x: eater.pos.x + 2, y: eater.pos.y }, life: 1, isEaten: false };
    const grid = makeGrid(config.GRID_COLS, config.GRID_ROWS);
    const idx = gridIndex(particle.pos.x, particle.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
    grid[idx].push({ type: 'particle', particleRef: particle });
    body.setSpatialGrid(grid);

    body._finalizeUpdateAndCheckStability(1 / 60);

    const expectedGain = config.ENERGY_PER_PARTICLE * Math.max(config.MIN_NUTRIENT_VALUE, 2.0 * config.globalNutrientMultiplier);
    assert.equal(Math.abs(body.energyGainedFromEating - expectedGain) < 1e-6, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.fluidField = runtimeBackup.fluidField;
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
  }
});

test('emitter and jet nodes inject density/velocity into fluid, and swimmer actuation moves the point', () => {
  const cfgBackup = {
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };
  const runtimeBackup = {
    fluidField: runtimeState.fluidField
  };

  try {
    config.DYE_ECOLOGY_ENABLED = false;
    const fluid = createMockFluidField(32);
    runtimeState.fluidField = fluid;

    // Emitter + Jet sanity
    const body = new SoftBody(501, 40, 40, null, false);
    const emitter = body.massPoints[0];
    const jet = new SoftBody(502, 40, 40, null, false).massPoints[0];

    emitter.nodeType = NodeType.EMITTER;
    emitter.currentExertionLevel = 1;
    emitter.dyeColor = [255, 10, 20];
    emitter.movementType = MovementType.FIXED;

    jet.nodeType = NodeType.JET;
    jet.currentExertionLevel = 1;
    jet.jetData = { currentMagnitude: 2, currentAngle: 0 };
    jet.maxEffectiveJetVelocity = 100;
    jet.movementType = MovementType.FIXED;

    body.massPoints = [emitter, jet];
    body.springs = [];
    body._performPhysicalUpdates(1 / 60, fluid);

    assert.equal(fluid.calls.addDensity.length > 0, true);
    assert.equal(fluid.calls.addVelocity.length > 0, true);

    // Swimmer sanity
    const swimmerBody = new SoftBody(503, 45, 45, null, false);
    const swimmer = swimmerBody.massPoints[0];
    swimmer.nodeType = NodeType.SWIMMER;
    swimmer.swimmerActuation = { magnitude: 3, angle: 0 };
    swimmer.movementType = MovementType.NEUTRAL;
    swimmer.isGrabbing = false;

    swimmerBody.massPoints = [swimmer];
    swimmerBody.springs = [];

    const beforeX = swimmer.pos.x;
    swimmerBody._performPhysicalUpdates(1 / 60, fluid);
    assert.equal(swimmer.pos.x > beforeX, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.fluidField = runtimeBackup.fluidField;
  }
});

test('eater does not consume particles outside eating radius', () => {
  const cfgBackup = {
    GRID_CELL_SIZE: config.GRID_CELL_SIZE,
    GRID_COLS: config.GRID_COLS,
    GRID_ROWS: config.GRID_ROWS,
    ENERGY_PER_PARTICLE: config.ENERGY_PER_PARTICLE,
    EATING_RADIUS_MULTIPLIER_BASE: config.EATING_RADIUS_MULTIPLIER_BASE,
    EATING_RADIUS_MULTIPLIER_MAX_BONUS: config.EATING_RADIUS_MULTIPLIER_MAX_BONUS,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };

  try {
    config.GRID_CELL_SIZE = 20;
    config.GRID_COLS = 8;
    config.GRID_ROWS = 8;
    config.ENERGY_PER_PARTICLE = 25;
    config.EATING_RADIUS_MULTIPLIER_BASE = 1.0;
    config.EATING_RADIUS_MULTIPLIER_MAX_BONUS = 0;
    config.DYE_ECOLOGY_ENABLED = false;

    const body = new SoftBody(601, 50, 50, null, false);
    const eater = body.massPoints[0];
    eater.nodeType = NodeType.EATER;
    eater.currentExertionLevel = 1;
    eater.radius = 6;
    eater.movementType = MovementType.NEUTRAL;
    eater.isGrabbing = false;

    body.massPoints = [eater];
    body.springs = [];
    body.creatureEnergy = 100;
    body.energyGainedFromEating = 0;

    const farParticle = { pos: { x: eater.pos.x + 80, y: eater.pos.y + 80 }, life: 1, isEaten: false };
    const grid = makeGrid(config.GRID_COLS, config.GRID_ROWS);
    const idx = gridIndex(farParticle.pos.x, farParticle.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
    grid[idx].push({ type: 'particle', particleRef: farParticle });
    body.setSpatialGrid(grid);

    body._finalizeUpdateAndCheckStability(1 / 60);

    assert.equal(farParticle.isEaten, false);
    assert.equal(farParticle.life, 1);
    assert.equal(body.energyGainedFromEating, 0);
    assert.equal(body.creatureEnergy, 100);
  } finally {
    Object.assign(config, cfgBackup);
  }
});

test('predator does not sap prey outside predation radius', () => {
  const cfgBackup = {
    GRID_CELL_SIZE: config.GRID_CELL_SIZE,
    GRID_COLS: config.GRID_COLS,
    GRID_ROWS: config.GRID_ROWS,
    PREDATION_RADIUS_MULTIPLIER_BASE: config.PREDATION_RADIUS_MULTIPLIER_BASE,
    PREDATION_RADIUS_MULTIPLIER_MAX_BONUS: config.PREDATION_RADIUS_MULTIPLIER_MAX_BONUS,
    ENERGY_SAPPED_PER_PREDATION_BASE: config.ENERGY_SAPPED_PER_PREDATION_BASE,
    ENERGY_SAPPED_PER_PREDATION_MAX_BONUS: config.ENERGY_SAPPED_PER_PREDATION_MAX_BONUS,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };

  try {
    config.GRID_CELL_SIZE = 20;
    config.GRID_COLS = 8;
    config.GRID_ROWS = 8;
    config.PREDATION_RADIUS_MULTIPLIER_BASE = 1.0;
    config.PREDATION_RADIUS_MULTIPLIER_MAX_BONUS = 0;
    config.ENERGY_SAPPED_PER_PREDATION_BASE = 10;
    config.ENERGY_SAPPED_PER_PREDATION_MAX_BONUS = 0;
    config.DYE_ECOLOGY_ENABLED = false;

    const predator = new SoftBody(701, 40, 40, null, false);
    const predPoint = predator.massPoints[0];
    predPoint.nodeType = NodeType.PREDATOR;
    predPoint.currentExertionLevel = 1;
    predPoint.radius = 6;
    predPoint.movementType = MovementType.NEUTRAL;
    predPoint.isGrabbing = false;

    predator.massPoints = [predPoint];
    predator.springs = [];
    predator.creatureEnergy = 100;
    predator.energyGainedFromPredation = 0;

    const prey = new SoftBody(702, 180, 180, null, false);
    const preyPoint = prey.massPoints[0];
    preyPoint.nodeType = NodeType.EATER;
    preyPoint.radius = 6;
    preyPoint.movementType = MovementType.NEUTRAL;
    preyPoint.isGrabbing = false;

    prey.massPoints = [preyPoint];
    prey.springs = [];
    prey.creatureEnergy = 120;

    const grid = makeGrid(config.GRID_COLS, config.GRID_ROWS);
    const predIdx = gridIndex(predPoint.pos.x, predPoint.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
    const preyIdx = gridIndex(preyPoint.pos.x, preyPoint.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
    grid[predIdx].push({ type: 'softbody_point', pointRef: predPoint, bodyRef: predator });
    grid[preyIdx].push({ type: 'softbody_point', pointRef: preyPoint, bodyRef: prey });

    predator.setSpatialGrid(grid);

    predator._finalizeUpdateAndCheckStability(1 / 60);

    assert.equal(predator.energyGainedFromPredation, 0);
    assert.equal(predator.creatureEnergy, 100);
    assert.equal(prey.creatureEnergy, 120);
  } finally {
    Object.assign(config, cfgBackup);
  }
});

test('same particle is not double-counted by multiple eater nodes in one tick', () => {
  const cfgBackup = {
    GRID_CELL_SIZE: config.GRID_CELL_SIZE,
    GRID_COLS: config.GRID_COLS,
    GRID_ROWS: config.GRID_ROWS,
    ENERGY_PER_PARTICLE: config.ENERGY_PER_PARTICLE,
    EATING_RADIUS_MULTIPLIER_BASE: config.EATING_RADIUS_MULTIPLIER_BASE,
    EATING_RADIUS_MULTIPLIER_MAX_BONUS: config.EATING_RADIUS_MULTIPLIER_MAX_BONUS,
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED
  };

  try {
    config.GRID_CELL_SIZE = 20;
    config.GRID_COLS = 8;
    config.GRID_ROWS = 8;
    config.ENERGY_PER_PARTICLE = 30;
    config.EATING_RADIUS_MULTIPLIER_BASE = 1.0;
    config.EATING_RADIUS_MULTIPLIER_MAX_BONUS = 0;
    config.DYE_ECOLOGY_ENABLED = false;

    const body = new SoftBody(801, 50, 50, null, false);
    const eaterA = body.massPoints[0];
    const eaterB = new SoftBody(802, 52, 50, null, false).massPoints[0];
    eaterA.nodeType = NodeType.EATER;
    eaterB.nodeType = NodeType.EATER;
    eaterA.currentExertionLevel = 1;
    eaterB.currentExertionLevel = 1;
    eaterA.radius = 8;
    eaterB.radius = 8;
    eaterA.movementType = MovementType.NEUTRAL;
    eaterB.movementType = MovementType.NEUTRAL;
    eaterA.isGrabbing = false;
    eaterB.isGrabbing = false;

    body.massPoints = [eaterA, eaterB];
    body.springs = [];
    body.currentMaxEnergy = 1000;
    body.creatureEnergy = 100;
    body.energyGainedFromEating = 0;

    const particle = { pos: { x: eaterA.pos.x + 2, y: eaterA.pos.y }, life: 1, isEaten: false };
    const grid = makeGrid(config.GRID_COLS, config.GRID_ROWS);
    const idx = gridIndex(particle.pos.x, particle.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
    grid[idx].push({ type: 'particle', particleRef: particle });
    body.setSpatialGrid(grid);

    body._finalizeUpdateAndCheckStability(1 / 60);

    assert.equal(particle.isEaten, true);
    assert.equal(body.energyGainedFromEating, config.ENERGY_PER_PARTICLE);
    assert.equal(body.creatureEnergy, 100 + config.ENERGY_PER_PARTICLE);
  } finally {
    Object.assign(config, cfgBackup);
  }
});
