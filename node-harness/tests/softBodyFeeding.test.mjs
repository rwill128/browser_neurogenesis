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
    EATING_RADIUS_MULTIPLIER_MAX_BONUS: config.EATING_RADIUS_MULTIPLIER_MAX_BONUS
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
    ENERGY_SAPPED_PER_PREDATION_MAX_BONUS: config.ENERGY_SAPPED_PER_PREDATION_MAX_BONUS
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

    runtimeState.fluidField = null;
    runtimeState.mutationStats = {};

    const predatorBody = new SoftBody(201, 60, 60, null, false);
    const predatorPoint = predatorBody.massPoints[0];
    predatorPoint.nodeType = NodeType.PREDATOR;
    predatorPoint.currentExertionLevel = 1;
    predatorPoint.radius = 10;
    predatorPoint.isGrabbing = false;
    predatorPoint.movementType = MovementType.NEUTRAL;

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
