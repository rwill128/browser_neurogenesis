import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { MovementType, NodeType } from '../../js/classes/constants.js';

function makeGrid(cols, rows) {
  return Array.from({ length: cols * rows }, () => []);
}

function gridIndex(x, y, cell, cols, rows) {
  const gx = Math.max(0, Math.min(cols - 1, Math.floor(x / cell)));
  const gy = Math.max(0, Math.min(rows - 1, Math.floor(y / cell)));
  return gx + gy * cols;
}

function distance(a, b) {
  const dx = a.pos.x - b.pos.x;
  const dy = a.pos.y - b.pos.y;
  return Math.hypot(dx, dy);
}

test('impermeable soft-vs-soft contact resolves tight overlap within tolerance', () => {
  const cfgBackup = {
    GRID_CELL_SIZE: config.GRID_CELL_SIZE,
    GRID_COLS: config.GRID_COLS,
    GRID_ROWS: config.GRID_ROWS,
    BODY_REPULSION_STRENGTH: config.BODY_REPULSION_STRENGTH,
    BODY_REPULSION_RADIUS_FACTOR: config.BODY_REPULSION_RADIUS_FACTOR,
    WORLD_WIDTH: config.WORLD_WIDTH,
    WORLD_HEIGHT: config.WORLD_HEIGHT,
    KILL_ON_OUT_OF_BOUNDS: config.KILL_ON_OUT_OF_BOUNDS,
    PHYSICS_MOTION_GUARD_ENABLED: config.PHYSICS_MOTION_GUARD_ENABLED,
  };

  try {
    config.GRID_CELL_SIZE = 20;
    config.GRID_COLS = 20;
    config.GRID_ROWS = 20;
    config.BODY_REPULSION_STRENGTH = 140;
    config.BODY_REPULSION_RADIUS_FACTOR = 1.0;
    config.WORLD_WIDTH = 400;
    config.WORLD_HEIGHT = 400;
    config.KILL_ON_OUT_OF_BOUNDS = false;
    config.PHYSICS_MOTION_GUARD_ENABLED = true;

    const bodyA = new SoftBody(9501, 200, 200, null, false);
    const bodyB = new SoftBody(9502, 206, 200, null, false);

    const pA = bodyA.massPoints[0];
    const pB = bodyB.massPoints[0];

    pA.radius = 6;
    pB.radius = 6;
    pA.nodeType = NodeType.SWIMMER;
    pB.nodeType = NodeType.SWIMMER;
    pA.movementType = MovementType.NEUTRAL;
    pB.movementType = MovementType.NEUTRAL;

    pA.pos.x = 200; pA.pos.y = 200; pA.prevPos.x = 200; pA.prevPos.y = 200;
    pB.pos.x = 206; pB.pos.y = 200; pB.prevPos.x = 206; pB.prevPos.y = 200;

    bodyA.massPoints = [pA];
    bodyB.massPoints = [pB];
    bodyA.springs = [];
    bodyB.springs = [];

    const interactionRadius = (pA.radius + pB.radius) * config.BODY_REPULSION_RADIUS_FACTOR;
    const beforeDist = distance(pA, pB);

    for (let step = 0; step < 24; step++) {
      // Integrate currently accumulated forces.
      bodyA._performPhysicalUpdates(1 / 60, null);
      bodyB._performPhysicalUpdates(1 / 60, null);

      // Rebuild world grid and run inter-body contact logic for the next integration step.
      const grid = makeGrid(config.GRID_COLS, config.GRID_ROWS);
      const idxA = gridIndex(pA.pos.x, pA.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
      const idxB = gridIndex(pB.pos.x, pB.pos.y, config.GRID_CELL_SIZE, config.GRID_COLS, config.GRID_ROWS);
      grid[idxA].push({ type: 'softbody_point', pointRef: pA, bodyRef: bodyA });
      grid[idxB].push({ type: 'softbody_point', pointRef: pB, bodyRef: bodyB });

      bodyA.setSpatialGrid(grid);
      bodyB.setSpatialGrid(grid);
      bodyA._finalizeUpdateAndCheckStability(1 / 60);
      bodyB._finalizeUpdateAndCheckStability(1 / 60);
    }

    // Apply one final integration pass so the last contact impulse is realized in position space.
    bodyA._performPhysicalUpdates(1 / 60, null);
    bodyB._performPhysicalUpdates(1 / 60, null);

    const afterDist = distance(pA, pB);
    const overlapAfter = Math.max(0, interactionRadius - afterDist);

    assert.equal(bodyA.isUnstable, false);
    assert.equal(bodyB.isUnstable, false);
    assert.ok(afterDist > beforeDist, `expected separation increase; before=${beforeDist.toFixed(3)} after=${afterDist.toFixed(3)}`);
    assert.ok(
      overlapAfter <= 0.25,
      `expected tight-space soft-body overlap to resolve below tolerance; overlap=${overlapAfter.toFixed(3)} interactionRadius=${interactionRadius.toFixed(3)}`,
    );
  } finally {
    Object.assign(config, cfgBackup);
  }
});
