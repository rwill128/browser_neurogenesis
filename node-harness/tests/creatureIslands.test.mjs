import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildCreatureInteractionIslands,
  computeIslandNeighborRadiusCells
} from '../../js/engine/creatureIslands.mjs';

function makeGrid(cols, rows) {
  return Array.from({ length: cols * rows }, () => []);
}

function cellIndex(cols, x, y) {
  return x + y * cols;
}

function addBodyPoint(grid, cols, x, y, body) {
  grid[cellIndex(cols, x, y)].push({
    type: 'softbody_point',
    bodyRef: body,
    pointRef: {}
  });
}

test('buildCreatureInteractionIslands groups nearby bodies into connected components', () => {
  const cols = 4;
  const rows = 4;
  const grid = makeGrid(cols, rows);

  const bodyA = { id: 7, isUnstable: false };
  const bodyB = { id: 2, isUnstable: false };
  const bodyC = { id: 9, isUnstable: false };

  addBodyPoint(grid, cols, 0, 0, bodyA);
  addBodyPoint(grid, cols, 1, 0, bodyB);
  addBodyPoint(grid, cols, 3, 3, bodyC);

  const islands = buildCreatureInteractionIslands({
    softBodyPopulation: [bodyA, bodyB, bodyC],
    spatialGrid: grid,
    gridCols: cols,
    gridRows: rows,
    neighborRadiusCells: 1
  });

  assert.equal(islands.bodyCount, 3);
  assert.equal(islands.islands.length, 2);
  assert.deepEqual(islands.islands[0].map((b) => b.id), [2, 7]);
  assert.deepEqual(islands.islands[1].map((b) => b.id), [9]);
});

test('buildCreatureInteractionIslands supports strict same-cell partitioning', () => {
  const cols = 4;
  const rows = 4;
  const grid = makeGrid(cols, rows);

  const bodyA = { id: 1, isUnstable: false };
  const bodyB = { id: 2, isUnstable: false };
  const bodyC = { id: 3, isUnstable: false };

  addBodyPoint(grid, cols, 0, 0, bodyA);
  addBodyPoint(grid, cols, 1, 0, bodyB);
  addBodyPoint(grid, cols, 3, 3, bodyC);

  const islands = buildCreatureInteractionIslands({
    softBodyPopulation: [bodyA, bodyB, bodyC],
    spatialGrid: grid,
    gridCols: cols,
    gridRows: rows,
    neighborRadiusCells: 0
  });

  assert.equal(islands.islands.length, 3);
  assert.deepEqual(islands.islands.map((group) => group[0].id), [1, 2, 3]);
});

test('computeIslandNeighborRadiusCells derives a conservative positive radius', () => {
  const cells = computeIslandNeighborRadiusCells({
    GRID_CELL_SIZE: 100,
    BODY_REPULSION_RADIUS_FACTOR: 5,
    ATTRACTION_RADIUS_MULTIPLIER_BASE: 0.1,
    ATTRACTION_RADIUS_MULTIPLIER_MAX_BONUS: 20,
    REPULSION_RADIUS_MULTIPLIER_BASE: 0.1,
    REPULSION_RADIUS_MULTIPLIER_MAX_BONUS: 20,
    PREDATOR_RADIUS_GENE_MAX: 14
  }, 100);

  assert.equal(Number.isInteger(cells), true);
  assert.equal(cells >= 1, true);
});
