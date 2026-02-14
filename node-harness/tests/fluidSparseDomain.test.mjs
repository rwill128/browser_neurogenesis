import test from 'node:test';
import assert from 'node:assert/strict';

import { FluidField } from '../../js/classes/FluidField.js';

function makeField(size = 16) {
  return new FluidField(size, 0.0005, 0.0009, 1 / 60, 1, 1);
}

test('sparse domain builds merged row spans from disjoint active tiles', () => {
  const f = makeField(16);
  f.activeTileSize = 1;
  f.activeTileHalo = 0;

  const tiles = new Map([
    ['3:3', 5],
    ['4:3', 5],
    ['10:3', 5],
    ['11:3', 5]
  ]);

  const d = f._buildSparseDomainFromTiles(tiles);
  assert.ok(d);
  const row3 = d.rowSpans.get(3);
  assert.deepEqual(row3, [
    [2, 6],
    [9, 13]
  ]);
});

test('normalizeDomain converts bounds into full row coverage spans', () => {
  const f = makeField(16);
  const d = f._normalizeDomain({ xMin: 2, xMax: 4, yMin: 5, yMax: 6 });
  assert.ok(d?.rowSpans instanceof Map);
  assert.deepEqual(d.rowSpans.get(5), [[2, 4]]);
  assert.deepEqual(d.rowSpans.get(6), [[2, 4]]);
});

test('mergeDomains preserves separated sparse spans', () => {
  const f = makeField(16);
  f.activeTileSize = 1;
  f.activeTileHalo = 0;

  const a = f._buildSparseDomainFromTiles(new Map([
    ['2:2', 5]
  ]));
  const b = f._buildSparseDomainFromTiles(new Map([
    ['9:2', 5]
  ]));

  const merged = f._mergeDomains(a, b);
  assert.ok(merged);
  assert.deepEqual(merged.rowSpans.get(2), [
    [1, 4],
    [8, 11]
  ]);
});
