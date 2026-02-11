import test from 'node:test';
import assert from 'node:assert/strict';

import {
  computeDensityFertilityScale,
  evaluateResourceCoupling,
  applyReproductionResourceDebit
} from '../../js/engine/reproductionControls.mjs';

test('density fertility scale declines under global+local crowding', () => {
  const low = computeDensityFertilityScale({
    population: 100,
    floor: 100,
    ceiling: 10000,
    globalSoftMultiplier: 2,
    globalHardMultiplier: 4,
    localNeighbors: 2,
    localSoftNeighbors: 6,
    localHardNeighbors: 18
  });

  const high = computeDensityFertilityScale({
    population: 390,
    floor: 100,
    ceiling: 10000,
    globalSoftMultiplier: 2,
    globalHardMultiplier: 4,
    localNeighbors: 16,
    localSoftNeighbors: 6,
    localHardNeighbors: 18
  });

  assert.equal(low.scale, 1);
  assert.ok(high.scale < low.scale);
  assert.ok(high.scale >= 0);
});

test('resource coupling blocks reproduction when nutrient/light are below minima', () => {
  const blocked = evaluateResourceCoupling({
    nutrientValue: 0.3,
    lightValue: 0.2,
    minNutrient: 0.55,
    minLight: 0.35
  });

  const allowed = evaluateResourceCoupling({
    nutrientValue: 0.8,
    lightValue: 0.6,
    minNutrient: 0.55,
    minLight: 0.35
  });

  assert.equal(blocked.allow, false);
  assert.ok(blocked.fertilityScale < 1);
  assert.equal(allowed.allow, true);
  assert.equal(allowed.fertilityScale, 1);
});

test('resource debit applies bounded subtraction to both fields', () => {
  const nutrient = new Float32Array([1.0]);
  const light = new Float32Array([0.7]);

  const out = applyReproductionResourceDebit({
    nutrientField: nutrient,
    lightField: light,
    index: 0,
    nutrientDebit: 0.2,
    lightDebit: 0.15,
    nutrientMin: 0,
    lightMin: 0
  });

  assert.equal(Number(nutrient[0].toFixed(6)), 0.8);
  assert.equal(Number(light[0].toFixed(6)), 0.55);
  assert.equal(Number(out.nutrientAfter.toFixed(6)), 0.8);
  assert.equal(Number(out.lightAfter.toFixed(6)), 0.55);
});
