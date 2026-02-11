import test from 'node:test';
import assert from 'node:assert/strict';

import {
  computeGrowthPopulationThrottle,
  computeGrowthSizeCostMultiplier
} from '../../js/engine/growthControls.mjs';

test('population throttle is full-strength below soft limit', () => {
  const out = computeGrowthPopulationThrottle({
    population: 120,
    floor: 100,
    ceiling: 10000,
    softLimitMultiplier: 2,
    hardLimitMultiplier: 4,
    minThrottleScale: 0.05
  });

  assert.equal(out.allowGrowth, true);
  assert.equal(out.scale, 1);
  assert.equal(out.softLimit, 200);
});

test('population throttle blocks growth at/above hard limit', () => {
  const out = computeGrowthPopulationThrottle({
    population: 401,
    floor: 100,
    ceiling: 10000,
    softLimitMultiplier: 2,
    hardLimitMultiplier: 4,
    minThrottleScale: 0.05
  });

  assert.equal(out.allowGrowth, false);
  assert.equal(out.scale, 0);
  assert.equal(out.hardLimit, 400);
});

test('population throttle decays smoothly between soft and hard limits', () => {
  const out = computeGrowthPopulationThrottle({
    population: 300,
    floor: 100,
    ceiling: 10000,
    softLimitMultiplier: 2,
    hardLimitMultiplier: 4,
    minThrottleScale: 0.05
  });

  assert.equal(out.allowGrowth, true);
  assert.ok(out.scale < 1 && out.scale > 0.05);
});

test('size-cost multiplier is bounded and monotonic', () => {
  const small = computeGrowthSizeCostMultiplier({
    currentPoints: 10,
    maxPoints: 100,
    exponent: 1.15,
    maxMultiplier: 4
  });
  const medium = computeGrowthSizeCostMultiplier({
    currentPoints: 50,
    maxPoints: 100,
    exponent: 1.15,
    maxMultiplier: 4
  });
  const large = computeGrowthSizeCostMultiplier({
    currentPoints: 100,
    maxPoints: 100,
    exponent: 1.15,
    maxMultiplier: 4
  });

  assert.ok(small >= 1 && small <= 4);
  assert.ok(medium >= small);
  assert.ok(large >= medium);
  assert.equal(large, 4);
});
