import test from 'node:test';
import assert from 'node:assert/strict';

import { FluidField } from '../../js/classes/FluidField.js';

function makeField(size = 16) {
  return new FluidField(size, 0.0005, 0.0009, 1 / 60, 1, 1);
}

test('active tile telemetry tracks activity from density/velocity writes', () => {
  const f = makeField(16);

  f.addDensity(4, 4, 255, 0, 0, 50);
  f.addVelocity(4, 4, 1.2, -0.4);
  f.step();

  const t = f.getActiveTileTelemetry();
  assert.ok(Number(t.totalTiles) >= 1);
  assert.ok(Number(t.carrierActiveTiles) >= 1);
  assert.ok(Number(t.momentumTilesTotal) >= 1);
  assert.ok(Number(t.carrierTouchedTiles) >= 1);
  assert.ok(Number(t.carrierPct) > 0);
});

test('active tiles can be seeded from body centers', () => {
  const f = makeField(16);
  const bodies = [
    {
      isUnstable: false,
      getAveragePosition() {
        return { x: 3, y: 3 };
      }
    }
  ];

  f.seedCarrierTilesFromBodies(bodies);
  f.step();
  const t = f.getActiveTileTelemetry();
  assert.ok(Number(t.carrierActiveTiles) >= 1);
});

test('velocity splats ignore non-finite impulses but still clamp oversized finite impulses', () => {
  const f = makeField(16);
  f.maxVelComponent = 2.5;

  f.addVelocity(8, 8, Infinity, NaN);
  let idx = f.IX(8, 8);
  assert.equal(f.Vx[idx], 0, 'Infinity/NaN splat should be ignored, not clamped into a max kick');
  assert.equal(f.Vy[idx], 0, 'Infinity/NaN splat should be ignored, not clamped into a max kick');

  f.addVelocity(8, 8, 99, -99);
  idx = f.IX(8, 8);
  assert.equal(f.Vx[idx], 2.5, 'finite oversized splat should still clamp to max velocity');
  assert.equal(f.Vy[idx], -2.5, 'finite oversized splat should still clamp to min velocity');
});

test('boundary update sanitizes NaN/Inf velocity contamination without killing interior motion', () => {
  const baseline = makeField(24);
  baseline.maxVelComponent = 4;
  baseline.addVelocity(12, 12, 1.8, 0.6);
  baseline.step();
  const centerIdx = baseline.IX(12, 12);
  const baselineCenterSpeed = Math.hypot(baseline.Vx[centerIdx], baseline.Vy[centerIdx]);

  const f = makeField(24);
  f.maxVelComponent = 4;
  f.addVelocity(12, 12, 1.8, 0.6);
  f.Vx[f.IX(0, 12)] = Number.NaN;
  f.Vx[f.IX(23, 12)] = Number.POSITIVE_INFINITY;
  f.Vy[f.IX(0, 12)] = Number.NEGATIVE_INFINITY;
  f.step();

  for (let i = 0; i < f.Vx.length; i++) {
    assert.ok(Number.isFinite(f.Vx[i]), `Vx[${i}] should be finite`);
    assert.ok(Number.isFinite(f.Vy[i]), `Vy[${i}] should be finite`);
    assert.ok(Math.abs(f.Vx[i]) <= f.maxVelComponent + 1e-6, `Vx[${i}] should remain clamped`);
    assert.ok(Math.abs(f.Vy[i]) <= f.maxVelComponent + 1e-6, `Vy[${i}] should remain clamped`);
  }

  const centerSpeed = Math.hypot(f.Vx[centerIdx], f.Vy[centerIdx]);
  assert.ok(centerSpeed >= baselineCenterSpeed * 0.75,
    `interior should stay close to baseline energy (${centerSpeed} vs baseline ${baselineCenterSpeed})`);
});
