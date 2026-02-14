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
  assert.ok(Number(t.momentumActiveTiles) >= 1);
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
