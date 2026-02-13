import test from 'node:test';
import assert from 'node:assert/strict';

import { getScenario } from '../scenarios.mjs';
import { RealWorld } from '../realWorld.mjs';

test('snapshot normalizes instability telemetry totals against reason counts', () => {
  const world = new RealWorld(getScenario('baseline'), 1234);

  world.worldState.instabilityTelemetry = {
    totalRemoved: 3,
    totalPhysicsRemoved: 0,
    totalNonPhysicsRemoved: 3,
    totalUnknownRemoved: 0,
    removedByReason: {
      age_limit: 5,
      energy_depleted: 2
    },
    removedByPhysicsKind: {},
    removedByBirthOrigin: { floor_spawn: 7 },
    removedByLifecycleStage: { floor_spawn: 7 },
    recentDeaths: [{ id: 1 }, { id: 2 }, { id: 3 }],
    sampledDiagnostics: [{ id: 'a' }]
  };

  const snap = world.snapshot();
  const t = snap.instabilityTelemetry;

  assert.equal(t.totalRemoved, 7);
  assert.equal(t.totalPhysicsRemoved, 0);
  assert.equal(t.totalNonPhysicsRemoved, 7);
  assert.equal(t.totalUnknownRemoved, 0);
  assert.equal((t.removedByReason?.age_limit || 0), 5);
  assert.equal((t.removedByReason?.energy_depleted || 0), 2);
});
