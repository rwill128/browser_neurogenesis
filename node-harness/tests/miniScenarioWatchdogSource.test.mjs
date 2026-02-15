import test from 'node:test';
import assert from 'node:assert/strict';

import { makeScenario } from '../miniScenarioWatchdog.mjs';

function expectComboCounts(combo, summary) {
  if (combo === 'rigid-rigid') {
    assert.ok((summary?.rigidBodies || 0) >= 2, `expected >=2 rigid bodies for ${combo}`);
    assert.equal(summary?.softBodies || 0, 0, `expected no soft bodies for ${combo}`);
    return;
  }
  if (combo === 'soft-soft') {
    assert.equal(summary?.rigidBodies || 0, 0, `expected no rigid bodies for ${combo}`);
    assert.ok((summary?.softBodies || 0) >= 2, `expected >=2 soft bodies for ${combo}`);
    return;
  }
  assert.ok((summary?.rigidBodies || 0) >= 1, `expected >=1 rigid body for ${combo}`);
  assert.ok((summary?.softBodies || 0) >= 1, `expected >=1 soft body for ${combo}`);
}

test('mini watchdog scenarios are generated from mesh-lab field/spec pipeline', () => {
  const seeds = [0, 1, 2];
  for (const seed of seeds) {
    const scenario = makeScenario(seed);
    assert.equal(scenario.source, 'mesh-lab-field', `seed=${seed} should use mesh-lab field source`);
    assert.equal(scenario.specSummary?.schemaVersion, 'creature-spec.v2', `seed=${seed} should produce creature-spec.v2`);
    assert.ok(scenario.meshMeta && Number(scenario.meshMeta.rigidTriangles + scenario.meshMeta.softTriangles) > 0,
      `seed=${seed} should produce non-empty compiler output`);
    expectComboCounts(scenario.combo, scenario.specSummary);
  }
});
