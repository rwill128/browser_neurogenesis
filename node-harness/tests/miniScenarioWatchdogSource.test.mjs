import test from 'node:test';
import assert from 'node:assert/strict';

import { makeScenario, sanitizeFlowSample } from '../miniScenarioWatchdog.mjs';

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

test('sanitizeFlowSample clamps non-finite and extreme fluid samples deterministically', () => {
  const nanFlow = sanitizeFlowSample({ vx: Number.NaN, vy: Number.POSITIVE_INFINITY }, 2.5);
  assert.deepEqual(nanFlow, { vx: 0, vy: 0, speed: 0 });

  const capped = sanitizeFlowSample({ vx: 30, vy: 40 }, 2.5);
  assert.ok(Math.abs(capped.speed - 2.5) < 1e-9);
  assert.ok(Math.abs(capped.vx - 1.5) < 1e-9);
  assert.ok(Math.abs(capped.vy - 2.0) < 1e-9);
});
