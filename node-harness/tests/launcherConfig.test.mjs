import test from 'node:test';
import assert from 'node:assert/strict';

import { buildRandomWorldLaunchConfig, RANDOM_WORLD_BOUNDS } from '../../js/engine/launcherConfig.mjs';

function createSequenceRng(values) {
  let i = 0;
  return () => {
    const v = values[i % values.length];
    i += 1;
    return v;
  };
}

test('random world launch config keeps relational constraints valid', () => {
  const rng = createSequenceRng([0.1, 0.9, 0.3, 0.7, 0.2, 0.8]);
  const cfg = buildRandomWorldLaunchConfig(rng).browserConfig;

  assert.ok(cfg.CREATURE_POPULATION_CEILING >= cfg.CREATURE_POPULATION_FLOOR);
  assert.ok(cfg.PARTICLE_POPULATION_CEILING >= cfg.PARTICLE_POPULATION_FLOOR);
  assert.ok(cfg.GROWTH_BASE_CHANCE_MAX >= cfg.GROWTH_BASE_CHANCE_MIN);
  assert.ok(cfg.GROWTH_POP_HARD_LIMIT_MULTIPLIER >= cfg.GROWTH_POP_SOFT_LIMIT_MULTIPLIER);
  assert.ok(cfg.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER >= cfg.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER);
});

test('random world launch config values remain inside configured bounds', () => {
  for (let i = 0; i < 200; i++) {
    const cfg = buildRandomWorldLaunchConfig().browserConfig;

    assert.ok(cfg.WORLD_WIDTH >= RANDOM_WORLD_BOUNDS.WORLD_WIDTH[0]);
    assert.ok(cfg.WORLD_WIDTH <= RANDOM_WORLD_BOUNDS.WORLD_WIDTH[1]);
    assert.ok(cfg.WORLD_HEIGHT >= RANDOM_WORLD_BOUNDS.WORLD_HEIGHT[0]);
    assert.ok(cfg.WORLD_HEIGHT <= RANDOM_WORLD_BOUNDS.WORLD_HEIGHT[1]);

    assert.ok(cfg.FLUID_GRID_SIZE_CONTROL >= RANDOM_WORLD_BOUNDS.FLUID_GRID_SIZE_CONTROL[0]);
    assert.ok(cfg.FLUID_GRID_SIZE_CONTROL <= RANDOM_WORLD_BOUNDS.FLUID_GRID_SIZE_CONTROL[1]);

    assert.ok(cfg.FLUID_DIFFUSION >= RANDOM_WORLD_BOUNDS.FLUID_DIFFUSION[0]);
    assert.ok(cfg.FLUID_DIFFUSION <= RANDOM_WORLD_BOUNDS.FLUID_DIFFUSION[1]);
    assert.ok(cfg.FLUID_VISCOSITY >= RANDOM_WORLD_BOUNDS.FLUID_VISCOSITY[0]);
    assert.ok(cfg.FLUID_VISCOSITY <= RANDOM_WORLD_BOUNDS.FLUID_VISCOSITY[1]);

    assert.ok(cfg.GROWTH_BASE_CHANCE_MIN >= RANDOM_WORLD_BOUNDS.GROWTH_BASE_CHANCE_MIN[0]);
    assert.ok(cfg.GROWTH_BASE_CHANCE_MIN <= RANDOM_WORLD_BOUNDS.GROWTH_BASE_CHANCE_MIN[1]);
    assert.ok(cfg.GROWTH_BASE_CHANCE_MAX <= RANDOM_WORLD_BOUNDS.GROWTH_BASE_CHANCE_MAX_CAP);

    assert.ok(cfg.GROWTH_POP_HARD_LIMIT_MULTIPLIER <= RANDOM_WORLD_BOUNDS.GROWTH_POP_HARD_LIMIT_MULTIPLIER_CAP);
    assert.ok(cfg.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER <= RANDOM_WORLD_BOUNDS.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER_CAP);
  }
});
