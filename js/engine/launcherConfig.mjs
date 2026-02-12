/**
 * Launch-time configuration sampling helpers for scenario picker UX.
 *
 * The random-world sampler intentionally uses conservative sub-ranges inspired by
 * UI slider limits so generated worlds are varied but still likely to be runnable.
 */

export const RANDOM_WORLD_BOUNDS = Object.freeze({
  WORLD_WIDTH: [800, 14000],
  WORLD_HEIGHT: [600, 10000],
  CREATURE_POPULATION_FLOOR: [4, 180],
  CREATURE_POPULATION_CEILING_DELTA: [20, 420],
  CREATURE_POPULATION_CEILING_MAX: 2200,
  PARTICLE_POPULATION_FLOOR: [0, 8000],
  PARTICLE_POPULATION_CEILING_DELTA: [200, 12000],
  PARTICLE_POPULATION_CEILING_MAX: 30000,
  PARTICLES_PER_SECOND: [0, 700],
  FLUID_GRID_SIZE_CONTROL: [32, 512],
  FLUID_DIFFUSION: [0, 0.001],
  FLUID_VISCOSITY: [0, 0.0045],
  BODY_REPULSION_STRENGTH: [20, 180],
  BODY_REPULSION_RADIUS_FACTOR: [1.5, 8.5],
  GROWTH_ENERGY_COST_SCALAR: [0.3, 2.2],
  GROWTH_BASE_CHANCE_MIN: [0.002, 0.07],
  GROWTH_BASE_CHANCE_MAX_DELTA: [0.005, 0.085],
  GROWTH_BASE_CHANCE_MAX_CAP: 0.2,
  GROWTH_POP_SOFT_LIMIT_MULTIPLIER: [1.2, 4.5],
  GROWTH_POP_HARD_LIMIT_MULTIPLIER_DELTA: [0.6, 3.2],
  GROWTH_POP_HARD_LIMIT_MULTIPLIER_CAP: 9.0,
  GROWTH_SIZE_COST_MAX_MULTIPLIER: [1.5, 8.5],
  REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER: [1.2, 4.5],
  REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER_DELTA: [0.6, 3.5],
  REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER_CAP: 10.0,
  REPRO_RESOURCE_MIN_NUTRIENT: [0.2, 1.2],
  REPRO_RESOURCE_MIN_LIGHT: [0.1, 1.0]
});

function randomInRange(rng, min, max) {
  return min + rng() * (max - min);
}

function randomInt(rng, min, max) {
  const lo = Math.ceil(Math.min(min, max));
  const hi = Math.floor(Math.max(min, max));
  return Math.floor(randomInRange(rng, lo, hi + 1));
}

function round(value, digits = 4) {
  const p = 10 ** digits;
  return Math.round(value * p) / p;
}

/**
 * Build a randomized browser launch preset from bounded ranges.
 */
export function buildRandomWorldLaunchConfig(rng = Math.random) {
  const B = RANDOM_WORLD_BOUNDS;

  const worldWidth = randomInt(rng, ...B.WORLD_WIDTH);
  const worldHeight = randomInt(rng, ...B.WORLD_HEIGHT);

  const creatureFloor = randomInt(rng, ...B.CREATURE_POPULATION_FLOOR);
  const creatureCeiling = Math.min(
    B.CREATURE_POPULATION_CEILING_MAX,
    creatureFloor + randomInt(rng, ...B.CREATURE_POPULATION_CEILING_DELTA)
  );

  const particleFloor = randomInt(rng, ...B.PARTICLE_POPULATION_FLOOR);
  const particleCeiling = Math.min(
    B.PARTICLE_POPULATION_CEILING_MAX,
    particleFloor + randomInt(rng, ...B.PARTICLE_POPULATION_CEILING_DELTA)
  );

  const growthChanceMin = round(randomInRange(rng, ...B.GROWTH_BASE_CHANCE_MIN), 4);
  const growthChanceMax = Math.min(
    B.GROWTH_BASE_CHANCE_MAX_CAP,
    round(growthChanceMin + randomInRange(rng, ...B.GROWTH_BASE_CHANCE_MAX_DELTA), 4)
  );

  const growthSoftMultiplier = round(randomInRange(rng, ...B.GROWTH_POP_SOFT_LIMIT_MULTIPLIER), 2);
  const growthHardMultiplier = Math.min(
    B.GROWTH_POP_HARD_LIMIT_MULTIPLIER_CAP,
    round(growthSoftMultiplier + randomInRange(rng, ...B.GROWTH_POP_HARD_LIMIT_MULTIPLIER_DELTA), 2)
  );

  const reproSoftMultiplier = round(randomInRange(rng, ...B.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER), 2);
  const reproHardMultiplier = Math.min(
    B.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER_CAP,
    round(reproSoftMultiplier + randomInRange(rng, ...B.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER_DELTA), 2)
  );

  return {
    name: 'random_world',
    description: 'Randomized world sampled from bounded slider-inspired ranges',
    browserConfig: {
      WORLD_WIDTH: worldWidth,
      WORLD_HEIGHT: worldHeight,
      CREATURE_POPULATION_FLOOR: creatureFloor,
      CREATURE_POPULATION_CEILING: creatureCeiling,
      PARTICLE_POPULATION_FLOOR: particleFloor,
      PARTICLE_POPULATION_CEILING: particleCeiling,
      PARTICLES_PER_SECOND: randomInt(rng, ...B.PARTICLES_PER_SECOND),
      FLUID_GRID_SIZE_CONTROL: randomInt(rng, ...B.FLUID_GRID_SIZE_CONTROL),
      FLUID_DIFFUSION: round(randomInRange(rng, ...B.FLUID_DIFFUSION), 6),
      FLUID_VISCOSITY: round(randomInRange(rng, ...B.FLUID_VISCOSITY), 6),
      BODY_REPULSION_STRENGTH: round(randomInRange(rng, ...B.BODY_REPULSION_STRENGTH), 2),
      BODY_REPULSION_RADIUS_FACTOR: round(randomInRange(rng, ...B.BODY_REPULSION_RADIUS_FACTOR), 2),
      GROWTH_ENERGY_COST_SCALAR: round(randomInRange(rng, ...B.GROWTH_ENERGY_COST_SCALAR), 2),
      GROWTH_BASE_CHANCE_MIN: growthChanceMin,
      GROWTH_BASE_CHANCE_MAX: growthChanceMax,
      GROWTH_POP_SOFT_LIMIT_MULTIPLIER: growthSoftMultiplier,
      GROWTH_POP_HARD_LIMIT_MULTIPLIER: growthHardMultiplier,
      GROWTH_SIZE_COST_MAX_MULTIPLIER: round(randomInRange(rng, ...B.GROWTH_SIZE_COST_MAX_MULTIPLIER), 2),
      REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER: reproSoftMultiplier,
      REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER: reproHardMultiplier,
      REPRO_RESOURCE_MIN_NUTRIENT: round(randomInRange(rng, ...B.REPRO_RESOURCE_MIN_NUTRIENT), 3),
      REPRO_RESOURCE_MIN_LIGHT: round(randomInRange(rng, ...B.REPRO_RESOURCE_MIN_LIGHT), 3),
      AUTO_FOLLOW_CREATURE: true
    }
  };
}
