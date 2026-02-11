import { createEnvironmentFields, createLightField, createNutrientField, createViscosityField } from './environmentFields.js';

function defaultRangeRandom(rng, min, max) {
  return min + rng() * (max - min);
}

function attachWorldRefsToBody(body, worldState) {
  body.setNutrientField(worldState.nutrientField);
  body.setLightField(worldState.lightField);
  body.setParticles(worldState.particles);
  body.setSpatialGrid(worldState.spatialGrid);
}

export function initializeSpatialGrid(worldState, config) {
  const total = Math.max(1, config.GRID_COLS * config.GRID_ROWS);
  worldState.spatialGrid = new Array(total);
  for (let i = 0; i < total; i++) {
    worldState.spatialGrid[i] = [];
  }
  return worldState.spatialGrid;
}

export function initializePopulation(worldState, {
  config,
  SoftBodyClass,
  count = config.CREATURE_POPULATION_FLOOR,
  spawnMargin = 50,
  rng = Math.random
} = {}) {
  if (!SoftBodyClass) throw new Error('initializePopulation requires SoftBodyClass');

  worldState.softBodyPopulation = [];
  worldState.nextSoftBodyId = 0;

  for (let i = 0; i < count; i++) {
    const x = defaultRangeRandom(rng, spawnMargin, config.WORLD_WIDTH - spawnMargin);
    const y = defaultRangeRandom(rng, spawnMargin, config.WORLD_HEIGHT - spawnMargin);
    const body = new SoftBodyClass(worldState.nextSoftBodyId++, x, y, null);
    attachWorldRefsToBody(body, worldState);
    worldState.softBodyPopulation.push(body);
  }

  return worldState.softBodyPopulation;
}

export function initializeParticles(worldState, {
  config,
  ParticleClass = null,
  count = 0,
  rng = Math.random
} = {}) {
  worldState.particles = [];

  if (ParticleClass && worldState.fluidField && count > 0) {
    for (let i = 0; i < count; i++) {
      worldState.particles.push(
        new ParticleClass(
          rng() * config.WORLD_WIDTH,
          rng() * config.WORLD_HEIGHT,
          worldState.fluidField
        )
      );
    }
  }

  return worldState.particles;
}

export function initializeEnvironmentMaps(worldState, {
  config,
  size = Math.round(config.FLUID_GRID_SIZE_CONTROL),
  rng = Math.random
} = {}) {
  const env = createEnvironmentFields({ size, random: rng });
  worldState.nutrientField = env.nutrientField;
  worldState.lightField = env.lightField;
  worldState.viscosityField = env.viscosityField;
  return env;
}

export function initializeNutrientMap(worldState, {
  config,
  size = Math.round(config.FLUID_GRID_SIZE_CONTROL),
  rng = Math.random
} = {}) {
  worldState.nutrientField = createNutrientField(size, rng);
  return worldState.nutrientField;
}

export function initializeLightMap(worldState, {
  config,
  size = Math.round(config.FLUID_GRID_SIZE_CONTROL),
  rng = Math.random
} = {}) {
  worldState.lightField = createLightField(size, rng);
  return worldState.lightField;
}

export function initializeViscosityMap(worldState, {
  config,
  size = Math.round(config.FLUID_GRID_SIZE_CONTROL),
  rng = Math.random
} = {}) {
  worldState.viscosityField = createViscosityField(size, rng);
  return worldState.viscosityField;
}
