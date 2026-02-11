import { createEnvironmentFields, createLightField, createNutrientField, createViscosityField } from './environmentFields.js';
import { withRandomSource } from './randomScope.mjs';
import { resolveConfigViews } from './configViews.mjs';

function defaultRangeRandom(rng, min, max) {
  return min + rng() * (max - min);
}

function attachWorldRefsToBody(body, worldState) {
  body.setNutrientField(worldState.nutrientField);
  body.setLightField(worldState.lightField);
  body.setParticles(worldState.particles);
  body.setSpatialGrid(worldState.spatialGrid);
}

export function initializeSpatialGrid(worldState, configOrViews) {
  const { runtime: config } = resolveConfigViews(configOrViews);
  const total = Math.max(1, config.GRID_COLS * config.GRID_ROWS);
  worldState.spatialGrid = new Array(total);
  for (let i = 0; i < total; i++) {
    worldState.spatialGrid[i] = [];
  }
  return worldState.spatialGrid;
}

export function initializePopulation(worldState, {
  config,
  configViews = null,
  SoftBodyClass,
  count = (configViews?.runtime || config).CREATURE_POPULATION_FLOOR,
  spawnMargin = 50,
  rng = Math.random
} = {}) {
  const { runtime: runtimeConfig } = resolveConfigViews(configViews || config);
  if (!SoftBodyClass) throw new Error('initializePopulation requires SoftBodyClass');

  worldState.softBodyPopulation = [];
  worldState.nextSoftBodyId = 0;

  for (let i = 0; i < count; i++) {
    const x = defaultRangeRandom(rng, spawnMargin, runtimeConfig.WORLD_WIDTH - spawnMargin);
    const y = defaultRangeRandom(rng, spawnMargin, runtimeConfig.WORLD_HEIGHT - spawnMargin);
    const body = withRandomSource(rng, () => new SoftBodyClass(worldState.nextSoftBodyId++, x, y, null));
    attachWorldRefsToBody(body, worldState);
    worldState.softBodyPopulation.push(body);
  }

  return worldState.softBodyPopulation;
}

export function initializeParticles(worldState, {
  config,
  configViews = null,
  ParticleClass = null,
  count = 0,
  rng = Math.random
} = {}) {
  const { runtime: runtimeConfig } = resolveConfigViews(configViews || config);
  worldState.particles = [];

  if (ParticleClass && worldState.fluidField && count > 0) {
    for (let i = 0; i < count; i++) {
      worldState.particles.push(
        withRandomSource(
          rng,
          () => new ParticleClass(
            rng() * runtimeConfig.WORLD_WIDTH,
            rng() * runtimeConfig.WORLD_HEIGHT,
            worldState.fluidField
          )
        )
      );
    }
  }

  return worldState.particles;
}

export function initializeEnvironmentMaps(worldState, {
  config,
  configViews = null,
  size = null,
  rng = Math.random
} = {}) {
  const { runtime: runtimeConfig } = resolveConfigViews(configViews || config);
  const resolvedSize = size ?? Math.round(runtimeConfig.FLUID_GRID_SIZE_CONTROL);
  const env = createEnvironmentFields({ size: resolvedSize, random: rng });
  worldState.nutrientField = env.nutrientField;
  worldState.lightField = env.lightField;
  worldState.viscosityField = env.viscosityField;
  return env;
}

export function initializeNutrientMap(worldState, {
  config,
  configViews = null,
  size = null,
  rng = Math.random
} = {}) {
  const { runtime: runtimeConfig } = resolveConfigViews(configViews || config);
  const resolvedSize = size ?? Math.round(runtimeConfig.FLUID_GRID_SIZE_CONTROL);
  worldState.nutrientField = createNutrientField(resolvedSize, rng);
  return worldState.nutrientField;
}

export function initializeLightMap(worldState, {
  config,
  configViews = null,
  size = null,
  rng = Math.random
} = {}) {
  const { runtime: runtimeConfig } = resolveConfigViews(configViews || config);
  const resolvedSize = size ?? Math.round(runtimeConfig.FLUID_GRID_SIZE_CONTROL);
  worldState.lightField = createLightField(resolvedSize, rng);
  return worldState.lightField;
}

export function initializeViscosityMap(worldState, {
  config,
  configViews = null,
  size = null,
  rng = Math.random
} = {}) {
  const { runtime: runtimeConfig } = resolveConfigViews(configViews || config);
  const resolvedSize = size ?? Math.round(runtimeConfig.FLUID_GRID_SIZE_CONTROL);
  worldState.viscosityField = createViscosityField(resolvedSize, rng);
  return worldState.viscosityField;
}
