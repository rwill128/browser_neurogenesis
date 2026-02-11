import config from '../config.js';
import { perlin } from '../utils.js';

function createField(size) {
  if (!Number.isFinite(size) || size <= 0) return new Float32Array(0);
  return new Float32Array(size * size);
}

export function createNutrientField(size = Math.round(config.FLUID_GRID_SIZE_CONTROL), random = Math.random) {
  const field = createField(size);
  if (field.length === 0) return field;

  const noiseScale = 0.05;
  const noiseOffsetX = random() * 1000 + 1000;
  const noiseOffsetY = random() * 1000 + 1000;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const noiseValue = perlin.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
      const mappedValue = ((noiseValue + 1) / 2) * (config.MAX_NUTRIENT_VALUE - config.MIN_NUTRIENT_VALUE) + config.MIN_NUTRIENT_VALUE;
      field[y * size + x] = Math.max(config.MIN_NUTRIENT_VALUE, Math.min(config.MAX_NUTRIENT_VALUE, mappedValue));
    }
  }

  return field;
}

export function createLightField(size = Math.round(config.FLUID_GRID_SIZE_CONTROL), random = Math.random) {
  const field = createField(size);
  if (field.length === 0) return field;

  const noiseScale = 0.05;
  const noiseOffsetX = random() * 1000;
  const noiseOffsetY = random() * 1000;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let noiseValue = perlin.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
      noiseValue = (noiseValue + 1) / 2;
      field[y * size + x] = Math.max(config.MIN_LIGHT_VALUE, Math.min(config.MAX_LIGHT_VALUE, noiseValue));
    }
  }

  return field;
}

export function createViscosityField(size = Math.round(config.FLUID_GRID_SIZE_CONTROL), random = Math.random) {
  const field = createField(size);
  if (field.length === 0) return field;

  const noiseScale = 0.06;
  const noiseOffsetX = random() * 1000 + 2000;
  const noiseOffsetY = random() * 1000 + 2000;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const noiseValue = perlin.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
      const mappedValue = ((noiseValue + 1) / 2) * (config.MAX_VISCOSITY_MULTIPLIER - config.MIN_VISCOSITY_MULTIPLIER) + config.MIN_VISCOSITY_MULTIPLIER;
      field[y * size + x] = Math.max(config.MIN_VISCOSITY_MULTIPLIER, Math.min(config.MAX_VISCOSITY_MULTIPLIER, mappedValue));
    }
  }

  return field;
}

export function createEnvironmentFields({ size = Math.round(config.FLUID_GRID_SIZE_CONTROL), random = Math.random } = {}) {
  return {
    nutrientField: createNutrientField(size, random),
    lightField: createLightField(size, random),
    viscosityField: createViscosityField(size, random)
  };
}
