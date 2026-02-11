import config from '../config.js';
import { perlin as sharedPerlin } from '../utils.js';

function createField(size) {
  if (!Number.isFinite(size) || size <= 0) return new Float32Array(0);
  return new Float32Array(size * size);
}

export function createPerlinNoise(random = Math.random) {
  const p = new Uint8Array(512);
  for (let i = 0; i < 256; i++) p[i] = Math.floor(random() * 256);
  for (let i = 0; i < 256; i++) p[256 + i] = p[i];

  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  const lerp = (t, a, b) => a + t * (b - a);
  const grad = (hash, x, y) => {
    const h = hash & 15;
    const u = h < 8 ? x : y;
    const v = h < 4 ? y : h === 12 || h === 14 ? x : 0;
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
  };

  return {
    noise(x, y) {
      const X = Math.floor(x) & 255;
      const Y = Math.floor(y) & 255;
      const xf = x - Math.floor(x);
      const yf = y - Math.floor(y);
      const u = fade(xf);
      const v = fade(yf);
      const A = p[X] + Y;
      const B = p[X + 1] + Y;
      return lerp(
        v,
        lerp(u, grad(p[A], xf, yf), grad(p[B], xf - 1, yf)),
        lerp(u, grad(p[A + 1], xf, yf - 1), grad(p[B + 1], xf - 1, yf - 1))
      );
    }
  };
}

export function createNutrientField(
  size = Math.round(config.FLUID_GRID_SIZE_CONTROL),
  random = Math.random,
  noise = sharedPerlin
) {
  const field = createField(size);
  if (field.length === 0) return field;

  const noiseScale = 0.05;
  const noiseOffsetX = random() * 1000 + 1000;
  const noiseOffsetY = random() * 1000 + 1000;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const noiseValue = noise.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
      const mappedValue = ((noiseValue + 1) / 2) * (config.MAX_NUTRIENT_VALUE - config.MIN_NUTRIENT_VALUE) + config.MIN_NUTRIENT_VALUE;
      field[y * size + x] = Math.max(config.MIN_NUTRIENT_VALUE, Math.min(config.MAX_NUTRIENT_VALUE, mappedValue));
    }
  }

  return field;
}

export function createLightField(
  size = Math.round(config.FLUID_GRID_SIZE_CONTROL),
  random = Math.random,
  noise = sharedPerlin
) {
  const field = createField(size);
  if (field.length === 0) return field;

  const noiseScale = 0.05;
  const noiseOffsetX = random() * 1000;
  const noiseOffsetY = random() * 1000;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let noiseValue = noise.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
      noiseValue = (noiseValue + 1) / 2;
      field[y * size + x] = Math.max(config.MIN_LIGHT_VALUE, Math.min(config.MAX_LIGHT_VALUE, noiseValue));
    }
  }

  return field;
}

export function createViscosityField(
  size = Math.round(config.FLUID_GRID_SIZE_CONTROL),
  random = Math.random,
  noise = sharedPerlin
) {
  const field = createField(size);
  if (field.length === 0) return field;

  const noiseScale = 0.06;
  const noiseOffsetX = random() * 1000 + 2000;
  const noiseOffsetY = random() * 1000 + 2000;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const noiseValue = noise.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
      const mappedValue = ((noiseValue + 1) / 2) * (config.MAX_VISCOSITY_MULTIPLIER - config.MIN_VISCOSITY_MULTIPLIER) + config.MIN_VISCOSITY_MULTIPLIER;
      field[y * size + x] = Math.max(config.MIN_VISCOSITY_MULTIPLIER, Math.min(config.MAX_VISCOSITY_MULTIPLIER, mappedValue));
    }
  }

  return field;
}

export function createEnvironmentFields({ size = Math.round(config.FLUID_GRID_SIZE_CONTROL), random = Math.random } = {}) {
  const noise = createPerlinNoise(random);
  return {
    nutrientField: createNutrientField(size, random, noise),
    lightField: createLightField(size, random, noise),
    viscosityField: createViscosityField(size, random, noise)
  };
}
