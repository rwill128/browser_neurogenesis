import { mulberry32 } from '../js/engine/random.mjs';

export function createSeededRandom(seed = 42) {
  return mulberry32(seed >>> 0);
}

export function withRandom(randomFn, work) {
  const prev = Math.random;
  Math.random = randomFn;
  try {
    return work();
  } finally {
    Math.random = prev;
  }
}
