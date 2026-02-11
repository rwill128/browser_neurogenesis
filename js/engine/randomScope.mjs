export function withRandomSource(rng, work) {
  if (typeof work !== 'function') {
    throw new Error('withRandomSource requires a callback');
  }

  if (typeof rng !== 'function' || rng === Math.random) {
    return work();
  }

  const prevRandom = Math.random;
  Math.random = rng;
  try {
    return work();
  } finally {
    Math.random = prevRandom;
  }
}
