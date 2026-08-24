/**
 * Runs {@link work} with a temporary `Math.random` implementation.
 *
 * This is intentionally tiny and synchronous: it exists so deterministic harness
 * paths can execute browser-oriented code that still reads `Math.random`
 * internally, without permanently mutating global RNG state.
 *
 * @template T
 * @param {(() => number) | null | undefined} rng - Temporary RNG callback.
 *   When omitted/invalid (or already `Math.random`) the callback is executed
 *   without patching globals.
 * @param {() => T} work - Synchronous callback executed under the RNG scope.
 * @returns {T}
 * @throws {Error} When `work` is not a function.
 */
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
