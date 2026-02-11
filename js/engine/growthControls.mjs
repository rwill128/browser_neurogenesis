/**
 * Growth-control helpers for tuning stability under developmental body growth.
 *
 * These helpers are pure and shared so they can be unit-tested independently
 * from the full simulation runtime.
 */

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Compute a population-based throttle for growth events.
 *
 * @param {object} params
 * @param {number} params.population - Current creature population.
 * @param {number} params.floor - Creature population floor.
 * @param {number} params.ceiling - Creature population ceiling.
 * @param {number} params.softLimitMultiplier - Multiplier on floor where throttling begins.
 * @param {number} params.hardLimitMultiplier - Multiplier on floor where growth is blocked.
 * @param {number} params.minThrottleScale - Minimum allowed scaling while throttled.
 * @returns {{allowGrowth:boolean, scale:number, softLimit:number, hardLimit:number}}
 */
export function computeGrowthPopulationThrottle({
  population,
  floor,
  ceiling,
  softLimitMultiplier = 2,
  hardLimitMultiplier = 4,
  minThrottleScale = 0.05
}) {
  const safeFloor = Math.max(1, Math.floor(Number(floor) || 1));
  const safeCeiling = Math.max(safeFloor + 1, Math.floor(Number(ceiling) || (safeFloor + 1)));
  const pop = Math.max(0, Math.floor(Number(population) || 0));

  const softLimit = clamp(
    Math.floor(safeFloor * Math.max(1, Number(softLimitMultiplier) || 1)),
    safeFloor,
    safeCeiling - 1
  );

  const hardLimitRaw = Math.floor(safeFloor * Math.max(1.01, Number(hardLimitMultiplier) || 1.01));
  const hardLimit = clamp(Math.max(softLimit + 1, hardLimitRaw), softLimit + 1, safeCeiling);

  if (pop >= hardLimit) {
    return {
      allowGrowth: false,
      scale: 0,
      softLimit,
      hardLimit
    };
  }

  if (pop <= softLimit) {
    return {
      allowGrowth: true,
      scale: 1,
      softLimit,
      hardLimit
    };
  }

  const progress = (pop - softLimit) / Math.max(1, (hardLimit - softLimit));
  const minScale = clamp(Number(minThrottleScale) || 0.05, 0, 1);
  const scale = clamp(1 - progress, minScale, 1);

  return {
    allowGrowth: true,
    scale,
    softLimit,
    hardLimit
  };
}

/**
 * Compute multiplicative growth-cost pressure based on body size.
 *
 * As creatures approach their per-body point cap, growth gets progressively
 * more expensive. This dampens runaway expansion without fully disabling growth.
 */
export function computeGrowthSizeCostMultiplier({
  currentPoints,
  maxPoints,
  exponent = 1.15,
  maxMultiplier = 4
}) {
  const current = Math.max(0, Number(currentPoints) || 0);
  const max = Math.max(1, Number(maxPoints) || 1);
  const ratio = clamp(current / max, 0, 1);
  const exp = Math.max(0.5, Number(exponent) || 1.15);
  const cap = Math.max(1, Number(maxMultiplier) || 1);

  const normalized = Math.pow(ratio, exp);
  return clamp(1 + normalized * (cap - 1), 1, cap);
}
