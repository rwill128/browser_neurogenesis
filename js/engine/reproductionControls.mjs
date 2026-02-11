/**
 * Reproduction control helpers.
 *
 * These pure utilities provide density-dependent fertility scaling and
 * resource-coupled gating/debit logic for reproduction.
 */

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function scaleFromPressure(current, softLimit, hardLimit, minScale) {
  const soft = Number.isFinite(softLimit) ? softLimit : 0;
  const hard = Number.isFinite(hardLimit) ? hardLimit : (soft + 1);
  const min = clamp(Number(minScale) || 0, 0, 1);
  const value = Math.max(0, Number(current) || 0);

  if (value <= soft) return 1;
  if (hard <= soft) return min;
  if (value >= hard) return min;

  const progress = (value - soft) / (hard - soft);
  return clamp(1 - progress * (1 - min), min, 1);
}

/**
 * Compute multiplicative fertility scaling under global + local crowding.
 *
 * @returns {{scale:number, globalScale:number, localScale:number, softPopulation:number, hardPopulation:number}}
 */
export function computeDensityFertilityScale({
  population,
  floor,
  ceiling,
  globalSoftMultiplier = 2,
  globalHardMultiplier = 4,
  globalMinScale = 0.1,
  localNeighbors = 0,
  localSoftNeighbors = 6,
  localHardNeighbors = 18,
  localMinScale = 0.2
}) {
  const safeFloor = Math.max(1, Math.floor(Number(floor) || 1));
  const safeCeiling = Math.max(safeFloor + 1, Math.floor(Number(ceiling) || (safeFloor + 1)));
  const pop = Math.max(0, Math.floor(Number(population) || 0));

  const softPopulation = clamp(
    Math.floor(safeFloor * Math.max(1, Number(globalSoftMultiplier) || 1)),
    safeFloor,
    safeCeiling - 1
  );
  const hardPopulation = clamp(
    Math.floor(safeFloor * Math.max(1.01, Number(globalHardMultiplier) || 1.01)),
    softPopulation + 1,
    safeCeiling
  );

  const globalScale = scaleFromPressure(pop, softPopulation, hardPopulation, globalMinScale);
  const localScale = scaleFromPressure(
    Math.max(0, Math.floor(Number(localNeighbors) || 0)),
    Math.max(0, Math.floor(Number(localSoftNeighbors) || 0)),
    Math.max(1, Math.floor(Number(localHardNeighbors) || 1)),
    localMinScale
  );

  return {
    scale: clamp(globalScale * localScale, 0, 1),
    globalScale,
    localScale,
    softPopulation,
    hardPopulation
  };
}

/**
 * Evaluate whether local nutrients/light permit reproduction this tick.
 */
export function evaluateResourceCoupling({
  nutrientValue,
  lightValue,
  minNutrient,
  minLight
}) {
  const nutrient = Number(nutrientValue);
  const light = Number(lightValue);
  const minN = Math.max(0.000001, Number(minNutrient) || 0.000001);
  const minL = Math.max(0.000001, Number(minLight) || 0.000001);

  const nutrientRatio = Number.isFinite(nutrient) ? nutrient / minN : 1;
  const lightRatio = Number.isFinite(light) ? light / minL : 1;
  const fertilityScale = clamp(Math.min(nutrientRatio, lightRatio), 0, 1);

  return {
    allow: fertilityScale >= 1,
    fertilityScale,
    nutrientRatio,
    lightRatio
  };
}

/**
 * Apply nutrient/light debit after successful offspring placement.
 */
export function applyReproductionResourceDebit({
  nutrientField,
  lightField,
  index,
  nutrientDebit = 0,
  lightDebit = 0,
  nutrientMin = 0,
  lightMin = 0
}) {
  const idx = Math.max(0, Math.floor(Number(index) || 0));
  const out = {
    nutrientAfter: null,
    lightAfter: null
  };

  if (nutrientField && idx >= 0 && idx < nutrientField.length) {
    const next = Math.max(Number(nutrientMin) || 0, (Number(nutrientField[idx]) || 0) - Math.max(0, Number(nutrientDebit) || 0));
    nutrientField[idx] = next;
    out.nutrientAfter = next;
  }

  if (lightField && idx >= 0 && idx < lightField.length) {
    const next = Math.max(Number(lightMin) || 0, (Number(lightField[idx]) || 0) - Math.max(0, Number(lightDebit) || 0));
    lightField[idx] = next;
    out.lightAfter = next;
  }

  return out;
}
