/**
 * Ecology summary helpers for soak/regression reporting.
 *
 * Keeps metrics computation pure and testable.
 */

/**
 * Build aggregate node-type counts from live soft bodies.
 *
 * @param {Array} bodies
 * @param {Record<number|string,string>} nodeTypeNameById
 * @returns {Record<string, number>}
 */
export function aggregateNodeTypeCounts(bodies, nodeTypeNameById = {}) {
  const counts = {};
  for (const b of bodies || []) {
    if (!b || b.isUnstable || !Array.isArray(b.massPoints)) continue;
    for (const p of b.massPoints) {
      const key = nodeTypeNameById[p?.nodeType] || `TYPE_${p?.nodeType}`;
      counts[key] = (counts[key] || 0) + 1;
    }
  }
  return counts;
}

/**
 * Compute node diversity metrics from counts.
 *
 * Returns richness and Shannon entropy metrics commonly used in ecology.
 */
export function computeNodeDiversity(nodeTypeCounts = {}) {
  const entries = Object.entries(nodeTypeCounts).filter(([, v]) => Number(v) > 0);
  const totalNodes = entries.reduce((acc, [, v]) => acc + Number(v), 0);

  if (totalNodes <= 0) {
    return {
      totalNodes: 0,
      richness: 0,
      shannonEntropy: 0,
      shannonEvenness: 0
    };
  }

  let shannon = 0;
  for (const [, v] of entries) {
    const p = Number(v) / totalNodes;
    if (p > 0) shannon -= p * Math.log(p);
  }

  const richness = entries.length;
  const maxShannon = richness > 1 ? Math.log(richness) : 0;
  const evenness = maxShannon > 0 ? shannon / maxShannon : 0;

  return {
    totalNodes,
    richness,
    shannonEntropy: shannon,
    shannonEvenness: evenness
  };
}

/**
 * Compute growth-event cohort metrics for currently living creatures.
 */
export function summarizeGrowthCohorts(bodies, {
  activeThreshold = 1,
  highThreshold = 5
} = {}) {
  const living = (bodies || []).filter((b) => b && !b.isUnstable);
  const n = living.length;

  if (n === 0) {
    return {
      livingCreatures: 0,
      avgGrowthEvents: 0,
      medianGrowthEvents: 0,
      activeGrowers: 0,
      activeGrowerFraction: 0,
      highGrowers: 0,
      highGrowerFraction: 0
    };
  }

  const events = living.map((b) => Math.max(0, Number(b.growthEventsCompleted) || 0)).sort((a, b) => a - b);
  const sum = events.reduce((acc, x) => acc + x, 0);
  const avg = sum / n;
  const median = n % 2 === 1
    ? events[(n - 1) / 2]
    : (events[n / 2 - 1] + events[n / 2]) / 2;

  const active = events.filter((x) => x >= activeThreshold).length;
  const high = events.filter((x) => x >= highThreshold).length;

  return {
    livingCreatures: n,
    avgGrowthEvents: avg,
    medianGrowthEvents: median,
    activeGrowers: active,
    activeGrowerFraction: active / n,
    highGrowers: high,
    highGrowerFraction: high / n
  };
}
