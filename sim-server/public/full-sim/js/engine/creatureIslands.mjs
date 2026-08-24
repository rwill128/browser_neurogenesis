/**
 * Creature-island graph helpers.
 *
 * This module builds connected components of creatures that may interact during
 * the current tick. Components can later map to local parallel workers or
 * distributed jobs; for now they are used for serial batch execution planning.
 */

function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function bodyIdForSort(body, fallback = 0) {
  const id = Number(body?.id);
  return Number.isFinite(id) ? id : fallback;
}

function sortBodiesStableById(bodies = []) {
  return bodies
    .map((body, index) => ({ body, index, id: bodyIdForSort(body, Number.MAX_SAFE_INTEGER) }))
    .sort((a, b) => {
      if (a.id !== b.id) return a.id - b.id;
      return a.index - b.index;
    })
    .map((entry) => entry.body);
}

/**
 * Conservative world-space interaction radius used to infer island neighborhood.
 */
export function computeMaxCrossBodyInteractionRadius(config, {
  maxPointRadius = 12
} = {}) {
  const pointRadius = Math.max(0.1, safeNumber(maxPointRadius, 12));

  const repulsionRadius = pointRadius * Math.max(0, safeNumber(config?.BODY_REPULSION_RADIUS_FACTOR, 0)) * 2;

  const attractionMultiplier = Math.max(0,
    safeNumber(config?.ATTRACTION_RADIUS_MULTIPLIER_BASE, 0)
    + safeNumber(config?.ATTRACTION_RADIUS_MULTIPLIER_MAX_BONUS, 0)
  );
  const repulsionMultiplier = Math.max(0,
    safeNumber(config?.REPULSION_RADIUS_MULTIPLIER_BASE, 0)
    + safeNumber(config?.REPULSION_RADIUS_MULTIPLIER_MAX_BONUS, 0)
  );

  const predatorMultiplier = Number.isFinite(Number(config?.PREDATOR_RADIUS_GENE_MAX))
    ? Math.max(0, Number(config.PREDATOR_RADIUS_GENE_MAX))
    : Math.max(0,
        safeNumber(config?.PREDATION_RADIUS_MULTIPLIER_BASE, 0)
        + safeNumber(config?.PREDATION_RADIUS_MULTIPLIER_MAX_BONUS, 0)
      );

  return Math.max(
    repulsionRadius,
    pointRadius * attractionMultiplier,
    pointRadius * repulsionMultiplier,
    pointRadius * predatorMultiplier
  );
}

/**
 * Derive a conservative neighborhood radius in grid cells for island edges.
 */
export function computeIslandNeighborRadiusCells(config, gridCellSize, {
  maxPointRadius = 12,
  minCells = 1,
  maxCells = 8
} = {}) {
  const cellSize = Math.max(1, safeNumber(gridCellSize, 1));
  const maxRadius = computeMaxCrossBodyInteractionRadius(config, { maxPointRadius });
  const rawCells = Math.ceil(maxRadius / cellSize);
  const lo = Math.max(0, Math.floor(safeNumber(minCells, 1)));
  const hi = Math.max(lo, Math.floor(safeNumber(maxCells, 8)));
  return Math.max(lo, Math.min(hi, rawCells));
}

/**
 * Build creature interaction islands using occupied broad-phase grid cells.
 */
export function buildCreatureInteractionIslands({
  softBodyPopulation,
  spatialGrid,
  gridCols,
  gridRows,
  neighborRadiusCells = 1
} = {}) {
  const livingBodies = sortBodiesStableById(
    (Array.isArray(softBodyPopulation) ? softBodyPopulation : []).filter((body) => body && !body.isUnstable)
  );

  const bodyCount = livingBodies.length;
  if (bodyCount === 0) {
    return {
      islands: [],
      bodyCount: 0,
      edgeCount: 0,
      occupiedCells: 0,
      neighborRadiusCells: Math.max(0, Math.floor(safeNumber(neighborRadiusCells, 0)))
    };
  }

  const cols = Math.max(1, Math.floor(safeNumber(gridCols, 1)));
  const rows = Math.max(1, Math.floor(safeNumber(gridRows, 1)));
  const totalCells = cols * rows;
  const radius = Math.max(0, Math.floor(safeNumber(neighborRadiusCells, 1)));

  const bodyIndex = new Map();
  for (let i = 0; i < livingBodies.length; i++) {
    bodyIndex.set(livingBodies[i], i);
  }

  const cellBodySets = new Map();
  const buckets = Array.isArray(spatialGrid) ? spatialGrid : [];
  const limit = Math.min(totalCells, buckets.length);

  for (let cellIndex = 0; cellIndex < limit; cellIndex++) {
    const bucket = buckets[cellIndex];
    if (!Array.isArray(bucket) || bucket.length === 0) continue;

    let set = null;
    for (const item of bucket) {
      if (item?.type !== 'softbody_point') continue;
      const idx = bodyIndex.get(item.bodyRef);
      if (!Number.isInteger(idx)) continue;
      if (!set) set = new Set();
      set.add(idx);
    }

    if (set && set.size > 0) {
      cellBodySets.set(cellIndex, set);
    }
  }

  const adjacency = Array.from({ length: bodyCount }, () => new Set());
  let edgeCount = 0;

  for (const [cellIndex, sourceSet] of cellBodySets.entries()) {
    const gx = cellIndex % cols;
    const gy = Math.floor(cellIndex / cols);

    const minX = Math.max(0, gx - radius);
    const maxX = Math.min(cols - 1, gx + radius);
    const minY = Math.max(0, gy - radius);
    const maxY = Math.min(rows - 1, gy + radius);

    for (let ny = minY; ny <= maxY; ny++) {
      for (let nx = minX; nx <= maxX; nx++) {
        const neighborCell = nx + ny * cols;
        const neighborSet = cellBodySets.get(neighborCell);
        if (!neighborSet) continue;

        for (const a of sourceSet) {
          for (const b of neighborSet) {
            if (a === b) continue;
            if (adjacency[a].has(b)) continue;
            adjacency[a].add(b);
            adjacency[b].add(a);
            edgeCount += 1;
          }
        }
      }
    }
  }

  const visited = new Array(bodyCount).fill(false);
  const islands = [];

  for (let i = 0; i < bodyCount; i++) {
    if (visited[i]) continue;

    const stack = [i];
    visited[i] = true;
    const islandIndices = [];

    while (stack.length > 0) {
      const idx = stack.pop();
      islandIndices.push(idx);

      for (const next of adjacency[idx]) {
        if (!visited[next]) {
          visited[next] = true;
          stack.push(next);
        }
      }
    }

    islandIndices.sort((a, b) => bodyIdForSort(livingBodies[a], a) - bodyIdForSort(livingBodies[b], b));
    islands.push(islandIndices.map((idx) => livingBodies[idx]));
  }

  islands.sort((a, b) => {
    const aId = bodyIdForSort(a[0], Number.MAX_SAFE_INTEGER);
    const bId = bodyIdForSort(b[0], Number.MAX_SAFE_INTEGER);
    return aId - bId;
  });

  return {
    islands,
    bodyCount,
    edgeCount,
    occupiedCells: cellBodySets.size,
    neighborRadiusCells: radius
  };
}
