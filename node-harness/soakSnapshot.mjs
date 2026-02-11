/**
 * Build renderable snapshots from a live soak world-state.
 */

function invertEnum(obj) {
  return Object.fromEntries(Object.entries(obj).map(([k, v]) => [v, k]));
}

/**
 * Build a single renderable snapshot frame compatible with renderTimelineFrames.
 */
export function buildRenderableSoakSnapshot({
  worldState,
  worldWidth,
  worldHeight,
  nodeTypeEnum
}) {
  const nodeTypeNameById = invertEnum(nodeTypeEnum || {});
  const creatures = [];

  for (const body of worldState?.softBodyPopulation || []) {
    if (!body || body.isUnstable || !Array.isArray(body.massPoints) || body.massPoints.length === 0) continue;

    const center = body.getAveragePosition();
    const pointIndex = new Map();
    body.massPoints.forEach((p, idx) => pointIndex.set(p, idx));

    creatures.push({
      id: body.id,
      energy: Number(body.creatureEnergy || 0),
      center: { x: Number(center.x || 0), y: Number(center.y || 0) },
      vertices: body.massPoints.map((p, idx) => ({
        index: idx,
        x: Number(p?.pos?.x || 0),
        y: Number(p?.pos?.y || 0),
        radius: Number(p?.radius || 0),
        nodeType: p?.nodeType,
        nodeTypeName: nodeTypeNameById[p?.nodeType] || null
      })),
      springs: (body.springs || [])
        .map((s) => ({
          a: pointIndex.get(s.p1),
          b: pointIndex.get(s.p2),
          isRigid: Boolean(s.isRigid)
        }))
        .filter((s) => Number.isInteger(s.a) && Number.isInteger(s.b) && s.a !== s.b)
    });
  }

  return {
    world: {
      width: Number(worldWidth) || 1,
      height: Number(worldHeight) || 1
    },
    populations: {
      creatures: creatures.length,
      particles: Array.isArray(worldState?.particles) ? worldState.particles.length : 0
    },
    creatures
  };
}
