function round(value, digits = 2) {
  const p = 10 ** digits;
  return Math.round((Number(value) || 0) * p) / p;
}

export function summarizeCreatureForTelemetry(body) {
  const center = body.getAveragePosition();
  const bbox = body.getBoundingBox();
  return {
    id: body.id,
    unstable: Boolean(body.isUnstable),
    points: body.massPoints.length,
    energy: round(body.creatureEnergy, 2),
    center: { x: round(center.x, 1), y: round(center.y, 1) },
    size: { w: round(bbox.maxX - bbox.minX, 1), h: round(bbox.maxY - bbox.minY, 1) }
  };
}

export function buildTelemetrySnapshot({
  tick,
  scenario,
  seed,
  fluidField,
  softBodyPopulation,
  particles,
  selectedBody
}) {
  const live = softBodyPopulation.filter((b) => !b.isUnstable);
  const selected = selectedBody ? summarizeCreatureForTelemetry(selectedBody) : null;

  return {
    tick,
    scenario: scenario || 'baseline',
    seed: seed ?? null,
    mode: fluidField && fluidField.gpuEnabled ? 'GPU' : 'CPU',
    populations: {
      creatures: softBodyPopulation.length,
      liveCreatures: live.length,
      particles: particles.length
    },
    selected,
    sampleCreatures: live.slice(0, 5).map(summarizeCreatureForTelemetry)
  };
}

function isCreatureInsideRect(creature, rect) {
  const x0 = rect.x;
  const y0 = rect.y;
  const x1 = rect.x + rect.width;
  const y1 = rect.y + rect.height;

  if (Array.isArray(creature.vertices) && creature.vertices.length > 0) {
    return creature.vertices.some((v) => v.x >= x0 && v.x <= x1 && v.y >= y0 && v.y <= y1);
  }

  const cx = creature.center?.x;
  const cy = creature.center?.y;
  return Number.isFinite(cx) && Number.isFinite(cy) && cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1;
}

export function querySnapshotRect({ snapshot, rect, particles = [] }) {
  const x0 = rect.x;
  const y0 = rect.y;
  const x1 = rect.x + rect.width;
  const y1 = rect.y + rect.height;

  const creatures = (snapshot.creatures || []).filter((c) => isCreatureInsideRect(c, rect));

  let particleCount = 0;
  const particleSample = [];
  for (const p of particles) {
    const pos = p?.pos;
    if (!pos || p.life <= 0) continue;
    if (pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1) {
      particleCount += 1;
      if (particleSample.length < 100) {
        particleSample.push({
          x: round(pos.x, 2),
          y: round(pos.y, 2),
          life: round(p.life, 3)
        });
      }
    }
  }

  return {
    rect,
    populations: snapshot.populations,
    creatures,
    particles: {
      count: particleCount,
      sample: particleSample
    }
  };
}

export function querySnapshotCreature({ snapshot, creatureId }) {
  const creature = (snapshot.creatures || []).find((c) => Number(c.id) === Number(creatureId)) || null;
  return {
    creatureId: Number(creatureId),
    found: Boolean(creature),
    creature
  };
}

export function selectRenderableCreatures(snapshot) {
  if (Array.isArray(snapshot?.creatures) && snapshot.creatures.length > 0) {
    return snapshot.creatures;
  }
  if (Array.isArray(snapshot?.sampleCreatures)) {
    return snapshot.sampleCreatures;
  }
  return [];
}
