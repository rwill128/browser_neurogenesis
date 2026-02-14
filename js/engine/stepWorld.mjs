/**
 * Shared world-step pipeline used by both browser and node harnesses.
 *
 * This module centralizes population maintenance, reproduction, fluid stepping,
 * and spatial-grid rebuilds so both runtimes execute the same simulation path.
 */

import { withRandomSource } from './randomScope.mjs';
import { resolveConfigViews } from './configViews.mjs';
import { stabilizeNewbornBody } from './newbornStability.mjs';
import {
  buildCreatureInteractionIslands,
  computeIslandNeighborRadiusCells
} from './creatureIslands.mjs';

/**
 * Draw a deterministic random value from [min, max) using the provided RNG.
 */
function randomInRange(rng, min, max) {
  return min + rng() * (max - min);
}

function round(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  const p = 10 ** digits;
  return Math.round(n * p) / p;
}

function deepClone(value) {
  if (value === null || value === undefined) return value;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return null;
  }
}

function countNodeTypes(points) {
  const out = {};
  for (const p of points || []) {
    const key = String(Number.isFinite(Number(p?.nodeType)) ? Number(p.nodeType) : -1);
    out[key] = (out[key] || 0) + 1;
  }
  return out;
}

function normalizeBirthOrigin(value) {
  const raw = String(value || '').trim().toLowerCase();
  if (!raw) return 'unknown';
  if (raw === 'floor' || raw === 'floorfill' || raw === 'floor_fill' || raw === 'floor_spawn') return 'floor_spawn';
  if (raw === 'reproduction' || raw === 'offspring' || raw === 'reproduction_offspring') return 'reproduction_offspring';
  if (raw === 'initial' || raw === 'initial_population' || raw === 'seed_population') return 'initial_population';
  if (raw === 'restored' || raw === 'restore' || raw === 'restored_checkpoint' || raw === 'imported_blueprint') return 'restored_checkpoint';
  return raw;
}

function summarizeLifecycleForRemoval(body) {
  const birthOrigin = normalizeBirthOrigin(body?.birthOrigin);
  const reproductionEventsCompleted = Math.max(0, Math.floor(Number(body?.reproductionEventsCompleted) || 0));
  const ticksSinceLastReproduction = Number.isFinite(Number(body?.ticksSinceLastReproduction))
    ? Math.max(0, Math.floor(Number(body.ticksSinceLastReproduction)))
    : null;
  const absoluteAgeTicks = Number.isFinite(Number(body?.absoluteAgeTicks))
    ? Math.max(0, Math.floor(Number(body.absoluteAgeTicks)))
    : null;
  const parentBodyId = Number.isFinite(Number(body?.parentBodyId)) ? Number(body.parentBodyId) : null;
  const lineageRootId = Number.isFinite(Number(body?.lineageRootId)) ? Number(body.lineageRootId) : null;
  const generation = Number.isFinite(Number(body?.generation)) ? Math.max(0, Math.floor(Number(body.generation))) : null;

  const isPostReproductionParent = reproductionEventsCompleted > 0;
  let lifecycleStage = 'unknown';
  if (isPostReproductionParent) lifecycleStage = 'post_reproduction_parent';
  else if (birthOrigin === 'floor_spawn') lifecycleStage = 'floor_spawn';
  else if (birthOrigin === 'reproduction_offspring') lifecycleStage = 'reproduction_offspring';
  else if (birthOrigin === 'initial_population') lifecycleStage = 'initial_population';
  else if (birthOrigin === 'restored_checkpoint') lifecycleStage = 'restored_checkpoint';

  return {
    birthOrigin,
    lifecycleStage,
    isPostReproductionParent,
    reproductionEventsCompleted,
    ticksSinceLastReproduction,
    absoluteAgeTicks,
    parentBodyId,
    lineageRootId,
    generation
  };
}

function ensureInstabilityTelemetryState(state, maxRecentDeaths = 1000) {
  if (!state.instabilityTelemetry || typeof state.instabilityTelemetry !== 'object') {
    state.instabilityTelemetry = {
      totalRemoved: 0,
      totalPhysicsRemoved: 0,
      totalNonPhysicsRemoved: 0,
      totalUnknownRemoved: 0,
      removedByReason: {},
      removedByPhysicsKind: {},
      removedByBirthOrigin: {},
      removedByLifecycleStage: {},
      recentDeaths: [],
      maxRecentDeaths,
      lastDeathSeq: 0,
      sampledDiagnostics: [],
      maxSampledDiagnostics: 50
    };
  }

  const t = state.instabilityTelemetry;
  t.totalRemoved = Number(t.totalRemoved) || 0;
  t.totalPhysicsRemoved = Number(t.totalPhysicsRemoved) || 0;
  t.totalNonPhysicsRemoved = Number(t.totalNonPhysicsRemoved) || 0;
  t.totalUnknownRemoved = Number(t.totalUnknownRemoved) || 0;
  t.removedByReason = t.removedByReason && typeof t.removedByReason === 'object' ? t.removedByReason : {};
  t.removedByPhysicsKind = t.removedByPhysicsKind && typeof t.removedByPhysicsKind === 'object' ? t.removedByPhysicsKind : {};
  t.removedByBirthOrigin = t.removedByBirthOrigin && typeof t.removedByBirthOrigin === 'object' ? t.removedByBirthOrigin : {};
  t.removedByLifecycleStage = t.removedByLifecycleStage && typeof t.removedByLifecycleStage === 'object' ? t.removedByLifecycleStage : {};
  t.recentDeaths = Array.isArray(t.recentDeaths) ? t.recentDeaths : [];
  t.sampledDiagnostics = Array.isArray(t.sampledDiagnostics) ? t.sampledDiagnostics : [];
  t.lastDeathSeq = Number(t.lastDeathSeq) || 0;
  t.maxRecentDeaths = Math.max(10, Math.floor(Number(t.maxRecentDeaths) || maxRecentDeaths));
  t.maxSampledDiagnostics = Math.max(5, Math.floor(Number(t.maxSampledDiagnostics) || 50));
  return t;
}

function percentileSorted(sorted, p) {
  if (!Array.isArray(sorted) || sorted.length === 0) return 0;
  const clampedP = Math.max(0, Math.min(100, Number(p) || 0));
  if (clampedP <= 0) return sorted[0];
  if (clampedP >= 100) return sorted[sorted.length - 1];

  const idx = (sorted.length - 1) * (clampedP / 100);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function ensureEdgeLengthTelemetryState(state, runtimeConfig) {
  if (!state.edgeLengthTelemetry || typeof state.edgeLengthTelemetry !== 'object') {
    state.edgeLengthTelemetry = {
      samplesCollected: 0,
      totalSpringSamples: 0,
      totalHugeOutliers: 0,
      latest: null,
      recentSamples: []
    };
  }

  const t = state.edgeLengthTelemetry;
  t.samplesCollected = Number(t.samplesCollected) || 0;
  t.totalSpringSamples = Number(t.totalSpringSamples) || 0;
  t.totalHugeOutliers = Number(t.totalHugeOutliers) || 0;
  t.latest = t.latest && typeof t.latest === 'object' ? t.latest : null;
  t.recentSamples = Array.isArray(t.recentSamples) ? t.recentSamples : [];

  t.enabled = runtimeConfig.EDGE_LENGTH_TELEMETRY_ENABLED !== false;
  t.sampleEveryNSteps = Math.max(1, Math.floor(Number(runtimeConfig.EDGE_LENGTH_TELEMETRY_SAMPLE_EVERY_N_STEPS) || 10));
  t.modeBinSize = Math.max(0.0001, Number(runtimeConfig.EDGE_LENGTH_TELEMETRY_MODE_BIN_SIZE) || 0.01);
  t.hugeOutlierIqrMultiplier = Math.max(1, Number(runtimeConfig.EDGE_LENGTH_TELEMETRY_HUGE_OUTLIER_IQR_MULTIPLIER) || 3);
  t.historyMaxSamples = Math.max(1, Math.floor(Number(runtimeConfig.EDGE_LENGTH_TELEMETRY_HISTORY_MAX_SAMPLES) || 120));
  t.maxRecordedOutliers = Math.max(0, Math.floor(Number(runtimeConfig.EDGE_LENGTH_TELEMETRY_MAX_RECORDED_OUTLIERS) || 24));
  return t;
}

function updateEdgeLengthTelemetry(state, runtimeConfig) {
  const telemetry = ensureEdgeLengthTelemetryState(state, runtimeConfig);
  if (!telemetry.enabled) return;

  const step = Number(state.simulationStep) || 0;
  if (step <= 0 || (step % telemetry.sampleEveryNSteps) !== 0) return;

  const lengths = [];
  const ratios = [];
  const records = [];

  for (const body of state.softBodyPopulation || []) {
    if (!body || !Array.isArray(body.springs)) continue;
    for (let i = 0; i < body.springs.length; i++) {
      const spring = body.springs[i];
      if (!spring?.p1?.pos || !spring?.p2?.pos) continue;

      const dx = (Number(spring.p1.pos.x) || 0) - (Number(spring.p2.pos.x) || 0);
      const dy = (Number(spring.p1.pos.y) || 0) - (Number(spring.p2.pos.y) || 0);
      const currentLength = Math.hypot(dx, dy);
      if (!Number.isFinite(currentLength) || currentLength <= 0) continue;

      const restLength = Math.max(1e-9, Number(spring.restLength) || 1e-9);
      const ratio = currentLength / restLength;

      lengths.push(currentLength);
      ratios.push(ratio);
      records.push({
        bodyId: Number.isFinite(Number(body.id)) ? Number(body.id) : null,
        springIndex: i,
        currentLength,
        restLength,
        ratio
      });
    }
  }

  if (lengths.length === 0) {
    telemetry.latest = {
      sampledAtStep: step,
      sampledAtIso: new Date().toISOString(),
      springCount: 0
    };
    return;
  }

  lengths.sort((a, b) => a - b);
  ratios.sort((a, b) => a - b);

  const count = lengths.length;
  const sum = lengths.reduce((acc, v) => acc + v, 0);
  const mean = sum / count;
  const median = percentileSorted(lengths, 50);
  const q1 = percentileSorted(lengths, 25);
  const q3 = percentileSorted(lengths, 75);
  const iqr = Math.max(0, q3 - q1);
  const hugeThreshold = q3 + (telemetry.hugeOutlierIqrMultiplier * iqr);
  const hugeOutliers = records.filter((r) => r.currentLength > hugeThreshold).sort((a, b) => b.currentLength - a.currentLength);

  const modeBinCounts = new Map();
  for (const value of lengths) {
    const bin = Math.round(value / telemetry.modeBinSize) * telemetry.modeBinSize;
    modeBinCounts.set(bin, (modeBinCounts.get(bin) || 0) + 1);
  }
  let modeValue = lengths[0];
  let modeCount = 0;
  for (const [bin, cnt] of modeBinCounts.entries()) {
    if (cnt > modeCount) {
      modeCount = cnt;
      modeValue = bin;
    }
  }

  const stretchLimit = Math.max(1, Number(runtimeConfig.MAX_SPRING_STRETCH_FACTOR) || 1);
  const ratioAboveStretchLimit = ratios.filter((r) => r > stretchLimit).length;

  const summary = {
    sampledAtStep: step,
    sampledAtIso: new Date().toISOString(),
    springCount: count,
    meanCurrentLength: round(mean, 6),
    medianCurrentLength: round(median, 6),
    modeCurrentLength: round(modeValue, 6),
    modeCount,
    p95CurrentLength: round(percentileSorted(lengths, 95), 6),
    p99CurrentLength: round(percentileSorted(lengths, 99), 6),
    maxCurrentLength: round(lengths[lengths.length - 1], 6),
    q1CurrentLength: round(q1, 6),
    q3CurrentLength: round(q3, 6),
    iqrCurrentLength: round(iqr, 6),
    hugeOutlierThreshold: round(hugeThreshold, 6),
    hugeOutlierCount: hugeOutliers.length,
    hugeOutlierPct: round((hugeOutliers.length / count) * 100, 6),
    meanStretchRatio: round(ratios.reduce((acc, v) => acc + v, 0) / ratios.length, 6),
    medianStretchRatio: round(percentileSorted(ratios, 50), 6),
    p95StretchRatio: round(percentileSorted(ratios, 95), 6),
    p99StretchRatio: round(percentileSorted(ratios, 99), 6),
    maxStretchRatio: round(ratios[ratios.length - 1], 6),
    stretchLimit,
    aboveStretchLimitCount: ratioAboveStretchLimit,
    aboveStretchLimitPct: round((ratioAboveStretchLimit / ratios.length) * 100, 6),
    hugeOutliersTop: hugeOutliers.slice(0, telemetry.maxRecordedOutliers).map((r) => ({
      bodyId: r.bodyId,
      springIndex: r.springIndex,
      currentLength: round(r.currentLength, 6),
      restLength: round(r.restLength, 6),
      ratio: round(r.ratio, 6)
    }))
  };

  telemetry.samplesCollected += 1;
  telemetry.totalSpringSamples += count;
  telemetry.totalHugeOutliers += hugeOutliers.length;
  telemetry.latest = summary;
  telemetry.recentSamples.push(summary);
  if (telemetry.recentSamples.length > telemetry.historyMaxSamples) {
    telemetry.recentSamples.splice(0, telemetry.recentSamples.length - telemetry.historyMaxSamples);
  }
}

function classifyInstabilityReason(reason) {
  const r = String(reason || 'unknown');
  if (r === 'physics_out_of_bounds') {
    return { unstableClass: 'physics', unstablePhysicsKind: 'boundary_exit' };
  }
  // Backward-compat legacy bucket.
  if (r === 'physics_invalid_motion_or_nan') {
    return { unstableClass: 'physics', unstablePhysicsKind: 'numeric_or_nan' };
  }
  if (r === 'physics_invalid_motion') {
    return { unstableClass: 'physics', unstablePhysicsKind: 'invalid_motion' };
  }
  if (r === 'physics_nan_position' || r === 'physics_non_finite_position') {
    return { unstableClass: 'physics', unstablePhysicsKind: 'non_finite_numeric' };
  }
  if (r === 'physics_spring_overstretch' || r === 'physics_span_exceeded') {
    return { unstableClass: 'physics', unstablePhysicsKind: 'geometric_explosion' };
  }
  if (r.startsWith('physics_')) {
    return { unstableClass: 'physics', unstablePhysicsKind: 'other_physics' };
  }
  if (r === 'unknown') {
    return { unstableClass: 'unknown', unstablePhysicsKind: null };
  }
  return { unstableClass: 'non_physics', unstablePhysicsKind: null };
}

function summarizePhenotype(body) {
  const massPoints = Array.isArray(body?.massPoints) ? body.massPoints : [];
  const springs = Array.isArray(body?.springs) ? body.springs : [];
  const pointIndex = new Map();
  for (let i = 0; i < massPoints.length; i++) pointIndex.set(massPoints[i], i);

  let center = { x: 0, y: 0 };
  if (typeof body?.getAveragePosition === 'function') {
    const c = body.getAveragePosition();
    center = { x: Number(c?.x) || 0, y: Number(c?.y) || 0 };
  } else if (massPoints.length > 0) {
    let sx = 0;
    let sy = 0;
    for (const p of massPoints) {
      sx += Number(p?.pos?.x) || 0;
      sy += Number(p?.pos?.y) || 0;
    }
    center = { x: sx / massPoints.length, y: sy / massPoints.length };
  }

  let bbox = null;
  if (typeof body?.getBoundingBox === 'function') {
    const b = body.getBoundingBox();
    if (b && Number.isFinite(b.minX) && Number.isFinite(b.maxX) && Number.isFinite(b.minY) && Number.isFinite(b.maxY)) {
      bbox = {
        minX: round(b.minX, 3),
        minY: round(b.minY, 3),
        maxX: round(b.maxX, 3),
        maxY: round(b.maxY, 3),
        width: round(b.maxX - b.minX, 3),
        height: round(b.maxY - b.minY, 3)
      };
    }
  }

  return {
    pointCount: massPoints.length,
    springCount: springs.length,
    nodeTypeCounts: countNodeTypes(massPoints),
    avgStiffness: round(typeof body?.getAverageStiffness === 'function' ? body.getAverageStiffness() : 0, 5),
    avgDamping: round(typeof body?.getAverageDamping === 'function' ? body.getAverageDamping() : 0, 5),
    bbox,
    points: massPoints.map((p) => ({
      relX: round((Number(p?.pos?.x) || 0) - center.x, 3),
      relY: round((Number(p?.pos?.y) || 0) - center.y, 3),
      radius: round(p?.radius, 4),
      mass: round(p?.mass, 5),
      nodeType: Number.isFinite(Number(p?.nodeType)) ? Number(p.nodeType) : null,
      movementType: Number.isFinite(Number(p?.movementType)) ? Number(p.movementType) : null,
      canBeGrabber: Boolean(p?.canBeGrabber),
      eyeTargetType: Number.isFinite(Number(p?.eyeTargetType)) ? Number(p.eyeTargetType) : null,
      isDesignatedEye: Boolean(p?.isDesignatedEye)
    })),
    springs: springs
      .map((s) => ({
        p1Index: pointIndex.get(s?.p1),
        p2Index: pointIndex.get(s?.p2),
        restLength: round(s?.restLength, 5),
        isRigid: Boolean(s?.isRigid),
        stiffness: round(s?.stiffness, 5),
        damping: round(s?.dampingFactor, 5)
      }))
      .filter((s) => Number.isInteger(s.p1Index) && Number.isInteger(s.p2Index) && s.p1Index !== s.p2Index)
  };
}

function summarizeHereditaryBlueprint(body) {
  const points = Array.isArray(body?.blueprintPoints) ? body.blueprintPoints : [];
  const springs = Array.isArray(body?.blueprintSprings) ? body.blueprintSprings : [];

  return {
    pointCount: points.length,
    springCount: springs.length,
    nodeTypeCounts: countNodeTypes(points),
    points: points.map((p) => ({
      relX: round(p?.relX, 3),
      relY: round(p?.relY, 3),
      radius: round(p?.radius, 4),
      mass: round(p?.mass, 5),
      nodeType: Number.isFinite(Number(p?.nodeType)) ? Number(p.nodeType) : null,
      movementType: Number.isFinite(Number(p?.movementType)) ? Number(p.movementType) : null,
      canBeGrabber: Boolean(p?.canBeGrabber),
      eyeTargetType: Number.isFinite(Number(p?.eyeTargetType)) ? Number(p.eyeTargetType) : null
    })),
    springs: springs
      .map((s) => ({
        p1Index: Number.isFinite(Number(s?.p1Index)) ? Number(s.p1Index) : null,
        p2Index: Number.isFinite(Number(s?.p2Index)) ? Number(s.p2Index) : null,
        restLength: round(s?.restLength, 5),
        isRigid: Boolean(s?.isRigid),
        stiffness: round(s?.stiffness, 5),
        damping: round(s?.damping, 5)
      }))
      .filter((s) => Number.isInteger(s.p1Index) && Number.isInteger(s.p2Index) && s.p1Index !== s.p2Index)
  };
}

function summarizeHeritableParameters(body) {
  return {
    stiffness: round(body?.stiffness, 5),
    springDamping: round(body?.springDamping, 5),
    motorImpulseInterval: round(body?.motorImpulseInterval, 5),
    motorImpulseMagnitudeCap: round(body?.motorImpulseMagnitudeCap, 5),
    emitterStrength: round(body?.emitterStrength, 5),
    emitterDirection: {
      x: round(body?.emitterDirection?.x, 5),
      y: round(body?.emitterDirection?.y, 5)
    },
    numOffspring: Number.isFinite(Number(body?.numOffspring)) ? Number(body.numOffspring) : null,
    offspringSpawnRadius: round(body?.offspringSpawnRadius, 5),
    pointAddChance: round(body?.pointAddChance, 6),
    springConnectionRadius: round(body?.springConnectionRadius, 5),
    jetMaxVelocityGene: round(body?.jetMaxVelocityGene, 5),
    reproductionEnergyThreshold: round(body?.reproductionEnergyThreshold, 5),
    reproductionCooldownGene: round(body?.reproductionCooldownGene, 5),
    growthGenome: deepClone(body?.growthGenome || null)
  };
}

function summarizeDecisionState(body) {
  const brainNode = body?.brain?.brainNode || null;
  const nd = brainNode?.neuronData || null;

  const inputLabeled = Array.isArray(nd?.currentFrameInputVectorWithLabels)
    ? nd.currentFrameInputVectorWithLabels.slice(0, 24).map((e) => ({
        label: String(e?.label || ''),
        value: round(e?.value, 6)
      }))
    : [];

  const rawOutputs = Array.isArray(nd?.rawOutputs)
    ? nd.rawOutputs.slice(0, 24).map((v) => round(v, 6))
    : [];

  return {
    hasBrainNode: Boolean(brainNode),
    brainPointIndex: Array.isArray(body?.massPoints) ? body.massPoints.indexOf(brainNode) : -1,
    rewardStrategy: Number.isFinite(Number(body?.rewardStrategy)) ? Number(body.rewardStrategy) : null,
    rlAlgorithmType: Number.isFinite(Number(body?.rlAlgorithmType)) ? Number(body.rlAlgorithmType) : null,
    inputVectorSize: Number.isFinite(Number(nd?.inputVectorSize)) ? Number(nd.inputVectorSize) : null,
    outputVectorSize: Number.isFinite(Number(nd?.outputVectorSize)) ? Number(nd.outputVectorSize) : null,
    hiddenLayerSize: Number.isFinite(Number(nd?.hiddenLayerSize)) ? Number(nd.hiddenLayerSize) : null,
    sampledInputs: inputLabeled,
    sampledRawOutputs: rawOutputs,
    ticksSinceBirth: Number.isFinite(Number(body?.ticksSinceBirth)) ? Number(body.ticksSinceBirth) : null,
    canReproduce: Boolean(body?.canReproduce),
    creatureEnergy: round(body?.creatureEnergy, 6),
    currentMaxEnergy: round(body?.currentMaxEnergy, 6)
  };
}

function buildInstabilityRemovalEvent(state, body) {
  const reason = String(body?.unstableReason || 'unknown');
  const classification = classifyInstabilityReason(reason);
  const lifecycle = summarizeLifecycleForRemoval(body);
  const telemetry = ensureInstabilityTelemetryState(state);
  const deathSeq = (telemetry.lastDeathSeq || 0) + 1;
  telemetry.lastDeathSeq = deathSeq;

  return {
    deathSeq,
    simulationStep: Number(state.simulationStep) || 0,
    bodyId: Number.isFinite(Number(body?.id)) ? Number(body.id) : null,
    unstableReason: reason,
    unstableReasonDetails: deepClone(body?.unstableReasonDetails || null),
    unstableClass: classification.unstableClass,
    unstablePhysicsKind: classification.unstablePhysicsKind,
    physicsStabilityDeath: classification.unstableClass === 'physics',
    ticksSinceBirth: Number.isFinite(Number(body?.ticksSinceBirth)) ? Number(body.ticksSinceBirth) : null,
    birthOrigin: lifecycle.birthOrigin,
    lifecycleStage: lifecycle.lifecycleStage,
    isPostReproductionParent: lifecycle.isPostReproductionParent,
    reproductionEventsCompleted: lifecycle.reproductionEventsCompleted,
    ticksSinceLastReproduction: lifecycle.ticksSinceLastReproduction,
    absoluteAgeTicks: lifecycle.absoluteAgeTicks,
    parentBodyId: lifecycle.parentBodyId,
    lineageRootId: lifecycle.lineageRootId,
    generation: lifecycle.generation,
    creatureEnergy: round(body?.creatureEnergy, 6),
    currentMaxEnergy: round(body?.currentMaxEnergy, 6),
    hereditaryBlueprint: summarizeHereditaryBlueprint(body),
    physiology: summarizePhenotype(body),
    heritableParameters: summarizeHeritableParameters(body),
    decisionSnapshot: summarizeDecisionState(body)
  };
}

/**
 * Rebuild broad-phase occupancy for body points and particles.
 */
function updateSpatialGrid(state, config, constants) {
  const { spatialGrid, softBodyPopulation, particles } = state;
  const gridCellSize = constants.GRID_CELL_SIZE ?? config.GRID_CELL_SIZE;
  if (!spatialGrid) return;

  for (let i = 0; i < spatialGrid.length; i++) {
    spatialGrid[i] = [];
  }

  for (const body of softBodyPopulation) {
    if (body.isUnstable) continue;
    for (let i = 0; i < body.massPoints.length; i++) {
      const point = body.massPoints[i];
      const gx = Math.floor(point.pos.x / gridCellSize);
      const gy = Math.floor(point.pos.y / gridCellSize);
      const index = gx + gy * config.GRID_COLS;
      if (index >= 0 && index < spatialGrid.length) {
        spatialGrid[index].push({
          type: 'softbody_point',
          pointRef: point,
          bodyRef: body,
          originalIndex: i
        });
      }
    }
  }

  for (const particle of particles) {
    if (particle.life <= 0) continue;
    const gx = Math.floor(particle.pos.x / gridCellSize);
    const gy = Math.floor(particle.pos.y / gridCellSize);
    const index = gx + gy * config.GRID_COLS;
    if (index >= 0 && index < spatialGrid.length) {
      spatialGrid[index].push({
        type: 'particle',
        particleRef: particle
      });
    }
  }
}

/**
 * Inject user-driven velocity emitters into the fluid field.
 */
function applyVelocityEmitters(state, config) {
  if (!state.fluidField || config.EMITTER_STRENGTH <= 0) return;
  for (const emitter of config.velocityEmitters) {
    state.fluidField.addVelocity(
      emitter.gridX,
      emitter.gridY,
      emitter.forceX * config.EMITTER_STRENGTH,
      emitter.forceY * config.EMITTER_STRENGTH
    );
  }
}

function hueToRgb255(h) {
  const hue = ((Number(h) % 1) + 1) % 1;
  const i = Math.floor(hue * 6);
  const f = hue * 6 - i;
  const q = 1 - f;

  let r = 0;
  let g = 0;
  let b = 0;

  switch (i % 6) {
    case 0: r = 1; g = f; b = 0; break;
    case 1: r = q; g = 1; b = 0; break;
    case 2: r = 0; g = 1; b = f; break;
    case 3: r = 0; g = q; b = 1; break;
    case 4: r = f; g = 0; b = 1; break;
    case 5: r = 1; g = 0; b = q; break;
    default: r = 1; g = 1; b = 1; break;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function ensureLandscapeDyeEmitters(state, config, rng) {
  if (!state.fluidField) {
    state.landscapeDyeEmitters = [];
    return state.landscapeDyeEmitters;
  }

  const enabled = config.LANDSCAPE_DYE_EMITTERS_ENABLED === true;
  const count = Math.max(0, Math.floor(Number(config.LANDSCAPE_DYE_EMITTER_COUNT) || 0));
  if (!enabled || count <= 0) {
    state.landscapeDyeEmitters = [];
    return state.landscapeDyeEmitters;
  }

  if (Array.isArray(state.landscapeDyeEmitters) && state.landscapeDyeEmitters.length === count) {
    return state.landscapeDyeEmitters;
  }

  const strengthMin = Math.max(0, Number(config.LANDSCAPE_DYE_EMITTER_STRENGTH_MIN) || 0);
  const strengthMax = Math.max(strengthMin, Number(config.LANDSCAPE_DYE_EMITTER_STRENGTH_MAX) || strengthMin);
  const pulseMin = Math.max(0, Number(config.LANDSCAPE_DYE_EMITTER_PULSE_HZ_MIN) || 0);
  const pulseMax = Math.max(pulseMin, Number(config.LANDSCAPE_DYE_EMITTER_PULSE_HZ_MAX) || pulseMin);

  state.landscapeDyeEmitters = Array.from({ length: count }, () => {
    const hue = rng();
    const [r, g, b] = hueToRgb255(hue);
    return {
      x: randomInRange(rng, 0, config.WORLD_WIDTH),
      y: randomInRange(rng, 0, config.WORLD_HEIGHT),
      r,
      g,
      b,
      baseStrength: randomInRange(rng, strengthMin, strengthMax),
      pulseHz: randomInRange(rng, pulseMin, pulseMax),
      phase: randomInRange(rng, 0, Math.PI * 2)
    };
  });

  return state.landscapeDyeEmitters;
}

function applyLandscapeDyeEmitters(state, config, dt, rng) {
  if (!state.fluidField) return;

  const emitters = ensureLandscapeDyeEmitters(state, config, rng);
  if (!emitters.length) return;

  const radiusCells = Math.max(0, Math.floor(Number(config.LANDSCAPE_DYE_EMITTER_RADIUS_CELLS) || 0));
  const t = (Number(state.simulationStep) || 0) * Math.max(0, Number(dt) || 0.01);

  for (const emitter of emitters) {
    if (!emitter) continue;
    const gx = Math.floor(Number(emitter.x) / state.fluidField.scaleX);
    const gy = Math.floor(Number(emitter.y) / state.fluidField.scaleY);

    const pulseHz = Math.max(0, Number(emitter.pulseHz) || 0);
    const phase = Number(emitter.phase) || 0;
    const pulse = 0.5 + 0.5 * Math.sin((Math.PI * 2 * pulseHz * t) + phase);
    const strength = Math.max(0, Number(emitter.baseStrength) || 0) * (0.45 + 0.55 * pulse);

    if (radiusCells <= 0) {
      state.fluidField.addDensity(gx, gy, emitter.r, emitter.g, emitter.b, strength);
      continue;
    }

    for (let dy = -radiusCells; dy <= radiusCells; dy++) {
      for (let dx = -radiusCells; dx <= radiusCells; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > radiusCells) continue;

        const falloff = Math.max(0.2, 1 - (dist / (radiusCells + 1e-9)));
        state.fluidField.addDensity(gx + dx, gy + dy, emitter.r, emitter.g, emitter.b, strength * falloff);
      }
    }
  }
}

function ensureLandscapeVelocityEmitters(state, config, rng) {
  if (!state.fluidField) {
    state.landscapeVelocityEmitters = [];
    return state.landscapeVelocityEmitters;
  }

  const enabled = config.LANDSCAPE_VELOCITY_EMITTERS_ENABLED === true;
  const count = Math.max(0, Math.floor(Number(config.LANDSCAPE_VELOCITY_EMITTER_COUNT) || 0));
  if (!enabled || count <= 0) {
    state.landscapeVelocityEmitters = [];
    return state.landscapeVelocityEmitters;
  }

  if (Array.isArray(state.landscapeVelocityEmitters) && state.landscapeVelocityEmitters.length === count) {
    return state.landscapeVelocityEmitters;
  }

  const strengthMin = Math.max(0, Number(config.LANDSCAPE_VELOCITY_EMITTER_STRENGTH_MIN) || 0);
  const strengthMax = Math.max(strengthMin, Number(config.LANDSCAPE_VELOCITY_EMITTER_STRENGTH_MAX) || strengthMin);
  const pulseMin = Math.max(0, Number(config.LANDSCAPE_VELOCITY_EMITTER_PULSE_HZ_MIN) || 0);
  const pulseMax = Math.max(pulseMin, Number(config.LANDSCAPE_VELOCITY_EMITTER_PULSE_HZ_MAX) || pulseMin);

  state.landscapeVelocityEmitters = Array.from({ length: count }, () => {
    const angle = randomInRange(rng, 0, Math.PI * 2);
    return {
      x: randomInRange(rng, 0, config.WORLD_WIDTH),
      y: randomInRange(rng, 0, config.WORLD_HEIGHT),
      dirX: Math.cos(angle),
      dirY: Math.sin(angle),
      baseStrength: randomInRange(rng, strengthMin, strengthMax),
      pulseHz: randomInRange(rng, pulseMin, pulseMax),
      phase: randomInRange(rng, 0, Math.PI * 2),
      swirl: rng() < 0.5 ? -1 : 1
    };
  });

  return state.landscapeVelocityEmitters;
}

function applyLandscapeVelocityEmitters(state, config, dt, rng) {
  if (!state.fluidField) return;

  const emitters = ensureLandscapeVelocityEmitters(state, config, rng);
  if (!emitters.length) return;

  const radiusCells = Math.max(0, Math.floor(Number(config.LANDSCAPE_VELOCITY_EMITTER_RADIUS_CELLS) || 0));
  const t = (Number(state.simulationStep) || 0) * Math.max(0, Number(dt) || 0.01);

  for (const emitter of emitters) {
    if (!emitter) continue;
    const gx = Math.floor(Number(emitter.x) / state.fluidField.scaleX);
    const gy = Math.floor(Number(emitter.y) / state.fluidField.scaleY);

    const pulseHz = Math.max(0, Number(emitter.pulseHz) || 0);
    const phase = Number(emitter.phase) || 0;
    const pulse = 0.5 + 0.5 * Math.sin((Math.PI * 2 * pulseHz * t) + phase);
    const strength = Math.max(0, Number(emitter.baseStrength) || 0) * (0.45 + 0.55 * pulse);

    for (let dy = -radiusCells; dy <= radiusCells; dy++) {
      for (let dx = -radiusCells; dx <= radiusCells; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > Math.max(0, radiusCells)) continue;
        const falloff = radiusCells <= 0 ? 1 : Math.max(0.15, 1 - (dist / (radiusCells + 1e-9)));

        // Tangential component creates a local swirl, plus a directional drift component.
        const tx = dist > 1e-9 ? (-dy / dist) * emitter.swirl : 0;
        const ty = dist > 1e-9 ? (dx / dist) * emitter.swirl : 0;
        const vx = (emitter.dirX * 0.55 + tx * 0.45) * strength * falloff;
        const vy = (emitter.dirY * 0.55 + ty * 0.45) * strength * falloff;
        state.fluidField.addVelocity(gx + dx, gy + dy, vx, vy);
      }
    }
  }
}

/**
 * If a fixed point is being dragged, transfer displacement impulse into fluid.
 */
function maybeApplySelectedPointFluidPush(state, config) {
  if (!state.fluidField || !config.selectedSoftBodyPoint || !config.selectedSoftBodyPoint.point?.isFixed) return;

  const point = config.selectedSoftBodyPoint.point;
  const displacementX = point.pos.x - point.prevPos.x;
  const displacementY = point.pos.y - point.prevPos.y;
  const movementMagnitudeSq = displacementX * displacementX + displacementY * displacementY;
  const movementThresholdSq = 0.01 * 0.01;

  if (movementMagnitudeSq > movementThresholdSq) {
    const fluidGridX = Math.floor(point.pos.x / state.fluidField.scaleX);
    const fluidGridY = Math.floor(point.pos.y / state.fluidField.scaleY);

    state.fluidField.addVelocity(
      fluidGridX,
      fluidGridY,
      displacementX * config.SOFT_BODY_PUSH_STRENGTH / state.fluidField.scaleX,
      displacementY * config.SOFT_BODY_PUSH_STRENGTH / state.fluidField.scaleY
    );
    state.fluidField.addDensity(fluidGridX, fluidGridY, 60, 60, 80, 15);
  }
}

/**
 * Spawn one particle at a random world coordinate.
 */
function spawnParticle(state, config, ParticleClass, rng) {
  const x = rng() * config.WORLD_WIDTH;
  const y = rng() * config.WORLD_HEIGHT;
  const particle = withRandomSource(rng, () => new ParticleClass(x, y, state.fluidField));
  state.particles.push(particle);
}

/**
 * Spawn one creature while wiring world references needed by runtime systems.
 */
function spawnCreature(state, config, SoftBodyClass, rng, dt, margin = 50) {
  const x = randomInRange(rng, margin, config.WORLD_WIDTH - margin);
  const y = randomInRange(rng, margin, config.WORLD_HEIGHT - margin);
  const body = withRandomSource(rng, () => new SoftBodyClass(state.nextSoftBodyId++, x, y, null));
  body.birthOrigin = 'floor_spawn';
  body.parentBodyId = null;
  body.generation = Number.isFinite(Number(body.generation)) ? Math.max(0, Math.floor(Number(body.generation))) : 0;
  body.lineageRootId = Number.isFinite(Number(body.lineageRootId)) ? Number(body.lineageRootId) : body.id;
  body.absoluteAgeTicks = 0;
  body.reproductionEventsCompleted = Number.isFinite(Number(body.reproductionEventsCompleted))
    ? Math.max(0, Math.floor(Number(body.reproductionEventsCompleted)))
    : 0;
  body.ticksSinceLastReproduction = null;
  body.setNutrientField(state.nutrientField);
  body.setLightField(state.lightField);
  body.setParticles(state.particles);
  body.setSpatialGrid(state.spatialGrid);

  // Spawn-time correction: keep newborn geometry inside world bounds and damp tiny-world rigidity.
  stabilizeNewbornBody(body, {
    config,
    dt
  });
  body.__newbornStabilityApplied = true;

  state.softBodyPopulation.push(body);
  return body;
}

/**
 * Fold per-creature energy accounting into global aggregates before removal.
 */
function accumulateRemovedBodyEnergy(state, body) {
  if (!state.globalEnergyGains || !state.globalEnergyCosts) return;

  state.globalEnergyGains.photosynthesis += body.energyGainedFromPhotosynthesis || 0;
  state.globalEnergyGains.eating += body.energyGainedFromEating || 0;
  state.globalEnergyGains.predation += body.energyGainedFromPredation || 0;

  state.globalEnergyCosts.baseNodes += body.energyCostFromBaseNodes || 0;
  state.globalEnergyCosts.emitterNodes += body.energyCostFromEmitterNodes || 0;
  state.globalEnergyCosts.eaterNodes += body.energyCostFromEaterNodes || 0;
  state.globalEnergyCosts.predatorNodes += body.energyCostFromPredatorNodes || 0;
  state.globalEnergyCosts.neuronNodes += body.energyCostFromNeuronNodes || 0;
  state.globalEnergyCosts.swimmerNodes += body.energyCostFromSwimmerNodes || 0;
  state.globalEnergyCosts.photosyntheticNodes += body.energyCostFromPhotosyntheticNodes || 0;
  state.globalEnergyCosts.grabbingNodes += body.energyCostFromGrabbingNodes || 0;
  state.globalEnergyCosts.eyeNodes += body.energyCostFromEyeNodes || 0;
  state.globalEnergyCosts.jetNodes += body.energyCostFromJetNodes || 0;
  state.globalEnergyCosts.attractorNodes += body.energyCostFromAttractorNodes || 0;
  state.globalEnergyCosts.repulsorNodes += body.energyCostFromRepulsorNodes || 0;
}

/**
 * Advance particles and cull expired entries.
 */
function removeDeadParticles(state, dt, rng) {
  for (let i = state.particles.length - 1; i >= 0; i--) {
    const particle = state.particles[i];
    withRandomSource(rng, () => particle.update(dt));
    if (particle.life <= 0) {
      state.particles.splice(i, 1);
    }
  }
}

function shuffleArrayInPlace(items, rng = Math.random) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.max(0, Math.min(0.999999, Number(rng()) || 0)) * (i + 1));
    const tmp = items[i];
    items[i] = items[j];
    items[j] = tmp;
  }
  return items;
}

function normalizeCreatureExecutionMode(mode) {
  const value = String(mode || '').trim().toLowerCase();
  if (!value) return 'legacy_reverse';

  if (value === 'legacy' || value === 'legacy_reverse' || value === 'reverse') {
    return 'legacy_reverse';
  }
  if (value === 'islands' || value === 'islands_deterministic' || value === 'islands_serial') {
    return 'islands_deterministic';
  }
  if (value === 'islands_shuffled' || value === 'shuffled' || value === 'islands_randomized') {
    return 'islands_shuffled';
  }
  return 'legacy_reverse';
}

function resolveCreatureExecutionPlan(state, runtimeConfig, constants, rng, {
  creatureExecutionMode = null,
  creatureIslandNeighborRadiusCells = null,
  creatureShuffleWithinIsland = false
} = {}) {
  const mode = normalizeCreatureExecutionMode(
    creatureExecutionMode ?? runtimeConfig.CREATURE_EXECUTION_MODE ?? 'legacy_reverse'
  );

  if (mode === 'legacy_reverse') {
    const order = [];
    for (let i = state.softBodyPopulation.length - 1; i >= 0; i--) {
      order.push(state.softBodyPopulation[i]);
    }
    return {
      order,
      telemetry: {
        mode,
        islandCount: 0,
        largestIslandSize: 0,
        avgIslandSize: 0,
        neighborRadiusCells: null,
        shuffled: false,
        shuffledWithinIsland: false,
        bodyCount: order.length
      }
    };
  }

  const gridCellSize = constants.GRID_CELL_SIZE ?? runtimeConfig.GRID_CELL_SIZE;
  const autoNeighborRadius = computeIslandNeighborRadiusCells(runtimeConfig, gridCellSize);
  const neighborRadiusCells = Number.isFinite(Number(creatureIslandNeighborRadiusCells))
    ? Math.max(0, Math.floor(Number(creatureIslandNeighborRadiusCells)))
    : autoNeighborRadius;

  const islandBuild = buildCreatureInteractionIslands({
    softBodyPopulation: state.softBodyPopulation,
    spatialGrid: state.spatialGrid,
    gridCols: runtimeConfig.GRID_COLS,
    gridRows: runtimeConfig.GRID_ROWS,
    neighborRadiusCells
  });

  const islands = islandBuild.islands.map((group) => group.slice());
  const shuffled = mode === 'islands_shuffled';
  if (shuffled) {
    shuffleArrayInPlace(islands, rng);
    if (creatureShuffleWithinIsland) {
      for (const group of islands) {
        shuffleArrayInPlace(group, rng);
      }
    }
  }

  const order = islands.flat();
  const islandSizes = islands.map((g) => g.length);
  const largestIslandSize = islandSizes.length > 0 ? Math.max(...islandSizes) : 0;
  const avgIslandSize = islandSizes.length > 0
    ? islandSizes.reduce((sum, n) => sum + n, 0) / islandSizes.length
    : 0;

  return {
    order,
    telemetry: {
      mode,
      islandCount: islands.length,
      largestIslandSize,
      avgIslandSize: round(avgIslandSize, 4),
      neighborRadiusCells,
      autoNeighborRadius,
      shuffled,
      shuffledWithinIsland: Boolean(creatureShuffleWithinIsland),
      bodyCount: order.length,
      occupiedCells: islandBuild.occupiedCells,
      graphEdgeCount: islandBuild.edgeCount
    }
  };
}

function maybeLogInstabilityDiagnostic(telemetry, removalEvent, {
  diagnosticEveryN = 100,
  diagnosticReasons = null
} = {}) {
  const everyN = Math.max(1, Math.floor(Number(diagnosticEveryN) || 100));
  const reason = String(removalEvent?.unstableReason || 'unknown');

  const watched = Array.isArray(diagnosticReasons) && diagnosticReasons.length > 0
    ? new Set(diagnosticReasons.map((r) => String(r)))
    : new Set(['physics_invalid_motion', 'physics_nan_position', 'physics_non_finite_position', 'physics_invalid_motion_or_nan']);

  if (!watched.has(reason)) return;

  const reasonCount = Number(telemetry?.removedByReason?.[reason]) || 0;
  if (reasonCount <= 0 || (reasonCount % everyN) !== 0) return;

  const sample = {
    sampledAt: new Date().toISOString(),
    reasonCountForReason: reasonCount,
    event: removalEvent
  };

  telemetry.sampledDiagnostics.push(sample);
  if (telemetry.sampledDiagnostics.length > telemetry.maxSampledDiagnostics) {
    telemetry.sampledDiagnostics.splice(0, telemetry.sampledDiagnostics.length - telemetry.maxSampledDiagnostics);
  }

  try {
    console.warn(`[instability-diagnostic] ${JSON.stringify(sample)}`);
  } catch {
    console.warn('[instability-diagnostic] (json serialization failed)');
  }
}

/**
 * Remove unstable bodies and capture rich removal telemetry for diagnostics.
 */
function removeUnstableBodies(state, {
  captureInstabilityTelemetry = true,
  maxRecentDeaths = 1000,
  diagnosticEveryN = 100,
  diagnosticReasons = null
} = {}) {
  let removedCount = 0;
  const removedBodies = [];
  const telemetry = ensureInstabilityTelemetryState(state, maxRecentDeaths);

  for (let i = state.softBodyPopulation.length - 1; i >= 0; i--) {
    const body = state.softBodyPopulation[i];
    if (!body.isUnstable) continue;

    const removalEvent = buildInstabilityRemovalEvent(state, body);

    if (captureInstabilityTelemetry) {
      telemetry.totalRemoved += 1;
      if (removalEvent.unstableClass === 'physics') telemetry.totalPhysicsRemoved += 1;
      else if (removalEvent.unstableClass === 'non_physics') telemetry.totalNonPhysicsRemoved += 1;
      else telemetry.totalUnknownRemoved += 1;

      telemetry.removedByReason[removalEvent.unstableReason] = (telemetry.removedByReason[removalEvent.unstableReason] || 0) + 1;
      if (removalEvent.unstablePhysicsKind) {
        telemetry.removedByPhysicsKind[removalEvent.unstablePhysicsKind] = (telemetry.removedByPhysicsKind[removalEvent.unstablePhysicsKind] || 0) + 1;
      }
      telemetry.removedByBirthOrigin[removalEvent.birthOrigin] = (telemetry.removedByBirthOrigin[removalEvent.birthOrigin] || 0) + 1;
      telemetry.removedByLifecycleStage[removalEvent.lifecycleStage] = (telemetry.removedByLifecycleStage[removalEvent.lifecycleStage] || 0) + 1;
      telemetry.recentDeaths.push(removalEvent);
      if (telemetry.recentDeaths.length > telemetry.maxRecentDeaths) {
        telemetry.recentDeaths.splice(0, telemetry.recentDeaths.length - telemetry.maxRecentDeaths);
      }

      maybeLogInstabilityDiagnostic(telemetry, removalEvent, {
        diagnosticEveryN,
        diagnosticReasons
      });
    }

    removedBodies.push(removalEvent);
    accumulateRemovedBodyEnergy(state, body);
    state.softBodyPopulation.splice(i, 1);
    removedCount++;
  }

  return { removedCount, removedBodies };
}

/**
 * Execute one simulation tick using the shared real-engine update path.
 *
 * @param {object} state - Mutable world state (bodies, particles, fields, grid).
 * @param {number} dt - Delta time in seconds.
 * @param {object} options - Runtime controls and injectable classes.
 * @returns {{removedCount:number,removedBodies:object[],currentAnyUnstable:boolean,spawnTelemetry:object,reproductionTelemetry:object,computeTelemetry:object,populations:{creatures:number,particles:number}}}
 */
export function stepWorld(state, dt, options = {}) {
  const tStep0 = Date.now();
  const {
    config,
    configViews = null,
    rng = Math.random,
    SoftBodyClass = null,
    ParticleClass = null,
    allowReproduction = true,
    maintainCreatureFloor = true,
    maintainParticleFloor = true,
    applyEmitters = true,
    applySelectedPointPush = true,
    creatureSpawnMargin = 50,
    creatureExecutionMode = null,
    creatureIslandNeighborRadiusCells = null,
    creatureShuffleWithinIsland = false,
    captureInstabilityTelemetry = true,
    maxRecentInstabilityDeaths = 1000,
    instabilityDiagnosticEveryN = null,
    instabilityDiagnosticReasons = null
  } = options;

  const { runtime: runtimeConfig, constants } = resolveConfigViews(configViews || config);
  if (!runtimeConfig) {
    throw new Error('stepWorld requires options.config or options.configViews');
  }

  const stepTiming = {
    totalMs: 0,
    newbornStabilizeMs: 0,
    spatialGridMs: 0,
    emittersMs: 0,
    particlesMs: 0,
    fluidMs: 0,
    creatureUpdateMs: 0,
    reproductionMs: 0,
    telemetryMs: 0,
    cleanupMs: 0,
    floorMs: 0
  };

  state.simulationStep = (Number(state.simulationStep) || 0) + 1;
  ensureInstabilityTelemetryState(state, maxRecentInstabilityDeaths);

  // One-time newborn stabilization pass for already-present creatures (initial population,
  // freshly loaded worlds, or externally inserted bodies).
  let t0 = Date.now();
  for (const body of state.softBodyPopulation) {
    if (!body || body.isUnstable || body.__newbornStabilityApplied) continue;
    if (Number.isFinite(Number(body.ticksSinceBirth)) && Number(body.ticksSinceBirth) > 1) {
      body.__newbornStabilityApplied = true;
      continue;
    }

    stabilizeNewbornBody(body, {
      config: runtimeConfig,
      dt
    });
    body.__newbornStabilityApplied = true;
  }

  stepTiming.newbornStabilizeMs += (Date.now() - t0);

  t0 = Date.now();
  updateSpatialGrid(state, runtimeConfig, constants);
  stepTiming.spatialGridMs += (Date.now() - t0);

  t0 = Date.now();
  if (applyEmitters) {
    applyVelocityEmitters(state, runtimeConfig);
  }
  stepTiming.emittersMs += (Date.now() - t0);

  t0 = Date.now();
  if (ParticleClass && state.fluidField) {
    if (maintainParticleFloor && state.particles.length < runtimeConfig.PARTICLE_POPULATION_FLOOR) {
      let particlesToSpawnToFloor = runtimeConfig.PARTICLE_POPULATION_FLOOR - state.particles.length;
      for (let i = 0; i < particlesToSpawnToFloor; i++) {
        if (state.particles.length >= runtimeConfig.PARTICLE_POPULATION_CEILING) break;
        spawnParticle(state, runtimeConfig, ParticleClass, rng);
      }
      runtimeConfig.particleEmissionDebt = 0;
    } else if (
      state.particles.length < runtimeConfig.PARTICLE_POPULATION_CEILING &&
      runtimeConfig.PARTICLES_PER_SECOND > 0
    ) {
      runtimeConfig.particleEmissionDebt += runtimeConfig.PARTICLES_PER_SECOND * dt;
      while (runtimeConfig.particleEmissionDebt >= 1 && state.particles.length < runtimeConfig.PARTICLE_POPULATION_CEILING) {
        spawnParticle(state, runtimeConfig, ParticleClass, rng);
        runtimeConfig.particleEmissionDebt -= 1;
      }
    }
  }
  stepTiming.particlesMs += (Date.now() - t0);

  t0 = Date.now();
  if (applySelectedPointPush) {
    maybeApplySelectedPointFluidPush(state, runtimeConfig);
  }

  applyLandscapeDyeEmitters(state, runtimeConfig, dt, rng);
  applyLandscapeVelocityEmitters(state, runtimeConfig, dt, rng);

  if (state.fluidField) {
    if (typeof state.fluidField.seedCarrierTilesFromBodies === 'function') {
      state.fluidField.seedCarrierTilesFromBodies(state.softBodyPopulation);
    }
    const fluidStepEvery = Math.max(1, Math.floor(Number(runtimeConfig.FLUID_STEP_EVERY_N_TICKS) || 1));
    const worldTick = Math.max(0, Math.floor(Number(state.tick) || 0));
    if ((worldTick % fluidStepEvery) === 0) {
      state.fluidField.dt = dt * fluidStepEvery;
      state.fluidField.step(worldTick);
    }
  }
  stepTiming.fluidMs += (Date.now() - t0);

  const creatureCeiling = runtimeConfig.CREATURE_POPULATION_CEILING;
  const canCreaturesReproduceGlobally = allowReproduction && state.softBodyPopulation.length < creatureCeiling;
  const newOffspring = [];
  let reproductionBirths = 0;
  let floorSpawns = 0;
  let currentAnyUnstable = false;
  const reproductionTelemetry = {
    consideredBodies: 0,
    attemptedParents: 0,
    successfulParents: 0,
    successfulBirths: 0,
    attemptsWithoutBirths: 0,
    suppressedByGlobalDisabled: 0,
    suppressedByGlobalCeiling: 0,
    suppressedByCanReproduce: 0,
    suppressedByEnergy: 0,
    suppressedByCooldown: 0,
    suppressedByResources: 0,
    suppressedByDensity: 0,
    suppressedByFertilityRoll: 0,
    suppressedByDye: 0,
    suppressedByPlacementOrOther: 0
  };

  const creatureExecutionPlan = resolveCreatureExecutionPlan(state, runtimeConfig, constants, rng, {
    creatureExecutionMode,
    creatureIslandNeighborRadiusCells,
    creatureShuffleWithinIsland
  });
  state.lastComputeTelemetry = creatureExecutionPlan.telemetry;

  const creatureLoopStart = Date.now();
  for (const body of creatureExecutionPlan.order) {
    if (!body || body.isUnstable) continue;

    let tUpdate = Date.now();
    withRandomSource(rng, () => body.updateSelf(dt, state.fluidField));
    stepTiming.creatureUpdateMs += (Date.now() - tUpdate);
    if (body.isUnstable) {
      currentAnyUnstable = true;
      continue;
    }

    reproductionTelemetry.consideredBodies += 1;

    if (!allowReproduction) {
      reproductionTelemetry.suppressedByGlobalDisabled += 1;
      continue;
    }

    const remainingCreatureSlots = creatureCeiling - (state.softBodyPopulation.length + newOffspring.length);
    if (!canCreaturesReproduceGlobally || remainingCreatureSlots <= 0) {
      reproductionTelemetry.suppressedByGlobalCeiling += 1;
      continue;
    }

    if (!body.canReproduce) {
      reproductionTelemetry.suppressedByCanReproduce += 1;
      continue;
    }

    if (body.creatureEnergy < body.reproductionEnergyThreshold) {
      reproductionTelemetry.suppressedByEnergy += 1;
      continue;
    }

    const cooldownBefore = Number(body.failedReproductionCooldown) || 0;
    const densityBefore = Number(body.reproductionSuppressedByDensity) || 0;
    const resourcesBefore = Number(body.reproductionSuppressedByResources) || 0;
    const fertilityBefore = Number(body.reproductionSuppressedByFertilityRoll) || 0;
    const dyeBefore = Number(body.reproductionSuppressedByDye) || 0;

    reproductionTelemetry.attemptedParents += 1;
    const offspring = withRandomSource(rng, () => body.reproduce({ maxOffspring: remainingCreatureSlots }));

    const densityDelta = Math.max(0, (Number(body.reproductionSuppressedByDensity) || 0) - densityBefore);
    const resourcesDelta = Math.max(0, (Number(body.reproductionSuppressedByResources) || 0) - resourcesBefore);
    const fertilityDelta = Math.max(0, (Number(body.reproductionSuppressedByFertilityRoll) || 0) - fertilityBefore);
    const dyeDelta = Math.max(0, (Number(body.reproductionSuppressedByDye) || 0) - dyeBefore);

    reproductionTelemetry.suppressedByDensity += densityDelta;
    reproductionTelemetry.suppressedByResources += resourcesDelta;
    reproductionTelemetry.suppressedByFertilityRoll += fertilityDelta;
    reproductionTelemetry.suppressedByDye += dyeDelta;

    if (cooldownBefore > 0) {
      reproductionTelemetry.suppressedByCooldown += 1;
    }

    if (offspring && offspring.length) {
      reproductionBirths += offspring.length;
      reproductionTelemetry.successfulParents += 1;
      reproductionTelemetry.successfulBirths += offspring.length;
      newOffspring.push(...offspring);
      continue;
    }

    reproductionTelemetry.attemptsWithoutBirths += 1;

    if (
      cooldownBefore <= 0 &&
      densityDelta === 0 &&
      resourcesDelta === 0 &&
      fertilityDelta === 0 &&
      dyeDelta === 0
    ) {
      reproductionTelemetry.suppressedByPlacementOrOther += 1;
    }
  }
  const creatureLoopMs = Date.now() - creatureLoopStart;
  stepTiming.reproductionMs += Math.max(0, creatureLoopMs - stepTiming.creatureUpdateMs);

  t0 = Date.now();
  if (newOffspring.length) {
    for (const child of newOffspring) {
      stabilizeNewbornBody(child, {
        config: runtimeConfig,
        dt
      });
      child.__newbornStabilityApplied = true;
    }
    state.softBodyPopulation.push(...newOffspring);
  }

  updateEdgeLengthTelemetry(state, runtimeConfig);
  stepTiming.telemetryMs += (Date.now() - t0);

  t0 = Date.now();
  removeDeadParticles(state, dt, rng);

  if (currentAnyUnstable && !runtimeConfig.isAnySoftBodyUnstable) {
    runtimeConfig.isAnySoftBodyUnstable = true;
  } else if (!currentAnyUnstable && runtimeConfig.isAnySoftBodyUnstable && !state.softBodyPopulation.some((b) => b.isUnstable)) {
    runtimeConfig.isAnySoftBodyUnstable = false;
  }

  const removal = removeUnstableBodies(state, {
    captureInstabilityTelemetry,
    maxRecentDeaths: maxRecentInstabilityDeaths,
    diagnosticEveryN: instabilityDiagnosticEveryN ?? runtimeConfig.INSTABILITY_DIAGNOSTIC_EVERY_N ?? 100,
    diagnosticReasons: instabilityDiagnosticReasons ?? runtimeConfig.INSTABILITY_DIAGNOSTIC_REASONS ?? null
  });
  stepTiming.cleanupMs += (Date.now() - t0);
  const removedCount = removal.removedCount;
  const removedBodies = removal.removedBodies;

  t0 = Date.now();
  if (maintainCreatureFloor && SoftBodyClass) {
    const neededToMaintainFloor = runtimeConfig.CREATURE_POPULATION_FLOOR - state.softBodyPopulation.length;
    if (neededToMaintainFloor > 0) {
      for (let i = 0; i < neededToMaintainFloor; i++) {
        if (state.softBodyPopulation.length >= runtimeConfig.CREATURE_POPULATION_CEILING) break;
        const spawned = spawnCreature(state, runtimeConfig, SoftBodyClass, rng, dt, creatureSpawnMargin);
        if (spawned) floorSpawns += 1;
      }
    }
  }
  stepTiming.floorMs += (Date.now() - t0);
  stepTiming.totalMs = Date.now() - tStep0;

  return {
    removedCount,
    removedBodies,
    currentAnyUnstable,
    spawnTelemetry: {
      reproductionBirths,
      floorSpawns,
      totalSpawns: reproductionBirths + floorSpawns
    },
    reproductionTelemetry,
    computeTelemetry: creatureExecutionPlan.telemetry,
    stepTiming,
    populations: {
      creatures: state.softBodyPopulation.length,
      particles: state.particles.length
    }
  };
}

/**
 * Public helper to rebuild spatial occupancy after bulk state changes (e.g. load).
 */
export function rebuildSpatialGrid(state, configOrViews) {
  const { runtime, constants } = resolveConfigViews(configOrViews);
  updateSpatialGrid(state, runtime, constants);
}
