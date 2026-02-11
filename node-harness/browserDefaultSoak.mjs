#!/usr/bin/env node

/**
 * Browser-default soak runner (real path only).
 *
 * Purpose:
 * - Reproduce browser-runtime issues without needing a live browser session.
 * - Use index.html-like startup defaults (world size, creature floor, particle emission, etc.).
 * - Run for a long time and persist crash snapshots/reports automatically.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

import config from '../js/config.js';
import { SoftBody } from '../js/classes/SoftBody.js';
import { Particle } from '../js/classes/Particle.js';
import { Spring } from '../js/classes/Spring.js';
import { FluidField } from '../js/classes/FluidField.js';
import { stepWorld } from '../js/engine/stepWorld.mjs';
import { createWorldState } from '../js/engine/worldState.mjs';
import { createConfigViews } from '../js/engine/configViews.mjs';
import {
  initializeSpatialGrid,
  initializeEnvironmentMaps,
  initializeParticles,
  initializePopulation
} from '../js/engine/initWorld.mjs';
import { saveWorldStateSnapshot } from '../js/engine/worldPersistence.mjs';
import { syncRuntimeState } from '../js/engine/runtimeState.js';
import { createSeededRandom } from './seededRandomScope.mjs';

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

function toNumber(value, fallback) {
  if (value === null || value === undefined || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function toInt(value, fallback) {
  const n = toNumber(value, fallback);
  return Number.isFinite(n) ? Math.floor(n) : fallback;
}

function nowStamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function safeWriteJson(filePath, data) {
  writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n', 'utf8');
}

function buildMutationStatsShape() {
  return {
    springStiffness: 0,
    springDamping: 0,
    motorInterval: 0,
    motorCap: 0,
    emitterStrength: 0,
    emitterDirection: 0,
    numOffspring: 0,
    offspringSpawnRadius: 0,
    pointAddChanceGene: 0,
    springConnectionRadiusGene: 0,
    reproductionEnergyThreshold: 0,
    nodeTypeChange: 0,
    movementTypeChange: 0,
    springDeletion: 0,
    springAddition: 0,
    springRestLength: 0,
    springRigidityFlip: 0,
    pointAddActual: 0,
    springSubdivision: 0,
    segmentDuplication: 0,
    symmetricBodyDuplication: 0,
    bodyScale: 0,
    rewardStrategyChange: 0,
    grabberGeneChange: 0,
    eyeTargetTypeChange: 0,
    jetMaxVelocityGene: 0,
    reproductionCooldownGene: 0,
    blueprintMassRadiusChange: 0,
    blueprintDyeColorChange: 0,
    blueprintCoordinateChange: 0,
    blueprintNeuronHiddenSizeChange: 0,
    shapeAddition: 0,
    growthGenomeMutations: 0
  };
}

/**
 * Apply index.html-like runtime defaults so this harness behaves close to browser startup.
 */
function applyBrowserDefaults(options) {
  config.IS_HEADLESS_MODE = true;
  config.USE_GPU_FLUID_SIMULATION = false;
  config.IS_SIMULATION_PAUSED = false;

  config.WORLD_WIDTH = options.worldWidth;
  config.WORLD_HEIGHT = options.worldHeight;

  config.CREATURE_POPULATION_FLOOR = options.creatureFloor;
  config.CREATURE_POPULATION_CEILING = options.creatureCeiling;
  config.PARTICLE_POPULATION_FLOOR = options.particleFloor;
  config.PARTICLE_POPULATION_CEILING = options.particleCeiling;
  config.PARTICLES_PER_SECOND = options.particlesPerSecond;

  config.canCreaturesReproduceGlobally = true;
  config.MAX_DELTA_TIME_MS = options.maxDeltaMs;

  config.GRID_COLS = Math.ceil(config.WORLD_WIDTH / config.GRID_CELL_SIZE);
  config.GRID_ROWS = Math.ceil(config.WORLD_HEIGHT / config.GRID_CELL_SIZE);

  // No UI-driven drag/emitter state in node harness.
  config.selectedSoftBodyPoint = null;
  config.velocityEmitters = [];
}

function createBrowserDefaultWorld(rng, options) {
  applyBrowserDefaults(options);
  const configViews = createConfigViews(config);

  const worldState = createWorldState({
    spatialGrid: null,
    softBodyPopulation: [],
    particles: [],
    nutrientField: null,
    lightField: null,
    viscosityField: null,
    nextSoftBodyId: 0,
    mutationStats: buildMutationStatsShape()
  });

  initializeSpatialGrid(worldState, configViews);

  worldState.fluidField = new FluidField(
    config.FLUID_GRID_SIZE_CONTROL,
    config.FLUID_DIFFUSION,
    config.FLUID_VISCOSITY,
    options.dt,
    config.WORLD_WIDTH / config.FLUID_GRID_SIZE_CONTROL,
    config.WORLD_HEIGHT / config.FLUID_GRID_SIZE_CONTROL
  );

  syncRuntimeState({
    fluidField: worldState.fluidField,
    softBodyPopulation: worldState.softBodyPopulation,
    mutationStats: worldState.mutationStats
  });

  initializeEnvironmentMaps(worldState, {
    configViews,
    config,
    size: Math.round(config.FLUID_GRID_SIZE_CONTROL),
    rng
  });
  worldState.fluidField.setViscosityField(worldState.viscosityField);

  initializeParticles(worldState, {
    configViews,
    config,
    ParticleClass: Particle,
    count: options.initialParticles,
    rng
  });

  initializePopulation(worldState, {
    configViews,
    config,
    SoftBodyClass: SoftBody,
    count: options.initialCreatures,
    spawnMargin: 50,
    rng
  });

  syncRuntimeState({
    fluidField: worldState.fluidField,
    softBodyPopulation: worldState.softBodyPopulation,
    mutationStats: worldState.mutationStats
  });

  return { worldState, configViews };
}

function summarize(worldState, tick, timeSec) {
  let maxPoints = 0;
  let totalPoints = 0;
  let totalSprings = 0;
  let totalEnergy = 0;
  let unstable = 0;

  let growthEvents = 0;
  let growthNodesAdded = 0;
  let growthEnergySpent = 0;
  let growthSuppressedByPopulation = 0;
  let growthSuppressedByEnergy = 0;
  let growthSuppressedByCooldown = 0;
  let rlTopologyResets = 0;
  let topologyVersionTotal = 0;

  let reproductionSuppressedByDensity = 0;
  let reproductionSuppressedByResources = 0;
  let reproductionSuppressedByFertilityRoll = 0;
  let reproductionResourceDebitApplied = 0;

  for (const b of worldState.softBodyPopulation) {
    if (b.isUnstable) unstable++;
    const points = Array.isArray(b.massPoints) ? b.massPoints.length : 0;
    const springs = Array.isArray(b.springs) ? b.springs.length : 0;
    totalPoints += points;
    totalSprings += springs;
    if (points > maxPoints) maxPoints = points;
    totalEnergy += Number(b.creatureEnergy || 0);

    growthEvents += Number(b.growthEventsCompleted || 0);
    growthNodesAdded += Number(b.growthNodesAdded || 0);
    growthEnergySpent += Number(b.totalGrowthEnergySpent || 0);
    growthSuppressedByPopulation += Number(b.growthSuppressedByPopulation || 0);
    growthSuppressedByEnergy += Number(b.growthSuppressedByEnergy || 0);
    growthSuppressedByCooldown += Number(b.growthSuppressedByCooldown || 0);
    rlTopologyResets += Number(b.rlBufferResetsDueToTopology || 0);
    topologyVersionTotal += Number(b.nnTopologyVersion || 0);

    reproductionSuppressedByDensity += Number(b.reproductionSuppressedByDensity || 0);
    reproductionSuppressedByResources += Number(b.reproductionSuppressedByResources || 0);
    reproductionSuppressedByFertilityRoll += Number(b.reproductionSuppressedByFertilityRoll || 0);
    reproductionResourceDebitApplied += Number(b.reproductionResourceDebitApplied || 0);
  }

  return {
    tick,
    timeSec,
    creatures: worldState.softBodyPopulation.length,
    particles: worldState.particles.length,
    unstable,
    totalPoints,
    maxPointsPerCreature: maxPoints,
    totalSprings,
    totalEnergy,
    growthEvents,
    growthNodesAdded,
    growthEnergySpent,
    growthSuppressedByPopulation,
    growthSuppressedByEnergy,
    growthSuppressedByCooldown,
    rlTopologyResets,
    topologyVersionTotal,
    reproductionSuppressedByDensity,
    reproductionSuppressedByResources,
    reproductionSuppressedByFertilityRoll,
    reproductionResourceDebitApplied
  };
}

/**
 * Sanity checks to surface silent corruption before total failure.
 */
function assertFiniteWorld(worldState) {
  for (const b of worldState.softBodyPopulation) {
    if (!Array.isArray(b.massPoints)) continue;
    for (const p of b.massPoints) {
      if (!Number.isFinite(p?.pos?.x) || !Number.isFinite(p?.pos?.y)) {
        throw new Error(`non-finite point position in body ${b.id}`);
      }
      if (!Number.isFinite(p?.prevPos?.x) || !Number.isFinite(p?.prevPos?.y)) {
        throw new Error(`non-finite point prevPos in body ${b.id}`);
      }
    }
    for (const s of (b.springs || [])) {
      if (!Number.isFinite(s?.restLength)) {
        throw new Error(`non-finite spring restLength in body ${b.id}`);
      }
    }
  }
}

const seed = (toInt(arg('seed', '42'), 42) >>> 0);
const steps = Math.max(1, toInt(arg('steps', '20000'), 20000));
const dt = toNumber(arg('dt', '0.01'), 0.01);
const logEvery = Math.max(1, toInt(arg('logEvery', '500'), 500));
const outDir = resolve(arg('out', '/tmp/browser-default-soak'));

const options = {
  dt,
  maxDeltaMs: toNumber(arg('maxDeltaMs', '10'), 10),
  worldWidth: toInt(arg('worldWidth', String(config.WORLD_WIDTH)), config.WORLD_WIDTH),
  worldHeight: toInt(arg('worldHeight', String(config.WORLD_HEIGHT)), config.WORLD_HEIGHT),
  creatureFloor: toInt(arg('creatureFloor', String(config.CREATURE_POPULATION_FLOOR)), config.CREATURE_POPULATION_FLOOR),
  creatureCeiling: toInt(arg('creatureCeiling', String(config.CREATURE_POPULATION_CEILING)), config.CREATURE_POPULATION_CEILING),
  particleFloor: toInt(arg('particleFloor', String(config.PARTICLE_POPULATION_FLOOR)), config.PARTICLE_POPULATION_FLOOR),
  particleCeiling: toInt(arg('particleCeiling', String(config.PARTICLE_POPULATION_CEILING)), config.PARTICLE_POPULATION_CEILING),
  particlesPerSecond: toNumber(arg('particlesPerSecond', String(config.PARTICLES_PER_SECOND)), config.PARTICLES_PER_SECOND),
  initialCreatures: toInt(arg('initialCreatures', String(config.CREATURE_POPULATION_FLOOR)), config.CREATURE_POPULATION_FLOOR),
  initialParticles: toInt(arg('initialParticles', '0'), 0)
};

mkdirSync(outDir, { recursive: true });

const rng = createSeededRandom(seed);
const { worldState, configViews } = createBrowserDefaultWorld(rng, options);

const checkpoints = [];
const startedAt = Date.now();

console.log(`[SOAK] browser-default real-path start seed=${seed} steps=${steps} dt=${dt}`);
console.log(`[SOAK] world=${options.worldWidth}x${options.worldHeight} creatures=${options.initialCreatures}/${options.creatureFloor}-${options.creatureCeiling} particles=${options.initialParticles} pps=${options.particlesPerSecond}`);

let tick = 0;
let crashed = null;

for (tick = 1; tick <= steps; tick++) {
  try {
    stepWorld(worldState, dt, {
      configViews,
      config,
      rng,
      SoftBodyClass: SoftBody,
      ParticleClass: Particle,
      allowReproduction: true,
      maintainCreatureFloor: true,
      maintainParticleFloor: true,
      applyEmitters: true,
      applySelectedPointPush: false,
      creatureSpawnMargin: 50
    });

    assertFiniteWorld(worldState);

    if (tick % logEvery === 0 || tick === 1) {
      const s = summarize(worldState, tick, tick * dt);
      checkpoints.push(s);
      console.log(
        `[SOAK] tick=${s.tick} creatures=${s.creatures} particles=${s.particles} ` +
        `totalPoints=${s.totalPoints} maxPts=${s.maxPointsPerCreature} ` +
        `growthEvents=${s.growthEvents} rlTopologyResets=${s.rlTopologyResets} ` +
        `reproSuppDensity=${s.reproductionSuppressedByDensity} reproSuppResource=${s.reproductionSuppressedByResources}`
      );
    }
  } catch (error) {
    crashed = {
      tick,
      timeSec: tick * dt,
      error: {
        name: error?.name || 'Error',
        message: error?.message || String(error),
        stack: error?.stack || null
      }
    };
    break;
  }
}

const finishedAt = Date.now();
const finalSummary = summarize(worldState, Math.min(tick, steps), Math.min(tick, steps) * dt);

const report = {
  run: {
    kind: 'browser-default-soak',
    seed,
    stepsRequested: steps,
    dt,
    startedAt: new Date(startedAt).toISOString(),
    finishedAt: new Date(finishedAt).toISOString(),
    durationMs: finishedAt - startedAt,
    options
  },
  status: crashed ? 'crashed' : 'ok',
  final: finalSummary,
  checkpoints,
  crash: crashed
};

if (crashed) {
  const crashSnap = saveWorldStateSnapshot({
    worldState,
    configOrViews: configViews,
    rng,
    meta: {
      source: 'node-browser-default-soak',
      seed,
      tick: crashed.tick,
      timeSec: crashed.timeSec,
      reason: crashed.error.message
    }
  });

  const crashBase = `${nowStamp()}-seed${seed}-tick${crashed.tick}`;
  const crashReportPath = resolve(outDir, `${crashBase}-crash-report.json`);
  const crashSnapshotPath = resolve(outDir, `${crashBase}-snapshot.json`);
  safeWriteJson(crashReportPath, report);
  safeWriteJson(crashSnapshotPath, crashSnap);

  console.error(`[SOAK] CRASH at tick=${crashed.tick}: ${crashed.error.message}`);
  console.error(`[SOAK] wrote ${crashReportPath}`);
  console.error(`[SOAK] wrote ${crashSnapshotPath}`);
  process.exit(1);
}

const okReportPath = resolve(outDir, `${nowStamp()}-seed${seed}-ok-report.json`);
safeWriteJson(okReportPath, report);
console.log(`[SOAK] completed ${steps} steps without crash.`);
console.log(`[SOAK] wrote ${okReportPath}`);
