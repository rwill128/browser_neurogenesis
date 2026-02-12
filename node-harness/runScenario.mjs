#!/usr/bin/env node
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { getScenario } from './scenarios.mjs';
import { RealWorld } from './realWorld.mjs';

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

function parseBoolArg(name, fallback) {
  const raw = arg(name, null);
  if (raw === null || raw === undefined) return fallback;
  const v = String(raw).trim().toLowerCase();
  if (['1', 'true', 'yes', 'y', 'on'].includes(v)) return true;
  if (['0', 'false', 'no', 'n', 'off'].includes(v)) return false;
  throw new Error(`Invalid boolean for --${name}: ${raw}`);
}

function parseNum(raw, fallback = null) {
  if (raw === null || raw === undefined || raw === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

const scenarioName = arg('scenario', 'micro_stability');
const seed = Number(arg('seed', '42'));
const stepsArg = arg('steps', null);
const outDir = resolve(arg('out', './artifacts'));

if (arg('engine', null) !== null || arg('allowMini', null) !== null) {
  throw new Error('Engine selection is no longer supported. This runner always uses the real simulation code path.');
}

const scenario = getScenario(scenarioName);
const worldW = parseNum(arg('worldW', null), null);
const worldH = parseNum(arg('worldH', null), null);
const creaturesArg = parseNum(arg('creatures', null), null);
const creatureFloorArg = parseNum(arg('creatureFloor', null), null);
const creatureCeilingArg = parseNum(arg('creatureCeiling', null), null);
const particlesArg = parseNum(arg('particles', null), null);
const particleFloorArg = parseNum(arg('particleFloor', null), null);
const particleCeilingArg = parseNum(arg('particleCeiling', null), null);
const particlesPerSecondArg = parseNum(arg('particlesPerSecond', null), null);
const dtArg = parseNum(arg('dt', null), null);

const creatureFloor = Math.max(0, Math.floor(
  creatureFloorArg ?? creaturesArg ?? scenario.creatureFloor ?? scenario.creatures ?? 0
));
const creatureCeiling = Math.max(
  creatureFloor,
  Math.floor(
    creatureCeilingArg ??
    scenario.creatureCeiling ??
    (creatureFloor > 0 ? Math.max(creatureFloor + 1, Math.ceil(creatureFloor * 2)) : 0)
  )
);

const particleFloor = Math.max(0, Math.floor(
  particleFloorArg ?? particlesArg ?? scenario.particleFloor ?? scenario.particles ?? 0
));
const particleCeiling = Math.max(
  particleFloor,
  Math.floor(
    particleCeilingArg ??
    scenario.particleCeiling ??
    (particleFloor > 0 ? Math.max(particleFloor + 1, Math.ceil(particleFloor * 2)) : 0)
  )
);

const runtimeScenario = {
  ...scenario,
  world: {
    width: Number.isFinite(worldW) ? worldW : scenario.world.width,
    height: Number.isFinite(worldH) ? worldH : scenario.world.height
  },
  creatureFloor,
  creatureCeiling,
  particleFloor,
  particleCeiling,
  particlesPerSecond: Number.isFinite(particlesPerSecondArg)
    ? particlesPerSecondArg
    : (Number.isFinite(Number(scenario.particlesPerSecond)) ? Number(scenario.particlesPerSecond) : 0),
  dt: Number.isFinite(dtArg) ? dtArg : scenario.dt,
  stepBehavior: {
    allowReproduction: parseBoolArg('allowReproduction', true),
    maintainCreatureFloor: parseBoolArg('maintainCreatureFloor', true),
    maintainParticleFloor: parseBoolArg('maintainParticleFloor', true),
    applyEmitters: parseBoolArg('applyEmitters', true),
    applySelectedPointPush: parseBoolArg('applySelectedPointPush', false),
    captureInstabilityTelemetry: parseBoolArg('captureInstabilityTelemetry', true),
    maxRecentInstabilityDeaths: parseNum(arg('maxRecentInstabilityDeaths', null), 5000),
    creatureSpawnMargin: parseNum(arg('creatureSpawnMargin', null), 50)
  }
};

const steps = stepsArg ? Number(stepsArg) : runtimeScenario.steps;
const world = new RealWorld(runtimeScenario, seed);

mkdirSync(outDir, { recursive: true });
const instabilityPath = resolve(outDir, `${scenario.name}-seed${seed}-steps${steps}-instability-deaths.jsonl`);
writeFileSync(instabilityPath, '', 'utf8');

const timeline = [];
let totalReproductionBirths = 0;
let totalFloorSpawns = 0;
for (let i = 0; i < steps; i++) {
  const stepResult = world.step(runtimeScenario.dt);
  totalReproductionBirths += Number(stepResult?.spawnTelemetry?.reproductionBirths || 0);
  totalFloorSpawns += Number(stepResult?.spawnTelemetry?.floorSpawns || 0);

  if (Array.isArray(stepResult?.removedBodies) && stepResult.removedBodies.length) {
    for (const death of stepResult.removedBodies) {
      appendFileSync(instabilityPath, JSON.stringify(death) + '\n', 'utf8');
      console.warn(`[UNSTABLE_DEATH] ${JSON.stringify(death)}`);
    }
  }

  if (i % 10 === 0 || i === steps - 1) timeline.push(world.snapshot());
}

const outPath = resolve(outDir, `${scenario.name}-seed${seed}-steps${steps}.json`);
const payload = {
  scenario: runtimeScenario.name,
  engine: 'real',
  seed,
  steps,
  dt: runtimeScenario.dt,
  world: runtimeScenario.world,
  creatures: runtimeScenario.creatureFloor,
  particles: runtimeScenario.particleFloor,
  creatureFloor: runtimeScenario.creatureFloor,
  creatureCeiling: runtimeScenario.creatureCeiling,
  particleFloor: runtimeScenario.particleFloor,
  particleCeiling: runtimeScenario.particleCeiling,
  particlesPerSecond: runtimeScenario.particlesPerSecond,
  stepBehavior: runtimeScenario.stepBehavior,
  generatedAt: new Date().toISOString(),
  instabilityTelemetry: world.worldState.instabilityTelemetry || {},
  spawnTelemetry: {
    totalReproductionBirths,
    totalFloorSpawns,
    totalSpawns: totalReproductionBirths + totalFloorSpawns
  },
  artifacts: {
    instabilityDeathsPath: instabilityPath
  },
  timeline
};
writeFileSync(outPath, JSON.stringify(payload, null, 2), 'utf8');
console.log(`Wrote ${outPath}`);
console.log(`Wrote ${instabilityPath}`);
