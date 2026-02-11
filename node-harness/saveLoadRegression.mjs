#!/usr/bin/env node
import config from '../js/config.js';
import { SoftBody } from '../js/classes/SoftBody.js';
import { Particle } from '../js/classes/Particle.js';
import { stepWorld } from '../js/engine/stepWorld.mjs';
import { getScenario } from './scenarios.mjs';
import { RealWorld } from './realWorld.mjs';

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

function round(v, digits = 6) {
  const p = 10 ** digits;
  return Math.round((Number(v) || 0) * p) / p;
}

class BrowserLikeWorld extends RealWorld {
  step(dt) {
    this.tick += 1;
    this.time += dt;

    this._applyEvents();

    stepWorld(this.worldState, dt, {
      configViews: this.configViews,
      config,
      rng: this.rand,
      SoftBodyClass: SoftBody,
      ParticleClass: Particle,
      allowReproduction: true,
      maintainCreatureFloor: true,
      maintainParticleFloor: true,
      applyEmitters: true,
      applySelectedPointPush: false,
      creatureSpawnMargin: 50
    });

    this._syncAliasesFromWorldState();
  }
}

function summarizeInvariants(world) {
  let totalEnergy = 0;
  let massPoints = 0;
  let springs = 0;
  for (const body of world.softBodyPopulation) {
    if (body.isUnstable) continue;
    totalEnergy += Number(body.creatureEnergy || 0);
    massPoints += Array.isArray(body.massPoints) ? body.massPoints.length : 0;
    springs += Array.isArray(body.springs) ? body.springs.length : 0;
  }

  let fluidDye = 0;
  let fluidVel = 0;
  const f = world.fluidField;
  if (f && f.densityR && f.densityG && f.densityB && f.Vx && f.Vy) {
    const len = Math.min(f.densityR.length, f.densityG.length, f.densityB.length, f.Vx.length, f.Vy.length);
    for (let i = 0; i < len; i++) {
      fluidDye += (f.densityR[i] || 0) + (f.densityG[i] || 0) + (f.densityB[i] || 0);
      const vx = f.Vx[i] || 0;
      const vy = f.Vy[i] || 0;
      fluidVel += Math.sqrt(vx * vx + vy * vy);
    }
  }

  return {
    tick: world.tick,
    time: round(world.time, 6),
    creatures: world.softBodyPopulation.length,
    particles: world.particles.length,
    massPoints,
    springs,
    totalEnergy: round(totalEnergy, 6),
    fluidDye: round(fluidDye, 5),
    fluidVel: round(fluidVel, 6)
  };
}

function compareInvariant(a, b, tolerances) {
  const diffs = [];
  const exactKeys = ['tick', 'creatures', 'particles', 'massPoints', 'springs'];
  for (const key of exactKeys) {
    if (a[key] !== b[key]) {
      diffs.push({ key, expected: a[key], actual: b[key] });
    }
  }

  const numericKeys = ['time', 'totalEnergy', 'fluidDye', 'fluidVel'];
  for (const key of numericKeys) {
    const delta = Math.abs(a[key] - b[key]);
    const tol = tolerances[key] ?? 1e-6;
    if (delta > tol) {
      diffs.push({ key, expected: a[key], actual: b[key], delta, tolerance: tol });
    }
  }
  return diffs;
}

function runSaveLoadRegression(WorldClass, runtimeScenario, seed, beforeSteps, afterSteps, tolerances) {
  const dt = runtimeScenario.dt;
  const base = new WorldClass(runtimeScenario, seed);

  for (let i = 0; i < beforeSteps; i++) {
    base.step(dt);
  }

  const saved = base.saveStateSnapshot({
    tick: base.tick,
    time: base.time,
    scenario: runtimeScenario.name
  });

  const restored = new WorldClass(runtimeScenario, seed + 1000);
  restored.loadStateSnapshot(saved);

  const mismatches = [];
  for (let i = 0; i < afterSteps; i++) {
    base.step(dt);
    restored.step(dt);

    const left = summarizeInvariants(base);
    const right = summarizeInvariants(restored);
    const diffs = compareInvariant(left, right, tolerances);
    if (diffs.length > 0) {
      mismatches.push({ step: i + 1, left, right, diffs });
      break;
    }
  }

  return {
    pass: mismatches.length === 0,
    mismatches
  };
}

const scenarioName = arg('scenario', 'micro_one_creature_100');
const seed = (toNumber(arg('seed', '42'), 42) >>> 0);
const beforeSteps = Math.max(1, Math.floor(toNumber(arg('beforeSteps', '30'), 30)));
const afterSteps = Math.max(1, Math.floor(toNumber(arg('afterSteps', '60'), 60)));
const dtOverride = toNumber(arg('dt', null), null);
const runtime = arg('runtime', 'both');

const scenario = getScenario(scenarioName);
const runtimeScenario = {
  ...scenario,
  dt: dtOverride ?? scenario.dt
};

const tolerances = {
  time: toNumber(arg('timeTol', '1e-9'), 1e-9),
  totalEnergy: toNumber(arg('energyTol', '1e-6'), 1e-6),
  fluidDye: toNumber(arg('fluidDyeTol', '1e-4'), 1e-4),
  fluidVel: toNumber(arg('fluidVelTol', '1e-4'), 1e-4)
};

const results = [];
if (runtime === 'both' || runtime === 'headless') {
  results.push({
    runtime: 'headless-real',
    ...runSaveLoadRegression(RealWorld, runtimeScenario, seed, beforeSteps, afterSteps, tolerances)
  });
}
if (runtime === 'both' || runtime === 'browserLike') {
  results.push({
    runtime: 'browser-like-real',
    ...runSaveLoadRegression(BrowserLikeWorld, runtimeScenario, seed, beforeSteps, afterSteps, tolerances)
  });
}

const pass = results.every((r) => r.pass);
console.log(JSON.stringify({
  scenario: runtimeScenario.name,
  seed,
  beforeSteps,
  afterSteps,
  dt: runtimeScenario.dt,
  runtime,
  pass,
  results
}, null, 2));

process.exit(pass ? 0 : 1);
