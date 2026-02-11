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

function round(v, digits = 6) {
  const p = 10 ** digits;
  return Math.round((Number(v) || 0) * p) / p;
}

function toNumber(value, fallback) {
  if (value === null || value === undefined || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
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
  const creatures = world.softBodyPopulation || [];
  const particles = world.particles || [];

  let totalEnergy = 0;
  let massPoints = 0;
  let springs = 0;
  for (const body of creatures) {
    if (body.isUnstable) continue;
    totalEnergy += Number(body.creatureEnergy || 0);
    massPoints += Array.isArray(body.massPoints) ? body.massPoints.length : 0;
    springs += Array.isArray(body.springs) ? body.springs.length : 0;
  }

  let fluidDyeSum = 0;
  let fluidVelMagSum = 0;
  const f = world.fluidField;
  if (f && f.densityR && f.densityG && f.densityB && f.Vx && f.Vy) {
    const size = Math.min(f.densityR.length, f.densityG.length, f.densityB.length, f.Vx.length, f.Vy.length);
    for (let i = 0; i < size; i++) {
      fluidDyeSum += (f.densityR[i] || 0) + (f.densityG[i] || 0) + (f.densityB[i] || 0);
      const vx = f.Vx[i] || 0;
      const vy = f.Vy[i] || 0;
      fluidVelMagSum += Math.sqrt(vx * vx + vy * vy);
    }
  }

  return {
    tick: world.tick,
    creatures: creatures.length,
    particles: particles.length,
    massPoints,
    springs,
    totalEnergy: round(totalEnergy, 5),
    fluidDyeSum: round(fluidDyeSum, 4),
    fluidVelMagSum: round(fluidVelMagSum, 5)
  };
}

function runWorld(WorldClass, scenario, seed, dt, steps, checkpointEvery = 10) {
  const world = new WorldClass(scenario, seed);
  const timeline = [];

  for (let i = 0; i < steps; i++) {
    world.step(dt);
    if (i % checkpointEvery === 0 || i === steps - 1) {
      timeline.push(summarizeInvariants(world));
    }
  }

  return { world, timeline };
}

function compareTimelines(aTimeline, bTimeline, { energyTolerance = 1e-4, fluidTolerance = 1e-3 } = {}) {
  const mismatches = [];
  const length = Math.min(aTimeline.length, bTimeline.length);

  for (let i = 0; i < length; i++) {
    const a = aTimeline[i];
    const b = bTimeline[i];

    const exactKeys = ['tick', 'creatures', 'particles', 'massPoints', 'springs'];
    for (const key of exactKeys) {
      if (a[key] !== b[key]) {
        mismatches.push({ index: i, tick: a.tick, key, browserLike: a[key], headlessReal: b[key] });
      }
    }

    const energyDelta = Math.abs(a.totalEnergy - b.totalEnergy);
    if (energyDelta > energyTolerance) {
      mismatches.push({ index: i, tick: a.tick, key: 'totalEnergy', browserLike: a.totalEnergy, headlessReal: b.totalEnergy, delta: energyDelta });
    }

    const dyeDelta = Math.abs(a.fluidDyeSum - b.fluidDyeSum);
    if (dyeDelta > fluidTolerance) {
      mismatches.push({ index: i, tick: a.tick, key: 'fluidDyeSum', browserLike: a.fluidDyeSum, headlessReal: b.fluidDyeSum, delta: dyeDelta });
    }

    const velDelta = Math.abs(a.fluidVelMagSum - b.fluidVelMagSum);
    if (velDelta > fluidTolerance) {
      mismatches.push({ index: i, tick: a.tick, key: 'fluidVelMagSum', browserLike: a.fluidVelMagSum, headlessReal: b.fluidVelMagSum, delta: velDelta });
    }
  }

  if (aTimeline.length !== bTimeline.length) {
    mismatches.push({
      key: 'timelineLength',
      browserLike: aTimeline.length,
      headlessReal: bTimeline.length
    });
  }

  return mismatches;
}

const scenarioName = arg('scenario', 'micro_one_creature_100');
const seed = toNumber(arg('seed', '42'), 42) >>> 0;
const steps = Math.max(1, Math.floor(toNumber(arg('steps', '120'), 120)));
const dtOverride = toNumber(arg('dt', null), null);
const checkpointEvery = Math.max(1, Math.floor(toNumber(arg('checkpointEvery', '10'), 10)));

const scenario = getScenario(scenarioName);
const runtimeScenario = {
  ...scenario,
  dt: dtOverride ?? scenario.dt
};

const browserRun = runWorld(BrowserLikeWorld, runtimeScenario, seed, runtimeScenario.dt, steps, checkpointEvery);
const headlessRun = runWorld(RealWorld, runtimeScenario, seed, runtimeScenario.dt, steps, checkpointEvery);

const mismatches = compareTimelines(browserRun.timeline, headlessRun.timeline, {
  energyTolerance: toNumber(arg('energyTolerance', '1e-4'), 1e-4),
  fluidTolerance: toNumber(arg('fluidTolerance', '1e-3'), 1e-3)
});

const result = {
  scenario: runtimeScenario.name,
  seed,
  steps,
  dt: runtimeScenario.dt,
  checkpointEvery,
  browserLikeSamples: browserRun.timeline.length,
  headlessSamples: headlessRun.timeline.length,
  mismatches: mismatches.slice(0, 50),
  mismatchCount: mismatches.length,
  pass: mismatches.length === 0
};

console.log(JSON.stringify(result, null, 2));
process.exit(result.pass ? 0 : 1);
