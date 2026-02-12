#!/usr/bin/env node

/**
 * Worker thread: owns one authoritative world (isolated module state).
 *
 * Why a worker?
 * - The simulation code currently uses a module-level config singleton (js/config.js).
 * - Workers isolate that singleton per world, making true multi-world safe.
 */

import { parentPort, workerData } from 'node:worker_threads';

import config from '../js/config.js';
import { applyConfigOverrides } from '../js/engine/configOverride.mjs';
import { getScenario } from '../node-harness/scenarios.mjs';
import { scenarioDefs } from '../js/engine/scenarioDefs.mjs';
import { RealWorld } from '../node-harness/realWorld.mjs';

if (!parentPort) {
  throw new Error('worldWorker must be run as a worker thread');
}

function asU32(n, fallback = 23) {
  const x = Number.isFinite(Number(n)) ? Number(n) : fallback;
  return (x >>> 0);
}

let id = String(workerData?.id || 'world');
let scenarioName = String(workerData?.scenario || 'micro_repro_sustain');
let seed = asU32(workerData?.seed, 23);
let paused = Boolean(workerData?.paused);

let dt = 1 / 60;
let world = null;

let accumulatorMs = 0;
let lastStepWallMs = 0;
let stepsThisSecond = 0;
let stepsPerSecond = 0;
let lastStepsPerSecondAt = Date.now();

let crashed = false;
let crash = null;

function resetWorld(nextScenarioName, nextSeed) {
  const s = String(nextScenarioName || '').trim();
  if (!s || !(s in scenarioDefs)) {
    throw new Error(`Unknown scenario: ${s}`);
  }

  scenarioName = s;
  seed = asU32(nextSeed, seed);

  const scenario = getScenario(scenarioName);
  dt = Number(scenario.dt) || (1 / 60);

  world = new RealWorld(scenario, seed);
  accumulatorMs = 0;

  lastStepWallMs = 0;
  stepsThisSecond = 0;
  stepsPerSecond = 0;
  lastStepsPerSecondAt = Date.now();

  crashed = false;
  crash = null;
}

resetWorld(scenarioName, seed);

function computeStatus() {
  return {
    id,
    ok: true,
    scenario: scenarioName,
    seed,
    paused,
    crashed,
    crash,
    dt,
    tick: Number(world?.tick) || 0,
    time: Number(world?.time) || 0,
    world: world?.config?.world || null,
    populations: {
      creatures: Array.isArray(world?.softBodyPopulation) ? world.softBodyPopulation.length : null,
      liveCreatures: Array.isArray(world?.softBodyPopulation) ? world.softBodyPopulation.length : null,
      particles: Array.isArray(world?.particles) ? world.particles.length : null
    },
    stepsPerSecond,
    lastStepWallMs
  };
}

function computeSnapshot(mode = 'render') {
  const snap = world.snapshot();
  const worldDims = world?.config?.world || null;

  if (mode === 'lite') {
    return {
      id,
      scenario: scenarioName,
      tick: snap.tick,
      time: snap.time,
      seed: snap.seed,
      world: worldDims,
      populations: snap.populations,
      instabilityTelemetry: snap.instabilityTelemetry,
      mutationStats: snap.mutationStats,
      sampleCreatures: snap.sampleCreatures
    };
  }

  if (mode === 'render') {
    return {
      id,
      scenario: scenarioName,
      tick: snap.tick,
      time: snap.time,
      seed: snap.seed,
      world: worldDims,
      populations: snap.populations,
      fluid: snap.fluid,
      creatures: snap.creatures
    };
  }

  return {
    ...snap,
    id,
    scenario: scenarioName,
    world: worldDims
  };
}

// Fixed-step simulation loop with accumulator.
const LOOP_WALL_MS = 10;
let loopLastAt = Date.now();
const MAX_STEPS_PER_LOOP = 50;

setInterval(() => {
  const now = Date.now();
  const elapsed = now - loopLastAt;
  loopLastAt = now;

  if (!world || paused || crashed) {
    accumulatorMs = 0;
    stepsThisSecond = 0;
    return;
  }

  accumulatorMs += elapsed;
  const dtMs = dt * 1000;

  const t0 = Date.now();
  let steps = 0;
  try {
    while (accumulatorMs >= dtMs && steps < MAX_STEPS_PER_LOOP) {
      world.step(dt);
      accumulatorMs -= dtMs;
      steps += 1;
      stepsThisSecond += 1;
    }
  } catch (err) {
    crashed = true;
    paused = true;
    crash = {
      at: new Date().toISOString(),
      message: String(err?.message || err),
      stack: String(err?.stack || '')
    };
  }

  if (steps > 0) {
    lastStepWallMs = Date.now() - t0;
  }

  if (now - lastStepsPerSecondAt >= 1000) {
    stepsPerSecond = stepsThisSecond;
    stepsThisSecond = 0;
    lastStepsPerSecondAt = now;
  }
}, LOOP_WALL_MS);

// RPC handling
const pending = new Map();

function reply(requestId, payload) {
  parentPort.postMessage({ type: 'rpcResult', requestId, ...payload });
}

parentPort.on('message', (msg) => {
  try {
    if (!msg || typeof msg !== 'object') return;

    if (msg.type === 'rpc') {
      const { requestId, method, args } = msg;

      if (method === 'getStatus') {
        return reply(requestId, { ok: true, result: computeStatus() });
      }

      if (method === 'getSnapshot') {
        const mode = String(args?.mode || 'render');
        return reply(requestId, { ok: true, result: computeSnapshot(mode) });
      }

      if (method === 'pause') {
        paused = true;
        return reply(requestId, { ok: true, result: { paused } });
      }

      if (method === 'resume') {
        paused = false;
        return reply(requestId, { ok: true, result: { paused } });
      }

      if (method === 'setScenario') {
        const nextScenario = String(args?.name || args?.scenario || '').trim();
        const nextSeed = args?.seed;
        resetWorld(nextScenario, nextSeed);
        return reply(requestId, { ok: true, result: { scenario: scenarioName, seed, dt } });
      }

      if (method === 'applyConfigOverrides') {
        const overrides = args?.overrides;
        if (!overrides || typeof overrides !== 'object') {
          return reply(requestId, { ok: false, error: 'overrides must be an object' });
        }

        const tokens = [];
        for (const [k, v] of Object.entries(overrides)) {
          tokens.push({ key: k, value: v });
        }

        const out = applyConfigOverrides(config, tokens);
        return reply(requestId, { ok: true, result: out });
      }

      return reply(requestId, { ok: false, error: `Unknown method: ${method}` });
    }
  } catch (err) {
    if (msg?.type === 'rpc') {
      reply(msg.requestId, { ok: false, error: String(err?.message || err) });
    }
  }
});
