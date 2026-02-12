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

function collectAllParticles() {
  if (!Array.isArray(world?.particles)) return [];
  const out = [];
  for (const p of world.particles) {
    if (!p?.pos) continue;
    out.push({
      x: Number(p.pos.x) || 0,
      y: Number(p.pos.y) || 0,
      life: Number(p.life) || 0,
      size: Number(p.size) || 1,
      isEaten: Boolean(p.isEaten)
    });
  }
  return out;
}

function collectFluidDenseRGBA() {
  const fluid = world?.fluidField;
  const dims = world?.config?.world;
  const N = Math.round(Number(fluid?.size) || 0);
  if (!fluid || !dims || !Number.isFinite(N) || N <= 0) {
    return null;
  }

  const total = N * N;
  const rgba = new Uint8ClampedArray(total * 4);

  for (let i = 0; i < total; i++) {
    const r = Math.min(255, Math.max(0, Math.floor(fluid.densityR?.[i] || 0)));
    const g = Math.min(255, Math.max(0, Math.floor(fluid.densityG?.[i] || 0)));
    const b = Math.min(255, Math.max(0, Math.floor(fluid.densityB?.[i] || 0)));

    // If dye is low, still show speed as faint grayscale.
    const vx = Number(fluid.Vx?.[i] || 0);
    const vy = Number(fluid.Vy?.[i] || 0);
    const speed = Math.sqrt(vx * vx + vy * vy);

    const dye = r + g + b;
    const alpha = Math.min(200, Math.max(0, Math.floor((dye > 3 ? 90 : 0) + Math.min(110, speed * 22))));

    const o = i * 4;
    rgba[o + 0] = r || (dye <= 3 ? 90 : 0);
    rgba[o + 1] = g || (dye <= 3 ? 90 : 0);
    rgba[o + 2] = b || (dye <= 3 ? 120 : 0);
    rgba[o + 3] = alpha;
  }

  const worldCell = {
    width: Number((dims.width / N).toFixed(4)),
    height: Number((dims.height / N).toFixed(4))
  };

  return {
    gridSize: N,
    worldCell,
    // base64-encoded RGBA for ImageData
    rgbaBase64: Buffer.from(rgba.buffer).toString('base64')
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
      worldStats: snap.worldStats,
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
      worldStats: snap.worldStats,
      instabilityTelemetry: snap.instabilityTelemetry,
      mutationStats: snap.mutationStats,
      fluid: snap.fluid,
      creatures: snap.creatures
    };
  }

  if (mode === 'renderFull') {
    return {
      id,
      scenario: scenarioName,
      tick: snap.tick,
      time: snap.time,
      seed: snap.seed,
      world: worldDims,
      populations: snap.populations,
      worldStats: snap.worldStats,
      instabilityTelemetry: snap.instabilityTelemetry,
      mutationStats: snap.mutationStats,
      fluid: snap.fluid,
      fluidDense: collectFluidDenseRGBA(),
      particles: collectAllParticles(),
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

      if (method === 'saveCheckpoint') {
        if (!world || typeof world.saveStateSnapshot !== 'function') {
          return reply(requestId, { ok: false, error: 'world does not support saveStateSnapshot' });
        }
        const meta = args?.meta && typeof args.meta === 'object' ? args.meta : {};
        const snapshot = world.saveStateSnapshot({
          ...meta,
          source: 'sim-server'
        });
        return reply(requestId, { ok: true, result: snapshot });
      }

      if (method === 'loadCheckpoint') {
        if (!world || typeof world.loadStateSnapshot !== 'function') {
          return reply(requestId, { ok: false, error: 'world does not support loadStateSnapshot' });
        }
        const snapshot = args?.snapshot;
        if (!snapshot || typeof snapshot !== 'object') {
          return reply(requestId, { ok: false, error: 'snapshot must be an object' });
        }
        const loadInfo = world.loadStateSnapshot(snapshot);
        return reply(requestId, { ok: true, result: { loaded: true, loadInfo } });
      }

      if (method === 'getConfig') {
        const keys = Array.isArray(args?.keys) ? args.keys : null;
        const allow = new Set(keys || [
          'PHOTOSYNTHESIS_EFFICIENCY',
          'globalNutrientMultiplier',
          'globalLightMultiplier',
          'ENERGY_PER_PARTICLE',
          'BASE_NODE_EXISTENCE_COST',
          'EMITTER_NODE_ENERGY_COST',
          'EATER_NODE_ENERGY_COST',
          'PREDATOR_NODE_ENERGY_COST',
          'NEURON_NODE_ENERGY_COST',
          'SWIMMER_NODE_ENERGY_COST',
          'PHOTOSYNTHETIC_NODE_ENERGY_COST',
          'GRABBING_NODE_ENERGY_COST',
          'EYE_NODE_ENERGY_COST',
          'JET_NODE_ENERGY_COST',
          'ATTRACTOR_NODE_ENERGY_COST',
          'REPULSOR_NODE_ENERGY_COST',
          'FLUID_CURRENT_STRENGTH_ON_BODY',
          'SOFT_BODY_PUSH_STRENGTH',
          'BODY_REPULSION_STRENGTH',
          'BODY_REPULSION_RADIUS_FACTOR',
          'MAX_FLUID_VELOCITY_COMPONENT',
          'FLUID_FADE_RATE',
          'LANDSCAPE_DYE_EMITTERS_ENABLED',
          'LANDSCAPE_DYE_EMITTER_COUNT',
          'LANDSCAPE_DYE_EMITTER_STRENGTH_MIN',
          'LANDSCAPE_DYE_EMITTER_STRENGTH_MAX',
          'LANDSCAPE_DYE_EMITTER_RADIUS_CELLS',
          'LANDSCAPE_DYE_EMITTER_PULSE_HZ_MIN',
          'LANDSCAPE_DYE_EMITTER_PULSE_HZ_MAX',
          'REPRO_RESOURCE_MIN_NUTRIENT',
          'REPRO_RESOURCE_MIN_LIGHT',
          'REPRO_FERTILITY_GLOBAL_MIN_SCALE',
          'REPRO_FERTILITY_LOCAL_MIN_SCALE',
          'REPRO_MIN_FERTILITY_SCALE',
          'FAILED_REPRODUCTION_COOLDOWN_TICKS'
        ]);

        const out = {};
        for (const key of allow) {
          if (Object.prototype.hasOwnProperty.call(config, key)) {
            const v = config[key];
            if (v === null || v === undefined) out[key] = v;
            else if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') out[key] = v;
            else out[key] = v; // best-effort; UI will ignore non-primitives
          }
        }

        return reply(requestId, { ok: true, result: { id, scenario: scenarioName, seed, values: out } });
      }

      return reply(requestId, { ok: false, error: `Unknown method: ${method}` });
    }
  } catch (err) {
    if (msg?.type === 'rpc') {
      reply(msg.requestId, { ok: false, error: String(err?.message || err) });
    }
  }
});
