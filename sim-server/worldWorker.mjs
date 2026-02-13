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

function normalizeRunMode(raw, fallback = RUN_MODE_REALTIME) {
  const mode = String(raw || '').trim().toLowerCase();
  if (mode === RUN_MODE_MAX || mode === 'fast' || mode === 'asap' || mode === 'max_speed') {
    return RUN_MODE_MAX;
  }
  if (mode === RUN_MODE_REALTIME || mode === 'normal') {
    return RUN_MODE_REALTIME;
  }
  return fallback;
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

const RUN_MODE_REALTIME = 'realtime';
const RUN_MODE_MAX = 'max';
let runMode = RUN_MODE_REALTIME;

let crashed = false;
let crash = null;

// Render-frame buffering for UI decoupling:
// simulation advances independently, clients consume recent cached frames.
const FRAME_HISTORY_MAX = 100;
const FRAME_CAPTURE_STEP_STRIDE = 1;
const FRAME_BUFFER_IDLE_MS = 30_000;
let frameHistory = [];
let frameSeq = 0;
let lastFrameCaptureTick = -1;
let frameBufferLastAccessAt = 0;

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

  frameHistory = [];
  frameSeq = 0;
  lastFrameCaptureTick = -1;
  frameBufferLastAccessAt = 0;
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
    runMode,
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
    lastStepWallMs,
    edgeLengthTelemetryLatest: world?.worldState?.edgeLengthTelemetry?.latest || null,
    frameBuffer: {
      active: isFrameBufferActive(),
      available: frameHistory.length,
      max: FRAME_HISTORY_MAX,
      stepStride: FRAME_CAPTURE_STEP_STRIDE,
      latestSeq: frameHistory.length ? Number(frameHistory[frameHistory.length - 1]?.__frameSeq || 0) : null,
      latestTick: frameHistory.length ? Number(frameHistory[frameHistory.length - 1]?.tick || 0) : null
    }
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

function markFrameBufferAccess() {
  frameBufferLastAccessAt = Date.now();
}

function isFrameBufferActive() {
  if (!frameBufferLastAccessAt) return false;
  return (Date.now() - frameBufferLastAccessAt) <= FRAME_BUFFER_IDLE_MS;
}

function captureRenderFrameIfNeeded({ force = false } = {}) {
  if (!world) return null;

  const tick = Number(world?.tick) || 0;
  if (!force) {
    if (!isFrameBufferActive()) return null;
    if (tick <= 0) return null;
    if (tick === lastFrameCaptureTick) return null;
    if ((tick % FRAME_CAPTURE_STEP_STRIDE) !== 0) return null;
  }

  const frame = computeSnapshot('render');
  frameSeq += 1;
  frame.__frameSeq = frameSeq;
  frame.__capturedAtIso = new Date().toISOString();

  frameHistory.push(frame);
  if (frameHistory.length > FRAME_HISTORY_MAX) {
    frameHistory.splice(0, frameHistory.length - FRAME_HISTORY_MAX);
  }

  lastFrameCaptureTick = tick;
  return frame;
}

function getLatestRenderFrame() {
  if (frameHistory.length === 0) {
    const seeded = captureRenderFrameIfNeeded({ force: true });
    return seeded || null;
  }
  return frameHistory[frameHistory.length - 1] || null;
}

function getRenderFrameByOffset(frameOffset = 0) {
  const offset = Math.max(0, Math.floor(Number(frameOffset) || 0));
  const idx = frameHistory.length - 1 - offset;
  if (idx < 0 || idx >= frameHistory.length) return null;
  return frameHistory[idx] || null;
}

function getRenderFrameBySeq(seq) {
  const target = Math.floor(Number(seq));
  if (!Number.isFinite(target)) return null;
  for (let i = frameHistory.length - 1; i >= 0; i--) {
    const frame = frameHistory[i];
    if (Number(frame?.__frameSeq) === target) return frame;
  }
  return null;
}

function getFrameTimeline() {
  const frames = frameHistory.map((f) => ({
    seq: Number(f?.__frameSeq) || 0,
    tick: Number(f?.tick) || 0,
    time: Number(f?.time) || 0,
    capturedAtIso: f?.__capturedAtIso || null
  }));
  const latest = frames[frames.length - 1] || null;
  const oldest = frames[0] || null;

  return {
    worldId: id,
    available: frames.length,
    max: FRAME_HISTORY_MAX,
    stepStride: FRAME_CAPTURE_STEP_STRIDE,
    oldestSeq: oldest?.seq ?? null,
    latestSeq: latest?.seq ?? null,
    latestTick: latest?.tick ?? null,
    frames
  };
}

// Fixed-step simulation loop: realtime pacing (default) or max-throughput mode.
const LOOP_WALL_MS = 10;
let loopLastAt = Date.now();
const MAX_STEPS_PER_LOOP_REALTIME = 50;
const MAX_STEPS_PER_LOOP_MAX = 5000;
const MAX_MODE_BATCH_WALL_MS = 25;

function scheduleNextLoop() {
  if (runMode === RUN_MODE_MAX) {
    setImmediate(runLoop);
  } else {
    setTimeout(runLoop, LOOP_WALL_MS);
  }
}

function runLoop() {
  const now = Date.now();
  const elapsed = Math.max(0, now - loopLastAt);
  loopLastAt = now;

  if (!world || paused || crashed) {
    accumulatorMs = 0;
    if (now - lastStepsPerSecondAt >= 1000) {
      stepsPerSecond = stepsThisSecond;
      stepsThisSecond = 0;
      lastStepsPerSecondAt = now;
    }
    scheduleNextLoop();
    return;
  }

  const t0 = Date.now();
  let steps = 0;

  try {
    if (runMode === RUN_MODE_MAX) {
      const batchStart = Date.now();
      while (steps < MAX_STEPS_PER_LOOP_MAX && (Date.now() - batchStart) < MAX_MODE_BATCH_WALL_MS) {
        world.step(dt);
        steps += 1;
      }
    } else {
      accumulatorMs += elapsed;
      const dtMs = dt * 1000;
      while (accumulatorMs >= dtMs && steps < MAX_STEPS_PER_LOOP_REALTIME) {
        world.step(dt);
        accumulatorMs -= dtMs;
        steps += 1;
      }
    }

    stepsThisSecond += steps;
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
    captureRenderFrameIfNeeded({ force: false });
  }

  if (now - lastStepsPerSecondAt >= 1000) {
    stepsPerSecond = stepsThisSecond;
    stepsThisSecond = 0;
    lastStepsPerSecondAt = now;
  }

  scheduleNextLoop();
}

scheduleNextLoop();

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

        if (mode === 'render') {
          markFrameBufferAccess();

          const requestedSeq = args?.frameSeq;
          if (requestedSeq !== undefined && requestedSeq !== null && requestedSeq !== '') {
            const bySeq = getRenderFrameBySeq(requestedSeq);
            if (bySeq) return reply(requestId, { ok: true, result: bySeq });
          }

          const requestedOffset = args?.frameOffset;
          if (requestedOffset !== undefined && requestedOffset !== null && requestedOffset !== '') {
            const byOffset = getRenderFrameByOffset(requestedOffset);
            if (byOffset) return reply(requestId, { ok: true, result: byOffset });
          }

          const latest = getLatestRenderFrame() || computeSnapshot('render');
          return reply(requestId, { ok: true, result: latest });
        }

        return reply(requestId, { ok: true, result: computeSnapshot(mode) });
      }

      if (method === 'getFrameTimeline') {
        markFrameBufferAccess();
        // Ensure timeline has at least one frame after first UI access.
        if (frameHistory.length === 0) {
          captureRenderFrameIfNeeded({ force: true });
        }
        return reply(requestId, { ok: true, result: getFrameTimeline() });
      }

      if (method === 'pause') {
        paused = true;
        return reply(requestId, { ok: true, result: { paused } });
      }

      if (method === 'resume') {
        paused = false;
        return reply(requestId, { ok: true, result: { paused } });
      }

      if (method === 'setRunMode') {
        runMode = normalizeRunMode(args?.mode, runMode);
        accumulatorMs = 0;
        loopLastAt = Date.now();
        return reply(requestId, { ok: true, result: { runMode } });
      }

      if (method === 'getRunMode') {
        return reply(requestId, { ok: true, result: { runMode } });
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
          'CREATURE_POPULATION_FLOOR',
          'CREATURE_POPULATION_CEILING',
          'PARTICLE_POPULATION_FLOOR',
          'PARTICLE_POPULATION_CEILING',
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
          'FLUID_SOLVER_ITERATIONS_VELOCITY',
          'FLUID_SOLVER_ITERATIONS_PRESSURE',
          'FLUID_SOLVER_ITERATIONS_DENSITY',
          'FLUID_FADE_RATE',
          'MIN_VISCOSITY_MULTIPLIER',
          'MAX_VISCOSITY_MULTIPLIER',
          'VISCOSITY_LANDSCAPE_NOISE_SCALE',
          'VISCOSITY_LANDSCAPE_OCTAVES',
          'VISCOSITY_LANDSCAPE_LACUNARITY',
          'VISCOSITY_LANDSCAPE_GAIN',
          'VISCOSITY_LANDSCAPE_CONTRAST',
          'VISCOSITY_LANDSCAPE_BANDS',
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
          'FAILED_REPRODUCTION_COOLDOWN_TICKS',
          'EDGE_LENGTH_HARD_CAP_ENABLED',
          'EDGE_LENGTH_HARD_CAP_FACTOR',
          'EDGE_LENGTH_TELEMETRY_ENABLED',
          'EDGE_LENGTH_TELEMETRY_SAMPLE_EVERY_N_STEPS',
          'EDGE_LENGTH_TELEMETRY_MODE_BIN_SIZE',
          'EDGE_LENGTH_TELEMETRY_HUGE_OUTLIER_IQR_MULTIPLIER',
          'PHYSICS_MOTION_GUARD_ENABLED',
          'PHYSICS_NONFINITE_FORCE_ZERO',
          'PHYSICS_MAX_ACCELERATION_MAGNITUDE',
          'PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP'
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
