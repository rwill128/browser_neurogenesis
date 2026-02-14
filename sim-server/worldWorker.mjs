#!/usr/bin/env node

/**
 * Worker thread: owns one authoritative world (isolated module state).
 *
 * Why a worker?
 * - The simulation code currently uses a module-level config singleton (js/config.js).
 * - Workers isolate that singleton per world, making true multi-world safe.
 */

import { parentPort, workerData } from 'node:worker_threads';
import { serialize as v8Serialize } from 'node:v8';

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
const FRAME_CAPTURE_STEP_STRIDE = 50;
const FRAME_BUFFER_IDLE_MS = 30_000;
const ARCHIVE_RICH_STATS_EVERY_N_TICKS = 1000;
let frameHistory = [];
let frameArchive = [];
let frameSeq = 0;
let lastFrameCaptureTick = -1;
let frameBufferLastAccessAt = 0;
let lastRealtimeStepAt = 0;

const perf = {
  stepSamples: 0,
  stepMsTotal: 0,
  stepMsMax: 0,
  archiveSerializeSamples: 0,
  archiveSerializeMsTotal: 0,
  archiveSerializeMsMax: 0,
  archivePostSamples: 0,
  archivePostMsTotal: 0,
  archivePostMsMax: 0,
  snapshotSamples: 0,
  snapshotMsTotal: 0,
  snapshotMsMax: 0
};

function perfAdd(kind, ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value < 0) return;
  if (kind === 'step') {
    perf.stepSamples += 1;
    perf.stepMsTotal += value;
    perf.stepMsMax = Math.max(perf.stepMsMax, value);
    return;
  }
  if (kind === 'archiveSerialize') {
    perf.archiveSerializeSamples += 1;
    perf.archiveSerializeMsTotal += value;
    perf.archiveSerializeMsMax = Math.max(perf.archiveSerializeMsMax, value);
    return;
  }
  if (kind === 'archivePost') {
    perf.archivePostSamples += 1;
    perf.archivePostMsTotal += value;
    perf.archivePostMsMax = Math.max(perf.archivePostMsMax, value);
    return;
  }
  if (kind === 'snapshot') {
    perf.snapshotSamples += 1;
    perf.snapshotMsTotal += value;
    perf.snapshotMsMax = Math.max(perf.snapshotMsMax, value);
  }
}

function perfAvg(total, samples) {
  return samples > 0 ? Number((total / samples).toFixed(3)) : 0;
}

function resetWorld(nextScenarioName, nextSeed) {
  const s = String(nextScenarioName || '').trim();
  if (!s || !(s in scenarioDefs)) {
    throw new Error(`Unknown scenario: ${s}`);
  }

  scenarioName = s;
  seed = asU32(nextSeed, seed);

  const scenario = getScenario(scenarioName);

  // Enforce stable defaults across all scenarios for the current archive-first phase.
  scenario.particles = 0;
  scenario.particleFloor = 0;
  scenario.particleCeiling = 0;
  scenario.particlesPerSecond = 0;
  scenario.configOverrides = {
    ...(scenario.configOverrides || {}),
    PARTICLE_POPULATION_FLOOR: 0,
    PARTICLE_POPULATION_CEILING: 0,
    PARTICLES_PER_SECOND: 0,
    INITIAL_TRIANGULATED_PRIMITIVES_ENABLED: true,
    INITIAL_TRI_TEMPLATE_WEIGHT_TRIANGLE: 1,
    INITIAL_TRI_TEMPLATE_WEIGHT_DIAMOND: 0,
    INITIAL_TRI_TEMPLATE_WEIGHT_HEXAGON: 0,
    INITIAL_TRI_MESH_EDGE_RIGID_CHANCE: 1,
    FORCE_ALL_SPRINGS_RIGID: true,
    RIGID_CONSTRAINT_PROJECTION_ENABLED: true,
    RIGID_CONSTRAINT_PROJECTION_ITERATIONS: 8,
    RIGID_CONSTRAINT_MAX_RELATIVE_ERROR: 0.001,
    EDGE_LENGTH_HARD_CAP_ENABLED: true,
    EDGE_LENGTH_HARD_CAP_FACTOR: 6,
    PHYSICS_MOTION_GUARD_ENABLED: true,
    PHYSICS_NONFINITE_FORCE_ZERO: true
  };

  dt = Number(scenario.dt) || (1 / 60);

  world = new RealWorld(scenario, seed);
  accumulatorMs = 0;

  lastStepWallMs = 0;
  stepsThisSecond = 0;
  stepsPerSecond = 0;
  lastStepsPerSecondAt = Date.now();
  lastRealtimeStepAt = 0;

  crashed = false;
  crash = null;

  frameHistory = [];
  frameArchive = [];
  frameSeq = 0;
  lastFrameCaptureTick = -1;
  frameBufferLastAccessAt = 0;

  perf.stepSamples = 0;
  perf.stepMsTotal = 0;
  perf.stepMsMax = 0;
  perf.archiveSerializeSamples = 0;
  perf.archiveSerializeMsTotal = 0;
  perf.archiveSerializeMsMax = 0;
  perf.archivePostSamples = 0;
  perf.archivePostMsTotal = 0;
  perf.archivePostMsMax = 0;
  perf.snapshotSamples = 0;
  perf.snapshotMsTotal = 0;
  perf.snapshotMsMax = 0;

  // Seed tick-0 frame so history is addressable from frame 0.
  archiveCurrentStepFrame();
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
      latestTick: frameHistory.length ? Number(frameHistory[frameHistory.length - 1]?.tick || 0) : null,
      archiveFrames: frameArchive.length,
      archiveOldestTick: frameArchive.length ? Number(frameArchive[0]?.tick || 0) : null,
      archiveLatestTick: frameArchive.length ? Number(frameArchive[frameArchive.length - 1]?.tick || 0) : null
    },
    workerPerf: {
      stepMsAvg: perfAvg(perf.stepMsTotal, perf.stepSamples),
      stepMsMax: Number(perf.stepMsMax.toFixed(3)),
      archiveSerializeMsAvg: perfAvg(perf.archiveSerializeMsTotal, perf.archiveSerializeSamples),
      archiveSerializeMsMax: Number(perf.archiveSerializeMsMax.toFixed(3)),
      archivePostMsAvg: perfAvg(perf.archivePostMsTotal, perf.archivePostSamples),
      archivePostMsMax: Number(perf.archivePostMsMax.toFixed(3)),
      snapshotMsAvg: perfAvg(perf.snapshotMsTotal, perf.snapshotSamples),
      snapshotMsMax: Number(perf.snapshotMsMax.toFixed(3)),
      latestStepBreakdown: world?.worldState?.lastStepTiming || null,
      samples: {
        step: perf.stepSamples,
        archiveSerialize: perf.archiveSerializeSamples,
        archivePost: perf.archivePostSamples,
        snapshot: perf.snapshotSamples
      }
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

function computeSnapshot(mode = 'render', { includeRichStats = false } = {}) {
  const snap = world.snapshot();
  const worldDims = world?.config?.world || null;

  // Archive/storage hygiene: strip bulky/derived telemetry from most per-tick frames.
  if (snap?.instabilityTelemetry && typeof snap.instabilityTelemetry === 'object') {
    delete snap.instabilityTelemetry.recentDeaths;
  }
  if (!includeRichStats && Array.isArray(snap?.creatures)) {
    for (const c of snap.creatures) {
      if (!c || typeof c !== 'object') continue;
      delete c.fullStats;
      delete c.actuationTelemetry;
    }
  }

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

function captureRenderFrameIfNeeded({ force = false, includeArchive = false } = {}) {
  if (!world) return null;

  const tick = Number(world?.tick) || 0;
  if (!force) {
    if (!isFrameBufferActive()) return null;
    if (tick <= 0) return null;
    if (tick === lastFrameCaptureTick) return null;
    if ((tick % FRAME_CAPTURE_STEP_STRIDE) !== 0) return null;
  } else {
    // Forced capture should still avoid pushing duplicate frames for the same tick.
    if (frameHistory.length > 0 && tick === lastFrameCaptureTick) {
      return frameHistory[frameHistory.length - 1] || null;
    }
  }

  const includeRichStats = includeArchive && tick > 0 && (tick % ARCHIVE_RICH_STATS_EVERY_N_TICKS === 0);
  const frame = computeSnapshot('render', { includeRichStats });
  frameSeq += 1;
  frame.__frameSeq = frameSeq;
  frame.__capturedAtIso = new Date().toISOString();
  frame.__richStats = includeRichStats;

  frameHistory.push(frame);
  if (frameHistory.length > FRAME_HISTORY_MAX) {
    frameHistory.splice(0, frameHistory.length - FRAME_HISTORY_MAX);
  }

  if (includeArchive) {
    frameArchive.push(frame);
  }

  lastFrameCaptureTick = tick;
  return frame;
}

function buildLeanArchiveFrame(frame) {
  if (!frame || typeof frame !== 'object') return frame;

  const creatures = Array.isArray(frame.creatures)
    ? frame.creatures.map((c) => {
        if (!c || typeof c !== 'object') return c;
        const out = {
          id: c.id,
          energy: c.energy,
          center: c.center,
          nodeTypeCounts: c.nodeTypeCounts,
          vertices: c.vertices,
          springs: c.springs
        };
        if (frame.__richStats === true) {
          out.fullStats = c.fullStats;
          out.actuationTelemetry = c.actuationTelemetry;
        }
        return out;
      })
    : [];

  return {
    id: frame.id,
    scenario: frame.scenario,
    tick: frame.tick,
    time: frame.time,
    seed: frame.seed,
    world: frame.world,
    populations: frame.populations,
    worldStats: frame.worldStats,
    instabilityTelemetry: frame.instabilityTelemetry,
    mutationStats: frame.mutationStats,
    fluid: frame.fluid,
    creatures,
    __frameSeq: frame.__frameSeq,
    __capturedAtIso: frame.__capturedAtIso,
    __richStats: frame.__richStats === true
  };
}

function archiveCurrentStepFrame() {
  const tick = Number(world?.tick) || 0;
  if (tick > 0 && (tick % FRAME_CAPTURE_STEP_STRIDE) !== 0) return null;

  const frame = captureRenderFrameIfNeeded({ force: true, includeArchive: true });
  if (frame) {
    const lean = buildLeanArchiveFrame(frame);
    const tSer0 = Date.now();
    const wire = v8Serialize(lean);
    perfAdd('archiveSerialize', Date.now() - tSer0);
    const ab = wire.buffer.slice(wire.byteOffset, wire.byteOffset + wire.byteLength);
    const tPost0 = Date.now();
    parentPort.postMessage({
      type: 'frameArchive',
      worldId: id,
      tick: Number(frame.tick) || 0,
      frameBuf: ab
    }, [ab]);
    perfAdd('archivePost', Date.now() - tPost0);
  }
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
    archiveFrames: frameArchive.length,
    archiveOldestTick: frameArchive.length ? Number(frameArchive[0]?.tick || 0) : null,
    archiveLatestTick: frameArchive.length ? Number(frameArchive[frameArchive.length - 1]?.tick || 0) : null,
    frames
  };
}

function getArchivedFrameByTick(tick) {
  const target = Math.floor(Number(tick));
  if (!Number.isFinite(target)) return null;
  // Start from tail because most queries are near latest.
  for (let i = frameArchive.length - 1; i >= 0; i--) {
    const f = frameArchive[i];
    if (Number(f?.tick) === target) return f;
    if (Number(f?.tick) < target) break;
  }
  return null;
}

function getArchivedFramesRange(startTick, endTick, stride = 1) {
  const start = Math.floor(Number(startTick));
  const end = Math.floor(Number(endTick));
  const step = Math.max(1, Math.floor(Number(stride) || 1));
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return [];

  const out = [];
  for (const f of frameArchive) {
    const t = Number(f?.tick);
    if (t < start) continue;
    if (t > end) break;
    if (((t - start) % step) === 0) out.push(f);
  }
  return out;
}

// Fixed-step simulation loop: realtime pacing (default) or max-throughput mode.
const LOOP_WALL_MS = 10;
let loopLastAt = Date.now();
const MAX_STEPS_PER_LOOP_REALTIME = 200;
const MAX_STEPS_PER_LOOP_MAX = 5000;
const MAX_MODE_BATCH_WALL_MS = 25;
const REALTIME_BATCH_WALL_MS = 8; // reserve event-loop time for status/snapshot RPC responsiveness
const REALTIME_MAX_STEPS_PER_SEC = Number.POSITIVE_INFINITY;
const REALTIME_MIN_STEP_INTERVAL_MS = 0;

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
        const tStep = Date.now();
        world.step(dt);
        perfAdd('step', Date.now() - tStep);
        archiveCurrentStepFrame();
        steps += 1;
      }
    } else {
      accumulatorMs += elapsed;
      const dtMs = dt * 1000;
      const realtimeBatchStart = Date.now();
      while (accumulatorMs >= dtMs && steps < MAX_STEPS_PER_LOOP_REALTIME) {
        if ((Date.now() - realtimeBatchStart) >= REALTIME_BATCH_WALL_MS) {
          break;
        }
        const nowStep = Date.now();
        if (lastRealtimeStepAt && (nowStep - lastRealtimeStepAt) < REALTIME_MIN_STEP_INTERVAL_MS) {
          break;
        }
        const tStep = Date.now();
        world.step(dt);
        perfAdd('step', Date.now() - tStep);
        archiveCurrentStepFrame();
        accumulatorMs -= dtMs;
        steps += 1;
        lastRealtimeStepAt = nowStep;
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

      if (method === 'getStatusAndSnapshot') {
        const tSnap0 = Date.now();
        const mode = String(args?.mode || 'render');
        const status = computeStatus();
        let snapshot;

        if (mode === 'render') {
          markFrameBufferAccess();

          const requestedSeq = args?.frameSeq;
          if (requestedSeq !== undefined && requestedSeq !== null && requestedSeq !== '') {
            snapshot = getRenderFrameBySeq(requestedSeq);
          }

          if (!snapshot) {
            const requestedOffset = args?.frameOffset;
            if (requestedOffset !== undefined && requestedOffset !== null && requestedOffset !== '') {
              snapshot = getRenderFrameByOffset(requestedOffset);
            }
          }

          if (!snapshot) {
            // Archive-first policy: prefer already-buffered render frames; do not force fresh capture.
            snapshot = getLatestRenderFrame() || captureRenderFrameIfNeeded({ force: true }) || computeSnapshot('render');
          }
        } else {
          snapshot = computeSnapshot(mode);
        }

        perfAdd('snapshot', Date.now() - tSnap0);
        return reply(requestId, { ok: true, result: { status, snapshot } });
      }

      if (method === 'getSnapshot') {
        const tSnap0 = Date.now();
        const mode = String(args?.mode || 'render');

        if (mode === 'render') {
          markFrameBufferAccess();

          const requestedSeq = args?.frameSeq;
          if (requestedSeq !== undefined && requestedSeq !== null && requestedSeq !== '') {
            const bySeq = getRenderFrameBySeq(requestedSeq);
            if (bySeq) {
              perfAdd('snapshot', Date.now() - tSnap0);
              return reply(requestId, { ok: true, result: bySeq });
            }
          }

          const requestedOffset = args?.frameOffset;
          if (requestedOffset !== undefined && requestedOffset !== null && requestedOffset !== '') {
            const byOffset = getRenderFrameByOffset(requestedOffset);
            if (byOffset) {
              perfAdd('snapshot', Date.now() - tSnap0);
              return reply(requestId, { ok: true, result: byOffset });
            }
          }

          // Archive-first policy: return latest buffered frame by default; only seed one if empty.
          const latest = getLatestRenderFrame() || captureRenderFrameIfNeeded({ force: true }) || computeSnapshot('render');
          perfAdd('snapshot', Date.now() - tSnap0);
          return reply(requestId, { ok: true, result: latest });
        }

        const result = computeSnapshot(mode);
        perfAdd('snapshot', Date.now() - tSnap0);
        return reply(requestId, { ok: true, result });
      }

      if (method === 'getFrameTimeline') {
        markFrameBufferAccess();
        // Ensure timeline has at least one frame after first UI access.
        if (frameHistory.length === 0) {
          captureRenderFrameIfNeeded({ force: true });
        }
        return reply(requestId, { ok: true, result: getFrameTimeline() });
      }

      if (method === 'getArchivedFrameByTick') {
        const tick = args?.tick;
        const frame = getArchivedFrameByTick(tick);
        if (!frame) return reply(requestId, { ok: false, error: `frame not found for tick ${tick}` });
        return reply(requestId, { ok: true, result: frame });
      }

      if (method === 'getArchivedFramesRange') {
        const startTick = args?.startTick;
        const endTick = args?.endTick;
        const stride = args?.stride;
        const frames = getArchivedFramesRange(startTick, endTick, stride);
        return reply(requestId, {
          ok: true,
          result: {
            worldId: id,
            startTick: Math.floor(Number(startTick)),
            endTick: Math.floor(Number(endTick)),
            stride: Math.max(1, Math.floor(Number(stride) || 1)),
            count: frames.length,
            frames
          }
        });
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
          'FLUID_STEP_EVERY_N_TICKS',
          'FLUID_MOMENTUM_ONLY_STEP_EVERY_N_TICKS',
          'FLUID_MOMENTUM_ACTIVITY_SPEED_THRESHOLD',
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
          'LANDSCAPE_VELOCITY_EMITTERS_ENABLED',
          'LANDSCAPE_VELOCITY_EMITTER_COUNT',
          'LANDSCAPE_VELOCITY_EMITTER_STRENGTH_MIN',
          'LANDSCAPE_VELOCITY_EMITTER_STRENGTH_MAX',
          'LANDSCAPE_VELOCITY_EMITTER_RADIUS_CELLS',
          'LANDSCAPE_VELOCITY_EMITTER_LOCAL_SPEED_CAP',
          'LANDSCAPE_VELOCITY_EMITTER_BUDGET_MAX',
          'LANDSCAPE_VELOCITY_EMITTER_BUDGET_REFILL_PER_SEC',
          'REPRO_RESOURCE_MIN_NUTRIENT',
          'REPRO_RESOURCE_MIN_LIGHT',
          'REPRO_FERTILITY_GLOBAL_MIN_SCALE',
          'REPRO_FERTILITY_LOCAL_MIN_SCALE',
          'REPRO_MIN_FERTILITY_SCALE',
          'FAILED_REPRODUCTION_COOLDOWN_TICKS',
          'EDGE_LENGTH_HARD_CAP_ENABLED',
          'EDGE_LENGTH_HARD_CAP_FACTOR',
          'SPRING_OVERSTRETCH_KILL_ENABLED',
          'FORCE_ALL_SPRINGS_RIGID',
          'RIGID_CONSTRAINT_PROJECTION_ENABLED',
          'RIGID_CONSTRAINT_PROJECTION_ITERATIONS',
          'RIGID_CONSTRAINT_MAX_RELATIVE_ERROR',
          'TRIANGLE_EXTRUSION_MUTATION_CHANCE_MULTIPLIER',
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
