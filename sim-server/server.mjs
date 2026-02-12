#!/usr/bin/env node

/**
 * Authoritative simulation server (single process, multi-world).
 *
 * - Runs the real engine in Node (RealWorld + stepWorld).
 * - Fastify HTTP API for worlds/status/snapshots/controls.
 * - WebSocket stream per world for snapshots + status.
 * - Serves a simple browser client from ./public.
 */

import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import fastifyWebsocket from '@fastify/websocket';

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { getScenario } from '../node-harness/scenarios.mjs';
import { scenarioDefs } from '../js/engine/scenarioDefs.mjs';
import { RealWorld } from '../node-harness/realWorld.mjs';

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

function parseNum(raw, fallback = null) {
  if (raw === null || raw === undefined || raw === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function makeId(prefix = 'w') {
  const rand = Math.random().toString(16).slice(2, 10);
  return `${prefix}_${rand}`;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, '..');
const publicDir = resolve(__dirname, 'public');

const port = Math.floor(parseNum(arg('port', null), 8787));
const initialScenarioName = arg('scenario', 'micro_repro_sustain');
const initialSeed = (parseNum(arg('seed', null), 23) >>> 0);
const maxWorlds = Math.floor(parseNum(arg('maxWorlds', null), 8));

const DEFAULT_WORLD_ID = 'w0';

/**
 * @typedef {object} WorldEntry
 * @property {string} id
 * @property {string} scenario
 * @property {number} seed
 * @property {boolean} paused
 * @property {boolean} crashed
 * @property {object|null} crash
 * @property {number} dt
 * @property {RealWorld} instance
 * @property {number} accumulatorMs
 * @property {number} lastStepWallMs
 * @property {number} stepsThisSecond
 * @property {number} stepsPerSecond
 * @property {number} lastStepsPerSecondAt
 * @property {number} createdAt
 */

/** @type {Map<string, WorldEntry>} */
const worlds = new Map();

function createWorld({ id = null, scenario, seed } = {}) {
  const scenarioName = String(scenario || '').trim();
  if (!scenarioName || !(scenarioName in scenarioDefs)) {
    throw new Error(`Unknown scenario: ${scenarioName}`);
  }

  const resolvedId = id ? String(id).trim() : makeId('w');
  if (!resolvedId) throw new Error('Invalid world id');
  if (worlds.has(resolvedId)) throw new Error(`World already exists: ${resolvedId}`);
  if (worlds.size >= maxWorlds) throw new Error(`Too many worlds (maxWorlds=${maxWorlds})`);

  const scenarioConfig = getScenario(scenarioName);
  const dt = Number(scenarioConfig.dt) || (1 / 60);
  const seedU32 = (Number.isFinite(Number(seed)) ? Number(seed) : 23) >>> 0;

  const instance = new RealWorld(scenarioConfig, seedU32);

  const entry = {
    id: resolvedId,
    scenario: scenarioName,
    seed: seedU32,
    paused: false,
    crashed: false,
    crash: null,
    dt,
    instance,
    accumulatorMs: 0,
    lastStepWallMs: 0,
    stepsThisSecond: 0,
    stepsPerSecond: 0,
    lastStepsPerSecondAt: Date.now(),
    createdAt: Date.now()
  };

  worlds.set(resolvedId, entry);
  return entry;
}

function deleteWorld(id) {
  if (id === DEFAULT_WORLD_ID) throw new Error('Refusing to delete default world');
  const entry = worlds.get(id);
  if (!entry) throw new Error(`Unknown world: ${id}`);
  worlds.delete(id);
}

function resetWorld(entry, { scenario, seed } = {}) {
  const scenarioName = String(scenario || '').trim();
  if (!scenarioName || !(scenarioName in scenarioDefs)) {
    throw new Error(`Unknown scenario: ${scenarioName}`);
  }

  const scenarioConfig = getScenario(scenarioName);
  const dt = Number(scenarioConfig.dt) || (1 / 60);
  const seedU32 = (Number.isFinite(Number(seed)) ? Number(seed) : entry.seed) >>> 0;

  entry.scenario = scenarioName;
  entry.seed = seedU32;
  entry.dt = dt;
  entry.instance = new RealWorld(scenarioConfig, seedU32);
  entry.accumulatorMs = 0;
  entry.lastStepWallMs = 0;
  entry.stepsThisSecond = 0;
  entry.stepsPerSecond = 0;
  entry.lastStepsPerSecondAt = Date.now();
  entry.paused = false;
  entry.crashed = false;
  entry.crash = null;
}

function getWorldOrThrow(id) {
  const key = String(id || '').trim();
  const entry = worlds.get(key);
  if (!entry) throw new Error(`Unknown world: ${key}`);
  return entry;
}

function worldPopulations(entry) {
  const inst = entry.instance;
  return {
    creatures: Array.isArray(inst?.softBodyPopulation) ? inst.softBodyPopulation.length : null,
    liveCreatures: Array.isArray(inst?.softBodyPopulation) ? inst.softBodyPopulation.length : null,
    particles: Array.isArray(inst?.particles) ? inst.particles.length : null
  };
}

function computeWorldStatus(entry) {
  const inst = entry.instance;
  return {
    id: entry.id,
    ok: true,
    scenario: entry.scenario,
    seed: entry.seed,
    paused: entry.paused,
    crashed: entry.crashed,
    crash: entry.crash,
    dt: entry.dt,
    tick: Number(inst?.tick) || 0,
    time: Number(inst?.time) || 0,
    world: inst?.config?.world || null,
    populations: worldPopulations(entry),
    stepsPerSecond: entry.stepsPerSecond,
    lastStepWallMs: entry.lastStepWallMs,
    createdAt: entry.createdAt
  };
}

function computeWorldSnapshot(entry, mode) {
  const snap = entry.instance.snapshot();
  const worldDims = entry.instance?.config?.world || null;

  if (mode === 'lite') {
    return {
      id: entry.id,
      scenario: entry.scenario,
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
      id: entry.id,
      scenario: entry.scenario,
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
    id: entry.id,
    scenario: entry.scenario,
    world: worldDims
  };
}

// Bootstrap default world.
createWorld({ id: DEFAULT_WORLD_ID, scenario: initialScenarioName, seed: initialSeed });

// Simulation loop (fixed dt per world with accumulator)
const LOOP_WALL_MS = 10;
let loopLastAt = Date.now();
const MAX_STEPS_PER_LOOP_PER_WORLD = 50;

setInterval(() => {
  const now = Date.now();
  const elapsed = now - loopLastAt;
  loopLastAt = now;

  for (const entry of worlds.values()) {
    if (!entry.instance || entry.paused || entry.crashed) {
      entry.accumulatorMs = 0;
      entry.stepsThisSecond = 0;
      continue;
    }

    entry.accumulatorMs += elapsed;
    const dtMs = entry.dt * 1000;

    const t0 = Date.now();
    let steps = 0;

    try {
      while (entry.accumulatorMs >= dtMs && steps < MAX_STEPS_PER_LOOP_PER_WORLD) {
        entry.instance.step(entry.dt);
        entry.accumulatorMs -= dtMs;
        steps += 1;
        entry.stepsThisSecond += 1;
      }
    } catch (err) {
      entry.crashed = true;
      entry.paused = true;
      entry.crash = {
        at: new Date().toISOString(),
        message: String(err?.message || err),
        stack: String(err?.stack || '')
      };
    }

    if (steps > 0) {
      entry.lastStepWallMs = Date.now() - t0;
    }

    if (now - entry.lastStepsPerSecondAt >= 1000) {
      entry.stepsPerSecond = entry.stepsThisSecond;
      entry.stepsThisSecond = 0;
      entry.lastStepsPerSecondAt = now;
    }
  }
}, LOOP_WALL_MS);

const app = Fastify({
  logger: false
});

await app.register(fastifyWebsocket);
await app.register(fastifyStatic, {
  root: publicDir,
  prefix: '/',
  decorateReply: false
});

// --- API ---

// Legacy single-world aliases (operate on DEFAULT_WORLD_ID)
app.get('/api/status', async () => computeWorldStatus(getWorldOrThrow(DEFAULT_WORLD_ID)));
app.get('/api/snapshot', async (req) => {
  const mode = (req.query?.mode || 'render');
  return computeWorldSnapshot(getWorldOrThrow(DEFAULT_WORLD_ID), mode);
});
app.post('/api/control/pause', async () => {
  const w = getWorldOrThrow(DEFAULT_WORLD_ID);
  w.paused = true;
  return { ok: true, paused: w.paused };
});
app.post('/api/control/resume', async () => {
  const w = getWorldOrThrow(DEFAULT_WORLD_ID);
  w.paused = false;
  return { ok: true, paused: w.paused };
});
app.post('/api/control/setScenario', async (req, reply) => {
  const nextScenario = String(req.body?.name || '').trim();
  const nextSeed = (Number.isFinite(Number(req.body?.seed)) ? Number(req.body.seed) : initialSeed) >>> 0;
  if (!nextScenario || !(nextScenario in scenarioDefs)) {
    reply.code(400);
    return { ok: false, error: `Unknown scenario: ${nextScenario}` };
  }
  const w = getWorldOrThrow(DEFAULT_WORLD_ID);
  resetWorld(w, { scenario: nextScenario, seed: nextSeed });
  return { ok: true, scenario: w.scenario, seed: w.seed, dt: w.dt };
});

// Scenarios
app.get('/api/scenarios', async () => {
  const items = Object.keys(scenarioDefs).map((name) => ({
    name,
    description: scenarioDefs[name]?.description || ''
  }));
  return { ok: true, scenarios: items };
});

// Worlds
app.get('/api/worlds', async () => {
  const items = Array.from(worlds.values()).map((w) => computeWorldStatus(w));
  return { ok: true, worlds: items, defaultWorldId: DEFAULT_WORLD_ID, maxWorlds };
});

app.post('/api/worlds', async (req, reply) => {
  try {
    const scenario = String(req.body?.scenario || req.body?.name || initialScenarioName).trim();
    const seed = (Number.isFinite(Number(req.body?.seed)) ? Number(req.body.seed) : initialSeed) >>> 0;
    const id = (req.body?.id !== undefined && req.body?.id !== null) ? String(req.body.id) : null;

    const w = createWorld({ id, scenario, seed });
    return { ok: true, id: w.id };
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.delete('/api/worlds/:id', async (req, reply) => {
  try {
    deleteWorld(req.params.id);
    return { ok: true };
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.get('/api/worlds/:id/status', async (req, reply) => {
  try {
    return computeWorldStatus(getWorldOrThrow(req.params.id));
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.get('/api/worlds/:id/snapshot', async (req, reply) => {
  try {
    const mode = (req.query?.mode || 'render');
    return computeWorldSnapshot(getWorldOrThrow(req.params.id), mode);
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/control/pause', async (req, reply) => {
  try {
    const w = getWorldOrThrow(req.params.id);
    w.paused = true;
    return { ok: true, paused: w.paused };
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/control/resume', async (req, reply) => {
  try {
    const w = getWorldOrThrow(req.params.id);
    w.paused = false;
    return { ok: true, paused: w.paused };
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/control/setScenario', async (req, reply) => {
  try {
    const w = getWorldOrThrow(req.params.id);
    const nextScenario = String(req.body?.name || req.body?.scenario || '').trim();
    const nextSeed = (Number.isFinite(Number(req.body?.seed)) ? Number(req.body.seed) : w.seed) >>> 0;
    if (!nextScenario || !(nextScenario in scenarioDefs)) {
      reply.code(400);
      return { ok: false, error: `Unknown scenario: ${nextScenario}` };
    }
    resetWorld(w, { scenario: nextScenario, seed: nextSeed });
    return { ok: true, scenario: w.scenario, seed: w.seed, dt: w.dt };
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

// --- WebSocket stream ---
// ws://host/ws?world=w0&mode=render&hz=10
app.get('/ws', { websocket: true }, (socket, req) => {
  const url = new URL(req.url, 'http://localhost');
  const mode = url.searchParams.get('mode') || 'render';
  const worldId = url.searchParams.get('world') || DEFAULT_WORLD_ID;
  const hzRaw = Number(url.searchParams.get('hz') || 10);
  const hz = Number.isFinite(hzRaw) ? Math.max(1, Math.min(60, Math.floor(hzRaw))) : 10;
  const intervalMs = Math.max(16, Math.floor(1000 / hz));

  let entry;
  try {
    entry = getWorldOrThrow(worldId);
  } catch (err) {
    try {
      socket.send(JSON.stringify({ kind: 'error', error: String(err?.message || err) }));
    } finally {
      socket.close();
    }
    return;
  }

  const send = (payload) => {
    try {
      socket.send(JSON.stringify(payload));
    } catch {
      // ignore
    }
  };

  send({ kind: 'status', data: computeWorldStatus(entry) });
  send({ kind: 'snapshot', data: computeWorldSnapshot(entry, mode) });

  const timer = setInterval(() => {
    const latest = worlds.get(entry.id);
    if (!latest) {
      send({ kind: 'error', error: 'world_deleted' });
      socket.close();
      return;
    }
    send({ kind: 'status', data: computeWorldStatus(latest) });
    send({ kind: 'snapshot', data: computeWorldSnapshot(latest, mode) });
  }, intervalMs);

  socket.on('close', () => {
    clearInterval(timer);
  });
});

await app.listen({ port, host: '0.0.0.0' });
// eslint-disable-next-line no-console
console.log(`[sim-server] listening on http://localhost:${port}`);
// eslint-disable-next-line no-console
console.log(`[sim-server] defaultWorld=${DEFAULT_WORLD_ID} scenario=${initialScenarioName} seed=${initialSeed} maxWorlds=${maxWorlds}`);
