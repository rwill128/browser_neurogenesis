#!/usr/bin/env node

/**
 * Authoritative simulation server.
 *
 * - Runs the real engine in Node (RealWorld + stepWorld).
 * - Fastify HTTP API for status/snapshots/controls.
 * - WebSocket stream for snapshots + status.
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

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, '..');
const publicDir = resolve(__dirname, 'public');

const port = Math.floor(parseNum(arg('port', null), 8787));
const initialScenarioName = arg('scenario', 'micro_repro_sustain');
const initialSeed = (parseNum(arg('seed', null), 23) >>> 0);

let scenarioName = initialScenarioName;
let seed = initialSeed;
let paused = false;
let dt = 1 / 60;

let world = null;
let lastStepWallMs = 0;
let stepsThisSecond = 0;
let stepsPerSecond = 0;
let lastStepsPerSecondAt = Date.now();

function resetWorld(nextScenarioName, nextSeed) {
  scenarioName = nextScenarioName;
  seed = (nextSeed >>> 0);

  const scenario = getScenario(scenarioName);
  dt = Number(scenario.dt) || (1 / 60);

  world = new RealWorld(scenario, seed);
}

resetWorld(scenarioName, seed);

function computeStatus() {
  const snapshot = world?.snapshot?.();
  return {
    ok: true,
    scenario: scenarioName,
    seed,
    paused,
    dt,
    tick: snapshot?.tick ?? null,
    time: snapshot?.time ?? null,
    populations: snapshot?.populations ?? null,
    stepsPerSecond,
    lastStepWallMs
  };
}

function computeSnapshot(mode) {
  const snap = world.snapshot();
  const worldDims = world?.config?.world || null;

  if (mode === 'lite') {
    return {
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
    world: worldDims
  };
}

// Simulation loop (fixed dt with accumulator)
const LOOP_WALL_MS = 10;
let accumulatorMs = 0;
let loopLastAt = Date.now();
const MAX_STEPS_PER_LOOP = 50;

setInterval(() => {
  const now = Date.now();
  const elapsed = now - loopLastAt;
  loopLastAt = now;

  if (!world || paused) {
    accumulatorMs = 0;
    stepsThisSecond = 0;
    return;
  }

  accumulatorMs += elapsed;
  const dtMs = dt * 1000;

  const t0 = Date.now();
  let steps = 0;
  while (accumulatorMs >= dtMs && steps < MAX_STEPS_PER_LOOP) {
    world.step(dt);
    accumulatorMs -= dtMs;
    steps += 1;
    stepsThisSecond += 1;
  }
  lastStepWallMs = Date.now() - t0;

  if (now - lastStepsPerSecondAt >= 1000) {
    stepsPerSecond = stepsThisSecond;
    stepsThisSecond = 0;
    lastStepsPerSecondAt = now;
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

// API
app.get('/api/status', async () => computeStatus());

app.get('/api/scenarios', async () => {
  const items = Object.keys(scenarioDefs).map((name) => ({
    name,
    description: scenarioDefs[name]?.description || ''
  }));
  return { ok: true, scenarios: items };
});

app.get('/api/snapshot', async (req) => {
  const mode = (req.query?.mode || 'render');
  return computeSnapshot(mode);
});

app.post('/api/control/pause', async () => {
  paused = true;
  return { ok: true, paused };
});

app.post('/api/control/resume', async () => {
  paused = false;
  return { ok: true, paused };
});

app.post('/api/control/setScenario', async (req, reply) => {
  const nextScenario = String(req.body?.name || '').trim();
  const nextSeed = (Number.isFinite(Number(req.body?.seed)) ? Number(req.body.seed) : seed) >>> 0;

  if (!nextScenario || !(nextScenario in scenarioDefs)) {
    reply.code(400);
    return { ok: false, error: `Unknown scenario: ${nextScenario}` };
  }

  resetWorld(nextScenario, nextSeed);
  return { ok: true, scenario: scenarioName, seed, dt };
});

// WebSocket stream
app.get('/ws', { websocket: true }, (socket, req) => {
  const url = new URL(req.url, 'http://localhost');
  const mode = url.searchParams.get('mode') || 'render';
  const hzRaw = Number(url.searchParams.get('hz') || 10);
  const hz = Number.isFinite(hzRaw) ? Math.max(1, Math.min(60, Math.floor(hzRaw))) : 10;
  const intervalMs = Math.max(16, Math.floor(1000 / hz));

  const send = (payload) => {
    try {
      socket.send(JSON.stringify(payload));
    } catch {
      // ignore
    }
  };

  send({ kind: 'status', data: computeStatus() });
  send({ kind: 'snapshot', data: computeSnapshot(mode) });

  const timer = setInterval(() => {
    send({ kind: 'status', data: computeStatus() });
    send({ kind: 'snapshot', data: computeSnapshot(mode) });
  }, intervalMs);

  socket.on('close', () => {
    clearInterval(timer);
  });
});

await app.listen({ port, host: '0.0.0.0' });
// eslint-disable-next-line no-console
console.log(`[sim-server] listening on http://localhost:${port}`);
// eslint-disable-next-line no-console
console.log(`[sim-server] scenario=${scenarioName} seed=${seed} dt=${dt}`);
