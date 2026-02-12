#!/usr/bin/env node

/**
 * Minimal authoritative simulation server.
 *
 * - Runs the real engine in Node (RealWorld + stepWorld).
 * - Exposes HTTP JSON endpoints for status/snapshot/control.
 * - Serves a simple browser client from ./public.
 *
 * No external deps (Node http).
 */

import http from 'node:http';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
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

function json(res, status, payload) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store'
  });
  res.end(body);
}

function text(res, status, body, contentType = 'text/plain; charset=utf-8') {
  res.writeHead(status, { 'content-type': contentType, 'cache-control': 'no-store' });
  res.end(body);
}

function readJsonBody(req) {
  return new Promise((resolvePromise, rejectPromise) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      if (!chunks.length) return resolvePromise(null);
      try {
        const raw = Buffer.concat(chunks).toString('utf8');
        resolvePromise(JSON.parse(raw));
      } catch (err) {
        rejectPromise(err);
      }
    });
    req.on('error', rejectPromise);
  });
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
let lastStepAt = Date.now();
let lastStepWallMs = 0;
let stepsThisSecond = 0;
let stepsPerSecond = 0;
let lastStepsPerSecondAt = Date.now();

let cached = {
  status: { at: 0, value: null },
  lite: { at: 0, value: null },
  render: { at: 0, value: null },
  full: { at: 0, value: null }
};

function resetWorld(nextScenarioName, nextSeed) {
  scenarioName = nextScenarioName;
  seed = (nextSeed >>> 0);

  const scenario = getScenario(scenarioName);
  dt = Number(scenario.dt) || (1 / 60);

  world = new RealWorld(scenario, seed);

  cached = {
    status: { at: 0, value: null },
    lite: { at: 0, value: null },
    render: { at: 0, value: null },
    full: { at: 0, value: null }
  };
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

function getCached(key, ttlMs, computeFn) {
  const now = Date.now();
  const entry = cached[key];
  if (entry && entry.value && (now - entry.at) < ttlMs) return entry.value;
  const value = computeFn();
  cached[key] = { at: now, value };
  return value;
}

function computeSnapshot(mode) {
  const snap = world.snapshot();
  if (mode === 'lite') {
    return {
      tick: snap.tick,
      time: snap.time,
      seed: snap.seed,
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
      populations: snap.populations,
      fluid: snap.fluid,
      creatures: snap.creatures
    };
  }
  return snap;
}

function mimeTypeFor(path) {
  const ext = extname(path).toLowerCase();
  if (ext === '.html') return 'text/html; charset=utf-8';
  if (ext === '.js' || ext === '.mjs') return 'text/javascript; charset=utf-8';
  if (ext === '.css') return 'text/css; charset=utf-8';
  if (ext === '.json') return 'application/json; charset=utf-8';
  if (ext === '.png') return 'image/png';
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg';
  if (ext === '.svg') return 'image/svg+xml; charset=utf-8';
  return 'application/octet-stream';
}

function serveStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  let rel = decodeURIComponent(url.pathname);
  if (rel === '/') rel = '/index.html';

  // basic traversal guard
  if (rel.includes('..')) {
    return text(res, 400, 'Bad path');
  }

  const filePath = resolve(publicDir, '.' + rel);
  if (!filePath.startsWith(publicDir)) {
    return text(res, 400, 'Bad path');
  }

  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    return text(res, 404, 'Not found');
  }

  const data = readFileSync(filePath);
  res.writeHead(200, {
    'content-type': mimeTypeFor(filePath),
    'cache-control': 'no-store'
  });
  res.end(data);
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);

    // API
    if (url.pathname === '/api/status' && req.method === 'GET') {
      return json(res, 200, getCached('status', 150, computeStatus));
    }

    if (url.pathname === '/api/scenarios' && req.method === 'GET') {
      const items = Object.keys(scenarioDefs).map((name) => ({
        name,
        description: scenarioDefs[name]?.description || ''
      }));
      return json(res, 200, { ok: true, scenarios: items });
    }

    if (url.pathname === '/api/snapshot' && req.method === 'GET') {
      const mode = url.searchParams.get('mode') || 'render';
      const ttl = mode === 'render' ? 120 : 300;
      const key = (mode === 'lite' || mode === 'render') ? mode : 'full';
      return json(res, 200, getCached(key, ttl, () => computeSnapshot(mode)));
    }

    if (url.pathname === '/api/control/pause' && req.method === 'POST') {
      paused = true;
      return json(res, 200, { ok: true, paused });
    }

    if (url.pathname === '/api/control/resume' && req.method === 'POST') {
      paused = false;
      return json(res, 200, { ok: true, paused });
    }

    if (url.pathname === '/api/control/setScenario' && req.method === 'POST') {
      const body = await readJsonBody(req);
      const nextScenario = String(body?.name || '').trim();
      const nextSeed = (Number.isFinite(Number(body?.seed)) ? Number(body.seed) : seed) >>> 0;
      if (!nextScenario || !(nextScenario in scenarioDefs)) {
        return json(res, 400, { ok: false, error: `Unknown scenario: ${nextScenario}` });
      }
      resetWorld(nextScenario, nextSeed);
      return json(res, 200, { ok: true, scenario: scenarioName, seed, dt });
    }

    // Static
    return serveStatic(req, res);
  } catch (err) {
    return json(res, 500, { ok: false, error: String(err?.message || err) });
  }
});

// Simulation loop: catch up using an accumulator so dt can be small (e.g. 1/120).
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
  lastStepAt = now;

  if (now - lastStepsPerSecondAt >= 1000) {
    stepsPerSecond = stepsThisSecond;
    stepsThisSecond = 0;
    lastStepsPerSecondAt = now;
  }
}, LOOP_WALL_MS);

server.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`[sim-server] listening on http://localhost:${port}`);
  console.log(`[sim-server] scenario=${scenarioName} seed=${seed} dt=${dt}`);
});
