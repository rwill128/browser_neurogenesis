#!/usr/bin/env node

/**
 * Authoritative simulation server (single process, multi-world via workers).
 *
 * Rationale:
 * - Simulation code uses a module-level config singleton (`js/config.js`).
 * - Running multiple worlds in one thread would stomp shared config.
 * - Worker threads isolate module state per world, while sharing one HTTP server.
 */

import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import fastifyWebsocket from '@fastify/websocket';

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { mkdirSync } from 'node:fs';
import { gzipSync, gunzipSync } from 'node:zlib';
import { DatabaseSync } from 'node:sqlite';

import { scenarioDefs } from '../js/engine/scenarioDefs.mjs';

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
const dataDir = resolve(__dirname, 'data');
const dbPath = resolve(dataDir, 'sim.sqlite');
const workerUrl = new URL('./worldWorker.mjs', import.meta.url);

mkdirSync(dataDir, { recursive: true });

const db = new DatabaseSync(dbPath);
db.exec(`
  PRAGMA journal_mode = WAL;
  CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worldId TEXT NOT NULL,
    createdAt TEXT NOT NULL,
    label TEXT,
    scenario TEXT,
    seed INTEGER,
    tick INTEGER,
    time REAL,
    bytes INTEGER,
    payloadGzip BLOB NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_checkpoints_world_created ON checkpoints(worldId, createdAt DESC);
`);

const stmtInsertCheckpoint = db.prepare(`
  INSERT INTO checkpoints (worldId, createdAt, label, scenario, seed, tick, time, bytes, payloadGzip)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const stmtListCheckpoints = db.prepare(`
  SELECT id, worldId, createdAt, label, scenario, seed, tick, time, bytes
  FROM checkpoints
  WHERE worldId = ?
  ORDER BY createdAt DESC
  LIMIT ?
`);

const stmtGetCheckpoint = db.prepare(`
  SELECT id, worldId, createdAt, label, scenario, seed, tick, time, bytes, payloadGzip
  FROM checkpoints
  WHERE id = ?
`);

const stmtPruneCheckpoints = db.prepare(`
  DELETE FROM checkpoints
  WHERE id IN (
    SELECT id FROM checkpoints
    WHERE worldId = ?
    ORDER BY createdAt DESC
    LIMIT -1 OFFSET ?
  )
`);

const port = Math.floor(parseNum(arg('port', null), 8787));
const initialScenarioName = arg('scenario', 'micro_repro_sustain');
const initialSeed = (parseNum(arg('seed', null), 23) >>> 0);
const maxWorlds = Math.floor(parseNum(arg('maxWorlds', null), 8));

const checkpointEverySec = Math.floor(parseNum(arg('checkpointEverySec', null), 60));
const checkpointKeep = Math.floor(parseNum(arg('checkpointKeep', null), 50));

const DEFAULT_WORLD_ID = 'w0';

let nextRequestId = 1;

class WorldHandle {
  constructor({ id, scenario, seed }) {
    this.id = id;
    this.worker = new Worker(workerUrl, {
      type: 'module',
      workerData: { id, scenario, seed }
    });
    this.pending = new Map();

    this.worker.on('message', (msg) => {
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === 'rpcResult') {
        const { requestId, ok, result, error } = msg;
        const pending = this.pending.get(requestId);
        if (!pending) return;
        this.pending.delete(requestId);
        if (ok) pending.resolve(result);
        else pending.reject(new Error(error || 'rpc error'));
      }
    });

    this.worker.on('error', (err) => {
      // reject all pending
      for (const p of this.pending.values()) {
        p.reject(err);
      }
      this.pending.clear();
    });

    this.worker.on('exit', (code) => {
      const err = new Error(`world worker exited (id=${id}, code=${code})`);
      for (const p of this.pending.values()) {
        p.reject(err);
      }
      this.pending.clear();
    });
  }

  rpc(method, args = {}) {
    const requestId = nextRequestId++;
    const payload = { type: 'rpc', requestId, method, args };

    return new Promise((resolvePromise, rejectPromise) => {
      this.pending.set(requestId, { resolve: resolvePromise, reject: rejectPromise });
      this.worker.postMessage(payload);

      // timeout safety
      setTimeout(() => {
        const pending = this.pending.get(requestId);
        if (!pending) return;
        this.pending.delete(requestId);
        pending.reject(new Error(`rpc timeout: ${method}`));
      }, 5000);
    });
  }

  async terminate() {
    try {
      await this.worker.terminate();
    } catch {
      // ignore
    }
  }
}

/** @type {Map<string, WorldHandle>} */
const worlds = new Map();

function ensureScenarioExists(name) {
  const n = String(name || '').trim();
  if (!n || !(n in scenarioDefs)) {
    throw new Error(`Unknown scenario: ${n}`);
  }
  return n;
}

async function createWorld({ id = null, scenario, seed } = {}) {
  const scenarioName = ensureScenarioExists(scenario);
  const resolvedId = id ? String(id).trim() : makeId('w');
  if (!resolvedId) throw new Error('Invalid world id');
  if (worlds.has(resolvedId)) throw new Error(`World already exists: ${resolvedId}`);
  if (worlds.size >= maxWorlds) throw new Error(`Too many worlds (maxWorlds=${maxWorlds})`);

  const seedU32 = (Number.isFinite(Number(seed)) ? Number(seed) : initialSeed) >>> 0;
  const handle = new WorldHandle({ id: resolvedId, scenario: scenarioName, seed: seedU32 });
  worlds.set(resolvedId, handle);

  // smoke ping
  await handle.rpc('getStatus');

  return handle;
}

async function deleteWorld(id) {
  const key = String(id || '').trim();
  if (key === DEFAULT_WORLD_ID) throw new Error('Refusing to delete default world');
  const handle = worlds.get(key);
  if (!handle) throw new Error(`Unknown world: ${key}`);
  worlds.delete(key);
  await handle.terminate();
}

function getWorldOrThrow(id) {
  const key = String(id || '').trim();
  const handle = worlds.get(key);
  if (!handle) throw new Error(`Unknown world: ${key}`);
  return handle;
}

function encodeCheckpoint(snapshot) {
  const json = JSON.stringify(snapshot);
  const gz = gzipSync(Buffer.from(json, 'utf8'));
  return { jsonBytes: Buffer.byteLength(json, 'utf8'), gz };
}

function decodeCheckpoint(gzBuffer) {
  const json = gunzipSync(gzBuffer).toString('utf8');
  return JSON.parse(json);
}

async function checkpointWorld(worldId, { label = null } = {}) {
  const handle = getWorldOrThrow(worldId);
  const status = await handle.rpc('getStatus');
  const createdAt = new Date().toISOString();

  const snapshot = await handle.rpc('saveCheckpoint', {
    meta: {
      worldId,
      label,
      createdAt,
      scenario: status?.scenario,
      seed: status?.seed
    }
  });

  const encoded = encodeCheckpoint(snapshot);

  stmtInsertCheckpoint.run(
    worldId,
    createdAt,
    label,
    String(status?.scenario || ''),
    Number(status?.seed) || 0,
    Number(status?.tick) || 0,
    Number(status?.time) || 0,
    encoded.gz.length,
    encoded.gz
  );

  if (Number.isFinite(checkpointKeep) && checkpointKeep > 0) {
    stmtPruneCheckpoints.run(worldId, checkpointKeep);
  }

  const row = db.prepare('SELECT last_insert_rowid() AS id').get();

  return {
    ok: true,
    id: Number(row?.id) || null,
    worldId,
    createdAt,
    label,
    scenario: status?.scenario,
    seed: status?.seed,
    tick: status?.tick,
    time: status?.time,
    bytes: encoded.gz.length
  };
}

// Bootstrap default world.
await createWorld({ id: DEFAULT_WORLD_ID, scenario: initialScenarioName, seed: initialSeed });

const app = Fastify({ logger: false });
await app.register(fastifyWebsocket);
await app.register(fastifyStatic, {
  root: publicDir,
  prefix: '/',
  decorateReply: false
});

// --- API ---

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
  const items = [];
  for (const [id, handle] of worlds.entries()) {
    const status = await handle.rpc('getStatus');
    items.push(status);
  }
  return { ok: true, worlds: items, defaultWorldId: DEFAULT_WORLD_ID, maxWorlds };
});

app.post('/api/worlds', async (req, reply) => {
  try {
    const scenario = String(req.body?.scenario || req.body?.name || initialScenarioName).trim();
    const seed = (Number.isFinite(Number(req.body?.seed)) ? Number(req.body.seed) : initialSeed) >>> 0;
    const id = (req.body?.id !== undefined && req.body?.id !== null) ? String(req.body.id) : null;

    const w = await createWorld({ id, scenario, seed });
    return { ok: true, id: w.id };
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.delete('/api/worlds/:id', async (req, reply) => {
  try {
    await deleteWorld(req.params.id);
    return { ok: true };
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.get('/api/worlds/:id/status', async (req, reply) => {
  try {
    return await getWorldOrThrow(req.params.id).rpc('getStatus');
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.get('/api/worlds/:id/snapshot', async (req, reply) => {
  try {
    const mode = String(req.query?.mode || 'render');
    return await getWorldOrThrow(req.params.id).rpc('getSnapshot', { mode });
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/control/pause', async (req, reply) => {
  try {
    return await getWorldOrThrow(req.params.id).rpc('pause');
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/control/resume', async (req, reply) => {
  try {
    return await getWorldOrThrow(req.params.id).rpc('resume');
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/control/setScenario', async (req, reply) => {
  try {
    const nextScenario = String(req.body?.name || req.body?.scenario || '').trim();
    const nextSeed = (Number.isFinite(Number(req.body?.seed)) ? Number(req.body.seed) : initialSeed) >>> 0;
    ensureScenarioExists(nextScenario);
    return await getWorldOrThrow(req.params.id).rpc('setScenario', { name: nextScenario, seed: nextSeed });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/control/configOverrides', async (req, reply) => {
  try {
    const overrides = req.body?.overrides;
    return await getWorldOrThrow(req.params.id).rpc('applyConfigOverrides', { overrides });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

// Persistence
app.post('/api/worlds/:id/checkpoints', async (req, reply) => {
  try {
    const label = req.body?.label ? String(req.body.label) : null;
    return await checkpointWorld(req.params.id, { label });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.get('/api/worlds/:id/checkpoints', async (req, reply) => {
  try {
    const worldId = String(req.params.id);
    // verify world exists
    getWorldOrThrow(worldId);

    const limitRaw = Number(req.query?.limit ?? 20);
    const limit = Number.isFinite(limitRaw) ? Math.max(1, Math.min(200, Math.floor(limitRaw))) : 20;

    const rows = stmtListCheckpoints.all(worldId, limit);
    return { ok: true, worldId, checkpoints: rows };
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/restore', async (req, reply) => {
  try {
    const worldId = String(req.params.id);
    const checkpointId = Number(req.body?.checkpointId);
    if (!Number.isFinite(checkpointId)) {
      reply.code(400);
      return { ok: false, error: 'checkpointId must be a number' };
    }

    const row = stmtGetCheckpoint.get(checkpointId);
    if (!row) {
      reply.code(404);
      return { ok: false, error: `checkpoint not found: ${checkpointId}` };
    }
    if (String(row.worldId) !== worldId) {
      reply.code(400);
      return { ok: false, error: `checkpoint ${checkpointId} belongs to world ${row.worldId}, not ${worldId}` };
    }

    const snapshot = decodeCheckpoint(row.payloadGzip);
    const out = await getWorldOrThrow(worldId).rpc('loadCheckpoint', { snapshot });
    return { ok: true, restored: true, checkpointId, worldId, load: out };
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

// Legacy aliases (operate on default world w0)
app.get('/api/status', async () => getWorldOrThrow(DEFAULT_WORLD_ID).rpc('getStatus'));
app.get('/api/snapshot', async (req) => {
  const mode = String(req.query?.mode || 'render');
  return getWorldOrThrow(DEFAULT_WORLD_ID).rpc('getSnapshot', { mode });
});
app.post('/api/control/pause', async () => getWorldOrThrow(DEFAULT_WORLD_ID).rpc('pause'));
app.post('/api/control/resume', async () => getWorldOrThrow(DEFAULT_WORLD_ID).rpc('resume'));
app.post('/api/control/setScenario', async (req, reply) => {
  try {
    const nextScenario = String(req.body?.name || '').trim();
    const nextSeed = (Number.isFinite(Number(req.body?.seed)) ? Number(req.body.seed) : initialSeed) >>> 0;
    ensureScenarioExists(nextScenario);
    return getWorldOrThrow(DEFAULT_WORLD_ID).rpc('setScenario', { name: nextScenario, seed: nextSeed });
  } catch (err) {
    reply.code(400);
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

  let handle;
  try {
    handle = getWorldOrThrow(worldId);
  } catch (err) {
    try {
      socket.send(JSON.stringify({ kind: 'error', error: String(err?.message || err) }));
    } finally {
      socket.close();
    }
    return;
  }

  let active = true;
  let inFlight = false;

  const tick = async () => {
    if (!active || inFlight) return;
    inFlight = true;
    try {
      const status = await handle.rpc('getStatus');
      const snapshot = await handle.rpc('getSnapshot', { mode });
      socket.send(JSON.stringify({ kind: 'status', data: status }));
      socket.send(JSON.stringify({ kind: 'snapshot', data: snapshot }));
    } catch (err) {
      try {
        socket.send(JSON.stringify({ kind: 'error', error: String(err?.message || err) }));
      } finally {
        socket.close();
      }
    } finally {
      inFlight = false;
    }
  };

  // send immediately
  tick();

  const timer = setInterval(tick, intervalMs);

  socket.on('close', () => {
    active = false;
    clearInterval(timer);
  });
});

await app.listen({ port, host: '0.0.0.0' });
// eslint-disable-next-line no-console
console.log(`[sim-server] listening on http://localhost:${port}`);
// eslint-disable-next-line no-console
console.log(`[sim-server] defaultWorld=${DEFAULT_WORLD_ID} scenario=${initialScenarioName} seed=${initialSeed} maxWorlds=${maxWorlds}`);
console.log(`[sim-server] db=${dbPath} checkpointEverySec=${checkpointEverySec} checkpointKeep=${checkpointKeep}`);

// Auto-checkpoint loop
if (checkpointEverySec > 0) {
  setInterval(() => {
    for (const worldId of worlds.keys()) {
      checkpointWorld(worldId, { label: 'auto' }).catch(() => {
        // best-effort; keep server running
      });
    }
  }, checkpointEverySec * 1000);
}
