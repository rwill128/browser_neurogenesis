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

import { resolve, basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { gzipSync, gunzipSync } from 'node:zlib';

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

function parseBool(raw, fallback = false) {
  if (raw === null || raw === undefined || raw === '') return fallback;
  const s = String(raw).trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(s)) return true;
  if (['0', 'false', 'no', 'off'].includes(s)) return false;
  return fallback;
}

function hasFlag(name) {
  return process.argv.includes(`--${name}`);
}

function makeId(prefix = 'w') {
  const rand = Math.random().toString(16).slice(2, 10);
  return `${prefix}_${rand}`;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, '..');
const publicDir = resolve(__dirname, 'public');
const dataDir = resolve(__dirname, 'data');
const capturesDir = resolve(dataDir, 'captures');
const dbPath = resolve(dataDir, 'sim.sqlite');
const workerUrl = new URL('./worldWorker.mjs', import.meta.url);

mkdirSync(dataDir, { recursive: true });
mkdirSync(capturesDir, { recursive: true });

let db = null;
let dbDriver = null;
let dbInitError = null;

try {
  const mod = await import('node:sqlite');
  const DatabaseSync = mod?.DatabaseSync;
  if (typeof DatabaseSync === 'function') {
    db = new DatabaseSync(dbPath);
    dbDriver = 'node:sqlite';
  }
} catch (err) {
  dbInitError = err;
}

if (!db) {
  try {
    const mod = await import('better-sqlite3');
    const BetterSqlite = mod?.default || mod;
    db = new BetterSqlite(dbPath);
    dbDriver = 'better-sqlite3';
    dbInitError = null;
  } catch (err) {
    dbInitError = err;
  }
}

if (!db) {
  throw new Error(`Unable to initialize SQLite backend. Last error: ${String(dbInitError?.message || dbInitError)}`);
}

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

const stmtGetLatestCheckpointForWorld = db.prepare(`
  SELECT id, worldId, createdAt, label, scenario, seed, tick, time, bytes, payloadGzip
  FROM checkpoints
  WHERE worldId = ?
  ORDER BY createdAt DESC
  LIMIT 1
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
const restoreLatest = hasFlag('restoreLatest')
  ? parseBool(arg('restoreLatest', 'true'), true)
  : false;

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

function xmlEscape(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function nodeColor(name) {
  const key = String(name || '').toUpperCase();
  switch (key) {
    case 'PREDATOR': return '#ff4d4d';
    case 'EATER': return '#ff9f1c';
    case 'PHOTOSYNTHETIC': return '#7cff6b';
    case 'NEURON': return '#b48bff';
    case 'EMITTER': return '#4dd6ff';
    case 'SWIMMER': return '#4d7cff';
    case 'EYE': return '#ffffff';
    case 'JET': return '#00e5ff';
    case 'ATTRACTOR': return '#ffd24d';
    case 'REPULSOR': return '#ff4de1';
    default: return '#d7ddff';
  }
}

function renderCreatureSvg({
  creature,
  worldId,
  tick,
  time,
  size = 800,
  padding = 40,
  zoomOutFactor = 1,
  allCreatures = null,
  fluid = null,
  includeFluid = true,
  includeNeighbors = true
}) {
  const vertices = Array.isArray(creature?.vertices) ? creature.vertices : [];
  const springs = Array.isArray(creature?.springs) ? creature.springs : [];
  if (vertices.length === 0) {
    throw new Error('creature has no vertices to render');
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const v of vertices) {
    const x = Number(v?.x);
    const y = Number(v?.y);
    const r = Math.max(0.5, Number(v?.radius) || 1);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    minX = Math.min(minX, x - r);
    minY = Math.min(minY, y - r);
    maxX = Math.max(maxX, x + r);
    maxY = Math.max(maxY, y + r);
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    throw new Error('creature bounds invalid');
  }

  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const centerX = (minX + maxX) * 0.5;
  const centerY = (minY + maxY) * 0.5;
  const zoom = Math.max(1, Math.min(50, Number(zoomOutFactor) || 1));

  const viewWidth = Math.max(1, width * zoom);
  const viewHeight = Math.max(1, height * zoom);
  const viewMinX = centerX - viewWidth * 0.5;
  const viewMinY = centerY - viewHeight * 0.5;

  const svgSize = Math.max(256, Math.floor(Number(size) || 800));
  const inner = Math.max(16, svgSize - 2 * padding);
  const scale = Math.min(inner / viewWidth, inner / viewHeight);

  const drawW = viewWidth * scale;
  const drawH = viewHeight * scale;
  const offsetX = (svgSize - drawW) * 0.5;
  const offsetY = (svgSize - drawH) * 0.5;

  const sx = (x) => ((x - viewMinX) * scale + offsetX);
  const sy = (y) => ((y - viewMinY) * scale + offsetY);

  const selectedSoft = [];
  const selectedRigid = [];
  const selectedDots = [];
  const neighborSoft = [];
  const neighborRigid = [];
  const neighborDots = [];

  const appendCreatureGeometry = (srcCreature, {
    softSink,
    rigidSink,
    dotSink,
    dotOpacity = 1,
    dotRadiusScale = 1
  } = {}) => {
    const srcVertices = Array.isArray(srcCreature?.vertices) ? srcCreature.vertices : [];
    const srcSprings = Array.isArray(srcCreature?.springs) ? srcCreature.springs : [];
    if (srcVertices.length === 0) return;

    for (const s of srcSprings) {
      const a = Number(s?.a);
      const b = Number(s?.b);
      if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0 || a >= srcVertices.length || b >= srcVertices.length) continue;
      const va = srcVertices[a];
      const vb = srcVertices[b];
      const x1 = sx(Number(va?.x));
      const y1 = sy(Number(va?.y));
      const x2 = sx(Number(vb?.x));
      const y2 = sy(Number(vb?.y));
      if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
      const line = `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"/>`;
      if (s?.isRigid) rigidSink.push(line);
      else softSink.push(line);
    }

    for (const v of srcVertices) {
      const x = sx(Number(v?.x));
      const y = sy(Number(v?.y));
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const r = Math.max(1.0, (Number(v?.radius) || 1) * scale * 0.6 * dotRadiusScale);
      const fill = nodeColor(v?.nodeTypeName || v?.nodeType);
      const opacityAttr = dotOpacity < 0.999 ? ` fill-opacity="${Math.max(0, Math.min(1, dotOpacity)).toFixed(3)}"` : '';
      dotSink.push(`<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${Math.min(10, r).toFixed(2)}" fill="${fill}"${opacityAttr}/>`);
    }
  };

  const selectedId = String(creature?.id ?? '');
  const creaturePool = Array.isArray(allCreatures) && allCreatures.length > 0 ? allCreatures : [creature];

  appendCreatureGeometry(creature, {
    softSink: selectedSoft,
    rigidSink: selectedRigid,
    dotSink: selectedDots,
    dotOpacity: 1,
    dotRadiusScale: 1
  });

  let neighborsRendered = 0;
  if (includeNeighbors) {
    for (const c of creaturePool) {
      if (!c) continue;
      if (String(c?.id ?? '') === selectedId) continue;
      appendCreatureGeometry(c, {
        softSink: neighborSoft,
        rigidSink: neighborRigid,
        dotSink: neighborDots,
        dotOpacity: 0.35,
        dotRadiusScale: 0.8
      });
      neighborsRendered += 1;
    }
  }

  const fluidRects = [];
  let fluidCellsRendered = 0;
  if (includeFluid && fluid && Array.isArray(fluid?.cells) && fluid.cells.length > 0) {
    const cellWWorld = Math.max(1e-6, Number(fluid?.worldCell?.width) || 0);
    const cellHWorld = Math.max(1e-6, Number(fluid?.worldCell?.height) || 0);
    const cellW = Math.max(0.8, cellWWorld * scale);
    const cellH = Math.max(0.8, cellHWorld * scale);

    for (const cell of fluid.cells) {
      const cx = sx(Number(cell?.x));
      const cy = sy(Number(cell?.y));
      if (!Number.isFinite(cx) || !Number.isFinite(cy)) continue;

      const r = Math.max(0, Math.min(255, Math.round(Number(cell?.r) || 0)));
      const g = Math.max(0, Math.min(255, Math.round(Number(cell?.g) || 0)));
      const b = Math.max(0, Math.min(255, Math.round(Number(cell?.b) || 0)));
      const dye = Math.max(0, Number(cell?.dye) || 0);
      const alpha = Math.max(0.04, Math.min(0.45, dye / 200));

      fluidRects.push(
        `<rect x="${(cx - cellW * 0.5).toFixed(2)}" y="${(cy - cellH * 0.5).toFixed(2)}" width="${cellW.toFixed(2)}" height="${cellH.toFixed(2)}" fill="rgb(${r},${g},${b})" fill-opacity="${alpha.toFixed(3)}"/>`
      );
      fluidCellsRendered += 1;
    }
  }

  const caption = `world=${worldId} creature=${creature?.id} tick=${tick} t=${Number(time || 0).toFixed(2)} points=${vertices.length} springs=${springs.length} zoom=${zoom.toFixed(2)}x neighbors=${neighborsRendered} fluidCells=${fluidCellsRendered}`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${svgSize}" height="${svgSize}" viewBox="0 0 ${svgSize} ${svgSize}">
  <rect x="0" y="0" width="${svgSize}" height="${svgSize}" fill="#070b1a"/>
  <g>
    ${fluidRects.join('\n    ')}
  </g>
  <g opacity="0.14" stroke="#a8b7ff" stroke-width="0.9" fill="none">
    ${neighborSoft.join('\n    ')}
  </g>
  <g opacity="0.22" stroke="#ffe16b" stroke-width="1.0" fill="none">
    ${neighborRigid.join('\n    ')}
  </g>
  <g>
    ${neighborDots.join('\n    ')}
  </g>
  <g opacity="0.35" stroke="#cfd6ff" stroke-width="1.2" fill="none">
    ${selectedSoft.join('\n    ')}
  </g>
  <g opacity="0.70" stroke="#ffe16b" stroke-width="1.5" fill="none">
    ${selectedRigid.join('\n    ')}
  </g>
  <g>
    ${selectedDots.join('\n    ')}
  </g>
  <text x="14" y="${svgSize - 14}" font-family="ui-monospace, Menlo, monospace" font-size="12" fill="#d7deff">${xmlEscape(caption)}</text>
</svg>`;
}

function readCaptureFilePayload(fileParam) {
  const fileName = basename(String(fileParam || ''));
  const lower = fileName.toLowerCase();
  const isSvg = lower.endsWith('.svg');
  const isPng = lower.endsWith('.png');
  const isMp4 = lower.endsWith('.mp4');
  const isGif = lower.endsWith('.gif');
  const isWebm = lower.endsWith('.webm');

  if (!fileName || (!isSvg && !isPng && !isMp4 && !isGif && !isWebm)) {
    throw new Error('invalid capture file name');
  }

  const absPath = resolve(capturesDir, fileName);
  if (!absPath.startsWith(capturesDir)) {
    throw new Error('invalid capture path');
  }

  if (isSvg) {
    return {
      payload: readFileSync(absPath, 'utf8'),
      contentType: 'image/svg+xml; charset=utf-8'
    };
  }

  const contentType = isPng
    ? 'image/png'
    : (isMp4
      ? 'video/mp4'
      : (isGif
        ? 'image/gif'
        : 'video/webm'));

  return {
    payload: readFileSync(absPath),
    contentType
  };
}

async function captureCreaturePortrait({
  worldId,
  creatureId = null,
  random = true,
  size = 800,
  zoomOutFactor = 1,
  includeFluid = true,
  includeNeighbors = true
}) {
  const handle = getWorldOrThrow(worldId);
  const snap = await handle.rpc('getSnapshot', { mode: 'render' });
  const creatures = Array.isArray(snap?.creatures) ? snap.creatures : [];
  if (creatures.length === 0) {
    throw new Error(`no creatures available in world ${worldId}`);
  }

  let selected = null;
  if (creatureId !== null && creatureId !== undefined && creatureId !== '') {
    const target = String(creatureId);
    selected = creatures.find((c) => String(c?.id) === target) || null;
    if (!selected) {
      throw new Error(`creature not found: ${target}`);
    }
  } else if (random) {
    selected = creatures[Math.floor(Math.random() * creatures.length)] || creatures[0];
  } else {
    selected = creatures[0];
  }

  const svg = renderCreatureSvg({
    creature: selected,
    worldId,
    tick: snap?.tick,
    time: snap?.time,
    size,
    zoomOutFactor,
    allCreatures: creatures,
    fluid: snap?.fluid,
    includeFluid,
    includeNeighbors
  });

  const safeWorldId = String(worldId).replace(/[^a-zA-Z0-9_-]+/g, '_');
  const safeCreatureId = String(selected?.id ?? 'unknown').replace(/[^a-zA-Z0-9_-]+/g, '_');
  const fileName = `creature-${safeWorldId}-${safeCreatureId}-tick${Number(snap?.tick) || 0}-${Date.now()}.svg`;
  const absPath = resolve(capturesDir, fileName);
  writeFileSync(absPath, svg, 'utf8');

  return {
    ok: true,
    worldId,
    scenario: snap?.scenario || null,
    tick: Number(snap?.tick) || 0,
    time: Number(snap?.time) || 0,
    creatureId: selected?.id ?? null,
    vertices: Array.isArray(selected?.vertices) ? selected.vertices.length : 0,
    springs: Array.isArray(selected?.springs) ? selected.springs.length : 0,
    zoomOutFactor: Math.max(1, Math.min(50, Number(zoomOutFactor) || 1)),
    includeFluid: Boolean(includeFluid),
    includeNeighbors: Boolean(includeNeighbors),
    fileName,
    filePath: absPath,
    downloadUrl: `/api/worlds/${encodeURIComponent(worldId)}/captures/${encodeURIComponent(fileName)}`
  };
}

function waitMs(ms) {
  const delay = Math.max(0, Number(ms) || 0);
  return new Promise((resolvePromise) => setTimeout(resolvePromise, delay));
}

async function captureCreatureClip({
  worldId,
  creatureId = null,
  random = true,
  size = 800,
  durationSec = 5,
  fps = 12,
  zoomOutFactor = 1,
  includeFluid = true,
  includeNeighbors = true
}) {
  const handle = getWorldOrThrow(worldId);

  const clipDurationSec = Math.max(1, Math.min(20, Number(durationSec) || 5));
  const clipFps = Math.max(4, Math.min(30, Math.floor(Number(fps) || 12)));
  const frameCountTarget = Math.max(2, Math.round(clipDurationSec * clipFps));
  const frameIntervalMs = 1000 / clipFps;
  const renderSize = Math.max(256, Math.floor(Number(size) || 800));

  const firstSnap = await handle.rpc('getSnapshot', { mode: 'render' });
  const firstCreatures = Array.isArray(firstSnap?.creatures) ? firstSnap.creatures : [];
  if (firstCreatures.length === 0) {
    throw new Error(`no creatures available in world ${worldId}`);
  }

  let selectedId = null;
  if (creatureId !== null && creatureId !== undefined && creatureId !== '') {
    selectedId = String(creatureId);
    const exists = firstCreatures.some((c) => String(c?.id) === selectedId);
    if (!exists) throw new Error(`creature not found: ${selectedId}`);
  } else if (random) {
    selectedId = String(firstCreatures[Math.floor(Math.random() * firstCreatures.length)]?.id);
  } else {
    selectedId = String(firstCreatures[0]?.id);
  }

  const frames = [];
  let endedEarly = false;

  for (let i = 0; i < frameCountTarget; i++) {
    if (i > 0) await waitMs(frameIntervalMs);

    const snap = (i === 0)
      ? firstSnap
      : await handle.rpc('getSnapshot', { mode: 'render' });

    const creatures = Array.isArray(snap?.creatures) ? snap.creatures : [];
    const creature = creatures.find((c) => String(c?.id) === selectedId) || null;
    if (!creature) {
      endedEarly = true;
      break;
    }

    frames.push({
      tick: Number(snap?.tick) || 0,
      time: Number(snap?.time) || 0,
      creature,
      creatures,
      fluid: snap?.fluid || null
    });
  }

  if (frames.length === 0) {
    throw new Error(`creature ${selectedId} disappeared before capture started`);
  }

  if (frames.length === 1) {
    frames.push({ ...frames[0] });
    endedEarly = true;
  }

  const safeWorldId = String(worldId).replace(/[^a-zA-Z0-9_-]+/g, '_');
  const safeCreatureId = String(selectedId || 'unknown').replace(/[^a-zA-Z0-9_-]+/g, '_');
  const stamp = Date.now();
  const fileName = `creature-clip-${safeWorldId}-${safeCreatureId}-tick${frames[0].tick}-${stamp}.mp4`;
  const absPath = resolve(capturesDir, fileName);

  const tempRoot = mkdtempSync(join(tmpdir(), 'bn-creature-clip-'));
  const framesDir = join(tempRoot, 'frames');
  mkdirSync(framesDir, { recursive: true });

  try {
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      const svg = renderCreatureSvg({
        creature: f.creature,
        worldId,
        tick: f.tick,
        time: f.time,
        size: renderSize,
        zoomOutFactor,
        allCreatures: f.creatures,
        fluid: f.fluid,
        includeFluid,
        includeNeighbors
      });

      const idx = String(i).padStart(4, '0');
      const svgPath = join(framesDir, `frame-${idx}.svg`);
      const pngPath = join(framesDir, `frame-${idx}.png`);
      writeFileSync(svgPath, svg, 'utf8');

      execFileSync('sips', ['-s', 'format', 'png', svgPath, '--out', pngPath], {
        stdio: 'ignore'
      });
    }

    execFileSync('ffmpeg', [
      '-y',
      '-hide_banner',
      '-loglevel', 'error',
      '-framerate', String(clipFps),
      '-start_number', '0',
      '-i', join(framesDir, 'frame-%04d.png'),
      '-c:v', 'libx264',
      '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart',
      absPath
    ], {
      stdio: 'ignore'
    });
  } catch (err) {
    throw new Error(`failed to build creature clip: ${String(err?.message || err)}`);
  } finally {
    rmSync(tempRoot, { recursive: true, force: true });
  }

  return {
    ok: true,
    worldId,
    scenario: firstSnap?.scenario || null,
    creatureId: Number.isFinite(Number(selectedId)) ? Number(selectedId) : selectedId,
    tickStart: Number(frames[0]?.tick) || 0,
    tickEnd: Number(frames[frames.length - 1]?.tick) || 0,
    timeStart: Number(frames[0]?.time) || 0,
    timeEnd: Number(frames[frames.length - 1]?.time) || 0,
    fps: clipFps,
    zoomOutFactor: Math.max(1, Math.min(50, Number(zoomOutFactor) || 1)),
    includeFluid: Boolean(includeFluid),
    includeNeighbors: Boolean(includeNeighbors),
    durationRequestedSec: clipDurationSec,
    durationCapturedSec: Number(((frames.length - 1) / clipFps).toFixed(3)),
    framesCaptured: frames.length,
    endedEarly,
    fileName,
    filePath: absPath,
    downloadUrl: `/api/worlds/${encodeURIComponent(worldId)}/captures/${encodeURIComponent(fileName)}`
  };
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

async function restoreLatestCheckpointForWorld(worldId) {
  const row = stmtGetLatestCheckpointForWorld.get(worldId);
  if (!row) {
    return {
      ok: true,
      worldId,
      restored: false,
      reason: 'no_checkpoint_found'
    };
  }

  const snapshot = decodeCheckpoint(row.payloadGzip);
  const load = await getWorldOrThrow(worldId).rpc('loadCheckpoint', { snapshot });

  return {
    ok: true,
    worldId,
    restored: true,
    checkpointId: Number(row.id) || null,
    scenario: String(row.scenario || ''),
    seed: Number(row.seed) || 0,
    tick: Number(row.tick) || 0,
    time: Number(row.time) || 0,
    createdAt: String(row.createdAt || ''),
    bytes: Number(row.bytes) || 0,
    load
  };
}

// Bootstrap default world.
await createWorld({ id: DEFAULT_WORLD_ID, scenario: initialScenarioName, seed: initialSeed });

let startupRestore = {
  enabled: restoreLatest,
  attempted: false,
  restored: false,
  reason: 'disabled'
};

if (restoreLatest) {
  startupRestore.attempted = true;
  try {
    startupRestore = {
      enabled: true,
      attempted: true,
      ...(await restoreLatestCheckpointForWorld(DEFAULT_WORLD_ID))
    };
  } catch (err) {
    startupRestore = {
      enabled: true,
      attempted: true,
      restored: false,
      reason: 'restore_error',
      error: String(err?.message || err)
    };
  }
}

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
    const frameOffsetRaw = req.query?.frameOffset;
    const frameSeqRaw = req.query?.frameSeq;

    const args = { mode };
    if (frameOffsetRaw !== undefined) {
      const n = Number(frameOffsetRaw);
      if (Number.isFinite(n)) args.frameOffset = Math.max(0, Math.floor(n));
    }
    if (frameSeqRaw !== undefined) {
      const n = Number(frameSeqRaw);
      if (Number.isFinite(n)) args.frameSeq = Math.floor(n);
    }

    return await getWorldOrThrow(req.params.id).rpc('getSnapshot', args);
  } catch (err) {
    reply.code(404);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.get('/api/worlds/:id/frameTimeline', async (req, reply) => {
  try {
    return await getWorldOrThrow(req.params.id).rpc('getFrameTimeline');
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

app.post('/api/worlds/:id/control/runMode', async (req, reply) => {
  try {
    const mode = String(req.body?.mode || '').trim();
    return await getWorldOrThrow(req.params.id).rpc('setRunMode', { mode });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.get('/api/worlds/:id/control/runMode', async (req, reply) => {
  try {
    return await getWorldOrThrow(req.params.id).rpc('getRunMode');
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

app.get('/api/worlds/:id/config', async (req, reply) => {
  try {
    const keys = Array.isArray(req.query?.keys)
      ? req.query.keys
      : null;
    return await getWorldOrThrow(req.params.id).rpc('getConfig', { keys });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/capture/randomCreature', async (req, reply) => {
  try {
    const worldId = String(req.params.id);
    const creatureId = req.body?.creatureId ?? null;
    const random = creatureId == null;
    const sizeRaw = Number(req.body?.size ?? req.query?.size ?? 800);
    const zoomRaw = Number(req.body?.zoomOutFactor ?? req.query?.zoomOutFactor ?? 1);
    const includeFluid = parseBool(req.body?.includeFluid ?? req.query?.includeFluid ?? true, true);
    const includeNeighbors = parseBool(req.body?.includeNeighbors ?? req.query?.includeNeighbors ?? true, true);
    const size = Number.isFinite(sizeRaw) ? Math.max(256, Math.min(2048, Math.floor(sizeRaw))) : 800;
    const zoomOutFactor = Number.isFinite(zoomRaw) ? Math.max(1, Math.min(50, zoomRaw)) : 1;

    return await captureCreaturePortrait({ worldId, creatureId, random, size, zoomOutFactor, includeFluid, includeNeighbors });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.post('/api/worlds/:id/capture/creatureClip', async (req, reply) => {
  try {
    const worldId = String(req.params.id);
    const creatureId = req.body?.creatureId ?? null;
    const random = creatureId == null;
    const sizeRaw = Number(req.body?.size ?? req.query?.size ?? 800);
    const durationRaw = Number(req.body?.durationSec ?? req.query?.durationSec ?? 5);
    const fpsRaw = Number(req.body?.fps ?? req.query?.fps ?? 12);
    const zoomRaw = Number(req.body?.zoomOutFactor ?? req.query?.zoomOutFactor ?? 1);
    const includeFluid = parseBool(req.body?.includeFluid ?? req.query?.includeFluid ?? true, true);
    const includeNeighbors = parseBool(req.body?.includeNeighbors ?? req.query?.includeNeighbors ?? true, true);

    const size = Number.isFinite(sizeRaw) ? Math.max(256, Math.min(2048, Math.floor(sizeRaw))) : 800;
    const durationSec = Number.isFinite(durationRaw) ? Math.max(1, Math.min(20, durationRaw)) : 5;
    const fps = Number.isFinite(fpsRaw) ? Math.max(4, Math.min(30, Math.floor(fpsRaw))) : 12;
    const zoomOutFactor = Number.isFinite(zoomRaw) ? Math.max(1, Math.min(50, zoomRaw)) : 1;

    return await captureCreatureClip({ worldId, creatureId, random, size, durationSec, fps, zoomOutFactor, includeFluid, includeNeighbors });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});

app.get('/api/worlds/:id/captures/:file', async (req, reply) => {
  try {
    const out = readCaptureFilePayload(req.params.file);
    reply.header('content-type', out.contentType);
    reply.header('cache-control', 'no-store');
    return reply.send(out.payload);
  } catch (err) {
    const msg = String(err?.message || err);
    reply.code(msg.includes('invalid capture') ? 400 : 404);
    return { ok: false, error: msg };
  }
});

// Legacy aliases (operate on default world w0)
app.get('/api/status', async () => getWorldOrThrow(DEFAULT_WORLD_ID).rpc('getStatus'));
app.get('/api/snapshot', async (req) => {
  const mode = String(req.query?.mode || 'render');
  const args = { mode };

  if (req.query?.frameOffset !== undefined) {
    const n = Number(req.query.frameOffset);
    if (Number.isFinite(n)) args.frameOffset = Math.max(0, Math.floor(n));
  }
  if (req.query?.frameSeq !== undefined) {
    const n = Number(req.query.frameSeq);
    if (Number.isFinite(n)) args.frameSeq = Math.floor(n);
  }

  return getWorldOrThrow(DEFAULT_WORLD_ID).rpc('getSnapshot', args);
});
app.get('/api/frameTimeline', async () => getWorldOrThrow(DEFAULT_WORLD_ID).rpc('getFrameTimeline'));
app.post('/api/capture/randomCreature', async (req, reply) => {
  try {
    const creatureId = req.body?.creatureId ?? null;
    const random = creatureId == null;
    const sizeRaw = Number(req.body?.size ?? req.query?.size ?? 800);
    const zoomRaw = Number(req.body?.zoomOutFactor ?? req.query?.zoomOutFactor ?? 1);
    const includeFluid = parseBool(req.body?.includeFluid ?? req.query?.includeFluid ?? true, true);
    const includeNeighbors = parseBool(req.body?.includeNeighbors ?? req.query?.includeNeighbors ?? true, true);
    const size = Number.isFinite(sizeRaw) ? Math.max(256, Math.min(2048, Math.floor(sizeRaw))) : 800;
    const zoomOutFactor = Number.isFinite(zoomRaw) ? Math.max(1, Math.min(50, zoomRaw)) : 1;
    return await captureCreaturePortrait({ worldId: DEFAULT_WORLD_ID, creatureId, random, size, zoomOutFactor, includeFluid, includeNeighbors });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});
app.post('/api/capture/creatureClip', async (req, reply) => {
  try {
    const creatureId = req.body?.creatureId ?? null;
    const random = creatureId == null;
    const sizeRaw = Number(req.body?.size ?? req.query?.size ?? 800);
    const durationRaw = Number(req.body?.durationSec ?? req.query?.durationSec ?? 5);
    const fpsRaw = Number(req.body?.fps ?? req.query?.fps ?? 12);
    const zoomRaw = Number(req.body?.zoomOutFactor ?? req.query?.zoomOutFactor ?? 1);
    const includeFluid = parseBool(req.body?.includeFluid ?? req.query?.includeFluid ?? true, true);
    const includeNeighbors = parseBool(req.body?.includeNeighbors ?? req.query?.includeNeighbors ?? true, true);

    const size = Number.isFinite(sizeRaw) ? Math.max(256, Math.min(2048, Math.floor(sizeRaw))) : 800;
    const durationSec = Number.isFinite(durationRaw) ? Math.max(1, Math.min(20, durationRaw)) : 5;
    const fps = Number.isFinite(fpsRaw) ? Math.max(4, Math.min(30, Math.floor(fpsRaw))) : 12;
    const zoomOutFactor = Number.isFinite(zoomRaw) ? Math.max(1, Math.min(50, zoomRaw)) : 1;

    return await captureCreatureClip({ worldId: DEFAULT_WORLD_ID, creatureId, random, size, durationSec, fps, zoomOutFactor, includeFluid, includeNeighbors });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});
app.get('/api/captures/:file', async (req, reply) => {
  try {
    const out = readCaptureFilePayload(req.params.file);
    reply.header('content-type', out.contentType);
    reply.header('cache-control', 'no-store');
    return reply.send(out.payload);
  } catch (err) {
    const msg = String(err?.message || err);
    reply.code(msg.includes('invalid capture') ? 400 : 404);
    return { ok: false, error: msg };
  }
});
app.post('/api/control/pause', async () => getWorldOrThrow(DEFAULT_WORLD_ID).rpc('pause'));
app.post('/api/control/resume', async () => getWorldOrThrow(DEFAULT_WORLD_ID).rpc('resume'));
app.post('/api/control/runMode', async (req, reply) => {
  try {
    const mode = String(req.body?.mode || '').trim();
    return getWorldOrThrow(DEFAULT_WORLD_ID).rpc('setRunMode', { mode });
  } catch (err) {
    reply.code(400);
    return { ok: false, error: String(err?.message || err) };
  }
});
app.get('/api/control/runMode', async () => getWorldOrThrow(DEFAULT_WORLD_ID).rpc('getRunMode'));
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
console.log(`[sim-server] db=${dbPath} driver=${dbDriver} checkpointEverySec=${checkpointEverySec} checkpointKeep=${checkpointKeep}`);
console.log(`[sim-server] restoreLatest=${restoreLatest}`);
if (restoreLatest) {
  console.log(`[sim-server] startupRestore=${JSON.stringify(startupRestore)}`);
}

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
