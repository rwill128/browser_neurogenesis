#!/usr/bin/env node
import readline from 'node:readline';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { getScenario } from './scenarios.mjs';
import { MiniWorld } from './miniWorld.mjs';
import { RealWorld } from './realWorld.mjs';

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

function toNumber(v, fallback = null) {
  if (v === null || v === undefined || v === '') return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clampInt(v, min = 0) {
  return Math.max(min, Math.floor(Number(v) || 0));
}

function nowIso() {
  return new Date().toISOString();
}

class InteractiveSim {
  constructor({ scenarioName, engine, seed, dt, worldW = null, worldH = null, creatures = null, particles = null }) {
    this.engine = engine;
    this.seed = seed;
    this.timer = null;
    this.rateHz = 12;
    this.batchSteps = 1;

    this.scenarioName = scenarioName;
    this.baseScenario = getScenario(scenarioName);
    this.runtimeScenario = this._makeRuntimeScenario({ dt, worldW, worldH, creatures, particles });
    this.world = this._createWorld();
  }

  _makeRuntimeScenario({ dt = null, worldW = null, worldH = null, creatures = null, particles = null } = {}) {
    return {
      ...this.baseScenario,
      world: {
        width: worldW ?? this.baseScenario.world.width,
        height: worldH ?? this.baseScenario.world.height
      },
      creatures: creatures ?? this.baseScenario.creatures,
      particles: particles ?? this.baseScenario.particles,
      dt: dt ?? this.baseScenario.dt
    };
  }

  _createWorld() {
    const WorldImpl = this.engine === 'real' ? RealWorld : MiniWorld;
    return new WorldImpl(this.runtimeScenario, this.seed);
  }

  reconfigure({ scenarioName, engine, seed, dt, worldW, worldH, creatures, particles } = {}) {
    if (scenarioName && scenarioName !== this.scenarioName) {
      this.scenarioName = scenarioName;
      this.baseScenario = getScenario(scenarioName);
    }
    if (engine) this.engine = engine;
    if (Number.isFinite(seed)) this.seed = seed;

    this.runtimeScenario = this._makeRuntimeScenario({
      dt: dt ?? this.runtimeScenario.dt,
      worldW: worldW ?? this.runtimeScenario.world.width,
      worldH: worldH ?? this.runtimeScenario.world.height,
      creatures: creatures ?? this.runtimeScenario.creatures,
      particles: particles ?? this.runtimeScenario.particles
    });

    this.reset();
  }

  reset() {
    this.pause();
    this.world = this._createWorld();
  }

  step(steps = 1) {
    const n = clampInt(steps, 0);
    for (let i = 0; i < n; i++) {
      this.world.step(this.runtimeScenario.dt);
    }
    return this.world.snapshot();
  }

  gotoTick(targetTick) {
    const t = clampInt(targetTick, 0);
    this.pause();
    this.world = this._createWorld();
    for (let i = 0; i < t; i++) this.world.step(this.runtimeScenario.dt);
    return this.world.snapshot();
  }

  back(steps = 1) {
    const current = this.world.snapshot().tick || 0;
    return this.gotoTick(Math.max(0, current - clampInt(steps, 0)));
  }

  play(rateHz = this.rateHz, batchSteps = this.batchSteps) {
    this.pause();
    this.rateHz = Math.max(0.1, Number(rateHz) || this.rateHz);
    this.batchSteps = Math.max(1, clampInt(batchSteps, 1));
    const intervalMs = Math.max(1, Math.floor(1000 / this.rateHz));

    this.timer = setInterval(() => {
      this.step(this.batchSteps);
    }, intervalMs);

    return { running: true, rateHz: this.rateHz, batchSteps: this.batchSteps, intervalMs };
  }

  pause() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    return { running: false };
  }

  isRunning() {
    return Boolean(this.timer);
  }

  _particlesInRect(x0, y0, x1, y1) {
    if (!Array.isArray(this.world.particles)) return null;

    let count = 0;
    const sample = [];
    for (const p of this.world.particles) {
      const pos = p?.pos;
      if (!pos || p.life <= 0) continue;
      if (pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1) {
        count += 1;
        if (sample.length < 100) {
          sample.push({
            x: Number(pos.x.toFixed(2)),
            y: Number(pos.y.toFixed(2)),
            life: Number((p.life || 0).toFixed(3))
          });
        }
      }
    }
    return { count, sample };
  }

  snapshotFull() {
    return {
      generatedAt: nowIso(),
      engine: this.engine,
      scenario: this.runtimeScenario.name,
      dt: this.runtimeScenario.dt,
      world: this.runtimeScenario.world,
      tick: this.world.snapshot().tick || 0,
      running: this.isRunning(),
      snapshot: this.world.snapshot()
    };
  }

  snapshotRect(x, y, w, h) {
    const x0 = Number(x);
    const y0 = Number(y);
    const ww = Number(w);
    const hh = Number(h);
    if (![x0, y0, ww, hh].every(Number.isFinite) || ww < 0 || hh < 0) {
      throw new Error('rect requires numeric x y width height');
    }

    const x1 = x0 + ww;
    const y1 = y0 + hh;
    const snap = this.world.snapshot();
    const creatures = (snap.creatures || []).filter((c) => {
      if (Array.isArray(c.vertices) && c.vertices.length > 0) {
        return c.vertices.some((v) => v.x >= x0 && v.x <= x1 && v.y >= y0 && v.y <= y1);
      }
      const cx = c.center?.x;
      const cy = c.center?.y;
      return Number.isFinite(cx) && Number.isFinite(cy) && cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1;
    });

    return {
      generatedAt: nowIso(),
      engine: this.engine,
      scenario: this.runtimeScenario.name,
      tick: snap.tick,
      running: this.isRunning(),
      rect: { x: x0, y: y0, width: ww, height: hh },
      populations: snap.populations,
      creatures,
      particles: this._particlesInRect(x0, y0, x1, y1)
    };
  }

  snapshotCreature(id) {
    const creatureId = Number(id);
    if (!Number.isFinite(creatureId)) {
      throw new Error('creature id must be numeric');
    }

    const snap = this.world.snapshot();
    const creature = (snap.creatures || []).find((c) => Number(c.id) === creatureId) || null;

    return {
      generatedAt: nowIso(),
      engine: this.engine,
      scenario: this.runtimeScenario.name,
      tick: snap.tick,
      running: this.isRunning(),
      creatureId,
      found: Boolean(creature),
      creature
    };
  }

  status() {
    const snap = this.world.snapshot();
    return {
      engine: this.engine,
      scenario: this.runtimeScenario.name,
      seed: this.seed,
      dt: this.runtimeScenario.dt,
      world: this.runtimeScenario.world,
      configured: {
        creatures: this.runtimeScenario.creatures,
        particles: this.runtimeScenario.particles
      },
      tick: snap.tick,
      populations: snap.populations,
      running: this.isRunning(),
      rateHz: this.rateHz,
      batchSteps: this.batchSteps
    };
  }
}

function parseOutArg(parts) {
  const idx = parts.indexOf('--out');
  if (idx === -1) return { outPath: null, parts };
  const outPath = parts[idx + 1] || null;
  const next = parts.slice(0, idx).concat(parts.slice(idx + 2));
  return { outPath, parts: next };
}

function printJson(obj, outPath = null) {
  const json = JSON.stringify(obj, null, 2);
  if (outPath) {
    const abs = resolve(outPath);
    writeFileSync(abs, json + '\n', 'utf8');
    console.log(`Wrote ${abs}`);
  } else {
    console.log(json);
  }
}

function printHelp() {
  console.log(`Commands:
  help
  status
  step [n]              # advance by n steps (default 1)
  forward [n]           # alias for step
  back [n]              # rewind n steps (deterministic replay)
  goto <tick>           # jump to absolute tick via replay
  play [hz] [batch]     # run continuously (default hz=12, batch=1)
  pause                 # pause continuous run
  reset                 # rebuild world at tick 0

  snapshot [--out path]
  snapshot full [--out path]
  snapshot rect <x> <y> <w> <h> [--out path]
  snapshot creature <id> [--out path]

  set dt <value>
  set seed <int>
  set engine <mini|real>
  set scenario <name>
  set world <width> <height>
  set creatures <n>
  set particles <n>

  exit | quit
`);
}

const allowMiniCli = arg('allowMini', null) !== null;
const requestedEngine = arg('engine', 'real');
if (requestedEngine !== 'real' && !allowMiniCli) {
  throw new Error(`Engine '${requestedEngine}' is blocked by default. Use --engine real or pass --allowMini for explicit surrogate runs.`);
}

const sim = new InteractiveSim({
  scenarioName: arg('scenario', 'micro_one_creature_100'),
  engine: requestedEngine,
  seed: Number(arg('seed', '42')),
  dt: toNumber(arg('dt', null), null),
  worldW: toNumber(arg('worldW', null), null),
  worldH: toNumber(arg('worldH', null), null),
  creatures: toNumber(arg('creatures', null), null),
  particles: toNumber(arg('particles', null), null)
});

console.log('Interactive sim CLI ready. Type `help` for commands.');
console.log(JSON.stringify(sim.status(), null, 2));

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'sim> '
});

rl.prompt();
rl.on('line', (line) => {
  const raw = line.trim();
  if (!raw) {
    rl.prompt();
    return;
  }

  let parts = raw.split(/\s+/);
  const cmd = parts[0].toLowerCase();
  parts = parts.slice(1);

  try {
    if (cmd === 'help') {
      printHelp();
    } else if (cmd === 'status') {
      printJson(sim.status());
    } else if (cmd === 'step' || cmd === 'forward') {
      const n = parts[0] ? clampInt(parts[0], 1) : 1;
      printJson(sim.step(n));
    } else if (cmd === 'back') {
      const n = parts[0] ? clampInt(parts[0], 1) : 1;
      printJson(sim.back(n));
    } else if (cmd === 'goto') {
      if (!parts[0]) throw new Error('usage: goto <tick>');
      printJson(sim.gotoTick(parts[0]));
    } else if (cmd === 'play') {
      const hz = parts[0] ? Number(parts[0]) : sim.rateHz;
      const batch = parts[1] ? clampInt(parts[1], 1) : sim.batchSteps;
      printJson(sim.play(hz, batch));
    } else if (cmd === 'pause' || cmd === 'stop') {
      printJson(sim.pause());
    } else if (cmd === 'reset') {
      sim.reset();
      printJson(sim.status());
    } else if (cmd === 'snapshot') {
      const parsed = parseOutArg(parts);
      parts = parsed.parts;
      const outPath = parsed.outPath;

      const mode = (parts[0] || 'full').toLowerCase();
      if (mode === 'full') {
        printJson(sim.snapshotFull(), outPath);
      } else if (mode === 'rect') {
        if (parts.length < 5) throw new Error('usage: snapshot rect <x> <y> <w> <h> [--out path]');
        printJson(sim.snapshotRect(parts[1], parts[2], parts[3], parts[4]), outPath);
      } else if (mode === 'creature') {
        if (!parts[1]) throw new Error('usage: snapshot creature <id> [--out path]');
        printJson(sim.snapshotCreature(parts[1]), outPath);
      } else {
        throw new Error('snapshot mode must be full|rect|creature');
      }
    } else if (cmd === 'set') {
      const key = (parts[0] || '').toLowerCase();
      if (!key) throw new Error('usage: set <field> <value...>');

      if (key === 'dt') {
        const dt = toNumber(parts[1], null);
        if (!Number.isFinite(dt) || dt <= 0) throw new Error('dt must be > 0');
        sim.reconfigure({ dt });
      } else if (key === 'seed') {
        const seed = toNumber(parts[1], null);
        if (!Number.isFinite(seed)) throw new Error('seed must be numeric');
        sim.reconfigure({ seed });
      } else if (key === 'engine') {
        const engine = (parts[1] || '').toLowerCase();
        if (!['mini', 'real'].includes(engine)) throw new Error('engine must be mini|real');
        if (engine !== 'real' && !allowMiniCli) {
          throw new Error('mini engine blocked by default; pass --allowMini if you explicitly want surrogate mode');
        }
        sim.reconfigure({ engine });
      } else if (key === 'scenario') {
        const scenarioName = parts[1];
        if (!scenarioName) throw new Error('usage: set scenario <name>');
        sim.reconfigure({ scenarioName });
      } else if (key === 'world') {
        const w = toNumber(parts[1], null);
        const h = toNumber(parts[2], null);
        if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
          throw new Error('usage: set world <width> <height>');
        }
        sim.reconfigure({ worldW: w, worldH: h });
      } else if (key === 'creatures') {
        const n = toNumber(parts[1], null);
        if (!Number.isFinite(n) || n < 0) throw new Error('creatures must be >= 0');
        sim.reconfigure({ creatures: Math.floor(n) });
      } else if (key === 'particles') {
        const n = toNumber(parts[1], null);
        if (!Number.isFinite(n) || n < 0) throw new Error('particles must be >= 0');
        sim.reconfigure({ particles: Math.floor(n) });
      } else {
        throw new Error(`unknown set field: ${key}`);
      }

      printJson(sim.status());
    } else if (cmd === 'exit' || cmd === 'quit') {
      sim.pause();
      rl.close();
      return;
    } else {
      throw new Error(`unknown command: ${cmd}`);
    }
  } catch (err) {
    console.error(`Error: ${err.message}`);
  }

  rl.prompt();
});

rl.on('close', () => {
  sim.pause();
  process.exit(0);
});
