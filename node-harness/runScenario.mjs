#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { getScenario } from './scenarios.mjs';
import { RealWorld } from './realWorld.mjs';

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

const scenarioName = arg('scenario', 'micro_stability');
const seed = Number(arg('seed', '42'));
const stepsArg = arg('steps', null);
const outDir = resolve(arg('out', './artifacts'));

if (arg('engine', null) !== null || arg('allowMini', null) !== null) {
  throw new Error('Engine selection is no longer supported. This runner always uses the real simulation code path.');
}

const scenario = getScenario(scenarioName);
const worldW = arg('worldW', null);
const worldH = arg('worldH', null);
const creaturesArg = arg('creatures', null);
const particlesArg = arg('particles', null);
const dtArg = arg('dt', null);

const runtimeScenario = {
  ...scenario,
  world: {
    width: worldW ? Number(worldW) : scenario.world.width,
    height: worldH ? Number(worldH) : scenario.world.height
  },
  creatures: creaturesArg ? Number(creaturesArg) : scenario.creatures,
  particles: particlesArg ? Number(particlesArg) : scenario.particles,
  dt: dtArg ? Number(dtArg) : scenario.dt
};

const steps = stepsArg ? Number(stepsArg) : runtimeScenario.steps;
const world = new RealWorld(runtimeScenario, seed);

const timeline = [];
for (let i = 0; i < steps; i++) {
  world.step(runtimeScenario.dt);
  if (i % 10 === 0 || i === steps - 1) timeline.push(world.snapshot());
}

mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, `${scenario.name}-seed${seed}-steps${steps}.json`);
const payload = {
  scenario: runtimeScenario.name,
  engine: 'real',
  seed,
  steps,
  dt: runtimeScenario.dt,
  world: runtimeScenario.world,
  creatures: runtimeScenario.creatures,
  particles: runtimeScenario.particles,
  generatedAt: new Date().toISOString(),
  timeline
};
writeFileSync(outPath, JSON.stringify(payload, null, 2), 'utf8');
console.log(`Wrote ${outPath}`);
