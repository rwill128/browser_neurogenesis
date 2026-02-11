#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { getScenario } from './scenarios.mjs';
import { MiniWorld } from './miniWorld.mjs';

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

const scenarioName = arg('scenario', 'micro_stability');
const seed = Number(arg('seed', '42'));
const stepsArg = arg('steps', null);
const outDir = resolve(arg('out', './artifacts'));

const scenario = getScenario(scenarioName);
const steps = stepsArg ? Number(stepsArg) : scenario.steps;
const world = new MiniWorld(scenario, seed);

const timeline = [];
for (let i = 0; i < steps; i++) {
  world.step(scenario.dt);
  if (i % 10 === 0 || i === steps - 1) timeline.push(world.snapshot());
}

mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, `${scenario.name}-seed${seed}-steps${steps}.json`);
const payload = {
  scenario: scenario.name,
  seed,
  steps,
  dt: scenario.dt,
  world: scenario.world,
  generatedAt: new Date().toISOString(),
  timeline
};
writeFileSync(outPath, JSON.stringify(payload, null, 2), 'utf8');
console.log(`Wrote ${outPath}`);
