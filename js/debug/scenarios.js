import config from '../config.js';
import { mulberry32 } from '../engine/random.mjs';
import { getScenarioDef, scenarioDefs } from '../engine/scenarioDefs.mjs';

export const SCENARIOS = scenarioDefs;

export function getScenarioNameFromUrl() {
  const p = new URLSearchParams(window.location.search);
  return p.get('scenario') || p.get('mini') || 'baseline';
}

function applySeedFromUrl() {
  const p = new URLSearchParams(window.location.search);
  const raw = p.get('seed');
  if (!raw) return null;

  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return null;
  const seed = parsed >>> 0;

  if (!window.__originalMathRandom) {
    window.__originalMathRandom = Math.random;
  }
  Math.random = mulberry32(seed);
  config.DEBUG_SEED = seed;
  return seed;
}

export function applyScenarioFromUrl() {
  const name = getScenarioNameFromUrl();
  const scenario = getScenarioDef(name);
  const browserConfig = scenario.browserConfig || {};
  for (const [k, v] of Object.entries(browserConfig)) {
    config[k] = v;
  }
  const seed = applySeedFromUrl();
  config.DEBUG_SCENARIO = scenario.name;
  return { name: scenario.name, description: scenario.description || '', seed };
}
