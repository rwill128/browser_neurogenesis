import config from '../config.js';

export const SCENARIOS = {
  baseline: {
    description: 'Default full simulation settings',
    apply: () => {}
  },
  micro_stability: {
    description: 'Small world for stability debugging',
    apply: () => {
      config.WORLD_WIDTH = 1800;
      config.WORLD_HEIGHT = 1200;
      config.CREATURE_POPULATION_FLOOR = 24;
      config.CREATURE_POPULATION_CEILING = 48;
      config.PARTICLE_POPULATION_FLOOR = 1500;
      config.PARTICLE_POPULATION_CEILING = 5000;
      config.PARTICLES_PER_SECOND = 120;
      config.FLUID_GRID_SIZE_CONTROL = 96;
      config.FLUID_DIFFUSION = 0.0005;
      config.FLUID_VISCOSITY = 0.0009;
      config.AUTO_FOLLOW_CREATURE = true;
    }
  },
  micro_predation: {
    description: 'Compact world with higher interactions',
    apply: () => {
      config.WORLD_WIDTH = 1400;
      config.WORLD_HEIGHT = 900;
      config.CREATURE_POPULATION_FLOOR = 36;
      config.CREATURE_POPULATION_CEILING = 72;
      config.PARTICLE_POPULATION_FLOOR = 2200;
      config.PARTICLE_POPULATION_CEILING = 7000;
      config.PARTICLES_PER_SECOND = 180;
      config.FLUID_GRID_SIZE_CONTROL = 96;
      config.FLUID_DIFFUSION = 0.00045;
      config.FLUID_VISCOSITY = 0.00075;
      config.AUTO_FOLLOW_CREATURE = true;
    }
  }
};

export function getScenarioNameFromUrl() {
  const p = new URLSearchParams(window.location.search);
  return p.get('scenario') || p.get('mini') || 'baseline';
}

export function applyScenarioFromUrl() {
  const name = getScenarioNameFromUrl();
  const scenario = SCENARIOS[name] || SCENARIOS.baseline;
  scenario.apply();
  config.DEBUG_SCENARIO = name;
  return { name, description: scenario.description || '' };
}
