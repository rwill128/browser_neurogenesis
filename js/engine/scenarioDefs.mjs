export const scenarioDefs = {
  baseline: {
    name: 'baseline',
    description: 'Default full simulation settings',
    browserConfig: {},
    nodeConfig: { world: { width: 140, height: 90 }, creatures: 10, particles: 240, dt: 1/30, steps: 300 }
  },
  micro_stability: {
    name: 'micro_stability',
    description: 'Small world for stability debugging',
    browserConfig: {
      WORLD_WIDTH: 1800,
      WORLD_HEIGHT: 1200,
      CREATURE_POPULATION_FLOOR: 24,
      CREATURE_POPULATION_CEILING: 48,
      PARTICLE_POPULATION_FLOOR: 1500,
      PARTICLE_POPULATION_CEILING: 5000,
      PARTICLES_PER_SECOND: 120,
      FLUID_GRID_SIZE_CONTROL: 96,
      FLUID_DIFFUSION: 0.0005,
      FLUID_VISCOSITY: 0.0009,
      AUTO_FOLLOW_CREATURE: true
    },
    nodeConfig: {
      world: { width: 120, height: 80 }, creatures: 8, particles: 200, dt: 1/30, steps: 300,
      events: [
        { tick: 60, kind: 'energySpike', amount: 8 },
        { tick: 120, kind: 'velocityKick', amount: 1.2 }
      ]
    }
  },
  micro_predation: {
    name: 'micro_predation',
    description: 'Compact world with higher interactions',
    browserConfig: {
      WORLD_WIDTH: 1400,
      WORLD_HEIGHT: 900,
      CREATURE_POPULATION_FLOOR: 36,
      CREATURE_POPULATION_CEILING: 72,
      PARTICLE_POPULATION_FLOOR: 2200,
      PARTICLE_POPULATION_CEILING: 7000,
      PARTICLES_PER_SECOND: 180,
      FLUID_GRID_SIZE_CONTROL: 96,
      FLUID_DIFFUSION: 0.00045,
      FLUID_VISCOSITY: 0.00075,
      AUTO_FOLLOW_CREATURE: true
    },
    nodeConfig: {
      world: { width: 100, height: 70 }, creatures: 12, particles: 260, dt: 1/30, steps: 300,
      events: [
        { tick: 80, kind: 'energyDrain', amount: 12 },
        { tick: 140, kind: 'velocityKick', amount: 1.6 }
      ]
    }
  },
  micro_single_100: {
    name: 'micro_single_100',
    description: 'Tiny 100x100 world with one creature for render debugging',
    browserConfig: {
      WORLD_WIDTH: 100,
      WORLD_HEIGHT: 100,
      CREATURE_POPULATION_FLOOR: 1,
      CREATURE_POPULATION_CEILING: 1,
      PARTICLE_POPULATION_FLOOR: 40,
      PARTICLE_POPULATION_CEILING: 120,
      PARTICLES_PER_SECOND: 4,
      FLUID_GRID_SIZE_CONTROL: 48,
      AUTO_FOLLOW_CREATURE: true
    },
    nodeConfig: {
      world: { width: 100, height: 100 }, creatures: 1, particles: 40, dt: 1/30, steps: 240,
      events: [
        { tick: 60, kind: 'velocityKick', amount: 0.4 },
        { tick: 150, kind: 'energySpike', amount: 4 }
      ]
    }
  },
  micro_blank_100: {
    name: 'micro_blank_100',
    description: '100x100 blank sanity check: no creatures, no particles, no fluid forcing',
    browserConfig: {
      WORLD_WIDTH: 100,
      WORLD_HEIGHT: 100,
      CREATURE_POPULATION_FLOOR: 0,
      CREATURE_POPULATION_CEILING: 0,
      PARTICLE_POPULATION_FLOOR: 0,
      PARTICLE_POPULATION_CEILING: 0,
      PARTICLES_PER_SECOND: 0,
      FLUID_GRID_SIZE_CONTROL: 32,
      FLUID_DIFFUSION: 0,
      FLUID_VISCOSITY: 0,
      FLUID_FADE_RATE: 0,
      EMITTER_STRENGTH: 0,
      FLUID_CURRENT_STRENGTH_ON_BODY: 0,
      AUTO_FOLLOW_CREATURE: false
    },
    nodeConfig: {
      world: { width: 100, height: 100 }, creatures: 0, particles: 0, dt: 1/30, steps: 120,
      events: []
    }
  }
};

export function getScenarioDef(name = 'baseline') {
  return scenarioDefs[name] || scenarioDefs.baseline;
}
