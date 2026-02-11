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
  }
};

export function getScenarioDef(name = 'baseline') {
  return scenarioDefs[name] || scenarioDefs.baseline;
}
