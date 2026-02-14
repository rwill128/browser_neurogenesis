export const scenarioDefs = {
  baseline: {
    name: 'baseline',
    description: 'Default full simulation settings',
    browserConfig: {},
    nodeConfig: { world: { width: 140, height: 90 }, creatures: 10, particles: 240, dt: 1/3, steps: 300 }
  },
  browser_default_big: {
    name: 'browser_default_big',
    description: 'Big browser-like default world (80000x64000, creature floor=0, particle-free)',
    browserConfig: {
      WORLD_WIDTH: 80000,
      WORLD_HEIGHT: 64000,
      CREATURE_POPULATION_FLOOR: 0,
      CREATURE_POPULATION_CEILING: 10000,
      PARTICLE_POPULATION_FLOOR: 0,
      PARTICLE_POPULATION_CEILING: 0,
      PARTICLES_PER_SECOND: 0,
      EATER_NODE_ENERGY_COST: 0.25,
      FLUID_FADE_RATE: 0.003,
      LANDSCAPE_DYE_EMITTERS_ENABLED: true,
      LANDSCAPE_DYE_EMITTER_COUNT: 42,
      LANDSCAPE_DYE_EMITTER_STRENGTH_MIN: 10,
      LANDSCAPE_DYE_EMITTER_STRENGTH_MAX: 24,
      LANDSCAPE_DYE_EMITTER_RADIUS_CELLS: 1,
      MIN_VISCOSITY_MULTIPLIER: 0.1,
      MAX_VISCOSITY_MULTIPLIER: 12.0,
      VISCOSITY_LANDSCAPE_CONTRAST: 0.9,
      VISCOSITY_LANDSCAPE_BANDS: 12,
      FLUID_GRID_SIZE_CONTROL: 128,
      FLUID_STEP_EVERY_N_TICKS: 4,
      FLUID_MOMENTUM_ONLY_STEP_EVERY_N_TICKS: 10,
      AUTO_FOLLOW_CREATURE: true
    },
    nodeConfig: {
      world: { width: 80000, height: 64000 },
      creatures: 600,
      creatureFloor: 0,
      creatureCeiling: 10000,
      particles: 0,
      particleFloor: 0,
      particleCeiling: 0,
      particlesPerSecond: 0,
      dt: 0.001,
      steps: 0,
      events: [],
      configOverrides: {
        EATER_NODE_ENERGY_COST: 0.25,
        FLUID_GRID_SIZE_CONTROL: 128,
        FLUID_STEP_EVERY_N_TICKS: 4,
        FLUID_MOMENTUM_ONLY_STEP_EVERY_N_TICKS: 10,
        FLUID_FADE_RATE: 0.003,
        LANDSCAPE_DYE_EMITTERS_ENABLED: true,
        LANDSCAPE_DYE_EMITTER_COUNT: 42,
        LANDSCAPE_DYE_EMITTER_STRENGTH_MIN: 10,
        LANDSCAPE_DYE_EMITTER_STRENGTH_MAX: 24,
        LANDSCAPE_DYE_EMITTER_RADIUS_CELLS: 1,
        MIN_VISCOSITY_MULTIPLIER: 0.1,
        MAX_VISCOSITY_MULTIPLIER: 12.0,
        VISCOSITY_LANDSCAPE_CONTRAST: 0.9,
        VISCOSITY_LANDSCAPE_BANDS: 12
      },
      stepBehavior: {
        allowReproduction: true,
        maintainCreatureFloor: true,
        maintainParticleFloor: true,
        applyEmitters: true
      }
    }
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
      world: { width: 120, height: 80 }, creatures: 8, creatureCeiling: 16, particles: 200, particleCeiling: 400, dt: 1/3, steps: 300,
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
      world: { width: 100, height: 70 }, creatures: 12, creatureCeiling: 24, particles: 260, particleCeiling: 520, dt: 1/3, steps: 300,
      events: [
        { tick: 80, kind: 'energyDrain', amount: 12 },
        { tick: 140, kind: 'velocityKick', amount: 1.6 }
      ]
    }
  },
  micro_repro_sustain: {
    name: 'micro_repro_sustain',
    description: 'Micro ecology tuned for reproduction-led persistence (no creature floor refill)',
    browserConfig: {
      WORLD_WIDTH: 320,
      WORLD_HEIGHT: 210,
      CREATURE_POPULATION_FLOOR: 0,
      CREATURE_POPULATION_CEILING: 24,
      PARTICLE_POPULATION_FLOOR: 380,
      PARTICLE_POPULATION_CEILING: 1500,
      PARTICLES_PER_SECOND: 36,
      FLUID_GRID_SIZE_CONTROL: 72,
      FLUID_DIFFUSION: 0.00035,
      FLUID_VISCOSITY: 0.0007,
      AUTO_FOLLOW_CREATURE: true,
      globalNutrientMultiplier: 1.8,
      globalLightMultiplier: 1.55,
      PHOTOSYNTHESIS_EFFICIENCY: 175,
      ENERGY_PER_PARTICLE: 35,
      FLUID_CURRENT_STRENGTH_ON_BODY: 2.0,
      SOFT_BODY_PUSH_STRENGTH: 0.04,
      BODY_REPULSION_STRENGTH: 25,
      BODY_REPULSION_RADIUS_FACTOR: 3.5,
      RIGID_SPRING_STIFFNESS: 140000,
      RIGID_SPRING_DAMPING: 80,
      MAX_FLUID_VELOCITY_COMPONENT: 6.0,
      EATER_NODE_ENERGY_COST: 6.0,
      PREDATOR_NODE_ENERGY_COST: 6.0,
      SWIMMER_NODE_ENERGY_COST: 0.06,
      JET_NODE_ENERGY_COST: 0.22,
      ATTRACTOR_NODE_ENERGY_COST: 0.45,
      REPULSOR_NODE_ENERGY_COST: 0.45,
      REPRO_RESOURCE_MIN_NUTRIENT: 0.28,
      REPRO_RESOURCE_MIN_LIGHT: 0.18,
      REPRO_LOCAL_DENSITY_RADIUS: 260,
      REPRO_FERTILITY_LOCAL_SOFT_NEIGHBORS: 4,
      REPRO_FERTILITY_LOCAL_HARD_NEIGHBORS: 10,
      OFFSPRING_INITIAL_ENERGY_SHARE: 0.4,
      REPRODUCTION_ADDITIONAL_COST_FACTOR: 0.05,
      FAILED_REPRODUCTION_COOLDOWN_TICKS: 30,
      INITIAL_REPRODUCTION_COOLDOWN_GENE_MIN: 12,
      INITIAL_REPRODUCTION_COOLDOWN_GENE_MAX: 48
    },
    nodeConfig: {
      world: { width: 240, height: 160 },
      creatures: 6,
      creatureFloor: 0,
      creatureCeiling: 18,
      particles: 260,
      particleCeiling: 900,
      particlesPerSecond: 22,
      dt: 1/12,
      steps: 3600,
      events: [],
      stepBehavior: {
        allowReproduction: true,
        maintainCreatureFloor: false,
        maintainParticleFloor: true,
        applyEmitters: false,
        creatureSpawnMargin: 18
      },
      configOverrides: {
        globalNutrientMultiplier: 1.8,
        globalLightMultiplier: 1.55,
        PHOTOSYNTHESIS_EFFICIENCY: 175,
        ENERGY_PER_PARTICLE: 35,
        FLUID_CURRENT_STRENGTH_ON_BODY: 2.0,
        SOFT_BODY_PUSH_STRENGTH: 0.04,
        BODY_REPULSION_STRENGTH: 25,
        BODY_REPULSION_RADIUS_FACTOR: 3.5,
        RIGID_SPRING_STIFFNESS: 140000,
        RIGID_SPRING_DAMPING: 80,
        MAX_FLUID_VELOCITY_COMPONENT: 6.0,
        EATER_NODE_ENERGY_COST: 6.0,
        PREDATOR_NODE_ENERGY_COST: 6.0,
        SWIMMER_NODE_ENERGY_COST: 0.06,
        JET_NODE_ENERGY_COST: 0.22,
        ATTRACTOR_NODE_ENERGY_COST: 0.45,
        REPULSOR_NODE_ENERGY_COST: 0.45,
        REPRO_RESOURCE_MIN_NUTRIENT: 0.28,
        REPRO_RESOURCE_MIN_LIGHT: 0.18,
        REPRO_LOCAL_DENSITY_RADIUS: 260,
        REPRO_FERTILITY_LOCAL_SOFT_NEIGHBORS: 4,
        REPRO_FERTILITY_LOCAL_HARD_NEIGHBORS: 10,
        OFFSPRING_INITIAL_ENERGY_SHARE: 0.4,
        REPRODUCTION_ADDITIONAL_COST_FACTOR: 0.05,
        FAILED_REPRODUCTION_COOLDOWN_TICKS: 30,
        INITIAL_REPRODUCTION_COOLDOWN_GENE_MIN: 12,
        INITIAL_REPRODUCTION_COOLDOWN_GENE_MAX: 48,
        KILL_ON_OUT_OF_BOUNDS: false
      }
    }
  },
  micro_repro_turbulent: {
    name: 'micro_repro_turbulent',
    description: 'Micro reproduction scenario with moderate fluid forcing and no creature floor refill',
    browserConfig: {
      WORLD_WIDTH: 320,
      WORLD_HEIGHT: 210,
      CREATURE_POPULATION_FLOOR: 0,
      CREATURE_POPULATION_CEILING: 30,
      PARTICLE_POPULATION_FLOOR: 460,
      PARTICLE_POPULATION_CEILING: 1900,
      PARTICLES_PER_SECOND: 45,
      FLUID_GRID_SIZE_CONTROL: 80,
      FLUID_DIFFUSION: 0.0004,
      FLUID_VISCOSITY: 0.0008,
      AUTO_FOLLOW_CREATURE: true,
      globalNutrientMultiplier: 1.6,
      globalLightMultiplier: 1.4,
      PHOTOSYNTHESIS_EFFICIENCY: 150,
      ENERGY_PER_PARTICLE: 32,
      FLUID_CURRENT_STRENGTH_ON_BODY: 2.4,
      SOFT_BODY_PUSH_STRENGTH: 0.05,
      BODY_REPULSION_STRENGTH: 30,
      BODY_REPULSION_RADIUS_FACTOR: 3.75,
      RIGID_SPRING_STIFFNESS: 160000,
      RIGID_SPRING_DAMPING: 90,
      MAX_FLUID_VELOCITY_COMPONENT: 7.0,
      EATER_NODE_ENERGY_COST: 6.75,
      PREDATOR_NODE_ENERGY_COST: 6.75,
      SWIMMER_NODE_ENERGY_COST: 0.07,
      JET_NODE_ENERGY_COST: 0.25,
      ATTRACTOR_NODE_ENERGY_COST: 0.6,
      REPULSOR_NODE_ENERGY_COST: 0.6,
      REPRO_RESOURCE_MIN_NUTRIENT: 0.32,
      REPRO_RESOURCE_MIN_LIGHT: 0.22,
      REPRO_LOCAL_DENSITY_RADIUS: 300,
      REPRO_FERTILITY_LOCAL_SOFT_NEIGHBORS: 5,
      REPRO_FERTILITY_LOCAL_HARD_NEIGHBORS: 12,
      OFFSPRING_INITIAL_ENERGY_SHARE: 0.38,
      REPRODUCTION_ADDITIONAL_COST_FACTOR: 0.07,
      FAILED_REPRODUCTION_COOLDOWN_TICKS: 35,
      INITIAL_REPRODUCTION_COOLDOWN_GENE_MIN: 16,
      INITIAL_REPRODUCTION_COOLDOWN_GENE_MAX: 72,
      KILL_ON_OUT_OF_BOUNDS: false
    },
    nodeConfig: {
      world: { width: 240, height: 160 },
      creatures: 8,
      creatureFloor: 0,
      creatureCeiling: 24,
      particles: 320,
      particleCeiling: 1100,
      particlesPerSecond: 28,
      dt: 1/12,
      steps: 4200,
      events: [
        { tick: 360, kind: 'velocityKick', amount: 0.45 },
        { tick: 1200, kind: 'energyDrain', amount: 4 }
      ],
      stepBehavior: {
        allowReproduction: true,
        maintainCreatureFloor: false,
        maintainParticleFloor: true,
        applyEmitters: true,
        creatureSpawnMargin: 18
      },
      configOverrides: {
        globalNutrientMultiplier: 1.6,
        globalLightMultiplier: 1.4,
        PHOTOSYNTHESIS_EFFICIENCY: 150,
        ENERGY_PER_PARTICLE: 32,
        FLUID_CURRENT_STRENGTH_ON_BODY: 2.4,
        SOFT_BODY_PUSH_STRENGTH: 0.05,
        BODY_REPULSION_STRENGTH: 30,
        BODY_REPULSION_RADIUS_FACTOR: 3.75,
        RIGID_SPRING_STIFFNESS: 160000,
        RIGID_SPRING_DAMPING: 90,
        MAX_FLUID_VELOCITY_COMPONENT: 7.0,
        EATER_NODE_ENERGY_COST: 6.75,
        PREDATOR_NODE_ENERGY_COST: 6.75,
        SWIMMER_NODE_ENERGY_COST: 0.07,
        JET_NODE_ENERGY_COST: 0.25,
        ATTRACTOR_NODE_ENERGY_COST: 0.6,
        REPULSOR_NODE_ENERGY_COST: 0.6,
        REPRO_RESOURCE_MIN_NUTRIENT: 0.32,
        REPRO_RESOURCE_MIN_LIGHT: 0.22,
        REPRO_LOCAL_DENSITY_RADIUS: 300,
        REPRO_FERTILITY_LOCAL_SOFT_NEIGHBORS: 5,
        REPRO_FERTILITY_LOCAL_HARD_NEIGHBORS: 12,
        OFFSPRING_INITIAL_ENERGY_SHARE: 0.38,
        REPRODUCTION_ADDITIONAL_COST_FACTOR: 0.07,
        FAILED_REPRODUCTION_COOLDOWN_TICKS: 35,
        INITIAL_REPRODUCTION_COOLDOWN_GENE_MIN: 16,
        INITIAL_REPRODUCTION_COOLDOWN_GENE_MAX: 72,
        KILL_ON_OUT_OF_BOUNDS: false
      }
    }
  },
  micro_single_100: {
    name: 'micro_single_100',
    description: 'Tiny 100x100 world with one creature for render debugging',
    browserConfig: {
      WORLD_WIDTH: 100,
      WORLD_HEIGHT: 100,
      CREATURE_POPULATION_FLOOR: 1,
      CREATURE_POPULATION_CEILING: 16,
      PARTICLE_POPULATION_FLOOR: 40,
      PARTICLE_POPULATION_CEILING: 120,
      PARTICLES_PER_SECOND: 4,
      FLUID_GRID_SIZE_CONTROL: 48,
      AUTO_FOLLOW_CREATURE: true
    },
    nodeConfig: {
      world: { width: 100, height: 100 }, creatures: 1, creatureFloor: 1, creatureCeiling: 16, particles: 40, particleCeiling: 80, dt: 1/3, steps: 240,
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
      world: { width: 100, height: 100 }, creatures: 0, creatureCeiling: 0, particles: 0, particleCeiling: 0, dt: 1/3, steps: 120,
      events: []
    }
  },
  micro_one_creature_100: {
    name: 'micro_one_creature_100',
    description: '100x100 with exactly 1 creature, no particles, no fluid forcing',
    browserConfig: {
      WORLD_WIDTH: 100,
      WORLD_HEIGHT: 100,
      CREATURE_POPULATION_FLOOR: 1,
      CREATURE_POPULATION_CEILING: 16,
      PARTICLE_POPULATION_FLOOR: 0,
      PARTICLE_POPULATION_CEILING: 0,
      PARTICLES_PER_SECOND: 0,
      FLUID_GRID_SIZE_CONTROL: 32,
      FLUID_DIFFUSION: 0,
      FLUID_VISCOSITY: 0,
      FLUID_FADE_RATE: 0,
      EMITTER_STRENGTH: 0,
      FLUID_CURRENT_STRENGTH_ON_BODY: 0,
      AUTO_FOLLOW_CREATURE: true
    },
    nodeConfig: {
      world: { width: 100, height: 100 }, creatures: 1, creatureFloor: 1, creatureCeiling: 16, particles: 0, particleCeiling: 0, dt: 1/3, steps: 120,
      events: []
    }
  }
};

const STABLE_DEFAULTS = {
  FORCE_ALL_SPRINGS_RIGID: true,
  RIGID_CONSTRAINT_PROJECTION_ENABLED: true,
  RIGID_CONSTRAINT_PROJECTION_ITERATIONS: 8,
  RIGID_CONSTRAINT_MAX_RELATIVE_ERROR: 0.001,
  EDGE_LENGTH_HARD_CAP_ENABLED: true,
  EDGE_LENGTH_HARD_CAP_FACTOR: 6,
  SPRING_OVERSTRETCH_KILL_ENABLED: false,
  PARTICLE_POPULATION_FLOOR: 0,
  PARTICLE_POPULATION_CEILING: 0,
  PARTICLES_PER_SECOND: 0
};

for (const def of Object.values(scenarioDefs)) {
  def.browserConfig = {
    ...(def.browserConfig || {}),
    ...STABLE_DEFAULTS
  };

  if (def.nodeConfig) {
    def.nodeConfig.particles = 0;
    def.nodeConfig.particleFloor = 0;
    def.nodeConfig.particleCeiling = 0;
    def.nodeConfig.particlesPerSecond = 0;

    def.nodeConfig.configOverrides = {
      ...(def.nodeConfig.configOverrides || {}),
      ...STABLE_DEFAULTS
    };
  }
}

export function getScenarioDef(name = 'baseline') {
  return scenarioDefs[name] || scenarioDefs.baseline;
}
