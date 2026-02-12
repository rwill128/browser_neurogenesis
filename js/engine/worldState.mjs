function createDefaultEnergyGains() {
  return {
    photosynthesis: 0,
    eating: 0,
    predation: 0
  };
}

function createDefaultInstabilityTelemetry() {
  return {
    totalRemoved: 0,
    totalPhysicsRemoved: 0,
    totalNonPhysicsRemoved: 0,
    totalUnknownRemoved: 0,
    removedByReason: {},
    removedByPhysicsKind: {},
    // Recent detailed removal records (including physiology + hereditary blueprint snapshots).
    recentDeaths: [],
    maxRecentDeaths: 1000,
    lastDeathSeq: 0
  };
}

function createDefaultEnergyCosts() {
  return {
    baseNodes: 0,
    emitterNodes: 0,
    eaterNodes: 0,
    predatorNodes: 0,
    neuronNodes: 0,
    swimmerNodes: 0,
    photosyntheticNodes: 0,
    grabbingNodes: 0,
    eyeNodes: 0,
    jetNodes: 0,
    attractorNodes: 0,
    repulsorNodes: 0
  };
}

export function createWorldState(initial = {}) {
  return {
    spatialGrid: null,
    softBodyPopulation: [],
    fluidField: null,
    particles: [],
    nextSoftBodyId: 0,
    nutrientField: null,
    lightField: null,
    viscosityField: null,
    mutationStats: {},
    globalEnergyGains: createDefaultEnergyGains(),
    globalEnergyCosts: createDefaultEnergyCosts(),
    simulationStep: 0,
    instabilityTelemetry: createDefaultInstabilityTelemetry(),
    ...initial
  };
}

export function resetPopulationState(worldState) {
  worldState.softBodyPopulation = [];
  worldState.nextSoftBodyId = 0;
}

export function resetParticleState(worldState) {
  worldState.particles = [];
}
