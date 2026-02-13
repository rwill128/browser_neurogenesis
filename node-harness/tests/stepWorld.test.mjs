import test from 'node:test';
import assert from 'node:assert/strict';

import { stepWorld } from '../../js/engine/stepWorld.mjs';

class FakeFluidField {
  constructor() {
    this.scaleX = 1;
    this.scaleY = 1;
    this.dt = 0;
    this.velocityCalls = 0;
    this.densityCalls = 0;
  }

  addVelocity() {
    this.velocityCalls += 1;
  }

  addDensity() {
    this.densityCalls += 1;
  }

  step() {
    // noop
  }
}

class FakeParticle {
  constructor(x, y, fluidField) {
    this.pos = { x, y };
    this.vel = { x: 0, y: 0 };
    this.life = 1;
    this.fluidField = fluidField;
  }

  update() {
    this.life -= 0.25;
  }
}

class FakeSoftBody {
  constructor(id, x, y) {
    this.id = id;
    this.isUnstable = false;
    this.massPoints = [{ pos: { x, y }, prevPos: { x, y }, force: { x: 0, y: 0 }, isFixed: false }];
    this.creatureEnergy = 10;
    this.reproductionEnergyThreshold = 5;
    this.canReproduce = false;
    this.failedReproductionCooldown = 0;
    this.reproduceCalls = 0;
    this.reproductionSuppressedByDensity = 0;
    this.reproductionSuppressedByResources = 0;
    this.reproductionSuppressedByFertilityRoll = 0;

    this.energyGainedFromPhotosynthesis = 1;
    this.energyGainedFromEating = 2;
    this.energyGainedFromPredation = 3;
    this.energyCostFromBaseNodes = 4;
    this.energyCostFromEmitterNodes = 5;
    this.energyCostFromEaterNodes = 6;
    this.energyCostFromPredatorNodes = 7;
    this.energyCostFromNeuronNodes = 8;
    this.energyCostFromSwimmerNodes = 9;
    this.energyCostFromPhotosyntheticNodes = 10;
    this.energyCostFromGrabbingNodes = 11;
    this.energyCostFromEyeNodes = 12;
    this.energyCostFromJetNodes = 13;
    this.energyCostFromAttractorNodes = 14;
    this.energyCostFromRepulsorNodes = 15;
  }

  setNutrientField(field) {
    this.nutrientField = field;
  }

  setLightField(field) {
    this.lightField = field;
  }

  setParticles(particles) {
    this.particles = particles;
  }

  setSpatialGrid(grid) {
    this.spatialGrid = grid;
  }

  updateSelf() {
    // noop
  }

  reproduce() {
    this.reproduceCalls += 1;
    return [];
  }
}

function createRuntimeConfig(overrides = {}) {
  return {
    GRID_CELL_SIZE: 10,
    GRID_COLS: 20,
    GRID_ROWS: 20,
    WORLD_WIDTH: 200,
    WORLD_HEIGHT: 200,
    PARTICLE_POPULATION_FLOOR: 0,
    PARTICLE_POPULATION_CEILING: 100,
    PARTICLES_PER_SECOND: 0,
    particleEmissionDebt: 0,
    CREATURE_POPULATION_FLOOR: 0,
    CREATURE_POPULATION_CEILING: 100,
    velocityEmitters: [],
    EMITTER_STRENGTH: 1,
    LANDSCAPE_DYE_EMITTERS_ENABLED: false,
    LANDSCAPE_DYE_EMITTER_COUNT: 0,
    LANDSCAPE_DYE_EMITTER_STRENGTH_MIN: 8,
    LANDSCAPE_DYE_EMITTER_STRENGTH_MAX: 18,
    LANDSCAPE_DYE_EMITTER_RADIUS_CELLS: 0,
    LANDSCAPE_DYE_EMITTER_PULSE_HZ_MIN: 0.02,
    LANDSCAPE_DYE_EMITTER_PULSE_HZ_MAX: 0.09,
    EDGE_LENGTH_TELEMETRY_ENABLED: true,
    EDGE_LENGTH_TELEMETRY_SAMPLE_EVERY_N_STEPS: 10,
    EDGE_LENGTH_TELEMETRY_MODE_BIN_SIZE: 0.01,
    EDGE_LENGTH_TELEMETRY_HUGE_OUTLIER_IQR_MULTIPLIER: 3,
    EDGE_LENGTH_TELEMETRY_HISTORY_MAX_SAMPLES: 120,
    EDGE_LENGTH_TELEMETRY_MAX_RECORDED_OUTLIERS: 24,
    selectedSoftBodyPoint: null,
    SOFT_BODY_PUSH_STRENGTH: 1,
    isAnySoftBodyUnstable: false,
    ...overrides
  };
}

function createWorldState({ runtime, softBodies = [], particles = [], fluidField = null } = {}) {
  const total = runtime.GRID_COLS * runtime.GRID_ROWS;
  return {
    fluidField,
    particles,
    softBodyPopulation: softBodies,
    nextSoftBodyId: softBodies.length,
    nutrientField: new Float32Array(total),
    lightField: new Float32Array(total),
    viscosityField: new Float32Array(total),
    spatialGrid: Array.from({ length: total }, () => []),
    globalEnergyGains: { photosynthesis: 0, eating: 0, predation: 0 },
    globalEnergyCosts: {
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
    }
  };
}

test('stepWorld maintains creature/particle floors and wires spawned creatures', () => {
  const runtime = createRuntimeConfig({
    PARTICLE_POPULATION_FLOOR: 3,
    CREATURE_POPULATION_FLOOR: 2,
    PARTICLE_POPULATION_CEILING: 8,
    CREATURE_POPULATION_CEILING: 5
  });

  const state = createWorldState({
    runtime,
    softBodies: [],
    particles: [],
    fluidField: new FakeFluidField()
  });

  const result = stepWorld(state, 0.01, {
    config: runtime,
    rng: () => 0.5,
    SoftBodyClass: FakeSoftBody,
    ParticleClass: FakeParticle,
    maintainCreatureFloor: true,
    maintainParticleFloor: true
  });

  assert.equal(result.populations.creatures, 2);
  assert.equal(result.populations.particles, 3);
  assert.equal(runtime.particleEmissionDebt, 0);

  for (const body of state.softBodyPopulation) {
    assert.equal(body.nutrientField, state.nutrientField);
    assert.equal(body.lightField, state.lightField);
    assert.equal(body.particles, state.particles);
    assert.equal(body.spatialGrid, state.spatialGrid);
  }
});

test('stepWorld injects landscape dye emitters into fluid when enabled', () => {
  const runtime = createRuntimeConfig({
    LANDSCAPE_DYE_EMITTERS_ENABLED: true,
    LANDSCAPE_DYE_EMITTER_COUNT: 3,
    LANDSCAPE_DYE_EMITTER_STRENGTH_MIN: 4,
    LANDSCAPE_DYE_EMITTER_STRENGTH_MAX: 4,
    LANDSCAPE_DYE_EMITTER_RADIUS_CELLS: 0
  });

  const fluidField = new FakeFluidField();
  const state = createWorldState({
    runtime,
    softBodies: [],
    particles: [],
    fluidField
  });

  stepWorld(state, 0.01, {
    config: runtime,
    rng: () => 0.25,
    maintainCreatureFloor: false,
    maintainParticleFloor: false,
    applyEmitters: false,
    applySelectedPointPush: false
  });

  assert.equal(Array.isArray(state.landscapeDyeEmitters), true);
  assert.equal(state.landscapeDyeEmitters.length, 3);
  assert.equal(fluidField.densityCalls, 3);
});

test('stepWorld allows reproduction only when enabled and globally allowed', () => {
  const runtime = createRuntimeConfig({
    CREATURE_POPULATION_FLOOR: 0,
    CREATURE_POPULATION_CEILING: 3
  });

  const parent = new FakeSoftBody(1, 20, 20);
  parent.canReproduce = true;
  parent.reproduce = () => [new FakeSoftBody(2, 21, 21)];

  const state = createWorldState({ runtime, softBodies: [parent], particles: [] });

  const enabledResult = stepWorld(state, 0.02, {
    config: runtime,
    rng: () => 0.25,
    allowReproduction: true,
    maintainCreatureFloor: false,
    maintainParticleFloor: false
  });

  assert.equal(enabledResult.populations.creatures, 2);

  const blockedRuntime = createRuntimeConfig({ CREATURE_POPULATION_CEILING: 2 });
  const blockedParent = new FakeSoftBody(10, 30, 30);
  blockedParent.canReproduce = true;
  let blockedCalled = 0;
  blockedParent.reproduce = () => {
    blockedCalled += 1;
    return [new FakeSoftBody(11, 31, 31)];
  };

  const blockedState = createWorldState({ runtime: blockedRuntime, softBodies: [blockedParent, new FakeSoftBody(12, 32, 32)] });

  const blockedResult = stepWorld(blockedState, 0.02, {
    config: blockedRuntime,
    rng: () => 0.25,
    allowReproduction: true,
    maintainCreatureFloor: false,
    maintainParticleFloor: false
  });

  assert.equal(blockedCalled, 0);
  assert.equal(blockedResult.populations.creatures, 2);
});

test('stepWorld decrements reproduction cooldown during gated attempts', () => {
  const runtime = createRuntimeConfig({
    CREATURE_POPULATION_FLOOR: 0,
    CREATURE_POPULATION_CEILING: 8
  });

  const parent = new FakeSoftBody(21, 20, 20);
  parent.canReproduce = true;
  parent.creatureEnergy = 20;
  parent.reproductionEnergyThreshold = 5;
  parent.failedReproductionCooldown = 2;
  parent.reproduce = function reproduceWithCooldownTick() {
    this.reproduceCalls += 1;
    if (this.failedReproductionCooldown > 0) {
      this.failedReproductionCooldown -= 1;
      return [];
    }
    return [new FakeSoftBody(22, 22, 22)];
  };

  const state = createWorldState({ runtime, softBodies: [parent], particles: [] });
  const result = stepWorld(state, 0.02, {
    config: runtime,
    rng: () => 0.25,
    allowReproduction: true,
    maintainCreatureFloor: false,
    maintainParticleFloor: false
  });

  assert.equal(parent.reproduceCalls, 1);
  assert.equal(parent.failedReproductionCooldown, 1);
  assert.equal(result.reproductionTelemetry.suppressedByCooldown, 1);
  assert.equal(result.reproductionTelemetry.attemptedParents, 1);
  assert.equal(result.reproductionTelemetry.successfulBirths, 0);
});

test('stepWorld clamps per-parent births to remaining creature ceiling slots', () => {
  const runtime = createRuntimeConfig({
    CREATURE_POPULATION_FLOOR: 0,
    CREATURE_POPULATION_CEILING: 2
  });

  class MultiBirthBody extends FakeSoftBody {
    constructor(id, x, y) {
      super(id, x, y);
      this.canReproduce = true;
    }

    reproduce({ maxOffspring = null } = {}) {
      this.reproduceCalls += 1;
      const n = Number.isFinite(Number(maxOffspring)) ? Math.max(0, Math.floor(Number(maxOffspring))) : 3;
      return Array.from({ length: n }, (_, idx) => new FakeSoftBody(200 + idx, 60 + idx, 60 + idx));
    }
  }

  const parent = new MultiBirthBody(1, 20, 20);
  parent.creatureEnergy = 100;
  parent.reproductionEnergyThreshold = 5;

  const state = createWorldState({ runtime, softBodies: [parent], particles: [] });

  const result = stepWorld(state, 0.02, {
    config: runtime,
    rng: () => 0.25,
    allowReproduction: true,
    maintainCreatureFloor: false,
    maintainParticleFloor: false
  });

  assert.equal(parent.reproduceCalls, 1);
  assert.equal(result.spawnTelemetry.reproductionBirths, 1);
  assert.equal(result.populations.creatures, 2);
});

test('stepWorld reports per-step reproduction suppression reasons and successful births', () => {
  const runtime = createRuntimeConfig({
    CREATURE_POPULATION_FLOOR: 0,
    CREATURE_POPULATION_CEILING: 20
  });

  const energyBlocked = new FakeSoftBody(30, 10, 10);
  energyBlocked.canReproduce = true;
  energyBlocked.creatureEnergy = 1;
  energyBlocked.reproductionEnergyThreshold = 5;

  const cooldownBlocked = new FakeSoftBody(31, 20, 20);
  cooldownBlocked.canReproduce = true;
  cooldownBlocked.creatureEnergy = 20;
  cooldownBlocked.reproductionEnergyThreshold = 5;
  cooldownBlocked.failedReproductionCooldown = 1;
  cooldownBlocked.reproduce = function cooldownGate() {
    this.reproduceCalls += 1;
    if (this.failedReproductionCooldown > 0) {
      this.failedReproductionCooldown -= 1;
      return [];
    }
    return [];
  };

  const resourceBlocked = new FakeSoftBody(32, 30, 30);
  resourceBlocked.canReproduce = true;
  resourceBlocked.creatureEnergy = 20;
  resourceBlocked.reproductionEnergyThreshold = 5;
  resourceBlocked.reproduce = function resourceGate() {
    this.reproduceCalls += 1;
    this.reproductionSuppressedByResources += 1;
    return [];
  };

  const densityBlocked = new FakeSoftBody(33, 40, 40);
  densityBlocked.canReproduce = true;
  densityBlocked.creatureEnergy = 20;
  densityBlocked.reproductionEnergyThreshold = 5;
  densityBlocked.reproduce = function densityGate() {
    this.reproduceCalls += 1;
    this.reproductionSuppressedByDensity += 1;
    this.reproductionSuppressedByFertilityRoll += 1;
    return [];
  };

  const success = new FakeSoftBody(34, 50, 50);
  success.canReproduce = true;
  success.creatureEnergy = 20;
  success.reproductionEnergyThreshold = 5;
  success.reproduce = function successBirth() {
    this.reproduceCalls += 1;
    return [new FakeSoftBody(35, 51, 51)];
  };

  const state = createWorldState({
    runtime,
    softBodies: [energyBlocked, cooldownBlocked, resourceBlocked, densityBlocked, success],
    particles: []
  });

  const result = stepWorld(state, 0.02, {
    config: runtime,
    rng: () => 0.25,
    allowReproduction: true,
    maintainCreatureFloor: false,
    maintainParticleFloor: false
  });

  assert.equal(result.spawnTelemetry.reproductionBirths, 1);
  assert.equal(result.reproductionTelemetry.consideredBodies, 5);
  assert.equal(result.reproductionTelemetry.attemptedParents, 4);
  assert.equal(result.reproductionTelemetry.successfulParents, 1);
  assert.equal(result.reproductionTelemetry.successfulBirths, 1);
  assert.equal(result.reproductionTelemetry.attemptsWithoutBirths, 3);
  assert.equal(result.reproductionTelemetry.suppressedByEnergy, 1);
  assert.equal(result.reproductionTelemetry.suppressedByCooldown, 1);
  assert.equal(result.reproductionTelemetry.suppressedByResources, 1);
  assert.equal(result.reproductionTelemetry.suppressedByDensity, 1);
  assert.equal(result.reproductionTelemetry.suppressedByFertilityRoll, 1);
  assert.equal(result.populations.creatures, 6);
});

test('stepWorld removes unstable bodies and accumulates energy accounting', () => {
  const runtime = createRuntimeConfig({ isAnySoftBodyUnstable: true });
  const unstable = new FakeSoftBody(99, 40, 40);
  unstable.isUnstable = true;

  const state = createWorldState({ runtime, softBodies: [unstable], particles: [] });

  const result = stepWorld(state, 0.01, {
    config: runtime,
    allowReproduction: false,
    maintainCreatureFloor: false,
    maintainParticleFloor: false
  });

  assert.equal(result.removedCount, 1);
  assert.equal(state.softBodyPopulation.length, 0);
  assert.equal(state.globalEnergyGains.photosynthesis, 1);
  assert.equal(state.globalEnergyGains.eating, 2);
  assert.equal(state.globalEnergyGains.predation, 3);
  assert.equal(state.globalEnergyCosts.baseNodes, 4);
  assert.equal(state.globalEnergyCosts.repulsorNodes, 15);
});

test('stepWorld captures detailed instability telemetry for removed bodies', () => {
  const runtime = createRuntimeConfig({ isAnySoftBodyUnstable: true });
  const unstable = new FakeSoftBody(100, 40, 40);
  unstable.isUnstable = true;
  unstable.unstableReason = 'physics_spring_overstretch';
  unstable.blueprintPoints = [
    { relX: 0, relY: 0, radius: 1, mass: 1, nodeType: 1, movementType: 2 },
    { relX: 1, relY: 0, radius: 1, mass: 1, nodeType: 1, movementType: 2 }
  ];
  unstable.blueprintSprings = [{ p1Index: 0, p2Index: 1, restLength: 1, isRigid: false, stiffness: 200, damping: 1 }];
  unstable.birthOrigin = 'reproduction_offspring';
  unstable.parentBodyId = 42;
  unstable.lineageRootId = 4;
  unstable.generation = 3;
  unstable.reproductionEventsCompleted = 2;
  unstable.ticksSinceLastReproduction = 7;
  unstable.absoluteAgeTicks = 120;

  const state = createWorldState({ runtime, softBodies: [unstable], particles: [] });

  const result = stepWorld(state, 0.01, {
    config: runtime,
    allowReproduction: false,
    maintainCreatureFloor: false,
    maintainParticleFloor: false,
    captureInstabilityTelemetry: true
  });

  assert.equal(result.removedCount, 1);
  assert.equal(result.removedBodies.length, 1);
  assert.equal(result.removedBodies[0].unstableReason, 'physics_spring_overstretch');
  assert.equal(result.removedBodies[0].physicsStabilityDeath, true);
  assert.equal(result.removedBodies[0].unstablePhysicsKind, 'geometric_explosion');
  assert.equal(result.removedBodies[0].birthOrigin, 'reproduction_offspring');
  assert.equal(result.removedBodies[0].lifecycleStage, 'post_reproduction_parent');
  assert.equal(result.removedBodies[0].isPostReproductionParent, true);
  assert.equal(result.removedBodies[0].ticksSinceLastReproduction, 7);
  assert.equal(result.removedBodies[0].absoluteAgeTicks, 120);
  assert.ok(result.removedBodies[0].physiology);
  assert.ok(result.removedBodies[0].hereditaryBlueprint);

  const telemetry = state.instabilityTelemetry;
  assert.equal(telemetry.totalRemoved, 1);
  assert.equal(telemetry.totalPhysicsRemoved, 1);
  assert.equal(telemetry.removedByReason.physics_spring_overstretch, 1);
  assert.equal(telemetry.removedByPhysicsKind.geometric_explosion, 1);
  assert.equal(telemetry.removedByBirthOrigin.reproduction_offspring, 1);
  assert.equal(telemetry.removedByLifecycleStage.post_reproduction_parent, 1);
  assert.equal(Array.isArray(telemetry.recentDeaths), true);
  assert.equal(telemetry.recentDeaths.length, 1);
});

test('stepWorld separates instability removals by lifecycle stage and birth origin', () => {
  const runtime = createRuntimeConfig({ isAnySoftBodyUnstable: true });

  const floorSpawned = new FakeSoftBody(150, 20, 20);
  floorSpawned.isUnstable = true;
  floorSpawned.unstableReason = 'physics_invalid_motion';
  floorSpawned.birthOrigin = 'floor_spawn';
  floorSpawned.reproductionEventsCompleted = 0;

  const postReproductionParent = new FakeSoftBody(151, 24, 20);
  postReproductionParent.isUnstable = true;
  postReproductionParent.unstableReason = 'energy_depleted';
  postReproductionParent.birthOrigin = 'initial_population';
  postReproductionParent.reproductionEventsCompleted = 1;
  postReproductionParent.ticksSinceLastReproduction = 3;

  const state = createWorldState({ runtime, softBodies: [floorSpawned, postReproductionParent], particles: [] });

  const result = stepWorld(state, 0.01, {
    config: runtime,
    allowReproduction: false,
    maintainCreatureFloor: false,
    maintainParticleFloor: false,
    captureInstabilityTelemetry: true
  });

  assert.equal(result.removedCount, 2);
  assert.equal(state.instabilityTelemetry.removedByBirthOrigin.floor_spawn, 1);
  assert.equal(state.instabilityTelemetry.removedByLifecycleStage.floor_spawn, 1);
  assert.equal(state.instabilityTelemetry.removedByLifecycleStage.post_reproduction_parent, 1);

  const lifecycleStages = new Set(result.removedBodies.map((e) => e.lifecycleStage));
  assert.equal(lifecycleStages.has('floor_spawn'), true);
  assert.equal(lifecycleStages.has('post_reproduction_parent'), true);
});

test('stepWorld separates invalid-motion vs nan/non-finite reasons and samples diagnostics', () => {
  const runtime = createRuntimeConfig({ isAnySoftBodyUnstable: true });

  const badMotion = new FakeSoftBody(201, 30, 30);
  badMotion.isUnstable = true;
  badMotion.unstableReason = 'physics_invalid_motion';
  badMotion.unstableReasonDetails = { pointIndex: 0, displacementSq: 999 };

  const badNaN = new FakeSoftBody(202, 32, 30);
  badNaN.isUnstable = true;
  badNaN.unstableReason = 'physics_nan_position';
  badNaN.unstableReasonDetails = { pointIndex: 0, pos: { x: NaN, y: 1 } };

  const badInfinite = new FakeSoftBody(203, 34, 30);
  badInfinite.isUnstable = true;
  badInfinite.unstableReason = 'physics_non_finite_position';
  badInfinite.unstableReasonDetails = { pointIndex: 0, pos: { x: Infinity, y: 1 } };

  const state = createWorldState({ runtime, softBodies: [badMotion, badNaN, badInfinite], particles: [] });

  const originalWarn = console.warn;
  console.warn = () => {};
  let result;
  try {
    result = stepWorld(state, 0.01, {
      config: runtime,
      allowReproduction: false,
      maintainCreatureFloor: false,
      maintainParticleFloor: false,
      captureInstabilityTelemetry: true,
      instabilityDiagnosticEveryN: 1
    });
  } finally {
    console.warn = originalWarn;
  }

  assert.equal(result.removedCount, 3);
  assert.equal(state.instabilityTelemetry.removedByReason.physics_invalid_motion, 1);
  assert.equal(state.instabilityTelemetry.removedByReason.physics_nan_position, 1);
  assert.equal(state.instabilityTelemetry.removedByReason.physics_non_finite_position, 1);

  assert.equal(state.instabilityTelemetry.removedByPhysicsKind.invalid_motion, 1);
  assert.equal(state.instabilityTelemetry.removedByPhysicsKind.non_finite_numeric, 2);

  assert.equal(Array.isArray(state.instabilityTelemetry.sampledDiagnostics), true);
  assert.equal(state.instabilityTelemetry.sampledDiagnostics.length >= 3, true);
  const reasons = state.instabilityTelemetry.sampledDiagnostics.map((s) => s?.event?.unstableReason);
  assert.equal(reasons.includes('physics_invalid_motion'), true);
  assert.equal(reasons.includes('physics_nan_position'), true);
  assert.equal(reasons.includes('physics_non_finite_position'), true);
});

test('stepWorld floor-spawn applies newborn fit + spring clamping', () => {
  class SpawnedBody {
    constructor(id, x, y) {
      this.id = id;
      this.isUnstable = false;
      this.ticksSinceBirth = 0;
      this.massPoints = [
        { pos: { x: x - 80, y }, prevPos: { x: x - 80, y }, radius: 4, isFixed: false },
        { pos: { x: x + 80, y }, prevPos: { x: x + 80, y }, radius: 4, isFixed: false }
      ];
      this.springs = [
        {
          isRigid: true,
          stiffness: 500000,
          dampingFactor: 150,
          restLength: 160,
          p1: this.massPoints[0],
          p2: this.massPoints[1]
        }
      ];
    }

    setNutrientField(field) { this.nutrientField = field; }
    setLightField(field) { this.lightField = field; }
    setParticles(particlesRef) { this.particles = particlesRef; }
    setSpatialGrid(grid) { this.spatialGrid = grid; }
  }

  const runtime = createRuntimeConfig({
    WORLD_WIDTH: 100,
    WORLD_HEIGHT: 100,
    CREATURE_POPULATION_FLOOR: 1,
    CREATURE_POPULATION_CEILING: 2,
    NEWBORN_STIFFNESS_CLAMP_ENABLED: true,
    NEWBORN_STIFFNESS_WORLD_REF_DIM: 1200,
    NEWBORN_STIFFNESS_DT_REF: 1 / 30,
    NEWBORN_STIFFNESS_DT_EXPONENT: 2,
    NEWBORN_RIGID_STIFFNESS_WORLD_EXPONENT: 2,
    NEWBORN_NON_RIGID_STIFFNESS_WORLD_EXPONENT: 1,
    NEWBORN_RIGID_STIFFNESS_MIN_SCALE: 0.005,
    NEWBORN_NON_RIGID_STIFFNESS_MIN_SCALE: 0.05,
    NEWBORN_NON_RIGID_STIFFNESS_BASE_CAP: 10000,
    NEWBORN_NON_RIGID_DAMPING_BASE_CAP: 80,
    RIGID_SPRING_STIFFNESS: 500000,
    RIGID_SPRING_DAMPING: 150
  });

  const state = createWorldState({ runtime, softBodies: [], particles: [] });

  const result = stepWorld(state, 1 / 30, {
    config: runtime,
    rng: () => 0.5,
    SoftBodyClass: SpawnedBody,
    allowReproduction: false,
    maintainCreatureFloor: true,
    maintainParticleFloor: false
  });

  assert.equal(result.populations.creatures, 1);
  const spawned = state.softBodyPopulation[0];
  assert.equal(spawned.__newbornStabilityApplied, true);

  for (const p of spawned.massPoints) {
    assert.ok(p.pos.x >= p.radius);
    assert.ok(p.pos.x <= runtime.WORLD_WIDTH - p.radius);
    assert.ok(p.pos.y >= p.radius);
    assert.ok(p.pos.y <= runtime.WORLD_HEIGHT - p.radius);
  }

  // Clamp should significantly reduce rigid newborn stiffness in tiny worlds.
  assert.ok(spawned.springs[0].stiffness < runtime.RIGID_SPRING_STIFFNESS);
  assert.ok(spawned.springs[0].stiffness <= 10000);
});

test('stepWorld legacy_reverse execution mode preserves historical reverse order', () => {
  const runtime = createRuntimeConfig({
    CREATURE_POPULATION_FLOOR: 0,
    CREATURE_POPULATION_CEILING: 10,
    PARTICLE_POPULATION_FLOOR: 0,
    PARTICLE_POPULATION_CEILING: 0
  });

  const order = [];
  class OrderedBody extends FakeSoftBody {
    updateSelf() {
      order.push(this.id);
    }
  }

  const bodies = [
    new OrderedBody(1, 10, 10),
    new OrderedBody(2, 40, 40),
    new OrderedBody(3, 70, 70)
  ];

  const state = createWorldState({ runtime, softBodies: bodies, particles: [] });
  const result = stepWorld(state, 0.01, {
    config: runtime,
    rng: () => 0.5,
    allowReproduction: false,
    maintainCreatureFloor: false,
    maintainParticleFloor: false,
    creatureExecutionMode: 'legacy_reverse'
  });

  assert.deepEqual(order, [3, 2, 1]);
  assert.equal(result.computeTelemetry.mode, 'legacy_reverse');
});

test('stepWorld islands_deterministic executes serially by deterministic island order', () => {
  const runtime = createRuntimeConfig({
    CREATURE_POPULATION_FLOOR: 0,
    CREATURE_POPULATION_CEILING: 10,
    PARTICLE_POPULATION_FLOOR: 0,
    PARTICLE_POPULATION_CEILING: 0
  });

  const order = [];
  class OrderedBody extends FakeSoftBody {
    updateSelf() {
      order.push(this.id);
    }
  }

  const bodies = [
    new OrderedBody(3, 5, 5),
    new OrderedBody(1, 95, 95),
    new OrderedBody(2, 150, 150)
  ];

  const state = createWorldState({ runtime, softBodies: bodies, particles: [] });
  const result = stepWorld(state, 0.01, {
    config: runtime,
    rng: () => 0.5,
    allowReproduction: false,
    maintainCreatureFloor: false,
    maintainParticleFloor: false,
    creatureExecutionMode: 'islands_deterministic',
    creatureIslandNeighborRadiusCells: 0
  });

  assert.deepEqual(order, [1, 2, 3]);
  assert.equal(result.computeTelemetry.mode, 'islands_deterministic');
  assert.equal(result.computeTelemetry.islandCount, 3);
});

test('stepWorld islands_shuffled randomizes serial island execution order', () => {
  const runtime = createRuntimeConfig({
    CREATURE_POPULATION_FLOOR: 0,
    CREATURE_POPULATION_CEILING: 10,
    PARTICLE_POPULATION_FLOOR: 0,
    PARTICLE_POPULATION_CEILING: 0
  });

  const order = [];
  class OrderedBody extends FakeSoftBody {
    updateSelf() {
      order.push(this.id);
    }
  }

  const bodies = [
    new OrderedBody(1, 5, 5),
    new OrderedBody(2, 95, 95),
    new OrderedBody(3, 150, 150)
  ];

  const state = createWorldState({ runtime, softBodies: bodies, particles: [] });
  const result = stepWorld(state, 0.01, {
    config: runtime,
    rng: () => 0,
    allowReproduction: false,
    maintainCreatureFloor: false,
    maintainParticleFloor: false,
    creatureExecutionMode: 'islands_shuffled',
    creatureIslandNeighborRadiusCells: 0
  });

  assert.deepEqual(order, [2, 3, 1]);
  assert.equal(result.computeTelemetry.mode, 'islands_shuffled');
  assert.equal(result.computeTelemetry.shuffled, true);
});

test('stepWorld samples edge-length telemetry (mean/median/mode + huge outliers)', () => {
  const runtime = createRuntimeConfig({
    EDGE_LENGTH_TELEMETRY_ENABLED: true,
    EDGE_LENGTH_TELEMETRY_SAMPLE_EVERY_N_STEPS: 1,
    EDGE_LENGTH_TELEMETRY_MODE_BIN_SIZE: 0.01,
    EDGE_LENGTH_TELEMETRY_HUGE_OUTLIER_IQR_MULTIPLIER: 3,
    EDGE_LENGTH_TELEMETRY_HISTORY_MAX_SAMPLES: 8,
    EDGE_LENGTH_TELEMETRY_MAX_RECORDED_OUTLIERS: 4
  });

  const body = new FakeSoftBody(1, 0, 0);
  body.canReproduce = false;
  const p1 = { pos: { x: 0, y: 0 }, prevPos: { x: 0, y: 0 }, force: { x: 0, y: 0 }, movementType: 2, mass: 1 };
  const p2 = { pos: { x: 4, y: 0 }, prevPos: { x: 4, y: 0 }, force: { x: 0, y: 0 }, movementType: 2, mass: 1 };
  const p3 = { pos: { x: 8, y: 0 }, prevPos: { x: 8, y: 0 }, force: { x: 0, y: 0 }, movementType: 2, mass: 1 };
  const p4 = { pos: { x: 50, y: 0 }, prevPos: { x: 50, y: 0 }, force: { x: 0, y: 0 }, movementType: 2, mass: 1 };
  body.massPoints = [p1, p2, p3, p4];

  const makeSpring = (a, b, restLength) => ({ p1: a, p2: b, restLength, isRigid: false, stiffness: 1000, dampingFactor: 10 });
  body.springs = [
    makeSpring(p1, p2, 4),
    makeSpring(p1, p2, 4),
    makeSpring(p1, p2, 4),
    makeSpring(p2, p3, 4),
    makeSpring(p2, p3, 4),
    makeSpring(p2, p3, 4),
    makeSpring(p1, p3, 8),
    makeSpring(p1, p3, 8),
    makeSpring(p1, p4, 4)
  ];

  const state = createWorldState({
    runtime,
    softBodies: [body],
    particles: [],
    fluidField: new FakeFluidField()
  });

  stepWorld(state, 0.01, {
    config: runtime,
    rng: () => 0.5,
    SoftBodyClass: FakeSoftBody,
    ParticleClass: FakeParticle,
    allowReproduction: false,
    maintainCreatureFloor: false,
    maintainParticleFloor: false
  });

  assert.ok(state.edgeLengthTelemetry);
  assert.ok(state.edgeLengthTelemetry.latest);
  assert.equal(state.edgeLengthTelemetry.latest.springCount, 9);
  assert.ok(state.edgeLengthTelemetry.latest.meanCurrentLength > 0);
  assert.ok(state.edgeLengthTelemetry.latest.medianCurrentLength > 0);
  assert.ok(state.edgeLengthTelemetry.latest.modeCurrentLength > 0);
  assert.equal(state.edgeLengthTelemetry.latest.hugeOutlierCount >= 1, true);
  assert.equal(Array.isArray(state.edgeLengthTelemetry.latest.hugeOutliersTop), true);
  assert.equal(state.edgeLengthTelemetry.samplesCollected >= 1, true);
});
