import config from '../js/config.js';
import { SoftBody } from '../js/classes/SoftBody.js';
import { Particle } from '../js/classes/Particle.js';
import { Spring } from '../js/classes/Spring.js';
import { FluidField } from '../js/classes/FluidField.js';
import { syncRuntimeState } from '../js/engine/runtimeState.js';
import { NodeType, MovementType, EyeTargetType } from '../js/classes/constants.js';
import { createSeededRandom } from './seededRandomScope.mjs';
import { stepWorld } from '../js/engine/stepWorld.mjs';
import { createWorldState } from '../js/engine/worldState.mjs';
import { collectFluidSnapshot } from './fluidSnapshot.mjs';
import {
  initializeSpatialGrid as initializeSharedSpatialGrid,
  initializeEnvironmentMaps as initializeSharedEnvironmentMaps,
  initializeParticles as initializeSharedParticles,
  initializePopulation as initializeSharedPopulation
} from '../js/engine/initWorld.mjs';
import { createConfigViews } from '../js/engine/configViews.mjs';
import { saveWorldStateSnapshot, loadWorldStateSnapshot } from '../js/engine/worldPersistence.mjs';

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function round(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  const p = 10 ** digits;
  return Math.round(n * p) / p;
}

function invertEnum(obj) {
  return Object.fromEntries(Object.entries(obj).map(([k, v]) => [v, k]));
}

const nodeTypeNameById = invertEnum(NodeType);
const movementTypeNameById = invertEnum(MovementType);
const eyeTargetTypeNameById = invertEnum(EyeTargetType);

function summarizeNodeTypes(points) {
  const counts = {};
  for (const p of points) {
    const key = nodeTypeNameById[p.nodeType] || `UNKNOWN_${p.nodeType}`;
    counts[key] = (counts[key] || 0) + 1;
  }
  return counts;
}

function summarizePopulationNodeTypes(softBodies) {
  const counts = {};
  let totalNodes = 0;

  for (const body of softBodies || []) {
    for (const p of body?.massPoints || []) {
      const key = nodeTypeNameById[p.nodeType] || `UNKNOWN_${p.nodeType}`;
      counts[key] = (counts[key] || 0) + 1;
      totalNodes += 1;
    }
  }

  const ratios = {};
  for (const [k, v] of Object.entries(counts)) {
    ratios[k] = totalNodes > 0 ? round(v / totalNodes, 6) : 0;
  }

  return { counts, ratios, totalNodes };
}

function normalizeInstabilityTelemetry(raw) {
  const t = raw && typeof raw === 'object' ? raw : {};

  const removedByReason = { ...(t.removedByReason && typeof t.removedByReason === 'object' ? t.removedByReason : {}) };
  const removedByPhysicsKind = { ...(t.removedByPhysicsKind && typeof t.removedByPhysicsKind === 'object' ? t.removedByPhysicsKind : {}) };
  const removedByBirthOrigin = { ...(t.removedByBirthOrigin && typeof t.removedByBirthOrigin === 'object' ? t.removedByBirthOrigin : {}) };
  const removedByLifecycleStage = { ...(t.removedByLifecycleStage && typeof t.removedByLifecycleStage === 'object' ? t.removedByLifecycleStage : {}) };

  let sumReasons = 0;
  let derivedPhysicsFromReasons = 0;
  for (const [reason, countRaw] of Object.entries(removedByReason)) {
    const count = Math.max(0, Number(countRaw) || 0);
    sumReasons += count;
    if (String(reason).startsWith('physics_')) {
      derivedPhysicsFromReasons += count;
    }
  }
  const derivedNonPhysicsFromReasons = Math.max(0, sumReasons - derivedPhysicsFromReasons);

  let totalPhysicsRemoved = Math.max(0, Number(t.totalPhysicsRemoved) || 0, derivedPhysicsFromReasons);
  let totalNonPhysicsRemoved = Math.max(0, Number(t.totalNonPhysicsRemoved) || 0, derivedNonPhysicsFromReasons);
  let totalUnknownRemoved = Math.max(0, Number(t.totalUnknownRemoved) || 0);

  let totalRemoved = Math.max(
    0,
    Number(t.totalRemoved) || 0,
    sumReasons,
    totalPhysicsRemoved + totalNonPhysicsRemoved + totalUnknownRemoved
  );

  const knownClassTotal = totalPhysicsRemoved + totalNonPhysicsRemoved;
  if (knownClassTotal > totalRemoved) {
    totalRemoved = knownClassTotal;
  }
  totalUnknownRemoved = Math.max(totalUnknownRemoved, totalRemoved - knownClassTotal);

  return {
    totalRemoved,
    totalPhysicsRemoved,
    totalNonPhysicsRemoved,
    totalUnknownRemoved,
    removedByReason,
    removedByPhysicsKind,
    removedByBirthOrigin,
    removedByLifecycleStage,
    recentDeaths: Array.isArray(t.recentDeaths) ? t.recentDeaths.slice(-20) : [],
    sampledDiagnostics: Array.isArray(t.sampledDiagnostics) ? t.sampledDiagnostics.slice(-10) : []
  };
}

export class RealWorld {
  constructor(scenario, seed = 42) {
    this.config = scenario;
    this.seed = seed >>> 0;
    this.rand = createSeededRandom(this.seed);
    this.tick = 0;
    this.time = 0;
    this.nextSoftBodyId = 0;
    this.events = Array.isArray(scenario.events) ? scenario.events.slice() : [];
    this.eventLog = [];

    const stepBehavior = scenario.stepBehavior || {};
    this.stepBehavior = {
      allowReproduction: Boolean(stepBehavior.allowReproduction),
      maintainCreatureFloor: Boolean(stepBehavior.maintainCreatureFloor),
      maintainParticleFloor: Boolean(stepBehavior.maintainParticleFloor),
      applyEmitters: Boolean(stepBehavior.applyEmitters),
      applySelectedPointPush: Boolean(stepBehavior.applySelectedPointPush),
      captureInstabilityTelemetry: stepBehavior.captureInstabilityTelemetry !== false,
      maxRecentInstabilityDeaths: Number.isFinite(Number(stepBehavior.maxRecentInstabilityDeaths))
        ? Number(stepBehavior.maxRecentInstabilityDeaths)
        : 5000,
      creatureSpawnMargin: Number.isFinite(Number(stepBehavior.creatureSpawnMargin))
        ? Number(stepBehavior.creatureSpawnMargin)
        : 10,
      creatureExecutionMode: typeof stepBehavior.creatureExecutionMode === 'string'
        ? stepBehavior.creatureExecutionMode
        : null,
      creatureIslandNeighborRadiusCells: Number.isFinite(Number(stepBehavior.creatureIslandNeighborRadiusCells))
        ? Math.max(0, Math.floor(Number(stepBehavior.creatureIslandNeighborRadiusCells)))
        : null,
      creatureShuffleWithinIsland: Boolean(stepBehavior.creatureShuffleWithinIsland)
    };

    this._applyScenarioConfig();
    this.configViews = createConfigViews(config);

    this.worldState = createWorldState({
      spatialGrid: null,
      softBodyPopulation: [],
      particles: [],
      nutrientField: null,
      lightField: null,
      viscosityField: null,
      nextSoftBodyId: 0
    });
    initializeSharedSpatialGrid(this.worldState, this.configViews);

    this.worldState.fluidField = new FluidField(
      config.FLUID_GRID_SIZE_CONTROL,
      config.FLUID_DIFFUSION,
      config.FLUID_VISCOSITY,
      scenario.dt,
      config.WORLD_WIDTH / config.FLUID_GRID_SIZE_CONTROL,
      config.WORLD_HEIGHT / config.FLUID_GRID_SIZE_CONTROL
    );

    this._syncAliasesFromWorldState();

    syncRuntimeState({
      fluidField: this.fluidField,
      softBodyPopulation: this.softBodyPopulation,
      mutationStats: {}
    });

    initializeSharedEnvironmentMaps(this.worldState, {
      configViews: this.configViews,
      config,
      size: Math.round(config.FLUID_GRID_SIZE_CONTROL),
      rng: this.rand
    });
    this.worldState.fluidField.setViscosityField(this.worldState.viscosityField);

    initializeSharedParticles(this.worldState, {
      configViews: this.configViews,
      config,
      ParticleClass: Particle,
      count: scenario.particles,
      rng: this.rand
    });

    initializeSharedPopulation(this.worldState, {
      configViews: this.configViews,
      config,
      SoftBodyClass: SoftBody,
      count: scenario.creatures,
      spawnMargin: this.stepBehavior.creatureSpawnMargin,
      newbornDt: scenario.dt,
      rng: this.rand
    });

    this._syncAliasesFromWorldState();
  }

  _syncAliasesFromWorldState() {
    this.spatialGrid = this.worldState.spatialGrid;
    this.softBodyPopulation = this.worldState.softBodyPopulation;
    this.particles = this.worldState.particles;
    this.nutrientField = this.worldState.nutrientField;
    this.lightField = this.worldState.lightField;
    this.viscosityField = this.worldState.viscosityField;
    this.fluidField = this.worldState.fluidField;
    this.nextSoftBodyId = this.worldState.nextSoftBodyId;
  }

  _applyScenarioConfig() {
    config.IS_HEADLESS_MODE = true;
    config.USE_GPU_FLUID_SIMULATION = false;
    config.WORLD_WIDTH = this.config.world.width;
    config.WORLD_HEIGHT = this.config.world.height;

    const creatureFloor = Number.isFinite(Number(this.config.creatureFloor))
      ? Math.max(0, Math.floor(Number(this.config.creatureFloor)))
      : Math.max(0, Math.floor(Number(this.config.creatures) || 0));
    const creatureCeiling = Number.isFinite(Number(this.config.creatureCeiling))
      ? Math.max(creatureFloor, Math.floor(Number(this.config.creatureCeiling)))
      : creatureFloor;

    const particleFloor = Number.isFinite(Number(this.config.particleFloor))
      ? Math.max(0, Math.floor(Number(this.config.particleFloor)))
      : Math.max(0, Math.floor(Number(this.config.particles) || 0));
    const particleCeiling = Number.isFinite(Number(this.config.particleCeiling))
      ? Math.max(particleFloor, Math.floor(Number(this.config.particleCeiling)))
      : particleFloor;

    config.CREATURE_POPULATION_FLOOR = creatureFloor;
    config.CREATURE_POPULATION_CEILING = creatureCeiling;
    config.PARTICLE_POPULATION_FLOOR = particleFloor;
    config.PARTICLE_POPULATION_CEILING = particleCeiling;

    // Let caller choose emission profile; default remains deterministic/no-emission.
    config.PARTICLES_PER_SECOND = Number.isFinite(Number(this.config.particlesPerSecond))
      ? Number(this.config.particlesPerSecond)
      : 0;

    const overrides = this.config?.configOverrides;
    if (overrides && typeof overrides === 'object' && !Array.isArray(overrides)) {
      for (const [key, value] of Object.entries(overrides)) {
        if (Object.prototype.hasOwnProperty.call(config, key)) {
          config[key] = value;
        }
      }
    }

    config.GRID_COLS = Math.ceil(config.WORLD_WIDTH / config.GRID_CELL_SIZE);
    config.GRID_ROWS = Math.ceil(config.WORLD_HEIGHT / config.GRID_CELL_SIZE);
  }

  /**
   * Inject a localized fluid burst (color + optional velocity) into the real fluid grid.
   * Used for diagnostic scenarios where we need visible fluid behavior checks in video output.
   */
  _applyFluidBurst(ev) {
    if (!this.fluidField) return;

    const worldX = Number.isFinite(ev.worldX)
      ? ev.worldX
      : (Number.isFinite(ev.xNorm) ? ev.xNorm * config.WORLD_WIDTH : config.WORLD_WIDTH * 0.5);
    const worldY = Number.isFinite(ev.worldY)
      ? ev.worldY
      : (Number.isFinite(ev.yNorm) ? ev.yNorm * config.WORLD_HEIGHT : config.WORLD_HEIGHT * 0.5);

    const centerGX = Math.floor(worldX / this.fluidField.scaleX);
    const centerGY = Math.floor(worldY / this.fluidField.scaleY);
    const radiusCells = Math.max(0, Math.floor(ev.radiusCells ?? 2));
    const r = Number.isFinite(ev.r) ? ev.r : 255;
    const g = Number.isFinite(ev.g) ? ev.g : 255;
    const b = Number.isFinite(ev.b) ? ev.b : 255;
    const strength = Number.isFinite(ev.strength) ? ev.strength : 220;
    const velX = Number.isFinite(ev.velX) ? ev.velX : 0;
    const velY = Number.isFinite(ev.velY) ? ev.velY : 0;

    for (let dy = -radiusCells; dy <= radiusCells; dy++) {
      for (let dx = -radiusCells; dx <= radiusCells; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > radiusCells) continue;

        const falloff = radiusCells <= 0 ? 1 : Math.max(0.15, 1 - (dist / (radiusCells + 1e-9)));
        const gx = centerGX + dx;
        const gy = centerGY + dy;
        this.fluidField.addDensity(gx, gy, r, g, b, strength * falloff);
        if (velX !== 0 || velY !== 0) {
          this.fluidField.addVelocity(gx, gy, velX * falloff, velY * falloff);
        }
      }
    }
  }

  _applyEvents() {
    for (const ev of this.events) {
      if (ev.tick !== this.tick) continue;
      if (ev.kind === 'energySpike') {
        for (const b of this.softBodyPopulation) if (!b.isUnstable) b.creatureEnergy += (ev.amount || 5);
      } else if (ev.kind === 'energyDrain') {
        for (const b of this.softBodyPopulation) if (!b.isUnstable) b.creatureEnergy -= (ev.amount || 5);
      } else if (ev.kind === 'velocityKick') {
        for (const b of this.softBodyPopulation) {
          for (const p of b.massPoints) {
            const kickX = (this.rand() - 0.5) * (ev.amount || 1);
            const kickY = (this.rand() - 0.5) * (ev.amount || 1);
            if (p.vel && Number.isFinite(p.vel.x) && Number.isFinite(p.vel.y)) {
              p.vel.x += kickX;
              p.vel.y += kickY;
            } else if (p.prevPos && p.pos) {
              // Verlet bodies encode velocity as (pos - prevPos), so shift prevPos opposite the kick.
              p.prevPos.x -= kickX;
              p.prevPos.y -= kickY;
            }
          }
        }
      } else if (ev.kind === 'fluidBurst') {
        this._applyFluidBurst(ev);
      }
      this.eventLog.push({ tick: this.tick, ...ev });
    }
  }

  step(dt) {
    this.tick += 1;
    this.time += dt;

    this._applyEvents();

    const stepResult = stepWorld(this.worldState, dt, {
      configViews: this.configViews,
      config,
      rng: this.rand,
      SoftBodyClass: SoftBody,
      ParticleClass: Particle,
      allowReproduction: this.stepBehavior.allowReproduction,
      maintainCreatureFloor: this.stepBehavior.maintainCreatureFloor,
      maintainParticleFloor: this.stepBehavior.maintainParticleFloor,
      applyEmitters: this.stepBehavior.applyEmitters,
      applySelectedPointPush: this.stepBehavior.applySelectedPointPush,
      captureInstabilityTelemetry: this.stepBehavior.captureInstabilityTelemetry,
      maxRecentInstabilityDeaths: this.stepBehavior.maxRecentInstabilityDeaths,
      creatureSpawnMargin: this.stepBehavior.creatureSpawnMargin,
      creatureExecutionMode: this.stepBehavior.creatureExecutionMode,
      creatureIslandNeighborRadiusCells: this.stepBehavior.creatureIslandNeighborRadiusCells,
      creatureShuffleWithinIsland: this.stepBehavior.creatureShuffleWithinIsland
    });

    this._syncAliasesFromWorldState();
    return stepResult;
  }

  saveStateSnapshot(meta = {}) {
    return saveWorldStateSnapshot({
      worldState: this.worldState,
      configOrViews: this.configViews,
      rng: this.rand,
      meta: {
        tick: this.tick,
        time: this.time,
        scenario: this.config?.name || null,
        source: 'node-real',
        ...meta
      }
    });
  }

  loadStateSnapshot(snapshot) {
    const loadInfo = loadWorldStateSnapshot(snapshot, {
      worldState: this.worldState,
      configOrViews: this.configViews,
      classes: {
        SoftBodyClass: SoftBody,
        ParticleClass: Particle,
        SpringClass: Spring,
        FluidFieldClass: FluidField
      },
      rng: this.rand
    });

    this.tick = Number(loadInfo?.meta?.tick) || 0;
    this.time = Number(loadInfo?.meta?.time) || 0;
    this._syncAliasesFromWorldState();

    syncRuntimeState({
      fluidField: this.fluidField,
      softBodyPopulation: this.softBodyPopulation,
      mutationStats: this.worldState.mutationStats || {}
    });

    return loadInfo;
  }

  snapshot() {
    const creatures = this.softBodyPopulation.map((b) => {
      const center = b.getAveragePosition();
      const pointIndex = new Map();
      b.massPoints.forEach((p, idx) => pointIndex.set(p, idx));

      const intervalSamples = Number(b.actuationIntervalSamples || 0);
      const intervalTotal = Number(b.actuationIntervalTotal || 0);

      return {
        id: b.id,
        energy: Number((b.creatureEnergy || 0).toFixed(2)),
        center: { x: Number(center.x.toFixed(2)), y: Number(center.y.toFixed(2)) },
        nodeTypeCounts: summarizeNodeTypes(b.massPoints),
        actuationTelemetry: {
          evaluations: Number(b.actuationEvaluations || 0),
          skips: Number(b.actuationSkips || 0),
          evaluationsByNodeType: b.actuationEvaluationsByNodeType || {},
          skipsByNodeType: b.actuationSkipsByNodeType || {},
          avgEffectiveInterval: intervalSamples > 0 ? Number((intervalTotal / intervalSamples).toFixed(3)) : 0,
          energyCostUpkeep: Number((b.energyCostFromActuationUpkeep || 0).toFixed(3)),
          energyCostEvents: Number((b.energyCostFromActuationEvents || 0).toFixed(3))
        },
        fullStats: {
          stiffness: round(typeof b.getAverageStiffness === 'function' ? b.getAverageStiffness() : b.stiffness, 4),
          damping: round(typeof b.getAverageDamping === 'function' ? b.getAverageDamping() : b.springDamping, 4),
          motorImpulseInterval: Number.isFinite(Number(b.motorImpulseInterval)) ? Number(b.motorImpulseInterval) : null,
          motorImpulseMagnitudeCap: round(b.motorImpulseMagnitudeCap, 4),
          emitterStrength: round(b.emitterStrength, 4),
          emitterDirection: {
            x: round(b.emitterDirection?.x, 4),
            y: round(b.emitterDirection?.y, 4)
          },
          numOffspring: Number.isFinite(Number(b.numOffspring)) ? Number(b.numOffspring) : null,
          offspringSpawnRadius: round(b.offspringSpawnRadius, 4),
          pointAddChance: round(b.pointAddChance, 6),
          springConnectionRadius: round(b.springConnectionRadius, 4),
          reproductionEnergyThreshold: round(b.reproductionEnergyThreshold, 4),
          currentMaxEnergy: round(b.currentMaxEnergy, 4),
          ticksSinceBirth: Number.isFinite(Number(b.ticksSinceBirth)) ? Number(b.ticksSinceBirth) : null,
          absoluteAgeTicks: Number.isFinite(Number(b.absoluteAgeTicks)) ? Number(b.absoluteAgeTicks) : null,
          birthOrigin: b.birthOrigin || 'unknown',
          parentBodyId: Number.isFinite(Number(b.parentBodyId)) ? Number(b.parentBodyId) : null,
          lineageRootId: Number.isFinite(Number(b.lineageRootId)) ? Number(b.lineageRootId) : null,
          generation: Number.isFinite(Number(b.generation)) ? Number(b.generation) : null,
          reproductionEventsCompleted: Number.isFinite(Number(b.reproductionEventsCompleted)) ? Number(b.reproductionEventsCompleted) : 0,
          ticksSinceLastReproduction: Number.isFinite(Number(b.ticksSinceLastReproduction)) ? Number(b.ticksSinceLastReproduction) : null,
          canReproduce: Boolean(b.canReproduce),
          rewardStrategy: b.rewardStrategy ?? null,
          dyePreferredHue: round(b.dyePreferredHue, 5),
          dyeHueTolerance: round(b.dyeHueTolerance, 5),
          dyeResponseGain: round(b.dyeResponseGain, 5),
          dyeResponseSign: Number(b.dyeResponseSign) < 0 ? -1 : 1,
          dyeNodeTypeAffinity: b.dyeNodeTypeAffinity || {},
          reproductionCooldownGene: Number.isFinite(Number(b.reproductionCooldownGene)) ? Number(b.reproductionCooldownGene) : null,
          effectiveReproductionCooldown: Number.isFinite(Number(b.effectiveReproductionCooldown)) ? Number(b.effectiveReproductionCooldown) : null,
          energyGains: {
            photosynthesis: round(b.energyGainedFromPhotosynthesis, 4),
            eating: round(b.energyGainedFromEating, 4),
            predation: round(b.energyGainedFromPredation, 4)
          },
          growth: {
            eventsCompleted: Number(b.growthEventsCompleted || 0),
            nodesAdded: Number(b.growthNodesAdded || 0),
            totalEnergySpent: round(b.totalGrowthEnergySpent, 4),
            suppressedByEnergy: Number(b.growthSuppressedByEnergy || 0),
            suppressedByCooldown: Number(b.growthSuppressedByCooldown || 0),
            suppressedByPopulation: Number(b.growthSuppressedByPopulation || 0),
            suppressedByDye: Number(b.growthSuppressedByDye || 0),
            suppressedByMaxPoints: Number(b.growthSuppressedByMaxPoints || 0),
            suppressedByNoCapacity: Number(b.growthSuppressedByNoCapacity || 0),
            suppressedByChanceRoll: Number(b.growthSuppressedByChanceRoll || 0),
            suppressedByPlacement: Number(b.growthSuppressedByPlacement || 0)
          },
          topology: {
            nnTopologyVersion: Number(b.nnTopologyVersion || 0),
            rlTopologyResets: Number(b.rlBufferResetsDueToTopology || 0)
          },
          motionGuards: {
            accelerationClampEvents: Number(b.motionGuardAccelerationClampEvents || 0),
            velocityClampEvents: Number(b.motionGuardVelocityClampEvents || 0),
            nonFiniteForceResets: Number(b.motionGuardNonFiniteForceResets || 0),
            positionResets: Number(b.motionGuardPositionResets || 0),
            maxAccelerationBefore: round(b.motionGuardMaxAccelerationBefore, 4),
            maxVelocityBefore: round(b.motionGuardMaxVelocityBefore, 4)
          },
          rigidConstraints: {
            projectionCorrections: Number(b.rigidConstraintProjectionCorrections || 0),
            maxRelativeErrorBefore: round(b.rigidConstraintProjectionMaxRelativeError, 6)
          },
          reproductionSuppression: {
            density: Number(b.reproductionSuppressedByDensity || 0),
            resources: Number(b.reproductionSuppressedByResources || 0),
            fertilityRoll: Number(b.reproductionSuppressedByFertilityRoll || 0),
            dye: Number(b.reproductionSuppressedByDye || 0),
            resourceDebits: Number(b.reproductionResourceDebitApplied || 0)
          },
          energyCostsByType: {
            base: round(b.energyCostFromBaseNodes, 4),
            emitter: round(b.energyCostFromEmitterNodes, 4),
            eater: round(b.energyCostFromEaterNodes, 4),
            predator: round(b.energyCostFromPredatorNodes, 4),
            neuron: round(b.energyCostFromNeuronNodes, 4),
            swimmer: round(b.energyCostFromSwimmerNodes, 4),
            photosynthetic: round(b.energyCostFromPhotosyntheticNodes, 4),
            grabbing: round(b.energyCostFromGrabbingNodes, 4),
            eye: round(b.energyCostFromEyeNodes, 4),
            jet: round(b.energyCostFromJetNodes, 4),
            attractor: round(b.energyCostFromAttractorNodes, 4),
            repulsor: round(b.energyCostFromRepulsorNodes, 4)
          }
        },
        vertices: b.massPoints.map((p, idx) => ({
          index: idx,
          x: Number(clamp(p.pos.x, 0, config.WORLD_WIDTH).toFixed(2)),
          y: Number(clamp(p.pos.y, 0, config.WORLD_HEIGHT).toFixed(2)),
          radius: Number((p.radius || 0).toFixed(2)),
          mass: Number((p.mass || 0).toFixed(4)),
          nodeType: p.nodeType,
          nodeTypeName: nodeTypeNameById[p.nodeType] || null,
          movementType: p.movementType,
          movementTypeName: movementTypeNameById[p.movementType] || null,
          eyeTargetType: p.eyeTargetType,
          eyeTargetTypeName: eyeTargetTypeNameById[p.eyeTargetType] || null,
          canBeGrabber: Boolean(p.canBeGrabber),
          isDesignatedEye: Boolean(p.isDesignatedEye),
          isGrabbing: Boolean(p.isGrabbing),
          currentExertionLevel: round(p.currentExertionLevel, 4),
          seesTarget: Boolean(p.seesTarget),
          nearestTargetMagnitude: round(p.nearestTargetMagnitude, 4),
          nearestTargetDirection: round(p.nearestTargetDirection, 4)
        })),
        springs: b.springs
          .map((s) => {
            const a = pointIndex.get(s.p1);
            const bIndex = pointIndex.get(s.p2);
            const dx = (s.p1?.pos?.x || 0) - (s.p2?.pos?.x || 0);
            const dy = (s.p1?.pos?.y || 0) - (s.p2?.pos?.y || 0);
            const currentLength = Math.hypot(dx, dy);
            const restLength = Number(s.restLength) || 0;
            const strain = restLength > 1e-8 ? ((currentLength - restLength) / restLength) : 0;
            return {
              a,
              b: bIndex,
              isRigid: Boolean(s.isRigid),
              restLength: round(restLength, 4),
              currentLength: round(currentLength, 4),
              strain: round(strain, 5),
              stiffness: round(s.stiffness, 4),
              dampingFactor: round(s.dampingFactor, 4)
            };
          })
          .filter((s) => Number.isInteger(s.a) && Number.isInteger(s.b) && s.a !== s.b)
      };
    });

    const fluid = collectFluidSnapshot({
      fluidField: this.fluidField,
      world: this.config.world,
      minDye: 2,
      minSpeed: 0.02,
      maxCells: 1200
    });

    const instabilityTelemetry = normalizeInstabilityTelemetry(this.worldState.instabilityTelemetry || {});
    const edgeLengthTelemetry = this.worldState.edgeLengthTelemetry || {};
    const nodeTypeSummary = summarizePopulationNodeTypes(this.softBodyPopulation);

    return {
      tick: this.tick,
      time: Number(this.time.toFixed(3)),
      seed: this.seed,
      recentEvents: this.eventLog.slice(-5),
      populations: {
        creatures: creatures.length,
        liveCreatures: creatures.length,
        particles: this.particles.length
      },
      worldStats: {
        globalEnergyGains: this.worldState?.globalEnergyGains || {},
        globalEnergyCosts: this.worldState?.globalEnergyCosts || {},
        computeTelemetry: this.worldState?.lastComputeTelemetry || null,
        nodeTypeCounts: nodeTypeSummary.counts,
        nodeTypeRatios: nodeTypeSummary.ratios,
        totalNodes: nodeTypeSummary.totalNodes,
        edgeLengthTelemetryLatest: edgeLengthTelemetry?.latest || null
      },
      nodeTypeCounts: nodeTypeSummary.counts,
      nodeTypeRatios: nodeTypeSummary.ratios,
      totalNodes: nodeTypeSummary.totalNodes,
      instabilityTelemetry,
      edgeLengthTelemetry: {
        enabled: edgeLengthTelemetry.enabled !== false,
        sampleEveryNSteps: Number(edgeLengthTelemetry.sampleEveryNSteps) || 0,
        samplesCollected: Number(edgeLengthTelemetry.samplesCollected) || 0,
        totalSpringSamples: Number(edgeLengthTelemetry.totalSpringSamples) || 0,
        totalHugeOutliers: Number(edgeLengthTelemetry.totalHugeOutliers) || 0,
        latest: edgeLengthTelemetry.latest || null,
        recentSamples: Array.isArray(edgeLengthTelemetry.recentSamples)
          ? edgeLengthTelemetry.recentSamples.slice(-20)
          : []
      },
      mutationStats: this.worldState?.mutationStats || {},
      fluid,
      creatures,
      sampleCreatures: creatures.slice(0, 5)
    };
  }
}
