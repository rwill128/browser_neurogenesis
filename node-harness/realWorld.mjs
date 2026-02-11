import config from '../js/config.js';
import { SoftBody } from '../js/classes/SoftBody.js';
import { Particle } from '../js/classes/Particle.js';
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

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
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
      spawnMargin: 10,
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
    config.CREATURE_POPULATION_FLOOR = this.config.creatures;
    config.CREATURE_POPULATION_CEILING = this.config.creatures;
    config.PARTICLE_POPULATION_FLOOR = this.config.particles;
    config.PARTICLE_POPULATION_CEILING = this.config.particles;
    config.PARTICLES_PER_SECOND = 0;
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

    stepWorld(this.worldState, dt, {
      configViews: this.configViews,
      config,
      rng: this.rand,
      SoftBodyClass: SoftBody,
      ParticleClass: Particle,
      allowReproduction: false,
      maintainCreatureFloor: false,
      maintainParticleFloor: false,
      applyEmitters: false,
      applySelectedPointPush: false
    });

    this._syncAliasesFromWorldState();
  }

  snapshot() {
    const creatures = this.softBodyPopulation.map((b) => {
      const center = b.getAveragePosition();
      const pointIndex = new Map();
      b.massPoints.forEach((p, idx) => pointIndex.set(p, idx));

      return {
        id: b.id,
        energy: Number((b.creatureEnergy || 0).toFixed(2)),
        center: { x: Number(center.x.toFixed(2)), y: Number(center.y.toFixed(2)) },
        nodeTypeCounts: summarizeNodeTypes(b.massPoints),
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
          isDesignatedEye: Boolean(p.isDesignatedEye)
        })),
        springs: b.springs
          .map((s) => ({
            a: pointIndex.get(s.p1),
            b: pointIndex.get(s.p2),
            isRigid: Boolean(s.isRigid)
          }))
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
      fluid,
      creatures,
      sampleCreatures: creatures.slice(0, 5)
    };
  }
}
