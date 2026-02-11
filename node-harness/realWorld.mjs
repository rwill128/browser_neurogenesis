import config from '../js/config.js';
import { SoftBody } from '../js/classes/SoftBody.js';
import { Particle } from '../js/classes/Particle.js';
import { FluidField } from '../js/classes/FluidField.js';
import { createEnvironmentFields } from '../js/engine/environmentFields.js';
import { syncRuntimeState } from '../js/engine/runtimeState.js';
import { NodeType, MovementType, EyeTargetType } from '../js/classes/constants.js';
import { createSeededRandom, withRandom } from './seededRandomScope.mjs';
import { stepWorld } from '../js/engine/stepWorld.mjs';
import { createWorldState } from '../js/engine/worldState.mjs';

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

    this.worldState = createWorldState({
      spatialGrid: this._createSpatialGrid(),
      softBodyPopulation: [],
      particles: [],
      nutrientField: null,
      lightField: null,
      viscosityField: null,
      nextSoftBodyId: 0
    });

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

    withRandom(this.rand, () => {
      const fieldSize = Math.round(config.FLUID_GRID_SIZE_CONTROL);
      const envFields = createEnvironmentFields({ size: fieldSize, random: this.rand });
      this.worldState.nutrientField = envFields.nutrientField;
      this.worldState.lightField = envFields.lightField;
      this.worldState.viscosityField = envFields.viscosityField;
      this.worldState.fluidField.setViscosityField(this.worldState.viscosityField);

      for (let i = 0; i < scenario.particles; i++) {
        this.worldState.particles.push(new Particle(this.rand() * config.WORLD_WIDTH, this.rand() * config.WORLD_HEIGHT, this.worldState.fluidField));
      }

      for (let i = 0; i < scenario.creatures; i++) {
        const margin = 10;
        const x = margin + this.rand() * Math.max(1, (config.WORLD_WIDTH - margin * 2));
        const y = margin + this.rand() * Math.max(1, (config.WORLD_HEIGHT - margin * 2));
        const body = new SoftBody(this.worldState.nextSoftBodyId++, x, y, null);
        body.setNutrientField(this.worldState.nutrientField);
        body.setLightField(this.worldState.lightField);
        body.setParticles(this.worldState.particles);
        body.setSpatialGrid(this.worldState.spatialGrid);
        this.worldState.softBodyPopulation.push(body);
      }
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

  _createSpatialGrid() {
    const total = Math.max(1, config.GRID_COLS * config.GRID_ROWS);
    const grid = new Array(total);
    for (let i = 0; i < total; i++) grid[i] = [];
    return grid;
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
            p.vel.x += (this.rand() - 0.5) * (ev.amount || 1);
            p.vel.y += (this.rand() - 0.5) * (ev.amount || 1);
          }
        }
      }
      this.eventLog.push({ tick: this.tick, ...ev });
    }
  }

  step(dt) {
    this.tick += 1;
    this.time += dt;

    withRandom(this.rand, () => {
      this._applyEvents();

      stepWorld(this.worldState, dt, {
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
      creatures,
      sampleCreatures: creatures.slice(0, 5)
    };
  }
}
