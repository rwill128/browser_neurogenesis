import config from '../js/config.js';
import { SoftBody } from '../js/classes/SoftBody.js';
import { Particle } from '../js/classes/Particle.js';
import { FluidField } from '../js/classes/FluidField.js';
import { createEnvironmentFields } from '../js/engine/environmentFields.js';
import { syncRuntimeState } from '../js/engine/runtimeState.js';
import { createSeededRandom, withRandom } from './seededRandomScope.mjs';

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
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
    this.spatialGrid = this._createSpatialGrid();
    this.softBodyPopulation = [];
    this.particles = [];
    this.nutrientField = null;
    this.lightField = null;
    this.viscosityField = null;

    this.fluidField = new FluidField(
      config.FLUID_GRID_SIZE_CONTROL,
      config.FLUID_DIFFUSION,
      config.FLUID_VISCOSITY,
      scenario.dt,
      config.WORLD_WIDTH / config.FLUID_GRID_SIZE_CONTROL,
      config.WORLD_HEIGHT / config.FLUID_GRID_SIZE_CONTROL
    );

    syncRuntimeState({
      fluidField: this.fluidField,
      softBodyPopulation: this.softBodyPopulation,
      mutationStats: {}
    });

    withRandom(this.rand, () => {
      const fieldSize = Math.round(config.FLUID_GRID_SIZE_CONTROL);
      const envFields = createEnvironmentFields({ size: fieldSize, random: this.rand });
      this.nutrientField = envFields.nutrientField;
      this.lightField = envFields.lightField;
      this.viscosityField = envFields.viscosityField;
      this.fluidField.setViscosityField(this.viscosityField);

      for (let i = 0; i < scenario.particles; i++) {
        this.particles.push(new Particle(this.rand() * config.WORLD_WIDTH, this.rand() * config.WORLD_HEIGHT, this.fluidField));
      }

      for (let i = 0; i < scenario.creatures; i++) {
        const margin = 10;
        const x = margin + this.rand() * Math.max(1, (config.WORLD_WIDTH - margin * 2));
        const y = margin + this.rand() * Math.max(1, (config.WORLD_HEIGHT - margin * 2));
        const body = new SoftBody(this.nextSoftBodyId++, x, y, null);
        body.setNutrientField(this.nutrientField);
        body.setLightField(this.lightField);
        body.setParticles(this.particles);
        body.setSpatialGrid(this.spatialGrid);
        this.softBodyPopulation.push(body);
      }
    });
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

  _updateSpatialGrid() {
    for (let i = 0; i < this.spatialGrid.length; i++) this.spatialGrid[i] = [];

    for (const body of this.softBodyPopulation) {
      if (body.isUnstable) continue;
      for (let i = 0; i < body.massPoints.length; i++) {
        const p = body.massPoints[i];
        const gx = Math.floor(p.pos.x / config.GRID_CELL_SIZE);
        const gy = Math.floor(p.pos.y / config.GRID_CELL_SIZE);
        const idx = gx + gy * config.GRID_COLS;
        if (idx >= 0 && idx < this.spatialGrid.length) {
          this.spatialGrid[idx].push({ type: 'softbody_point', pointRef: p, bodyRef: body, originalIndex: i });
        }
      }
    }

    for (const particle of this.particles) {
      if (particle.life <= 0) continue;
      const gx = Math.floor(particle.pos.x / config.GRID_CELL_SIZE);
      const gy = Math.floor(particle.pos.y / config.GRID_CELL_SIZE);
      const idx = gx + gy * config.GRID_COLS;
      if (idx >= 0 && idx < this.spatialGrid.length) {
        this.spatialGrid[idx].push({ type: 'particle', particleRef: particle });
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
      this._updateSpatialGrid();

      this.fluidField.dt = dt;
      this.fluidField.step();

      for (let i = this.softBodyPopulation.length - 1; i >= 0; i--) {
        const body = this.softBodyPopulation[i];
        if (!body.isUnstable) body.updateSelf(dt, this.fluidField);
        if (body.isUnstable) this.softBodyPopulation.splice(i, 1);
      }

      for (let i = this.particles.length - 1; i >= 0; i--) {
        const p = this.particles[i];
        p.update(dt);
        if (p.life <= 0) this.particles.splice(i, 1);
      }
    });
  }

  snapshot() {
    const creatures = this.softBodyPopulation.map((b) => {
      const center = b.getAveragePosition();
      return {
        id: b.id,
        energy: Number((b.creatureEnergy || 0).toFixed(2)),
        center: { x: Number(center.x.toFixed(2)), y: Number(center.y.toFixed(2)) },
        vertices: b.massPoints.map((p) => ({
          x: Number(clamp(p.pos.x, 0, config.WORLD_WIDTH).toFixed(2)),
          y: Number(clamp(p.pos.y, 0, config.WORLD_HEIGHT).toFixed(2))
        }))
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
