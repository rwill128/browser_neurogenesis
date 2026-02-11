import { withRandomSource } from './randomScope.mjs';

function randomInRange(rng, min, max) {
  return min + rng() * (max - min);
}

function updateSpatialGrid(state, config) {
  const { spatialGrid, softBodyPopulation, particles } = state;
  if (!spatialGrid) return;

  for (let i = 0; i < spatialGrid.length; i++) {
    spatialGrid[i] = [];
  }

  for (const body of softBodyPopulation) {
    if (body.isUnstable) continue;
    for (let i = 0; i < body.massPoints.length; i++) {
      const point = body.massPoints[i];
      const gx = Math.floor(point.pos.x / config.GRID_CELL_SIZE);
      const gy = Math.floor(point.pos.y / config.GRID_CELL_SIZE);
      const index = gx + gy * config.GRID_COLS;
      if (index >= 0 && index < spatialGrid.length) {
        spatialGrid[index].push({
          type: 'softbody_point',
          pointRef: point,
          bodyRef: body,
          originalIndex: i
        });
      }
    }
  }

  for (const particle of particles) {
    if (particle.life <= 0) continue;
    const gx = Math.floor(particle.pos.x / config.GRID_CELL_SIZE);
    const gy = Math.floor(particle.pos.y / config.GRID_CELL_SIZE);
    const index = gx + gy * config.GRID_COLS;
    if (index >= 0 && index < spatialGrid.length) {
      spatialGrid[index].push({
        type: 'particle',
        particleRef: particle
      });
    }
  }
}

function applyVelocityEmitters(state, config) {
  if (!state.fluidField || config.EMITTER_STRENGTH <= 0) return;
  for (const emitter of config.velocityEmitters) {
    state.fluidField.addVelocity(
      emitter.gridX,
      emitter.gridY,
      emitter.forceX * config.EMITTER_STRENGTH,
      emitter.forceY * config.EMITTER_STRENGTH
    );
  }
}

function maybeApplySelectedPointFluidPush(state, config) {
  if (!state.fluidField || !config.selectedSoftBodyPoint || !config.selectedSoftBodyPoint.point?.isFixed) return;

  const point = config.selectedSoftBodyPoint.point;
  const displacementX = point.pos.x - point.prevPos.x;
  const displacementY = point.pos.y - point.prevPos.y;
  const movementMagnitudeSq = displacementX * displacementX + displacementY * displacementY;
  const movementThresholdSq = 0.01 * 0.01;

  if (movementMagnitudeSq > movementThresholdSq) {
    const fluidGridX = Math.floor(point.pos.x / state.fluidField.scaleX);
    const fluidGridY = Math.floor(point.pos.y / state.fluidField.scaleY);

    state.fluidField.addVelocity(
      fluidGridX,
      fluidGridY,
      displacementX * config.SOFT_BODY_PUSH_STRENGTH / state.fluidField.scaleX,
      displacementY * config.SOFT_BODY_PUSH_STRENGTH / state.fluidField.scaleY
    );
    state.fluidField.addDensity(fluidGridX, fluidGridY, 60, 60, 80, 15);
  }
}

function spawnParticle(state, config, ParticleClass, rng) {
  const x = rng() * config.WORLD_WIDTH;
  const y = rng() * config.WORLD_HEIGHT;
  const particle = withRandomSource(rng, () => new ParticleClass(x, y, state.fluidField));
  state.particles.push(particle);
}

function spawnCreature(state, config, SoftBodyClass, rng, margin = 50) {
  const x = randomInRange(rng, margin, config.WORLD_WIDTH - margin);
  const y = randomInRange(rng, margin, config.WORLD_HEIGHT - margin);
  const body = withRandomSource(rng, () => new SoftBodyClass(state.nextSoftBodyId++, x, y, null));
  body.setNutrientField(state.nutrientField);
  body.setLightField(state.lightField);
  body.setParticles(state.particles);
  body.setSpatialGrid(state.spatialGrid);
  state.softBodyPopulation.push(body);
  return body;
}

function accumulateRemovedBodyEnergy(state, body) {
  if (!state.globalEnergyGains || !state.globalEnergyCosts) return;

  state.globalEnergyGains.photosynthesis += body.energyGainedFromPhotosynthesis || 0;
  state.globalEnergyGains.eating += body.energyGainedFromEating || 0;
  state.globalEnergyGains.predation += body.energyGainedFromPredation || 0;

  state.globalEnergyCosts.baseNodes += body.energyCostFromBaseNodes || 0;
  state.globalEnergyCosts.emitterNodes += body.energyCostFromEmitterNodes || 0;
  state.globalEnergyCosts.eaterNodes += body.energyCostFromEaterNodes || 0;
  state.globalEnergyCosts.predatorNodes += body.energyCostFromPredatorNodes || 0;
  state.globalEnergyCosts.neuronNodes += body.energyCostFromNeuronNodes || 0;
  state.globalEnergyCosts.swimmerNodes += body.energyCostFromSwimmerNodes || 0;
  state.globalEnergyCosts.photosyntheticNodes += body.energyCostFromPhotosyntheticNodes || 0;
  state.globalEnergyCosts.grabbingNodes += body.energyCostFromGrabbingNodes || 0;
  state.globalEnergyCosts.eyeNodes += body.energyCostFromEyeNodes || 0;
  state.globalEnergyCosts.jetNodes += body.energyCostFromJetNodes || 0;
  state.globalEnergyCosts.attractorNodes += body.energyCostFromAttractorNodes || 0;
  state.globalEnergyCosts.repulsorNodes += body.energyCostFromRepulsorNodes || 0;
}

function removeDeadParticles(state, dt, rng) {
  for (let i = state.particles.length - 1; i >= 0; i--) {
    const particle = state.particles[i];
    withRandomSource(rng, () => particle.update(dt));
    if (particle.life <= 0) {
      state.particles.splice(i, 1);
    }
  }
}

function removeUnstableBodies(state) {
  let removedCount = 0;
  for (let i = state.softBodyPopulation.length - 1; i >= 0; i--) {
    const body = state.softBodyPopulation[i];
    if (!body.isUnstable) continue;
    accumulateRemovedBodyEnergy(state, body);
    state.softBodyPopulation.splice(i, 1);
    removedCount++;
  }
  return removedCount;
}

export function stepWorld(state, dt, options = {}) {
  const {
    config,
    rng = Math.random,
    SoftBodyClass = null,
    ParticleClass = null,
    allowReproduction = true,
    maintainCreatureFloor = true,
    maintainParticleFloor = true,
    applyEmitters = true,
    applySelectedPointPush = true,
    creatureSpawnMargin = 50
  } = options;

  if (!config) {
    throw new Error('stepWorld requires options.config');
  }

  updateSpatialGrid(state, config);

  if (applyEmitters) {
    applyVelocityEmitters(state, config);
  }

  if (ParticleClass && state.fluidField) {
    if (maintainParticleFloor && state.particles.length < config.PARTICLE_POPULATION_FLOOR) {
      let particlesToSpawnToFloor = config.PARTICLE_POPULATION_FLOOR - state.particles.length;
      for (let i = 0; i < particlesToSpawnToFloor; i++) {
        if (state.particles.length >= config.PARTICLE_POPULATION_CEILING) break;
        spawnParticle(state, config, ParticleClass, rng);
      }
      config.particleEmissionDebt = 0;
    } else if (
      state.particles.length < config.PARTICLE_POPULATION_CEILING &&
      config.PARTICLES_PER_SECOND > 0
    ) {
      config.particleEmissionDebt += config.PARTICLES_PER_SECOND * dt;
      while (config.particleEmissionDebt >= 1 && state.particles.length < config.PARTICLE_POPULATION_CEILING) {
        spawnParticle(state, config, ParticleClass, rng);
        config.particleEmissionDebt -= 1;
      }
    }
  }

  if (applySelectedPointPush) {
    maybeApplySelectedPointFluidPush(state, config);
  }

  if (state.fluidField) {
    state.fluidField.dt = dt;
    state.fluidField.step();
  }

  const canCreaturesReproduceGlobally = allowReproduction && state.softBodyPopulation.length < config.CREATURE_POPULATION_CEILING;
  const newOffspring = [];
  let currentAnyUnstable = false;

  for (let i = state.softBodyPopulation.length - 1; i >= 0; i--) {
    const body = state.softBodyPopulation[i];
    if (body.isUnstable) continue;

    withRandomSource(rng, () => body.updateSelf(dt, state.fluidField));
    if (body.isUnstable) {
      currentAnyUnstable = true;
      continue;
    }

    if (
      allowReproduction &&
      body.creatureEnergy >= body.reproductionEnergyThreshold &&
      body.canReproduce &&
      canCreaturesReproduceGlobally &&
      body.failedReproductionCooldown <= 0
    ) {
      const offspring = withRandomSource(rng, () => body.reproduce());
      if (offspring && offspring.length) newOffspring.push(...offspring);
    }
  }

  if (newOffspring.length) {
    state.softBodyPopulation.push(...newOffspring);
  }

  removeDeadParticles(state, dt, rng);

  if (currentAnyUnstable && !config.isAnySoftBodyUnstable) {
    config.isAnySoftBodyUnstable = true;
  } else if (!currentAnyUnstable && config.isAnySoftBodyUnstable && !state.softBodyPopulation.some((b) => b.isUnstable)) {
    config.isAnySoftBodyUnstable = false;
  }

  const removedCount = removeUnstableBodies(state);

  if (maintainCreatureFloor && SoftBodyClass) {
    const neededToMaintainFloor = config.CREATURE_POPULATION_FLOOR - state.softBodyPopulation.length;
    if (neededToMaintainFloor > 0) {
      for (let i = 0; i < neededToMaintainFloor; i++) {
        if (state.softBodyPopulation.length >= config.CREATURE_POPULATION_CEILING) break;
        spawnCreature(state, config, SoftBodyClass, rng, creatureSpawnMargin);
      }
    }
  }

  return {
    removedCount,
    currentAnyUnstable,
    populations: {
      creatures: state.softBodyPopulation.length,
      particles: state.particles.length
    }
  };
}

export function rebuildSpatialGrid(state, config) {
  updateSpatialGrid(state, config);
}
