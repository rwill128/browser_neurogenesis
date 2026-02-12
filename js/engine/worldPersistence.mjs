/**
 * Runtime snapshot persistence for the shared real-engine world model.
 *
 * The browser and node harness both use this module for round-trippable save/load
 * so topology, fluid state, particles, and selection context can be restored.
 */

import { resolveConfigViews } from './configViews.mjs';
import { rebuildSpatialGrid } from './stepWorld.mjs';
import { captureRngSnapshot, applyRngSnapshot } from './rngState.mjs';

const CONFIG_EXCLUDE_KEYS = new Set([
  'selectedInspectBody',
  'selectedInspectPoint',
  'currentEmitterPreview',
  'emitterDragStartCell',
  'IMPORTED_CREATURE_DATA'
]);

const BODY_EXCLUDE_KEYS = new Set([
  'massPoints',
  'springs',
  'particles',
  'spatialGrid',
  'nutrientField',
  'lightField',
  'brain',
  'primaryEyePoint',
  'blueprintPoints',
  'blueprintSprings',
  'blueprintRadius',
  '_tempVec1',
  '_tempVec2',
  '_tempDiffPos',
  '_tempDirection',
  '_tempRelVel',
  '_tempP1Vel',
  '_tempP2Vel',
  '_tempSpringForceVec',
  '_tempDampingForceVec',
  '_tempTotalForceVec'
]);

/**
 * JSON-safe deep clone utility for serializable runtime state.
 */
function deepClone(value) {
  if (value === null || value === undefined) return value;
  if (typeof value !== 'object') return value;
  return JSON.parse(JSON.stringify(value));
}

function isSerializableValue(value) {
  if (value === null || value === undefined) return true;
  const t = typeof value;
  if (t === 'string' || t === 'number' || t === 'boolean') return true;
  if (Array.isArray(value)) return value.every(isSerializableValue);
  if (t === 'object') {
    return Object.values(value).every(isSerializableValue);
  }
  return false;
}

/**
 * Capture own enumerable properties that can safely survive JSON serialization.
 */
function captureSerializableOwnProps(source, excludeKeys = new Set()) {
  const out = {};
  for (const [key, value] of Object.entries(source || {})) {
    if (excludeKeys.has(key)) continue;
    if (typeof value === 'function') continue;
    if (!isSerializableValue(value)) continue;
    out[key] = deepClone(value);
  }
  return out;
}

function captureVector(vecLike) {
  if (!vecLike) return { x: 0, y: 0 };
  return {
    x: Number(vecLike.x) || 0,
    y: Number(vecLike.y) || 0
  };
}

function assignVector(target, snapshot) {
  const x = Number(snapshot?.x) || 0;
  const y = Number(snapshot?.y) || 0;

  if (!target || typeof target !== 'object') {
    return { x, y };
  }

  target.x = x;
  target.y = y;
  return target;
}

function toFloatArray(values) {
  if (!Array.isArray(values) && !(values instanceof Float32Array)) {
    return new Float32Array(0);
  }
  return Float32Array.from(values);
}

function overwriteFloatArray(targetArray, values) {
  targetArray.fill(0);
  if (!Array.isArray(values) && !(values instanceof Float32Array)) return;

  const max = Math.min(targetArray.length, values.length);
  for (let i = 0; i < max; i++) {
    targetArray[i] = Number(values[i]) || 0;
  }
}

function serializeMassPoint(point) {
  const state = captureSerializableOwnProps(point, new Set(['pos', 'prevPos', 'force']));
  return {
    pos: captureVector(point.pos),
    prevPos: captureVector(point.prevPos),
    force: captureVector(point.force),
    state
  };
}

function restoreMassPoint(point, pointSnapshot) {
  assignVector(point.pos, pointSnapshot.pos);
  assignVector(point.prevPos, pointSnapshot.prevPos);
  point.force = assignVector(point.force, pointSnapshot.force);

  for (const [key, value] of Object.entries(pointSnapshot.state || {})) {
    if (key === 'sensedFluidVelocity') {
      point.sensedFluidVelocity = assignVector(point.sensedFluidVelocity, value);
      continue;
    }
    if (key === 'jetData' && value && typeof value === 'object') {
      point.jetData = { ...point.jetData, ...deepClone(value) };
      continue;
    }
    point[key] = deepClone(value);
  }

  point.invMass = point.mass !== 0 ? 1 / point.mass : 0;
}

function serializeSpring(spring, pointsIndex) {
  return {
    p1Index: pointsIndex.get(spring.p1),
    p2Index: pointsIndex.get(spring.p2),
    restLength: Number(spring.restLength) || 0,
    stiffness: Number(spring.stiffness) || 0,
    dampingFactor: Number(spring.dampingFactor) || 0,
    isRigid: Boolean(spring.isRigid)
  };
}

/**
 * Build a constructor-safe blueprint from the current phenotype.
 *
 * This is used only for reconstructing the current body shape on load,
 * not as the heritable reproductive blueprint.
 */
function buildPhenotypeBlueprint(body, pointsIndex) {
  const points = Array.isArray(body?.massPoints) ? body.massPoints : [];
  const center = typeof body?.getAveragePosition === 'function'
    ? body.getAveragePosition()
    : { x: 0, y: 0 };

  const blueprintPoints = points.map((p) => ({
    relX: Number(p?.pos?.x || 0) - Number(center?.x || 0),
    relY: Number(p?.pos?.y || 0) - Number(center?.y || 0),
    radius: Number(p?.radius || 0),
    mass: Number(p?.mass || 0),
    nodeType: Number.isFinite(Number(p?.nodeType)) ? Number(p.nodeType) : 1,
    movementType: Number.isFinite(Number(p?.movementType)) ? Number(p.movementType) : 2,
    dyeColor: Array.isArray(p?.dyeColor) ? deepClone(p.dyeColor) : [200, 50, 50],
    canBeGrabber: Boolean(p?.canBeGrabber),
    eyeTargetType: Number.isFinite(Number(p?.eyeTargetType)) ? Number(p.eyeTargetType) : 0,
    neuronDataBlueprint: p?.neuronData
      ? { hiddenLayerSize: Number(p.neuronData.hiddenLayerSize) || null }
      : null
  }));

  const blueprintSprings = (body?.springs || [])
    .map((s) => ({
      p1Index: pointsIndex.get(s.p1),
      p2Index: pointsIndex.get(s.p2),
      restLength: Number(s.restLength) || 0,
      isRigid: Boolean(s.isRigid),
      stiffness: Number(s.stiffness) || 0,
      damping: Number(s.dampingFactor) || 0
    }))
    .filter((s) => Number.isInteger(s.p1Index) && Number.isInteger(s.p2Index) && s.p1Index !== s.p2Index);

  const exported = typeof body?.exportBlueprint === 'function' ? deepClone(body.exportBlueprint()) : {};

  return {
    ...exported,
    version: Number(exported?.version) || 2,
    blueprintPoints,
    blueprintSprings
  };
}

/**
 * Convert a snapshot payload back into a usable blueprint.
 */
function buildBlueprintFromBodySnapshot(bodySnapshot) {
  const points = Array.isArray(bodySnapshot?.massPoints) ? bodySnapshot.massPoints : [];
  const springs = Array.isArray(bodySnapshot?.springs) ? bodySnapshot.springs : [];

  if (points.length === 0) {
    return deepClone(bodySnapshot?.blueprint || null);
  }

  let sumX = 0;
  let sumY = 0;
  for (const p of points) {
    sumX += Number(p?.pos?.x) || 0;
    sumY += Number(p?.pos?.y) || 0;
  }
  const cx = sumX / points.length;
  const cy = sumY / points.length;

  const blueprintPoints = points.map((p) => {
    const state = p?.state || {};
    return {
      relX: (Number(p?.pos?.x) || 0) - cx,
      relY: (Number(p?.pos?.y) || 0) - cy,
      radius: Number(state.radius || 0),
      mass: Number(state.mass || 0),
      nodeType: Number.isFinite(Number(state.nodeType)) ? Number(state.nodeType) : 1,
      movementType: Number.isFinite(Number(state.movementType)) ? Number(state.movementType) : 2,
      dyeColor: Array.isArray(state.dyeColor) ? deepClone(state.dyeColor) : [200, 50, 50],
      canBeGrabber: Boolean(state.canBeGrabber),
      eyeTargetType: Number.isFinite(Number(state.eyeTargetType)) ? Number(state.eyeTargetType) : 0,
      neuronDataBlueprint: state?.neuronData
        ? { hiddenLayerSize: Number(state.neuronData.hiddenLayerSize) || null }
        : null
    };
  });

  const blueprintSprings = springs
    .map((s) => ({
      p1Index: Number(s?.p1Index),
      p2Index: Number(s?.p2Index),
      restLength: Number(s?.restLength) || 0,
      isRigid: Boolean(s?.isRigid),
      stiffness: Number(s?.stiffness) || 0,
      damping: Number(s?.dampingFactor) || 0
    }))
    .filter((s) => Number.isInteger(s.p1Index) && Number.isInteger(s.p2Index) && s.p1Index !== s.p2Index);

  const base = deepClone(bodySnapshot?.phenotypeBlueprint || bodySnapshot?.blueprint || {});
  return {
    ...base,
    version: Number(base?.version) || 2,
    blueprintPoints,
    blueprintSprings
  };
}

/**
 * Convert a SoftBody instance into a blueprint+state snapshot payload.
 */
function serializeSoftBody(body) {
  const pointsIndex = new Map();
  body.massPoints.forEach((p, idx) => pointsIndex.set(p, idx));

  const reproductiveBlueprint = typeof body.exportBlueprint === 'function'
    ? deepClone(body.exportBlueprint())
    : {
        version: 2,
        blueprintPoints: deepClone(body.blueprintPoints || []),
        blueprintSprings: deepClone(body.blueprintSprings || [])
      };

  return {
    id: body.id,
    blueprint: reproductiveBlueprint,
    phenotypeBlueprint: buildPhenotypeBlueprint(body, pointsIndex),
    state: captureSerializableOwnProps(body, BODY_EXCLUDE_KEYS),
    primaryEyePointIndex: body.primaryEyePoint ? pointsIndex.get(body.primaryEyePoint) : -1,
    massPoints: body.massPoints.map(serializeMassPoint),
    springs: body.springs.map((spring) => serializeSpring(spring, pointsIndex))
  };
}

/**
 * Rehydrate one SoftBody from snapshot data and reconnect world references.
 */
function restoreSoftBody(bodySnapshot, { SoftBodyClass, SpringClass, worldState }) {
  const firstPoint = bodySnapshot.massPoints?.[0];
  const initialX = Number(firstPoint?.pos?.x) || 0;
  const initialY = Number(firstPoint?.pos?.y) || 0;

  const restorePointCount = (bodySnapshot.massPoints || []).length;

  const reproductiveBlueprint = deepClone(bodySnapshot.blueprint || null);
  const constructorBlueprint = bodySnapshot.phenotypeBlueprint || bodySnapshot.blueprint || null;

  let body = new SoftBodyClass(
    Number(bodySnapshot.id) || 0,
    initialX,
    initialY,
    constructorBlueprint || null,
    Boolean(constructorBlueprint)
  );

  // Legacy save compatibility: stale blueprint shapes may not match runtime points.
  if (restorePointCount !== body.massPoints.length) {
    const rebuiltBlueprint = buildBlueprintFromBodySnapshot(bodySnapshot);
    body = new SoftBodyClass(
      Number(bodySnapshot.id) || 0,
      initialX,
      initialY,
      rebuiltBlueprint || null,
      Boolean(rebuiltBlueprint)
    );
  }

  for (const [key, value] of Object.entries(bodySnapshot.state || {})) {
    body[key] = deepClone(value);
  }

  if (restorePointCount !== body.massPoints.length) {
    throw new Error(
      `SoftBody ${bodySnapshot.id} restore failed: mass point count mismatch ` +
      `(${body.massPoints.length} != ${restorePointCount})`
    );
  }

  for (let i = 0; i < body.massPoints.length; i++) {
    restoreMassPoint(body.massPoints[i], bodySnapshot.massPoints[i]);
  }

  body.springs = [];
  for (const springSnapshot of bodySnapshot.springs || []) {
    const p1 = body.massPoints[springSnapshot.p1Index];
    const p2 = body.massPoints[springSnapshot.p2Index];
    if (!p1 || !p2 || p1 === p2) continue;

    const spring = new SpringClass(
      p1,
      p2,
      Number(springSnapshot.stiffness) || body.stiffness || 0,
      Number(springSnapshot.dampingFactor) || body.springDamping || 0,
      Number(springSnapshot.restLength) || null,
      Boolean(springSnapshot.isRigid)
    );

    if (!spring.isRigid) {
      spring.stiffness = Number(springSnapshot.stiffness) || spring.stiffness;
      spring.dampingFactor = Number(springSnapshot.dampingFactor) || spring.dampingFactor;
    }

    body.springs.push(spring);
  }

  const eyeIdx = Number(bodySnapshot.primaryEyePointIndex);
  body.primaryEyePoint = Number.isInteger(eyeIdx) && eyeIdx >= 0 && eyeIdx < body.massPoints.length
    ? body.massPoints[eyeIdx]
    : null;

  // Preserve heritable reproductive blueprint separately from current phenotype.
  if (reproductiveBlueprint && Array.isArray(reproductiveBlueprint.blueprintPoints) && Array.isArray(reproductiveBlueprint.blueprintSprings)) {
    body.blueprintPoints = deepClone(reproductiveBlueprint.blueprintPoints);
    body.blueprintSprings = deepClone(reproductiveBlueprint.blueprintSprings);
    if (typeof body._calculateBlueprintRadius === 'function') {
      body._calculateBlueprintRadius();
    }
  }

  body.setNutrientField(worldState.nutrientField);
  body.setLightField(worldState.lightField);
  body.setParticles(worldState.particles);
  body.setSpatialGrid(worldState.spatialGrid);

  return body;
}

function serializeParticle(particle) {
  const state = captureSerializableOwnProps(particle, new Set(['pos', 'vel', 'fluidField']));
  return {
    pos: captureVector(particle.pos),
    vel: captureVector(particle.vel),
    state
  };
}

function restoreParticle(particleSnapshot, { ParticleClass, fluidField }) {
  const p = new ParticleClass(
    Number(particleSnapshot.pos?.x) || 0,
    Number(particleSnapshot.pos?.y) || 0,
    fluidField
  );

  assignVector(p.pos, particleSnapshot.pos);
  assignVector(p.vel, particleSnapshot.vel);
  for (const [key, value] of Object.entries(particleSnapshot.state || {})) {
    if (key === 'fluidField') continue;
    p[key] = deepClone(value);
  }
  p.fluidField = fluidField;
  return p;
}

/**
 * Serialize CPU fluid field buffers for deterministic restore.
 */
function serializeFluidField(fluidField) {
  if (!fluidField) return null;

  const requiredArrays = ['densityR', 'densityG', 'densityB', 'densityR0', 'densityG0', 'densityB0', 'Vx', 'Vy', 'Vx0', 'Vy0'];
  for (const key of requiredArrays) {
    if (!(fluidField[key] instanceof Float32Array)) {
      throw new Error(`Fluid serialization requires CPU FluidField arrays; missing ${key}`);
    }
  }

  return {
    size: Number(fluidField.size) || 0,
    dt: Number(fluidField.dt) || 0,
    diffusion: Number(fluidField.diffusion) || 0,
    viscosity: Number(fluidField.viscosity) || 0,
    scaleX: Number(fluidField.scaleX) || 1,
    scaleY: Number(fluidField.scaleY) || 1,
    useWrapping: Boolean(fluidField.useWrapping),
    maxVelComponent: Number(fluidField.maxVelComponent) || 0,
    iterations: Number(fluidField.iterations) || 0,
    densityR: Array.from(fluidField.densityR),
    densityG: Array.from(fluidField.densityG),
    densityB: Array.from(fluidField.densityB),
    densityR0: Array.from(fluidField.densityR0),
    densityG0: Array.from(fluidField.densityG0),
    densityB0: Array.from(fluidField.densityB0),
    Vx: Array.from(fluidField.Vx),
    Vy: Array.from(fluidField.Vy),
    Vx0: Array.from(fluidField.Vx0),
    Vy0: Array.from(fluidField.Vy0)
  };
}

/**
 * Restore CPU fluid field buffers from persisted arrays.
 */
function restoreFluidField(fluidSnapshot, { FluidFieldClass }) {
  if (!fluidSnapshot) return null;

  const fluidField = new FluidFieldClass(
    fluidSnapshot.size,
    fluidSnapshot.diffusion,
    fluidSnapshot.viscosity,
    fluidSnapshot.dt,
    fluidSnapshot.scaleX,
    fluidSnapshot.scaleY
  );

  fluidField.useWrapping = Boolean(fluidSnapshot.useWrapping);
  fluidField.maxVelComponent = Number(fluidSnapshot.maxVelComponent) || fluidField.maxVelComponent;
  if (Number.isFinite(fluidSnapshot.iterations)) {
    fluidField.iterations = Math.max(1, Math.floor(fluidSnapshot.iterations));
  }

  overwriteFloatArray(fluidField.densityR, fluidSnapshot.densityR);
  overwriteFloatArray(fluidField.densityG, fluidSnapshot.densityG);
  overwriteFloatArray(fluidField.densityB, fluidSnapshot.densityB);
  overwriteFloatArray(fluidField.densityR0, fluidSnapshot.densityR0);
  overwriteFloatArray(fluidField.densityG0, fluidSnapshot.densityG0);
  overwriteFloatArray(fluidField.densityB0, fluidSnapshot.densityB0);
  overwriteFloatArray(fluidField.Vx, fluidSnapshot.Vx);
  overwriteFloatArray(fluidField.Vy, fluidSnapshot.Vy);
  overwriteFloatArray(fluidField.Vx0, fluidSnapshot.Vx0);
  overwriteFloatArray(fluidField.Vy0, fluidSnapshot.Vy0);

  return fluidField;
}

/**
 * Capture JSON-safe runtime config fields used by world state.
 */
export function captureConfigRuntimeSnapshot(configOrViews) {
  const { runtime } = resolveConfigViews(configOrViews);
  return captureSerializableOwnProps(runtime, CONFIG_EXCLUDE_KEYS);
}

/**
 * Apply a previously captured runtime config snapshot to current config runtime.
 */
export function applyConfigRuntimeSnapshot(configOrViews, runtimeSnapshot = {}) {
  const { runtime } = resolveConfigViews(configOrViews);
  for (const [key, value] of Object.entries(runtimeSnapshot || {})) {
    if (!(key in runtime)) continue;
    if (CONFIG_EXCLUDE_KEYS.has(key)) continue;
    runtime[key] = deepClone(value);
  }
}

/**
 * Build a complete world snapshot payload suitable for cross-runtime restore.
 */
export function saveWorldStateSnapshot({
  worldState,
  configOrViews,
  rng = null,
  meta = {}
}) {
  const { runtime } = resolveConfigViews(configOrViews);

  const selectedBody = runtime.selectedInspectBody;
  const selectedBodyId = selectedBody ? selectedBody.id : null;
  const selectedPointIndex = selectedBody && runtime.selectedInspectPoint
    ? selectedBody.massPoints.indexOf(runtime.selectedInspectPoint)
    : -1;

  return {
    version: 1,
    enginePath: 'real',
    savedAt: new Date().toISOString(),
    meta: deepClone(meta),
    rng: captureRngSnapshot(rng),
    configRuntime: captureConfigRuntimeSnapshot(runtime),
    selection: {
      selectedBodyId,
      selectedPointIndex: Number.isInteger(selectedPointIndex) ? selectedPointIndex : -1
    },
    world: {
      nextSoftBodyId: Number(worldState.nextSoftBodyId) || 0,
      simulationStep: Number(worldState.simulationStep) || 0,
      nutrientField: Array.from(worldState.nutrientField || []),
      lightField: Array.from(worldState.lightField || []),
      viscosityField: Array.from(worldState.viscosityField || []),
      mutationStats: deepClone(worldState.mutationStats || {}),
      globalEnergyGains: deepClone(worldState.globalEnergyGains || {}),
      globalEnergyCosts: deepClone(worldState.globalEnergyCosts || {}),
      instabilityTelemetry: deepClone(worldState.instabilityTelemetry || {}),
      fluidField: serializeFluidField(worldState.fluidField),
      particles: (worldState.particles || []).map(serializeParticle),
      softBodies: (worldState.softBodyPopulation || []).map(serializeSoftBody)
    }
  };
}

/**
 * Restore a previously saved world snapshot into mutable world state/config.
 */
export function loadWorldStateSnapshot(snapshot, {
  worldState,
  configOrViews,
  classes,
  rng = null
}) {
  if (!snapshot || snapshot.enginePath !== 'real') {
    throw new Error('Unsupported world snapshot format. Expected enginePath="real".');
  }

  const {
    SoftBodyClass,
    ParticleClass,
    SpringClass,
    FluidFieldClass
  } = classes || {};

  if (!SoftBodyClass || !ParticleClass || !SpringClass || !FluidFieldClass) {
    throw new Error('loadWorldStateSnapshot requires SoftBodyClass, ParticleClass, SpringClass, and FluidFieldClass');
  }

  const { runtime } = resolveConfigViews(configOrViews);
  applyConfigRuntimeSnapshot(runtime, snapshot.configRuntime || {});

  const fluidField = restoreFluidField(snapshot.world?.fluidField, { FluidFieldClass });
  worldState.fluidField = fluidField;

  worldState.nutrientField = toFloatArray(snapshot.world?.nutrientField);
  worldState.lightField = toFloatArray(snapshot.world?.lightField);
  worldState.viscosityField = toFloatArray(snapshot.world?.viscosityField);

  if (worldState.fluidField) {
    worldState.fluidField.setViscosityField(worldState.viscosityField);
  }

  worldState.particles = (snapshot.world?.particles || []).map((p) =>
    restoreParticle(p, { ParticleClass, fluidField: worldState.fluidField })
  );

  worldState.softBodyPopulation = [];
  for (const bodySnapshot of snapshot.world?.softBodies || []) {
    const body = restoreSoftBody(bodySnapshot, { SoftBodyClass, SpringClass, worldState });
    worldState.softBodyPopulation.push(body);
  }

  worldState.nextSoftBodyId = Number(snapshot.world?.nextSoftBodyId) || worldState.softBodyPopulation.length;
  worldState.simulationStep = Number(snapshot.world?.simulationStep) || worldState.simulationStep || 0;
  worldState.mutationStats = deepClone(snapshot.world?.mutationStats || worldState.mutationStats || {});
  worldState.globalEnergyGains = deepClone(snapshot.world?.globalEnergyGains || worldState.globalEnergyGains || {});
  worldState.globalEnergyCosts = deepClone(snapshot.world?.globalEnergyCosts || worldState.globalEnergyCosts || {});
  worldState.instabilityTelemetry = deepClone(snapshot.world?.instabilityTelemetry || worldState.instabilityTelemetry || {
    totalRemoved: 0,
    totalPhysicsRemoved: 0,
    totalNonPhysicsRemoved: 0,
    totalUnknownRemoved: 0,
    removedByReason: {},
    recentDeaths: [],
    maxRecentDeaths: 1000,
    lastDeathSeq: 0
  });

  const totalCells = Math.max(1, runtime.GRID_COLS * runtime.GRID_ROWS);
  worldState.spatialGrid = new Array(totalCells);
  for (let i = 0; i < totalCells; i++) worldState.spatialGrid[i] = [];
  rebuildSpatialGrid(worldState, runtime);

  const selection = snapshot.selection || {};
  const selectedBodyId = selection.selectedBodyId;
  if (selectedBodyId === null || selectedBodyId === undefined) {
    runtime.selectedInspectBody = null;
    runtime.selectedInspectPoint = null;
    runtime.selectedInspectPointIndex = -1;
  } else {
    const selectedBody = worldState.softBodyPopulation.find((b) => b.id === selectedBodyId) || null;
    const pointIndex = Number(selection.selectedPointIndex);
    runtime.selectedInspectBody = selectedBody;
    runtime.selectedInspectPointIndex = selectedBody && Number.isInteger(pointIndex) ? pointIndex : -1;
    runtime.selectedInspectPoint =
      selectedBody && runtime.selectedInspectPointIndex >= 0 && runtime.selectedInspectPointIndex < selectedBody.massPoints.length
        ? selectedBody.massPoints[runtime.selectedInspectPointIndex]
        : null;
  }

  runtime.isAnySoftBodyUnstable = worldState.softBodyPopulation.some((b) => b.isUnstable);
  applyRngSnapshot(rng, snapshot.rng);

  return {
    meta: deepClone(snapshot.meta || {}),
    savedAt: snapshot.savedAt || null
  };
}
