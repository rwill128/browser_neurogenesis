import test from 'node:test';
import assert from 'node:assert/strict';

import {
  captureConfigRuntimeSnapshot,
  applyConfigRuntimeSnapshot,
  saveWorldStateSnapshot,
  loadWorldStateSnapshot
} from '../../js/engine/worldPersistence.mjs';

class FakeSpring {
  constructor(p1, p2, stiffness = 0.8, dampingFactor = 0.2, restLength = 6, isRigid = false) {
    this.p1 = p1;
    this.p2 = p2;
    this.restLength = restLength;
    this.stiffness = stiffness;
    this.dampingFactor = dampingFactor;
    this.isRigid = isRigid;
  }
}

class FakeParticle {
  constructor(x, y, fluidField) {
    this.pos = { x, y };
    this.vel = { x: 0.5, y: -0.25 };
    this.life = 0.9;
    this.fluidField = fluidField;
    this.kind = 'fake-particle';
  }
}

class FakeFluidField {
  constructor() {
    this.scaleX = 1;
    this.scaleY = 1;
  }

  setViscosityField() {
    // noop for tests
  }
}

class FakeSoftBody {
  constructor(id, x, y, blueprint = null) {
    this.id = id;
    this.isUnstable = false;
    this.creatureEnergy = 42;
    this.stiffness = 0.8;
    this.springDamping = 0.2;
    this.customSerializable = 'kept';

    this.blueprintPoints = blueprint?.blueprintPoints
      ? blueprint.blueprintPoints.map((p) => ({ ...p }))
      : [
          { relX: 0, relY: 0, mass: 1, radius: 3, nodeType: 1 },
          { relX: 6, relY: 0, mass: 1, radius: 4, nodeType: 2 }
        ];

    this.blueprintSprings = blueprint?.blueprintSprings
      ? blueprint.blueprintSprings.map((s) => ({ ...s }))
      : [
          { p1Index: 0, p2Index: 1, restLength: 6, stiffness: 0.8, damping: 0.2, isRigid: false }
        ];

    this.massPoints = this.blueprintPoints.map((point) => {
      const relX = Number(point.relX ?? point.x ?? 0);
      const relY = Number(point.relY ?? point.y ?? 0);
      const px = x + relX;
      const py = y + relY;
      return {
        pos: { x: px, y: py },
        prevPos: { x: px, y: py },
        force: { x: 0, y: 0 },
        mass: point.mass,
        invMass: point.mass ? 1 / point.mass : 0,
        radius: point.radius,
        nodeType: point.nodeType,
        sensedFluidVelocity: { x: 0, y: 0 },
        jetData: { thrust: 0 }
      };
    });

    this.springs = this.blueprintSprings.map((spring) => {
      const p1Index = Number(spring.p1Index ?? spring.a ?? 0);
      const p2Index = Number(spring.p2Index ?? spring.b ?? 0);
      return new FakeSpring(
        this.massPoints[p1Index],
        this.massPoints[p2Index],
        spring.stiffness,
        Number(spring.damping ?? spring.dampingFactor ?? 0.2),
        spring.restLength,
        spring.isRigid
      );
    });

    this.primaryEyePoint = this.massPoints[1] || null;
  }

  exportBlueprint() {
    return {
      version: 1,
      blueprintPoints: this.blueprintPoints.map((p) => ({ ...p })),
      blueprintSprings: this.blueprintSprings.map((s) => ({ ...s }))
    };
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
}

function createRuntime(overrides = {}) {
  return {
    GRID_COLS: 4,
    GRID_ROWS: 4,
    GRID_CELL_SIZE: 10,
    selectedInspectBody: null,
    selectedInspectPoint: null,
    selectedInspectPointIndex: -1,
    currentEmitterPreview: { ignored: true },
    emitterDragStartCell: { ignored: true },
    IMPORTED_CREATURE_DATA: { ignored: true },
    marker: 'runtime-marker',
    isAnySoftBodyUnstable: false,
    ...overrides
  };
}

function createWorldState(runtime, body) {
  const cells = runtime.GRID_COLS * runtime.GRID_ROWS;
  const particles = [new FakeParticle(15, 18, null)];
  return {
    nextSoftBodyId: 2,
    nutrientField: Float32Array.from({ length: cells }, (_, i) => i + 0.5),
    lightField: Float32Array.from({ length: cells }, (_, i) => 10 + i),
    viscosityField: Float32Array.from({ length: cells }, () => 1),
    mutationStats: { branch: 1 },
    globalEnergyGains: { photosynthesis: 2 },
    globalEnergyCosts: { baseNodes: 3 },
    fluidField: null,
    particles,
    softBodyPopulation: [body],
    spatialGrid: Array.from({ length: cells }, () => [])
  };
}

test('capture/apply config runtime snapshots skip excluded and unknown keys', () => {
  const runtime = createRuntime({ alpha: 10, beta: { nested: true } });
  const snapshot = captureConfigRuntimeSnapshot(runtime);

  assert.equal(snapshot.alpha, 10);
  assert.deepEqual(snapshot.beta, { nested: true });
  assert.equal('selectedInspectBody' in snapshot, false);
  assert.equal('IMPORTED_CREATURE_DATA' in snapshot, false);

  const target = createRuntime({ alpha: 0, beta: {}, gamma: 'keep' });
  applyConfigRuntimeSnapshot(target, {
    alpha: 22,
    beta: { nested: false },
    gamma: 'updated',
    doesNotExist: 123,
    selectedInspectPointIndex: 999
  });

  assert.equal(target.alpha, 22);
  assert.deepEqual(target.beta, { nested: false });
  assert.equal(target.gamma, 'updated');
  assert.equal(target.selectedInspectPointIndex, 999);
  assert.equal(target.doesNotExist, undefined);
});

test('save/load world snapshot round-trips body, particles, and selection', () => {
  const runtime = createRuntime();
  const body = new FakeSoftBody(1, 12, 14);
  body.massPoints[0].sensedFluidVelocity = { x: 2, y: 3 };
  body.massPoints[0].jetData = { thrust: 7 };

  const worldState = createWorldState(runtime, body);

  runtime.selectedInspectBody = body;
  runtime.selectedInspectPoint = body.massPoints[1];
  runtime.selectedInspectPointIndex = 1;

  const snapshot = saveWorldStateSnapshot({
    worldState,
    configOrViews: runtime,
    meta: { source: 'unit-test' }
  });

  const runtimeReloaded = createRuntime();
  const worldReloaded = {
    nextSoftBodyId: 0,
    nutrientField: new Float32Array(0),
    lightField: new Float32Array(0),
    viscosityField: new Float32Array(0),
    mutationStats: {},
    globalEnergyGains: {},
    globalEnergyCosts: {},
    fluidField: null,
    particles: [],
    softBodyPopulation: [],
    spatialGrid: []
  };

  const loaded = loadWorldStateSnapshot(snapshot, {
    worldState: worldReloaded,
    configOrViews: runtimeReloaded,
    classes: {
      SoftBodyClass: FakeSoftBody,
      ParticleClass: FakeParticle,
      SpringClass: FakeSpring,
      FluidFieldClass: FakeFluidField
    }
  });

  assert.equal(loaded.meta.source, 'unit-test');
  assert.equal(worldReloaded.softBodyPopulation.length, 1);
  assert.equal(worldReloaded.particles.length, 1);

  const restored = worldReloaded.softBodyPopulation[0];
  assert.equal(restored.id, 1);
  assert.equal(restored.massPoints.length, 2);
  assert.equal(restored.massPoints[1].pos.x, 18);
  assert.equal(restored.massPoints[1].pos.y, 14);
  assert.equal(restored.primaryEyePoint, restored.massPoints[1]);
  assert.equal(restored.customSerializable, 'kept');
  assert.equal(restored.nutrientField, worldReloaded.nutrientField);
  assert.equal(restored.lightField, worldReloaded.lightField);
  assert.equal(restored.particles, worldReloaded.particles);

  assert.equal(runtimeReloaded.selectedInspectBody.id, 1);
  assert.equal(runtimeReloaded.selectedInspectPointIndex, 1);
  assert.equal(runtimeReloaded.selectedInspectPoint, runtimeReloaded.selectedInspectBody.massPoints[1]);

  const occupiedCells = worldReloaded.spatialGrid.filter((cell) => cell.length > 0).length;
  assert.ok(occupiedCells > 0);
});

test('loadWorldStateSnapshot tolerates stale blueprint point counts by rebuilding from snapshot points', () => {
  const runtime = createRuntime();
  const body = new FakeSoftBody(7, 40, 44);
  const worldState = createWorldState(runtime, body);

  const snapshot = saveWorldStateSnapshot({
    worldState,
    configOrViews: runtime,
    meta: { source: 'stale-blueprint-test' }
  });

  // Simulate an older/bad save where blueprint shape no longer matches runtime point state.
  snapshot.world.softBodies[0].blueprint.blueprintPoints = snapshot.world.softBodies[0].blueprint.blueprintPoints.slice(0, 1);
  snapshot.world.softBodies[0].blueprint.blueprintSprings = [];

  const runtimeReloaded = createRuntime();
  const worldReloaded = {
    nextSoftBodyId: 0,
    nutrientField: new Float32Array(0),
    lightField: new Float32Array(0),
    viscosityField: new Float32Array(0),
    mutationStats: {},
    globalEnergyGains: {},
    globalEnergyCosts: {},
    fluidField: null,
    particles: [],
    softBodyPopulation: [],
    spatialGrid: []
  };

  loadWorldStateSnapshot(snapshot, {
    worldState: worldReloaded,
    configOrViews: runtimeReloaded,
    classes: {
      SoftBodyClass: FakeSoftBody,
      ParticleClass: FakeParticle,
      SpringClass: FakeSpring,
      FluidFieldClass: FakeFluidField
    }
  });

  assert.equal(worldReloaded.softBodyPopulation.length, 1);
  assert.equal(worldReloaded.softBodyPopulation[0].massPoints.length, snapshot.world.softBodies[0].massPoints.length);
});
