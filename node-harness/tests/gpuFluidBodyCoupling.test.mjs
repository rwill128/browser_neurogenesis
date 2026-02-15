import test from 'node:test';
import assert from 'node:assert/strict';

import { GPUFluidField } from '../../js/gpuFluidField.js';

function makeShadowOnlyField({ size = 32, dt = 1 / 30, scaleX = 1, scaleY = 1 } = {}) {
  const f = Object.create(GPUFluidField.prototype);
  f.size = size;
  f.dt = dt;
  f.scaleX = scaleX;
  f.scaleY = scaleY;
  f.maxVelComponent = 10;
  const cells = size * size;
  f.shadowVx = new Float32Array(cells).fill(0);
  f.shadowVy = new Float32Array(cells).fill(0);
  f.shadowDensityR = new Float32Array(cells).fill(0);
  f.shadowDensityG = new Float32Array(cells).fill(0);
  f.shadowDensityB = new Float32Array(cells).fill(0);
  f.shadowVxNext = new Float32Array(cells).fill(0);
  f.shadowVyNext = new Float32Array(cells).fill(0);
  f.shadowDensityRNext = new Float32Array(cells).fill(0);
  f.shadowDensityGNext = new Float32Array(cells).fill(0);
  f.shadowDensityBNext = new Float32Array(cells).fill(0);
  f._initShadowBackCompatViews();
  f.gpuEnabled = false;
  return f;
}

test('GPU fluid shadow path exposes deterministic velocity for body coupling and decays over step()', () => {
  const fluid = makeShadowOnlyField({ size: 32, dt: 0.1, scaleX: 1, scaleY: 1 });

  fluid.addVelocity(8, 8, 2.5, -1.0);
  const before = fluid.getVelocityAtWorld(8, 8);
  assert.ok(before.vx > 2.0);
  assert.ok(before.vy < -0.5);

  fluid.step();
  const after = fluid.getVelocityAtWorld(8, 8);

  assert.ok(after.vx < before.vx, `expected vx decay (${after.vx} < ${before.vx})`);
  assert.ok(Math.abs(after.vy) < Math.abs(before.vy), `expected vy decay (${after.vy} vs ${before.vy})`);
});

test('GPU fluid shadow path stores density splats for emitter/body sampling', () => {
  const fluid = makeShadowOnlyField({ size: 32, dt: 0.1, scaleX: 1, scaleY: 1 });

  fluid.addDensity(4, 6, 200, 100, 50, 70);
  const density = fluid.getDensityAtWorld(4, 6);

  assert.ok(density[0] > 100);
  assert.ok(density[1] > 40);
  assert.ok(density[2] > 20);
});

test('GPU fluid shadow path advects momentum to neighboring cells (current transport)', () => {
  const fluid = makeShadowOnlyField({ size: 32, dt: 0.2, scaleX: 1, scaleY: 1 });

  fluid.addVelocity(10, 10, 2.0, 0.0);
  const beforeNeighbor = fluid.getVelocityAtWorld(11, 10);
  assert.ok(beforeNeighbor.vx > 0, `expected local splat radius to seed neighboring momentum, got ${beforeNeighbor.vx}`);
  assert.ok(beforeNeighbor.vx < 2.0, `neighbor seed should remain weaker than source impulse, got ${beforeNeighbor.vx}`);

  fluid.step();

  const sourceAfter = fluid.getVelocityAtWorld(10, 10);
  const neighborAfter = fluid.getVelocityAtWorld(11, 10);

  assert.ok(neighborAfter.vx > 1e-4, `expected downstream transported velocity, got ${neighborAfter.vx}`);
  assert.ok(sourceAfter.vx < 2.0, `expected source to diffuse/advect, got ${sourceAfter.vx}`);
});

test('GPU fluid world-space sampling respects scaleX/scaleY near origin for coupling queries', () => {
  const fluid = makeShadowOnlyField({ size: 64, dt: 0.1, scaleX: 10, scaleY: 10 });

  // Grid injection at (4,3) should be sampled from world (40,30).
  fluid.addVelocity(4, 3, 3.25, -1.5);

  const sampled = fluid.getVelocityAtWorld(40, 30);
  const wrongCell = fluid.getVelocityAtWorld(4, 3);

  assert.ok(sampled.vx > 3.0 && sampled.vy < -1.0, `expected world->grid mapping at scaled coord, got (${sampled.vx}, ${sampled.vy})`);
  assert.ok(Math.abs(wrongCell.vx) < 1e-6 && Math.abs(wrongCell.vy) < 1e-6,
    `expected world cell (4,3) to remain untouched under scaled mapping, got (${wrongCell.vx}, ${wrongCell.vy})`);
});

test('GPU shadow arrays are exposed through FluidField-compatible views for coupling consumers', () => {
  const fluid = makeShadowOnlyField({ size: 32, dt: 0.1, scaleX: 1, scaleY: 1 });

  fluid.addVelocity(12, 9, 2.4, -0.8);
  fluid.addDensity(12, 9, 150, 90, 40, 65);

  const idx = fluid.IX(12, 9);
  assert.ok(fluid.Vx[idx] > 1.0 && fluid.Vy[idx] < -0.2, `expected Vx/Vy aliases to mirror shadow velocity at idx=${idx}`);
  assert.ok(fluid.densityR[idx] > 30 && fluid.densityG[idx] > 15 && fluid.densityB[idx] > 8,
    `expected density aliases to mirror shadow dye at idx=${idx}`);

  fluid.step();

  const idxAfter = fluid.IX(12, 9);
  assert.equal(fluid.Vx, fluid.shadowVx, 'Vx alias should track swapped shadow velocity buffer');
  assert.equal(fluid.densityR, fluid.shadowDensityR, 'densityR alias should track swapped shadow density buffer');
  assert.ok(Math.abs(fluid.Vx[idxAfter]) > 1e-4, 'expected non-zero transported momentum after step');
});

test('GPU shadow velocity splat and advection sanitize NaN/Inf and clamp unsafe magnitudes', () => {
  const fluid = makeShadowOnlyField({ size: 32, dt: 2.5, scaleX: 1, scaleY: 1 });
  fluid.maxVelComponent = 3;

  fluid.addVelocity(10, 10, Infinity, NaN, 24);
  fluid.addVelocity(10, 10, -999, 999, 24);

  const seeded = fluid.getVelocityAtWorld(10, 10);
  assert.ok(Number.isFinite(seeded.vx) && Number.isFinite(seeded.vy), 'seeded velocity should remain finite');
  assert.ok(Math.abs(seeded.vx) <= fluid.maxVelComponent + 1e-6, `vx should be clamped, got ${seeded.vx}`);
  assert.ok(Math.abs(seeded.vy) <= fluid.maxVelComponent + 1e-6, `vy should be clamped, got ${seeded.vy}`);

  fluid.step();
  const after = fluid.getVelocityAtWorld(10, 10);
  assert.ok(Number.isFinite(after.vx) && Number.isFinite(after.vy), 'advected velocity should remain finite');
  assert.ok(Math.abs(after.vx) <= fluid.maxVelComponent + 1e-6, `vx after step should remain clamped, got ${after.vx}`);
  assert.ok(Math.abs(after.vy) <= fluid.maxVelComponent + 1e-6, `vy after step should remain clamped, got ${after.vy}`);
});

test('GPU shadow boundary damping preserves interior dynamics while reducing edge spikes', () => {
  const fluid = makeShadowOnlyField({ size: 40, dt: 0.05, scaleX: 1, scaleY: 1 });
  fluid.maxVelComponent = 8;

  fluid.addVelocity(0, 20, 4.2, 0, 18);
  fluid.addVelocity(20, 20, 4.2, 0, 18);

  const edgeBefore = Math.abs(fluid.getVelocityAtWorld(0, 20).vx);
  const interiorBefore = Math.abs(fluid.getVelocityAtWorld(20, 20).vx);

  fluid.step();

  const edgeAfter = Math.abs(fluid.getVelocityAtWorld(0, 20).vx);
  const interiorAfter = Math.abs(fluid.getVelocityAtWorld(20, 20).vx);

  assert.ok(edgeAfter < edgeBefore, `expected edge damping (${edgeAfter} < ${edgeBefore})`);
  assert.ok(interiorAfter > 0.5, `expected interior flow to stay energetic, got ${interiorAfter}`);
  assert.ok(interiorAfter > edgeAfter * 0.6,
    `interior should not collapse relative to edge (${interiorAfter} vs ${edgeAfter})`);
});

test('GPU world-space sampling sanitizes corrupted shadow NaN/Inf values', () => {
  const fluid = makeShadowOnlyField({ size: 24, dt: 0.1, scaleX: 1, scaleY: 1 });
  fluid.maxVelComponent = 2.5;

  const idx = fluid.IX(6, 6);
  fluid.shadowVx[idx] = Infinity;
  fluid.shadowVy[idx] = NaN;
  fluid.shadowDensityR[idx] = Infinity;
  fluid.shadowDensityG[idx] = -Infinity;
  fluid.shadowDensityB[idx] = NaN;

  const vel = fluid.getVelocityAtWorld(6, 6);
  const density = fluid.getDensityAtWorld(6, 6);

  assert.deepEqual(vel, { vx: 0, vy: 0 }, 'velocity queries should sanitize non-finite shadow values');
  assert.deepEqual(density, [0, 0, 0, 1], 'density queries should sanitize non-finite shadow values');
});

test('GPU shadow advection prevents NaN/Inf contamination from spreading through bilinear samples', () => {
  const fluid = makeShadowOnlyField({ size: 28, dt: 0.2, scaleX: 1, scaleY: 1 });
  fluid.maxVelComponent = 4;

  fluid.addVelocity(14, 14, 2.2, 0.6, 16);
  fluid.addDensity(14, 14, 180, 40, 20, 60);

  const corruptIdx = fluid.IX(15, 14);
  fluid.shadowVx[corruptIdx] = NaN;
  fluid.shadowVy[corruptIdx] = Infinity;
  fluid.shadowDensityR[corruptIdx] = NaN;

  fluid.step();

  for (let y = 13; y <= 15; y++) {
    for (let x = 13; x <= 16; x++) {
      const sampled = fluid.getVelocityAtWorld(x, y);
      const dye = fluid.getDensityAtWorld(x, y);
      assert.ok(Number.isFinite(sampled.vx) && Number.isFinite(sampled.vy), `velocity at (${x},${y}) should stay finite`);
      assert.ok(Number.isFinite(dye[0]) && Number.isFinite(dye[1]) && Number.isFinite(dye[2]), `density at (${x},${y}) should stay finite`);
    }
  }
});

test('GPU shadow idle step keeps quiescent fields near-zero while preserving boundary finite guards', () => {
  const fluid = makeShadowOnlyField({ size: 20, dt: 0.1, scaleX: 1, scaleY: 1 });
  fluid.maxVelComponent = 2;

  const edgeIdx = fluid.IX(0, 10);
  fluid.shadowVx[edgeIdx] = Infinity;
  fluid.shadowVy[edgeIdx] = -Infinity;

  fluid.step();

  const edge = fluid.getVelocityAtWorld(0, 10);
  const center = fluid.getVelocityAtWorld(10, 10);
  assert.deepEqual(edge, { vx: 0, vy: 0 }, 'edge guard should sanitize non-finite boundary velocity');
  assert.ok(Math.abs(center.vx) < 1e-6 && Math.abs(center.vy) < 1e-6, 'idle interior should remain near zero');
});

test('GPU addVelocity sanitizes/clamps WebGPU splat uniforms before dispatch', () => {
  const prevW = globalThis.WORLD_WIDTH;
  const prevH = globalThis.WORLD_HEIGHT;
  globalThis.WORLD_WIDTH = 32;
  globalThis.WORLD_HEIGHT = 32;

  try {
    const fluid = makeShadowOnlyField({ size: 32, dt: 0.1, scaleX: 1, scaleY: 1 });
    fluid.maxVelComponent = 10;
    fluid.gpuEnabled = true;
    fluid.device = {};
    fluid.textures = { velocityPing: { id: 'ping' }, velocityPong: { id: 'pong' } };

    let captured = null;
    fluid._runShaderPass = (_pipeline, spec) => {
      captured = spec;
    };

    fluid.addVelocity(8, 8, Infinity, NaN, 15);
    assert.ok(captured, 'expected WebGPU splat dispatch');
    assert.deepEqual(captured.u_splatValue.slice(0, 2), [0, 0], 'non-finite impulses should sanitize to zero');
    assert.ok(Number.isFinite(captured.u_radius) && captured.u_radius > 0 && captured.u_radius <= 1,
      `radius should be finite normalized clamp, got ${captured.u_radius}`);

    fluid.addVelocity(8, 8, 999, -999, 15);
    assert.ok(Math.abs(captured.u_splatValue[0]) <= 1.0 + 1e-6, `vx splat should be clamped, got ${captured.u_splatValue[0]}`);
    assert.ok(Math.abs(captured.u_splatValue[1]) <= 1.0 + 1e-6, `vy splat should be clamped, got ${captured.u_splatValue[1]}`);
  } finally {
    if (prevW === undefined) delete globalThis.WORLD_WIDTH;
    else globalThis.WORLD_WIDTH = prevW;
    if (prevH === undefined) delete globalThis.WORLD_HEIGHT;
    else globalThis.WORLD_HEIGHT = prevH;
  }
});
