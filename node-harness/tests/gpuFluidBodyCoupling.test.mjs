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
