import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveRigidVsRigidPolygonCollision } from '../../sim-server/public/rigid-collision.js';

function makeConcaveL(x, y) {
  return {
    x,
    y,
    vx: 0,
    vy: 0,
    theta: 0,
    omega: 0,
    mass: 8,
    inertia: 220,
    r: 12,
    verticesLocal: [
      { x: -10, y: -10 },
      { x: 10, y: -10 },
      { x: 10, y: -4 },
      { x: -4, y: -4 },
      { x: -4, y: 10 },
      { x: -10, y: 10 },
    ],
  };
}

function makeBox(x, y, half = 2) {
  return {
    x,
    y,
    vx: 0,
    vy: 0,
    theta: 0,
    omega: 0,
    mass: 2,
    inertia: 20,
    r: half,
    verticesLocal: [
      { x: -half, y: -half },
      { x: half, y: -half },
      { x: half, y: half },
      { x: -half, y: half },
    ],
  };
}

test('rigid-rigid polygon collision: concave notch does not false-collide', () => {
  const a = makeConcaveL(32, 32);
  const b = makeBox(36, 36, 1.8); // in L notch void (outside solid)

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3);
  assert.equal(hit, false, 'notch occupancy should not produce rigid-rigid collision');
  assert.equal(a.x, 32);
  assert.equal(b.x, 36);
});

test('rigid-rigid polygon collision: overlapping boxes separate and exchange momentum', () => {
  const a = makeBox(20, 20, 3);
  const b = makeBox(24, 20, 3);
  a.vx = 1.5;
  b.vx = -0.5;

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.25);
  assert.equal(hit, true, 'expected overlap collision');
  assert.ok(a.x < 20.1, `A should be pushed left, got ${a.x}`);
  assert.ok(b.x > 23.9, `B should be pushed right, got ${b.x}`);
  assert.ok(a.vx < 1.5, `A should lose rightward speed, got ${a.vx}`);
  assert.ok(b.vx > -0.5, `B should gain rightward speed, got ${b.vx}`);
});

test('rigid-rigid polygon collision keeps finite state under high-speed impact', () => {
  const a = makeBox(20, 20, 4);
  const b = makeBox(26, 20, 4);
  a.vx = 60;
  b.vx = -40;

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3);
  assert.equal(hit, true, 'expected collision under overlap');
  for (const v of [a.vx, a.vy, b.vx, b.vy, a.omega, b.omega, a.x, b.x]) {
    assert.ok(Number.isFinite(v), `state should remain finite, got ${v}`);
  }
  assert.ok(Math.abs(a.vx) < 80, `guardrail should bound extreme velocity response, got ${a.vx}`);
  assert.ok(Math.abs(b.vx) < 80, `guardrail should bound extreme velocity response, got ${b.vx}`);
});
