import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveRigidVsRigidPolygonCollision, getRigidCollisionPolysWorld, rigidVerticesWorld, pointInPolygonInclusive } from '../../sim-server/public/rigid-collision.js';

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

test('rigid-rigid polygon collision: no force when bodies are clearly separated', () => {
  const a = makeConcaveL(20, 20);
  const b = makeBox(58, 20, 3);
  a.vx = 0.7;
  b.vx = -0.3;

  const ax = a.x; const ay = a.y;
  const bx = b.x; const by = b.y;
  const avx = a.vx; const bvx = b.vx;

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3);
  assert.equal(hit, false, 'separated bodies must not collide');
  assert.equal(a.x, ax);
  assert.equal(a.y, ay);
  assert.equal(b.x, bx);
  assert.equal(b.y, by);
  assert.equal(a.vx, avx);
  assert.equal(b.vx, bvx);
});

test('rigid-rigid proxy decomposition stays inside concave hull and logs triggering proxy pair', () => {
  const a = makeConcaveL(32, 32);
  const b = makeBox(26, 26, 2.5);

  const hullA = rigidVerticesWorld(a);
  const polysA = getRigidCollisionPolysWorld(a);
  for (const poly of polysA) {
    let cx = 0; let cy = 0;
    for (const p of poly) { cx += p.x; cy += p.y; }
    cx /= Math.max(1, poly.length);
    cy /= Math.max(1, poly.length);
    assert.equal(pointInPolygonInclusive(cx, cy, hullA), true, 'proxy centroid should remain inside source concave hull');
  }

  const debug = { contacts: [], aIndex: 3, bIndex: 7, iter: 0, phase: 'test' };
  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3, debug);
  assert.equal(hit, true, 'expected overlap for proxy-pair logging case');
  assert.ok(debug.contacts.length >= 1, 'expected proxy contact log entry');
  const c = debug.contacts[0];
  assert.equal(c.a, 3);
  assert.equal(c.b, 7);
  assert.ok(Number.isInteger(c.proxyA) && c.proxyA >= 0);
  assert.ok(Number.isInteger(c.proxyB) && c.proxyB >= 0);
});
