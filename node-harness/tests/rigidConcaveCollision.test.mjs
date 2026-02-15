import test from 'node:test';
import assert from 'node:assert/strict';
import {
  rigidVerticesWorld,
  pointInPolygonInclusive,
  resolveRigidVsSoftNodeCollision,
} from '../../sim-server/public/rigid-collision.js';

function makeConcaveRigid() {
  return {
    x: 32,
    y: 32,
    vx: 0,
    vy: 0,
    theta: 0,
    omega: 0,
    mass: 8,
    inertia: 220,
    // Concave L (clockwise): notch in bottom-right quadrant.
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

test('concave rigid collision: node in L-notch does not collide (no false circle hit)', () => {
  const rigid = makeConcaveRigid();
  const verts = rigidVerticesWorld(rigid);
  const node = { x: 36, y: 36, vx: 0, vy: 0, mass: 1, r: 1 };

  assert.equal(pointInPolygonInclusive(node.x, node.y, verts), false, 'sanity: node should be in concave notch (outside polygon)');

  const hit = resolveRigidVsSoftNodeCollision(rigid, node, verts, 0.25);
  assert.equal(hit, false, 'node in concave notch should not register collision');
  assert.equal(node.x, 36);
  assert.equal(node.y, 36);
});

test('concave rigid collision: node inside solid region is pushed out', () => {
  const rigid = makeConcaveRigid();
  const verts = rigidVerticesWorld(rigid);
  const node = { x: 24, y: 30, vx: 0, vy: 0, mass: 1, r: 1.2 };

  assert.equal(pointInPolygonInclusive(node.x, node.y, verts), true, 'sanity: node starts inside solid region');

  const hit = resolveRigidVsSoftNodeCollision(rigid, node, verts, 0.25);
  assert.equal(hit, true, 'inside node should collide');
  assert.equal(pointInPolygonInclusive(node.x, node.y, verts), false, 'inside node should be expelled from rigid polygon');
});

test('concave rigid collision: impulse transfers momentum from soft node into rigid body', () => {
  const rigid = makeConcaveRigid();
  const verts = rigidVerticesWorld(rigid);
  const node = { x: 21.4, y: 30, vx: 2.0, vy: 0, mass: 1, r: 1.0 };

  const hit = resolveRigidVsSoftNodeCollision(rigid, node, verts, 0.3);
  assert.equal(hit, true, 'expected contact at left rigid edge');
  assert.ok(node.vx < 1.2, `expected node to lose forward speed, got ${node.vx}`);
  assert.ok(rigid.vx > 0, `expected rigid to gain momentum, got ${rigid.vx}`);
});

test('concave rigid collision: high-speed contact is guardrailed to bounded impulse', () => {
  const rigid = makeConcaveRigid();
  const verts = rigidVerticesWorld(rigid);
  const node = { x: 21.4, y: 30, vx: 240.0, vy: 0, mass: 1, r: 1.0 };

  const hit = resolveRigidVsSoftNodeCollision(rigid, node, verts, 0.3);
  assert.equal(hit, true, 'expected contact at left rigid edge');
  assert.ok(Number.isFinite(rigid.vx) && Number.isFinite(rigid.omega), 'rigid velocity should remain finite');
  assert.ok(Math.abs(rigid.vx) < 40, `guardrail should cap rigid velocity injection, got ${rigid.vx}`);
  assert.ok(Math.abs(rigid.omega) < 3, `guardrail should cap rigid spin injection, got ${rigid.omega}`);
});

test('concave rigid collision: tiny near-edge overlap stays in slop band (no jitter impulse)', () => {
  const rigid = makeConcaveRigid();
  const verts = rigidVerticesWorld(rigid);
  const node = { x: 21.01, y: 30, vx: 0.03, vy: 0, mass: 1, r: 1.0 };

  const hit = resolveRigidVsSoftNodeCollision(rigid, node, verts, 0.3);
  assert.equal(hit, false, 'tiny overlap should remain in contact slop to avoid resting-contact jitter');
});

test('concave rigid collision: malformed non-finite polygon vertex is ignored safely', () => {
  const rigid = makeConcaveRigid();
  const verts = rigidVerticesWorld(rigid);
  verts[2] = { x: Number.NaN, y: Number.NaN };
  const node = { x: 21.4, y: 30, vx: 2.0, vy: 0, mass: 1, r: 1.0 };

  const hit = resolveRigidVsSoftNodeCollision(rigid, node, verts, 0.3);
  assert.equal(hit, true, 'collision should still resolve using remaining finite edges');
  assert.ok(Number.isFinite(node.x) && Number.isFinite(node.vx), 'node state should remain finite');
  assert.ok(Number.isFinite(rigid.x) && Number.isFinite(rigid.vx) && Number.isFinite(rigid.omega), 'rigid state should remain finite');
});
