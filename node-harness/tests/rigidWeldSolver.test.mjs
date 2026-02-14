import test from 'node:test';
import assert from 'node:assert/strict';
import { applyRigidWeldConstraints, buildRigidWeldPairSet } from '../../sim-server/public/rigid-weld.js';

function rigidVertexWorld(rb, vi) {
  const verts = rb.verticesLocal || [];
  const n = verts.length;
  if (!n) return { x: rb.x, y: rb.y };
  const i = ((vi % n) + n) % n;
  const v = verts[i];
  const th = rb.theta || 0;
  const c = Math.cos(th), s = Math.sin(th);
  return {
    x: rb.x + v.x * c - v.y * s,
    y: rb.y + v.x * s + v.y * c,
  };
}

function weldedEdgeGap(bodies, weld) {
  const ra = bodies.rigid[weld.a];
  const rb = bodies.rigid[weld.b];
  const a0 = rigidVertexWorld(ra, weld.a0);
  const a1 = rigidVertexWorld(ra, weld.a1);
  const b0 = rigidVertexWorld(rb, weld.b0);
  const b1 = rigidVertexWorld(rb, weld.b1);
  return 0.5 * (
    Math.hypot(a0.x - b0.x, a0.y - b0.y) +
    Math.hypot(a1.x - b1.x, a1.y - b1.y)
  );
}

test('buildRigidWeldPairSet normalizes ordering and deduplicates', () => {
  const set = buildRigidWeldPairSet([
    { a: 4, b: 1 },
    { a: 1, b: 4 },
    { a: 3, b: 3 },
    { a: 2, b: 5 },
  ]);
  assert.equal(set.size, 2);
  assert.ok(set.has('1:4'));
  assert.ok(set.has('2:5'));
});

test('applyRigidWeldConstraints limits first-step correction for far-separated bodies (guardrail)', () => {
  const bodies = {
    rigid: [
      {
        x: 0, y: 0, vx: 0, vy: 0, theta: 0, omega: 0,
        verticesLocal: [{ x: -1, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }],
      },
      {
        x: 120, y: 0, vx: 0, vy: 0, theta: 0, omega: 0,
        verticesLocal: [{ x: -1, y: 0 }, { x: 1, y: 0 }, { x: 0, y: -1 }],
      },
    ],
    rigidWelds: [
      { a: 0, b: 1, a0: 1, a1: 2, b0: 0, b1: 2 },
    ],
  };

  applyRigidWeldConstraints({
    rigid: bodies.rigid,
    rigidWelds: bodies.rigidWelds,
    rigidVertexWorld,
    stiffness: 0.06,
    errorScale: 0.95,
    maxPairError: 2.5,
  });

  // Two welded vertex pairs should each contribute at most stiffness*maxPairError.
  const maxExpectedDelta = 2 * 0.06 * 2.5 + 1e-9;
  assert.ok(Math.abs(bodies.rigid[0].vx) <= maxExpectedDelta);
  assert.ok(Math.abs(bodies.rigid[1].vx) <= maxExpectedDelta);
  assert.ok(Number.isFinite(bodies.rigid[0].omega) && Number.isFinite(bodies.rigid[1].omega));
});

test('applyRigidWeldConstraints pulls welded rigid pieces together over steps', () => {
  const bodies = {
    rigid: [
      {
        x: 0, y: 0, vx: 0, vy: 0, theta: 0, omega: 0,
        verticesLocal: [{ x: -1, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }],
      },
      {
        x: 8, y: 0, vx: 0, vy: 0, theta: 0, omega: 0,
        verticesLocal: [{ x: -1, y: 0 }, { x: 1, y: 0 }, { x: 0, y: -1 }],
      },
    ],
    rigidWelds: [
      { a: 0, b: 1, a0: 1, a1: 2, b0: 0, b1: 2 },
    ],
  };

  const weld = bodies.rigidWelds[0];
  const gapBefore = weldedEdgeGap(bodies, weld);

  for (let step = 0; step < 40; step++) {
    applyRigidWeldConstraints({ rigid: bodies.rigid, rigidWelds: bodies.rigidWelds, rigidVertexWorld });
    for (const rb of bodies.rigid) {
      rb.x += rb.vx * 0.06;
      rb.y += rb.vy * 0.06;
      rb.theta += rb.omega * 0.06;
      rb.vx *= 0.92;
      rb.vy *= 0.92;
      rb.omega *= 0.92;
      assert.ok(Number.isFinite(rb.vx) && Number.isFinite(rb.vy) && Number.isFinite(rb.omega));
    }
  }

  const gapAfter = weldedEdgeGap(bodies, weld);
  assert.ok(gapAfter < gapBefore, `expected welded gap to shrink (before=${gapBefore}, after=${gapAfter})`);
});
