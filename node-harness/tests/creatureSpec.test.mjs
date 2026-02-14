import test from 'node:test';
import assert from 'node:assert/strict';
import { createCreatureSpecFromMesh, parseCreatureSpec, buildBodiesFromCreatureSpec } from '../../sim-server/public/creature-spec.js';

test('CreatureSpec roundtrip parse and build bodies', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 2, y: 2, rigid: 1, soft: 0 },
      { id: 1, x: 6, y: 2, rigid: 1, soft: 0 },
      { id: 2, x: 4, y: 6, rigid: 1, soft: 0 },
      { id: 3, x: 10, y: 10, rigid: 0, soft: 1 },
      { id: 4, x: 14, y: 10, rigid: 0, soft: 1 },
      { id: 5, x: 12, y: 14, rigid: 0, soft: 1 },
    ],
    triangles: [
      { kind: 'rigid', a: 0, b: 1, c: 2 },
      { kind: 'soft', a: 3, b: 4, c: 5 },
    ],
    meta: { width: 16, height: 16 },
  };
  const spec = createCreatureSpecFromMesh(mesh, { name: 'test' });
  const parsed = parseCreatureSpec(JSON.stringify(spec));
  const bodies = buildBodiesFromCreatureSpec(parsed, 256, { massSoft: 0.6, massHeavy: 5.0 });

  assert.ok(Array.isArray(bodies.rigid));
  assert.ok(Array.isArray(bodies.soft.nodes));
  assert.ok(Array.isArray(bodies.soft.springs));
  assert.ok(Array.isArray(bodies.hybrid));
  assert.equal(bodies.rigid.length, 1);
  assert.ok(bodies.soft.nodes.length >= 3);
  for (const [a, b] of bodies.soft.springs) {
    assert.ok(Number.isInteger(a) && a >= 0 && a < bodies.soft.nodes.length);
    assert.ok(Number.isInteger(b) && b >= 0 && b < bodies.soft.nodes.length);
  }
});

test('shared rigid-soft boundary creates hybrid links', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 2, y: 2, rigid: 1, soft: 0 },
      { id: 1, x: 6, y: 2, rigid: 1, soft: 1 },
      { id: 2, x: 4, y: 6, rigid: 1, soft: 1 },
      { id: 3, x: 8, y: 6, rigid: 0, soft: 1 },
    ],
    triangles: [
      { kind: 'rigid', a: 0, b: 1, c: 2 },
      { kind: 'soft', a: 1, b: 2, c: 3 },
    ],
    meta: { width: 16, height: 16 },
  };
  const bodies = buildBodiesFromCreatureSpec(createCreatureSpecFromMesh(mesh), 256, { massSoft: 0.6, massHeavy: 5.0 });

  assert.ok(bodies.rigid.length === 1);
  assert.ok(bodies.soft.nodes.length >= 3);
  assert.ok(bodies.hybrid.length >= 1);
  for (const h of bodies.hybrid) {
    assert.ok(h.rigidIndex >= 0 && h.rigidIndex < bodies.rigid.length);
    assert.ok(h.nodeIndex >= 0 && h.nodeIndex < bodies.soft.nodes.length);
  }
});
