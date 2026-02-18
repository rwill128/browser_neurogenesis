import test from 'node:test';
import assert from 'node:assert/strict';
import { createCreatureSpecFromMesh, parseCreatureSpec, buildBodiesFromCreatureSpec, CREATURE_SPEC_VERSION } from '../../sim-server/public/creature-spec.js';

const CONTROLS = { massSoft: 0.6, massHeavy: 5.0 };

test('CreatureSpec v2 roundtrip parse and build bodies', () => {
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

  const spec = createCreatureSpecFromMesh(mesh, { name: 'test', fields: { rigidField: new Float32Array(16 * 16), softField: new Float32Array(16 * 16) } });
  assert.equal(spec.schemaVersion, CREATURE_SPEC_VERSION);
  assert.ok(Array.isArray(spec.rigidBodies));
  assert.ok(Array.isArray(spec.softBodies));
  // rigidWelds removed from public spec (compound rigid is fused at export)
  assert.ok(Array.isArray(spec.hybridJoints));
  assert.equal(spec.hybridJoints.length, 0);
  assert.equal(spec.mesh, undefined);

  const parsed = parseCreatureSpec(JSON.stringify(spec));
  const bodies = buildBodiesFromCreatureSpec(parsed, 256, CONTROLS);

  assert.ok(Array.isArray(bodies.rigid));
  assert.ok(Array.isArray(bodies.soft.nodes));
  assert.ok(Array.isArray(bodies.soft.springs));
  assert.ok(Array.isArray(bodies.hybrid));
  assert.equal(bodies.hybrid.length, 0);
  assert.equal(bodies.rigidWelds, undefined);
  assert.equal(bodies.rigid.length, 1);
  assert.ok(bodies.soft.nodes.length >= 3);
  for (const [a, b] of bodies.soft.springs) {
    assert.ok(Number.isInteger(a) && a >= 0 && a < bodies.soft.nodes.length);
    assert.ok(Number.isInteger(b) && b >= 0 && b < bodies.soft.nodes.length);
  }
});

// hybrid link generation removed from creature spec.

// hybrid link generation removed from creature spec.

test('createCreatureSpecFromMesh fuses compiler rigid pieces by compoundId (no spring-style inter-piece weld export)', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 1, soft: 0 },
      { id: 1, x: 4, y: 0, rigid: 1, soft: 0 },
      { id: 2, x: 4, y: 4, rigid: 1, soft: 0 },
      { id: 3, x: 0, y: 4, rigid: 1, soft: 0 },
    ],
    triangles: [],
    rigidPieces: [
      { id: 'p0', compoundId: 'compound_A', hull: [{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 1, y: 2 }], sourceNodeIds: [0, 1, 2] },
      { id: 'p1', compoundId: 'compound_A', hull: [{ x: 2, y: 0 }, { x: 4, y: 0 }, { x: 3, y: 2 }], sourceNodeIds: [1, 2, 3] },
      { id: 'p2', compoundId: 'compound_B', hull: [{ x: 2, y: 2 }, { x: 4, y: 2 }, { x: 3, y: 4 }], sourceNodeIds: [1, 2, 3] },
    ],
    rigidWelds: [
      { a: 0, b: 1, a0: 0, a1: 1, b0: 0, b1: 1 },
      { a: 1, b: 0, a0: 1, a1: 2, b0: 1, b1: 2 },
      { a: 1, b: 2, a0: 0, a1: 1, b0: 0, b1: 1 },
      { a: 99, b: 0, a0: 0, a1: 1, b0: 0, b1: 1 },
    ],
    meta: { width: 8, height: 8 },
  };

  const spec = createCreatureSpecFromMesh(mesh);
  assert.equal(spec.rigidBodies.length, 2);
  assert.equal(spec.rigidWelds, undefined);
  const fusedA = spec.rigidBodies.find((rb) => rb.compoundId === 'compound_A');
  assert.ok(fusedA);
  assert.ok(Array.isArray(fusedA.subHulls));
  assert.equal(fusedA.subHulls.length, 2);
});

test('buildBodiesFromCreatureSpec drops non-finite rigid hull vertices deterministically', () => {
  const spec = {
    schemaVersion: CREATURE_SPEC_VERSION,
    space: { width: 10, height: 10 },
    rigidBodies: [{
      id: 'rb0',
      hull: [
        { x: 1, y: 1 },
        { x: 5, y: 1 },
        { x: Number.POSITIVE_INFINITY, y: 3 },
        { x: 3, y: 6 },
      ],
      mass: 4,
    }],
    softBodies: [],
    hybridJoints: [],
  };

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  assert.equal(bodies.rigid.length, 1);

  const rb = bodies.rigid[0];
  assert.equal(rb.verticesLocal.length, 3);
  assert.ok(Number.isFinite(rb.x) && Number.isFinite(rb.y));
  assert.ok(Number.isFinite(rb.r) && rb.r > 0);
  assert.ok(Number.isFinite(rb.inertia) && rb.inertia > 0);
  for (const v of rb.verticesLocal) {
    assert.ok(Number.isFinite(v.x) && Number.isFinite(v.y));
  }
});

// hybrid link generation removed from creature spec.
