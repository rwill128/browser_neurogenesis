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
  assert.equal(spec.mesh, undefined);

  const parsed = parseCreatureSpec(JSON.stringify(spec));
  const bodies = buildBodiesFromCreatureSpec(parsed, 256, CONTROLS);

  assert.ok(Array.isArray(bodies.rigid));
  assert.ok(Array.isArray(bodies.soft.nodes));
  assert.ok(Array.isArray(bodies.soft.springs));
  assert.ok(Array.isArray(bodies.hybrid));
  assert.ok(Array.isArray(bodies.rigidWelds));
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

  const bodies = buildBodiesFromCreatureSpec(createCreatureSpecFromMesh(mesh), 256, CONTROLS);
  assert.equal(bodies.rigid.length, 1);
  assert.ok(bodies.soft.nodes.length >= 3);
  assert.ok(bodies.hybrid.length >= 1);
  for (const h of bodies.hybrid) {
    assert.ok(h.rigidIndex >= 0 && h.rigidIndex < bodies.rigid.length);
    assert.ok(h.nodeIndex >= 0 && h.nodeIndex < bodies.soft.nodes.length);
  }
});

test('hybrid links anchor to nearest rigid hull edge for irregular rigid meshes', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 1, soft: 0 },
      { id: 1, x: 20, y: 0, rigid: 1, soft: 1 },
      { id: 2, x: 18, y: 2, rigid: 1, soft: 1 },
      { id: 3, x: 0, y: 10, rigid: 1, soft: 0 },
      { id: 4, x: 24, y: 4, rigid: 0, soft: 1 },
    ],
    triangles: [
      { kind: 'rigid', a: 0, b: 1, c: 2 },
      { kind: 'rigid', a: 0, b: 2, c: 3 },
      { kind: 'soft', a: 1, b: 2, c: 4 },
    ],
    meta: { width: 24, height: 24 },
  };

  const spec = createCreatureSpecFromMesh(mesh);
  assert.equal(spec.rigidBodies.length, 1);
  assert.equal(spec.rigidWelds, undefined);

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  assert.equal(bodies.rigid.length, 1);
  assert.equal(bodies.rigidWelds.length, 0);
  assert.ok(bodies.hybrid.length >= 2);

  for (const h of bodies.hybrid) {
    assert.ok(h.restA < 110, `restA too large: ${h.restA}`);
    assert.ok(h.restB < 110, `restB too large: ${h.restB}`);
  }
});

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

test('hybrid links at rigid vertices choose a local incident edge (deterministic tie-break)', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 1, soft: 0 },
      { id: 1, x: 40, y: 0, rigid: 1, soft: 1 },
      { id: 2, x: 40, y: 3, rigid: 1, soft: 0 },
      { id: 3, x: 0, y: 12, rigid: 1, soft: 0 },
      { id: 4, x: 48, y: 6, rigid: 0, soft: 1 },
    ],
    triangles: [
      { kind: 'rigid', a: 0, b: 1, c: 2 },
      { kind: 'rigid', a: 0, b: 2, c: 3 },
      { kind: 'soft', a: 1, b: 2, c: 4 },
    ],
    meta: { width: 48, height: 48 },
  };

  const bodies = buildBodiesFromCreatureSpec(createCreatureSpecFromMesh(mesh), 256, CONTROLS);
  assert.equal(bodies.rigid.length, 1);

  const sharedVertexLink = bodies.hybrid.find((h) => {
    const rb0 = bodies.rigid[h.rigidIndex];
    const verts0 = rb0.verticesLocal.map((v) => ({ x: rb0.x + v.x, y: rb0.y + v.y }));
    const node0 = bodies.soft.nodes[h.nodeIndex];
    const da = Math.hypot(node0.x - verts0[h.vertexA].x, node0.y - verts0[h.vertexA].y);
    const db = Math.hypot(node0.x - verts0[h.vertexB].x, node0.y - verts0[h.vertexB].y);
    return Math.min(da, db) <= 1e-4;
  });
  assert.ok(sharedVertexLink, 'expected a hybrid link pinned on/near a rigid vertex');

  const rb = bodies.rigid[sharedVertexLink.rigidIndex];
  const verts = rb.verticesLocal.map((v) => ({ x: rb.x + v.x, y: rb.y + v.y }));
  const n = verts.length;

  const vertexIdx = sharedVertexLink.restA <= sharedVertexLink.restB ? sharedVertexLink.vertexA : sharedVertexLink.vertexB;
  const otherIdx = sharedVertexLink.restA <= sharedVertexLink.restB ? sharedVertexLink.vertexB : sharedVertexLink.vertexA;
  const prevIdx = (vertexIdx - 1 + n) % n;
  const nextIdx = (vertexIdx + 1) % n;

  const edgeLen = (i, j) => Math.hypot(verts[i].x - verts[j].x, verts[i].y - verts[j].y);
  const shortestIncident = Math.min(edgeLen(vertexIdx, prevIdx), edgeLen(vertexIdx, nextIdx));
  const chosenLen = edgeLen(vertexIdx, otherIdx);

  assert.ok(chosenLen <= shortestIncident + 1e-6,
    `expected local tie-break edge (chosen=${chosenLen}, shortestIncident=${shortestIncident})`);
});
