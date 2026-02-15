import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CREATURE_SPEC_VERSION,
  createCreatureSpecFromMesh,
  parseCreatureSpec,
  buildBodiesFromCreatureSpec,
} from '../../sim-server/public/creature-spec.js';

const CONTROLS = { massSoft: 0.6, massHeavy: 5.0 };

function isConcave(poly) {
  let hasPos = false;
  let hasNeg = false;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const c = poly[(i + 2) % poly.length];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (cross > 1e-6) hasPos = true;
    if (cross < -1e-6) hasNeg = true;
  }
  return hasPos && hasNeg;
}

function sampleMesh() {
  return {
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
}

test('parseCreatureSpec rejects unsupported schema versions', () => {
  assert.throws(() => parseCreatureSpec(JSON.stringify({ schemaVersion: 'creature-spec.v1' })), /Unsupported schemaVersion/);
});

test('createCreatureSpecFromMesh exports solver-ready v2 sections', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  assert.equal(spec.schemaVersion, CREATURE_SPEC_VERSION);
  assert.ok(Array.isArray(spec.rigidBodies));
  assert.ok(Array.isArray(spec.softBodies));
  assert.ok(Array.isArray(spec.hybridJoints));
  assert.equal(spec.rigidWelds, undefined);
  assert.equal(spec.mesh, undefined);

  const rb = spec.rigidBodies[0];
  assert.ok(Array.isArray(rb.hull));
  assert.equal(rb.nodes, undefined);
  assert.equal(rb.triangles, undefined);

  const sb = spec.softBodies[0];
  assert.ok(Array.isArray(sb.nodes));
  assert.ok(Array.isArray(sb.springs));
});

test('createCreatureSpecFromMesh carries compiler soft cross-beams into soft spring export', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 0, soft: 1 },
      { id: 1, x: 4, y: 0, rigid: 0, soft: 1 },
      { id: 2, x: 0, y: 4, rigid: 0, soft: 1 },
      { id: 3, x: 4, y: 4, rigid: 0, soft: 1 },
    ],
    triangles: [
      { kind: 'soft', a: 0, b: 1, c: 2 },
      { kind: 'soft', a: 1, b: 3, c: 2 },
    ],
    softCrossBeams: [[0, 3]],
    meta: { width: 8, height: 8 },
  };

  const spec = createCreatureSpecFromMesh(mesh);
  assert.equal(spec.softBodies.length, 1);
  const springs = spec.softBodies[0].springs || [];

  const hasPrimaryDiag = springs.some((s) => {
    const [a, b] = s;
    return (a === 1 && b === 2) || (a === 2 && b === 1);
  });
  const hasCrossBeamDiag = springs.some((s) => {
    const [a, b] = s;
    return (a === 0 && b === 3) || (a === 3 && b === 0);
  });

  assert.ok(hasPrimaryDiag, 'expected existing triangle diagonal spring');
  assert.ok(hasCrossBeamDiag, 'expected compiler cross-beam diagonal spring');
});

test('createCreatureSpecFromMesh merges adjacent rigid triangles into a single convex rigid body', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 1, soft: 0 },
      { id: 1, x: 4, y: 0, rigid: 1, soft: 0 },
      { id: 2, x: 4, y: 4, rigid: 1, soft: 0 },
      { id: 3, x: 0, y: 4, rigid: 1, soft: 0 },
    ],
    triangles: [
      { kind: 'rigid', a: 0, b: 1, c: 2 },
      { kind: 'rigid', a: 0, b: 2, c: 3 },
    ],
    meta: { width: 8, height: 8 },
  };

  const spec = createCreatureSpecFromMesh(mesh);
  assert.equal(spec.rigidBodies.length, 1);
  assert.equal(spec.rigidWelds, undefined);
});

test('createCreatureSpecFromMesh keeps compiler concave decomposition fused as one compound body', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 1, soft: 0 },
      { id: 1, x: 4, y: 0, rigid: 1, soft: 0 },
      { id: 2, x: 4, y: 4, rigid: 1, soft: 0 },
      { id: 3, x: 0, y: 4, rigid: 1, soft: 0 },
      { id: 4, x: 8, y: 0, rigid: 1, soft: 0 },
      { id: 5, x: 8, y: 4, rigid: 1, soft: 0 },
    ],
    triangles: [
      { kind: 'rigid', a: 0, b: 1, c: 2 },
      { kind: 'rigid', a: 0, b: 2, c: 3 },
      { kind: 'rigid', a: 1, b: 4, c: 5 },
      { kind: 'rigid', a: 1, b: 5, c: 2 },
    ],
    rigidPieces: [
      { id: 'rp0', compoundId: 'compound_concave', hull: [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 4 }, { x: 0, y: 4 }], sourceNodeIds: [0, 1, 2, 3] },
      { id: 'rp1', compoundId: 'compound_concave', hull: [{ x: 4, y: 0 }, { x: 8, y: 0 }, { x: 8, y: 4 }, { x: 4, y: 4 }], sourceNodeIds: [1, 4, 5, 2] },
    ],
    rigidWelds: [{ a: 0, b: 1, a0: 1, a1: 2, b0: 0, b1: 3 }],
    meta: { width: 8, height: 8 },
  };

  const spec = createCreatureSpecFromMesh(mesh);
  assert.equal(spec.rigidBodies.length, 1);
  assert.equal(spec.rigidWelds, undefined);
  assert.ok(Array.isArray(spec.rigidBodies[0].subHulls));
  assert.equal(spec.rigidBodies[0].subHulls.length, 2);
});

test('createCreatureSpecFromMesh preserves concave hull when compiler supplies single concave rigid piece', () => {
  const mesh = sampleMesh();
  mesh.rigidPieces = [
    {
      id: 'p0',
      compoundId: 'c0',
      hull: [
        { x: 0, y: 0 },
        { x: 6, y: 0 },
        { x: 6, y: 2 },
        { x: 2, y: 2 },
        { x: 2, y: 6 },
        { x: 0, y: 6 },
      ],
      sourceNodeIds: [0, 1, 2, 3],
    },
  ];

  const spec = createCreatureSpecFromMesh(mesh);
  assert.equal(spec.rigidBodies.length, 1);
  assert.equal(spec.rigidWelds, undefined);
  assert.equal(spec.rigidBodies[0].compoundId, 'c0');
  assert.equal(spec.rigidBodies[0].hull.length, 6);
  assert.ok(isConcave(spec.rigidBodies[0].hull));
});

test('createCreatureSpecFromMesh prefers compiler-provided rigid decomposition when present', () => {
  const mesh = sampleMesh();
  mesh.rigidPieces = [
    { id: 'p0', compoundId: 'c0', hull: [{ x: 2, y: 2 }, { x: 6, y: 2 }, { x: 4, y: 4 }], sourceNodeIds: [0, 1, 2] },
    { id: 'p1', compoundId: 'c0', hull: [{ x: 4, y: 4 }, { x: 6, y: 2 }, { x: 6, y: 6 }], sourceNodeIds: [1, 2, 3] },
  ];
  mesh.rigidWelds = [{ a: 0, b: 1, a0: 1, a1: 2, b0: 0, b1: 1 }];

  const spec = createCreatureSpecFromMesh(mesh);
  assert.equal(spec.rigidBodies.length, 1);
  assert.equal(spec.rigidWelds, undefined);
  assert.equal(spec.rigidBodies[0].compoundId, 'c0');
  assert.ok(Array.isArray(spec.rigidBodies[0].subHulls));
  assert.equal(spec.rigidBodies[0].subHulls.length, 2);
});

test('buildBodiesFromCreatureSpec ignores invalid joint references safely', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  spec.hybridJoints.push({ rigidBodyIndex: 999, softBodyIndex: 999, softNodeIndex: 999, edgeA: 0, edgeB: 1, restA: 1, restB: 1 });

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  assert.equal(bodies.rigidWelds, undefined);
  for (const h of bodies.hybrid) {
    assert.ok(h.rigidIndex >= 0 && h.rigidIndex < bodies.rigid.length);
    assert.ok(h.nodeIndex >= 0 && h.nodeIndex < bodies.soft.nodes.length);
  }
});

test('buildBodiesFromCreatureSpec clamps unknown edge dye modes to DEFLECT guardrail', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());

  // Strict mode: per-edge RGB tuples only; invalid channels clamp to DEFLECT.
  spec.rigidBodies[0].edgeDyeMode = [[99, 99, 99], [0, 0, 0], [2, 2, 2]];

  // Unknown per-channel values on soft springs should clamp channel-wise.
  const firstSpring = spec.softBodies[0].springs[0];
  if (Array.isArray(firstSpring)) firstSpring[4] = [7, -2, 2];

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  const rigidModes = bodies.rigid[0].edgeDyeMode[0];
  assert.deepEqual(rigidModes, [1, 1, 1]);
  assert.deepEqual(bodies.rigid[0].edgeDyeMode[1], [0, 0, 0]);
  assert.deepEqual(bodies.rigid[0].edgeDyeMode[2], [2, 2, 2]);

  const softSpringMode = bodies.soft.springs[0][4];
  assert.deepEqual(softSpringMode, [1, 1, 2]);
});

test('buildBodiesFromCreatureSpec preserves rigid per-edge permeability and interior consume-dye masks', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  spec.rigidBodies[0].edgePermeabilityRGB = [
    [1, 0, 1],
    [0, 0, 0],
    [2, -1, 0],
  ];
  spec.rigidBodies[0].consumeDyeRGB = [1, 0, 2];

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  assert.deepEqual(bodies.rigid[0].edgePermeabilityRGB[0], [1, 0, 1]);
  assert.deepEqual(bodies.rigid[0].edgePermeabilityRGB[1], [0, 0, 0]);
  assert.deepEqual(bodies.rigid[0].edgePermeabilityRGB[2], [1, 0, 0]);
  assert.deepEqual(bodies.rigid[0].consumeDyeRGB, [1, 0, 1]);
});

test('buildBodiesFromCreatureSpec de-degenerates hybrid joints when edgeA=edgeB', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  const j = spec.hybridJoints[0];
  assert.ok(j, 'sampleMesh should produce at least one hybrid joint');

  j.edgeA = 1;
  j.edgeB = 1;
  j.restA = Number.NaN;
  j.restB = Number.NaN;

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  const h = bodies.hybrid[0];
  assert.ok(h, 'expected at least one built hybrid joint');
  assert.notEqual(h.vertexA, h.vertexB, 'hybrid joint should span two rigid vertices');
  assert.ok(Number.isFinite(h.restA) && h.restA >= 0.8);
  assert.ok(Number.isFinite(h.restB) && h.restB >= 0.8);
});

test('buildBodiesFromCreatureSpec clamps oversized hybrid rest lengths', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  const j = spec.hybridJoints[0];
  assert.ok(j, 'sampleMesh should produce at least one hybrid joint');

  j.restA = 1e6;
  j.restB = 1e6;

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  const h = bodies.hybrid[0];
  const rb = bodies.rigid[h.rigidIndex];
  const maxRest = Math.max(6, rb.r * 1.5);

  assert.ok(h.restA <= maxRest + 1e-9);
  assert.ok(h.restB <= maxRest + 1e-9);
});

test('createCreatureSpecFromMesh preserves optional authoring field payload', () => {
  const w = 16;
  const rigidField = new Float32Array(w * w);
  const softField = new Float32Array(w * w);
  rigidField[3] = 0.75;
  softField[8] = 0.25;

  const spec = createCreatureSpecFromMesh(sampleMesh(), {
    fields: { rigidField, softField },
  });

  assert.ok(spec.authoring?.fields);
  assert.equal(spec.authoring.fields.width, 16);
  assert.equal(spec.authoring.fields.height, 16);
  assert.equal(spec.authoring.fields.rigid.length, w * w);
  assert.equal(spec.authoring.fields.soft.length, w * w);
  assert.equal(spec.authoring.fields.rigid[3], 0.75);
  assert.equal(spec.authoring.fields.soft[8], 0.25);
});
