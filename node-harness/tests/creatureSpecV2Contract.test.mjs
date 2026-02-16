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

function assertMembraneRingOnly(sb) {
  assert.ok(sb && Array.isArray(sb.nodes), 'membrane body should have nodes');
  assert.ok(Array.isArray(sb.springs), 'membrane body should have springs');
  const n = sb.nodes.length;
  assert.ok(n >= 3, 'membrane ring requires at least 3 nodes');
  assert.equal(sb.springs.length, n, 'ring-only membrane must have exactly one edge per node');

  const deg = new Array(n).fill(0);
  const adj = Array.from({ length: n }, () => []);
  const seen = new Set();

  for (const sp of sb.springs) {
    assert.ok(Array.isArray(sp) && sp.length >= 3, 'spring record malformed');
    const a = Number(sp[0]);
    const b = Number(sp[1]);
    assert.ok(Number.isInteger(a) && a >= 0 && a < n, `spring endpoint a out of range: ${a}`);
    assert.ok(Number.isInteger(b) && b >= 0 && b < n, `spring endpoint b out of range: ${b}`);
    assert.notEqual(a, b, 'ring spring cannot be self-loop');

    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    assert.equal(seen.has(key), false, `duplicate membrane edge detected: ${key}`);
    seen.add(key);

    deg[a] += 1;
    deg[b] += 1;
    adj[a].push(b);
    adj[b].push(a);

    const edgeBodyMode = Number(sp[3]);
    assert.equal(edgeBodyMode, 1, 'membrane ring edges should be blocking');
  }

  for (let i = 0; i < n; i++) {
    assert.equal(deg[i], 2, `ring node degree must be 2 (node ${i} has ${deg[i]})`);
  }

  // Confirm single-cycle connectivity.
  const visited = new Set([0]);
  const stack = [0];
  while (stack.length) {
    const u = stack.pop();
    for (const v of adj[u]) {
      if (visited.has(v)) continue;
      visited.add(v);
      stack.push(v);
    }
  }
  assert.equal(visited.size, n, 'membrane ring should be one connected cycle');
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

test('createCreatureSpecFromMesh compacts default rigid edge arrays', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  const rb = spec.rigidBodies[0];
  assert.ok(rb, 'expected rigid body export');
  assert.equal(rb.edgeBodyMode, undefined);
  assert.equal(rb.edgeDyeMode, undefined);
  assert.equal(rb.edgePermeabilityRGB, undefined);
});

test('buildBodiesFromCreatureSpec deterministically restores compacted rigid edge defaults', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  const rb = spec.rigidBodies[0];
  assert.ok(rb, 'expected rigid body export');
  // Export should omit default arrays for compactness.
  assert.equal(rb.edgeBodyMode, undefined);
  assert.equal(rb.edgeDyeMode, undefined);
  assert.equal(rb.edgePermeabilityRGB, undefined);

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  assert.equal(bodies.rigid.length, 1);

  const imported = bodies.rigid[0];
  assert.equal(imported.sides, imported.verticesLocal.length);

  // Missing arrays in spec must normalize to stable solver defaults.
  assert.equal(imported.edgeBodyMode.length, imported.sides);
  assert.ok(imported.edgeBodyMode.every((m) => Number(m) === 1));

  assert.equal(imported.edgeDyeMode.length, imported.sides);
  assert.ok(imported.edgeDyeMode.every((rgb) => Array.isArray(rgb)
    && rgb.length === 3
    && Number(rgb[0]) === 1
    && Number(rgb[1]) === 1
    && Number(rgb[2]) === 1));

  assert.equal(imported.edgePermeabilityRGB.length, imported.sides);
  assert.ok(imported.edgePermeabilityRGB.every((rgb) => Array.isArray(rgb)
    && rgb.length === 3
    && Number(rgb[0]) === 0
    && Number(rgb[1]) === 0
    && Number(rgb[2]) === 0));
});

test('soft solver mode is exported and membrane mode is mapped on import bodies', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 0, soft: 1 },
      { id: 1, x: 4, y: 0, rigid: 0, soft: 1 },
      { id: 2, x: 2, y: 4, rigid: 0, soft: 1 },
    ],
    triangles: [
      { kind: 'soft', a: 0, b: 1, c: 2 },
    ],
    meta: { width: 8, height: 8 },
  };

  const membraneSpec = createCreatureSpecFromMesh(mesh, {
    softSolverMode: 'membrane',
    softBoundaryRingSprings: false,
  });
  assert.equal(membraneSpec.softBodies[0].solverMode, 'membrane');
  assert.ok(Number(membraneSpec.softBodies[0].restArea) > 0);
  assert.ok(Number(membraneSpec.softBodies[0].shapeMemoryGain) > 0);
  assert.equal(Number(membraneSpec.softBodies[0].insideCorrectionEnabled), 1);
  assertMembraneRingOnly(membraneSpec.softBodies[0]);

  const importedMembrane = buildBodiesFromCreatureSpec(membraneSpec, 64, CONTROLS);
  assert.ok(Array.isArray(importedMembrane.softMembraneClusters));
  assert.equal(importedMembrane.softMembraneClusters.length, 1);
  assert.equal(importedMembrane.softMembraneClusters[0].clusterId, 0);
  assert.ok(Number(importedMembrane.softMembraneClusters[0].shapeMemoryGain) > 0);
  assert.equal(Number(importedMembrane.softMembraneClusters[0].insideCorrectionEnabled), 1);

  const springSpec = createCreatureSpecFromMesh(mesh, {
    softSolverMode: 'spring',
    softBoundaryRingSprings: false,
  });
  const importedSpring = buildBodiesFromCreatureSpec(springSpec, 64, CONTROLS);
  assert.equal(importedSpring.softMembraneClusters.length, 0);
});

test('unknown soft solver mode deterministically normalizes to spring (no membrane metadata)', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 1, y: 1, rigid: 0, soft: 1 },
      { id: 1, x: 5, y: 1, rigid: 0, soft: 1 },
      { id: 2, x: 3, y: 5, rigid: 0, soft: 1 },
    ],
    triangles: [
      { kind: 'soft', a: 0, b: 1, c: 2 },
    ],
    meta: { width: 8, height: 8 },
  };

  const spec = createCreatureSpecFromMesh(mesh, {
    softSolverMode: 'future-cellular-mode',
    softBoundaryRingSprings: false,
  });

  assert.equal(spec.softBodies.length, 1);
  assert.equal(spec.softBodies[0].solverMode, 'spring');
  assert.equal(spec.softBodies[0].restArea, undefined);

  const imported = buildBodiesFromCreatureSpec(spec, 64, CONTROLS);
  assert.equal(imported.softMembraneClusters.length, 0);
  assert.ok(imported.soft.nodes.length >= 3);
  assert.ok(imported.soft.springs.length >= 3);
});

test('unknown imported soft solver mode falls back to spring even when membrane-only fields are present', () => {
  const spec = {
    schemaVersion: CREATURE_SPEC_VERSION,
    space: { width: 24, height: 24 },
    rigidBodies: [],
    hybridJoints: [],
    softBodies: [{
      id: 'soft_unknown_solver',
      solverMode: 'future-anisotropic-membrane',
      nodes: [
        { x: 2, y: 2 },
        { x: 20, y: 2 },
        { x: 20, y: 20 },
        { x: 2, y: 20 },
      ],
      springs: [
        [0, 1, 18, 1, [1, 1, 1]],
        [1, 2, 18, 1, [1, 1, 1]],
        [2, 3, 18, 1, [1, 1, 1]],
        [3, 0, 18, 1, [1, 1, 1]],
      ],
      // These fields are membrane-specific and should be ignored when solverMode is unknown.
      restArea: 324,
      pressureGain: 0.08,
      radialDamping: 0.05,
      shapeMemoryGain: 0.04,
      insideCorrectionEnabled: 1,
    }],
  };

  const imported = buildBodiesFromCreatureSpec(spec, 96, CONTROLS);
  assert.equal(imported.softMembraneClusters.length, 0, 'unknown solver mode must not create membrane metadata');
  assert.equal(imported.soft.nodes.length, 4);
  assert.equal(imported.soft.springs.length, 4);
});

test('buildBodiesFromCreatureSpec sanitizes membrane imports to perimeter-only springs', () => {
  const spec = {
    schemaVersion: CREATURE_SPEC_VERSION,
    space: { width: 16, height: 16 },
    rigidBodies: [],
    hybridJoints: [],
    softBodies: [{
      id: 'membrane_0',
      solverMode: 'membrane',
      nodes: [
        { x: 2, y: 2 },
        { x: 14, y: 2 },
        { x: 14, y: 14 },
        { x: 2, y: 14 },
      ],
      // Includes one interior chord (0-2) that should be dropped on import.
      springs: [
        [0, 1, 12, 1, [1, 1, 1]],
        [1, 2, 12, 1, [1, 1, 1]],
        [2, 3, 12, 1, [1, 1, 1]],
        [3, 0, 12, 1, [1, 1, 1]],
        [0, 2, 17, 1, [1, 1, 1]],
      ],
      restArea: 144,
      pressureGain: 0.08,
      radialDamping: 0.05,
      shapeMemoryGain: 0.04,
      insideCorrectionEnabled: 1,
    }],
  };

  const imported = buildBodiesFromCreatureSpec(spec, 64, CONTROLS);
  assert.equal(imported.softMembraneClusters.length, 1, 'expected membrane cluster metadata');
  assert.equal(imported.soft.nodes.length, 4);

  const importedBody = {
    nodes: imported.soft.nodes,
    springs: imported.soft.springs.map(([a, b, rest, edgeBodyMode, edgeDyeMode]) => [a, b, rest, edgeBodyMode, edgeDyeMode]),
  };
  assertMembraneRingOnly(importedBody);

  const edgeKeys = new Set(importedBody.springs.map((sp) => {
    const a = Number(sp[0]);
    const b = Number(sp[1]);
    return a < b ? `${a}-${b}` : `${b}-${a}`;
  }));
  assert.equal(edgeKeys.has('0-2'), false, 'membrane import should drop interior chord springs');
});

test('buildBodiesFromCreatureSpec keeps concave membrane perimeter order instead of convexifying it', () => {
  const spec = {
    schemaVersion: CREATURE_SPEC_VERSION,
    space: { width: 24, height: 24 },
    rigidBodies: [],
    hybridJoints: [],
    softBodies: [{
      id: 'membrane_concave',
      solverMode: 'membrane',
      nodes: [
        { x: 2, y: 2 },
        { x: 18, y: 2 },
        { x: 18, y: 18 },
        { x: 10, y: 10 }, // concave notch (intentionally inside convex hull edge 2->4)
        { x: 2, y: 18 },
      ],
      springs: [
        [0, 1, 16, 1, [1, 1, 1]],
        [1, 2, 16, 1, [1, 1, 1]],
        [2, 3, 11.3, 1, [1, 1, 1]],
        [3, 4, 11.3, 1, [1, 1, 1]],
        [4, 0, 16, 1, [1, 1, 1]],
        [0, 2, 22.6, 1, [1, 1, 1]],
      ],
      restArea: 200,
      pressureGain: 0.08,
      radialDamping: 0.05,
      shapeMemoryGain: 0.04,
      insideCorrectionEnabled: 1,
    }],
  };

  const imported = buildBodiesFromCreatureSpec(spec, 96, CONTROLS);
  const importedBody = {
    nodes: imported.soft.nodes,
    springs: imported.soft.springs.map(([a, b, rest, edgeBodyMode, edgeDyeMode]) => [a, b, rest, edgeBodyMode, edgeDyeMode]),
  };
  assertMembraneRingOnly(importedBody);

  const touchesNotch = importedBody.springs.filter((sp) => Number(sp[0]) === 3 || Number(sp[1]) === 3);
  assert.equal(touchesNotch.length, 2, 'concave notch node should remain in perimeter ring with degree 2');
});

test('membrane export from authoring soft field is perimeter-only (no interior infill springs)', () => {
  const w = 24;
  const h = 24;
  const rigidField = new Float32Array(w * h);
  const softField = new Float32Array(w * h);

  for (let y = 6; y <= 18; y++) {
    for (let x = 7; x <= 19; x++) {
      const dx = x - 13;
      const dy = y - 12;
      if ((dx * dx) / 36 + (dy * dy) / 25 <= 1) {
        softField[y * w + x] = 1;
      }
    }
  }

  const spec = createCreatureSpecFromMesh({
    nodes: [],
    triangles: [],
    meta: { width: w, height: h },
  }, {
    name: 'field-membrane',
    softSolverMode: 'membrane',
    fields: { rigidField, softField },
    threshold: 0.35,
    membraneMinEdgeLength: 4,
  });

  assert.ok(Array.isArray(spec.softBodies));
  assert.equal(spec.softBodies.length, 1);
  const sb = spec.softBodies[0];
  assert.equal(sb.solverMode, 'membrane');
  assert.ok(sb.nodes.length >= 3, 'expected perimeter loop nodes from painted field contour');
  assertMembraneRingOnly(sb);
  const minRest = Math.min(...sb.springs.map((sp) => Number(sp?.[2]) || 0));
  assert.ok(minRest >= 2.8, `expected membrane edge rests to stay near configured minimum edge length (minRest=${minRest})`);

  const bodies = buildBodiesFromCreatureSpec(spec, 64, CONTROLS);
  assert.equal(bodies.softMembraneClusters.length, 1);
  assert.ok(bodies.soft.springs.length >= 3);
});

test('membrane mode enforces perimeter-only ring even when soft mesh has no detectable boundary edges', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 0, soft: 1 },
      { id: 1, x: 4, y: 0, rigid: 0, soft: 1 },
      { id: 2, x: 0, y: 4, rigid: 0, soft: 1 },
    ],
    triangles: [
      // Duplicate/opposite-winding triangle soup: every undirected edge appears twice,
      // so boundary extraction can be empty.
      { kind: 'soft', a: 0, b: 1, c: 2 },
      { kind: 'soft', a: 0, b: 2, c: 1 },
    ],
    meta: { width: 8, height: 8 },
  };

  const spec = createCreatureSpecFromMesh(mesh, {
    softSolverMode: 'membrane',
    softBoundaryRingSprings: false,
    softSeamWeldSprings: false,
  });

  assert.equal(spec.softBodies.length, 1);
  const sb = spec.softBodies[0];
  assert.equal(sb.solverMode, 'membrane');
  assertMembraneRingOnly(sb);

  const imported = buildBodiesFromCreatureSpec(spec, 64, CONTROLS);
  assert.equal(imported.softMembraneClusters.length, 1);
});

test('membrane edge-length/shape maps modulate exported ring spacing and per-vertex give', () => {
  const w = 32;
  const h = 32;
  const rigidField = new Float32Array(w * h);
  const softField = new Float32Array(w * h);
  const edgeMinMap = new Float32Array(w * h).fill(0);
  const edgeMaxMap = new Float32Array(w * h).fill(1);
  const shapeLowMap = new Float32Array(w * h).fill(0.2);

  for (let y = 7; y <= 25; y++) {
    for (let x = 8; x <= 24; x++) {
      const dx = x - 16;
      const dy = y - 16;
      if ((dx * dx) / 64 + (dy * dy) / 49 <= 1) {
        softField[y * w + x] = 1;
      }
    }
  }

  const mkSpec = (edgeMap) => createCreatureSpecFromMesh({
    nodes: [],
    triangles: [],
    meta: { width: w, height: h },
  }, {
    name: 'membrane-map-test',
    softSolverMode: 'membrane',
    fields: {
      rigidField,
      softField,
      membraneEdgeMap: edgeMap,
      membraneShapeMap: shapeLowMap,
    },
    threshold: 0.35,
    membraneMinEdgeLength: 3,
    membraneMaxEdgeLength: 9,
  });

  const specMin = mkSpec(edgeMinMap);
  const specMax = mkSpec(edgeMaxMap);
  const sbMin = specMin.softBodies[0];
  const sbMax = specMax.softBodies[0];
  assertMembraneRingOnly(sbMin);
  assertMembraneRingOnly(sbMax);

  const meanEdge = (sb) => {
    const vals = (sb.springs || []).map((sp) => Number(sp?.[2]) || 0).filter((v) => v > 0);
    return vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length);
  };
  assert.ok(meanEdge(sbMax) > meanEdge(sbMin) + 0.8, `expected edge map=1 to increase spacing (${meanEdge(sbMin)} -> ${meanEdge(sbMax)})`);

  const weights = (sbMin.nodes || []).map((n) => Number(n?.shapeMemoryWeight));
  const avgW = weights.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0) / Math.max(1, weights.length);
  assert.ok(avgW <= 0.35, `expected low shape map to lower per-vertex memory weights (avg=${avgW})`);

  const bodies = buildBodiesFromCreatureSpec(specMin, 96, CONTROLS);
  const importedW = bodies.soft.nodes.map((n) => Number(n?.shapeMemoryWeight) || 0);
  const importedAvg = importedW.reduce((a, b) => a + b, 0) / Math.max(1, importedW.length);
  assert.ok(importedAvg <= 0.35, `expected imported node weights to preserve low shape-memory map (avg=${importedAvg})`);
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

  const spec = createCreatureSpecFromMesh(mesh, { softBoundaryRingSprings: false });
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

test('createCreatureSpecFromMesh can reinforce soft perimeter with border-ring springs', () => {
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
    meta: { width: 8, height: 8 },
  };

  const withoutRing = createCreatureSpecFromMesh(mesh, { softBoundaryRingSprings: false });
  const withRing = createCreatureSpecFromMesh(mesh, { softBoundaryRingSprings: true, softBoundaryRingStride: 2 });

  const hasOpposingDiag = (springs) => springs.some((s) => {
    const [a, b] = s;
    return (a === 0 && b === 3) || (a === 3 && b === 0);
  });

  assert.ok(!hasOpposingDiag(withoutRing.softBodies[0].springs || []), 'without perimeter ring, opposite diagonal should be absent');
  assert.ok(hasOpposingDiag(withRing.softBodies[0].springs || []), 'perimeter ring should add reinforcing opposite diagonal spring');
});

test('createCreatureSpecFromMesh can add seam-weld springs across axis-aligned shared soft edges', () => {
  const mesh = {
    nodes: [
      { id: 0, x: 0, y: 0, rigid: 0, soft: 1 },
      { id: 1, x: 4, y: 0, rigid: 0, soft: 1 },
      { id: 2, x: 4, y: 4, rigid: 0, soft: 1 },
      { id: 3, x: 0, y: 4, rigid: 0, soft: 1 },
      { id: 4, x: 8, y: 0, rigid: 0, soft: 1 },
      { id: 5, x: 8, y: 4, rigid: 0, soft: 1 },
    ],
    triangles: [
      { kind: 'soft', a: 0, b: 1, c: 3 },
      { kind: 'soft', a: 1, b: 2, c: 3 },
      { kind: 'soft', a: 1, b: 4, c: 2 },
      { kind: 'soft', a: 4, b: 5, c: 2 },
    ],
    meta: { width: 16, height: 8 },
  };

  const withoutSeamWeld = createCreatureSpecFromMesh(mesh, {
    softBoundaryRingSprings: false,
    softSeamWeldSprings: false,
  });
  const withSeamWeld = createCreatureSpecFromMesh(mesh, {
    softBoundaryRingSprings: false,
    softSeamWeldSprings: true,
  });

  const edgeKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);
  const withoutEdges = new Set((withoutSeamWeld.softBodies[0].springs || []).map((s) => edgeKey(Number(s[0]), Number(s[1]))));
  const withSprings = withSeamWeld.softBodies[0].springs || [];
  const withEdges = new Set(withSprings.map((s) => edgeKey(Number(s[0]), Number(s[1]))));

  const added = [...withEdges].filter((k) => !withoutEdges.has(k));
  assert.ok(withEdges.size > withoutEdges.size, 'expected seam weld mode to add at least one additional spring');
  assert.ok(added.length >= 1, 'expected at least one new seam weld spring edge');

  const addedRests = withSprings
    .filter((s) => added.includes(edgeKey(Number(s[0]), Number(s[1]))))
    .map((s) => Number(s[2]))
    .filter((v) => Number.isFinite(v));
  assert.ok(addedRests.some((r) => r > 4.5), `expected seam weld spring to span across seam, added rests=${JSON.stringify(addedRests)}`);
});

test('adversarial pure-soft cross-beam reinforcement bounds long-span springs on thin strips', () => {
  const cols = 11;
  const top = Array.from({ length: cols }, (_, i) => ({ id: i, x: i * 4, y: 0, rigid: 0, soft: 1 }));
  const bottom = Array.from({ length: cols }, (_, i) => ({ id: cols + i, x: i * 4, y: 1, rigid: 0, soft: 1 }));
  const nodes = [...top, ...bottom];

  const triangles = [];
  for (let i = 0; i < cols - 1; i++) {
    const t0 = i;
    const t1 = i + 1;
    const b0 = cols + i;
    const b1 = cols + i + 1;
    triangles.push({ kind: 'soft', a: t0, b: t1, c: b0 });
    triangles.push({ kind: 'soft', a: t1, b: b1, c: b0 });
  }

  const mesh = {
    nodes,
    triangles,
    softCrossBeams: [[0, cols - 1]], // intentionally adversarial long-span cross-beam across thin strip
    meta: { width: 64, height: 16 },
  };

  const unbounded = createCreatureSpecFromMesh(mesh, {
    softBoundaryRingSprings: false,
    softCrossBeamMaxSpanFactor: Number.POSITIVE_INFINITY,
  });
  const bounded = createCreatureSpecFromMesh(mesh, {
    softBoundaryRingSprings: false,
  });

  const maxRest = (springs) => springs.reduce((m, sp) => Math.max(m, Number(sp?.[2]) || 0), 0);
  const maxUnbounded = maxRest(unbounded.softBodies[0].springs || []);
  const maxBounded = maxRest(bounded.softBodies[0].springs || []);

  assert.ok(maxUnbounded >= 30, `expected unbounded mode to keep adversarial long cross-beam (maxRest=${maxUnbounded})`);
  assert.ok(maxBounded < maxUnbounded, `expected bounded mode to reject long cross-beam (${maxUnbounded} -> ${maxBounded})`);
  assert.ok(maxBounded <= 11 + 1e-6, `expected bounded long-span cap near 2.75x median structural edge (~11), saw ${maxBounded}`);
});

test('adversarial pure-soft ring reinforcement bounds long-span springs on thin strips', () => {
  const cols = 11;
  const top = Array.from({ length: cols }, (_, i) => ({ id: i, x: i * 4, y: 0, rigid: 0, soft: 1 }));
  const bottom = Array.from({ length: cols }, (_, i) => ({ id: cols + i, x: i * 4, y: 1, rigid: 0, soft: 1 }));
  const nodes = [...top, ...bottom];

  const triangles = [];
  for (let i = 0; i < cols - 1; i++) {
    const t0 = i;
    const t1 = i + 1;
    const b0 = cols + i;
    const b1 = cols + i + 1;
    triangles.push({ kind: 'soft', a: t0, b: t1, c: b0 });
    triangles.push({ kind: 'soft', a: t1, b: b1, c: b0 });
  }

  const mesh = {
    nodes,
    triangles,
    meta: { width: 64, height: 16 },
  };

  const unbounded = createCreatureSpecFromMesh(mesh, {
    softBoundaryRingSprings: true,
    softBoundaryRingStride: 8,
    softBoundaryRingMaxSpanFactor: Number.POSITIVE_INFINITY,
  });
  const bounded = createCreatureSpecFromMesh(mesh, {
    softBoundaryRingSprings: true,
    softBoundaryRingStride: 8,
  });

  const maxRest = (springs) => springs.reduce((m, sp) => Math.max(m, Number(sp?.[2]) || 0), 0);
  const maxUnbounded = maxRest(unbounded.softBodies[0].springs || []);
  const maxBounded = maxRest(bounded.softBodies[0].springs || []);

  assert.ok(maxUnbounded >= 30, `expected adversarial long spring in unbounded mode, saw ${maxUnbounded}`);
  assert.ok(maxBounded < maxUnbounded, `expected bounded mode to reduce long-span springs (${maxUnbounded} -> ${maxBounded})`);
  assert.ok(maxBounded <= 11 + 1e-6, `expected bounded long-span cap near 2.75x median boundary edge (~11), saw ${maxBounded}`);
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

test('buildBodiesFromCreatureSpec normalizes per-edge body modes to binary pass/block', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());

  // Rigid edges: only mode=1 should remain blocking; everything else coerces to pass (0).
  spec.rigidBodies[0].edgeBodyMode = [1, 0, 42];

  // Soft spring edge body mode follows the same binary normalization.
  const firstSpring = spec.softBodies[0].springs[0];
  const secondSpring = spec.softBodies[0].springs[1];
  if (Array.isArray(firstSpring)) firstSpring[3] = 999;
  if (Array.isArray(secondSpring)) secondSpring[3] = 1;

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);

  assert.deepEqual(bodies.rigid[0].edgeBodyMode, [1, 0, 0]);
  assert.equal(Number(bodies.soft.springs[0][3]), 0);
  assert.equal(Number(bodies.soft.springs[1][3]), 1);
});

test('buildBodiesFromCreatureSpec deterministically defaults malformed soft spring trait slots', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh(), { softBoundaryRingSprings: false });
  const sb = spec.softBodies[0];
  assert.ok(sb?.springs?.length >= 1, 'sample soft body should include at least one spring');

  // Remove optional trait slots from first spring tuple.
  sb.springs[0] = [sb.springs[0][0], sb.springs[0][1], sb.springs[0][2]];

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  const imported = bodies.soft.springs[0];
  assert.ok(Array.isArray(imported), 'imported spring should be tuple-like');

  // Missing edgeBodyMode defaults to pass (0) and missing edgeDyeMode defaults
  // channel-wise to DEFLECT [1,1,1].
  assert.equal(Number(imported[3]), 0);
  assert.deepEqual(imported[4], [1, 1, 1]);
});

test('buildBodiesFromCreatureSpec deterministically normalizes malformed soft spring dye tuples', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh(), { softBoundaryRingSprings: false });
  const sb = spec.softBodies[0];
  assert.ok(sb?.springs?.length >= 3, 'sample soft body should include multiple springs');

  // Per-channel coercion should handle mixed invalid payloads deterministically.
  sb.springs[0][4] = null;
  sb.springs[1][4] = [2, 'oops', -99];
  sb.springs[2][4] = [0, 1, 2];

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  const imported = bodies.soft.springs;

  assert.deepEqual(imported[0][4], [1, 1, 1], 'missing tuple should default to DEFLECT RGB');
  assert.deepEqual(imported[1][4], [2, 1, 1], 'invalid channels should clamp to DEFLECT without disturbing valid channels');
  assert.deepEqual(imported[2][4], [0, 1, 2], 'valid dye tuple should survive normalization intact');
});

test('buildBodiesFromCreatureSpec deterministically backfills sparse rigid edge arrays with safe defaults', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  const rb = spec.rigidBodies[0];
  assert.ok(rb?.hull?.length >= 3, 'sample rigid body should have at least 3 edges');

  // Provide sparse arrays (fewer entries than hull edges).
  rb.edgeBodyMode = [0];
  rb.edgeDyeMode = [[2, 0, 1]];
  rb.edgePermeabilityRGB = [[1, 0, 0]];

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  const imported = bodies.rigid[0];
  assert.equal(imported.edgeBodyMode.length, imported.sides);
  assert.equal(imported.edgeDyeMode.length, imported.sides);
  assert.equal(imported.edgePermeabilityRGB.length, imported.sides);

  // Explicit first edge values survive normalization.
  assert.equal(imported.edgeBodyMode[0], 0);
  assert.deepEqual(imported.edgeDyeMode[0], [2, 0, 1]);
  assert.deepEqual(imported.edgePermeabilityRGB[0], [1, 0, 0]);

  // Missing trailing entries must backfill to deterministic solver defaults.
  for (let i = 1; i < imported.sides; i++) {
    assert.equal(imported.edgeBodyMode[i], 1);
    assert.deepEqual(imported.edgeDyeMode[i], [1, 1, 1]);
    assert.deepEqual(imported.edgePermeabilityRGB[i], [0, 0, 0]);
  }
});

test('buildBodiesFromCreatureSpec honors rigid inside-correction toggle per body', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  spec.rigidBodies[0].insideCorrectionEnabled = false;

  const bodiesOff = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  assert.equal(bodiesOff.rigid[0].insideCorrectionEnabled, false);

  spec.rigidBodies[0].insideCorrectionEnabled = true;
  const bodiesOn = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  assert.equal(bodiesOn.rigid[0].insideCorrectionEnabled, true);
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

test('buildBodiesFromCreatureSpec deterministically normalizes malformed rigid edge permeability payloads', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  spec.rigidBodies[0].edgePermeabilityRGB = [
    null,
    [2, 0.01, -5],
    // third edge intentionally omitted to verify sparse backfill behavior
  ];

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  const permeability = bodies.rigid[0].edgePermeabilityRGB;

  assert.deepEqual(permeability[0], [0, 0, 0], 'non-array payload should collapse to blocked RGB');
  assert.deepEqual(permeability[1], [1, 1, 0], 'channels should normalize by >0 permeability semantics');
  assert.deepEqual(permeability[2], [0, 0, 0], 'missing sparse trailing entries should backfill as blocked RGB');
});

test('buildBodiesFromCreatureSpec clamps and defaults soft per-vertex shapeMemoryWeight deterministically', () => {
  const spec = createCreatureSpecFromMesh(sampleMesh());
  const sb = spec.softBodies[0];
  assert.ok(sb?.nodes?.length >= 3, 'sample soft body should expose multiple nodes');

  sb.nodes[0].shapeMemoryWeight = -0.25; // underflow clamps to 0
  sb.nodes[1].shapeMemoryWeight = 1.8; // overflow clamps to 1
  delete sb.nodes[2].shapeMemoryWeight; // missing defaults to 1

  const bodies = buildBodiesFromCreatureSpec(spec, 256, CONTROLS);
  const weights = bodies.soft.nodes.slice(0, 3).map((n) => Number(n.shapeMemoryWeight));
  assert.deepEqual(weights, [0, 1, 1]);
});

test('buildBodiesFromCreatureSpec clamps malformed membrane shapeMemoryWeight values deterministically', () => {
  const spec = {
    schemaVersion: CREATURE_SPEC_VERSION,
    space: { width: 16, height: 16 },
    rigidBodies: [],
    hybridJoints: [],
    softBodies: [{
      id: 'membrane_shape_clamp',
      solverMode: 'membrane',
      nodes: [
        { x: 2, y: 2, shapeMemoryWeight: -5 },
        { x: 14, y: 2, shapeMemoryWeight: 3.2 },
        { x: 14, y: 14, shapeMemoryWeight: Number.NaN },
        { x: 2, y: 14 },
      ],
      springs: [
        [0, 1, 12, 1, [1, 1, 1]],
        [1, 2, 12, 1, [1, 1, 1]],
        [2, 3, 12, 1, [1, 1, 1]],
        [3, 0, 12, 1, [1, 1, 1]],
      ],
      restArea: 144,
      pressureGain: 0.08,
      radialDamping: 0.05,
      shapeMemoryGain: 0.04,
      insideCorrectionEnabled: 1,
    }],
  };

  const bodies = buildBodiesFromCreatureSpec(spec, 64, CONTROLS);
  assert.equal(bodies.softMembraneClusters.length, 1, 'expected membrane cluster metadata');

  const weights = bodies.soft.nodes.slice(0, 4).map((n) => Number(n.shapeMemoryWeight));
  assert.deepEqual(weights, [0, 1, 1, 1]);
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
