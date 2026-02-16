import test from 'node:test';
import assert from 'node:assert/strict';
import { EDGE_DYE_MODE, EDGE_BODY_MODE, normalizeEdgeDyeModeRGB, normalizePermeabilityRGB, applyBodyEdgeFieldBarriers } from '../../sim-server/public/dye-barrier.js';
import { createCreatureSpecFromMesh, buildBodiesFromCreatureSpec } from '../../sim-server/public/creature-spec.js';

function rigidVerticesWorld(rb) {
  if (!rb?.verticesLocal?.length) return [];
  const th = rb.theta || 0;
  const c = Math.cos(th), s = Math.sin(th);
  return rb.verticesLocal.map((v) => ({
    x: rb.x + v.x * c - v.y * s,
    y: rb.y + v.x * s + v.y * c,
  }));
}

test('normalizeEdgeDyeModeRGB clamps invalid channels to DEFLECT', () => {
  const m = normalizeEdgeDyeModeRGB([99, -2, Number.NaN]);
  assert.deepEqual(m, [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT]);
});

test('normalizePermeabilityRGB clamps to binary channel mask', () => {
  const m = normalizePermeabilityRGB([1, 0, -3]);
  assert.deepEqual(m, [1, 0, 0]);
});

test('applyBodyEdgeFieldBarriers applies PASS/DEFLECT/ABSORB channel behavior on soft edge', () => {
  const n = 16;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const x = 8;
  const y = 8;
  const i = y * n + x;
  const ti = y * n + (x + 1);

  r[i] = 100;
  g[i] = 100;
  b[i] = 100;
  vx[i] = 1.0; // tangential to horizontal edge
  vy[i] = 0.0;

  const sim = {
    controls: { n },
    bodies: {
      rigid: [],
      soft: {
        nodes: [
          { x: 5, y: 8 },
          { x: 11, y: 8 },
        ],
        springs: [
          [0, 1, 6, EDGE_BODY_MODE.BLOCK, [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.ABSORB]],
        ],
      },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(Math.abs(r[i] - 100) < 1e-6, `PASS channel should remain unchanged, got ${r[i]}`);
  assert.ok(g[i] < 100, `DEFLECT channel should lose local dye, got ${g[i]}`);
  assert.ok(g[ti] > 0, `DEFLECT channel should move dye tangentially, got target ${g[ti]}`);
  assert.ok(b[i] < g[i], `ABSORB channel should attenuate more strongly than DEFLECT, b=${b[i]} g=${g[i]}`);
});

test('BLOCK edge mode still deflects fluid velocity when dye channels are PASS', () => {
  const n = 16;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const x = 8;
  const y = 8;
  const i = y * n + x;
  vx[i] = 0.0;
  vy[i] = 2.0; // normal to horizontal edge

  const sim = {
    controls: { n },
    bodies: {
      rigid: [],
      soft: {
        nodes: [
          { x: 5, y: 8 },
          { x: 11, y: 8 },
        ],
        springs: [
          [0, 1, 6, EDGE_BODY_MODE.BLOCK, [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS]],
        ],
      },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(Math.abs(vy[i]) < 2.0, `normal velocity should be reduced by BLOCK body barrier, got vy=${vy[i]}`);
});

test('soft edge velocity mode PASS overrides legacy BLOCK body mode for fluid velocity barriering', () => {
  const n = 16;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const x = 8;
  const y = 8;
  const i = y * n + x;
  vx[i] = 0.0;
  vy[i] = 2.0; // normal to horizontal edge

  const sim = {
    controls: { n },
    bodies: {
      rigid: [],
      soft: {
        nodes: [
          { x: 5, y: 8 },
          { x: 11, y: 8 },
        ],
        springs: [
          [0, 1, 6, EDGE_BODY_MODE.BLOCK, [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS], EDGE_BODY_MODE.PASS],
        ],
      },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(Math.abs(vy[i] - 2.0) < 1e-6, `velocity-mode PASS should skip soft velocity barrier even when legacy body mode is BLOCK, got vy=${vy[i]}`);
});

test('rigid edge permeability RGB overrides channel pass/blocked behavior', () => {
  const n = 24;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const i = 12 * n + 12;
  const ti = 12 * n + 13;
  r[i] = 120;
  g[i] = 120;
  b[i] = 120;
  vx[i] = 1.0;
  vy[i] = 0.0;

  const sim = {
    controls: { n },
    bodies: {
      rigid: [
        {
          x: 12,
          y: 12,
          theta: 0,
          verticesLocal: [
            { x: -4, y: 0 },
            { x: 4, y: 0 },
            { x: 4, y: 5 },
            { x: -4, y: 5 },
          ],
          edgeBodyMode: [EDGE_BODY_MODE.BLOCK, EDGE_BODY_MODE.BLOCK, EDGE_BODY_MODE.BLOCK, EDGE_BODY_MODE.BLOCK],
          edgeDyeMode: [
            [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT],
            [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT],
            [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT],
            [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT],
          ],
          edgePermeabilityRGB: [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
          ],
        },
      ],
      soft: { nodes: [], springs: [] },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(Math.abs(r[i] - 120) < 1e-6, 'permeable red channel should PASS unchanged');
  assert.ok(g[i] < 120, 'impermeable green channel should be blocked/deflected');
  assert.ok(g[ti] > 0, 'impermeable green channel should deflect tangentially');
});

test('rigid global RGB tuple permeability is ignored in strict mode (requires per-edge tuples)', () => {
  const n = 24;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const i = 12 * n + 12;
  r[i] = 80;
  g[i] = 80;
  b[i] = 80;
  vx[i] = 1.0;

  const sim = {
    controls: { n },
    bodies: {
      rigid: [
        {
          x: 12,
          y: 12,
          theta: 0,
          verticesLocal: [
            { x: -4, y: 0 },
            { x: 4, y: 0 },
            { x: 4, y: 5 },
            { x: -4, y: 5 },
          ],
          // Broadcast tuple (not per-edge list).
          edgeBodyMode: EDGE_BODY_MODE.BLOCK,
          edgeDyeMode: [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT],
          edgePermeabilityRGB: [1, 0, 0],
        },
      ],
      soft: { nodes: [], springs: [] },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(r[i] < 80, 'strict mode should not treat global tuple as per-edge override; red should be blocked/deflected');
  assert.ok(g[i] < 80, 'green should be blocked');
  assert.ok(b[i] < 80, 'blue should be blocked');
});

test('exported rigid permeability traits survive import and drive runtime barrier behavior', () => {
  const n = 32;
  const size = n * n;
  const permeabilityMap = new Float32Array(n * n).fill(0);
  permeabilityMap[8 * n + 16] = 1; // top-edge midpoint permeable

  const mesh = {
    meta: { width: n, height: n },
    nodes: [
      { x: 8, y: 8, rigid: 1 },
      { x: 24, y: 8, rigid: 1 },
      { x: 24, y: 24, rigid: 1 },
      { x: 8, y: 24, rigid: 1 },
    ],
    triangles: [
      { kind: 'rigid', a: 0, b: 1, c: 2 },
      { kind: 'rigid', a: 0, b: 2, c: 3 },
    ],
  };

  const spec = createCreatureSpecFromMesh(mesh, {
    width: n,
    height: n,
    fields: { rigidPermeabilityMap: permeabilityMap },
    rigidPermeabilityThreshold: 0.5,
    includeAuthoring: false,
  });
  const bodies = buildBodiesFromCreatureSpec(spec, n, { massHeavy: 5 });
  assert.equal(bodies.rigid.length, 1);

  const rb = bodies.rigid[0];
  const edgeMask = rb.edgePermeabilityRGB || [];

  const verts = rigidVerticesWorld(rb);
  const midCell = (edgeIdx) => {
    const a = verts[edgeIdx];
    const b2 = verts[(edgeIdx + 1) % verts.length];
    const x = Math.max(0, Math.min(n - 1, Math.round((a.x + b2.x) * 0.5)));
    const y = Math.max(0, Math.min(n - 1, Math.round((a.y + b2.y) * 0.5)));
    return y * n + x;
  };
  const midPoint = (edgeIdx) => {
    const a = verts[edgeIdx];
    const b2 = verts[(edgeIdx + 1) % verts.length];
    return { x: (a.x + b2.x) * 0.5, y: (a.y + b2.y) * 0.5 };
  };

  const permeableEdges = edgeMask
    .map((rgb, i) => ({ rgb, i }))
    .filter(({ rgb }) => Array.isArray(rgb) && rgb[0] > 0 && rgb[1] > 0 && rgb[2] > 0)
    .map(({ i }) => i);
  const blockedEdge = edgeMask.findIndex((rgb) => Array.isArray(rgb) && rgb[0] <= 0 && rgb[1] <= 0 && rgb[2] <= 0);
  assert.ok(permeableEdges.length > 0, 'expected at least one permeable edge from painted map');
  assert.ok(blockedEdge >= 0, 'expected at least one blocked edge to compare against');

  const paintTarget = { x: 16, y: 8 };
  let permeableEdge = permeableEdges[0];
  let bestDist = Number.POSITIVE_INFINITY;
  for (const edgeIdx of permeableEdges) {
    const m = midPoint(edgeIdx);
    const d = Math.hypot(m.x - paintTarget.x, m.y - paintTarget.y);
    if (d < bestDist) {
      bestDist = d;
      permeableEdge = edgeIdx;
    }
  }
  assert.equal(permeableEdges.length, 1, `single painted hotspot should only open one segment, got ${permeableEdges.length}`);
  assert.ok(bestDist <= 0.01, `painted permeability should map to exact top-edge midpoint (16,8), nearest permeable midpoint distance=${bestDist.toFixed(3)}`);

  const oppositeTarget = { x: 16, y: 24 };
  let oppositeBestDist = Number.POSITIVE_INFINITY;
  for (const edgeIdx of permeableEdges) {
    const m = midPoint(edgeIdx);
    oppositeBestDist = Math.min(oppositeBestDist, Math.hypot(m.x - oppositeTarget.x, m.y - oppositeTarget.y));
  }
  assert.ok(oppositeBestDist >= 8, `permeability hotspot should stay localized and not leak to opposite edge, nearest opposite-edge permeable midpoint distance=${oppositeBestDist.toFixed(3)}`);

  const permeableCell = midCell(permeableEdge);
  const blockedCell = midCell(blockedEdge);

  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);
  r[permeableCell] = 120;
  g[permeableCell] = 90;
  b[permeableCell] = 60;
  r[blockedCell] = 120;
  g[blockedCell] = 90;
  b[blockedCell] = 60;
  // Give both cells strong edge-normal velocity so we can verify permeability only affects dye,
  // while contact blocking still projects out normal flow.
  const edgeNormal = (edgeIdx) => {
    const a = verts[edgeIdx];
    const b2 = verts[(edgeIdx + 1) % verts.length];
    const ex = b2.x - a.x;
    const ey = b2.y - a.y;
    const len = Math.max(1e-6, Math.hypot(ex, ey));
    return { nx: -ey / len, ny: ex / len };
  };
  const permeableNormal = edgeNormal(permeableEdge);
  const blockedNormal = edgeNormal(blockedEdge);
  vx[permeableCell] = permeableNormal.nx * 4.5;
  vy[permeableCell] = permeableNormal.ny * 4.5;
  vx[blockedCell] = blockedNormal.nx * 4.5;
  vy[blockedCell] = blockedNormal.ny * 4.5;

  const sim = {
    controls: { n },
    bodies: {
      rigid: [rb],
      soft: { nodes: [], springs: [] },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(r[permeableCell] >= 119.9, `permeable edge should preserve red dye (got ${r[permeableCell]})`);
  assert.ok(g[permeableCell] >= 89.9, `permeable edge should preserve green dye (got ${g[permeableCell]})`);
  assert.ok(b[permeableCell] >= 59.9, `permeable edge should preserve blue dye (got ${b[permeableCell]})`);

  assert.ok(r[blockedCell] < 120, `blocked edge should attenuate/deflect red dye (got ${r[blockedCell]})`);
  assert.ok(g[blockedCell] < 90, `blocked edge should attenuate/deflect green dye (got ${g[blockedCell]})`);
  assert.ok(b[blockedCell] < 60, `blocked edge should attenuate/deflect blue dye (got ${b[blockedCell]})`);

  const postPermeableNormal = vx[permeableCell] * permeableNormal.nx + vy[permeableCell] * permeableNormal.ny;
  const postBlockedNormal = vx[blockedCell] * blockedNormal.nx + vy[blockedCell] * blockedNormal.ny;
  assert.ok(postPermeableNormal < 4.5, `permeable dye edge should still block body-normal velocity (got ${postPermeableNormal})`);
  assert.ok(postBlockedNormal < 4.5, `blocked edge should block body-normal velocity (got ${postBlockedNormal})`);
});

test('exported rigid edge consume maps survive import and absorb configured dye channels', () => {
  const n = 32;
  const size = n * n;
  const consumeR = new Float32Array(n * n).fill(0);
  const consumeG = new Float32Array(n * n).fill(0);
  const consumeB = new Float32Array(n * n).fill(0);
  consumeR[8 * n + 16] = 1; // top-edge midpoint consumes red

  const mesh = {
    meta: { width: n, height: n },
    nodes: [
      { x: 8, y: 8, rigid: 1 },
      { x: 24, y: 8, rigid: 1 },
      { x: 24, y: 24, rigid: 1 },
      { x: 8, y: 24, rigid: 1 },
    ],
    triangles: [
      { kind: 'rigid', a: 0, b: 1, c: 2 },
      { kind: 'rigid', a: 0, b: 2, c: 3 },
    ],
  };

  const spec = createCreatureSpecFromMesh(mesh, {
    width: n,
    height: n,
    fields: {
      rigidEdgeConsumeMapR: consumeR,
      rigidEdgeConsumeMapG: consumeG,
      rigidEdgeConsumeMapB: consumeB,
    },
    rigidEdgeConsumeThreshold: 0.5,
    includeAuthoring: false,
  });

  const bodies = buildBodiesFromCreatureSpec(spec, n, { massHeavy: 5 });
  assert.equal(bodies.rigid.length, 1);
  const rb = bodies.rigid[0];

  const absorbEdge = (rb.edgeDyeMode || []).findIndex((rgb) => Array.isArray(rgb) && Number(rgb[0]) === EDGE_DYE_MODE.ABSORB);
  const deflectEdge = (rb.edgeDyeMode || []).findIndex((rgb) => Array.isArray(rgb) && Number(rgb[0]) === EDGE_DYE_MODE.DEFLECT);
  assert.ok(absorbEdge >= 0, 'expected at least one edge absorbing red dye from painted consume map');
  assert.ok(deflectEdge >= 0, 'expected at least one deflect edge for comparison');

  const verts = rigidVerticesWorld(rb);
  const midCell = (edgeIdx) => {
    const a = verts[edgeIdx];
    const b2 = verts[(edgeIdx + 1) % verts.length];
    const x = Math.max(0, Math.min(n - 1, Math.round((a.x + b2.x) * 0.5)));
    const y = Math.max(0, Math.min(n - 1, Math.round((a.y + b2.y) * 0.5)));
    return y * n + x;
  };

  const absorbCell = midCell(absorbEdge);
  const deflectCell = midCell(deflectEdge);

  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);
  r[absorbCell] = 120;
  r[deflectCell] = 120;

  const sim = {
    controls: { n },
    bodies: {
      rigid: [rb],
      soft: { nodes: [], springs: [] },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(r[absorbCell] < r[deflectCell], `absorb edge should consume more red dye than deflect edge (${r[absorbCell]} vs ${r[deflectCell]})`);
});

test('applyBodyEdgeFieldBarriers skips malformed rigid edges with non-finite vertices', () => {
  const n = 16;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const sim = {
    controls: { n },
    bodies: {
      rigid: [
        {
          x: 8,
          y: 8,
          theta: 0,
          verticesLocal: [
            { x: -2, y: 0 },
            { x: Number.NaN, y: 2 },
            { x: 2, y: 0 },
          ],
          edgeBodyMode: [EDGE_BODY_MODE.BLOCK, EDGE_BODY_MODE.BLOCK, EDGE_BODY_MODE.BLOCK],
          edgeDyeMode: [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT],
        },
      ],
      soft: { nodes: [], springs: [] },
    },
  };

  assert.doesNotThrow(() => applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld }));
});

test('applyBodyEdgeFieldBarriers skips malformed soft springs with missing node indices', () => {
  const n = 16;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const sim = {
    controls: { n },
    bodies: {
      rigid: [],
      soft: {
        nodes: [
          { x: 6, y: 8 },
          { x: 10, y: 8 },
        ],
        springs: [
          [0, 999, 4, EDGE_BODY_MODE.BLOCK, [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT]],
        ],
      },
    },
  };

  assert.doesNotThrow(() => applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld }));
});

test('collapsed soft spring edge is ignored (no point-barrier dye/velocity sink)', () => {
  const n = 16;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const i = 8 * n + 8;
  r[i] = 120;
  g[i] = 90;
  b[i] = 60;
  vx[i] = 1.5;
  vy[i] = -0.8;

  const sim = {
    controls: { n },
    bodies: {
      rigid: [],
      soft: {
        nodes: [
          { x: 8, y: 8 },
          { x: 8, y: 8 },
        ],
        springs: [
          [0, 1, 0, EDGE_BODY_MODE.BLOCK, [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT]],
        ],
      },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(Math.abs(r[i] - 120) < 1e-6, 'collapsed edge should not deflect/passively alter red dye');
  assert.ok(Math.abs(g[i] - 90) < 1e-6, 'collapsed edge should not absorb green dye');
  assert.ok(Math.abs(b[i] - 60) < 1e-6, 'collapsed edge should not alter blue dye');
  assert.ok(Math.abs(vx[i] - 1.5) < 1e-6, 'collapsed edge should not project velocity');
  assert.ok(Math.abs(vy[i] + 0.8) < 1e-6, 'collapsed edge should not project velocity');
});

test('collapsed rigid edge is ignored (does not attenuate local field)', () => {
  const n = 20;
  const size = n * n;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);
  const vx = new Float32Array(size);
  const vy = new Float32Array(size);

  const i = 10 * n + 10;
  r[i] = 100;
  g[i] = 100;
  b[i] = 100;
  vx[i] = 0.7;
  vy[i] = 1.2;

  const sim = {
    controls: { n },
    bodies: {
      rigid: [
        {
          x: 10,
          y: 10,
          theta: 0,
          // Two identical vertices => both traversed edges are collapsed.
          verticesLocal: [
            { x: 0, y: 0 },
            { x: 0, y: 0 },
          ],
          edgeBodyMode: [EDGE_BODY_MODE.BLOCK, EDGE_BODY_MODE.BLOCK],
          edgeDyeMode: [
            [EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.ABSORB],
            [EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.ABSORB],
          ],
        },
      ],
      soft: { nodes: [], springs: [] },
    },
  };

  applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld });

  assert.ok(Math.abs(r[i] - 100) < 1e-6, 'collapsed rigid edge should not absorb red dye');
  assert.ok(Math.abs(g[i] - 100) < 1e-6, 'collapsed rigid edge should not absorb green dye');
  assert.ok(Math.abs(b[i] - 100) < 1e-6, 'collapsed rigid edge should not absorb blue dye');
  assert.ok(Math.abs(vx[i] - 0.7) < 1e-6, 'collapsed rigid edge should not change vx');
  assert.ok(Math.abs(vy[i] - 1.2) < 1e-6, 'collapsed rigid edge should not change vy');
});
