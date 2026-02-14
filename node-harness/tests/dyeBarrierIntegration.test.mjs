import test from 'node:test';
import assert from 'node:assert/strict';
import { EDGE_DYE_MODE, EDGE_BODY_MODE, normalizeEdgeDyeModeRGB, applyBodyEdgeFieldBarriers } from '../../sim-server/public/dye-barrier.js';

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
