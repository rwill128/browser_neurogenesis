import test from 'node:test';
import assert from 'node:assert/strict';
import { compileFieldToMesh } from '../../sim-server/public/field-to-structure-core.js';

function pointInPolygon(px, py, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    const intersect = ((yi > py) !== (yj > py))
      && (px < ((xj - xi) * (py - yi)) / Math.max(1e-9, (yj - yi)) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

test('field-to-structure compiler creates rigid and soft triangles from painted regions', () => {
  const w = 16, h = 16;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // rigid patch on left
  for (let y = 3; y <= 10; y++) {
    for (let x = 2; x <= 6; x++) rigid[y * w + x] = 1;
  }
  // soft patch on right
  for (let y = 4; y <= 11; y++) {
    for (let x = 9; x <= 13; x++) soft[y * w + x] = 1;
  }

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.2,
    density: 2,
  });

  assert.ok(mesh.nodes.length > 0);
  assert.ok(mesh.triangles.length > 0);
  assert.ok(mesh.meta.rigidTriangles > 0);
  assert.ok(mesh.meta.softTriangles > 0);
});

test('higher threshold reduces generated triangles', () => {
  const w = 16, h = 16;
  const rigid = new Float32Array(w * h).fill(0.4);
  const soft = new Float32Array(w * h).fill(0.0);

  const low = compileFieldToMesh({ width: w, height: h, rigidField: rigid, softField: soft, threshold: 0.2, density: 2 });
  const high = compileFieldToMesh({ width: w, height: h, rigidField: rigid, softField: soft, threshold: 0.6, density: 2 });

  assert.ok(low.triangles.length > high.triangles.length);
});

test('connectivity largest mode drops disconnected islands', () => {
  const w = 20, h = 20;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // big island
  for (let y = 3; y <= 10; y++) for (let x = 3; x <= 10; x++) rigid[y * w + x] = 1;
  // tiny island
  for (let y = 14; y <= 15; y++) for (let x = 14; x <= 15; x++) rigid[y * w + x] = 1;

  const raw = compileFieldToMesh({ width: w, height: h, rigidField: rigid, softField: soft, threshold: 0.2, density: 2, connectivityMode: 'none' });
  const connected = compileFieldToMesh({ width: w, height: h, rigidField: rigid, softField: soft, threshold: 0.2, density: 2, connectivityMode: 'largest' });

  assert.ok(raw.triangles.length > connected.triangles.length);
  assert.ok(connected.meta.components >= 2);
  assert.equal(connected.meta.keptComponents, 1);
});

test('connectivity largest mode enforces single connected body across rigid+soft', () => {
  const w = 28, h = 20;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // dominant rigid component
  for (let y = 2; y <= 10; y++) for (let x = 2; x <= 12; x++) rigid[y * w + x] = 1;

  // disconnected soft component should be dropped under strict single-body policy
  for (let y = 3; y <= 10; y++) for (let x = 20; x <= 25; x++) soft[y * w + x] = 1;

  const connected = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.2,
    density: 2,
    connectivityMode: 'largest',
  });

  assert.equal(connected.meta.keptComponents, 1);
  assert.ok(connected.meta.rigidTriangles > 0);
  assert.equal(connected.meta.softTriangles, 0);
});

test('compiler exposes rigid decomposition pieces and welds at compile time', () => {
  const w = 24, h = 24;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // concave-ish rigid "L" shape
  for (let y = 4; y <= 16; y++) for (let x = 4; x <= 9; x++) rigid[y * w + x] = 1;
  for (let y = 12; y <= 16; y++) for (let x = 4; x <= 16; x++) rigid[y * w + x] = 1;

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.2,
    density: 2,
    connectivityMode: 'largest',
  });

  assert.ok(Array.isArray(mesh.rigidPieces));
  assert.ok(Array.isArray(mesh.rigidWelds));
  assert.ok(mesh.rigidPieces.length >= 1);
  for (const p of mesh.rigidPieces) {
    assert.ok(Array.isArray(p.hull));
    assert.ok(p.hull.length >= 3);
  }
  for (const wld of mesh.rigidWelds) {
    assert.ok(Number.isInteger(wld.a) && wld.a >= 0 && wld.a < mesh.rigidPieces.length);
    assert.ok(Number.isInteger(wld.b) && wld.b >= 0 && wld.b < mesh.rigidPieces.length);
  }
});

test('soft triangles do not overlap rigid contour in rigid+soft overlap zones', () => {
  const w = 64, h = 64;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // rigid L
  for (let y = 10; y <= 48; y++) for (let x = 10; x <= 20; x++) rigid[y * w + x] = 1;
  for (let y = 36; y <= 48; y++) for (let x = 10; x <= 48; x++) rigid[y * w + x] = 1;

  // soft patch intentionally overlapping lower-right interior of rigid L
  for (let y = 28; y <= 48; y++) for (let x = 18; x <= 42; x++) soft[y * w + x] = 1;

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 6,
    connectivityMode: 'largest',
  });

  assert.ok(mesh.rigidPieces.length >= 1);
  const rigidHulls = mesh.rigidPieces.map((p) => p.hull).filter((h) => Array.isArray(h) && h.length >= 3);

  for (const t of mesh.triangles) {
    if (t.kind !== 'soft') continue;
    const a = mesh.nodes[t.a];
    const b = mesh.nodes[t.b];
    const c = mesh.nodes[t.c];
    const probes = [
      { x: a.x, y: a.y },
      { x: b.x, y: b.y },
      { x: c.x, y: c.y },
      { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 },
      { x: (b.x + c.x) * 0.5, y: (b.y + c.y) * 0.5 },
      { x: (c.x + a.x) * 0.5, y: (c.y + a.y) * 0.5 },
      { x: (a.x + b.x + c.x) / 3, y: (a.y + b.y + c.y) / 3 },
    ];

    for (const hull of rigidHulls) {
      for (const p of probes) {
        assert.equal(pointInPolygon(p.x, p.y, hull), false, 'soft triangle probe overlapped rigid contour');
      }
    }
  }

  assert.ok((mesh.meta.softOverlapTrimmed || 0) > 0, 'expected overlap-trim guardrail to remove some soft triangles');
});
