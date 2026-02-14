import test from 'node:test';
import assert from 'node:assert/strict';
import { compileFieldToMesh } from '../../sim-server/public/field-to-structure-core.js';

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
