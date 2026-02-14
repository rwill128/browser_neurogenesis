import test from 'node:test';
import assert from 'node:assert/strict';
import { compileFieldToMesh } from '../../sim-server/public/field-to-structure-core.js';

function paintStroke(field, w, h, x0, y0, x1, y1, radius) {
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - radius - 1));
  const maxX = Math.min(w - 1, Math.ceil(Math.max(x0, x1) + radius + 1));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - radius - 1));
  const maxY = Math.min(h - 1, Math.ceil(Math.max(y0, y1) + radius + 1));
  const vx = x1 - x0;
  const vy = y1 - y0;
  const vv = Math.max(1e-6, vx * vx + vy * vy);

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5;
      const py = y + 0.5;
      const t = Math.max(0, Math.min(1, ((px - x0) * vx + (py - y0) * vy) / vv));
      const cx = x0 + vx * t;
      const cy = y0 + vy * t;
      const d = Math.hypot(px - cx, py - cy);
      if (d > radius) continue;
      const val = 1 - d / radius;
      const i = y * w + x;
      if (val > field[i]) field[i] = val;
    }
  }
}

test('rigid decomposition: simple rectangle stays compact', () => {
  const w = 64, h = 64;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  for (let y = 16; y <= 40; y++) {
    for (let x = 20; x <= 44; x++) rigid[y * w + x] = 1;
  }

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 4,
    connectivityMode: 'largest',
  });

  assert.ok(mesh.rigidPieces.length <= 3, `expected compact decomposition, got ${mesh.rigidPieces.length} pieces`);
});

test('rigid decomposition quality: brushed L-shape should avoid excessive tiny pieces', () => {
  const w = 128, h = 128;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  paintStroke(rigid, w, h, 40, 30, 40, 90, 18);
  paintStroke(rigid, w, h, 40, 72, 92, 72, 18);

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 7,
    connectivityMode: 'largest',
  });

  const totalHullVerts = mesh.rigidPieces.reduce((sum, p) => sum + (p.hull?.length || 0), 0);

  // This encodes desired quality bar for rigid decomposition from blurred fields.
  assert.ok(mesh.rigidPieces.length <= 8, `too many rigid pieces: ${mesh.rigidPieces.length}`);
  assert.ok(totalHullVerts <= 40, `too many hull vertices across rigid pieces: ${totalHullVerts}`);
  assert.ok(mesh.rigidWelds.length <= 8, `too many rigid weld seams after decomposition: ${mesh.rigidWelds.length}`);
});
