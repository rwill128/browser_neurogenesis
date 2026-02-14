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

function polygonArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    s += p.x * q.y - q.x * p.y;
  }
  return Math.abs(s * 0.5);
}

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
  const hullArea = mesh.rigidPieces.reduce((sum, p) => sum + polygonArea(p.hull || []), 0);
  let maskArea = 0;
  for (const v of rigid) if (v >= 0.35) maskArea++;
  const areaRatio = hullArea / Math.max(1, maskArea);

  // This encodes desired quality bar for rigid decomposition from blurred fields.
  assert.ok(mesh.rigidPieces.length >= 1, 'expected at least one rigid contour piece');
  assert.ok(mesh.rigidPieces.length <= 8, `too many rigid pieces: ${mesh.rigidPieces.length}`);
  assert.ok(totalHullVerts <= 40, `too many hull vertices across rigid pieces: ${totalHullVerts}`);
  assert.ok(mesh.rigidWelds.length <= 8, `too many rigid weld seams after decomposition: ${mesh.rigidWelds.length}`);
  assert.ok(areaRatio >= 0.8 && areaRatio <= 1.2, `rigid contour area drift too large: ratio=${areaRatio.toFixed(3)}`);
  assert.ok(isConcave(mesh.rigidPieces[0].hull), 'expected concave rigid contour for L-shape field');
});
