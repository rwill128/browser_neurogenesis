import test from 'node:test';
import assert from 'node:assert/strict';

import { computeVectorRms, computeRigidAlignedPoseResidual } from '../../sim-server/public/soft-deformation-metrics.js';

function makeLocal(points, cx = 0, cy = 0) {
  const out = new Float32Array(points.length * 2);
  for (let i = 0; i < points.length; i++) {
    out[i * 2] = points[i][0] - cx;
    out[i * 2 + 1] = points[i][1] - cy;
  }
  return out;
}

test('rigid-aligned pose residual is translation/rotation invariant', () => {
  const refPts = [
    [-2, -1],
    [2, -1],
    [2, 1],
    [-2, 1],
  ];
  const ref = makeLocal(refPts);

  const theta = Math.PI * 0.37;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const tx = 13.2;
  const ty = -8.7;
  const cur = new Float32Array(ref.length);
  for (let i = 0; i < refPts.length; i++) {
    const x = refPts[i][0];
    const y = refPts[i][1];
    cur[i * 2] = x * c - y * s + tx;
    cur[i * 2 + 1] = x * s + y * c + ty;
  }

  // Convert to centroid-relative current coordinates (what GPU lab uses).
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < refPts.length; i++) {
    cx += cur[i * 2];
    cy += cur[i * 2 + 1];
  }
  cx /= refPts.length;
  cy /= refPts.length;
  for (let i = 0; i < refPts.length; i++) {
    cur[i * 2] -= cx;
    cur[i * 2 + 1] -= cy;
  }

  const residual = computeRigidAlignedPoseResidual(ref, cur, computeVectorRms(ref));
  assert.ok(residual);
  assert.ok(residual.normalizedRms < 1e-6, `expected near-zero RMS residual, got ${residual.normalizedRms}`);
  assert.ok(residual.normalizedMax < 1e-6, `expected near-zero max residual, got ${residual.normalizedMax}`);
});

test('rigid-aligned pose residual rises under true non-rigid deformation', () => {
  const ref = new Float32Array([
    -2, -1,
    2, -1,
    2, 1,
    -2, 1,
  ]);
  const cur = new Float32Array(ref);

  // Apply a non-rigid deformation: pull one corner outward.
  cur[4] += 1.8; // third point x
  cur[5] += 1.2; // third point y

  const residual = computeRigidAlignedPoseResidual(ref, cur, computeVectorRms(ref));
  assert.ok(residual);
  assert.ok(residual.normalizedRms > 0.15, `expected noticeable RMS residual, got ${residual.normalizedRms}`);
  assert.ok(residual.normalizedMax > 0.3, `expected noticeable max residual, got ${residual.normalizedMax}`);
});
