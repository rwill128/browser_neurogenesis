import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const file = path.resolve('sim-server/public/runtime-solvers/stepSoftCollisionGpuOnly.js');
const src = fs.readFileSync(file, 'utf8');

test('soft collision gpu-only source includes WGSL node-node authoritative path and fallback routes', () => {
  assert.match(src, /const softNodeNodeCollisionWgsl\s*=\s*\/\* wgsl \*\//);
  assert.match(src, /wgsl-soft-node-node-authoritative/);
  assert.match(src, /cpu-soft-node-node-fallback-nonfinite/);
  assert.match(src, /cpu-soft-node-node-fallback-error/);
});

test('soft collision gpu-only source builds deterministic soft node-edge candidate layout for WGSL next stage bring-up', () => {
  assert.match(src, /function buildSoftNodeEdgeCandidateLayout/);
  assert.match(src, /cpu-soft-node-edge-candidate-layout/);
  assert.match(src, /wgsl-soft-node-edge-collision-pending/);
  assert.match(src, /lastSoftNodeEdgeCandidateLayoutSignature/);
});
