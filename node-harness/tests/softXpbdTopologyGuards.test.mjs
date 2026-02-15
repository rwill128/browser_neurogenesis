import test from 'node:test';
import assert from 'node:assert/strict';

import {
  sanitizeSoftSprings,
  ensureLambdaCacheSize,
  buildSoftClusterBoundaryLoops,
} from '../../sim-server/public/soft-xpbd.js';

test('sanitizeSoftSprings drops malformed/out-of-range entries deterministically', () => {
  const raw = [
    [0, 1, 2.5, 1, [1, 1, 1]],   // keep
    [6, 1, 1.0, 1, [1, 1, 1]],   // drop (a out of range)
    [1, -2, 1.0, 1, [1, 1, 1]],  // drop (b out of range)
    [1, 1, 1.0, 1, [1, 1, 1]],   // drop (degenerate)
    ['x', 2, 1.0, 1, [1, 1, 1]], // drop (non-integer)
    { a: 0, b: 2, rest: 1.2, edgeBodyMode: 1, edgeDyeMode: [1,1,1] }, // drop (non-array strict mode)
    [2, 3, NaN, 0, [0, 0, 0]],   // keep with rest fallback
  ];

  const out = sanitizeSoftSprings(raw, 5, { edgeBodyPass: 0, edgeBodyBlock: 1, restFloor: 1e-3 });
  assert.equal(out.springs.length, 2);
  assert.equal(out.dropped, 5);

  assert.deepEqual(out.springs[0].slice(0, 4), [0, 1, 2.5, 1]);
  assert.deepEqual(out.springs[1].slice(0, 2), [2, 3]);
  assert.ok(out.springs[1][2] >= 1e-3, 'rest should be clamped to finite floor');
});

test('buildSoftClusterBoundaryLoops prefers boundary BLOCK cycle and ignores interior nodes', () => {
  const nodes = [
    { x: 0, y: 0, clusterId: 0 },  // 0
    { x: 2, y: 0, clusterId: 0 },  // 1
    { x: 2, y: 2, clusterId: 0 },  // 2
    { x: 0, y: 2, clusterId: 0 },  // 3
    { x: 1, y: 1, clusterId: 0 },  // 4 interior node
  ];

  const springs = [
    [0, 1, 2, 1],
    [1, 2, 2, 1],
    [2, 3, 2, 1],
    [3, 0, 2, 1],
    [0, 4, 1.4, 0],
    [1, 4, 1.4, 0],
    [2, 4, 1.4, 0],
    [3, 4, 1.4, 0],
  ];

  const loops = buildSoftClusterBoundaryLoops(nodes, springs, { blockMode: 1 });
  assert.equal(loops.length, 1);
  assert.equal(loops[0].source, 'boundary');
  assert.equal(loops[0].indices.length, 4);
  assert.ok(!loops[0].indices.includes(4), 'interior node should not be part of boundary area loop');
});

test('buildSoftClusterBoundaryLoops falls back to convex hull when boundary cycle is unavailable', () => {
  const nodes = [
    { x: 0, y: 0, clusterId: 1 },
    { x: 2, y: 0, clusterId: 1 },
    { x: 2, y: 2, clusterId: 1 },
    { x: 0, y: 2, clusterId: 1 },
    { x: 1, y: 1, clusterId: 1 },
  ];

  // No BLOCK perimeter springs => force hull fallback.
  const springs = [
    [0, 4, 1.4, 0],
    [1, 4, 1.4, 0],
    [2, 4, 1.4, 0],
    [3, 4, 1.4, 0],
  ];

  const loops = buildSoftClusterBoundaryLoops(nodes, springs, { blockMode: 1 });
  assert.equal(loops.length, 1);
  assert.equal(loops[0].source, 'hull');
  assert.equal(loops[0].indices.length, 4);
  assert.ok(!loops[0].indices.includes(4), 'hull should discard interior node');
});

test('ensureLambdaCacheSize preserves warm-start values across resize and sanitizes non-finite entries', () => {
  const prev = new Float32Array([1.25, -2.5, Number.NaN, 99]);
  const next = ensureLambdaCacheSize(prev, 6, 20);
  assert.equal(next.length, 6);
  assert.equal(next[0], 1.25);
  assert.equal(next[1], -2.5);
  assert.equal(next[2], 0);
  assert.equal(next[3], 20);
  assert.equal(next[4], 0);
  assert.equal(next[5], 0);

  const shrunk = ensureLambdaCacheSize(next, 2, 20);
  assert.deepEqual([...shrunk], [1.25, -2.5]);
});
