import test from 'node:test';
import assert from 'node:assert/strict';

import { applyHybridAttachmentConstraintsGpuOnly } from '../../sim-server/public/runtime-solvers/stepHybridConstraintsGpuOnly.js';

function rigidVertexWorld(rb, idx) {
  const verts = rb?.verticesLocal || [];
  const v = verts[idx] || { x: 0, y: 0 };
  const c = Math.cos(rb.a || 0);
  const s = Math.sin(rb.a || 0);
  return {
    x: (rb.x || 0) + v.x * c - v.y * s,
    y: (rb.y || 0) + v.x * s + v.y * c,
  };
}

function applyHybridAttachmentConstraintsBaseline({ rigidBodies, soft, hybrid, dtNorm, iterations = 5 }) {
  for (let iter = 0; iter < iterations; iter++) {
    for (const h of (hybrid || [])) {
      const rb = rigidBodies[h.rigidIndex];
      const node = soft.nodes[h.nodeIndex];
      if (!rb || !node) continue;
      const va = rigidVertexWorld(rb, h.vertexA);
      const vb = rigidVertexWorld(rb, h.vertexB);
      const pairs = [[va, h.restA], [vb, h.restB]];
      for (const [anchor, rest] of pairs) {
        const dx = node.x - anchor.x;
        const dy = node.y - anchor.y;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const err = (d - rest) * 0.74;
        const nx = dx / d;
        const ny = dy / d;
        node.vx -= nx * err * 0.052 * dtNorm;
        node.vy -= ny * err * 0.052 * dtNorm;
        rb.vx += nx * err * 0.0075 * dtNorm;
        rb.vy += ny * err * 0.0075 * dtNorm;
        rb.omega = (rb.omega || 0) + (nx * ny) * err * 0.00075 * dtNorm;
      }
    }
  }
}

test('gpu-only hybrid attachment pass matches baseline weld-like rigid-soft constraint stepping', () => {
  const baselineRigid = [{
    x: 1.6,
    y: -0.8,
    a: 0.34,
    vx: 0.12,
    vy: -0.06,
    omega: 0.03,
    verticesLocal: [
      { x: -0.7, y: 0.1 },
      { x: 0.65, y: -0.25 },
      { x: 0.15, y: 0.75 },
    ],
  }];
  const baselineSoft = {
    nodes: [
      { x: 1.9, y: -0.25, vx: -0.08, vy: 0.1 },
      { x: 1.2, y: -0.9, vx: 0.02, vy: -0.03 },
    ],
  };
  const hybrid = [
    { rigidIndex: 0, nodeIndex: 0, vertexA: 0, vertexB: 1, restA: 0.68, restB: 0.86 },
    { rigidIndex: 0, nodeIndex: 1, vertexA: 1, vertexB: 2, restA: 0.75, restB: 0.92 },
  ];

  const gpuRigid = structuredClone(baselineRigid);
  const gpuSoft = structuredClone(baselineSoft);

  const dtNorm = 0.83;
  applyHybridAttachmentConstraintsBaseline({
    rigidBodies: baselineRigid,
    soft: baselineSoft,
    hybrid,
    dtNorm,
    iterations: 5,
  });

  applyHybridAttachmentConstraintsGpuOnly({
    rigidBodies: gpuRigid,
    soft: gpuSoft,
    hybrid,
    rigidVertexWorld,
    dtNorm,
    iterations: 5,
  });

  assert.deepEqual(gpuSoft, baselineSoft, 'gpu-only hybrid attachment pass should mutate soft nodes identically to baseline loops');
  assert.deepEqual(gpuRigid, baselineRigid, 'gpu-only hybrid attachment pass should mutate rigid bodies identically to baseline loops');
});
