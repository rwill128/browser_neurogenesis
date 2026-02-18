/**
 * GPU-only runtime rigid↔soft hybrid attachment constraint pass.
 *
 * Isolates weld-like rigid-soft hybrid spring stepping behind the gpu-only
 * runtime solver path while preserving baseline/default behavior contracts.
 */
export function applyHybridAttachmentConstraintsGpuOnly({
  rigidBodies,
  soft,
  hybrid,
  rigidVertexWorld,
  dtNorm,
  iterations = 5,
  nodeErrorGain = 0.74,
  nodeImpulseScale = 0.052,
  rigidImpulseScale = 0.0075,
  rigidAngularScale = 0.00075,
}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return;
  if (!soft || !Array.isArray(soft.nodes) || soft.nodes.length === 0) return;
  if (!Array.isArray(hybrid) || hybrid.length === 0) return;
  if (typeof rigidVertexWorld !== 'function') {
    throw new Error('gpu-only hybrid constraint pass requires rigidVertexWorld callback');
  }

  const safeDtNorm = Number.isFinite(dtNorm) ? dtNorm : 0;
  const totalIterations = Math.max(0, Number(iterations) | 0);
  if (totalIterations <= 0 || safeDtNorm === 0) return;

  for (let iter = 0; iter < totalIterations; iter++) {
    for (const h of hybrid) {
      const rb = rigidBodies[h?.rigidIndex];
      const node = soft.nodes[h?.nodeIndex];
      if (!rb || !node) continue;

      const va = rigidVertexWorld(rb, h.vertexA);
      const vb = rigidVertexWorld(rb, h.vertexB);
      const pairs = [[va, h.restA], [vb, h.restB]];
      for (const [anchor, rest] of pairs) {
        const dx = node.x - anchor.x;
        const dy = node.y - anchor.y;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const err = (d - rest) * nodeErrorGain;
        const nx = dx / d;
        const ny = dy / d;

        node.vx -= nx * err * nodeImpulseScale * safeDtNorm;
        node.vy -= ny * err * nodeImpulseScale * safeDtNorm;

        rb.vx += nx * err * rigidImpulseScale * safeDtNorm;
        rb.vy += ny * err * rigidImpulseScale * safeDtNorm;
        rb.omega = (rb.omega || 0) + (nx * ny) * err * rigidAngularScale * safeDtNorm;
      }
    }
  }
}
