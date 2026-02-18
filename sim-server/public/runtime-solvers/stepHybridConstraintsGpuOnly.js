/**
 * GPU-only runtime rigid↔soft hybrid attachment constraint pass.
 *
 * Isolates weld-like rigid-soft hybrid spring stepping behind the gpu-only
 * runtime solver path while preserving baseline/default behavior contracts.
 */

function clampFinite(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export function buildHybridAttachmentWgslPrep({ rigidBodies, soft, hybrid, rigidVertexWorld }) {
  const attachments = Array.isArray(hybrid) ? hybrid : [];
  const count = attachments.length;

  const nodeIndex = new Uint32Array(count);
  const rigidIndex = new Uint32Array(count);
  const anchorAX = new Float32Array(count);
  const anchorAY = new Float32Array(count);
  const anchorBX = new Float32Array(count);
  const anchorBY = new Float32Array(count);
  const restA = new Float32Array(count);
  const restB = new Float32Array(count);

  let validCount = 0;
  for (let i = 0; i < count; i++) {
    const h = attachments[i];
    const rb = rigidBodies[h?.rigidIndex];
    const node = soft?.nodes?.[h?.nodeIndex];
    if (!rb || !node) continue;

    const va = rigidVertexWorld(rb, h.vertexA);
    const vb = rigidVertexWorld(rb, h.vertexB);

    nodeIndex[i] = h.nodeIndex >>> 0;
    rigidIndex[i] = h.rigidIndex >>> 0;
    anchorAX[i] = clampFinite(va?.x);
    anchorAY[i] = clampFinite(va?.y);
    anchorBX[i] = clampFinite(vb?.x);
    anchorBY[i] = clampFinite(vb?.y);
    restA[i] = Math.max(0, clampFinite(h.restA));
    restB[i] = Math.max(0, clampFinite(h.restB));
    validCount += 1;
  }

  const layout = {
    nodeIndex,
    rigidIndex,
    anchorAX,
    anchorAY,
    anchorBX,
    anchorBY,
    restA,
    restB,
  };
  layout.byteLength = Object.values(layout).reduce((sum, arr) => sum + (arr?.byteLength || 0), 0);

  return {
    plan: {
      attachmentCount: count,
      validAttachmentCount: validCount,
      rigidBodyCount: Array.isArray(rigidBodies) ? rigidBodies.length : 0,
      softNodeCount: Array.isArray(soft?.nodes) ? soft.nodes.length : 0,
    },
    layout,
  };
}

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
  wgslOffload,
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

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const { plan, layout } = buildHybridAttachmentWgslPrep({ rigidBodies, soft, hybrid, rigidVertexWorld });
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.lastPreparedAttachmentCount = plan.attachmentCount;
    wgslOffload.state.lastPreparedValidAttachmentCount = plan.validAttachmentCount;
    wgslOffload.state.lastPreparedRigidBodyCount = plan.rigidBodyCount;
    wgslOffload.state.lastPreparedSoftNodeCount = plan.softNodeCount;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared';
    wgslOffload.state.lastError = null;
  }

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
