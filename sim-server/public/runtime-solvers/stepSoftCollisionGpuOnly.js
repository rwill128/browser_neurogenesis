/**
 * GPU-only runtime soft↔soft collision pass.
 * Isolates soft node-vs-node and soft node-vs-foreign-soft-edge collision stepping
 * for the gpu-only runtime path while preserving baseline/default behavior contracts.
 */
export function resolveSoftSoftCollisionPassGpuOnly({
  soft,
  resolveCircleCollision,
  resolveSoftNodeVsSoftEdgeCollision,
  edgeBodyModeBlock,
  nodeNodeSlop = 0.22,
  nodeEdgeSlop = 0.12,
}) {
  if (!soft || !Array.isArray(soft.nodes) || !Array.isArray(soft.springs)) return;
  if (typeof resolveCircleCollision !== 'function' || typeof resolveSoftNodeVsSoftEdgeCollision !== 'function') {
    throw new Error('gpu-only soft collision pass requires soft collision callbacks');
  }

  const nodes = soft.nodes;
  const springs = soft.springs;

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      resolveCircleCollision(nodes[i], nodes[j], nodeNodeSlop);
    }
  }

  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni];
    for (const [i, j, _rest, edgeBodyMode] of springs) {
      if (edgeBodyMode !== edgeBodyModeBlock) continue;
      if (i === ni || j === ni) continue;
      const a = nodes[i];
      const b = nodes[j];
      if (a.clusterId === node.clusterId && b.clusterId === node.clusterId) continue;
      resolveSoftNodeVsSoftEdgeCollision(node, a, b, nodeEdgeSlop);
    }
  }
}
