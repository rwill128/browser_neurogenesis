/**
 * GPU-only runtime rigid↔soft collision pass.
 * Keeps rigid-vs-soft collision stepping isolated for the gpu-only runtime path
 * while preserving baseline/default behavior and call contracts.
 */
export function resolveRigidSoftCollisionPassGpuOnly({
  rigidBodies,
  soft,
  hybridAttachedByRigid,
  resolveRigidVsSoftNodeCollision,
  resolveRigidVsSoftEdgeCollision,
  edgeBodyModeBlock,
  nodeSlop = 0.18,
  edgeSlop = 0.16,
}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return;
  if (!soft || !Array.isArray(soft.nodes) || !Array.isArray(soft.springs)) return;
  if (typeof resolveRigidVsSoftNodeCollision !== 'function' || typeof resolveRigidVsSoftEdgeCollision !== 'function') {
    throw new Error('gpu-only rigid-soft collision pass requires rigid-vs-soft collision callbacks');
  }

  for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
    const rb = rigidBodies[rbi];
    const attachedNodeSet = hybridAttachedByRigid?.get?.(rbi) || null;

    for (let ni = 0; ni < soft.nodes.length; ni++) {
      if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
      const sn = soft.nodes[ni];
      resolveRigidVsSoftNodeCollision(rb, sn, null, nodeSlop);
    }

    for (const [i, j, _rest, edgeBodyMode] of soft.springs) {
      if (edgeBodyMode !== edgeBodyModeBlock) continue;
      if (attachedNodeSet && (attachedNodeSet.has(i) || attachedNodeSet.has(j))) continue;
      resolveRigidVsSoftEdgeCollision(rb, soft.nodes[i], soft.nodes[j], edgeSlop);
    }
  }
}
