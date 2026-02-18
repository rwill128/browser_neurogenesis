/**
 * GPU-only runtime collision iteration orchestrator.
 * Migrates the collision iteration stepping schedule into an isolated module so
 * gpu-only runtime owns rigid↔rigid, rigid↔soft, soft↔soft, and boundary passes
 * without touching the baseline/default path.
 */

function buildHybridAttachedByRigidMap(hybrid, rigidCount, softNodeCount) {
  const attachedByRigid = new Map();
  if (!Array.isArray(hybrid) || hybrid.length === 0) return attachedByRigid;

  for (const joint of hybrid) {
    const rigidIndex = Number(joint?.rigidIndex) | 0;
    const nodeIndex = Number(joint?.nodeIndex) | 0;
    if (rigidIndex < 0 || rigidIndex >= rigidCount) continue;
    if (nodeIndex < 0 || nodeIndex >= softNodeCount) continue;
    if (!attachedByRigid.has(rigidIndex)) attachedByRigid.set(rigidIndex, new Set());
    attachedByRigid.get(rigidIndex).add(nodeIndex);
  }

  return attachedByRigid;
}

export async function runCollisionIterationsGpuOnly({
  rigidBodies,
  soft,
  hybrid,
  rigidContactDebug,
  collisionIterations = 2,
  rigidRigidSlop = 0.32,
  rigidSoftNodeSlop = 0.18,
  rigidSoftEdgeSlop = 0.16,
  softNodeNodeSlop = 0.22,
  softNodeEdgeSlop = 0.12,
  edgeBodyModeBlock,
  n,
  rigidBounce = 0.84,
  softBounce = 0.78,
  resolveRigidRigidCollisionPassGpuOnly,
  resolveRigidSoftCollisionPassGpuOnly,
  resolveSoftSoftCollisionPassGpuOnly,
  applyCollisionBoundaryPassGpuOnly,
  resolveRigidVsRigidPolygonCollision,
  resolveCircleCollision,
  resolveSoftNodeVsSoftEdgeCollision,
  applyBounceBoundary,
  wgslOffload,
}) {
  const rigid = Array.isArray(rigidBodies) ? rigidBodies : [];
  const softBody = soft || { nodes: [], springs: [] };

  const hybridAttachedByRigid = buildHybridAttachedByRigidMap(
    hybrid,
    rigid.length,
    Array.isArray(softBody.nodes) ? softBody.nodes.length : 0,
  );

  let boundaryRuntime = { mode: 'cpu', reason: 'not-run' };

  for (let iter = 0; iter < collisionIterations; iter++) {
    resolveRigidRigidCollisionPassGpuOnly({
      rigidBodies: rigid,
      slop: rigidRigidSlop,
      contacts: rigidContactDebug,
      iter,
      phase: 'pre-soft',
      resolveRigidVsRigidPolygonCollision,
    });

    await resolveRigidSoftCollisionPassGpuOnly({
      rigidBodies: rigid,
      soft: softBody,
      hybridAttachedByRigid,
      edgeBodyModeBlock,
      nodeSlop: rigidSoftNodeSlop,
      edgeSlop: rigidSoftEdgeSlop,
      wgslOffload,
    });

    resolveSoftSoftCollisionPassGpuOnly({
      soft: softBody,
      resolveCircleCollision,
      resolveSoftNodeVsSoftEdgeCollision,
      edgeBodyModeBlock,
      nodeNodeSlop: softNodeNodeSlop,
      nodeEdgeSlop: softNodeEdgeSlop,
    });

    resolveRigidRigidCollisionPassGpuOnly({
      rigidBodies: rigid,
      slop: rigidRigidSlop,
      contacts: rigidContactDebug,
      iter,
      phase: 'post-soft',
      resolveRigidVsRigidPolygonCollision,
    });

    boundaryRuntime = (await applyCollisionBoundaryPassGpuOnly({
      rigidBodies: rigid,
      soft: softBody,
      n,
      rigidBounce,
      softBounce,
      applyBounceBoundary,
      wgslOffload,
    })) || { mode: 'cpu-fallback', reason: 'unknown' };
  }

  return { hybridAttachedByRigid, boundaryRuntime };
}
