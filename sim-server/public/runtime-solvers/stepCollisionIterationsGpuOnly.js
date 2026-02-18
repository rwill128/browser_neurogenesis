/**
 * GPU-only runtime collision iteration orchestrator.
 * Migrates the collision iteration stepping schedule into an isolated module so
 * gpu-only runtime owns rigid↔rigid, rigid↔soft, soft↔soft, and boundary passes
 * without touching the baseline/default path.
 */

export async function runCollisionIterationsGpuOnly({
  rigidBodies,
  soft,
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

  let boundaryRuntime = { mode: 'cpu', reason: 'not-run' };

  const rigidRigidWgslOffload = wgslOffload?.rigidRigid || wgslOffload;
  const rigidSoftWgslOffload = wgslOffload?.rigidSoft || wgslOffload;
  const softSoftWgslOffload = wgslOffload?.softSoft || wgslOffload;
  const collisionBoundaryWgslOffload = wgslOffload?.boundary || wgslOffload;

  for (let iter = 0; iter < collisionIterations; iter++) {
    await resolveRigidRigidCollisionPassGpuOnly({
      rigidBodies: rigid,
      slop: rigidRigidSlop,
      contacts: rigidContactDebug,
      iter,
      phase: 'pre-soft',
      resolveRigidVsRigidPolygonCollision,
      wgslOffload: rigidRigidWgslOffload,
    });

    await resolveRigidSoftCollisionPassGpuOnly({
      rigidBodies: rigid,
      soft: softBody,
      edgeBodyModeBlock,
      nodeSlop: rigidSoftNodeSlop,
      edgeSlop: rigidSoftEdgeSlop,
      wgslOffload: rigidSoftWgslOffload,
    });

    await resolveSoftSoftCollisionPassGpuOnly({
      soft: softBody,
      resolveCircleCollision,
      resolveSoftNodeVsSoftEdgeCollision,
      edgeBodyModeBlock,
      nodeNodeSlop: softNodeNodeSlop,
      nodeEdgeSlop: softNodeEdgeSlop,
      wgslOffload: softSoftWgslOffload,
    });

    await resolveRigidRigidCollisionPassGpuOnly({
      rigidBodies: rigid,
      slop: rigidRigidSlop,
      contacts: rigidContactDebug,
      iter,
      phase: 'post-soft',
      resolveRigidVsRigidPolygonCollision,
      wgslOffload: rigidRigidWgslOffload,
    });

    boundaryRuntime = (await applyCollisionBoundaryPassGpuOnly({
      rigidBodies: rigid,
      soft: softBody,
      n,
      rigidBounce,
      softBounce,
      applyBounceBoundary,
      wgslOffload: collisionBoundaryWgslOffload,
    })) || { mode: 'cpu-fallback', reason: 'unknown' };
  }

  return { boundaryRuntime };
}
