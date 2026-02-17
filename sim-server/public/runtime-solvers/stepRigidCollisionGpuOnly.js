/**
 * GPU-only runtime rigid-rigid collision pass.
 * Keeps collision stepping isolated for the gpu-only runtime path while
 * preserving baseline/default behavior and call contracts.
 */
export function resolveRigidRigidCollisionPassGpuOnly({
  rigidBodies,
  slop,
  contacts,
  iter,
  phase,
  resolveRigidVsRigidPolygonCollision,
}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length < 2) return;
  if (typeof resolveRigidVsRigidPolygonCollision !== 'function') {
    throw new Error('gpu-only rigid collision pass requires resolveRigidVsRigidPolygonCollision callback');
  }

  for (let i = 0; i < rigidBodies.length; i++) {
    for (let j = i + 1; j < rigidBodies.length; j++) {
      resolveRigidVsRigidPolygonCollision(rigidBodies[i], rigidBodies[j], slop, {
        contacts,
        aIndex: i,
        bIndex: j,
        iter,
        phase,
      });
    }
  }
}
