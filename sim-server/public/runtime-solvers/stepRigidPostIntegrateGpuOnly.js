/**
 * GPU-only runtime rigid post-integration stabilization.
 * Owns rigid linear/angular velocity capping before collision passes so this
 * stepping responsibility lives in isolated gpu-only runtime modules.
 */
export function stabilizeRigidPostIntegrateGpuOnly({ rigidBodies, velocityCap = 4.0, omegaCap = 0.22 } = {}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return;

  const vCap = Math.max(0, Number(velocityCap) || 0);
  const wCap = Math.max(0, Number(omegaCap) || 0);

  for (const rb of rigidBodies) {
    const vmag = Math.hypot(rb.vx, rb.vy);
    if (vmag > vCap) {
      rb.vx = (rb.vx / vmag) * vCap;
      rb.vy = (rb.vy / vmag) * vCap;
    }

    rb.omega = Math.max(-wCap, Math.min(wCap, rb.omega || 0));
  }
}
