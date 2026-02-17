/**
 * GPU-only runtime soft integration path.
 * Isolated from baseline in gpu-lab.js so soft-body stepping responsibility
 * can migrate incrementally while baseline remains the default/reference path.
 */
export function integrateSoftBodiesGpuOnly({
  soft,
  n,
  dt,
  softIntegrationScale,
  hybridNodeVCap,
  applyBounceBoundary,
}) {
  for (const node of (soft?.nodes || [])) {
    const vmag = Math.hypot(node.vx, node.vy);
    if (vmag > hybridNodeVCap) {
      node.vx = (node.vx / vmag) * hybridNodeVCap;
      node.vy = (node.vy / vmag) * hybridNodeVCap;
    }
    node.x = node.x + node.vx * dt * softIntegrationScale;
    node.y = node.y + node.vy * dt * softIntegrationScale;
    applyBounceBoundary(node, n, 0.78);
  }
}
