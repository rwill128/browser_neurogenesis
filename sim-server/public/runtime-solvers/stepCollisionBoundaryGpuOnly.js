export function applyCollisionBoundaryPassGpuOnly({
  rigidBodies,
  soft,
  n,
  rigidBounce = 0.84,
  softBounce = 0.78,
  applyBounceBoundary,
}) {
  const rigidList = Array.isArray(rigidBodies) ? rigidBodies : [];
  const softNodes = Array.isArray(soft?.nodes) ? soft.nodes : [];

  for (const rb of rigidList) {
    if (!rb) continue;
    applyBounceBoundary(rb, n, rigidBounce);
  }

  for (const sn of softNodes) {
    if (!sn) continue;
    applyBounceBoundary(sn, n, softBounce);
  }
}
