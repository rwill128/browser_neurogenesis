export function computeVectorRms(vec2) {
  if (!(vec2 instanceof Float32Array) || vec2.length < 2) return 0;
  const n = Math.floor(vec2.length / 2);
  if (n <= 0) return 0;
  let s2 = 0;
  for (let i = 0; i < n; i++) {
    const x = Number(vec2[i * 2]) || 0;
    const y = Number(vec2[i * 2 + 1]) || 0;
    s2 += x * x + y * y;
  }
  return Math.sqrt(s2 / n);
}

// Rigid-aligned residual between two centroid-relative point clouds (same ordering, no scale term).
// Returns absolute and normalized errors against the reference RMS radius.
export function computeRigidAlignedPoseResidual(refLocal, curLocal, refRmsOverride = null) {
  if (!(refLocal instanceof Float32Array) || !(curLocal instanceof Float32Array)) return null;
  if (refLocal.length !== curLocal.length || refLocal.length < 6) return null;

  const n = Math.floor(refLocal.length / 2);
  if (n < 3) return null;

  let dot = 0;
  let cross = 0;
  for (let i = 0; i < n; i++) {
    const rx = Number(refLocal[i * 2]) || 0;
    const ry = Number(refLocal[i * 2 + 1]) || 0;
    const px = Number(curLocal[i * 2]) || 0;
    const py = Number(curLocal[i * 2 + 1]) || 0;
    dot += rx * px + ry * py;
    cross += rx * py - ry * px;
  }

  const theta = (Math.abs(dot) + Math.abs(cross)) > 1e-12 ? Math.atan2(cross, dot) : 0;
  const c = Math.cos(theta);
  const s = Math.sin(theta);

  let err2 = 0;
  let errMax = 0;
  for (let i = 0; i < n; i++) {
    const rx = Number(refLocal[i * 2]) || 0;
    const ry = Number(refLocal[i * 2 + 1]) || 0;
    const tx = rx * c - ry * s;
    const ty = rx * s + ry * c;

    const px = Number(curLocal[i * 2]) || 0;
    const py = Number(curLocal[i * 2 + 1]) || 0;
    const dx = px - tx;
    const dy = py - ty;
    const e = Math.hypot(dx, dy);
    err2 += e * e;
    if (e > errMax) errMax = e;
  }

  const rms = Math.sqrt(err2 / n);
  const refRmsRaw = Number.isFinite(Number(refRmsOverride)) ? Number(refRmsOverride) : computeVectorRms(refLocal);
  const refRms = Math.max(1e-3, refRmsRaw);

  return {
    theta,
    rms,
    max: errMax,
    normalizedRms: rms / refRms,
    normalizedMax: errMax / refRms,
    refRms,
  };
}
