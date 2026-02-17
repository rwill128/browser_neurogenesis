/**
 * GPU-only runtime soft spring XPBD pass.
 * Keeps spring-constraint stepping isolated for the gpu-only runtime path while
 * preserving baseline/default behavior and call contracts.
 */
export function applySoftSpringsXPBDVelocityGpuOnly({
  soft,
  dtPos,
  stiffnessScale,
  lambdaCache,
  softXpbdIters,
  softXpbdBaseCompliance,
  clamp,
  skipClusterSet = null,
}) {
  if (!soft?.nodes?.length || !soft?.springs?.length) return;
  if (!Array.isArray(lambdaCache) && !(lambdaCache instanceof Float32Array)) {
    throw new Error('gpu-only soft spring XPBD pass requires lambdaCache array-like');
  }
  if (typeof clamp !== 'function') {
    throw new Error('gpu-only soft spring XPBD pass requires clamp callback');
  }

  const alpha = (softXpbdBaseCompliance / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < softXpbdIters; iter++) {
    for (let si = 0; si < soft.springs.length; si++) {
      const [i, j, rest] = soft.springs[si];
      const a = soft.nodes[i];
      const b = soft.nodes[j];
      if (!a || !b) continue;
      if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

      const ax = a.x + a.vx * dtPos;
      const ay = a.y + a.vy * dtPos;
      const bx = b.x + b.vx * dtPos;
      const by = b.y + b.vy * dtPos;

      const dx = bx - ax;
      const dy = by - ay;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = dx / d;
      const ny = dy / d;
      const strainCap = Math.max(0.05, Math.abs(rest) * 0.45);
      const C = clamp(d - rest, -strainCap, strainCap);

      const wA = 1 / Math.max(0.02, a.mass || 1);
      const wB = 1 / Math.max(0.02, b.mass || 1);
      const wSum = wA + wB;
      if (wSum <= 1e-9) continue;

      const lambdaPrev = Number(lambdaCache[si]) || 0;
      let dl = (-C - alpha * lambdaPrev) / (wSum + alpha);
      if (!Number.isFinite(dl)) continue;
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      lambdaCache[si] = lambdaNext;

      const corrAx = -wA * dl * nx;
      const corrAy = -wA * dl * ny;
      const corrBx = wB * dl * nx;
      const corrBy = wB * dl * ny;

      a.vx += corrAx / dtPos;
      a.vy += corrAy / dtPos;
      b.vx += corrBx / dtPos;
      b.vy += corrBy / dtPos;
    }
  }
}
