/**
 * GPU-only runtime soft spring XPBD pass.
 * Keeps spring-constraint stepping isolated for the gpu-only runtime path while
 * preserving baseline/default behavior and call contracts.
 */
export function buildSoftSpringXpbdWgslPlan({
  soft,
  skipClusterSet = null,
} = {}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springs = Array.isArray(soft?.springs) ? soft.springs : [];

  const activeSpringIndices = [];
  const endpointNodeIndicesRaw = [];
  const endpointSpringIndicesRaw = [];
  const endpointSignsRaw = [];

  for (let si = 0; si < springs.length; si++) {
    const spring = springs[si];
    const i = Number(spring?.[0]);
    const j = Number(spring?.[1]);
    const rest = Number(spring?.[2]);
    if (!Number.isInteger(i) || !Number.isInteger(j)) continue;
    if (i < 0 || i >= nodes.length || j < 0 || j >= nodes.length) continue;
    if (!Number.isFinite(rest)) continue;

    const a = nodes[i];
    const b = nodes[j];
    if (!a || !b) continue;
    if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

    const activeIndex = activeSpringIndices.length;
    activeSpringIndices.push(si);

    endpointNodeIndicesRaw.push(i, j);
    endpointSpringIndicesRaw.push(activeIndex, activeIndex);
    endpointSignsRaw.push(-1, 1);
  }

  const endpointCount = endpointNodeIndicesRaw.length;
  const nodeEndpointCounts = new Uint32Array(nodes.length);
  for (let ei = 0; ei < endpointCount; ei++) {
    nodeEndpointCounts[endpointNodeIndicesRaw[ei]] += 1;
  }

  const nodeEndpointOffsets = new Uint32Array(nodes.length + 1);
  for (let ni = 0; ni < nodes.length; ni++) {
    nodeEndpointOffsets[ni + 1] = nodeEndpointOffsets[ni] + nodeEndpointCounts[ni];
  }

  const endpointNodeIndices = new Uint32Array(endpointCount);
  const endpointSpringIndices = new Uint32Array(endpointCount);
  const endpointSigns = new Int8Array(endpointCount);
  const cursor = nodeEndpointOffsets.slice(0, nodes.length);

  for (let ei = 0; ei < endpointCount; ei++) {
    const ni = endpointNodeIndicesRaw[ei];
    const dst = cursor[ni]++;
    endpointNodeIndices[dst] = ni;
    endpointSpringIndices[dst] = endpointSpringIndicesRaw[ei];
    endpointSigns[dst] = endpointSignsRaw[ei];
  }

  return {
    nodeCount: nodes.length,
    springCount: springs.length,
    activeSpringCount: activeSpringIndices.length,
    endpointCount,
    activeSpringIndices: Uint32Array.from(activeSpringIndices),
    endpointNodeIndices,
    endpointSpringIndices,
    endpointSigns,
    nodeEndpointOffsets,
  };
}

export function applySoftSpringsXPBDVelocityGpuOnly({
  soft,
  dtPos,
  stiffnessScale,
  lambdaCache,
  softXpbdIters,
  softXpbdBaseCompliance,
  clamp,
  skipClusterSet = null,
  wgslOffload,
}) {
  if (!soft?.nodes?.length || !soft?.springs?.length) return;
  if (!Array.isArray(lambdaCache) && !(lambdaCache instanceof Float32Array)) {
    throw new Error('gpu-only soft spring XPBD pass requires lambdaCache array-like');
  }
  if (typeof clamp !== 'function') {
    throw new Error('gpu-only soft spring XPBD pass requires clamp callback');
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    // Unblocker for upcoming WGSL XPBD stage: prepare deterministic CSR endpoint
    // ownership now so the compute stage can run spring solve + node reduction
    // without atomics changing integration ownership semantics.
    const plan = buildSoftSpringXpbdWgslPlan({ soft, skipClusterSet });
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.lastPreparedSpringCount = plan.activeSpringCount;
    wgslOffload.state.lastPreparedEndpointCount = plan.endpointCount;
    wgslOffload.state.lastMode = 'cpu-prepared';
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
