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
  const activeNodeAIndices = [];
  const activeNodeBIndices = [];
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
    activeNodeAIndices.push(i);
    activeNodeBIndices.push(j);

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

  // Deterministic conflict-free spring coloring (no shared nodes per color).
  // This is a direct unblocker for WGSL spring solves: each color can dispatch
  // in parallel without atomics while preserving Gauss-Seidel ownership order.
  const nodeUsedColors = Array.from({ length: nodes.length }, () => new Set());
  const springColors = new Uint32Array(activeSpringIndices.length);
  let colorCount = 0;
  for (let ai = 0; ai < activeSpringIndices.length; ai++) {
    const i = activeNodeAIndices[ai];
    const j = activeNodeBIndices[ai];
    let color = 0;
    while (nodeUsedColors[i].has(color) || nodeUsedColors[j].has(color)) color++;
    nodeUsedColors[i].add(color);
    nodeUsedColors[j].add(color);
    springColors[ai] = color;
    if (color + 1 > colorCount) colorCount = color + 1;
  }

  const springColorCounts = new Uint32Array(colorCount);
  for (let ai = 0; ai < springColors.length; ai++) springColorCounts[springColors[ai]] += 1;
  const springColorOffsets = new Uint32Array(colorCount + 1);
  for (let ci = 0; ci < colorCount; ci++) springColorOffsets[ci + 1] = springColorOffsets[ci] + springColorCounts[ci];
  const springColorOrderedIndices = new Uint32Array(activeSpringIndices.length);
  const springColorCursor = springColorOffsets.slice(0, colorCount);
  for (let ai = 0; ai < springColors.length; ai++) {
    const ci = springColors[ai];
    const dst = springColorCursor[ci]++;
    springColorOrderedIndices[dst] = ai;
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
    springColors,
    springColorOffsets,
    springColorOrderedIndices,
  };
}

export function buildSoftSpringXpbdWgslLayout({
  soft,
  plan,
} = {}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const activeSpringCount = Number(plan?.activeSpringCount) || 0;
  const activeSpringIndices = plan?.activeSpringIndices instanceof Uint32Array
    ? plan.activeSpringIndices
    : new Uint32Array(0);
  const springColorOrderedIndices = plan?.springColorOrderedIndices instanceof Uint32Array
    ? plan.springColorOrderedIndices
    : new Uint32Array(0);

  const springNodeA = new Uint32Array(activeSpringCount);
  const springNodeB = new Uint32Array(activeSpringCount);
  const springRest = new Float32Array(activeSpringCount);
  const springInvMassA = new Float32Array(activeSpringCount);
  const springInvMassB = new Float32Array(activeSpringCount);

  for (let ai = 0; ai < activeSpringCount; ai++) {
    const si = activeSpringIndices[ai];
    const spring = soft?.springs?.[si];
    const i = Number(spring?.[0]);
    const j = Number(spring?.[1]);
    const rest = Number(spring?.[2]);
    const a = nodes[i];
    const b = nodes[j];

    springNodeA[ai] = Number.isInteger(i) && i >= 0 ? i : 0;
    springNodeB[ai] = Number.isInteger(j) && j >= 0 ? j : 0;
    springRest[ai] = Number.isFinite(rest) ? rest : 0;
    springInvMassA[ai] = 1 / Math.max(0.02, Number(a?.mass) || 1);
    springInvMassB[ai] = 1 / Math.max(0.02, Number(b?.mass) || 1);
  }

  const springNodeAByColor = new Uint32Array(springColorOrderedIndices.length);
  const springNodeBByColor = new Uint32Array(springColorOrderedIndices.length);
  const springRestByColor = new Float32Array(springColorOrderedIndices.length);
  const springInvMassAByColor = new Float32Array(springColorOrderedIndices.length);
  const springInvMassBByColor = new Float32Array(springColorOrderedIndices.length);
  for (let oi = 0; oi < springColorOrderedIndices.length; oi++) {
    const ai = springColorOrderedIndices[oi];
    springNodeAByColor[oi] = springNodeA[ai];
    springNodeBByColor[oi] = springNodeB[ai];
    springRestByColor[oi] = springRest[ai];
    springInvMassAByColor[oi] = springInvMassA[ai];
    springInvMassBByColor[oi] = springInvMassB[ai];
  }

  // WebGPU storage buffers are naturally 4-byte addressed; widen signs now so
  // upcoming WGSL reduction kernels can consume endpoint direction directly.
  const endpointSignsI32 = new Int32Array(plan?.endpointCount || 0);
  for (let ei = 0; ei < endpointSignsI32.length; ei++) {
    endpointSignsI32[ei] = Number(plan?.endpointSigns?.[ei]) < 0 ? -1 : 1;
  }

  return {
    nodeEndpointOffsets: plan?.nodeEndpointOffsets || new Uint32Array(0),
    endpointSpringIndices: plan?.endpointSpringIndices || new Uint32Array(0),
    endpointSignsI32,
    springNodeA,
    springNodeB,
    springRest,
    springInvMassA,
    springInvMassB,
    springColorOffsets: plan?.springColorOffsets || new Uint32Array(0),
    springColorOrderedIndices,
    springNodeAByColor,
    springNodeBByColor,
    springRestByColor,
    springInvMassAByColor,
    springInvMassBByColor,
    byteLength:
      (plan?.nodeEndpointOffsets?.byteLength || 0)
      + (plan?.endpointSpringIndices?.byteLength || 0)
      + endpointSignsI32.byteLength
      + springNodeA.byteLength
      + springNodeB.byteLength
      + springRest.byteLength
      + springInvMassA.byteLength
      + springInvMassB.byteLength
      + (plan?.springColorOffsets?.byteLength || 0)
      + springColorOrderedIndices.byteLength
      + springNodeAByColor.byteLength
      + springNodeBByColor.byteLength
      + springRestByColor.byteLength
      + springInvMassAByColor.byteLength
      + springInvMassBByColor.byteLength,
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
    // ownership and spring SoA buffers now so the compute stage can run spring
    // solve + node reduction without atomics changing ownership semantics.
    const plan = buildSoftSpringXpbdWgslPlan({ soft, skipClusterSet });
    const layout = buildSoftSpringXpbdWgslLayout({ soft, plan });
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.lastPreparedSpringCount = plan.activeSpringCount;
    wgslOffload.state.lastPreparedEndpointCount = plan.endpointCount;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
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
