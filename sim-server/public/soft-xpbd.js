const EPS = 1e-8;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function signedAreaFromIndices(nodes, indices) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    s += a.x * b.y - b.x * a.y;
  }
  return s * 0.5;
}

function isFiniteNode(node) {
  return Number.isFinite(Number(node?.x)) && Number.isFinite(Number(node?.y));
}

function convexHullIndices(nodes, indices) {
  const unique = new Map();
  for (const idx of indices) {
    const n = nodes[idx];
    if (!isFiniteNode(n)) continue;
    const k = `${Math.round(n.x * 1e6)},${Math.round(n.y * 1e6)}`;
    if (!unique.has(k)) unique.set(k, idx);
  }

  const pts = [...unique.values()].map((idx) => ({ idx, x: nodes[idx].x, y: nodes[idx].y }));
  if (pts.length <= 2) return pts.map((p) => p.idx);
  pts.sort((a, b) => (a.x - b.x) || (a.y - b.y));

  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }

  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }

  const hull = lower.slice(0, -1).concat(upper.slice(0, -1)).map((p) => p.idx);
  return hull.length >= 3 ? hull : pts.map((p) => p.idx);
}

function traverseCycle(component, adjacency, nodes) {
  const set = new Set(component);
  const degreeTwo = component.every((idx) => (adjacency.get(idx)?.size || 0) === 2);
  if (!degreeTwo) return null;

  const start = [...component].sort((a, b) => (nodes[a].x - nodes[b].x) || (nodes[a].y - nodes[b].y))[0];
  const nbs = [...(adjacency.get(start) || [])].filter((nb) => set.has(nb));
  if (nbs.length !== 2) return null;

  let prev = start;
  let curr = nbs[0];
  if (nbs.length === 2) {
    const a = nodes[nbs[0]];
    const b = nodes[nbs[1]];
    const sa = Math.atan2(a.y - nodes[start].y, a.x - nodes[start].x);
    const sb = Math.atan2(b.y - nodes[start].y, b.x - nodes[start].x);
    curr = sa <= sb ? nbs[0] : nbs[1];
  }

  const loop = [start];
  const guard = component.length * 4 + 16;
  let steps = 0;

  while (steps++ < guard) {
    if (curr === start) break;
    if (!set.has(curr)) return null;
    loop.push(curr);

    const nextCandidates = [...(adjacency.get(curr) || [])].filter((nb) => set.has(nb) && nb !== prev);
    if (!nextCandidates.length) return null;
    if (nextCandidates.length > 1) {
      // Shouldn't happen in degree-2 cycle, but keep deterministic tie-break.
      nextCandidates.sort((ia, ib) => (nodes[ia].x - nodes[ib].x) || (nodes[ia].y - nodes[ib].y));
    }
    const next = nextCandidates[0];
    prev = curr;
    curr = next;
  }

  if (loop.length < 3) return null;
  if (curr !== start) return null;

  const unique = [...new Set(loop)];
  return unique.length >= 3 ? unique : null;
}

export function sanitizeSoftSprings(rawSprings, nodeCount, {
  edgeBodyPass = 0,
  edgeBodyBlock = 1,
  restFloor = 1e-3,
} = {}) {
  const out = [];
  const seenPairs = new Set();
  let dropped = 0;

  for (const sp of (Array.isArray(rawSprings) ? rawSprings : [])) {
    if (!Array.isArray(sp)) {
      dropped += 1;
      continue;
    }
    const [aRaw, bRaw, restRaw, edgeBodyRaw, edgeDyeRaw] = sp;

    const a = Number(aRaw);
    const b = Number(bRaw);
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      dropped += 1;
      continue;
    }
    if (a < 0 || b < 0 || a >= nodeCount || b >= nodeCount || a === b) {
      dropped += 1;
      continue;
    }

    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const key = `${lo}:${hi}`;
    if (seenPairs.has(key)) {
      dropped += 1;
      continue;
    }
    seenPairs.add(key);

    const rest = Math.max(restFloor, Number.isFinite(Number(restRaw)) ? Number(restRaw) : 1);
    const edgeBodyMode = Number(edgeBodyRaw) === edgeBodyPass ? edgeBodyPass : edgeBodyBlock;
    out.push([a, b, rest, edgeBodyMode, edgeDyeRaw]);
  }

  return { springs: out, dropped };
}

export function ensureLambdaCacheSize(previous, nextLength, clampAbs = 20) {
  const len = Math.max(0, Number(nextLength) | 0);
  const next = new Float32Array(len);
  if (!previous || typeof previous.length !== 'number') return next;

  const copy = Math.min(previous.length, len);
  for (let i = 0; i < copy; i++) {
    const v = Number(previous[i]);
    next[i] = Number.isFinite(v) ? clamp(v, -clampAbs, clampAbs) : 0;
  }
  return next;
}

export function decayLambdaCache(lambdaCache, {
  decay = 0.995,
  clampAbs = 20,
  deadband = 1e-4,
} = {}) {
  if (!lambdaCache || typeof lambdaCache.length !== 'number') return lambdaCache;
  const k = clamp(Number(decay), 0, 1);
  const eps = Math.max(0, Number(deadband) || 0);
  const limit = Math.max(0, Number(clampAbs) || 0);

  for (let i = 0; i < lambdaCache.length; i++) {
    const v = Number(lambdaCache[i]);
    if (!Number.isFinite(v)) {
      lambdaCache[i] = 0;
      continue;
    }
    const next = clamp(v * k, -limit, limit);
    lambdaCache[i] = Math.abs(next) <= eps ? 0 : next;
  }

  return lambdaCache;
}

export function recoverSoftSpringRests(springs, restBaseline, {
  recoverRate = 0.08,
  hardMinFactor = 0.6,
  hardMaxFactor = 1.6,
  jitterDeadband = 1e-4,
  adaptiveGainMax = 2.4,
  adaptiveExponent = 0.8,
  elongationBiasMax = 1.22,
  compressionBiasMax = 1.12,
  errorPivot = 0.16,
  convergenceWindow = 0.12,
  convergenceBiasMax = 0.96,
  convergenceNudge = 0.35,
  nearBaselineSnapWindow = 0.004,
  nearBaselineSnapBlend = 0.9,
  globalErrorCouplingMax = 1.25,
  globalDirectionalCouplingMax = 1.18,
  outlierRecoveryCouplingMax = 1.22,
  outlierErrorPivot = 0.75,
  localEndpointCouplingMax = 1.16,
  localDirectionalCouplingMax = 1.1,
  localImbalanceCouplingMax = 1.1,
  localErrorPivot = 0.2,
  polarityCouplingMax = 1.08,
  counterPolarityCouplingMax = 1,
  smallRestRecoveryCouplingMax = 1,
  smallRestPivot = 0.9,
  smallRestErrorGate = 0.08,
  midErrorRecoveryCouplingMax = 1,
  midErrorRecoveryCenter = 0.22,
  midErrorRecoveryHalfWidth = 0.22,
  highRateSnapRecoverThreshold = 0.055,
  highRateSnapErrorThreshold = 0.002,
} = {}) {
  if (!Array.isArray(springs) || !restBaseline || typeof restBaseline.length !== 'number') return 0;

  const k = clamp(Number(recoverRate), 0, 1);
  const minFactor = Math.max(0.05, Number(hardMinFactor) || 0.6);
  const maxFactor = Math.max(minFactor + 1e-3, Number(hardMaxFactor) || 1.6);
  const eps = Math.max(0, Number(jitterDeadband) || 0);
  const gainMax = Math.max(1, Number(adaptiveGainMax) || 1);
  const adaptiveMode = gainMax > 1.0001;
  const exponent = Math.max(0.25, Math.min(2, Number(adaptiveExponent) || 0.8));
  const elongationBias = Math.max(1, Number(elongationBiasMax) || 1);
  const compressionBias = Math.max(1, Number(compressionBiasMax) || 1);
  const pivot = Math.max(1e-6, Number(errorPivot) || 0.16);
  const window = Math.max(1e-6, Number(convergenceWindow) || 0.12);
  const biasMax = clamp(Number(convergenceBiasMax) || 0, 0, 0.999);
  const nudge = clamp(Number(convergenceNudge) || 0, 0, 1);
  const snapWindow = Math.max(0, Number(nearBaselineSnapWindow) || 0);
  const snapBlend = clamp(Number(nearBaselineSnapBlend) || 0, 0, 1);
  const globalCouplingMax = Math.max(1, Number(globalErrorCouplingMax) || 1);
  const directionalCouplingMax = Math.max(1, Number(globalDirectionalCouplingMax) || 1);
  const outlierCouplingMax = Math.max(1, Number(outlierRecoveryCouplingMax) || 1);
  const outlierPivot = Math.max(1e-6, Number(outlierErrorPivot) || 0.75);
  const localEndpointMax = Math.max(1, Number(localEndpointCouplingMax) || 1);
  const localDirectionalMax = Math.max(1, Number(localDirectionalCouplingMax) || 1);
  const localImbalanceMax = Math.max(1, Number(localImbalanceCouplingMax) || 1);
  const localPivot = Math.max(1e-6, Number(localErrorPivot) || 0.2);
  const polarityMax = Math.max(1, Number(polarityCouplingMax) || 1);
  const counterPolarityMax = Math.max(1, Number(counterPolarityCouplingMax) || 1);
  const smallRestCouplingMax = Math.max(1, Number(smallRestRecoveryCouplingMax) || 1);
  const smallRestErrPivot = Math.max(1e-6, Number(smallRestPivot) || 0.9);
  const smallRestErrGate = Math.max(1e-6, Number(smallRestErrorGate) || 0.08);
  const midErrorCouplingMax = Math.max(1, Number(midErrorRecoveryCouplingMax) || 1);
  const midErrorCenter = Math.max(1e-6, Number(midErrorRecoveryCenter) || 0.22);
  const midErrorHalfWidth = Math.max(1e-6, Number(midErrorRecoveryHalfWidth) || 0.22);
  const highRateSnapRateThreshold = Math.max(0, Number(highRateSnapRecoverThreshold) || 0);
  const highRateSnapErrThreshold = Math.max(0, Number(highRateSnapErrorThreshold) || 0);

  let touched = 0;
  const n = Math.min(springs.length, restBaseline.length);
  let maxNodeIndex = -1;
  for (let i = 0; i < n; i++) {
    const sp = springs[i];
    if (!Array.isArray(sp) || sp.length < 2) continue;
    const a = Number(sp[0]);
    const b = Number(sp[1]);
    if (Number.isInteger(a) && a >= 0) maxNodeIndex = Math.max(maxNodeIndex, a);
    if (Number.isInteger(b) && b >= 0) maxNodeIndex = Math.max(maxNodeIndex, b);
  }
  const nodeErrAbsSum = maxNodeIndex >= 0 ? new Float64Array(maxNodeIndex + 1) : null;
  const nodeErrSignedSum = maxNodeIndex >= 0 ? new Float64Array(maxNodeIndex + 1) : null;
  const nodeErrCount = maxNodeIndex >= 0 ? new Uint32Array(maxNodeIndex + 1) : null;

  let meanErrNorm = 0;
  let meanSignedErrNorm = 0;
  let meanErrCount = 0;
  let meanPositiveErrNorm = 0;
  let meanNegativeErrNorm = 0;
  let positiveErrCount = 0;
  let negativeErrCount = 0;
  let meanBaseRest = 0;
  let meanBaseRestCount = 0;
  if (n > 0 && (adaptiveMode || nodeErrAbsSum)) {
    for (let i = 0; i < n; i++) {
      const sp = springs[i];
      if (!Array.isArray(sp) || sp.length < 3) continue;
      const base = Math.max(1e-4, Number(restBaseline[i]) || 1e-4);
      meanBaseRest += base;
      meanBaseRestCount += 1;
      const current = Number(sp[2]);
      const cur = Number.isFinite(current) ? current : base;
      const signedErrNorm = (cur - base) / base;
      const absErrNorm = Math.abs(signedErrNorm);

      if (adaptiveMode && (globalCouplingMax > 1.0001 || polarityMax > 1.0001)) {
        meanErrNorm += absErrNorm;
        meanSignedErrNorm += signedErrNorm;
        meanErrCount += 1;
        if (signedErrNorm >= 0) {
          meanPositiveErrNorm += absErrNorm;
          positiveErrCount += 1;
        } else {
          meanNegativeErrNorm += absErrNorm;
          negativeErrCount += 1;
        }
      }

      if (nodeErrAbsSum) {
        const a = Number(sp[0]);
        const b = Number(sp[1]);
        if (Number.isInteger(a) && a >= 0 && a < nodeErrAbsSum.length) {
          nodeErrAbsSum[a] += absErrNorm;
          nodeErrSignedSum[a] += signedErrNorm;
          nodeErrCount[a] += 1;
        }
        if (Number.isInteger(b) && b >= 0 && b < nodeErrAbsSum.length) {
          nodeErrAbsSum[b] += absErrNorm;
          nodeErrSignedSum[b] += signedErrNorm;
          nodeErrCount[b] += 1;
        }
      }
    }
    meanErrNorm = meanErrCount > 0 ? (meanErrNorm / meanErrCount) : 0;
    meanSignedErrNorm = meanErrCount > 0 ? (meanSignedErrNorm / meanErrCount) : 0;
    meanPositiveErrNorm = positiveErrCount > 0 ? (meanPositiveErrNorm / positiveErrCount) : 0;
    meanNegativeErrNorm = negativeErrCount > 0 ? (meanNegativeErrNorm / negativeErrCount) : 0;
    meanBaseRest = meanBaseRestCount > 0 ? (meanBaseRest / meanBaseRestCount) : 0;
  }

  for (let i = 0; i < n; i++) {
    const sp = springs[i];
    if (!Array.isArray(sp) || sp.length < 3) continue;

    const base = Math.max(1e-4, Number(restBaseline[i]) || 1e-4);
    const current = Number(sp[2]);
    const cur = Number.isFinite(current) ? current : base;
    const signedErrNorm = (cur - base) / base;
    const errNorm = Math.abs(signedErrNorm);
    const adaptiveErr = Math.min(1, Math.pow(errNorm / pivot, exponent));
    const dirBoost = (signedErrNorm >= 0) ? elongationBias : compressionBias;
    const globalErrAlpha = clamp(meanErrNorm / pivot, 0, 1);
    const globalBoost = 1 + (globalCouplingMax - 1) * globalErrAlpha;
    const globalDirectionalAlpha = clamp(Math.abs(meanSignedErrNorm) / pivot, 0, 1);
    const directionalAligned = (signedErrNorm === 0 || meanSignedErrNorm === 0)
      ? 0
      : (Math.sign(signedErrNorm) === Math.sign(meanSignedErrNorm) ? 1 : 0);
    const directionalBoost = 1 + (directionalCouplingMax - 1) * globalDirectionalAlpha * directionalAligned;
    const polarityMeanErrNorm = signedErrNorm >= 0 ? meanPositiveErrNorm : meanNegativeErrNorm;
    const polarityAlpha = clamp(polarityMeanErrNorm / pivot, 0, 1);
    const polarityBoost = 1 + (polarityMax - 1) * polarityAlpha;
    const counterPolarityAligned = (signedErrNorm === 0 || meanSignedErrNorm === 0)
      ? 0
      : (Math.sign(signedErrNorm) === Math.sign(meanSignedErrNorm) ? 0 : 1);
    const counterPolarityBoost = 1 + (counterPolarityMax - 1) * globalDirectionalAlpha * counterPolarityAligned;
    const outlierRatio = (adaptiveMode && outlierCouplingMax > 1.0001 && meanErrCount > 0)
      ? (errNorm / Math.max(1e-6, meanErrNorm || 0))
      : 1;
    const outlierAlpha = clamp((outlierRatio - 1) / outlierPivot, 0, 1);
    const outlierBoost = 1 + (outlierCouplingMax - 1) * outlierAlpha;
    const shortRestRatio = (adaptiveMode && smallRestCouplingMax > 1.0001 && meanBaseRest > 1e-6)
      ? (meanBaseRest / base)
      : 1;
    const shortRestAlpha = clamp((shortRestRatio - 1) / smallRestErrPivot, 0, 1);
    const shortRestErrAlpha = clamp(errNorm / smallRestErrGate, 0, 1);
    const shortRestBoost = 1 + (smallRestCouplingMax - 1) * shortRestAlpha * shortRestErrAlpha;
    const midErrorDistance = Math.abs(errNorm - midErrorCenter);
    const midErrorAlpha = clamp(1 - (midErrorDistance / midErrorHalfWidth), 0, 1);
    const midErrorBoost = 1 + (midErrorCouplingMax - 1) * midErrorAlpha;

    let localBoost = 1;
    let localDirectionalBoost = 1;
    let localImbalanceBoost = 1;
    if (adaptiveMode && nodeErrAbsSum && (localEndpointMax > 1.0001 || localDirectionalMax > 1.0001 || localImbalanceMax > 1.0001)) {
      const a = Number(sp[0]);
      const b = Number(sp[1]);
      const validA = Number.isInteger(a) && a >= 0 && a < nodeErrAbsSum.length && nodeErrCount[a] > 0;
      const validB = Number.isInteger(b) && b >= 0 && b < nodeErrAbsSum.length && nodeErrCount[b] > 0;
      if (validA || validB) {
        const absA = validA ? (nodeErrAbsSum[a] / nodeErrCount[a]) : 0;
        const absB = validB ? (nodeErrAbsSum[b] / nodeErrCount[b]) : 0;
        const signedA = validA ? (nodeErrSignedSum[a] / nodeErrCount[a]) : 0;
        const signedB = validB ? (nodeErrSignedSum[b] / nodeErrCount[b]) : 0;
        const denom = (validA && validB) ? 2 : 1;
        const endpointAbs = (absA + absB) / denom;
        const endpointSigned = (signedA + signedB) / denom;

        const localAlpha = clamp(endpointAbs / localPivot, 0, 1);
        localBoost = 1 + (localEndpointMax - 1) * localAlpha;

        const localSignAligned = (signedErrNorm === 0 || endpointSigned === 0)
          ? 0
          : (Math.sign(signedErrNorm) === Math.sign(endpointSigned) ? 1 : 0);
        const localDirAlpha = clamp(Math.abs(endpointSigned) / localPivot, 0, 1);
        localDirectionalBoost = 1 + (localDirectionalMax - 1) * localDirAlpha * localSignAligned;

        const endpointOpposed = (signedA === 0 || signedB === 0)
          ? 0
          : (Math.sign(signedA) === Math.sign(signedB) ? 0 : 1);
        const imbalanceErr = Math.abs(signedA - signedB) * 0.5;
        const localImbalanceAlpha = clamp(imbalanceErr / localPivot, 0, 1);
        localImbalanceBoost = 1 + (localImbalanceMax - 1) * localImbalanceAlpha * endpointOpposed;
      }
    }

    const boost = (1 + (gainMax - 1) * adaptiveErr * dirBoost)
      * globalBoost
      * directionalBoost
      * polarityBoost
      * counterPolarityBoost
      * outlierBoost
      * shortRestBoost
      * midErrorBoost
      * localBoost
      * localDirectionalBoost
      * localImbalanceBoost;
    const recover = clamp(k * boost, 0, 1);

    // As we approach baseline, gradually tighten allowable rest-length range to reduce
    // long-tail spring-rest drift while preserving wide bounds during large deformations.
    const convergeAlpha = adaptiveMode ? clamp(1 - (errNorm / window), 0, 1) : 0;
    const tighten = convergeAlpha * biasMax;
    const localMinFactor = minFactor + (1 - minFactor) * tighten;
    const localMaxFactor = maxFactor - (maxFactor - 1) * tighten;
    const convergedRecover = clamp(recover + (1 - recover) * convergeAlpha * nudge, 0, 1);
    let target = clamp(cur + (base - cur) * convergedRecover, base * localMinFactor, base * localMaxFactor);

    if (adaptiveMode && snapWindow > 0) {
      const targetErrNorm = Math.abs(base - target) / base;
      if (targetErrNorm <= snapWindow) {
        target = clamp(target + (base - target) * snapBlend, base * localMinFactor, base * localMaxFactor);
      }
    }

    // High recover-rate near-baseline lock: prevent tiny residual oscillation/undershoot
    // under aggressive adaptive recovery by snapping exactly to baseline once error is small.
    if (adaptiveMode && k >= highRateSnapRateThreshold && highRateSnapErrThreshold > 0) {
      const targetErrNorm = Math.abs(base - target) / base;
      if (targetErrNorm <= highRateSnapErrThreshold) target = base;
    }

    // Deterministic tail lock: if we're already inside the configured jitter deadband,
    // collapse exactly to baseline so increased recoverRate cannot get stuck with tiny
    // residual rest drift from floating-point noise.
    const deadbandNorm = eps / base;
    if (Math.abs(base - target) / base <= deadbandNorm) target = base;

    if (Math.abs(target - cur) <= eps) {
      if (target === base && cur !== base) {
        sp[2] = base;
        touched += 1;
      }
      continue;
    }
    sp[2] = Math.max(1e-4, target);
    touched += 1;
  }

  return touched;
}

export function buildSoftClusterBoundaryLoops(nodes, springs, { blockMode = 1 } = {}) {
  const byClusterNodes = new Map();
  for (let i = 0; i < (nodes?.length || 0); i++) {
    const n = nodes[i];
    if (!isFiniteNode(n)) continue;
    const cid = n.clusterId ?? 0;
    if (!byClusterNodes.has(cid)) byClusterNodes.set(cid, new Set());
    byClusterNodes.get(cid).add(i);
  }

  const byClusterAdj = new Map();
  for (const sp of (springs || [])) {
    if (!Array.isArray(sp) || sp.length < 4) continue;
    const a = Number(sp[0]);
    const b = Number(sp[1]);
    const mode = Number(sp[3]);
    if (!Number.isInteger(a) || !Number.isInteger(b)) continue;
    if (a < 0 || b < 0 || a >= nodes.length || b >= nodes.length || a === b) continue;
    if (mode !== blockMode) continue;

    const na = nodes[a];
    const nb = nodes[b];
    if (!isFiniteNode(na) || !isFiniteNode(nb)) continue;
    const ca = na.clusterId ?? 0;
    const cb = nb.clusterId ?? 0;
    if (ca !== cb) continue;

    if (!byClusterAdj.has(ca)) byClusterAdj.set(ca, new Map());
    const adj = byClusterAdj.get(ca);
    if (!adj.has(a)) adj.set(a, new Set());
    if (!adj.has(b)) adj.set(b, new Set());
    adj.get(a).add(b);
    adj.get(b).add(a);
  }

  const loops = [];

  for (const [clusterId, nodeSet] of byClusterNodes.entries()) {
    const clusterNodes = [...nodeSet];
    if (clusterNodes.length < 3) continue;

    const adj = byClusterAdj.get(clusterId) || new Map();
    const visited = new Set();
    const candidateLoops = [];

    for (const start of adj.keys()) {
      if (visited.has(start)) continue;
      const stack = [start];
      visited.add(start);
      const comp = [];
      while (stack.length) {
        const cur = stack.pop();
        comp.push(cur);
        for (const nb of (adj.get(cur) || [])) {
          if (!visited.has(nb)) {
            visited.add(nb);
            stack.push(nb);
          }
        }
      }
      if (comp.length < 3) continue;
      const cycle = traverseCycle(comp, adj, nodes);
      if (cycle && cycle.length >= 3) candidateLoops.push(cycle);
    }

    let indices = null;
    let source = 'hull';
    if (candidateLoops.length) {
      candidateLoops.sort((a, b) => Math.abs(signedAreaFromIndices(nodes, b)) - Math.abs(signedAreaFromIndices(nodes, a)));
      indices = candidateLoops[0];
      source = 'boundary';
    } else {
      indices = convexHullIndices(nodes, clusterNodes);
    }

    if (!indices || indices.length < 3) continue;
    const area = signedAreaFromIndices(nodes, indices);
    if (Math.abs(area) < EPS) continue;
    if (area < 0) indices = [...indices].reverse();

    loops.push({ clusterId, indices, source });
  }

  return loops;
}
