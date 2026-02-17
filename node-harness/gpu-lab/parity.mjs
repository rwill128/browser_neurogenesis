const DEFAULT_TOLERANCES = Object.freeze({
  meanAbsDelta: 0.15,
  rms: 0.2,
  maxAbs: 0.5,
  totalMass: 0.1
});

function finiteNumber(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

export function normalizeSnapshotMetrics(metrics = {}) {
  return {
    meanAbsDelta: finiteNumber(Number(metrics.meanAbsDelta), 0),
    rms: finiteNumber(Number(metrics.rms), 0),
    maxAbs: finiteNumber(Number(metrics.maxAbs), 0),
    totalMass: finiteNumber(Number(metrics.totalMass), 0)
  };
}

export function evaluateParity(cpuMetrics, candidateMetrics, tolerances = {}) {
  const tol = { ...DEFAULT_TOLERANCES, ...(tolerances || {}) };
  const cpu = normalizeSnapshotMetrics(cpuMetrics);
  const candidate = normalizeSnapshotMetrics(candidateMetrics);

  const deltas = {
    meanAbsDelta: Math.abs(candidate.meanAbsDelta - cpu.meanAbsDelta),
    rms: Math.abs(candidate.rms - cpu.rms),
    maxAbs: Math.abs(candidate.maxAbs - cpu.maxAbs),
    totalMass: Math.abs(candidate.totalMass - cpu.totalMass)
  };

  const failing = Object.entries(deltas)
    .filter(([key, value]) => value > tol[key])
    .map(([key, value]) => ({ metric: key, delta: value, tolerance: tol[key] }));

  return {
    pass: failing.length === 0,
    deltas,
    tolerances: tol,
    failing
  };
}

export function buildParityEntry({ label, cpuMetrics, candidateMetrics = null, tolerances = null, reason = null }) {
  if (!candidateMetrics) {
    return {
      backend: label,
      comparable: false,
      reason: reason || 'candidate metrics unavailable'
    };
  }

  const comparison = evaluateParity(cpuMetrics, candidateMetrics, tolerances || undefined);
  return {
    backend: label,
    comparable: true,
    ...comparison
  };
}
