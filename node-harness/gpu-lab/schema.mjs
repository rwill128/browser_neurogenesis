export function createFluidState({ size }) {
  const n = Math.max(4, Math.floor(Number(size) || 128));
  const cells = n * n;
  return {
    size: n,
    densityR: new Float32Array(cells),
    densityG: new Float32Array(cells),
    densityB: new Float32Array(cells),
    densityR0: new Float32Array(cells),
    densityG0: new Float32Array(cells),
    densityB0: new Float32Array(cells),
    vx: new Float32Array(cells),
    vy: new Float32Array(cells),
    vx0: new Float32Array(cells),
    vy0: new Float32Array(cells)
  };
}

export function snapshotMetrics(state) {
  const { vx, vy, densityR, densityG, densityB } = state;
  let sumSpeed = 0;
  let maxSpeed = 0;
  let dyeCells = 0;
  for (let i = 0; i < vx.length; i++) {
    const s = Math.hypot(vx[i], vy[i]);
    sumSpeed += s;
    if (s > maxSpeed) maxSpeed = s;
    if ((densityR[i] + densityG[i] + densityB[i]) > 1.0) dyeCells++;
  }
  return {
    avgSpeed: sumSpeed / Math.max(1, vx.length),
    maxSpeed,
    dyeFootprintPct: dyeCells / Math.max(1, vx.length)
  };
}
