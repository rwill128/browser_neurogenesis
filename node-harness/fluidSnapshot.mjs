/**
 * Build a queryable fluid snapshot for all cells or a world-space rectangle.
 *
 * Output is intentionally bounded (`maxCells`) so JSON remains practical.
 * Cells are selected when dye/speed exceed thresholds and then ranked by intensity.
 */
export function collectFluidSnapshot({
  fluidField,
  world,
  rect = null,
  minDye = 6,
  minSpeed = 0.03,
  maxCells = 1500
}) {
  if (!fluidField || !world || !Number.isFinite(world.width) || !Number.isFinite(world.height)) {
    return {
      gridSize: 0,
      worldCell: { width: 0, height: 0 },
      scannedCells: 0,
      activeCells: 0,
      avgSpeed: 0,
      maxDye: 0,
      maxSpeed: 0,
      activeTileTelemetry: null,
      fluidStepPerf: null,
      boundsWorld: null,
      cells: []
    };
  }

  const N = Math.round(fluidField.size);
  if (!Number.isFinite(N) || N <= 0) {
    return {
      gridSize: 0,
      worldCell: { width: 0, height: 0 },
      scannedCells: 0,
      activeCells: 0,
      avgSpeed: 0,
      maxDye: 0,
      maxSpeed: 0,
      activeTileTelemetry: null,
      fluidStepPerf: null,
      boundsWorld: null,
      cells: []
    };
  }

  const cellW = world.width / N;
  const cellH = world.height / N;

  const x0 = rect ? Math.max(0, rect.x) : 0;
  const y0 = rect ? Math.max(0, rect.y) : 0;
  const x1 = rect ? Math.min(world.width, rect.x + rect.width) : world.width;
  const y1 = rect ? Math.min(world.height, rect.y + rect.height) : world.height;

  const gx0 = Math.max(0, Math.floor(x0 / cellW));
  const gy0 = Math.max(0, Math.floor(y0 / cellH));
  const gx1 = Math.min(N - 1, Math.floor(Math.max(0, x1 - 1e-9) / cellW));
  const gy1 = Math.min(N - 1, Math.floor(Math.max(0, y1 - 1e-9) / cellH));

  let scannedCells = 0;
  let activeCells = 0;
  let sumSpeed = 0;
  let maxSpeed = 0;
  let maxDye = 0;

  const candidates = [];

  for (let gy = gy0; gy <= gy1; gy++) {
    for (let gx = gx0; gx <= gx1; gx++) {
      scannedCells += 1;
      const idx = fluidField.IX(gx, gy);
      const r = fluidField.densityR[idx] || 0;
      const g = fluidField.densityG[idx] || 0;
      const b = fluidField.densityB[idx] || 0;
      const vx = fluidField.Vx[idx] || 0;
      const vy = fluidField.Vy[idx] || 0;
      const dye = r + g + b;
      const speed = Math.sqrt(vx * vx + vy * vy);

      if (speed > 0) sumSpeed += speed;
      if (speed > maxSpeed) maxSpeed = speed;
      if (dye > maxDye) maxDye = dye;

      if (dye >= minDye || speed >= minSpeed) {
        activeCells += 1;
        candidates.push({
          gx,
          gy,
          x: Number((((gx + 0.5) * cellW)).toFixed(3)),
          y: Number((((gy + 0.5) * cellH)).toFixed(3)),
          r: Number(r.toFixed(2)),
          g: Number(g.toFixed(2)),
          b: Number(b.toFixed(2)),
          vx: Number(vx.toFixed(4)),
          vy: Number(vy.toFixed(4)),
          dye: Number(dye.toFixed(2)),
          speed: Number(speed.toFixed(4)),
          score: dye + speed * 40
        });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const cells = candidates.slice(0, Math.max(1, Math.floor(maxCells))).map((c) => ({
    gx: c.gx,
    gy: c.gy,
    x: c.x,
    y: c.y,
    r: c.r,
    g: c.g,
    b: c.b,
    vx: c.vx,
    vy: c.vy,
    dye: c.dye,
    speed: c.speed
  }));

  return {
    gridSize: N,
    worldCell: {
      width: Number(cellW.toFixed(4)),
      height: Number(cellH.toFixed(4))
    },
    activeTileTelemetry: typeof fluidField.getActiveTileTelemetry === 'function'
      ? fluidField.getActiveTileTelemetry()
      : null,
    fluidStepPerf: typeof fluidField.getLastStepPerf === 'function'
      ? fluidField.getLastStepPerf()
      : null,
    activeTileDebug: typeof fluidField.getActiveTileDebugCells === 'function'
      ? fluidField.getActiveTileDebugCells(30000)
      : null,
    boundsWorld: {
      x: Number(x0.toFixed(3)),
      y: Number(y0.toFixed(3)),
      width: Number((x1 - x0).toFixed(3)),
      height: Number((y1 - y0).toFixed(3))
    },
    scannedCells,
    activeCells,
    avgSpeed: Number((sumSpeed / Math.max(1, scannedCells)).toFixed(5)),
    maxSpeed: Number(maxSpeed.toFixed(5)),
    maxDye: Number(maxDye.toFixed(2)),
    cells
  };
}
