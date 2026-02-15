export function sampleBilinear(field, width, height, x, y) {
  const cx = Math.max(0, Math.min(width - 1.001, x));
  const cy = Math.max(0, Math.min(height - 1.001, y));
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
  const sx = cx - x0, sy = cy - y0;
  const i00 = y0 * width + x0, i10 = y0 * width + x1, i01 = y1 * width + x0, i11 = y1 * width + x1;
  const a = field[i00] * (1 - sx) + field[i10] * sx;
  const b = field[i01] * (1 - sx) + field[i11] * sx;
  return a * (1 - sy) + b * sy;
}

const DENSITY_STEP_MIN = 1;
const DENSITY_STEP_MAX = 12;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function densityValueToStep(v) {
  const d = clamp(Number(v) || 0, 0, 1);
  return DENSITY_STEP_MIN + Math.round(d * (DENSITY_STEP_MAX - DENSITY_STEP_MIN));
}

function deriveBaseStepFromDensityField({ width, height, rigidField, softField, threshold, densityField, fallbackStep }) {
  if (!(densityField instanceof Float32Array) || densityField.length < width * height) {
    return fallbackStep;
  }

  let sum = 0;
  let count = 0;
  const occThreshold = Math.max(0.02, threshold * 0.5);
  for (let i = 0; i < width * height; i++) {
    if ((Number(rigidField?.[i]) || 0) < occThreshold && (Number(softField?.[i]) || 0) < occThreshold) continue;
    const d = clamp(Number(densityField[i]) || 0, 0, 1);
    sum += d;
    count += 1;
  }
  if (!count) return fallbackStep;

  const meanDensity = sum / count;
  return densityValueToStep(meanDensity);
}

function buildAdaptiveDensityCells({ width, height, rigidField, softField, threshold, densityField, softMinCellSize = 1, softMaxCellSize = 6, softBoundaryCellCap = 2, softNeighborStepDeltaCap = 0, softThinFeatureCellCap = 0, softBridgeCellCap = 2 }) {
  const minSoftStep = Math.max(1, Math.round(Number(softMinCellSize) || 1));
  const maxSoftStep = Math.max(minSoftStep, Math.round(Number(softMaxCellSize) || 6));
  const boundaryCap = Number.isFinite(Number(softBoundaryCellCap))
    ? Math.max(0, Math.round(Number(softBoundaryCellCap)))
    : 2;
  const neighborDeltaCap = Number.isFinite(Number(softNeighborStepDeltaCap))
    ? Math.max(0, Math.round(Number(softNeighborStepDeltaCap)))
    : 0;
  const thinFeatureCap = Number.isFinite(Number(softThinFeatureCellCap))
    ? Math.max(0, Math.round(Number(softThinFeatureCellCap)))
    : 0;
  const bridgeCap = Number.isFinite(Number(softBridgeCellCap))
    ? Math.max(0, Math.round(Number(softBridgeCellCap)))
    : 3;
  const cw = Math.max(1, width - 1);
  const ch = Math.max(1, height - 1);
  const cellCount = cw * ch;
  const kindGrid = new Uint8Array(cellCount); // 0 empty, 1 rigid, 2 soft
  const stepGrid = new Uint8Array(cellCount);
  const used = new Uint8Array(cellCount);
  const rigidMask = new Uint8Array(cellCount);

  const cIdx = (x, y) => y * cw + x;

  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const i = cIdx(x, y);
      const sx = Math.min(width - 1.001, x + 0.5);
      const sy = Math.min(height - 1.001, y + 0.5);
      const rv = sampleBilinear(rigidField, width, height, sx, sy);
      const sv = sampleBilinear(softField, width, height, sx, sy);
      const kind = rv >= threshold ? 1 : (sv >= threshold ? 2 : 0);
      kindGrid[i] = kind;
      const d = sampleBilinear(densityField, width, height, sx, sy);
      const localStep = densityValueToStep(d);
      stepGrid[i] = kind === 2
        ? Math.min(maxSoftStep, Math.max(localStep, minSoftStep))
        : localStep;
    }
  }

  if (boundaryCap > 0) {
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = cIdx(x, y);
        if (kindGrid[i] !== 2) continue;

        let touchesBoundary = (x === 0 || y === 0 || x === cw - 1 || y === ch - 1);
        if (!touchesBoundary) {
          for (let ny = y - 1; ny <= y + 1 && !touchesBoundary; ny++) {
            for (let nx = x - 1; nx <= x + 1; nx++) {
              if (nx === x && ny === y) continue;
              if (nx < 0 || ny < 0 || nx >= cw || ny >= ch) {
                touchesBoundary = true;
                break;
              }
              if (kindGrid[cIdx(nx, ny)] !== 2) {
                touchesBoundary = true;
                break;
              }
            }
          }
        }

        if (touchesBoundary) {
          stepGrid[i] = Math.max(minSoftStep, Math.min(stepGrid[i], boundaryCap));
        }
      }
    }
  }

  if (thinFeatureCap > 0) {
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = cIdx(x, y);
        if (kindGrid[i] !== 2) continue;

        let softNeighborCount = 0;
        for (let ny = y - 1; ny <= y + 1; ny++) {
          for (let nx = x - 1; nx <= x + 1; nx++) {
            if (nx === x && ny === y) continue;
            if (nx < 0 || ny < 0 || nx >= cw || ny >= ch) continue;
            if (kindGrid[cIdx(nx, ny)] === 2) softNeighborCount += 1;
          }
        }

        // Narrow tendrils/bridges have sparse 8-neighborhood support; cap their
        // primitive scale to preserve painted-shape fit and reduce long rest-span tails.
        if (softNeighborCount <= 5) {
          stepGrid[i] = Math.max(minSoftStep, Math.min(stepGrid[i], thinFeatureCap));
        }
      }
    }
  }

  if (bridgeCap > 0) {
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = cIdx(x, y);
        if (kindGrid[i] !== 2) continue;

        let cardinalSoft = 0;
        if (x > 0 && kindGrid[cIdx(x - 1, y)] === 2) cardinalSoft += 1;
        if (x + 1 < cw && kindGrid[cIdx(x + 1, y)] === 2) cardinalSoft += 1;
        if (y > 0 && kindGrid[cIdx(x, y - 1)] === 2) cardinalSoft += 1;
        if (y + 1 < ch && kindGrid[cIdx(x, y + 1)] === 2) cardinalSoft += 1;

        // 1-cell bridges can look well-supported diagonally while still acting like
        // long tendrils in the solver. Cap primitive scale by cardinal connectivity.
        if (cardinalSoft <= 2) {
          stepGrid[i] = Math.max(minSoftStep, Math.min(stepGrid[i], bridgeCap));
        }
      }
    }
  }

  if (neighborDeltaCap > 0) {
    for (let pass = 0; pass < 3; pass++) {
      let changed = false;
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) {
          const i = cIdx(x, y);
          if (kindGrid[i] !== 2) continue;

          let minNeighborStep = Number.POSITIVE_INFINITY;
          for (let ny = y - 1; ny <= y + 1; ny++) {
            for (let nx = x - 1; nx <= x + 1; nx++) {
              if (nx === x && ny === y) continue;
              if (nx < 0 || ny < 0 || nx >= cw || ny >= ch) continue;
              const ni = cIdx(nx, ny);
              if (kindGrid[ni] !== 2) continue;
              minNeighborStep = Math.min(minNeighborStep, stepGrid[ni] || minSoftStep);
            }
          }

          if (!Number.isFinite(minNeighborStep)) continue;
          const allowed = Math.max(minSoftStep, minNeighborStep + neighborDeltaCap);
          const nextStep = Math.max(minSoftStep, Math.min(stepGrid[i], allowed));
          if (nextStep < stepGrid[i]) {
            stepGrid[i] = nextStep;
            changed = true;
          }
        }
      }
      if (!changed) break;
    }
  }

  const cells = [];

  const canPlace = (kind, x0, y0, size) => {
    if (x0 + size > cw || y0 + size > ch) return false;
    for (let y = y0; y < y0 + size; y++) {
      for (let x = x0; x < x0 + size; x++) {
        const i = cIdx(x, y);
        if (used[i]) return false;
        if (kindGrid[i] !== kind) return false;
        if ((stepGrid[i] || 1) < size) return false;
      }
    }
    return true;
  };

  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const i = cIdx(x, y);
      if (used[i]) continue;
      const kind = kindGrid[i];
      if (!kind) continue;

      const target = Math.max(1, stepGrid[i] || 1);
      const minAllowed = kind === 2 ? minSoftStep : 1;
      let size = target;
      while (size > minAllowed && !canPlace(kind, x, y, size)) size -= 1;
      if (!canPlace(kind, x, y, size)) continue;

      for (let yy = y; yy < y + size; yy++) {
        for (let xx = x; xx < x + size; xx++) {
          const ii = cIdx(xx, yy);
          used[ii] = 1;
          if (kind === 1) rigidMask[ii] = 1;
        }
      }

      cells.push({
        kind: kind === 1 ? 'rigid' : 'soft',
        x,
        y,
        size,
      });
    }
  }

  return {
    cells,
    rigidCells: cells.filter((c) => c.kind === 'rigid'),
    rigidMask: { mask: rigidMask, cols: cw, rows: ch, step: 1 },
  };
}

export function compileFieldToMesh({
  width,
  height,
  rigidField,
  softField,
  dragField = null,
  permeabilityField = null,
  threshold = 0.35,
  density = 1,
  connectivityMode = 'none', // none | largest (strict single connected body)
  minComponentTriangles = 0,
  softInfillMode = 'triangles+cross', // triangles | triangles+cross
  softDensityField = null, // 0..1, controls local rigid+soft primitive size (darker=finer, brighter=coarser)
  softMinCellSize = 1, // lower bound on adaptive soft primitive size (cell units)
  softMaxCellSize = 6, // upper bound on adaptive soft primitive size (cell units)
  softBoundaryCellCap = 2, // cap soft primitive size on paint boundary to avoid seam stretch/fit loss
  softNeighborStepDeltaCap = 1, // max coarse-step delta between neighboring soft cells (0 disables)
  softThinFeatureCellCap = 2, // cap primitive size in narrow/tendril soft regions to preserve shape memory topology
  softBridgeCellCap = 2, // cap primitive size in low-cardinality bridge cells to reduce soft rest-span outliers
}) {
  const infillMode = softInfillMode === 'triangles' ? 'triangles' : 'triangles+cross';
  const fallbackStep = Math.max(1, density | 0);
  const maxAdaptiveStep = Math.max(1, Math.min(width - 1, height - 1));
  const softMinStep = Math.max(1, Math.min(maxAdaptiveStep, Math.round(Number(softMinCellSize) || 1)));
  const softMaxStep = Math.max(softMinStep, Math.min(maxAdaptiveStep, Math.round(Number(softMaxCellSize) || 6)));
  const softNeighborDeltaCap = Number.isFinite(Number(softNeighborStepDeltaCap))
    ? Math.max(0, Math.round(Number(softNeighborStepDeltaCap)))
    : 0;
  const softThinFeatureCap = Number.isFinite(Number(softThinFeatureCellCap))
    ? Math.max(0, Math.round(Number(softThinFeatureCellCap)))
    : 0;
  const softBridgeCap = Number.isFinite(Number(softBridgeCellCap))
    ? Math.max(0, Math.round(Number(softBridgeCellCap)))
    : 3;
  const step = deriveBaseStepFromDensityField({
    width,
    height,
    rigidField,
    softField,
    threshold,
    densityField: softDensityField,
    fallbackStep,
  });
  const nodeMap = new Map();
  const nodes = [];
  const triangles = [];

  const nodeId = (x, y) => {
    const k = `${x},${y}`;
    if (nodeMap.has(k)) return nodeMap.get(k);
    const rid = sampleBilinear(rigidField, width, height, x, y);
    const sof = sampleBilinear(softField, width, height, x, y);
    const drag = dragField ? sampleBilinear(dragField, width, height, x, y) : 0.5;
    const perm = permeabilityField ? sampleBilinear(permeabilityField, width, height, x, y) : 0.5;
    const idx = nodes.length;
    nodes.push({ id: idx, x, y, rigid: rid, soft: sof, drag, permeability: perm });
    nodeMap.set(k, idx);
    return idx;
  };

  const addTri = (ax, ay, bx, by, cx, cy) => {
    const mx = (ax + bx + cx) / 3;
    const my = (ay + by + cy) / 3;
    const rv = sampleBilinear(rigidField, width, height, mx, my);
    const sv = sampleBilinear(softField, width, height, mx, my);

    // Rigid has absolute precedence in overlap zones:
    // if rigid is above threshold, this triangle cannot be soft.
    const rigidSolid = rv >= threshold;
    const softSolid = sv >= threshold;
    if (!rigidSolid && !softSolid) return;

    const kind = rigidSolid ? 'rigid' : 'soft';
    const mag = kind === 'rigid' ? rv : sv;
    triangles.push({
      kind,
      a: nodeId(ax, ay),
      b: nodeId(bx, by),
      c: nodeId(cx, cy),
      strength: mag,
    });
  };

  let rigidMaskOverride = null;
  let adaptive = null;
  const hasDensityMap = (softDensityField instanceof Float32Array) && softDensityField.length >= width * height;

  if (hasDensityMap) {
    adaptive = buildAdaptiveDensityCells({
      width,
      height,
      rigidField,
      softField,
      threshold,
      densityField: softDensityField,
      softMinCellSize: softMinStep,
      softMaxCellSize: softMaxStep,
      softBoundaryCellCap,
      softNeighborStepDeltaCap: softNeighborDeltaCap,
      softThinFeatureCellCap: softThinFeatureCap,
      softBridgeCellCap: softBridgeCap,
    });
    for (const c of adaptive.cells) {
      const x0 = c.x;
      const y0 = c.y;
      const x1 = c.x + c.size;
      const y1 = c.y + c.size;
      addTri(x0, y0, x1, y0, x0, y1);
      addTri(x1, y0, x1, y1, x0, y1);
    }
    rigidMaskOverride = adaptive.rigidMask;
  } else {
    for (let y = 0; y < height - step; y += step) {
      for (let x = 0; x < width - step; x += step) {
        const x1 = x + step;
        const y1 = y + step;
        addTri(x, y, x1, y, x, y1);
        addTri(x1, y, x1, y1, x, y1);
      }
    }
  }

  const filtered = enforceConnectivity({ triangles, mode: connectivityMode, minComponentTriangles });
  const rigidDecomp = extractRigidContoursFromField({
    width,
    height,
    rigidField,
    threshold,
    cellSize: step,
    nodes,
    keptTriangles: filtered.triangles,
    rigidMaskOverride,
  });

  const noOverlap = removeSoftTrianglesOverlappingRigidContours(filtered.triangles, nodes, rigidDecomp.pieces);
  const densityApplied = hasDensityMap
    ? { triangles: noOverlap.triangles, culledSoft: 0, culledRigid: 0 }
    : applyInfillDensityMap(noOverlap.triangles, nodes, {
        width,
        height,
        densityField: softDensityField,
        step,
      });
  const final = enforceConnectivity({ triangles: densityApplied.triangles, mode: connectivityMode, minComponentTriangles });
  const useSoftCrossBeams = infillMode === 'triangles+cross';
  const softCrossBeams = useSoftCrossBeams ? buildSoftCrossBeams(final.triangles, nodes) : [];

  return {
    nodes,
    triangles: final.triangles,
    softCrossBeams,
    rigidPieces: rigidDecomp.pieces,
    meta: {
      width,
      height,
      density: step,
      densitySource: (softDensityField instanceof Float32Array) ? 'map' : 'fixed',
      threshold,
      connectivityMode,
      components: filtered.componentCount,
      keptComponents: filtered.keptComponents,
      componentsFinal: final.componentCount,
      keptComponentsFinal: final.keptComponents,
      rigidTriangles: final.triangles.filter((t) => t.kind === 'rigid').length,
      softTriangles: final.triangles.filter((t) => t.kind === 'soft').length,
      rigidPieces: rigidDecomp.pieces.length,
      softInfillMode: infillMode,
      softCrossBeams: softCrossBeams.length,
      droppedTriangles: triangles.length - final.triangles.length,
      softOverlapTrimmed: noOverlap.removed,
      softDensityCulled: densityApplied.culledSoft,
      rigidDensityCulled: densityApplied.culledRigid,
      softDensityLevels: softDensityField ? 15 : 0,
      softMinCellSize: softMinStep,
      softMaxCellSize: softMaxStep,
      softBoundaryCellCap: Number.isFinite(Number(softBoundaryCellCap)) ? Math.max(0, Math.round(Number(softBoundaryCellCap))) : 2,
      softNeighborStepDeltaCap: softNeighborDeltaCap,
      softThinFeatureCellCap: softThinFeatureCap,
      softBridgeCellCap: softBridgeCap,
    },
  };
}

function enforceConnectivity({ triangles, mode = 'none', minComponentTriangles = 0 }) {
  if (!triangles.length || mode === 'none') {
    return { triangles, componentCount: triangles.length ? 1 : 0, keptComponents: triangles.length ? 1 : 0 };
  }

  const allIndices = triangles.map((_, i) => i);
  const components = collectTriangleComponents(triangles, allIndices);

  // Strict policy: creature topology must be a single connected body.
  const keep = pickComponentsToKeep(components, mode, minComponentTriangles);
  const keepSet = new Set(keep.flat());
  const keptTriangles = triangles.filter((_, idx) => keepSet.has(idx));
  return { triangles: keptTriangles, componentCount: components.length, keptComponents: keep.length };
}

function collectTriangleComponents(triangles, indices) {
  const nodeToTris = new Map();
  for (const idx of indices) {
    const t = triangles[idx];
    for (const n of [t.a, t.b, t.c]) {
      if (!nodeToTris.has(n)) nodeToTris.set(n, []);
      nodeToTris.get(n).push(idx);
    }
  }

  const seen = new Set();
  const components = [];
  for (const startIdx of indices) {
    if (seen.has(startIdx)) continue;
    const stack = [startIdx];
    seen.add(startIdx);
    const comp = [];
    while (stack.length) {
      const triIdx = stack.pop();
      comp.push(triIdx);
      const tri = triangles[triIdx];
      for (const n of [tri.a, tri.b, tri.c]) {
        for (const nextIdx of (nodeToTris.get(n) || [])) {
          if (!seen.has(nextIdx)) {
            seen.add(nextIdx);
            stack.push(nextIdx);
          }
        }
      }
    }
    components.push(comp);
  }

  components.sort((a, b) => b.length - a.length);
  return components;
}

function pickComponentsToKeep(components, mode, minComponentTriangles) {
  if (!components.length) return [];
  if (mode === 'largest') return [components[0]];
  return components;
}

function extractRigidContoursFromAdaptiveCells({ rigidCells, nodes, keptTriangles, threshold }) {
  const cells = (rigidCells || []).filter((c) => c && c.kind === 'rigid' && c.size > 0);
  if (!cells.length) return { pieces: [] };

  const segMap = new Map();
  const segKey = (ax, ay, bx, by) => {
    if (ax < bx || (ax === bx && ay <= by)) return `${ax},${ay}|${bx},${by}`;
    return `${bx},${by}|${ax},${ay}`;
  };

  const addSeg = (ax, ay, bx, by) => {
    const key = segKey(ax, ay, bx, by);
    const entry = segMap.get(key);
    if (!entry) {
      segMap.set(key, { count: 1, ax, ay, bx, by });
    } else {
      entry.count += 1;
    }
  };

  for (const c of cells) {
    const x0 = c.x;
    const y0 = c.y;
    const x1 = c.x + c.size;
    const y1 = c.y + c.size;

    for (let x = x0; x < x1; x++) {
      addSeg(x, y0, x + 1, y0);
      addSeg(x + 1, y1, x, y1);
    }
    for (let y = y0; y < y1; y++) {
      addSeg(x1, y, x1, y + 1);
      addSeg(x0, y + 1, x0, y);
    }
  }

  const boundary = [...segMap.values()].filter((e) => e.count === 1);
  if (!boundary.length) return { pieces: [] };

  const outByFrom = new Map();
  const fromKey = (x, y) => `${x},${y}`;
  for (let i = 0; i < boundary.length; i++) {
    const e = boundary[i];
    const fk = fromKey(e.ax, e.ay);
    if (!outByFrom.has(fk)) outByFrom.set(fk, []);
    outByFrom.get(fk).push({ id: i, ax: e.ax, ay: e.ay, bx: e.bx, by: e.by });
  }

  const used = new Uint8Array(boundary.length);
  const loops = [];

  for (let i = 0; i < boundary.length; i++) {
    if (used[i]) continue;
    const start = boundary[i];
    const loop = [{ x: start.ax, y: start.ay }];
    let cx = start.ax;
    let cy = start.ay;
    let nx = start.bx;
    let ny = start.by;
    used[i] = 1;
    let guard = boundary.length * 4 + 32;

    while (guard-- > 0) {
      loop.push({ x: nx, y: ny });
      cx = nx;
      cy = ny;
      if (cx === start.ax && cy === start.ay) break;
      const fk = fromKey(cx, cy);
      const options = (outByFrom.get(fk) || []).filter((e) => !used[e.id]);
      if (!options.length) break;
      options.sort((a, b) => {
        const aa = Math.atan2(a.by - a.ay, a.bx - a.ax);
        const bb = Math.atan2(b.by - b.ay, b.bx - b.ax);
        return aa - bb;
      });
      const e = options[0];
      used[e.id] = 1;
      nx = e.bx;
      ny = e.by;
    }

    if (loop.length >= 4 && loop[0].x === loop[loop.length - 1].x && loop[0].y === loop[loop.length - 1].y) {
      loop.pop();
      loops.push(loop);
    }
  }

  if (!loops.length) return { pieces: [] };

  const keptRigidNodeIds = new Set();
  for (const t of keptTriangles || []) {
    if (t.kind !== 'rigid') continue;
    keptRigidNodeIds.add(t.a);
    keptRigidNodeIds.add(t.b);
    keptRigidNodeIds.add(t.c);
  }

  const pieces = [];
  for (let li = 0; li < loops.length; li++) {
    let hull = simplifyCollinear(loops[li]);
    hull = simplifyDouglasPeucker(hull, 0.05);
    hull = simplifyCollinear(hull);

    if (hull.length < 3) continue;
    if (signedPolygonArea(hull) < 0) hull = [...hull].reverse();

    const sourceNodeIds = [];
    for (const n of nodes || []) {
      if (!Number.isFinite(n?.x) || !Number.isFinite(n?.y)) continue;
      if ((Number(n.rigid) || 0) < threshold * 0.75) continue;
      if (!pointInPolygon(n.x, n.y, hull)) continue;
      sourceNodeIds.push(n.id);
    }

    if (keptRigidNodeIds.size > 0) {
      const touchesKeptRigid = sourceNodeIds.some((id) => keptRigidNodeIds.has(id));
      if (!touchesKeptRigid) continue;
    }

    pieces.push({
      id: `rigid_piece_${pieces.length}`,
      compoundId: `rigid_compound_${li}`,
      hull,
      sourceNodeIds,
    });
  }

  return { pieces };
}

function extractRigidContoursFromField({ width, height, rigidField, threshold, cellSize, nodes, keptTriangles, rigidMaskOverride = null }) {
  let step = Math.max(1, cellSize | 0);
  let cols = Math.max(1, Math.ceil(width / step));
  let rows = Math.max(1, Math.ceil(height / step));

  let mask = null;
  let usingMaskOverride = false;
  if (rigidMaskOverride && rigidMaskOverride.mask instanceof Uint8Array) {
    const m = rigidMaskOverride.mask;
    const c = Number(rigidMaskOverride.cols) | 0;
    const r = Number(rigidMaskOverride.rows) | 0;
    const s = Number(rigidMaskOverride.step) || 1;
    if (c > 0 && r > 0 && m.length >= c * r) {
      mask = m;
      cols = c;
      rows = r;
      step = Math.max(1, s);
      usingMaskOverride = true;
    }
  }

  if (!mask) {
    mask = new Uint8Array(cols * rows);
    const mIdx = (x, y) => y * cols + x;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const sx = Math.min(width - 0.501, x * step + step * 0.5);
        const sy = Math.min(height - 0.501, y * step + step * 0.5);
        const rv = sampleBilinear(rigidField, width, height, sx, sy);
        if (rv >= threshold) mask[mIdx(x, y)] = 1;
      }
    }
  }

  const components = collectMaskComponents(mask, cols, rows);
  if (!components.length) return { pieces: [] };

  const keptRigidNodeIds = new Set();
  for (const t of keptTriangles || []) {
    if (t.kind !== 'rigid') continue;
    keptRigidNodeIds.add(t.a);
    keptRigidNodeIds.add(t.b);
    keptRigidNodeIds.add(t.c);
  }

  const pieces = [];
  for (let ci = 0; ci < components.length; ci++) {
    const cellSet = components[ci];
    const loops = traceComponentLoops(cellSet, step);
    if (!loops.length) continue;

    let bestLoop = loops[0];
    let bestArea = Math.abs(signedPolygonArea(bestLoop));
    for (let i = 1; i < loops.length; i++) {
      const loop = loops[i];
      const area = Math.abs(signedPolygonArea(loop));
      if (area > bestArea) {
        bestArea = area;
        bestLoop = loop;
      }
    }

    let hull = simplifyCollinear(bestLoop);
    const simplifyEps = usingMaskOverride ? 0.08 : Math.max(0.75, step * 0.42);
    hull = simplifyDouglasPeucker(hull, simplifyEps);
    hull = simplifyCollinear(hull);

    if (hull.length < 3) continue;
    if (signedPolygonArea(hull) < 0) hull = [...hull].reverse();

    const sourceNodeIds = [];
    for (const n of nodes) {
      if (!Number.isFinite(n?.x) || !Number.isFinite(n?.y)) continue;
      if ((Number(n.rigid) || 0) < threshold * 0.75) continue;
      if (!pointInPolygon(n.x, n.y, hull)) continue;
      sourceNodeIds.push(n.id);
    }

    if (keptRigidNodeIds.size > 0) {
      const touchesKeptRigid = sourceNodeIds.some((id) => keptRigidNodeIds.has(id));
      if (!touchesKeptRigid) continue;
    }

    pieces.push({
      id: `rigid_piece_${pieces.length}`,
      compoundId: `rigid_compound_${ci}`,
      hull,
      sourceNodeIds,
    });
  }

  return { pieces };
}

function removeSoftTrianglesOverlappingRigidContours(triangles, nodes, rigidPieces) {
  const hulls = (rigidPieces || [])
    .map((p) => p?.hull)
    .filter((h) => Array.isArray(h) && h.length >= 3);
  if (!hulls.length) return { triangles, removed: 0 };

  const keep = [];
  let removed = 0;

  for (const t of triangles) {
    if (!t || t.kind !== 'soft') {
      keep.push(t);
      continue;
    }
    const a = nodes[t.a];
    const b = nodes[t.b];
    const c = nodes[t.c];
    if (!a || !b || !c) continue;

    const probes = [
      { x: a.x, y: a.y },
      { x: b.x, y: b.y },
      { x: c.x, y: c.y },
      { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 },
      { x: (b.x + c.x) * 0.5, y: (b.y + c.y) * 0.5 },
      { x: (c.x + a.x) * 0.5, y: (c.y + a.y) * 0.5 },
      { x: (a.x + b.x + c.x) / 3, y: (a.y + b.y + c.y) / 3 },
    ];

    let overlapsRigid = false;
    for (const hull of hulls) {
      if (probes.some((p) => pointInPolygon(p.x, p.y, hull))) {
        overlapsRigid = true;
        break;
      }
    }

    if (overlapsRigid) removed += 1;
    else keep.push(t);
  }

  return { triangles: keep, removed };
}

function hash01FromTri(a, b, c) {
  const x = [a, b, c].sort((m, n) => m - n);
  let h = ((x[0] * 73856093) ^ (x[1] * 19349663) ^ (x[2] * 83492791)) >>> 0;
  h = (Math.imul(h ^ (h >>> 16), 2246822519) + 3266489917) >>> 0;
  return h / 4294967295;
}

function applyInfillDensityMap(triangles, nodes, { width, height, densityField, step }) {
  if (!(densityField instanceof Float32Array) || densityField.length < width * height) {
    return { triangles, culledSoft: 0, culledRigid: 0 };
  }

  const triIdsByKind = { soft: [], rigid: [] };
  for (let i = 0; i < triangles.length; i++) {
    const k = triangles[i]?.kind;
    if (k === 'soft' || k === 'rigid') triIdsByKind[k].push(i);
  }
  if (!triIdsByKind.soft.length && !triIdsByKind.rigid.length) {
    return { triangles, culledSoft: 0, culledRigid: 0 };
  }

  const buildEdgeCounts = (ids) => {
    const edgeCount = new Map();
    const addEdge = (u, v) => {
      const key = u < v ? `${u}:${v}` : `${v}:${u}`;
      edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
    };
    for (const ti of ids) {
      const t = triangles[ti];
      addEdge(t.a, t.b);
      addEdge(t.b, t.c);
      addEdge(t.c, t.a);
    }
    return edgeCount;
  };

  const edgeCounts = {
    soft: buildEdgeCounts(triIdsByKind.soft),
    rigid: buildEdgeCounts(triIdsByKind.rigid),
  };

  const keep = [];
  let culledSoft = 0;
  let culledRigid = 0;

  for (const t of triangles) {
    if (!t || (t.kind !== 'soft' && t.kind !== 'rigid')) {
      keep.push(t);
      continue;
    }

    const counts = edgeCounts[t.kind];
    const edgeKeys = [
      t.a < t.b ? `${t.a}:${t.b}` : `${t.b}:${t.a}`,
      t.b < t.c ? `${t.b}:${t.c}` : `${t.c}:${t.b}`,
      t.c < t.a ? `${t.c}:${t.a}` : `${t.a}:${t.c}`,
    ];
    const isBoundaryTri = edgeKeys.some((k) => (counts.get(k) || 0) <= 1);
    if (isBoundaryTri) {
      keep.push(t);
      continue;
    }

    const a = nodes[t.a];
    const b = nodes[t.b];
    const c = nodes[t.c];
    const cx = (a.x + b.x + c.x) / 3;
    const cy = (a.y + b.y + c.y) / 3;

    const densCenter = Math.max(0, Math.min(1, sampleBilinear(densityField, width, height, cx, cy)));
    const gx = sampleBilinear(densityField, width, height, cx + step, cy) - sampleBilinear(densityField, width, height, cx - step, cy);
    const gy = sampleBilinear(densityField, width, height, cx, cy + step) - sampleBilinear(densityField, width, height, cx, cy - step);
    const grad = Math.hypot(gx, gy) * 0.5;

    // Local step target from green intensity:
    // darker green => denser => smaller local step, brighter => sparser => larger local step.
    const localStepTarget = densityValueToStep(densCenter);
    const baseStep = Math.max(1, step | 0);

    // If local step is denser-or-equal than base, keep all.
    // Only sparse regions (> base step) are probabilistically thinned.
    let keepProb = 1;
    if (localStepTarget > baseStep) {
      keepProb = Math.min(1, (baseStep * baseStep) / (localStepTarget * localStepTarget));
    }

    // Keep a transition belt where gradients are high to avoid seam tearing.
    const transitionBoost = Math.max(0, Math.min(1, grad * 1.8));
    keepProb = Math.max(keepProb, 0.62 * transitionBoost + 0.28);
    keepProb = Math.max(0.06, Math.min(1, keepProb));

    const h = hash01FromTri(t.a, t.b, t.c);
    if (h <= keepProb) keep.push(t);
    else if (t.kind === 'soft') culledSoft += 1;
    else culledRigid += 1;
  }

  return { triangles: keep, culledSoft, culledRigid };
}
function buildSoftCrossBeams(triangles, nodes) {
  const softTris = (triangles || []).filter((t) => t?.kind === 'soft');
  if (!softTris.length) return [];

  const edgeKey = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);
  const parseEdge = (k) => k.split(':').map((v) => Number(v));

  const existingEdges = new Set();
  const edgeToTris = new Map();

  for (let ti = 0; ti < softTris.length; ti++) {
    const t = softTris[ti];
    for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
      const k = edgeKey(u, v);
      existingEdges.add(k);
      if (!edgeToTris.has(k)) edgeToTris.set(k, []);
      edgeToTris.get(k).push(ti);
    }
  }

  const beams = [];
  const beamSet = new Set();

  for (const [sharedKey, triIds] of edgeToTris.entries()) {
    if (!Array.isArray(triIds) || triIds.length !== 2) continue;
    const [s0, s1] = parseEdge(sharedKey);
    const tA = softTris[triIds[0]];
    const tB = softTris[triIds[1]];
    if (!tA || !tB) continue;

    const idsA = [tA.a, tA.b, tA.c];
    const idsB = [tB.a, tB.b, tB.c];
    const oppA = idsA.find((id) => id !== s0 && id !== s1);
    const oppB = idsB.find((id) => id !== s0 && id !== s1);
    if (!Number.isInteger(oppA) || !Number.isInteger(oppB) || oppA === oppB) continue;

    const beamKey = edgeKey(oppA, oppB);
    if (existingEdges.has(beamKey) || beamSet.has(beamKey)) continue;

    const a = nodes[oppA];
    const b = nodes[oppB];
    if (!a || !b) continue;
    const len = Math.hypot((b.x - a.x), (b.y - a.y));
    if (!(len > 1e-6)) continue;

    beamSet.add(beamKey);
    beams.push([oppA, oppB]);
  }

  return beams;
}

function collectMaskComponents(mask, cols, rows) {
  const comps = [];
  const seen = new Uint8Array(mask.length);
  const idx = (x, y) => y * cols + x;

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const start = idx(x, y);
      if (!mask[start] || seen[start]) continue;
      const stack = [[x, y]];
      seen[start] = 1;
      const cells = new Set();

      while (stack.length) {
        const [cx, cy] = stack.pop();
        cells.add(`${cx},${cy}`);
        const n4 = [[cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]];
        for (const [nx, ny] of n4) {
          if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
          const ni = idx(nx, ny);
          if (!mask[ni] || seen[ni]) continue;
          seen[ni] = 1;
          stack.push([nx, ny]);
        }
      }

      if (cells.size) comps.push(cells);
    }
  }

  comps.sort((a, b) => b.size - a.size);
  return comps;
}

function traceComponentLoops(cellSet, step) {
  const edgeMap = new Map();
  const addEdge = (x0, y0, x1, y1) => {
    const p0 = `${x0},${y0}`;
    const p1 = `${x1},${y1}`;
    if (!edgeMap.has(p0)) edgeMap.set(p0, []);
    edgeMap.get(p0).push({ from: p0, to: p1 });
  };

  const hasCell = (x, y) => cellSet.has(`${x},${y}`);

  for (const key of cellSet) {
    const [cx, cy] = key.split(',').map((v) => Number(v));
    const x0 = cx * step;
    const y0 = cy * step;
    const x1 = x0 + step;
    const y1 = y0 + step;

    if (!hasCell(cx, cy - 1)) addEdge(x0, y0, x1, y0); // top
    if (!hasCell(cx + 1, cy)) addEdge(x1, y0, x1, y1); // right
    if (!hasCell(cx, cy + 1)) addEdge(x1, y1, x0, y1); // bottom
    if (!hasCell(cx - 1, cy)) addEdge(x0, y1, x0, y0); // left
  }

  const loops = [];
  while (true) {
    const startEntry = [...edgeMap.entries()].find(([, arr]) => arr.length > 0);
    if (!startEntry) break;

    let [start] = startEntry;
    let current = start;
    const loop = [];
    const guard = edgeMap.size * 8 + 64;
    let steps = 0;

    while (steps++ < guard) {
      const outgoing = edgeMap.get(current);
      if (!outgoing || outgoing.length === 0) break;
      const edge = outgoing.pop();
      const [x, y] = edge.from.split(',').map((v) => Number(v));
      loop.push({ x, y });
      current = edge.to;
      if (current === start) break;
    }

    if (loop.length >= 3) loops.push(loop);
  }

  return loops;
}

function signedPolygonArea(poly) {
  if (!poly || poly.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    s += p.x * q.y - q.x * p.y;
  }
  return s * 0.5;
}

function simplifyCollinear(poly) {
  if (!poly || poly.length < 3) return poly || [];
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[(i - 1 + poly.length) % poly.length];
    const b = poly[i];
    const c = poly[(i + 1) % poly.length];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(cross) <= 1e-6) continue;
    out.push(b);
  }
  return out.length >= 3 ? out : poly;
}

function simplifyDouglasPeucker(poly, epsilon) {
  if (!poly || poly.length < 4) return poly || [];
  const open = [...poly, poly[0]];
  const keep = new Uint8Array(open.length);
  keep[0] = 1;
  keep[open.length - 1] = 1;

  const stack = [[0, open.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    let maxDist = -1;
    let maxIdx = -1;
    const p0 = open[a];
    const p1 = open[b];
    for (let i = a + 1; i < b; i++) {
      const d = pointSegmentDistance(open[i], p0, p1);
      if (d > maxDist) {
        maxDist = d;
        maxIdx = i;
      }
    }
    if (maxDist > epsilon && maxIdx > a && maxIdx < b) {
      keep[maxIdx] = 1;
      stack.push([a, maxIdx], [maxIdx, b]);
    }
  }

  const simplified = [];
  for (let i = 0; i < open.length - 1; i++) {
    if (keep[i]) simplified.push(open[i]);
  }
  return simplified.length >= 3 ? simplified : poly;
}

function pointSegmentDistance(p, a, b) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const apx = p.x - a.x;
  const apy = p.y - a.y;
  const den = Math.max(1e-9, abx * abx + aby * aby);
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / den));
  const cx = a.x + abx * t;
  const cy = a.y + aby * t;
  return Math.hypot(p.x - cx, p.y - cy);
}

function pointInPolygon(px, py, poly, eps = 1e-6) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;

    if (pointOnSegment(px, py, xi, yi, xj, yj, eps)) return true;

    const intersect = ((yi > py) !== (yj > py))
      && (px < ((xj - xi) * (py - yi)) / Math.max(1e-9, (yj - yi)) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function pointOnSegment(px, py, ax, ay, bx, by, eps = 1e-6) {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const cross = Math.abs(abx * apy - aby * apx);
  if (cross > eps) return false;

  const dot = apx * abx + apy * aby;
  if (dot < -eps) return false;

  const len2 = abx * abx + aby * aby;
  if (dot > len2 + eps) return false;

  return true;
}

function centroid(points) {
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / Math.max(1, points.length), y: sy / Math.max(1, points.length) };
}
