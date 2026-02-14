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
}) {
  const step = Math.max(1, density | 0);
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

  for (let y = 0; y < height - step; y += step) {
    for (let x = 0; x < width - step; x += step) {
      const x1 = x + step;
      const y1 = y + step;
      addTri(x, y, x1, y, x, y1);
      addTri(x1, y, x1, y1, x, y1);
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
  });

  return {
    nodes,
    triangles: filtered.triangles,
    rigidPieces: rigidDecomp.pieces,
    rigidWelds: rigidDecomp.welds,
    meta: {
      width,
      height,
      density: step,
      threshold,
      connectivityMode,
      components: filtered.componentCount,
      keptComponents: filtered.keptComponents,
      rigidTriangles: filtered.triangles.filter((t) => t.kind === 'rigid').length,
      softTriangles: filtered.triangles.filter((t) => t.kind === 'soft').length,
      rigidPieces: rigidDecomp.pieces.length,
      rigidWelds: rigidDecomp.welds.length,
      droppedTriangles: triangles.length - filtered.triangles.length,
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

function extractRigidContoursFromField({ width, height, rigidField, threshold, cellSize, nodes, keptTriangles }) {
  const step = Math.max(1, cellSize | 0);
  const cols = Math.max(1, Math.ceil(width / step));
  const rows = Math.max(1, Math.ceil(height / step));

  const mask = new Uint8Array(cols * rows);
  const mIdx = (x, y) => y * cols + x;

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const sx = Math.min(width - 0.501, x * step + step * 0.5);
      const sy = Math.min(height - 0.501, y * step + step * 0.5);
      const rv = sampleBilinear(rigidField, width, height, sx, sy);
      if (rv >= threshold) mask[mIdx(x, y)] = 1;
    }
  }

  const components = collectMaskComponents(mask, cols, rows);
  if (!components.length) return { pieces: [], welds: [] };

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
    hull = simplifyDouglasPeucker(hull, Math.max(0.75, step * 0.42));
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

  return { pieces, welds: [] };
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

function pointInPolygon(px, py, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    const intersect = ((yi > py) !== (yj > py))
      && (px < ((xj - xi) * (py - yi)) / Math.max(1e-9, (yj - yi)) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function decomposeRigidTriangles(triangles, nodes) {
  const rigidIndices = [];
  for (let i = 0; i < triangles.length; i++) {
    if (triangles[i].kind === 'rigid') rigidIndices.push(i);
  }
  if (!rigidIndices.length) return { pieces: [], welds: [] };

  const rigidComps = collectTriangleComponents(triangles, rigidIndices);
  const pieces = [];
  const welds = [];

  for (let compIdx = 0; compIdx < rigidComps.length; compIdx++) {
    const compTriIdx = rigidComps[compIdx];
    const triPieces = [];
    for (const ti of compTriIdx) {
      const t = triangles[ti];
      const ids = [t.a, t.b, t.c];
      triPieces.push({
        nodeIds: new Set(ids),
        area: triangleAreaAbs(nodes[ids[0]], nodes[ids[1]], nodes[ids[2]]),
      });
    }

    const mergedPieces = mergeConvexPieces(triPieces, nodes);
    const localPieces = [];

    for (const mp of mergedPieces) {
      const hullWithIds = convexHullWithIds([...mp.nodeIds].map((id) => ({ id, x: nodes[id].x, y: nodes[id].y })));
      if (hullWithIds.length < 3) continue;
      const pieceIndex = pieces.length;
      pieces.push({
        id: `rigid_piece_${pieceIndex}`,
        compoundId: `rigid_compound_${compIdx}`,
        hull: hullWithIds.map((p) => ({ x: p.x, y: p.y })),
        sourceNodeIds: hullWithIds.map((p) => p.id),
      });
      localPieces.push({
        globalIndex: pieceIndex,
        nodeIds: new Set(mp.nodeIds),
        hullSourceIds: hullWithIds.map((p) => p.id),
      });
    }

    for (let i = 0; i < localPieces.length; i++) {
      for (let j = i + 1; j < localPieces.length; j++) {
        const a = localPieces[i];
        const b = localPieces[j];
        const shared = [...a.nodeIds].filter((id) => b.nodeIds.has(id));
        if (shared.length < 2) continue;

        const aHullSet = new Set(a.hullSourceIds);
        const bHullSet = new Set(b.hullSourceIds);
        const sharedHull = shared.filter((id) => aHullSet.has(id) && bHullSet.has(id));

        if (sharedHull.length >= 2) {
          let bestA = sharedHull[0];
          let bestB = sharedHull[1];
          let bestDist2 = -1;
          for (let p = 0; p < sharedHull.length; p++) {
            for (let q = p + 1; q < sharedHull.length; q++) {
              const idA = sharedHull[p];
              const idB = sharedHull[q];
              const dx = nodes[idA].x - nodes[idB].x;
              const dy = nodes[idA].y - nodes[idB].y;
              const d2 = dx * dx + dy * dy;
              if (d2 > bestDist2) {
                bestDist2 = d2;
                bestA = idA;
                bestB = idB;
              }
            }
          }

          const a0 = a.hullSourceIds.indexOf(bestA);
          const a1 = a.hullSourceIds.indexOf(bestB);
          const b0 = b.hullSourceIds.indexOf(bestA);
          const b1 = b.hullSourceIds.indexOf(bestB);
          if ([a0, a1, b0, b1].some((x) => x < 0)) continue;

          welds.push({ a: a.globalIndex, b: b.globalIndex, a0, a1, b0, b1 });
          continue;
        }

        const aHull = pieces[a.globalIndex].hull;
        const bHull = pieces[b.globalIndex].hull;
        const ca = centroid(aHull);
        const cb = centroid(bHull);
        const ea = nearestHullEdgeForPoint(aHull, cb.x, cb.y);
        const eb = nearestHullEdgeForPoint(bHull, ca.x, ca.y);
        welds.push({
          a: a.globalIndex,
          b: b.globalIndex,
          a0: ea.vA,
          a1: ea.vB,
          b0: eb.vA,
          b1: eb.vB,
        });
      }
    }
  }

  return { pieces, welds };
}

function mergeConvexPieces(triPieces, nodes) {
  const pieces = triPieces.map((p) => ({ nodeIds: new Set(p.nodeIds), area: p.area }));
  let changed = true;
  while (changed) {
    changed = false;
    outer: for (let i = 0; i < pieces.length; i++) {
      for (let j = i + 1; j < pieces.length; j++) {
        const a = pieces[i];
        const b = pieces[j];
        const shared = [...a.nodeIds].filter((id) => b.nodeIds.has(id));
        if (shared.length < 2) continue;

        const mergedIds = new Set([...a.nodeIds, ...b.nodeIds]);
        const hull = convexHull([...mergedIds].map((id) => ({ x: nodes[id].x, y: nodes[id].y })));
        const hullArea = polygonAreaAbs(hull);
        const sumArea = a.area + b.area;
        const eps = Math.max(1e-4, sumArea * 0.02);
        const convexCompatible = Math.abs(hullArea - sumArea) <= eps;
        const smallSliverMerge = Math.min(a.area, b.area) <= sumArea * 0.24 && hullArea <= sumArea * 1.22;
        if (convexCompatible || smallSliverMerge) {
          pieces[i] = { nodeIds: mergedIds, area: sumArea };
          pieces.splice(j, 1);
          changed = true;
          break outer;
        }
      }
    }
  }
  return pieces;
}

function nearestHullEdgeForPoint(hull, px, py) {
  if (!hull?.length) return { vA: 0, vB: 0 };
  if (hull.length === 1) return { vA: 0, vB: 0 };
  let bestIdx = 0;
  let bestD2 = Number.POSITIVE_INFINITY;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const cp = closestPointOnSegment(px, py, a.x, a.y, b.x, b.y);
    const dx = px - cp.x;
    const dy = py - cp.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      bestIdx = i;
    }
  }
  return { vA: bestIdx, vB: (bestIdx + 1) % hull.length };
}

function closestPointOnSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const den = Math.max(1e-9, abx * abx + aby * aby);
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / den));
  return { x: ax + abx * t, y: ay + aby * t };
}

function convexHullWithIds(pointsWithIds) {
  if (!pointsWithIds || pointsWithIds.length < 3) return pointsWithIds || [];
  const pts = [...pointsWithIds]
    .map((p) => ({ id: p.id, x: p.x, y: p.y }))
    .sort((a, b) => (a.x - b.x) || (a.y - b.y));

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
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function convexHull(points) {
  if (!points || points.length < 3) return points || [];
  const pts = [...points]
    .map((p) => ({ x: p.x, y: p.y }))
    .sort((a, b) => (a.x - b.x) || (a.y - b.y));

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
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function triangleAreaAbs(a, b, c) {
  return Math.abs((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) * 0.5);
}

function polygonAreaAbs(poly) {
  if (!poly || poly.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    s += p.x * q.y - q.x * p.y;
  }
  return Math.abs(s * 0.5);
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
