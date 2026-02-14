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
    const mag = Math.max(rv, sv);
    if (mag < threshold) return;
    const kind = rv >= sv ? 'rigid' : 'soft';
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

  return {
    nodes,
    triangles: filtered.triangles,
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
  // Keep exactly the largest connected component when connectivity mode is enabled.
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
  if (mode === 'largest') {
    return [components[0]];
  }
  return components;
}
