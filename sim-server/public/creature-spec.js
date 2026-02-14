export const CREATURE_SPEC_VERSION = 'creature-spec.v1';

export function createCreatureSpecFromMesh(mesh, options = {}) {
  return {
    schemaVersion: CREATURE_SPEC_VERSION,
    createdAt: new Date().toISOString(),
    name: options.name || 'unnamed-creature',
    space: {
      width: mesh?.meta?.width || options.width || 128,
      height: mesh?.meta?.height || options.height || 128,
    },
    mesh: {
      nodes: mesh?.nodes || [],
      triangles: mesh?.triangles || [],
      meta: mesh?.meta || {},
    },
  };
}

export function parseCreatureSpec(jsonText) {
  const obj = JSON.parse(jsonText);
  if (!obj || typeof obj !== 'object') throw new Error('Invalid JSON object');
  if (obj.schemaVersion !== CREATURE_SPEC_VERSION) throw new Error(`Unsupported schemaVersion: ${obj.schemaVersion}`);
  if (!obj.mesh || !Array.isArray(obj.mesh.nodes) || !Array.isArray(obj.mesh.triangles)) {
    throw new Error('CreatureSpec missing mesh.nodes/mesh.triangles');
  }
  return obj;
}

export function buildBodiesFromCreatureSpec(spec, n, controls) {
  const srcW = Math.max(1, spec.space?.width || n);
  const srcH = Math.max(1, spec.space?.height || n);
  const sx = n / srcW;
  const sy = n / srcH;

  const nodes = spec.mesh.nodes.map((p, i) => ({
    id: i,
    x: p.x * sx,
    y: p.y * sy,
    rigid: p.rigid || 0,
    soft: p.soft || 0,
  }));

  const triByKind = { rigid: [], soft: [] };
  for (const t of spec.mesh.triangles) {
    if (t.kind === 'rigid') triByKind.rigid.push(t);
    else triByKind.soft.push(t);
  }

  // Preserve imported geometry fidelity by compiling both rigid and soft triangle regions into
  // explicit node/spring meshes (instead of convex-hulling rigid regions into one regular polygon).
  const rigidLike = buildSoftFromTriangles(triByKind.rigid, nodes, controls, {
    clusterOffset: 10000,
    mass: controls.massHeavy,
    radius: 1.4,
    digestEnabled: false,
    digestRGB: [1, 1, 1],
  });
  const soft = buildSoftFromTriangles(triByKind.soft, nodes, controls, {
    clusterOffset: 0,
    mass: controls.massSoft,
    radius: 1.6,
    digestEnabled: false,
    digestRGB: [1, 1, 1],
  });

  return {
    rigid: [],
    soft: {
      nodes: [...rigidLike.nodes, ...soft.nodes],
      springs: [...rigidLike.springs, ...soft.springs],
    },
    hybrid: [],
  };
}

function buildSoftFromTriangles(tris, nodes, controls, opts = {}) {
  const comps = triangleComponents(tris);
  const softNodes = [];
  const springs = [];
  let clusterId = opts.clusterOffset || 0;
  const nodeMass = opts.mass ?? controls.massSoft;
  const nodeRadius = opts.radius ?? 1.6;
  const digestEnabled = !!opts.digestEnabled;
  const digestRGB = opts.digestRGB || [1, 1, 1];
  for (const comp of comps) {
    const pointIds = new Set();
    for (const t of comp) {
      pointIds.add(t.a); pointIds.add(t.b); pointIds.add(t.c);
    }
    const ids = [...pointIds];
    const base = softNodes.length;
    const remap = new Map();
    ids.forEach((id, i) => {
      remap.set(id, base + i);
      const p = nodes[id];
      softNodes.push({
        x: p.x,
        y: p.y,
        vx: 0,
        vy: 0,
        mass: nodeMass,
        r: nodeRadius,
        clusterId,
        digestEnabled,
        digestRGB: [...digestRGB],
      });
    });

    const edgeCount = new Map();
    for (const t of comp) {
      for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
        const k = u < v ? `${u}-${v}` : `${v}-${u}`;
        edgeCount.set(k, (edgeCount.get(k) || 0) + 1);
      }
    }

    for (const [k, c] of edgeCount.entries()) {
      const [ua, ub] = k.split('-').map((x) => Number(x));
      const a = softNodes[remap.get(ua)];
      const b = softNodes[remap.get(ub)];
      const rest = Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y));
      const boundary = c === 1;
      springs.push([
        remap.get(ua),
        remap.get(ub),
        rest,
        boundary ? 1 : 0,
        boundary ? [1, 1, 1] : [0, 0, 0],
      ]);
    }
    clusterId += 1;
  }
  return { nodes: softNodes, springs };
}

function triangleComponents(tris) {
  if (!tris.length) return [];
  const nodeToTri = new Map();
  for (let i = 0; i < tris.length; i++) {
    const t = tris[i];
    for (const n of [t.a, t.b, t.c]) {
      if (!nodeToTri.has(n)) nodeToTri.set(n, []);
      nodeToTri.get(n).push(i);
    }
  }
  const seen = new Uint8Array(tris.length);
  const comps = [];
  for (let i = 0; i < tris.length; i++) {
    if (seen[i]) continue;
    const stack = [i];
    seen[i] = 1;
    const comp = [];
    while (stack.length) {
      const ti = stack.pop();
      comp.push(tris[ti]);
      for (const n of [tris[ti].a, tris[ti].b, tris[ti].c]) {
        for (const ni of nodeToTri.get(n) || []) {
          if (!seen[ni]) {
            seen[ni] = 1;
            stack.push(ni);
          }
        }
      }
    }
    comps.push(comp);
  }
  return comps;
}

function approximateHullSides(points) {
  if (points.length < 3) return 3;
  const sorted = [...points].sort((a, b) => (a.x - b.x) || (a.y - b.y));
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return Math.max(3, lower.length + upper.length);
}
