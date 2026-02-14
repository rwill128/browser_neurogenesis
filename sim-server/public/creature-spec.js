export const CREATURE_SPEC_VERSION = 'creature-spec.v1';

const EDGE_BODY_BLOCK = 1;

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
  for (const t of spec.mesh.triangles || []) {
    if (t.kind === 'rigid') triByKind.rigid.push(t);
    else if (t.kind === 'soft') triByKind.soft.push(t);
  }

  const rigidBuild = buildRigidClusters(triByKind.rigid, nodes, controls);
  const softBuild = buildSoftFromTriangles(triByKind.soft, nodes, controls, {
    clusterOffset: 0,
    mass: controls.massSoft,
    radius: 1.6,
    digestEnabled: false,
    digestRGB: [1, 1, 1],
  });

  const hybrid = buildRigidSoftHybridLinks({
    rigidComps: rigidBuild.components,
    softComps: softBuild.components,
    rigidBodies: rigidBuild.rigid,
    softNodeRemap: softBuild.globalRemap,
    nodes,
  });

  return {
    rigid: rigidBuild.rigid,
    soft: {
      nodes: softBuild.nodes,
      springs: softBuild.springs,
    },
    hybrid,
  };
}

function buildRigidClusters(tris, nodes, controls) {
  const comps = triangleComponents(tris);
  const rigid = [];
  const components = [];
  for (let ci = 0; ci < comps.length; ci++) {
    const comp = comps[ci];
    const pointIds = new Set();
    for (const t of comp) {
      pointIds.add(t.a); pointIds.add(t.b); pointIds.add(t.c);
    }
    const pts = [...pointIds].map((i) => nodes[i]).filter(Boolean);
    if (!pts.length) continue;

    let cx = 0, cy = 0;
    for (const p of pts) { cx += p.x; cy += p.y; }
    cx /= pts.length; cy /= pts.length;

    let r = 0;
    for (const p of pts) r = Math.max(r, Math.hypot(p.x - cx, p.y - cy));
    r = Math.max(2.5, r);

    const hull = convexHull(pts);
    const verticesLocal = hull
      .map((p) => ({ x: p.x - cx, y: p.y - cy }))
      .filter((v) => Number.isFinite(v.x) && Number.isFinite(v.y));
    const sides = Math.max(3, verticesLocal.length || Math.min(12, approximateHullSides(pts)));
    const mass = controls.massHeavy;
    const body = {
      x: cx,
      y: cy,
      vx: 0,
      vy: 0,
      r,
      sides,
      verticesLocal: verticesLocal.length >= 3 ? verticesLocal : null,
      edgeDyeMode: Array.from({ length: sides }, () => [1, 1, 1]),
      edgeBodyMode: Array.from({ length: sides }, () => EDGE_BODY_BLOCK),
      digestEnabled: false,
      digestRGB: [1, 1, 1],
      mass,
      theta: 0,
      omega: 0,
      inertia: 0.5 * mass * r * r,
    };

    rigid.push(body);
    components.push({
      index: rigid.length - 1,
      nodeIds: pointIds,
      cx,
      cy,
      r,
      sides,
    });
  }
  return { rigid, components };
}

function buildSoftFromTriangles(tris, nodes, controls, opts = {}) {
  const comps = triangleComponents(tris);
  const softNodes = [];
  const springs = [];
  const components = [];
  const globalRemap = new Map();

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
      const outIdx = base + i;
      remap.set(id, outIdx);
      globalRemap.set(id, outIdx);
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
      const aIdx = remap.get(ua);
      const bIdx = remap.get(ub);
      const a = softNodes[aIdx];
      const b = softNodes[bIdx];
      const rest = Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y));
      const boundary = c === 1;
      springs.push([
        aIdx,
        bIdx,
        rest,
        boundary ? EDGE_BODY_BLOCK : 0,
        boundary ? [1, 1, 1] : [0, 0, 0],
      ]);
    }

    components.push({ nodeIds: pointIds, clusterId });
    clusterId += 1;
  }

  return { nodes: softNodes, springs, components, globalRemap };
}

function buildRigidSoftHybridLinks({ rigidComps, softComps, rigidBodies, softNodeRemap, nodes }) {
  const links = [];
  const used = new Set();

  for (const rc of rigidComps) {
    for (const sc of softComps) {
      const shared = [];
      for (const nid of rc.nodeIds) if (sc.nodeIds.has(nid)) shared.push(nid);
      for (const nid of shared) {
        const softIdx = softNodeRemap.get(nid);
        if (softIdx == null) continue;
        const key = `${rc.index}:${softIdx}`;
        if (used.has(key)) continue;
        used.add(key);

        const p = nodes[nid];
        const rb = rigidBodies[rc.index];
        if (!p || !rb) continue;

        const ang = Math.atan2(p.y - rb.y, p.x - rb.x);
        const twoPi = Math.PI * 2;
        const phase = ((ang - (rb.theta || 0)) % twoPi + twoPi) % twoPi;
        const f = (phase / twoPi) * rb.sides;
        const vA = Math.floor(f) % rb.sides;
        const vB = (vA + 1) % rb.sides;

        const aPos = rigidVertexWorld(rb, vA);
        const bPos = rigidVertexWorld(rb, vB);
        const restA = Math.max(0.8, Math.hypot(p.x - aPos.x, p.y - aPos.y));
        const restB = Math.max(0.8, Math.hypot(p.x - bPos.x, p.y - bPos.y));

        links.push({
          rigidIndex: rc.index,
          nodeIndex: softIdx,
          vertexA: vA,
          vertexB: vB,
          restA,
          restB,
        });
      }
    }
  }

  return links;
}

function rigidVertexWorld(rb, vi) {
  const ang = (vi / rb.sides) * Math.PI * 2 + (rb.theta || 0);
  return { x: rb.x + Math.cos(ang) * rb.r, y: rb.y + Math.sin(ang) * rb.r };
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
  lower.pop(); upper.pop();
  return [...lower, ...upper];
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
