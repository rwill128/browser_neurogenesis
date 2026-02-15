export const CREATURE_SPEC_VERSION = 'creature-spec.v2';

const EDGE_BODY_BLOCK = 1;
const EDGE_DYE_PASS = 0;
const EDGE_DYE_DEFLECT = 1;
const EDGE_DYE_ABSORB = 2;
const EDGE_DYE_DEFLECT_RGB = [EDGE_DYE_DEFLECT, EDGE_DYE_DEFLECT, EDGE_DYE_DEFLECT];

export function createCreatureSpecFromMesh(mesh, options = {}) {
  const width = mesh?.meta?.width || options.width || 128;
  const height = mesh?.meta?.height || options.height || 128;
  const nodes = (mesh?.nodes || []).map((p, i) => ({
    id: i,
    x: Number(p.x) || 0,
    y: Number(p.y) || 0,
    rigid: Number(p.rigid) || 0,
    soft: Number(p.soft) || 0,
  }));

  const triByKind = { rigid: [], soft: [] };
  for (const t of (mesh?.triangles || [])) {
    if (t?.kind === 'rigid') triByKind.rigid.push(t);
    else if (t?.kind === 'soft') triByKind.soft.push(t);
  }

  const rigidBuild = Array.isArray(mesh?.rigidPieces)
    ? buildRigidExportFromCompilerPieces(mesh.rigidPieces, nodes, options)
    : buildRigidExport(triByKind.rigid, nodes, options);
  const softBuild = buildSoftExport(triByKind.soft, nodes, options, mesh?.softCrossBeams || []);
  const hybridJoints = buildHybridExport({
    rigidComps: rigidBuild.components,
    softComps: softBuild.components,
    rigidBodies: rigidBuild.rigidBodies,
    sourceNodes: nodes,
  });

  const out = {
    schemaVersion: CREATURE_SPEC_VERSION,
    createdAt: new Date().toISOString(),
    name: options.name || 'unnamed-creature',
    space: { width, height },
    rigidBodies: rigidBuild.rigidBodies,
    softBodies: softBuild.softBodies,
    hybridJoints,
  };

  // Authoring payload is optional and intentionally separate from solver contract.
  if (options.includeAuthoring !== false) {
    const rigidField = options?.fields?.rigidField;
    const softField = options?.fields?.softField;
    const softDensityField = options?.fields?.softDensityField;
    if (rigidField && softField) {
      out.authoring = {
        fields: {
          width,
          height,
          rigid: Array.from(rigidField),
          soft: Array.from(softField),
          softDensity: softDensityField ? Array.from(softDensityField) : undefined,
        },
      };
    }
  }

  return out;
}

export function parseCreatureSpec(jsonText) {
  const obj = JSON.parse(jsonText);
  if (!obj || typeof obj !== 'object') throw new Error('Invalid JSON object');
  if (obj.schemaVersion !== CREATURE_SPEC_VERSION) {
    throw new Error(`Unsupported schemaVersion: ${obj.schemaVersion}`);
  }
  if (!obj.space || !Number.isFinite(obj.space.width) || !Number.isFinite(obj.space.height)) {
    throw new Error('CreatureSpec missing valid space.width/space.height');
  }
  if (!Array.isArray(obj.rigidBodies) || !Array.isArray(obj.softBodies) || !Array.isArray(obj.hybridJoints)) {
    throw new Error('CreatureSpec missing rigidBodies/softBodies/hybridJoints arrays');
  }
  for (const rb of obj.rigidBodies) {
    if (!Array.isArray(rb.hull) || rb.hull.length < 3) throw new Error('Rigid body missing hull vertices');
  }
  for (const sb of obj.softBodies) {
    if (!Array.isArray(sb.nodes) || !Array.isArray(sb.springs)) throw new Error('Soft body missing nodes/springs');
  }
  return obj;
}

export function buildBodiesFromCreatureSpec(spec, n, controls) {
  const srcW = Math.max(1, Number(spec.space?.width) || n);
  const srcH = Math.max(1, Number(spec.space?.height) || n);
  const sx = n / srcW;
  const sy = n / srcH;
  const sRest = 0.5 * (sx + sy);

  const rigid = [];
  for (let rbi = 0; rbi < (spec.rigidBodies || []).length; rbi++) {
    const rb = spec.rigidBodies[rbi];
    const hull = (rb.hull || []).map((p) => ({ x: (Number(p.x) || 0) * sx, y: (Number(p.y) || 0) * sy }));
    if (hull.length < 3) continue;

    const c = centroid(hull);
    const verticesLocal = hull.map((p) => ({ x: p.x - c.x, y: p.y - c.y }));
    const r = Math.max(2.5, ...verticesLocal.map((v) => Math.hypot(v.x, v.y)));
    const sides = Math.max(3, verticesLocal.length);
    const mass = finiteOr(Number(rb.mass), controls.massHeavy);

    const subPolysLocal = Array.isArray(rb.subHulls)
      ? rb.subHulls
          .map((poly) => (Array.isArray(poly) ? poly : []))
          .map((poly) => poly
            .map((p) => ({ x: (Number(p?.x) || 0) * sx - c.x, y: (Number(p?.y) || 0) * sy - c.y }))
            .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)))
          .filter((poly) => poly.length >= 3)
      : null;

    rigid.push({
      x: c.x,
      y: c.y,
      vx: 0,
      vy: 0,
      r,
      sides,
      verticesLocal,
      subPolysLocal,
      edgeDyeMode: normalizeEdgeDyeModeList(rb.edgeDyeMode, sides),
      edgeBodyMode: normalizeEdgeBodyModeList(rb.edgeBodyMode, sides),
      edgePermeabilityRGB: normalizeEdgePermeabilityList(rb.edgePermeabilityRGB, sides),
      digestEnabled: !!rb.digestEnabled,
      digestRGB: normalizeRGB(rb.digestRGB),
      consumeDyeRGB: normalizeBinaryRGB(rb.consumeDyeRGB ?? (rb.digestEnabled ? [1, 1, 1] : [0, 0, 0])),
      mass,
      theta: 0,
      omega: 0,
      inertia: finiteOr(Number(rb.inertia), 0.5 * mass * r * r),
    });
  }

  const soft = { nodes: [], springs: [] };
  const softNodeMap = new Map();
  let clusterId = 0;
  for (let sbi = 0; sbi < (spec.softBodies || []).length; sbi++) {
    const sb = spec.softBodies[sbi];
    const base = soft.nodes.length;
    for (let i = 0; i < sb.nodes.length; i++) {
      const p = sb.nodes[i] || {};
      soft.nodes.push({
        x: (Number(p.x) || 0) * sx,
        y: (Number(p.y) || 0) * sy,
        vx: 0,
        vy: 0,
        mass: finiteOr(Number(p.mass), controls.massSoft),
        r: finiteOr(Number(p.r), 1.6),
        clusterId,
        digestEnabled: !!p.digestEnabled,
        digestRGB: normalizeRGB(p.digestRGB),
      });
      softNodeMap.set(`${sbi}:${i}`, base + i);
    }

    for (const sp of sb.springs) {
      const [aRaw, bRaw, restRaw, edgeBodyRaw, edgeDyeRaw] = Array.isArray(sp)
        ? sp
        : [sp?.a, sp?.b, sp?.rest, sp?.edgeBodyMode, sp?.edgeDyeMode];
      const a = Number(aRaw);
      const b = Number(bRaw);
      if (!Number.isInteger(a) || !Number.isInteger(b)) continue;
      if (a < 0 || b < 0 || a >= sb.nodes.length || b >= sb.nodes.length) continue;
      soft.springs.push([
        base + a,
        base + b,
        Math.max(1e-4, finiteOr(Number(restRaw), 1) * sRest),
        normalizeEdgeBodyMode(edgeBodyRaw),
        normalizeEdgeDyeMode(edgeDyeRaw),
      ]);
    }

    clusterId += 1;
  }

  const hybrid = [];
  for (const j of (spec.hybridJoints || [])) {
    const rigidIndex = Number(j.rigidBodyIndex);
    const softBodyIndex = Number(j.softBodyIndex);
    const softNodeIndex = Number(j.softNodeIndex);
    if (!Number.isInteger(rigidIndex) || rigidIndex < 0 || rigidIndex >= rigid.length) continue;

    const globalSoft = softNodeMap.get(`${softBodyIndex}:${softNodeIndex}`);
    if (!Number.isInteger(globalSoft) || globalSoft < 0 || globalSoft >= soft.nodes.length) continue;

    const rb = rigid[rigidIndex];
    const edgeA = Math.max(0, Math.min(rb.sides - 1, Number(j.edgeA) | 0));
    let edgeB = Math.max(0, Math.min(rb.sides - 1, Number(j.edgeB) | 0));
    // Guardrail: avoid degenerate hybrid constraints that pin both rest links
    // to the same rigid vertex (can inject solver jitter under fluid load).
    if (rb.sides > 1 && edgeA === edgeB) edgeB = (edgeA + 1) % rb.sides;

    const aPos = rigidVertexWorld(rb, edgeA);
    const bPos = rigidVertexWorld(rb, edgeB);
    const p = soft.nodes[globalSoft];

    const defaultRestA = Math.hypot(p.x - aPos.x, p.y - aPos.y);
    const defaultRestB = Math.hypot(p.x - bPos.x, p.y - bPos.y);
    const maxRest = Math.max(6, rb.r * 1.5);
    const restA = Math.min(maxRest, Math.max(0.8, finiteOr(Number(j.restA), defaultRestA)));
    const restB = Math.min(maxRest, Math.max(0.8, finiteOr(Number(j.restB), defaultRestB)));

    hybrid.push({
      rigidIndex,
      nodeIndex: globalSoft,
      vertexA: edgeA,
      vertexB: edgeB,
      restA,
      restB,
    });
  }

  return { rigid, soft, hybrid };
}

function buildRigidExportFromCompilerPieces(rigidPieces, nodes, options) {
  const grouped = new Map();
  for (let i = 0; i < rigidPieces.length; i++) {
    const rp = rigidPieces[i] || {};
    const hull = (rp.hull || [])
      .map((p) => ({ x: Number(p?.x) || 0, y: Number(p?.y) || 0 }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (hull.length < 3) continue;
    const compoundId = rp.compoundId || `compound_piece_${i}`;
    if (!grouped.has(compoundId)) grouped.set(compoundId, []);
    grouped.get(compoundId).push({
      hull,
      sourceNodeIds: Array.isArray(rp.sourceNodeIds)
        ? rp.sourceNodeIds.map((id) => Number(id)).filter((id) => Number.isInteger(id) && nodes[id])
        : [],
    });
  }

  const rigidBodies = [];
  const components = [];

  for (const [compoundId, pieces] of grouped.entries()) {
    const allPoints = [];
    const sourceNodeSet = new Set();
    const subHulls = [];

    for (const p of pieces) {
      subHulls.push(p.hull.map((v) => ({ x: v.x, y: v.y })));
      for (const v of p.hull) allPoints.push(v);
      for (const sid of p.sourceNodeIds) sourceNodeSet.add(sid);
    }

    // Preserve compiler-authored concavity when a compound already comes as a single contour piece.
    // Only fall back to convex merge when multiple disjoint sub-hulls must be combined.
    const outerHull = pieces.length === 1
      ? pieces[0].hull.map((v) => ({ x: v.x, y: v.y }))
      : convexHull(allPoints);
    if (outerHull.length < 3) continue;

    const c = centroid(outerHull);
    const r = Math.max(2.5, ...outerHull.map((p) => Math.hypot(p.x - c.x, p.y - c.y)));
    const sides = outerHull.length;

    const rigidIndex = rigidBodies.length;
    rigidBodies.push({
      id: `rigid_${rigidIndex}`,
      compoundId,
      hull: outerHull,
      subHulls,
      mass: finiteOr(Number(options.massHeavy), 5),
      edgeBodyMode: Array.from({ length: sides }, () => EDGE_BODY_BLOCK),
      edgeDyeMode: Array.from({ length: sides }, () => [...EDGE_DYE_DEFLECT_RGB]),
      edgePermeabilityRGB: Array.from({ length: sides }, () => [0, 0, 0]),
      digestEnabled: false,
      digestRGB: [1, 1, 1],
      consumeDyeRGB: [0, 0, 0],
    });

    components.push({
      index: rigidIndex,
      nodeIds: sourceNodeSet,
      radius: r,
    });
  }

  // Compound fusion path: no runtime inter-piece spring/weld constraints.
  return { rigidBodies, components };
}

function buildRigidExport(tris, nodes, options) {
  const comps = triangleComponents(tris);
  const rigidBodies = [];
  const components = [];

  for (let compIdx = 0; compIdx < comps.length; compIdx++) {
    const comp = comps[compIdx];

    const triPieces = [];
    for (const t of comp) {
      const idsRaw = [Number(t.a), Number(t.b), Number(t.c)];
      const ids = [...new Set(idsRaw)].filter((id) => Number.isInteger(id) && nodes[id]);
      if (ids.length < 3) continue;
      triPieces.push({
        nodeIds: new Set(ids),
        area: triangleAreaAbs(nodes[ids[0]], nodes[ids[1]], nodes[ids[2]]),
      });
    }

    const mergedPieces = mergeRigidPiecesConvex(triPieces, nodes);

    for (const piece of mergedPieces) {
      const hullWithIds = convexHullWithIds([...piece.nodeIds].map((id) => ({ id, x: nodes[id].x, y: nodes[id].y })));
      if (hullWithIds.length < 3) continue;
      const hull = hullWithIds.map((p) => ({ x: p.x, y: p.y }));
      const hullSourceIds = hullWithIds.map((p) => p.id);

      const c = centroid(hull);
      const r = Math.max(2.5, ...hull.map((p) => Math.hypot(p.x - c.x, p.y - c.y)));
      const sides = hull.length;

      const rigidIndex = rigidBodies.length;
      rigidBodies.push({
        id: `rigid_${rigidIndex}`,
        compoundId: `compound_${compIdx}`,
        hull,
        mass: finiteOr(Number(options.massHeavy), 5),
        edgeBodyMode: Array.from({ length: sides }, () => EDGE_BODY_BLOCK),
        edgeDyeMode: Array.from({ length: sides }, () => [...EDGE_DYE_DEFLECT_RGB]),
        edgePermeabilityRGB: Array.from({ length: sides }, () => [0, 0, 0]),
        digestEnabled: false,
        digestRGB: [1, 1, 1],
        consumeDyeRGB: [0, 0, 0],
      });

      components.push({
        index: rigidIndex,
        nodeIds: new Set(piece.nodeIds),
        radius: r,
      });

    }

  }

  return { rigidBodies, components };
}

function mergeRigidPiecesConvex(triPieces, nodes) {
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
        const mergedHull = convexHull([...mergedIds].map((id) => ({ x: nodes[id].x, y: nodes[id].y })));
        const hullArea = polygonAreaAbs(mergedHull);
        const sumArea = a.area + b.area;
        const eps = Math.max(1e-4, sumArea * 0.02);
        if (Math.abs(hullArea - sumArea) <= eps) {
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

function buildSoftExport(tris, nodes, options, softCrossBeams = []) {
  const comps = triangleComponents(tris);
  const softBodies = [];
  const components = [];

  for (const comp of comps) {
    const pointIds = new Set();
    for (const t of comp) {
      pointIds.add(t.a); pointIds.add(t.b); pointIds.add(t.c);
    }
    const ids = [...pointIds];
    const remap = new Map();

    const softNodes = ids.map((id, i) => {
      remap.set(id, i);
      const p = nodes[id];
      return {
        x: p.x,
        y: p.y,
        mass: finiteOr(Number(options.massSoft), 0.6),
        r: 1.6,
        digestEnabled: false,
        digestRGB: [1, 1, 1],
      };
    });

    const edgeCount = new Map();
    for (const t of comp) {
      for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
        const k = u < v ? `${u}-${v}` : `${v}-${u}`;
        edgeCount.set(k, (edgeCount.get(k) || 0) + 1);
      }
    }

    const springs = [];
    const springSet = new Set();
    const springKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

    for (const [k, c] of edgeCount.entries()) {
      const [ua, ub] = k.split('-').map((x) => Number(x));
      const a = remap.get(ua);
      const b = remap.get(ub);
      if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
      const pa = softNodes[a];
      const pb = softNodes[b];
      if (!pa || !pb) continue;
      const rest = Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y));
      const boundary = c === 1;
      springs.push([
        a,
        b,
        rest,
        boundary ? EDGE_BODY_BLOCK : 0,
        boundary ? [...EDGE_DYE_DEFLECT_RGB] : [0, 0, 0],
      ]);
      springSet.add(springKey(a, b));
    }

    // Double cross-beam square reinforcement for soft lattice.
    for (const beam of (softCrossBeams || [])) {
      const ua = Number(beam?.[0]);
      const ub = Number(beam?.[1]);
      const a = remap.get(ua);
      const b = remap.get(ub);
      if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
      const key = springKey(a, b);
      if (springSet.has(key)) continue;
      const pa = softNodes[a];
      const pb = softNodes[b];
      if (!pa || !pb) continue;
      springs.push([
        a,
        b,
        Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y)),
        0,
        [0, 0, 0],
      ]);
      springSet.add(key);
    }

    const bodyIndex = softBodies.length;
    softBodies.push({
      id: `soft_${bodyIndex}`,
      nodes: softNodes,
      springs,
    });

    components.push({
      index: bodyIndex,
      nodeIds: pointIds,
      sourceToLocal: remap,
    });
  }

  return { softBodies, components };
}

function buildHybridExport({ rigidComps, softComps, rigidBodies, sourceNodes }) {
  const links = [];
  const used = new Set();

  for (const rc of rigidComps) {
    for (const sc of softComps) {
      const shared = [];
      for (const nid of rc.nodeIds) if (sc.nodeIds.has(nid)) shared.push(nid);
      for (const nid of shared) {
        const softNodeLocal = sc.sourceToLocal.get(nid);
        if (softNodeLocal == null) continue;

        const key = `${rc.index}:${sc.index}:${softNodeLocal}`;
        if (used.has(key)) continue;
        used.add(key);

        const p = sourceNodes[nid];
        const rb = rigidBodies[rc.index];
        if (!p || !rb || !rb.hull?.length) continue;

        const { vA, vB } = nearestHullEdgeForPoint(rb.hull, p.x, p.y);
        const aPos = rb.hull[vA];
        const bPos = rb.hull[vB];
        const maxRest = Math.max(6, rc.radius * 0.55);

        links.push({
          rigidBodyIndex: rc.index,
          softBodyIndex: sc.index,
          softNodeIndex: softNodeLocal,
          edgeA: vA,
          edgeB: vB,
          restA: Math.min(maxRest, Math.max(0.8, Math.hypot(p.x - aPos.x, p.y - aPos.y))),
          restB: Math.min(maxRest, Math.max(0.8, Math.hypot(p.x - bPos.x, p.y - bPos.y))),
        });
      }
    }
  }

  return links;
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

function rigidVerticesWorld(rb) {
  if (Array.isArray(rb.verticesLocal) && rb.verticesLocal.length >= 3) {
    const th = rb.theta || 0;
    const c = Math.cos(th), s = Math.sin(th);
    return rb.verticesLocal.map((v) => ({
      x: rb.x + v.x * c - v.y * s,
      y: rb.y + v.x * s + v.y * c,
    }));
  }
  const sides = Math.max(3, rb.sides || 3);
  const rot = (rb.theta || 0) + (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
  const verts = [];
  for (let i = 0; i < sides; i++) {
    const a = rot + (i / sides) * Math.PI * 2;
    verts.push({ x: rb.x + Math.cos(a) * rb.r, y: rb.y + Math.sin(a) * rb.r });
  }
  return verts;
}

function rigidVertexWorld(rb, vi) {
  const verts = rigidVerticesWorld(rb);
  const n = verts.length || 1;
  return verts[((vi % n) + n) % n];
}

function nearestHullEdgeForPoint(hull, px, py) {
  if (!hull?.length) return { vA: 0, vB: 0 };
  if (hull.length === 1) return { vA: 0, vB: 0 };

  let nearestVertex = 0;
  let nearestVertexD2 = Number.POSITIVE_INFINITY;
  for (let i = 0; i < hull.length; i++) {
    const dx = px - hull[i].x;
    const dy = py - hull[i].y;
    const d2 = dx * dx + dy * dy;
    if (d2 < nearestVertexD2) {
      nearestVertexD2 = d2;
      nearestVertex = i;
    }
  }

  // Deterministic tie-break for shared nodes that land directly on a rigid vertex.
  if (nearestVertexD2 <= 1e-6) {
    const n = hull.length;
    const prev = (nearestVertex - 1 + n) % n;
    const next = (nearestVertex + 1) % n;
    const prevLen2 = (hull[nearestVertex].x - hull[prev].x) ** 2 + (hull[nearestVertex].y - hull[prev].y) ** 2;
    const nextLen2 = (hull[nearestVertex].x - hull[next].x) ** 2 + (hull[nearestVertex].y - hull[next].y) ** 2;
    return nextLen2 <= prevLen2
      ? { vA: nearestVertex, vB: next }
      : { vA: prev, vB: nearestVertex };
  }

  let bestIdx = 0;
  let bestD2 = Number.POSITIVE_INFINITY;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const apx = px - a.x;
    const apy = py - a.y;
    const denom = Math.max(1e-6, abx * abx + aby * aby);
    const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / denom));
    const cx = a.x + abx * t;
    const cy = a.y + aby * t;
    const dx = px - cx;
    const dy = py - cy;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      bestIdx = i;
    }
  }

  return { vA: bestIdx, vB: (bestIdx + 1) % hull.length };
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

function normalizeEdgeBodyModeList(list, count) {
  if (!Array.isArray(list) || !list.length) return Array.from({ length: count }, () => EDGE_BODY_BLOCK);
  return Array.from({ length: count }, (_, i) => {
    const v = list[i];
    return v === undefined ? EDGE_BODY_BLOCK : normalizeEdgeBodyMode(v);
  });
}

function normalizeEdgeDyeModeList(list, count) {
  if (!Array.isArray(list) || !list.length) return Array.from({ length: count }, () => [...EDGE_DYE_DEFLECT_RGB]);
  return Array.from({ length: count }, (_, i) => {
    const v = list[i];
    return v === undefined ? [...EDGE_DYE_DEFLECT_RGB] : normalizeEdgeDyeMode(v);
  });
}

function normalizeEdgePermeabilityList(list, count) {
  if (!Array.isArray(list) || !list.length) return Array.from({ length: count }, () => [0, 0, 0]);
  return Array.from({ length: count }, (_, i) => {
    const v = list[i];
    return v === undefined ? [0, 0, 0] : normalizeBinaryRGB(v);
  });
}

function normalizeEdgeBodyMode(v) {
  return Number(v) === EDGE_BODY_BLOCK ? EDGE_BODY_BLOCK : 0;
}

function normalizeEdgeDyeMode(v) {
  if (Array.isArray(v) && v.length >= 3) {
    return [
      normalizeEdgeDyeModeChannel(v[0]),
      normalizeEdgeDyeModeChannel(v[1]),
      normalizeEdgeDyeModeChannel(v[2]),
    ];
  }
  const n = Number(v);
  if (Number.isFinite(n)) {
    const c = normalizeEdgeDyeModeChannel(n);
    return [c, c, c];
  }
  return [...EDGE_DYE_DEFLECT_RGB];
}

function normalizeEdgeDyeModeChannel(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return EDGE_DYE_DEFLECT;
  if (n === EDGE_DYE_PASS || n === EDGE_DYE_DEFLECT || n === EDGE_DYE_ABSORB) return n;
  return EDGE_DYE_DEFLECT;
}

function normalizeBinaryRGB(v) {
  if (!Array.isArray(v) || v.length < 3) return [0, 0, 0];
  return [
    Number(v[0]) > 0 ? 1 : 0,
    Number(v[1]) > 0 ? 1 : 0,
    Number(v[2]) > 0 ? 1 : 0,
  ];
}

function normalizeRGB(v) {
  if (!Array.isArray(v) || v.length < 3) return [1, 1, 1];
  return [finiteOr(Number(v[0]), 1), finiteOr(Number(v[1]), 1), finiteOr(Number(v[2]), 1)];
}

function finiteOr(v, fallback) {
  return Number.isFinite(v) ? v : fallback;
}
