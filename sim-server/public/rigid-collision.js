const EPS = 1e-6;
const RIGID_CONTACT_SLOP = 0.015;

function finiteOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function rigidVerticesWorld(body) {
  if (!body) return [];
  const theta = finiteOr(body.theta, 0);
  const bx = finiteOr(body.x, 0);
  const by = finiteOr(body.y, 0);

  if (Array.isArray(body.verticesLocal) && body.verticesLocal.length >= 3) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return body.verticesLocal.map((v) => ({
      x: bx + v.x * c - v.y * s,
      y: by + v.x * s + v.y * c,
    }));
  }

  const sides = Math.max(3, Math.trunc(finiteOr(body.sides, 3)) || 3);
  const rot = theta + (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
  const radius = Math.max(0, finiteOr(body.r, 0));
  const verts = [];
  for (let i = 0; i < sides; i++) {
    const a = rot + (i / sides) * Math.PI * 2;
    verts.push({ x: bx + Math.cos(a) * radius, y: by + Math.sin(a) * radius });
  }
  return verts;
}

export function pointOnSegment(px, py, ax, ay, bx, by, eps = EPS) {
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

export function pointInPolygonInclusive(px, py, verts, eps = EPS) {
  let inside = false;
  for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
    const xi = verts[i].x;
    const yi = verts[i].y;
    const xj = verts[j].x;
    const yj = verts[j].y;

    if (pointOnSegment(px, py, xi, yi, xj, yj, eps)) return true;

    const dy = yj - yi;
    if (Math.abs(dy) <= eps) continue;
    const intersect = ((yi > py) !== (yj > py))
      && (px < ((xj - xi) * (py - yi)) / dy + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function polygonCentroid(verts) {
  let sx = 0;
  let sy = 0;
  for (const v of verts) {
    sx += v.x;
    sy += v.y;
  }
  const k = 1 / Math.max(1, verts.length);
  return { x: sx * k, y: sy * k };
}

function sanitizeFinitePolygonVerts(verts, eps = EPS) {
  if (!Array.isArray(verts) || verts.length < 3) return [];
  const out = [];
  for (const v of verts) {
    const x = Number(v?.x);
    const y = Number(v?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const prev = out[out.length - 1];
    if (prev && Math.hypot(x - prev.x, y - prev.y) <= eps) continue;
    out.push({ x, y });
  }
  if (out.length >= 2) {
    const first = out[0];
    const last = out[out.length - 1];
    if (Math.hypot(first.x - last.x, first.y - last.y) <= eps) out.pop();
  }
  return out.length >= 3 ? out : [];
}

function closestPointOnSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const ab2 = abx * abx + aby * aby;
  if (ab2 < EPS) return { x: ax, y: ay, t: 0, abx, aby };
  const apx = px - ax;
  const apy = py - ay;
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / ab2));
  return { x: ax + abx * t, y: ay + aby * t, t, abx, aby };
}

function edgeOutwardNormal(ax, ay, bx, by, cx, cy) {
  const ex = bx - ax;
  const ey = by - ay;
  const len = Math.max(EPS, Math.hypot(ex, ey));
  let nx = -ey / len;
  let ny = ex / len;

  // Ensure normal points away from polygon centroid.
  const mx = (ax + bx) * 0.5;
  const my = (ay + by) * 0.5;
  const toCenterX = cx - mx;
  const toCenterY = cy - my;
  if (toCenterX * nx + toCenterY * ny > 0) {
    nx = -nx;
    ny = -ny;
  }
  return { nx, ny };
}

function isRigidSoftNodeCollisionCache(value) {
  return Boolean(
    value
    && !Array.isArray(value)
    && Array.isArray(value.verts)
    && Array.isArray(value.edges),
  );
}

export function buildRigidSoftNodeCollisionCache(rigid, vertsInput = null) {
  const verts = Array.isArray(vertsInput) && vertsInput.length >= 3
    ? vertsInput
    : rigidVerticesWorld(rigid);
  if (!Array.isArray(verts) || verts.length < 3) return null;

  const finiteVerts = sanitizeFinitePolygonVerts(verts);
  if (finiteVerts.length < 3) return null;

  const centroid = polygonCentroid(finiteVerts);
  const edges = [];
  for (let i = 0; i < finiteVerts.length; i++) {
    const a = finiteVerts[i];
    const b = finiteVerts[(i + 1) % finiteVerts.length];
    const out = edgeOutwardNormal(a.x, a.y, b.x, b.y, centroid.x, centroid.y);
    edges.push({
      ax: a.x,
      ay: a.y,
      bx: b.x,
      by: b.y,
      nx: out.nx,
      ny: out.ny,
    });
  }

  return {
    verts: finiteVerts,
    edges,
    centroid,
  };
}

// True polygonal rigid-vs-soft-node collision (concave-friendly).
export function resolveRigidVsSoftNodeCollision(rigid, node, vertsInput, restitution = 0.28) {
  const cache = isRigidSoftNodeCollisionCache(vertsInput)
    ? vertsInput
    : buildRigidSoftNodeCollisionCache(rigid, vertsInput);
  if (!cache) return false;

  const finiteVerts = cache.verts;
  const edges = cache.edges;

  let best = null;
  for (const edge of edges) {
    const cp = closestPointOnSegment(node.x, node.y, edge.ax, edge.ay, edge.bx, edge.by);
    if (!Number.isFinite(cp.x) || !Number.isFinite(cp.y)) continue;
    const dx = node.x - cp.x;
    const dy = node.y - cp.y;
    const d2 = dx * dx + dy * dy;
    if (!best || d2 < best.d2) {
      best = {
        d2,
        d: Math.sqrt(d2),
        cp,
        ax: edge.ax,
        ay: edge.ay,
        bx: edge.bx,
        by: edge.by,
        nx: edge.nx,
        ny: edge.ny,
      };
    }
  }
  if (!best) return false;

  const r = Math.max(0.4, node.r || 1.0);
  const inside = pointInPolygonInclusive(node.x, node.y, finiteVerts);
  const contactSlop = Math.max(0.015, r * 0.04);
  if (!inside && best.d >= (r - contactSlop)) return false;

  let nx = best.nx;
  let ny = best.ny;

  if (!inside) {
    // Outside contact should push node away from edge toward current node->edge side.
    const toNodeX = node.x - best.cp.x;
    const toNodeY = node.y - best.cp.y;
    if (toNodeX * nx + toNodeY * ny < 0) {
      nx = -nx;
      ny = -ny;
    }
  }

  const dist = Math.max(EPS, best.d);
  const penetration = inside ? (r + dist) : (r - dist);
  if (penetration <= contactSlop) return false;

  const mNode = Math.max(0.02, node.mass || 1);
  const mRigid = Math.max(0.05, rigid.mass || 1);
  const invNode = 1 / mNode;
  const invRigid = 1 / mRigid;
  const invSum = invNode + invRigid;

  const corrFactor = inside ? 0.82 : 0.68;
  const corr = (penetration / Math.max(EPS, invSum)) * corrFactor;
  node.x += nx * corr * invNode;
  node.y += ny * corr * invNode;
  rigid.x -= nx * corr * invRigid;
  rigid.y -= ny * corr * invRigid;

  const cpX = best.cp.x;
  const cpY = best.cp.y;
  const rx = cpX - rigid.x;
  const ry = cpY - rigid.y;
  const omega = rigid.omega || 0;
  const rigidPointVx = (rigid.vx || 0) - omega * ry;
  const rigidPointVy = (rigid.vy || 0) + omega * rx;
  const rvx = (node.vx || 0) - rigidPointVx;
  const rvy = (node.vy || 0) - rigidPointVy;
  const vn = rvx * nx + rvy * ny;
  if (vn >= -0.02) return true;

  const invInertia = 1 / Math.max(0.05, rigid.inertia || (0.5 * mRigid * Math.max(1, rigid.r || 1) ** 2));
  const rn = rx * ny - ry * nx;
  const denom = invNode + invRigid + (rn * rn) * invInertia;
  const rawJ = (-(1 + restitution) * vn) / Math.max(EPS, denom);

  // Guardrail: clamp very deep/high-speed contacts so one bad frame cannot inject
  // unbounded momentum into rigid angular velocity.
  const maxImpactSpeed = 24;
  const maxJ = ((1 + restitution) * Math.min(maxImpactSpeed, Math.abs(vn))) / Math.max(EPS, denom);
  const j = Math.max(-maxJ, Math.min(maxJ, rawJ));
  const jx = j * nx;
  const jy = j * ny;

  node.vx = (node.vx || 0) + jx * invNode;
  node.vy = (node.vy || 0) + jy * invNode;
  rigid.vx = (rigid.vx || 0) - jx * invRigid;
  rigid.vy = (rigid.vy || 0) - jy * invRigid;
  rigid.omega = (rigid.omega || 0) - (rx * jy - ry * jx) * invInertia;
  return true;
}

function rigidVerticesLocalFallback(body) {
  const sides = Math.max(3, body?.sides || 3);
  const rot = (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
  const r = Math.max(1, body?.r || 1);
  const verts = [];
  for (let i = 0; i < sides; i++) {
    const a = rot + (i / sides) * Math.PI * 2;
    verts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
  }
  return verts;
}

function signedArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    s += p.x * q.y - q.x * p.y;
  }
  return s * 0.5;
}

function ensureCCW(poly) {
  if (!Array.isArray(poly)) return [];
  if (poly.length < 3) return poly.map((p) => ({ x: p.x, y: p.y }));
  const out = poly.map((p) => ({ x: p.x, y: p.y }));
  if (signedArea(out) < 0) out.reverse();
  return out;
}

function isConvexPolygon(poly) {
  if (!Array.isArray(poly) || poly.length < 4) return true;
  let sign = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const c = poly[(i + 2) % poly.length];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(cross) <= 1e-7) continue;
    const s = cross > 0 ? 1 : -1;
    if (sign === 0) sign = s;
    else if (sign !== s) return false;
  }
  return true;
}

function pointInTriangle(px, py, a, b, c) {
  const v0x = c.x - a.x;
  const v0y = c.y - a.y;
  const v1x = b.x - a.x;
  const v1y = b.y - a.y;
  const v2x = px - a.x;
  const v2y = py - a.y;

  const dot00 = v0x * v0x + v0y * v0y;
  const dot01 = v0x * v1x + v0y * v1y;
  const dot02 = v0x * v2x + v0y * v2y;
  const dot11 = v1x * v1x + v1y * v1y;
  const dot12 = v1x * v2x + v1y * v2y;

  const invDen = 1 / Math.max(EPS, dot00 * dot11 - dot01 * dot01);
  const u = (dot11 * dot02 - dot01 * dot12) * invDen;
  const v = (dot00 * dot12 - dot01 * dot02) * invDen;
  return u >= -1e-6 && v >= -1e-6 && (u + v) <= 1 + 1e-6;
}

function triangulateEarClip(polyInput) {
  const poly = ensureCCW(polyInput);
  if (poly.length < 3) return [];
  if (poly.length === 3) return [poly];

  const indices = Array.from({ length: poly.length }, (_, i) => i);
  const tris = [];
  let guard = poly.length * poly.length;

  while (indices.length > 3 && guard-- > 0) {
    let clipped = false;
    for (let ii = 0; ii < indices.length; ii++) {
      const i0 = indices[(ii - 1 + indices.length) % indices.length];
      const i1 = indices[ii];
      const i2 = indices[(ii + 1) % indices.length];
      const a = poly[i0];
      const b = poly[i1];
      const c = poly[i2];

      const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
      if (cross <= 1e-7) continue;

      let contains = false;
      for (let jj = 0; jj < indices.length; jj++) {
        const k = indices[jj];
        if (k === i0 || k === i1 || k === i2) continue;
        const p = poly[k];
        if (pointInTriangle(p.x, p.y, a, b, c)) {
          contains = true;
          break;
        }
      }
      if (contains) continue;

      tris.push([a, b, c].map((p) => ({ x: p.x, y: p.y })));
      indices.splice(ii, 1);
      clipped = true;
      break;
    }

    if (!clipped) break;
  }

  if (indices.length === 3) {
    tris.push(indices.map((idx) => ({ x: poly[idx].x, y: poly[idx].y })));
  }

  if (!tris.length) {
    // Fallback fan triangulation (guardrail for degenerate cases).
    for (let i = 1; i < poly.length - 1; i++) {
      tris.push([
        { x: poly[0].x, y: poly[0].y },
        { x: poly[i].x, y: poly[i].y },
        { x: poly[i + 1].x, y: poly[i + 1].y },
      ]);
    }
  }

  return tris;
}

function triangleAreaAbs(tri) {
  const [a, b, c] = tri;
  return Math.abs((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) * 0.5);
}

function triangleAspectRatio(tri) {
  const d = (p, q) => Math.hypot((q.x - p.x), (q.y - p.y));
  const l0 = d(tri[0], tri[1]);
  const l1 = d(tri[1], tri[2]);
  const l2 = d(tri[2], tri[0]);
  const maxL = Math.max(l0, l1, l2);
  const minL = Math.max(EPS, Math.min(l0, l1, l2));
  return maxL / minL;
}

function triangleCentroid(tri) {
  return {
    x: (tri[0].x + tri[1].x + tri[2].x) / 3,
    y: (tri[0].y + tri[1].y + tri[2].y) / 3,
  };
}

function orientation(a, b, c) {
  const v = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  if (Math.abs(v) <= 1e-8) return 0;
  return v > 0 ? 1 : -1;
}

function pointsNear(a, b, eps = 1e-6) {
  return Math.abs(a.x - b.x) <= eps && Math.abs(a.y - b.y) <= eps;
}

function sharesEndpoint(a, b, c, d) {
  return pointsNear(a, c) || pointsNear(a, d) || pointsNear(b, c) || pointsNear(b, d);
}

function onSegment(a, b, p) {
  return pointOnSegment(p.x, p.y, a.x, a.y, b.x, b.y, 1e-6);
}

function segmentsIntersect(a, b, c, d) {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);

  if (o1 !== o2 && o3 !== o4) return true;
  if (o1 === 0 && onSegment(a, b, c)) return true;
  if (o2 === 0 && onSegment(a, b, d)) return true;
  if (o3 === 0 && onSegment(c, d, a)) return true;
  if (o4 === 0 && onSegment(c, d, b)) return true;
  return false;
}

function isProxyTriangleValid(tri, sourcePoly, maxAspect = 14) {
  if (!Array.isArray(tri) || tri.length !== 3) return false;
  const area = triangleAreaAbs(tri);
  if (!(area > 1e-5)) return false;

  const aspect = triangleAspectRatio(tri);
  if (!Number.isFinite(aspect) || aspect > maxAspect) return false;

  const centroid = triangleCentroid(tri);
  if (!pointInPolygonInclusive(centroid.x, centroid.y, sourcePoly)) return false;

  const mids = [
    { x: (tri[0].x + tri[1].x) * 0.5, y: (tri[0].y + tri[1].y) * 0.5 },
    { x: (tri[1].x + tri[2].x) * 0.5, y: (tri[1].y + tri[2].y) * 0.5 },
    { x: (tri[2].x + tri[0].x) * 0.5, y: (tri[2].y + tri[0].y) * 0.5 },
  ];
  if (!mids.every((m) => pointInPolygonInclusive(m.x, m.y, sourcePoly))) return false;

  const triEdges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
  for (const [ta, tb] of triEdges) {
    for (let i = 0; i < sourcePoly.length; i++) {
      const pa = sourcePoly[i];
      const pb = sourcePoly[(i + 1) % sourcePoly.length];
      if (sharesEndpoint(ta, tb, pa, pb)) continue;
      if (segmentsIntersect(ta, tb, pa, pb)) return false;
    }
  }

  return true;
}

function sanitizeFiniteLocalPolygon(poly, eps = EPS) {
  if (!Array.isArray(poly) || poly.length < 3) return [];
  const out = [];
  for (const v of poly) {
    const x = Number(v?.x);
    const y = Number(v?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const prev = out[out.length - 1];
    if (prev && Math.hypot(x - prev.x, y - prev.y) <= eps) continue;
    out.push({ x, y });
  }
  if (out.length >= 2) {
    const first = out[0];
    const last = out[out.length - 1];
    if (Math.hypot(first.x - last.x, first.y - last.y) <= eps) out.pop();
  }
  return out.length >= 3 ? out : [];
}

function buildCollisionPolysLocal(body) {
  const basePolys = Array.isArray(body?.subPolysLocal) && body.subPolysLocal.length
    ? body.subPolysLocal
    : [Array.isArray(body?.verticesLocal) && body.verticesLocal.length >= 3 ? body.verticesLocal : rigidVerticesLocalFallback(body)];

  const out = [];
  for (const poly of basePolys) {
    if (!Array.isArray(poly) || poly.length < 3) continue;
    const finitePoly = sanitizeFiniteLocalPolygon(poly);
    if (finitePoly.length < 3) continue;

    const clean = ensureCCW(finitePoly);
    if (isConvexPolygon(clean)) {
      out.push(clean);
      continue;
    }

    const rawTris = triangulateEarClip(clean);
    const valid = rawTris
      .map((tri) => ensureCCW(tri))
      .filter((tri) => isProxyTriangleValid(tri, clean));

    // Fallback safety: if strict filter rejects everything, keep only triangles whose centroid+edge mids stay inside.
    const fallback = rawTris
      .map((tri) => ensureCCW(tri))
      .filter((tri) => {
        const c = triangleCentroid(tri);
        if (!pointInPolygonInclusive(c.x, c.y, clean)) return false;
        const mids = [
          { x: (tri[0].x + tri[1].x) * 0.5, y: (tri[0].y + tri[1].y) * 0.5 },
          { x: (tri[1].x + tri[2].x) * 0.5, y: (tri[1].y + tri[2].y) * 0.5 },
          { x: (tri[2].x + tri[0].x) * 0.5, y: (tri[2].y + tri[0].y) * 0.5 },
        ];
        return mids.every((m) => pointInPolygonInclusive(m.x, m.y, clean));
      });

    const chosen = valid.length ? valid : fallback;
    for (const tri of chosen) out.push(tri);
  }
  return out;
}

function collisionPolyCacheKey(body) {
  if (!body) return 'none';
  const verts = Array.isArray(body.verticesLocal) ? body.verticesLocal : null;
  const sub = Array.isArray(body.subPolysLocal) ? body.subPolysLocal : null;
  const vKey = verts
    ? `v:${verts.length}:${verts.map((p) => `${Number(p?.x) || 0},${Number(p?.y) || 0}`).join('|')}`
    : 'v:none';
  const sKey = sub
    ? `s:${sub.length}:${sub.map((poly) => (Array.isArray(poly)
      ? poly.map((p) => `${Number(p?.x) || 0},${Number(p?.y) || 0}`).join('|')
      : 'bad')).join('||')}`
    : 's:none';
  return `${vKey};${sKey};r:${Number(body.r) || 0};sides:${Number(body.sides) || 0}`;
}

function bodyCollisionPolysLocal(body) {
  if (!body) return [];
  const cacheKey = collisionPolyCacheKey(body);
  if (!Array.isArray(body._collisionPolysLocal)
    || body._collisionPolysLocal.length === 0
    || body._collisionPolysLocalKey !== cacheKey) {
    body._collisionPolysLocal = buildCollisionPolysLocal(body);
    body._collisionPolysLocalKey = cacheKey;
  }
  return body._collisionPolysLocal;
}

function getRigidBroadphaseRadius(body) {
  if (!body) return 0.5;
  const polysLocal = bodyCollisionPolysLocal(body);
  const cacheKey = body._collisionPolysLocalKey || 'none';
  if (Number.isFinite(body._broadphaseRadiusLocal)
    && body._broadphaseRadiusLocal > 0
    && body._broadphaseRadiusLocalKey === cacheKey) {
    return body._broadphaseRadiusLocal;
  }

  let maxR = 0;
  for (const poly of polysLocal) {
    if (!Array.isArray(poly)) continue;
    for (const v of poly) {
      const x = Number(v?.x);
      const y = Number(v?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const d = Math.hypot(x, y);
      if (d > maxR) maxR = d;
    }
  }
  if (!(maxR > 0)) {
    maxR = Math.max(0.5, finiteOr(body?.r, 0.5));
  }
  body._broadphaseRadiusLocal = maxR;
  body._broadphaseRadiusLocalKey = cacheKey;
  return maxR;
}

export function getRigidCollisionPolysWorld(body) {
  const polysLocal = bodyCollisionPolysLocal(body);
  const theta = finiteOr(body?.theta, 0);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const bx = finiteOr(body?.x, 0);
  const by = finiteOr(body?.y, 0);
  return polysLocal.map((poly) => poly.map((v) => ({
    x: bx + v.x * c - v.y * s,
    y: by + v.x * s + v.y * c,
  })));
}

const SPATIAL_KEY_BIAS = 1 << 20; // 1,048,576 cells each direction
const SPATIAL_KEY_STRIDE = 1 << 21; // 2,097,152

function packSpatialCellKey(cx, cy) {
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  if (cx <= -SPATIAL_KEY_BIAS || cx >= SPATIAL_KEY_BIAS) return null;
  if (cy <= -SPATIAL_KEY_BIAS || cy >= SPATIAL_KEY_BIAS) return null;
  return (cx + SPATIAL_KEY_BIAS) * SPATIAL_KEY_STRIDE + (cy + SPATIAL_KEY_BIAS);
}

function buildRigidSpatialHashState(rigidBodies, cellSize) {
  const bodies = Array.isArray(rigidBodies) ? rigidBodies : [];
  const bodyCount = bodies.length;
  const bruteForcePairs = bodyCount > 1 ? (bodyCount * (bodyCount - 1)) / 2 : 0;

  const cells = new Map();
  const rigidMeta = new Array(bodyCount);
  const bodyCellKeys = new Array(bodyCount);
  const cellSpans = new Int32Array(Math.max(0, bodyCount * 4));
  let cellSpanSignature = 2166136261 >>> 0;
  const mixCellSpanSignature = (value) => {
    const v = (Number(value) | 0) >>> 0;
    cellSpanSignature ^= v;
    cellSpanSignature = Math.imul(cellSpanSignature, 16777619) >>> 0;
  };

  let occupiedBodyWrites = 0;

  for (let i = 0; i < bodyCount; i++) {
    const rb = bodies[i];
    let minCx = 0;
    let maxCx = -1;
    let minCy = 0;
    let maxCy = -1;

    const cellKeys = [];
    bodyCellKeys[i] = cellKeys;

    if (!rb) {
      rigidMeta[i] = {
        bx: 0,
        by: 0,
        radius: 0.5,
      };
    } else {
      const bx = finiteOr(rb.x, 0);
      const by = finiteOr(rb.y, 0);
      const radius = Math.max(0.5, getRigidBroadphaseRadius(rb));
      rigidMeta[i] = { bx, by, radius };

      minCx = Math.floor((bx - radius) / cellSize);
      maxCx = Math.floor((bx + radius) / cellSize);
      minCy = Math.floor((by - radius) / cellSize);
      maxCy = Math.floor((by + radius) / cellSize);

      for (let cy = minCy; cy <= maxCy; cy++) {
        for (let cx = minCx; cx <= maxCx; cx++) {
          const key = packSpatialCellKey(cx, cy);
          if (key == null) continue;
          let bucket = cells.get(key);
          if (!bucket) {
            bucket = [];
            cells.set(key, bucket);
          }
          bucket.push(i);
          cellKeys.push(key);
          occupiedBodyWrites += 1;
        }
      }
    }

    const spanOffset = i * 4;
    cellSpans[spanOffset] = minCx;
    cellSpans[spanOffset + 1] = maxCx;
    cellSpans[spanOffset + 2] = minCy;
    cellSpans[spanOffset + 3] = maxCy;
    mixCellSpanSignature(minCx);
    mixCellSpanSignature(maxCx);
    mixCellSpanSignature(minCy);
    mixCellSpanSignature(maxCy);
  }

  const sortedKeys = Array.from(cells.keys()).sort((a, b) => a - b);
  let maxBodiesPerCell = 0;
  for (const key of sortedKeys) {
    const ids = cells.get(key) || [];
    if (ids.length > maxBodiesPerCell) maxBodiesPerCell = ids.length;
  }

  return {
    bodies,
    bodyCount,
    bruteForcePairs,
    cellSize,
    cells,
    sortedKeys,
    rigidMeta,
    bodyCellKeys,
    occupiedBodyWrites,
    maxBodiesPerCell,
    cellSpans,
    cellSpanSignature,
  };
}

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function rigidCellSpansEqual(a, b) {
  if (!(a instanceof Int32Array) || !(b instanceof Int32Array)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function resolveRigidRigidCandidateBackend(rawBackend) {
  if (!rawBackend) return null;
  if (typeof rawBackend === 'function') {
    return { label: rawBackend.name || 'custom', buildCandidates: rawBackend };
  }
  if (typeof rawBackend === 'object' && typeof rawBackend.buildCandidates === 'function') {
    return {
      label: typeof rawBackend.label === 'string' && rawBackend.label.length > 0
        ? rawBackend.label
        : 'custom',
      buildCandidates: rawBackend.buildCandidates,
    };
  }
  return null;
}

function deriveRigidRigidCandidatesFromState(state, options = {}) {
  const bodyCount = Number(state?.bodyCount) | 0;
  const backend = resolveRigidRigidCandidateBackend(options?.backend);

  if (backend) {
    try {
      const backendResult = backend.buildCandidates(state, options);
      const backendPairs = Array.isArray(backendResult?.pairs) ? backendResult.pairs : null;
      if (backendPairs) {
        const checkedPairs = backendPairs.length;
        const prunedPairs = Math.max(0, state.bruteForcePairs - checkedPairs);
        const reductionPct = state.bruteForcePairs > 0
          ? (prunedPairs / state.bruteForcePairs) * 100
          : 0;
        const backendStats = backendResult?.stats && typeof backendResult.stats === 'object'
          ? backendResult.stats
          : {};
        const dedupeMs = Number(backendStats.dedupeMs) || 0;
        const sortMs = Number(backendStats.sortMs) || 0;
        const totalBuildMs = Number(backendStats.totalBuildMs) || (dedupeMs + sortMs);

        return {
          pairs: backendPairs,
          stats: {
            bodyCount: state.bodyCount,
            bruteForcePairs: state.bruteForcePairs,
            checkedPairs,
            prunedPairs,
            reductionPct,
            cellSize: state.cellSize,
            occupiedCells: state.sortedKeys.length,
            occupiedBodyWrites: state.occupiedBodyWrites,
            maxBodiesPerCell: state.maxBodiesPerCell,
            avgBodiesPerCell: state.sortedKeys.length > 0 ? (state.occupiedBodyWrites / state.sortedKeys.length) : 0,
            emitAttempts: Number(backendStats.emitAttempts) || 0,
            duplicatesRejected: Number(backendStats.duplicatesRejected) || 0,
            pairsOut: checkedPairs,
            dedupeMs,
            sortMs,
            totalBuildMs,
            cellSpanSignature: state.cellSpanSignature,
            backend: backend.label,
            reuseHit: false,
            reuseMiss: false,
          },
        };
      }
    } catch {
      // Fall back to JS candidate builder if backend throws.
    }
  }

  const candidatePairs = [];
  let emitAttempts = 0;
  let duplicatesRejected = 0;
  let dedupeMs = 0;
  let sortMs = 0;

  const cells = state?.cells;
  const bodyCellKeys = Array.isArray(state?.bodyCellKeys) ? state.bodyCellKeys : [];
  const seenStamp = new Int32Array(Math.max(0, bodyCount));
  let stampToken = 1;

  for (let i = 0; i < bodyCount; i++) {
    const cellKeys = bodyCellKeys[i];
    if (!Array.isArray(cellKeys) || cellKeys.length === 0) continue;
    const bodyDedupeStartMs = nowMs();
    const bodyStamp = stampToken++;
    const neighbors = [];

    for (const key of cellKeys) {
      const ids = cells?.get(key) || [];
      for (let idx = 0; idx < ids.length; idx++) {
        emitAttempts += 1;
        const j = Number(ids[idx]) | 0;
        if (j <= i) {
          duplicatesRejected += 1;
          continue;
        }
        if (seenStamp[j] === bodyStamp) {
          duplicatesRejected += 1;
          continue;
        }
        seenStamp[j] = bodyStamp;
        neighbors.push(j);
      }
    }

    dedupeMs += nowMs() - bodyDedupeStartMs;

    if (neighbors.length > 1) {
      const bodySortStartMs = nowMs();
      neighbors.sort((a, b) => a - b);
      sortMs += nowMs() - bodySortStartMs;
    }

    for (let k = 0; k < neighbors.length; k++) {
      candidatePairs.push([i, neighbors[k]]);
    }
  }

  const checkedPairs = candidatePairs.length;
  const prunedPairs = Math.max(0, state.bruteForcePairs - checkedPairs);
  const reductionPct = state.bruteForcePairs > 0
    ? (prunedPairs / state.bruteForcePairs) * 100
    : 0;

  return {
    pairs: candidatePairs,
    stats: {
      bodyCount: state.bodyCount,
      bruteForcePairs: state.bruteForcePairs,
      checkedPairs,
      prunedPairs,
      reductionPct,
      cellSize: state.cellSize,
      occupiedCells: state.sortedKeys.length,
      occupiedBodyWrites: state.occupiedBodyWrites,
      maxBodiesPerCell: state.maxBodiesPerCell,
      avgBodiesPerCell: state.sortedKeys.length > 0 ? (state.occupiedBodyWrites / state.sortedKeys.length) : 0,
      emitAttempts,
      duplicatesRejected,
      pairsOut: checkedPairs,
      dedupeMs,
      sortMs,
      totalBuildMs: dedupeMs + sortMs,
      cellSpanSignature: state.cellSpanSignature,
      backend: 'js',
      reuseHit: false,
      reuseMiss: false,
    },
  };
}

function buildSoftFeatureSpatialState(softNodes, softSprings, cellSize, edgeBodyModeBlock) {
  const nodes = Array.isArray(softNodes) ? softNodes : [];
  const springs = Array.isArray(softSprings) ? softSprings : [];
  const nodeCount = nodes.length;

  const nodeCells = new Map();
  let nodeWrites = 0;
  let maxNodesPerCell = 0;
  let maxSoftNodeRadius = 0;
  for (let ni = 0; ni < nodeCount; ni++) {
    const node = nodes[ni];
    if (!node) continue;
    const x = Number(node.x);
    const y = Number(node.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const radius = Math.max(0.2, finiteOr(node.r, 1));
    if (radius > maxSoftNodeRadius) maxSoftNodeRadius = radius;
    const cx = Math.floor(x / cellSize);
    const cy = Math.floor(y / cellSize);
    const key = packSpatialCellKey(cx, cy);
    if (key == null) continue;
    let bucket = nodeCells.get(key);
    if (!bucket) {
      bucket = [];
      nodeCells.set(key, bucket);
    }
    bucket.push(ni);
    nodeWrites += 1;
    if (bucket.length > maxNodesPerCell) maxNodesPerCell = bucket.length;
  }

  const edgeCells = new Map();
  let edgeWrites = 0;
  let maxEdgesPerCell = 0;
  let blockedEdgeCount = 0;

  for (let si = 0; si < springs.length; si++) {
    const sp = springs[si];
    if (!Array.isArray(sp)) continue;
    const bodyMode = Number(sp?.[3]);
    if (bodyMode !== edgeBodyModeBlock) continue;

    const ai = Number(sp?.[0]) | 0;
    const bi = Number(sp?.[1]) | 0;
    if (ai < 0 || bi < 0 || ai >= nodeCount || bi >= nodeCount) continue;
    const a = nodes[ai];
    const b = nodes[bi];
    if (!a || !b) continue;

    const ax = Number(a.x);
    const ay = Number(a.y);
    const bx = Number(b.x);
    const by = Number(b.y);
    if (![ax, ay, bx, by].every(Number.isFinite)) continue;

    blockedEdgeCount += 1;
    const minCx = Math.floor(Math.min(ax, bx) / cellSize);
    const maxCx = Math.floor(Math.max(ax, bx) / cellSize);
    const minCy = Math.floor(Math.min(ay, by) / cellSize);
    const maxCy = Math.floor(Math.max(ay, by) / cellSize);

    for (let cy = minCy; cy <= maxCy; cy++) {
      for (let cx = minCx; cx <= maxCx; cx++) {
        const key = packSpatialCellKey(cx, cy);
        if (key == null) continue;
        let bucket = edgeCells.get(key);
        if (!bucket) {
          bucket = [];
          edgeCells.set(key, bucket);
        }
        bucket.push(si);
        edgeWrites += 1;
        if (bucket.length > maxEdgesPerCell) maxEdgesPerCell = bucket.length;
      }
    }
  }

  return {
    nodes,
    springs,
    nodeCount,
    blockedEdgeCount,
    nodeCells,
    edgeCells,
    nodeWrites,
    edgeWrites,
    maxNodesPerCell,
    maxEdgesPerCell,
    maxSoftNodeRadius,
  };
}

function deriveRigidSoftCandidatesFromState(state, softState, options = {}) {
  const rigidCount = state.bodyCount;
  const nodeCount = softState.nodeCount;
  const nodePad = Number.isFinite(Number(options.nodePad)) ? Math.max(0, Number(options.nodePad)) : 0.8;
  const edgePad = Number.isFinite(Number(options.edgePad)) ? Math.max(0, Number(options.edgePad)) : 0.8;

  const nodeCandidatesByRigid = new Array(rigidCount);
  const edgeCandidatesByRigid = new Array(rigidCount);
  let candidateNodeChecks = 0;
  let candidateEdgeChecks = 0;
  let maxRigidRadius = 0;

  for (let rbi = 0; rbi < rigidCount; rbi++) {
    const meta = state.rigidMeta[rbi] || { bx: 0, by: 0, radius: 0.5 };
    const bx = Number(meta.bx) || 0;
    const by = Number(meta.by) || 0;
    const rigidRadius = Math.max(0.5, Number(meta.radius) || 0.5);
    if (rigidRadius > maxRigidRadius) maxRigidRadius = rigidRadius;

    const nodeReach = rigidRadius + softState.maxSoftNodeRadius + nodePad;
    const nodeSet = new Set();
    const nodeMinCx = Math.floor((bx - nodeReach) / state.cellSize);
    const nodeMaxCx = Math.floor((bx + nodeReach) / state.cellSize);
    const nodeMinCy = Math.floor((by - nodeReach) / state.cellSize);
    const nodeMaxCy = Math.floor((by + nodeReach) / state.cellSize);
    for (let cy = nodeMinCy; cy <= nodeMaxCy; cy++) {
      for (let cx = nodeMinCx; cx <= nodeMaxCx; cx++) {
        const key = packSpatialCellKey(cx, cy);
        if (key == null) continue;
        const bucket = softState.nodeCells.get(key);
        if (!bucket) continue;
        for (const ni of bucket) nodeSet.add(ni);
      }
    }
    const nodeCandidates = Array.from(nodeSet).sort((a, b) => a - b);
    nodeCandidatesByRigid[rbi] = nodeCandidates;
    candidateNodeChecks += nodeCandidates.length;

    const edgeReach = rigidRadius + edgePad;
    const edgeSet = new Set();
    const edgeMinCx = Math.floor((bx - edgeReach) / state.cellSize);
    const edgeMaxCx = Math.floor((bx + edgeReach) / state.cellSize);
    const edgeMinCy = Math.floor((by - edgeReach) / state.cellSize);
    const edgeMaxCy = Math.floor((by + edgeReach) / state.cellSize);
    for (let cy = edgeMinCy; cy <= edgeMaxCy; cy++) {
      for (let cx = edgeMinCx; cx <= edgeMaxCx; cx++) {
        const key = packSpatialCellKey(cx, cy);
        if (key == null) continue;
        const bucket = softState.edgeCells.get(key);
        if (!bucket) continue;
        for (const si of bucket) edgeSet.add(si);
      }
    }
    const edgeCandidates = Array.from(edgeSet).sort((a, b) => a - b);
    edgeCandidatesByRigid[rbi] = edgeCandidates;
    candidateEdgeChecks += edgeCandidates.length;
  }

  const rawNodeChecks = rigidCount * nodeCount;
  const rawEdgeChecks = rigidCount * softState.blockedEdgeCount;
  const prunedNodeChecks = Math.max(0, rawNodeChecks - candidateNodeChecks);
  const prunedEdgeChecks = Math.max(0, rawEdgeChecks - candidateEdgeChecks);

  return {
    nodeCandidatesByRigid,
    edgeCandidatesByRigid,
    stats: {
      rigidCount,
      nodeCount,
      blockedEdgeCount: softState.blockedEdgeCount,
      rawNodeChecks,
      candidateNodeChecks,
      prunedNodeChecks,
      nodeReductionPct: rawNodeChecks > 0 ? (prunedNodeChecks / rawNodeChecks) * 100 : 0,
      rawEdgeChecks,
      candidateEdgeChecks,
      prunedEdgeChecks,
      edgeReductionPct: rawEdgeChecks > 0 ? (prunedEdgeChecks / rawEdgeChecks) * 100 : 0,
      cellSize: state.cellSize,
      nodeOccupiedCells: softState.nodeCells.size,
      edgeOccupiedCells: softState.edgeCells.size,
      nodeWrites: softState.nodeWrites,
      edgeWrites: softState.edgeWrites,
      maxNodesPerCell: softState.maxNodesPerCell,
      maxEdgesPerCell: softState.maxEdgesPerCell,
      maxRigidRadius,
      maxSoftNodeRadius: softState.maxSoftNodeRadius,
      nodePad,
      edgePad,
    },
  };
}

export function buildCollisionPhaseSceneCache(rigidBodies, softNodes, softSprings, options = {}) {
  const rawCellSize = Number(options.cellSize);
  const cellSize = Number.isFinite(rawCellSize) ? Math.max(0.25, rawCellSize) : 12;
  const includeRigidRigid = options.includeRigidRigid !== false;
  const includeRigidSoft = options.includeRigidSoft !== false;
  const edgeBodyModeBlock = Number.isFinite(Number(options.edgeBodyModeBlock))
    ? Number(options.edgeBodyModeBlock)
    : 1;

  const rigidState = buildRigidSpatialHashState(rigidBodies, cellSize);
  const rigidRigidReuseCache = options?.rigidRigidReuseCache && typeof options.rigidRigidReuseCache === 'object'
    ? options.rigidRigidReuseCache
    : null;

  let rigidRigid = {
    pairs: [],
    stats: {
      bodyCount: rigidState.bodyCount,
      bruteForcePairs: rigidState.bruteForcePairs,
      checkedPairs: 0,
      prunedPairs: rigidState.bruteForcePairs,
      reductionPct: rigidState.bruteForcePairs > 0 ? 100 : 0,
      cellSize,
      occupiedCells: rigidState.sortedKeys.length,
      occupiedBodyWrites: rigidState.occupiedBodyWrites,
      maxBodiesPerCell: rigidState.maxBodiesPerCell,
      avgBodiesPerCell: rigidState.sortedKeys.length > 0
        ? (rigidState.occupiedBodyWrites / rigidState.sortedKeys.length)
        : 0,
      emitAttempts: 0,
      duplicatesRejected: 0,
      pairsOut: 0,
      dedupeMs: 0,
      sortMs: 0,
      totalBuildMs: 0,
      cellSpanSignature: rigidState.cellSpanSignature,
      reuseHit: false,
      reuseMiss: false,
    },
  };

  if (includeRigidRigid) {
    const canReuseRigidRigid = Boolean(
      rigidRigidReuseCache
      && rigidRigidReuseCache.lastBodyCount === rigidState.bodyCount
      && Number(rigidRigidReuseCache.lastCellSize) === Number(cellSize)
      && rigidCellSpansEqual(rigidRigidReuseCache.lastCellSpans, rigidState.cellSpans)
      && Array.isArray(rigidRigidReuseCache.lastPairs)
      && rigidRigidReuseCache.lastStats,
    );

    if (canReuseRigidRigid) {
      const cachedStats = rigidRigidReuseCache.lastStats;
      rigidRigid = {
        pairs: rigidRigidReuseCache.lastPairs,
        stats: {
          ...cachedStats,
          dedupeMs: 0,
          sortMs: 0,
          totalBuildMs: 0,
          emitAttempts: 0,
          duplicatesRejected: 0,
          pairsOut: Number(cachedStats.checkedPairs) || 0,
          cellSpanSignature: rigidState.cellSpanSignature,
          reuseHit: true,
          reuseMiss: false,
        },
      };
    } else {
      rigidRigid = deriveRigidRigidCandidatesFromState(rigidState, {
        backend: options?.rigidRigidCandidateBackend,
      });
      if (rigidRigidReuseCache) {
        rigidRigid.stats.reuseMiss = true;
        rigidRigidReuseCache.lastBodyCount = rigidState.bodyCount;
        rigidRigidReuseCache.lastCellSize = cellSize;
        rigidRigidReuseCache.lastCellSpans = rigidState.cellSpans.slice();
        rigidRigidReuseCache.lastPairs = rigidRigid.pairs;
        rigidRigidReuseCache.lastStats = {
          ...rigidRigid.stats,
          reuseHit: false,
          reuseMiss: false,
        };
      }
    }
  }

  let rigidSoft = {
    nodeCandidatesByRigid: new Array(rigidState.bodyCount).fill(null).map(() => []),
    edgeCandidatesByRigid: new Array(rigidState.bodyCount).fill(null).map(() => []),
    stats: {
      rigidCount: rigidState.bodyCount,
      nodeCount: Array.isArray(softNodes) ? softNodes.length : 0,
      blockedEdgeCount: 0,
      rawNodeChecks: 0,
      candidateNodeChecks: 0,
      prunedNodeChecks: 0,
      nodeReductionPct: 0,
      rawEdgeChecks: 0,
      candidateEdgeChecks: 0,
      prunedEdgeChecks: 0,
      edgeReductionPct: 0,
      cellSize,
      nodeOccupiedCells: 0,
      edgeOccupiedCells: 0,
      nodeWrites: 0,
      edgeWrites: 0,
      maxNodesPerCell: 0,
      maxEdgesPerCell: 0,
      maxRigidRadius: 0,
      maxSoftNodeRadius: 0,
      nodePad: Number.isFinite(Number(options.nodePad)) ? Math.max(0, Number(options.nodePad)) : 0.8,
      edgePad: Number.isFinite(Number(options.edgePad)) ? Math.max(0, Number(options.edgePad)) : 0.8,
    },
  };

  if (includeRigidSoft) {
    const softState = buildSoftFeatureSpatialState(softNodes, softSprings, cellSize, edgeBodyModeBlock);
    rigidSoft = deriveRigidSoftCandidatesFromState(rigidState, softState, options);
  }

  return {
    cellSize,
    rigidMeta: rigidState.rigidMeta,
    rigidRigid,
    rigidSoft,
  };
}

export function buildRigidRigidSpatialHashCandidates(rigidBodies, options = {}) {
  const scene = buildCollisionPhaseSceneCache(rigidBodies, [], [], {
    ...options,
    includeRigidRigid: true,
    includeRigidSoft: false,
  });
  return scene.rigidRigid;
}

export function buildRigidSoftSpatialHashCandidates(rigidBodies, softNodes, softSprings, options = {}) {
  const scene = buildCollisionPhaseSceneCache(rigidBodies, softNodes, softSprings, {
    ...options,
    includeRigidRigid: false,
    includeRigidSoft: true,
  });
  return scene.rigidSoft;
}

function polygonCenter(poly) {
  let sx = 0;
  let sy = 0;
  for (const p of poly) {
    sx += p.x;
    sy += p.y;
  }
  const k = 1 / Math.max(1, poly.length);
  return { x: sx * k, y: sy * k };
}

function projectOnAxis(poly, nx, ny) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const p of poly) {
    const d = p.x * nx + p.y * ny;
    if (d < min) min = d;
    if (d > max) max = d;
  }
  return { min, max };
}

function satConvexCollision(polyA, polyB) {
  let minOverlap = Number.POSITIVE_INFINITY;
  let bestNx = 0;
  let bestNy = 0;

  const checkAxes = (poly) => {
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const ex = b.x - a.x;
      const ey = b.y - a.y;
      const len = Math.hypot(ex, ey);
      if (len < 1e-8) continue;
      const nx = -ey / len;
      const ny = ex / len;

      const pa = projectOnAxis(polyA, nx, ny);
      const pb = projectOnAxis(polyB, nx, ny);
      const overlap = Math.min(pa.max, pb.max) - Math.max(pa.min, pb.min);
      if (overlap <= 0) return false;
      if (overlap < minOverlap) {
        minOverlap = overlap;
        bestNx = nx;
        bestNy = ny;
      }
    }
    return true;
  };

  if (!checkAxes(polyA)) return null;
  if (!checkAxes(polyB)) return null;

  const ca = polygonCenter(polyA);
  const cb = polygonCenter(polyB);
  const toBx = cb.x - ca.x;
  const toBy = cb.y - ca.y;
  if (toBx * bestNx + toBy * bestNy < 0) {
    bestNx = -bestNx;
    bestNy = -bestNy;
  }

  return { overlap: minOverlap, nx: bestNx, ny: bestNy };
}

function supportPoint(poly, nx, ny) {
  let best = poly[0];
  let bestD = best.x * nx + best.y * ny;
  for (let i = 1; i < poly.length; i++) {
    const p = poly[i];
    const d = p.x * nx + p.y * ny;
    if (d > bestD) {
      bestD = d;
      best = p;
    }
  }
  return best;
}

function buildCollisionPolyMeta(poly) {
  if (!Array.isArray(poly) || poly.length === 0) {
    return {
      minX: 0,
      maxX: 0,
      minY: 0,
      maxY: 0,
      cx: 0,
      cy: 0,
      radius: 0,
    };
  }

  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let sx = 0;
  let sy = 0;
  let count = 0;

  for (const p of poly) {
    const x = Number(p?.x);
    const y = Number(p?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    sx += x;
    sy += y;
    count += 1;
  }

  if (count <= 0) {
    return {
      minX: 0,
      maxX: 0,
      minY: 0,
      maxY: 0,
      cx: 0,
      cy: 0,
      radius: 0,
    };
  }

  const cx = sx / count;
  const cy = sy / count;
  let radius = 0;
  for (const p of poly) {
    const x = Number(p?.x);
    const y = Number(p?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const d = Math.hypot(x - cx, y - cy);
    if (d > radius) radius = d;
  }

  return { minX, maxX, minY, maxY, cx, cy, radius };
}

function overlapAabb(metaA, metaB) {
  if (!metaA || !metaB) return true;
  return !(
    metaA.maxX < metaB.minX
    || metaB.maxX < metaA.minX
    || metaA.maxY < metaB.minY
    || metaB.maxY < metaA.minY
  );
}

export function resolveRigidVsRigidPolygonCollision(a, b, restitution = 0.3, debugInfo = null, cached = null) {
  const polysA = Array.isArray(cached?.polysA) ? cached.polysA : getRigidCollisionPolysWorld(a);
  const polysB = Array.isArray(cached?.polysB) ? cached.polysB : getRigidCollisionPolysWorld(b);
  if (!polysA.length || !polysB.length) return false;

  const metaA = (Array.isArray(cached?.metaA) && cached.metaA.length === polysA.length)
    ? cached.metaA
    : polysA.map((poly) => buildCollisionPolyMeta(poly));
  const metaB = (Array.isArray(cached?.metaB) && cached.metaB.length === polysB.length)
    ? cached.metaB
    : polysB.map((poly) => buildCollisionPolyMeta(poly));
  const stats = cached?.stats && typeof cached.stats === 'object' ? cached.stats : null;

  let best = null;
  for (let ai = 0; ai < polysA.length; ai++) {
    const pa = polysA[ai];
    const ma = metaA[ai];
    for (let bi = 0; bi < polysB.length; bi++) {
      const pb = polysB[bi];
      const mb = metaB[bi];
      if (stats) stats.rigidRigidProxyPairChecks = (Number(stats.rigidRigidProxyPairChecks) || 0) + 1;

      if (!overlapAabb(ma, mb)) {
        if (stats) stats.rigidRigidAabbRejected = (Number(stats.rigidRigidAabbRejected) || 0) + 1;
        continue;
      }

      const dx = (Number(ma?.cx) || 0) - (Number(mb?.cx) || 0);
      const dy = (Number(ma?.cy) || 0) - (Number(mb?.cy) || 0);
      const rr = (Number(ma?.radius) || 0) + (Number(mb?.radius) || 0);
      if ((dx * dx + dy * dy) > (rr * rr)) {
        if (stats) stats.rigidRigidRadiusRejected = (Number(stats.rigidRigidRadiusRejected) || 0) + 1;
        continue;
      }

      if (stats) stats.rigidRigidSatCalls = (Number(stats.rigidRigidSatCalls) || 0) + 1;
      const sat = satConvexCollision(pa, pb);
      if (!sat) continue;
      if (stats) stats.rigidRigidSatHits = (Number(stats.rigidRigidSatHits) || 0) + 1;
      if (!best || sat.overlap < best.overlap) {
        best = { ...sat, pa, pb, ai, bi };
      }
    }
  }
  if (!best) return false;
  if (best.overlap <= RIGID_CONTACT_SLOP) return false;

  const ma = Math.max(0.05, a.mass || 1);
  const mb = Math.max(0.05, b.mass || 1);
  const invA = 1 / ma;
  const invB = 1 / mb;
  const invSum = invA + invB;

  const correction = (best.overlap / Math.max(EPS, invSum)) * 0.78;
  a.x -= best.nx * correction * invA;
  a.y -= best.ny * correction * invA;
  b.x += best.nx * correction * invB;
  b.y += best.ny * correction * invB;

  const sa = supportPoint(best.pa, best.nx, best.ny);
  const sb = supportPoint(best.pb, -best.nx, -best.ny);
  const cx = (sa.x + sb.x) * 0.5;
  const cy = (sa.y + sb.y) * 0.5;

  const rax = cx - a.x;
  const ray = cy - a.y;
  const rbx = cx - b.x;
  const rby = cy - b.y;

  const vaX = (a.vx || 0) - (a.omega || 0) * ray;
  const vaY = (a.vy || 0) + (a.omega || 0) * rax;
  const vbX = (b.vx || 0) - (b.omega || 0) * rby;
  const vbY = (b.vy || 0) + (b.omega || 0) * rbx;
  const rvx = vbX - vaX;
  const rvy = vbY - vaY;
  const vn = rvx * best.nx + rvy * best.ny;

  const appendDebug = (jVal = 0) => {
    if (!debugInfo || !Array.isArray(debugInfo.contacts)) return;
    debugInfo.contacts.push({
      a: Number.isInteger(debugInfo.aIndex) ? debugInfo.aIndex : -1,
      b: Number.isInteger(debugInfo.bIndex) ? debugInfo.bIndex : -1,
      proxyA: best.ai,
      proxyB: best.bi,
      overlap: +best.overlap.toFixed(4),
      normal: [ +best.nx.toFixed(4), +best.ny.toFixed(4) ],
      contact: [ +cx.toFixed(3), +cy.toFixed(3) ],
      vn: +vn.toFixed(4),
      j: +jVal.toFixed(4),
      phase: debugInfo.phase || 'main',
      iter: Number.isInteger(debugInfo.iter) ? debugInfo.iter : -1,
    });
  };

  if (vn >= 0) {
    appendDebug(0);
    return true;
  }

  const invIA = 1 / Math.max(0.05, a.inertia || (0.5 * ma * Math.max(1, a.r || 1) ** 2));
  const invIB = 1 / Math.max(0.05, b.inertia || (0.5 * mb * Math.max(1, b.r || 1) ** 2));
  const raN = rax * best.ny - ray * best.nx;
  const rbN = rbx * best.ny - rby * best.nx;
  const denom = invA + invB + (raN * raN) * invIA + (rbN * rbN) * invIB;
  const rawJ = (-(1 + restitution) * vn) / Math.max(EPS, denom);
  const maxImpactSpeed = 24;
  const maxJ = ((1 + restitution) * Math.min(maxImpactSpeed, Math.abs(vn))) / Math.max(EPS, denom);
  const j = Math.max(-maxJ, Math.min(maxJ, rawJ));

  const jx = j * best.nx;
  const jy = j * best.ny;
  a.vx = (a.vx || 0) - jx * invA;
  a.vy = (a.vy || 0) - jy * invA;
  b.vx = (b.vx || 0) + jx * invB;
  b.vy = (b.vy || 0) + jy * invB;
  a.omega = (a.omega || 0) - (rax * jy - ray * jx) * invIA;
  b.omega = (b.omega || 0) + (rbx * jy - rby * jx) * invIB;
  appendDebug(j);
  return true;
}
