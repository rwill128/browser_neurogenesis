const EPS = 1e-6;
const RIGID_CONTACT_SLOP = 0.015;

export function rigidVerticesWorld(body) {
  if (!body) return [];
  if (Array.isArray(body.verticesLocal) && body.verticesLocal.length >= 3) {
    const c = Math.cos(body.theta || 0);
    const s = Math.sin(body.theta || 0);
    return body.verticesLocal.map((v) => ({
      x: body.x + v.x * c - v.y * s,
      y: body.y + v.x * s + v.y * c,
    }));
  }

  const sides = Math.max(3, body.sides || 3);
  const rot = (body.theta || 0) + (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
  const verts = [];
  for (let i = 0; i < sides; i++) {
    const a = rot + (i / sides) * Math.PI * 2;
    verts.push({ x: body.x + Math.cos(a) * body.r, y: body.y + Math.sin(a) * body.r });
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

    const intersect = ((yi > py) !== (yj > py))
      && (px < ((xj - xi) * (py - yi)) / Math.max(EPS, (yj - yi)) + xi);
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

// True polygonal rigid-vs-soft-node collision (concave-friendly).
export function resolveRigidVsSoftNodeCollision(rigid, node, vertsInput, restitution = 0.28) {
  const verts = Array.isArray(vertsInput) && vertsInput.length >= 3
    ? vertsInput
    : rigidVerticesWorld(rigid);
  if (!verts.length) return false;

  const finiteVerts = verts.filter((v) => Number.isFinite(v?.x) && Number.isFinite(v?.y));
  if (finiteVerts.length < 3) return false;

  const centroid = polygonCentroid(finiteVerts);

  let best = null;
  for (let i = 0; i < finiteVerts.length; i++) {
    const a = finiteVerts[i];
    const b = finiteVerts[(i + 1) % finiteVerts.length];
    const cp = closestPointOnSegment(node.x, node.y, a.x, a.y, b.x, b.y);
    if (!Number.isFinite(cp.x) || !Number.isFinite(cp.y)) continue;
    const dx = node.x - cp.x;
    const dy = node.y - cp.y;
    const d2 = dx * dx + dy * dy;
    if (!best || d2 < best.d2) {
      const out = edgeOutwardNormal(a.x, a.y, b.x, b.y, centroid.x, centroid.y);
      best = {
        d2,
        d: Math.sqrt(d2),
        cp,
        ax: a.x,
        ay: a.y,
        bx: b.x,
        by: b.y,
        nx: out.nx,
        ny: out.ny,
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

function buildCollisionPolysLocal(body) {
  const basePolys = Array.isArray(body?.subPolysLocal) && body.subPolysLocal.length
    ? body.subPolysLocal
    : [Array.isArray(body?.verticesLocal) && body.verticesLocal.length >= 3 ? body.verticesLocal : rigidVerticesLocalFallback(body)];

  const out = [];
  for (const poly of basePolys) {
    if (!Array.isArray(poly) || poly.length < 3) continue;
    const clean = ensureCCW(poly);
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

function bodyCollisionPolysLocal(body) {
  if (!body) return [];
  if (!Array.isArray(body._collisionPolysLocal) || body._collisionPolysLocal.length === 0) {
    body._collisionPolysLocal = buildCollisionPolysLocal(body);
  }
  return body._collisionPolysLocal;
}

export function getRigidCollisionPolysWorld(body) {
  const polysLocal = bodyCollisionPolysLocal(body);
  const c = Math.cos(body?.theta || 0);
  const s = Math.sin(body?.theta || 0);
  const bx = body?.x || 0;
  const by = body?.y || 0;
  return polysLocal.map((poly) => poly.map((v) => ({
    x: bx + v.x * c - v.y * s,
    y: by + v.x * s + v.y * c,
  })));
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

export function resolveRigidVsRigidPolygonCollision(a, b, restitution = 0.3, debugInfo = null) {
  const polysA = getRigidCollisionPolysWorld(a);
  const polysB = getRigidCollisionPolysWorld(b);
  if (!polysA.length || !polysB.length) return false;

  let best = null;
  for (let ai = 0; ai < polysA.length; ai++) {
    const pa = polysA[ai];
    for (let bi = 0; bi < polysB.length; bi++) {
      const pb = polysB[bi];
      const sat = satConvexCollision(pa, pb);
      if (!sat) continue;
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
