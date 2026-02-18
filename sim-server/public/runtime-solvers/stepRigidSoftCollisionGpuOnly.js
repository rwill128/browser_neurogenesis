/**
 * GPU-only runtime rigid↔soft collision pass.
 * Owns rigid-vs-soft collision stepping mechanics in the isolated gpu-only path,
 * keeping baseline/default runtime untouched.
 */

const EPS = 1e-6;

function finiteOr(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function rigidVerticesWorld(body) {
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

function pointOnSegment(px, py, ax, ay, bx, by, eps = EPS) {
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

function pointInPolygonInclusive(px, py, verts, eps = EPS) {
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
  if (ab2 < EPS) return { x: ax, y: ay, t: 0, abx, aby, ab2 };
  const apx = px - ax;
  const apy = py - ay;
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / ab2));
  return { x: ax + abx * t, y: ay + aby * t, t, abx, aby, ab2 };
}

function edgeOutwardNormal(ax, ay, bx, by, cx, cy) {
  const ex = bx - ax;
  const ey = by - ay;
  const len = Math.max(EPS, Math.hypot(ex, ey));
  let nx = -ey / len;
  let ny = ex / len;

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

export function resolveRigidVsSoftNodeCollisionGpuOnly(rigid, node, vertsInput, restitution = 0.28) {
  const verts = Array.isArray(vertsInput) && vertsInput.length >= 3
    ? vertsInput
    : rigidVerticesWorld(rigid);
  if (!verts.length) return false;

  const finiteVerts = sanitizeFinitePolygonVerts(verts);
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
      best = { d2, d: Math.sqrt(d2), cp, nx: out.nx, ny: out.ny };
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

export function resolveRigidVsSoftEdgeCollisionGpuOnly(rigid, a, b, restitution = 0.28) {
  const cp = closestPointOnSegment(rigid.x, rigid.y, a.x, a.y, b.x, b.y);
  let nx = rigid.x - cp.x;
  let ny = rigid.y - cp.y;
  let dist = Math.hypot(nx, ny);
  const minDist = Math.max(0.8, rigid.r || 1);
  if (dist >= minDist) return false;

  if (dist < 1e-6) {
    const invLen = 1 / Math.max(1e-6, Math.hypot(-cp.aby, cp.abx));
    nx = -cp.aby * invLen;
    ny = cp.abx * invLen;
    dist = 1e-6;
  } else {
    nx /= dist;
    ny /= dist;
  }

  const penetration = minDist - dist;
  rigid.x += nx * penetration * 0.92;
  rigid.y += ny * penetration * 0.92;

  const edgeVx = (a.vx + b.vx) * 0.5;
  const edgeVy = (a.vy + b.vy) * 0.5;
  const rvx = rigid.vx - edgeVx;
  const rvy = rigid.vy - edgeVy;
  const vn = rvx * nx + rvy * ny;
  if (vn < 0) {
    const j = -(1 + restitution) * vn;
    rigid.vx += nx * j;
    rigid.vy += ny * j;
  }
  return true;
}

function fnv1aMix(seed, value) {
  let h = seed >>> 0;
  h ^= Number(value) >>> 0;
  h = Math.imul(h, 16777619) >>> 0;
  return h >>> 0;
}

function buildRigidSoftCollisionWgslLayout({ rigidBodies, soft, hybridAttachedByRigid, edgeBodyModeBlock }) {
  const rigid = Array.isArray(rigidBodies) ? rigidBodies : [];
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springs = Array.isArray(soft?.springs) ? soft.springs : [];

  const nodePairRigidIndex = [];
  const nodePairNodeIndex = [];
  const edgePairRigidIndex = [];
  const edgePairSpringIndex = [];

  let signature = 0x811c9dc5;

  for (let rbi = 0; rbi < rigid.length; rbi++) {
    const attachedNodeSet = hybridAttachedByRigid?.get?.(rbi) || null;

    for (let ni = 0; ni < nodes.length; ni++) {
      if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
      nodePairRigidIndex.push(rbi);
      nodePairNodeIndex.push(ni);
      signature = fnv1aMix(signature, rbi);
      signature = fnv1aMix(signature, ni);
    }

    for (let si = 0; si < springs.length; si++) {
      const spring = springs[si];
      if (!Array.isArray(spring) || spring.length < 4) continue;
      const i = Number(spring[0]) | 0;
      const j = Number(spring[1]) | 0;
      const edgeBodyMode = spring[3];
      if (edgeBodyMode !== edgeBodyModeBlock) continue;
      if (i < 0 || j < 0 || i >= nodes.length || j >= nodes.length) continue;
      if (attachedNodeSet && (attachedNodeSet.has(i) || attachedNodeSet.has(j))) continue;
      edgePairRigidIndex.push(rbi);
      edgePairSpringIndex.push(si);
      signature = fnv1aMix(signature, rbi);
      signature = fnv1aMix(signature, si);
    }
  }

  const layout = {
    nodePairRigidIndex: new Uint32Array(nodePairRigidIndex),
    nodePairNodeIndex: new Uint32Array(nodePairNodeIndex),
    edgePairRigidIndex: new Uint32Array(edgePairRigidIndex),
    edgePairSpringIndex: new Uint32Array(edgePairSpringIndex),
  };

  const byteLength = layout.nodePairRigidIndex.byteLength
    + layout.nodePairNodeIndex.byteLength
    + layout.edgePairRigidIndex.byteLength
    + layout.edgePairSpringIndex.byteLength;

  return {
    layout,
    nodePairCount: layout.nodePairRigidIndex.length,
    edgePairCount: layout.edgePairRigidIndex.length,
    byteLength,
    signature: signature >>> 0,
  };
}

export function resolveRigidSoftCollisionPassGpuOnly({
  rigidBodies,
  soft,
  hybridAttachedByRigid,
  resolveRigidVsSoftNodeCollision = resolveRigidVsSoftNodeCollisionGpuOnly,
  resolveRigidVsSoftEdgeCollision = resolveRigidVsSoftEdgeCollisionGpuOnly,
  edgeBodyModeBlock,
  nodeSlop = 0.18,
  edgeSlop = 0.16,
  wgslOffload,
}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return;
  if (!soft || !Array.isArray(soft.nodes) || !Array.isArray(soft.springs)) return;

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const prep = buildRigidSoftCollisionWgslLayout({
      rigidBodies,
      soft,
      hybridAttachedByRigid,
      edgeBodyModeBlock,
    });
    wgslOffload.state.preparedLayout = prep.layout;
    wgslOffload.state.lastPreparedNodePairCount = prep.nodePairCount;
    wgslOffload.state.lastPreparedEdgePairCount = prep.edgePairCount;
    wgslOffload.state.lastPreparedLayoutBytes = prep.byteLength;
    wgslOffload.state.lastPreparedLayoutSignature = prep.signature;
    wgslOffload.state.lastSourceRoute = 'cpu-rigid-soft-candidate-layout';
    wgslOffload.state.lastMode = 'cpu-prepared';
    wgslOffload.state.lastError = null;
  }

  for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
    const rb = rigidBodies[rbi];
    const attachedNodeSet = hybridAttachedByRigid?.get?.(rbi) || null;

    for (let ni = 0; ni < soft.nodes.length; ni++) {
      if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
      const sn = soft.nodes[ni];
      resolveRigidVsSoftNodeCollision(rb, sn, null, nodeSlop);
    }

    for (const [i, j, _rest, edgeBodyMode] of soft.springs) {
      if (edgeBodyMode !== edgeBodyModeBlock) continue;
      if (attachedNodeSet && (attachedNodeSet.has(i) || attachedNodeSet.has(j))) continue;
      resolveRigidVsSoftEdgeCollision(rb, soft.nodes[i], soft.nodes[j], edgeSlop);
    }
  }
}
