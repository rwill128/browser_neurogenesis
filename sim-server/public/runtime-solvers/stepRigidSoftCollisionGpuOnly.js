/**
 * GPU-only runtime rigid↔soft collision pass.
 * Owns rigid-vs-soft collision stepping mechanics in the isolated gpu-only path,
 * keeping baseline/default runtime untouched.
 */

const EPS = 1e-6;
const FAST_RIGID_SOFT_BROADPHASE_INTERVAL = 2;

function getGpuOnlyPipelineModeProfile(offload) {
  const modeProfile = String(offload?.modeProfile || offload?.pipelineMode || '').trim().toLowerCase();
  if (modeProfile === 'gpu-only-fast') return 'gpu-only-fast';
  if (modeProfile === 'gpu-only-validated') return 'gpu-only-validated';
  return 'standard';
}

function isGpuOnlyFastMode(offload) {
  return getGpuOnlyPipelineModeProfile(offload) === 'gpu-only-fast';
}

function normalizeFastRigidSoftBroadphaseInterval(raw) {
  const n = Math.round(Number(raw));
  if (!Number.isFinite(n)) return FAST_RIGID_SOFT_BROADPHASE_INTERVAL;
  return Math.max(1, Math.min(8, n));
}

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
  let finiteVerts = null;
  let centroid = null;

  if (Array.isArray(vertsInput) && vertsInput.length >= 3) {
    finiteVerts = sanitizeFinitePolygonVerts(vertsInput);
    if (finiteVerts.length < 3) return false;
    centroid = polygonCentroid(finiteVerts);
  } else if (
    vertsInput
    && Array.isArray(vertsInput.finiteVerts)
    && vertsInput.finiteVerts.length >= 3
    && vertsInput.centroid
    && Number.isFinite(vertsInput.centroid.x)
    && Number.isFinite(vertsInput.centroid.y)
  ) {
    finiteVerts = vertsInput.finiteVerts;
    centroid = vertsInput.centroid;
  } else {
    const verts = rigidVerticesWorld(rigid);
    if (!verts.length) return false;
    finiteVerts = sanitizeFinitePolygonVerts(verts);
    if (finiteVerts.length < 3) return false;
    centroid = polygonCentroid(finiteVerts);
  }

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

function buildRigidSoftCollisionWgslLayout({ rigidBodies, soft, edgeBodyModeBlock }) {
  const rigid = Array.isArray(rigidBodies) ? rigidBodies : [];
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springs = Array.isArray(soft?.springs) ? soft.springs : [];

  const nodePairRigidIndex = [];
  const nodePairNodeIndex = [];
  const edgePairRigidIndex = [];
  const edgePairSpringIndex = [];
  const edgePairNodeAIndex = [];
  const edgePairNodeBIndex = [];

  let signature = 0x811c9dc5;

  for (let rbi = 0; rbi < rigid.length; rbi++) {
    for (let ni = 0; ni < nodes.length; ni++) {
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
      edgePairRigidIndex.push(rbi);
      edgePairSpringIndex.push(si);
      edgePairNodeAIndex.push(i);
      edgePairNodeBIndex.push(j);
      signature = fnv1aMix(signature, rbi);
      signature = fnv1aMix(signature, si);
      signature = fnv1aMix(signature, i);
      signature = fnv1aMix(signature, j);
    }
  }

  const layout = {
    nodePairRigidIndex: new Uint32Array(nodePairRigidIndex),
    nodePairNodeIndex: new Uint32Array(nodePairNodeIndex),
    edgePairRigidIndex: new Uint32Array(edgePairRigidIndex),
    edgePairSpringIndex: new Uint32Array(edgePairSpringIndex),
    edgePairNodeAIndex: new Uint32Array(edgePairNodeAIndex),
    edgePairNodeBIndex: new Uint32Array(edgePairNodeBIndex),
  };

  const byteLength = layout.nodePairRigidIndex.byteLength
    + layout.nodePairNodeIndex.byteLength
    + layout.edgePairRigidIndex.byteLength
    + layout.edgePairSpringIndex.byteLength
    + layout.edgePairNodeAIndex.byteLength
    + layout.edgePairNodeBIndex.byteLength;

  return {
    layout,
    nodePairCount: layout.nodePairRigidIndex.length,
    edgePairCount: layout.edgePairRigidIndex.length,
    byteLength,
    signature: signature >>> 0,
  };
}

export function buildActiveRigidSoftNodeNarrowphasePairs({ prep, activeMask }) {
  const nodePairCount = Number(prep?.nodePairCount) || 0;
  const rigidIndex = prep?.layout?.nodePairRigidIndex;
  const nodeIndex = prep?.layout?.nodePairNodeIndex;
  if (!(rigidIndex instanceof Uint32Array) || !(nodeIndex instanceof Uint32Array)) return null;
  if (!(activeMask instanceof Uint32Array) || activeMask.length !== nodePairCount) return null;

  const activeRigid = [];
  const activeNode = [];
  let signature = 0x811c9dc5;
  for (let i = 0; i < nodePairCount; i++) {
    if ((activeMask[i] >>> 0) !== 1) continue;
    const rbi = rigidIndex[i] >>> 0;
    const ni = nodeIndex[i] >>> 0;
    activeRigid.push(rbi);
    activeNode.push(ni);
    signature = fnv1aMix(signature, rbi);
    signature = fnv1aMix(signature, ni);
  }

  const compactRigidIndex = Uint32Array.from(activeRigid);
  const compactNodeIndex = Uint32Array.from(activeNode);
  return {
    compactRigidIndex,
    compactNodeIndex,
    pairCount: compactRigidIndex.length,
    byteLength: compactRigidIndex.byteLength + compactNodeIndex.byteLength,
    signature: signature >>> 0,
  };
}

export function buildActiveRigidSoftEdgeNarrowphasePairs({ prep, activeMask }) {
  const edgePairCount = Number(prep?.edgePairCount) || 0;
  const rigidIndex = prep?.layout?.edgePairRigidIndex;
  const springIndex = prep?.layout?.edgePairSpringIndex;
  const nodeAIndex = prep?.layout?.edgePairNodeAIndex;
  const nodeBIndex = prep?.layout?.edgePairNodeBIndex;
  if (!(rigidIndex instanceof Uint32Array)
    || !(springIndex instanceof Uint32Array)
    || !(nodeAIndex instanceof Uint32Array)
    || !(nodeBIndex instanceof Uint32Array)) return null;
  if (!(activeMask instanceof Uint32Array) || activeMask.length !== edgePairCount) return null;

  const activeRigid = [];
  const activeSpring = [];
  const activeNodeA = [];
  const activeNodeB = [];
  let signature = 0x811c9dc5;
  for (let i = 0; i < edgePairCount; i++) {
    if ((activeMask[i] >>> 0) !== 1) continue;
    const rbi = rigidIndex[i] >>> 0;
    const si = springIndex[i] >>> 0;
    const ai = nodeAIndex[i] >>> 0;
    const bi = nodeBIndex[i] >>> 0;
    activeRigid.push(rbi);
    activeSpring.push(si);
    activeNodeA.push(ai);
    activeNodeB.push(bi);
    signature = fnv1aMix(signature, rbi);
    signature = fnv1aMix(signature, si);
    signature = fnv1aMix(signature, ai);
    signature = fnv1aMix(signature, bi);
  }

  const compactRigidIndex = Uint32Array.from(activeRigid);
  const compactSpringIndex = Uint32Array.from(activeSpring);
  const compactNodeAIndex = Uint32Array.from(activeNodeA);
  const compactNodeBIndex = Uint32Array.from(activeNodeB);
  return {
    compactRigidIndex,
    compactSpringIndex,
    compactNodeAIndex,
    compactNodeBIndex,
    pairCount: compactRigidIndex.length,
    byteLength: compactRigidIndex.byteLength
      + compactSpringIndex.byteLength
      + compactNodeAIndex.byteLength
      + compactNodeBIndex.byteLength,
    signature: signature >>> 0,
  };
}

export function buildCompactRigidSoftNodePairsFromActiveIndices({ prep, activePairIndices }) {
  const nodePairCount = Number(prep?.nodePairCount) || 0;
  const rigidIndex = prep?.layout?.nodePairRigidIndex;
  const nodeIndex = prep?.layout?.nodePairNodeIndex;
  if (!(rigidIndex instanceof Uint32Array) || !(nodeIndex instanceof Uint32Array)) return null;
  if (!(activePairIndices instanceof Uint32Array)) return null;

  const compactRigidIndex = new Uint32Array(activePairIndices.length);
  const compactNodeIndex = new Uint32Array(activePairIndices.length);
  let count = 0;
  let signature = 0x811c9dc5;

  for (let i = 0; i < activePairIndices.length; i++) {
    let pairIndex = activePairIndices[i] >>> 0;
    if (pairIndex >= (nodePairCount >>> 0)) pairIndex = pairIndex % Math.max(1, nodePairCount >>> 0);
    const rbi = rigidIndex[pairIndex] >>> 0;
    const ni = nodeIndex[pairIndex] >>> 0;
    compactRigidIndex[count] = rbi;
    compactNodeIndex[count] = ni;
    signature = fnv1aMix(signature, rbi);
    signature = fnv1aMix(signature, ni);
    count += 1;
  }

  return {
    compactRigidIndex: compactRigidIndex.subarray(0, count),
    compactNodeIndex: compactNodeIndex.subarray(0, count),
    pairCount: count,
    byteLength: count * 8,
    signature: signature >>> 0,
  };
}

export function buildCompactRigidSoftEdgePairsFromActiveIndices({ prep, activePairIndices }) {
  const edgePairCount = Number(prep?.edgePairCount) || 0;
  const rigidIndex = prep?.layout?.edgePairRigidIndex;
  const springIndex = prep?.layout?.edgePairSpringIndex;
  const nodeAIndex = prep?.layout?.edgePairNodeAIndex;
  const nodeBIndex = prep?.layout?.edgePairNodeBIndex;
  if (!(rigidIndex instanceof Uint32Array)
    || !(springIndex instanceof Uint32Array)
    || !(nodeAIndex instanceof Uint32Array)
    || !(nodeBIndex instanceof Uint32Array)) return null;
  if (!(activePairIndices instanceof Uint32Array)) return null;

  const compactRigidIndex = new Uint32Array(activePairIndices.length);
  const compactSpringIndex = new Uint32Array(activePairIndices.length);
  const compactNodeAIndex = new Uint32Array(activePairIndices.length);
  const compactNodeBIndex = new Uint32Array(activePairIndices.length);
  let count = 0;
  let signature = 0x811c9dc5;

  for (let i = 0; i < activePairIndices.length; i++) {
    let pairIndex = activePairIndices[i] >>> 0;
    if (pairIndex >= (edgePairCount >>> 0)) pairIndex = pairIndex % Math.max(1, edgePairCount >>> 0);
    const rbi = rigidIndex[pairIndex] >>> 0;
    const si = springIndex[pairIndex] >>> 0;
    const ai = nodeAIndex[pairIndex] >>> 0;
    const bi = nodeBIndex[pairIndex] >>> 0;
    compactRigidIndex[count] = rbi;
    compactSpringIndex[count] = si;
    compactNodeAIndex[count] = ai;
    compactNodeBIndex[count] = bi;
    signature = fnv1aMix(signature, rbi);
    signature = fnv1aMix(signature, si);
    signature = fnv1aMix(signature, ai);
    signature = fnv1aMix(signature, bi);
    count += 1;
  }

  return {
    compactRigidIndex: compactRigidIndex.subarray(0, count),
    compactSpringIndex: compactSpringIndex.subarray(0, count),
    compactNodeAIndex: compactNodeAIndex.subarray(0, count),
    compactNodeBIndex: compactNodeBIndex.subarray(0, count),
    pairCount: count,
    byteLength: count * 16,
    signature: signature >>> 0,
  };
}

export function buildRigidSoftNarrowphaseWgslLayout({
  nodePairs,
  edgePairs,
}) {
  const nodeRigid = nodePairs?.compactRigidIndex;
  const nodeIndex = nodePairs?.compactNodeIndex;
  const edgeRigid = edgePairs?.compactRigidIndex;
  const edgeSpring = edgePairs?.compactSpringIndex;
  const edgeNodeA = edgePairs?.compactNodeAIndex;
  const edgeNodeB = edgePairs?.compactNodeBIndex;

  if (!(nodeRigid instanceof Uint32Array)
    || !(nodeIndex instanceof Uint32Array)
    || !(edgeRigid instanceof Uint32Array)
    || !(edgeSpring instanceof Uint32Array)
    || !(edgeNodeA instanceof Uint32Array)
    || !(edgeNodeB instanceof Uint32Array)) {
    return null;
  }

  let signature = 0x811c9dc5;
  signature = fnv1aMix(signature, nodeRigid.length >>> 0);
  signature = fnv1aMix(signature, edgeRigid.length >>> 0);

  for (let i = 0; i < nodeRigid.length; i++) {
    signature = fnv1aMix(signature, nodeRigid[i] >>> 0);
    signature = fnv1aMix(signature, nodeIndex[i] >>> 0);
  }

  for (let i = 0; i < edgeRigid.length; i++) {
    signature = fnv1aMix(signature, edgeRigid[i] >>> 0);
    signature = fnv1aMix(signature, edgeSpring[i] >>> 0);
    signature = fnv1aMix(signature, edgeNodeA[i] >>> 0);
    signature = fnv1aMix(signature, edgeNodeB[i] >>> 0);
  }

  return {
    nodePairRigidIndex: nodeRigid,
    nodePairNodeIndex: nodeIndex,
    edgePairRigidIndex: edgeRigid,
    edgePairSpringIndex: edgeSpring,
    edgePairNodeAIndex: edgeNodeA,
    edgePairNodeBIndex: edgeNodeB,
    nodePairCount: nodeRigid.length,
    edgePairCount: edgeRigid.length,
    byteLength: nodeRigid.byteLength
      + nodeIndex.byteLength
      + edgeRigid.byteLength
      + edgeSpring.byteLength
      + edgeNodeA.byteLength
      + edgeNodeB.byteLength,
    signature: signature >>> 0,
  };
}

export function buildRigidSoftNarrowphaseSceneWgslLayout({ rigidBodies, soft }) {
  if (!Array.isArray(rigidBodies) || !soft || !Array.isArray(soft.nodes) || !Array.isArray(soft.springs)) {
    return null;
  }

  const rigidCount = rigidBodies.length;
  const nodeCount = soft.nodes.length;
  const springCount = soft.springs.length;

  const rigidX = new Float32Array(rigidCount);
  const rigidY = new Float32Array(rigidCount);
  const rigidTheta = new Float32Array(rigidCount);
  const rigidVx = new Float32Array(rigidCount);
  const rigidVy = new Float32Array(rigidCount);
  const rigidOmega = new Float32Array(rigidCount);
  const rigidInvMass = new Float32Array(rigidCount);
  const rigidInvInertia = new Float32Array(rigidCount);
  const rigidMinX = new Float32Array(rigidCount);
  const rigidMinY = new Float32Array(rigidCount);
  const rigidMaxX = new Float32Array(rigidCount);
  const rigidMaxY = new Float32Array(rigidCount);
  const rigidVertexStart = new Uint32Array(rigidCount + 1);

  const rigidVertexXList = [];
  const rigidVertexYList = [];

  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  const nodeVx = new Float32Array(nodeCount);
  const nodeVy = new Float32Array(nodeCount);
  const nodeR = new Float32Array(nodeCount);
  const nodeInvMass = new Float32Array(nodeCount);

  const springNodeA = new Uint32Array(springCount);
  const springNodeB = new Uint32Array(springCount);
  const springRestLen = new Float32Array(springCount);

  let signature = 0x811c9dc5;
  signature = fnv1aMix(signature, rigidCount >>> 0);
  signature = fnv1aMix(signature, nodeCount >>> 0);
  signature = fnv1aMix(signature, springCount >>> 0);

  for (let i = 0; i < rigidCount; i++) {
    const rb = rigidBodies[i] || {};
    const x = Number(rb.x) || 0;
    const y = Number(rb.y) || 0;
    const theta = Number(rb.theta) || 0;
    const vx = Number(rb.vx) || 0;
    const vy = Number(rb.vy) || 0;
    const omega = Number(rb.omega) || 0;
    const invMass = 1 / Math.max(0.05, Number(rb.mass) || 0.05);
    const invInertia = 1 / Math.max(0.05, Number(rb.inertia) || (0.5 * Math.max(0.05, Number(rb.mass) || 1) * Math.max(1, Number(rb.r) || 1) ** 2));
    const minX = Number(rb._aabb?.minX) || 0;
    const minY = Number(rb._aabb?.minY) || 0;
    const maxX = Number(rb._aabb?.maxX) || 0;
    const maxY = Number(rb._aabb?.maxY) || 0;
    rigidX[i] = x;
    rigidY[i] = y;
    rigidTheta[i] = theta;
    rigidVx[i] = vx;
    rigidVy[i] = vy;
    rigidOmega[i] = omega;
    rigidInvMass[i] = invMass;
    rigidInvInertia[i] = invInertia;
    rigidMinX[i] = minX;
    rigidMinY[i] = minY;
    rigidMaxX[i] = maxX;
    rigidMaxY[i] = maxY;

    rigidVertexStart[i] = rigidVertexXList.length >>> 0;
    const finiteVerts = sanitizeFinitePolygonVerts(rigidVerticesWorld(rb));
    for (let vi = 0; vi < finiteVerts.length; vi++) {
      const vxw = Number(finiteVerts[vi]?.x) || 0;
      const vyw = Number(finiteVerts[vi]?.y) || 0;
      rigidVertexXList.push(vxw);
      rigidVertexYList.push(vyw);
      signature = fnv1aMix(signature, Math.fround(vxw));
      signature = fnv1aMix(signature, Math.fround(vyw));
    }

    signature = fnv1aMix(signature, Math.fround(x));
    signature = fnv1aMix(signature, Math.fround(y));
    signature = fnv1aMix(signature, Math.fround(theta));
    signature = fnv1aMix(signature, Math.fround(vx));
    signature = fnv1aMix(signature, Math.fround(vy));
    signature = fnv1aMix(signature, Math.fround(omega));
    signature = fnv1aMix(signature, Math.fround(invMass));
    signature = fnv1aMix(signature, Math.fround(invInertia));
  }

  rigidVertexStart[rigidCount] = rigidVertexXList.length >>> 0;
  const rigidVertexX = Float32Array.from(rigidVertexXList);
  const rigidVertexY = Float32Array.from(rigidVertexYList);
  signature = fnv1aMix(signature, rigidVertexX.length >>> 0);
  for (let i = 0; i < rigidVertexStart.length; i++) {
    signature = fnv1aMix(signature, rigidVertexStart[i] >>> 0);
  }

  for (let i = 0; i < nodeCount; i++) {
    const node = soft.nodes[i] || {};
    const x = Number(node.x) || 0;
    const y = Number(node.y) || 0;
    const vx = Number(node.vx) || 0;
    const vy = Number(node.vy) || 0;
    const r = Math.max(0, Number(node.r) || 0);
    const invMass = 1 / Math.max(0.02, Number(node.mass) || 0.02);
    nodeX[i] = x;
    nodeY[i] = y;
    nodeVx[i] = vx;
    nodeVy[i] = vy;
    nodeR[i] = r;
    nodeInvMass[i] = invMass;
    signature = fnv1aMix(signature, Math.fround(x));
    signature = fnv1aMix(signature, Math.fround(y));
    signature = fnv1aMix(signature, Math.fround(vx));
    signature = fnv1aMix(signature, Math.fround(vy));
    signature = fnv1aMix(signature, Math.fround(r));
  }

  for (let i = 0; i < springCount; i++) {
    const spring = soft.springs[i];
    const a = Number(spring?.[0]) | 0;
    const b = Number(spring?.[1]) | 0;
    const rest = Math.max(0, Number(spring?.[2]) || 0);
    springNodeA[i] = (a >>> 0);
    springNodeB[i] = (b >>> 0);
    springRestLen[i] = rest;
    signature = fnv1aMix(signature, a >>> 0);
    signature = fnv1aMix(signature, b >>> 0);
    signature = fnv1aMix(signature, Math.fround(rest));
  }

  return {
    rigidX,
    rigidY,
    rigidTheta,
    rigidVx,
    rigidVy,
    rigidOmega,
    rigidInvMass,
    rigidInvInertia,
    rigidMinX,
    rigidMinY,
    rigidMaxX,
    rigidMaxY,
    rigidVertexStart,
    rigidVertexX,
    rigidVertexY,
    nodeX,
    nodeY,
    nodeVx,
    nodeVy,
    nodeR,
    nodeInvMass,
    springNodeA,
    springNodeB,
    springRestLen,
    byteLength: rigidX.byteLength
      + rigidY.byteLength
      + rigidTheta.byteLength
      + rigidVx.byteLength
      + rigidVy.byteLength
      + rigidOmega.byteLength
      + rigidInvMass.byteLength
      + rigidInvInertia.byteLength
      + rigidMinX.byteLength
      + rigidMinY.byteLength
      + rigidMaxX.byteLength
      + rigidMaxY.byteLength
      + rigidVertexStart.byteLength
      + rigidVertexX.byteLength
      + rigidVertexY.byteLength
      + nodeX.byteLength
      + nodeY.byteLength
      + nodeVx.byteLength
      + nodeVy.byteLength
      + nodeR.byteLength
      + nodeInvMass.byteLength
      + springNodeA.byteLength
      + springNodeB.byteLength
      + springRestLen.byteLength,
    signature: signature >>> 0,
  };
}

const WGSL_WORKGROUP_SIZE = 64;

const rigidSoftNodeBroadphaseWgsl = /* wgsl */`
struct Params {
  nodePairCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodePairRigidIndex: array<u32>;
@group(0) @binding(2) var<storage, read> nodePairNodeIndex: array<u32>;
@group(0) @binding(3) var<storage, read> nodeX: array<f32>;
@group(0) @binding(4) var<storage, read> nodeY: array<f32>;
@group(0) @binding(5) var<storage, read> nodeR: array<f32>;
@group(0) @binding(6) var<storage, read> rigidMinX: array<f32>;
@group(0) @binding(7) var<storage, read> rigidMinY: array<f32>;
@group(0) @binding(8) var<storage, read> rigidMaxX: array<f32>;
@group(0) @binding(9) var<storage, read> rigidMaxY: array<f32>;
@group(0) @binding(10) var<storage, read_write> activePackedOut: array<atomic<u32>>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairIndex = gid.x;
  if (pairIndex >= params.nodePairCount) { return; }

  let rigidIndex = nodePairRigidIndex[pairIndex];
  let nodeIndex = nodePairNodeIndex[pairIndex];

  let x = nodeX[nodeIndex];
  let y = nodeY[nodeIndex];
  let r = max(0.0, nodeR[nodeIndex]);

  let minX = rigidMinX[rigidIndex] - r;
  let minY = rigidMinY[rigidIndex] - r;
  let maxX = rigidMaxX[rigidIndex] + r;
  let maxY = rigidMaxY[rigidIndex] + r;

  let activeFlag = select(0u, 1u, x >= minX && x <= maxX && y >= minY && y <= maxY);
  if (activeFlag == 1u) {
    let outOffset = atomicAdd(&activePackedOut[0], 1u) + 1u;
    if (outOffset <= params.nodePairCount) {
      atomicStore(&activePackedOut[outOffset], pairIndex);
    }
  }
}
`;

const rigidSoftEdgeBroadphaseWgsl = /* wgsl */`
struct Params {
  edgePairCount: u32,
  _pad0: u32,
  _pad1: u32,
  edgeSlop: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> edgePairRigidIndex: array<u32>;
@group(0) @binding(2) var<storage, read> edgePairNodeAIndex: array<u32>;
@group(0) @binding(3) var<storage, read> edgePairNodeBIndex: array<u32>;
@group(0) @binding(4) var<storage, read> nodeX: array<f32>;
@group(0) @binding(5) var<storage, read> nodeY: array<f32>;
@group(0) @binding(6) var<storage, read> rigidMinX: array<f32>;
@group(0) @binding(7) var<storage, read> rigidMinY: array<f32>;
@group(0) @binding(8) var<storage, read> rigidMaxX: array<f32>;
@group(0) @binding(9) var<storage, read> rigidMaxY: array<f32>;
@group(0) @binding(10) var<storage, read_write> activePackedOut: array<atomic<u32>>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairIndex = gid.x;
  if (pairIndex >= params.edgePairCount) { return; }

  let rigidIndex = edgePairRigidIndex[pairIndex];
  let nodeAIndex = edgePairNodeAIndex[pairIndex];
  let nodeBIndex = edgePairNodeBIndex[pairIndex];

  let ax = nodeX[nodeAIndex];
  let ay = nodeY[nodeAIndex];
  let bx = nodeX[nodeBIndex];
  let by = nodeY[nodeBIndex];

  let edgeMinX = min(ax, bx) - params.edgeSlop;
  let edgeMinY = min(ay, by) - params.edgeSlop;
  let edgeMaxX = max(ax, bx) + params.edgeSlop;
  let edgeMaxY = max(ay, by) + params.edgeSlop;

  let rigidMinXv = rigidMinX[rigidIndex];
  let rigidMinYv = rigidMinY[rigidIndex];
  let rigidMaxXv = rigidMaxX[rigidIndex];
  let rigidMaxYv = rigidMaxY[rigidIndex];

  let overlaps = edgeMaxX >= rigidMinXv && edgeMinX <= rigidMaxXv && edgeMaxY >= rigidMinYv && edgeMinY <= rigidMaxYv;
  let activeFlag = select(0u, 1u, overlaps);
  if (activeFlag == 1u) {
    let outOffset = atomicAdd(&activePackedOut[0], 1u) + 1u;
    if (outOffset <= params.edgePairCount) {
      atomicStore(&activePackedOut[outOffset], pairIndex);
    }
  }
}
`;

const rigidSoftNodeNarrowphaseAabbProbeWgsl = /* wgsl */
`
struct Params {
  pairCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pairRigidIndex: array<u32>;
@group(0) @binding(2) var<storage, read> pairNodeIndex: array<u32>;
@group(0) @binding(3) var<storage, read> nodeX: array<f32>;
@group(0) @binding(4) var<storage, read> nodeY: array<f32>;
@group(0) @binding(5) var<storage, read> nodeR: array<f32>;
@group(0) @binding(6) var<storage, read> rigidMinX: array<f32>;
@group(0) @binding(7) var<storage, read> rigidMinY: array<f32>;
@group(0) @binding(8) var<storage, read> rigidMaxX: array<f32>;
@group(0) @binding(9) var<storage, read> rigidMaxY: array<f32>;
@group(0) @binding(10) var<storage, read_write> pairOut: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairIndex = gid.x;
  if (pairIndex >= params.pairCount) { return; }

  let rigidIndex = pairRigidIndex[pairIndex];
  let nodeIndex = pairNodeIndex[pairIndex];
  let x = nodeX[nodeIndex];
  let y = nodeY[nodeIndex];
  let r = max(0.0, nodeR[nodeIndex]);

  let minX = rigidMinX[rigidIndex];
  let minY = rigidMinY[rigidIndex];
  let maxX = rigidMaxX[rigidIndex];
  let maxY = rigidMaxY[rigidIndex];

  let dxLo = max((minX - x) - r, 0.0);
  let dxHi = max((x - maxX) - r, 0.0);
  let dyLo = max((minY - y) - r, 0.0);
  let dyHi = max((y - maxY) - r, 0.0);
  let dx = max(dxLo, dxHi);
  let dy = max(dyLo, dyHi);
  let separation = sqrt(dx * dx + dy * dy);

  let inside = x >= (minX - r) && x <= (maxX + r) && y >= (minY - r) && y <= (maxY + r);
  pairOut[pairIndex] = vec2<f32>(separation, select(0.0, 1.0, inside));
}
`;

const rigidSoftEdgeNarrowphaseAabbProbeWgsl = /* wgsl */
`
struct Params {
  pairCount: u32,
  edgeSlop: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pairRigidIndex: array<u32>;
@group(0) @binding(2) var<storage, read> pairNodeAIndex: array<u32>;
@group(0) @binding(3) var<storage, read> pairNodeBIndex: array<u32>;
@group(0) @binding(4) var<storage, read> nodeX: array<f32>;
@group(0) @binding(5) var<storage, read> nodeY: array<f32>;
@group(0) @binding(6) var<storage, read> rigidMinX: array<f32>;
@group(0) @binding(7) var<storage, read> rigidMinY: array<f32>;
@group(0) @binding(8) var<storage, read> rigidMaxX: array<f32>;
@group(0) @binding(9) var<storage, read> rigidMaxY: array<f32>;
@group(0) @binding(10) var<storage, read_write> pairOut: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairIndex = gid.x;
  if (pairIndex >= params.pairCount) { return; }

  let rigidIndex = pairRigidIndex[pairIndex];
  let nodeAIndex = pairNodeAIndex[pairIndex];
  let nodeBIndex = pairNodeBIndex[pairIndex];

  let ax = nodeX[nodeAIndex];
  let ay = nodeY[nodeAIndex];
  let bx = nodeX[nodeBIndex];
  let by = nodeY[nodeBIndex];

  let edgeMinX = min(ax, bx) - params.edgeSlop;
  let edgeMinY = min(ay, by) - params.edgeSlop;
  let edgeMaxX = max(ax, bx) + params.edgeSlop;
  let edgeMaxY = max(ay, by) + params.edgeSlop;

  let minX = rigidMinX[rigidIndex];
  let minY = rigidMinY[rigidIndex];
  let maxX = rigidMaxX[rigidIndex];
  let maxY = rigidMaxY[rigidIndex];

  let dx = max(max(minX - edgeMaxX, 0.0), max(edgeMinX - maxX, 0.0));
  let dy = max(max(minY - edgeMaxY, 0.0), max(edgeMinY - maxY, 0.0));
  let separation = sqrt(dx * dx + dy * dy);
  let inside = edgeMaxX >= minX && edgeMinX <= maxX && edgeMaxY >= minY && edgeMinY <= maxY;

  pairOut[pairIndex] = vec2<f32>(separation, select(0.0, 1.0, inside));
}
`;


function getWgslOffloadUnavailableReason(offload) {
  if (!offload) return 'wgsl-offload-missing';
  if (!offload.device) return 'wgsl-device-missing';
  if (offload.enabled !== true) return 'wgsl-offload-disabled';
  if (typeof offload.device.createComputePipelineAsync !== 'function') {
    return 'wgsl-device-missing-createComputePipelineAsync';
  }
  if (typeof globalThis.GPUBufferUsage === 'undefined') return 'wgsl-buffer-usage-unavailable';
  return null;
}

function canUseWgslOffload(offload) {
  return getWgslOffloadUnavailableReason(offload) === null;
}

function safeUnmapBuffer(buffer) {
  if (!buffer || typeof buffer.unmap !== 'function') return;
  try {
    buffer.unmap();
  } catch {
    // ignore invalid-state unmap attempts while recovering readback lifecycle
  }
}

function scheduleSerializedDispatch(state, key, task) {
  if (!state || typeof task !== 'function') return task();
  const chainKey = String(key || 'pendingDispatchPromise');
  const pending = (state[chainKey] || Promise.resolve())
    .catch(() => {})
    .then(task);
  state[chainKey] = pending.finally(() => {
    if (state[chainKey] === pending) state[chainKey] = null;
  });
  return state[chainKey];
}

function ensureRigidSoftNodeBroadphaseBuffers(offload, nodeCount, rigidCount, pairCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;

  if (!state.rigidSoftNodeBroadphaseParams) {
    state.rigidSoftNodeBroadphaseParams = device.createBuffer({
      size: 16,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.rigidSoftNodeBroadphaseBindGroup = null;
  }

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.rigidSoftNodeBroadphaseNodeCapacity || 0) < requiredNodeCapacity) {
    const cap = Math.max(requiredNodeCapacity, state.rigidSoftNodeBroadphaseNodeCapacity ? state.rigidSoftNodeBroadphaseNodeCapacity * 2 : 256);
    const bytes = cap * 4;
    state.rigidSoftNodeBroadphaseNodeX?.destroy?.();
    state.rigidSoftNodeBroadphaseNodeY?.destroy?.();
    state.rigidSoftNodeBroadphaseNodeR?.destroy?.();
    state.rigidSoftNodeBroadphaseNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseNodeR = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseNodeCapacity = cap;
    state.rigidSoftNodeBroadphaseBindGroup = null;
  }

  const requiredRigidCapacity = Math.max(1, rigidCount);
  if ((state.rigidSoftNodeBroadphaseRigidCapacity || 0) < requiredRigidCapacity) {
    const cap = Math.max(requiredRigidCapacity, state.rigidSoftNodeBroadphaseRigidCapacity ? state.rigidSoftNodeBroadphaseRigidCapacity * 2 : 64);
    const bytes = cap * 4;
    state.rigidSoftNodeBroadphaseMinX?.destroy?.();
    state.rigidSoftNodeBroadphaseMinY?.destroy?.();
    state.rigidSoftNodeBroadphaseMaxX?.destroy?.();
    state.rigidSoftNodeBroadphaseMaxY?.destroy?.();
    state.rigidSoftNodeBroadphaseMinX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseMinY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseMaxX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseMaxY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseRigidCapacity = cap;
    state.rigidSoftNodeBroadphaseBindGroup = null;
  }

  const requiredPairCapacity = Math.max(1, pairCount);
  if ((state.rigidSoftNodeBroadphasePairCapacity || 0) < requiredPairCapacity) {
    const cap = Math.max(requiredPairCapacity, state.rigidSoftNodeBroadphasePairCapacity ? state.rigidSoftNodeBroadphasePairCapacity * 2 : 512);
    const pairBytes = cap * 4;
    const packedBytes = (cap + 1) * 4;
    state.rigidSoftNodeBroadphasePairRigidIndex?.destroy?.();
    state.rigidSoftNodeBroadphasePairNodeIndex?.destroy?.();
    state.rigidSoftNodeBroadphaseActiveMaskOut?.destroy?.();
    state.rigidSoftNodeBroadphaseActiveMaskReadback?.destroy?.();
    state.rigidSoftNodeBroadphasePairRigidIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
    state.rigidSoftNodeBroadphasePairNodeIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseActiveMaskOut = device.createBuffer({
      size: packedBytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.rigidSoftNodeBroadphaseActiveMaskReadback = device.createBuffer({
      size: packedBytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.rigidSoftNodeBroadphasePairCapacity = cap;
    state.rigidSoftNodeBroadphaseBindGroup = null;
  }

  return state;
}

function ensureRigidSoftEdgeBroadphaseBuffers(offload, nodeCount, rigidCount, pairCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;

  if (!state.rigidSoftEdgeBroadphaseParams) {
    state.rigidSoftEdgeBroadphaseParams = device.createBuffer({
      size: 16,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.rigidSoftEdgeBroadphaseBindGroup = null;
  }

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.rigidSoftEdgeBroadphaseNodeCapacity || 0) < requiredNodeCapacity) {
    const cap = Math.max(requiredNodeCapacity, state.rigidSoftEdgeBroadphaseNodeCapacity ? state.rigidSoftEdgeBroadphaseNodeCapacity * 2 : 256);
    const bytes = cap * 4;
    state.rigidSoftEdgeBroadphaseNodeX?.destroy?.();
    state.rigidSoftEdgeBroadphaseNodeY?.destroy?.();
    state.rigidSoftEdgeBroadphaseNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphaseNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphaseNodeCapacity = cap;
    state.rigidSoftEdgeBroadphaseBindGroup = null;
  }

  const requiredRigidCapacity = Math.max(1, rigidCount);
  if ((state.rigidSoftEdgeBroadphaseRigidCapacity || 0) < requiredRigidCapacity) {
    const cap = Math.max(requiredRigidCapacity, state.rigidSoftEdgeBroadphaseRigidCapacity ? state.rigidSoftEdgeBroadphaseRigidCapacity * 2 : 64);
    const bytes = cap * 4;
    state.rigidSoftEdgeBroadphaseMinX?.destroy?.();
    state.rigidSoftEdgeBroadphaseMinY?.destroy?.();
    state.rigidSoftEdgeBroadphaseMaxX?.destroy?.();
    state.rigidSoftEdgeBroadphaseMaxY?.destroy?.();
    state.rigidSoftEdgeBroadphaseMinX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphaseMinY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphaseMaxX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphaseMaxY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphaseRigidCapacity = cap;
    state.rigidSoftEdgeBroadphaseBindGroup = null;
  }

  const requiredPairCapacity = Math.max(1, pairCount);
  if ((state.rigidSoftEdgeBroadphasePairCapacity || 0) < requiredPairCapacity) {
    const cap = Math.max(requiredPairCapacity, state.rigidSoftEdgeBroadphasePairCapacity ? state.rigidSoftEdgeBroadphasePairCapacity * 2 : 512);
    const pairBytes = cap * 4;
    const packedBytes = (cap + 1) * 4;
    state.rigidSoftEdgeBroadphasePairRigidIndex?.destroy?.();
    state.rigidSoftEdgeBroadphasePairNodeAIndex?.destroy?.();
    state.rigidSoftEdgeBroadphasePairNodeBIndex?.destroy?.();
    state.rigidSoftEdgeBroadphaseActiveMaskOut?.destroy?.();
    state.rigidSoftEdgeBroadphaseActiveMaskReadback?.destroy?.();
    state.rigidSoftEdgeBroadphasePairRigidIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphasePairNodeAIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphasePairNodeBIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphaseActiveMaskOut = device.createBuffer({
      size: packedBytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.rigidSoftEdgeBroadphaseActiveMaskReadback = device.createBuffer({
      size: packedBytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.rigidSoftEdgeBroadphasePairCapacity = cap;
    state.rigidSoftEdgeBroadphaseBindGroup = null;
  }

  return state;
}


function floatToSignatureWord(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return (Math.round(n * 1024) >>> 0);
}

export function buildRigidSoftNarrowphaseGeometryLayout(rigidBodies) {
  const rigid = Array.isArray(rigidBodies) ? rigidBodies : [];
  const edgeStart = new Uint32Array(rigid.length + 1);
  const centroidX = new Float32Array(rigid.length);
  const centroidY = new Float32Array(rigid.length);
  const edgeAx = [];
  const edgeAy = [];
  const edgeBx = [];
  const edgeBy = [];
  const edgeNx = [];
  const edgeNy = [];

  let edgeCursor = 0;
  let signature = 0x811c9dc5;

  for (let rbi = 0; rbi < rigid.length; rbi++) {
    const verts = sanitizeFinitePolygonVerts(rigidVerticesWorld(rigid[rbi]));
    const safeVerts = verts.length >= 3 ? verts : sanitizeFinitePolygonVerts(rigidVerticesWorld({
      ...rigid[rbi],
      verticesLocal: null,
      sides: Math.max(5, Math.trunc(finiteOr(rigid[rbi]?.sides, 6)) || 6),
    }));

    edgeStart[rbi] = edgeCursor;
    if (safeVerts.length < 3) {
      centroidX[rbi] = finiteOr(rigid[rbi]?.x, 0);
      centroidY[rbi] = finiteOr(rigid[rbi]?.y, 0);
      signature = fnv1aMix(signature, rbi);
      continue;
    }

    const c = polygonCentroid(safeVerts);
    centroidX[rbi] = c.x;
    centroidY[rbi] = c.y;
    signature = fnv1aMix(signature, rbi);
    signature = fnv1aMix(signature, floatToSignatureWord(c.x));
    signature = fnv1aMix(signature, floatToSignatureWord(c.y));

    for (let vi = 0; vi < safeVerts.length; vi++) {
      const a = safeVerts[vi];
      const b = safeVerts[(vi + 1) % safeVerts.length];
      const out = edgeOutwardNormal(a.x, a.y, b.x, b.y, c.x, c.y);
      edgeAx.push(a.x);
      edgeAy.push(a.y);
      edgeBx.push(b.x);
      edgeBy.push(b.y);
      edgeNx.push(out.nx);
      edgeNy.push(out.ny);
      edgeCursor += 1;

      signature = fnv1aMix(signature, floatToSignatureWord(a.x));
      signature = fnv1aMix(signature, floatToSignatureWord(a.y));
      signature = fnv1aMix(signature, floatToSignatureWord(b.x));
      signature = fnv1aMix(signature, floatToSignatureWord(b.y));
      signature = fnv1aMix(signature, floatToSignatureWord(out.nx));
      signature = fnv1aMix(signature, floatToSignatureWord(out.ny));
    }
  }
  edgeStart[rigid.length] = edgeCursor;

  const layout = {
    edgeStart,
    centroidX,
    centroidY,
    edgeAx: Float32Array.from(edgeAx),
    edgeAy: Float32Array.from(edgeAy),
    edgeBx: Float32Array.from(edgeBx),
    edgeBy: Float32Array.from(edgeBy),
    edgeNx: Float32Array.from(edgeNx),
    edgeNy: Float32Array.from(edgeNy),
  };

  const byteLength = layout.edgeStart.byteLength
    + layout.centroidX.byteLength
    + layout.centroidY.byteLength
    + layout.edgeAx.byteLength
    + layout.edgeAy.byteLength
    + layout.edgeBx.byteLength
    + layout.edgeBy.byteLength
    + layout.edgeNx.byteLength
    + layout.edgeNy.byteLength;

  return {
    layout,
    rigidCount: rigid.length,
    edgeCount: edgeCursor,
    byteLength,
    signature: signature >>> 0,
  };
}

function computeRigidBodyAabbs(rigidBodies) {
  const rigid = Array.isArray(rigidBodies) ? rigidBodies : [];
  const minX = new Float32Array(rigid.length);
  const minY = new Float32Array(rigid.length);
  const maxX = new Float32Array(rigid.length);
  const maxY = new Float32Array(rigid.length);

  for (let i = 0; i < rigid.length; i++) {
    const verts = rigidVerticesWorld(rigid[i]);
    let loX = Infinity;
    let loY = Infinity;
    let hiX = -Infinity;
    let hiY = -Infinity;
    for (const v of verts) {
      if (!Number.isFinite(v.x) || !Number.isFinite(v.y)) continue;
      if (v.x < loX) loX = v.x;
      if (v.y < loY) loY = v.y;
      if (v.x > hiX) hiX = v.x;
      if (v.y > hiY) hiY = v.y;
    }
    if (!Number.isFinite(loX)) {
      const bx = Number(rigid[i]?.x) || 0;
      const by = Number(rigid[i]?.y) || 0;
      const r = Math.max(0, Number(rigid[i]?.r) || 0);
      loX = bx - r;
      loY = by - r;
      hiX = bx + r;
      hiY = by + r;
    }
    minX[i] = loX;
    minY[i] = loY;
    maxX[i] = hiX;
    maxY[i] = hiY;
  }

  return { minX, minY, maxX, maxY };
}

async function dispatchRigidSoftNodeBroadphaseWgsl({ rigidBodies, soft, offload, layout, rigidAabb = null }) {
  if (!canUseWgslOffload(offload)) return null;
  const pairCount = Number(layout?.nodePairCount) || 0;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const rigidCount = Array.isArray(rigidBodies) ? rigidBodies.length : 0;
  if (pairCount <= 0 || nodes.length <= 0 || rigidCount <= 0) return null;

  const state = ensureRigidSoftNodeBroadphaseBuffers(offload, nodes.length, rigidCount, pairCount);
  const device = offload.device;

  if (!state.rigidSoftNodeBroadphasePipelinePromise) {
    const module = device.createShaderModule({ code: rigidSoftNodeBroadphaseWgsl });
    state.rigidSoftNodeBroadphasePipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    }).then((pipeline) => {
      state.rigidSoftNodeBroadphasePipeline = pipeline;
      state.rigidSoftNodeBroadphaseBindGroup = null;
      return pipeline;
    });
  }

  let pipeline = state.rigidSoftNodeBroadphasePipeline;
  if (!pipeline && state.rigidSoftNodeBroadphasePipelinePromise) {
    try {
      pipeline = await state.rigidSoftNodeBroadphasePipelinePromise;
      state.rigidSoftNodeBroadphasePipeline = pipeline || null;
    } catch (err) {
      const reason = `pipeline-error:${String(err?.message || err || 'unknown')}`;
      state.lastNodeBroadphaseDispatched = false;
      state.lastNodeBroadphaseReason = reason;
      state.lastError = reason;
      state.rigidSoftNodeBroadphasePipelinePromise = null;
      return null;
    }
  }
  if (!pipeline) {
    state.lastNodeBroadphaseDispatched = false;
    state.lastNodeBroadphaseReason = 'pipeline-pending';
    return null;
  }

  if (!state.rigidSoftNodeBroadphaseBindGroup) {
    state.rigidSoftNodeBroadphaseBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.rigidSoftNodeBroadphaseParams } },
        { binding: 1, resource: { buffer: state.rigidSoftNodeBroadphasePairRigidIndex } },
        { binding: 2, resource: { buffer: state.rigidSoftNodeBroadphasePairNodeIndex } },
        { binding: 3, resource: { buffer: state.rigidSoftNodeBroadphaseNodeX } },
        { binding: 4, resource: { buffer: state.rigidSoftNodeBroadphaseNodeY } },
        { binding: 5, resource: { buffer: state.rigidSoftNodeBroadphaseNodeR } },
        { binding: 6, resource: { buffer: state.rigidSoftNodeBroadphaseMinX } },
        { binding: 7, resource: { buffer: state.rigidSoftNodeBroadphaseMinY } },
        { binding: 8, resource: { buffer: state.rigidSoftNodeBroadphaseMaxX } },
        { binding: 9, resource: { buffer: state.rigidSoftNodeBroadphaseMaxY } },
        { binding: 10, resource: { buffer: state.rigidSoftNodeBroadphaseActiveMaskOut } },
      ],
    });
  }

  return scheduleSerializedDispatch(state, 'pendingRigidSoftNodeBroadphaseDispatchPromise', async () => {
    const timing = {
      uploadMs: 0,
      dispatchSubmitMs: 0,
      countReadbackMs: 0,
      indexReadbackMs: 0,
      compactBuildMs: 0,
      totalMs: 0,
    };
    const totalStartMs = performance.now();
    state.lastNodeBroadphaseTiming = timing;

    const nodeX = new Float32Array(nodes.length);
    const nodeY = new Float32Array(nodes.length);
    const nodeR = new Float32Array(nodes.length);
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i] || {};
      nodeX[i] = Number(n.x) || 0;
      nodeY[i] = Number(n.y) || 0;
      nodeR[i] = Math.max(0, Number(n.r) || 1);
    }
    const computedRigidAabb = rigidAabb || computeRigidBodyAabbs(rigidBodies);

    const paramsBytes = new Uint32Array([pairCount >>> 0, 0, 0, 0]);
    const uploadStartMs = performance.now();
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseParams, 0, paramsBytes);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphasePairRigidIndex, 0, layout.layout.nodePairRigidIndex);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphasePairNodeIndex, 0, layout.layout.nodePairNodeIndex);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseNodeX, 0, nodeX);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseNodeY, 0, nodeY);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseNodeR, 0, nodeR);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseMinX, 0, computedRigidAabb.minX);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseMinY, 0, computedRigidAabb.minY);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseMaxX, 0, computedRigidAabb.maxX);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseMaxY, 0, computedRigidAabb.maxY);
    device.queue.writeBuffer(state.rigidSoftNodeBroadphaseActiveMaskOut, 0, new Uint32Array([0]));
    timing.uploadMs += Math.max(0, performance.now() - uploadStartMs);

    const dispatchCount = Math.ceil(pairCount / WGSL_WORKGROUP_SIZE);
    const packedReadbackBytes = (pairCount + 1) * 4;
    const dispatchSubmitStartMs = performance.now();
    const countEncoder = device.createCommandEncoder();
    const pass = countEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, state.rigidSoftNodeBroadphaseBindGroup);
    pass.dispatchWorkgroups(dispatchCount);
    pass.end();
    countEncoder.copyBufferToBuffer(
      state.rigidSoftNodeBroadphaseActiveMaskOut,
      0,
      state.rigidSoftNodeBroadphaseActiveMaskReadback,
      0,
      packedReadbackBytes,
    );
    device.queue.submit([countEncoder.finish()]);
    timing.dispatchSubmitMs += Math.max(0, performance.now() - dispatchSubmitStartMs);

    state.lastNodeBroadphaseDispatched = true;
    state.lastNodeBroadphaseReason = null;
    state.lastNodeBroadphaseDispatch = dispatchCount;
    state.lastNodeBroadphasePairCount = pairCount;
    state.lastNodeBroadphaseCandidatePairCount = pairCount;
    state.lastNodeBroadphaseSourceRoute = 'wgsl-rigid-soft-node-broadphase-proposal';

    safeUnmapBuffer(state.rigidSoftNodeBroadphaseActiveMaskReadback);

    try {
      const readbackStartMs = performance.now();
      await state.rigidSoftNodeBroadphaseActiveMaskReadback.mapAsync(globalThis.GPUMapMode.READ, 0, packedReadbackBytes);
      const mappedPacked = state.rigidSoftNodeBroadphaseActiveMaskReadback.getMappedRange(0, packedReadbackBytes);
      const packedU32 = new Uint32Array(mappedPacked.slice(0));
      timing.countReadbackMs += Math.max(0, performance.now() - readbackStartMs);

      const activeCountRaw = packedU32[0] >>> 0;
      const activeCount = Math.min(pairCount >>> 0, activeCountRaw >>> 0);
      state.lastNodeBroadphaseActivePairCount = activeCount;
      state.lastNodeBroadphaseReadbackCount = activeCount;
      state.lastNodeBroadphaseReadbackSourceRoute = 'wgsl-rigid-soft-node-broadphase-readback';

      if (activeCount === 0) {
        timing.totalMs = Math.max(0, performance.now() - totalStartMs);
        return {
          candidatePairCount: pairCount,
          activePairCount: 0,
          compactPairs: {
            compactRigidIndex: new Uint32Array(0),
            compactNodeIndex: new Uint32Array(0),
            pairCount: 0,
            byteLength: 0,
            signature: 0x811c9dc5,
          },
        };
      }

      const indexExtractStartMs = performance.now();
      const activePairIndices = packedU32.slice(1, 1 + activeCount);
      timing.indexReadbackMs += Math.max(0, performance.now() - indexExtractStartMs);

      const compactStartMs = performance.now();
      const compactPairs = buildCompactRigidSoftNodePairsFromActiveIndices({
        prep: layout,
        activePairIndices,
      });
      timing.compactBuildMs += Math.max(0, performance.now() - compactStartMs);
      if (!compactPairs) {
        timing.totalMs = Math.max(0, performance.now() - totalStartMs);
        return null;
      }
      timing.totalMs = Math.max(0, performance.now() - totalStartMs);
      return {
        candidatePairCount: pairCount,
        activePairCount: compactPairs.pairCount | 0,
        compactPairs,
      };
    } catch (err) {
      state.lastNodeBroadphaseReason = String(err?.message || err || 'readback-failed');
      state.lastNodeBroadphaseReadbackSourceRoute = 'cpu-rigid-soft-node-broadphase-readback-fallback';
      timing.totalMs = Math.max(0, performance.now() - totalStartMs);
      return null;
    } finally {
      safeUnmapBuffer(state.rigidSoftNodeBroadphaseActiveMaskReadback);
    }
  });
}

async function dispatchRigidSoftEdgeBroadphaseWgsl({ rigidBodies, soft, offload, layout, edgeSlop = 0, rigidAabb = null }) {
  if (!canUseWgslOffload(offload)) return null;
  const pairCount = Number(layout?.edgePairCount) || 0;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const rigidCount = Array.isArray(rigidBodies) ? rigidBodies.length : 0;
  if (pairCount <= 0 || nodes.length <= 0 || rigidCount <= 0) return null;

  const state = ensureRigidSoftEdgeBroadphaseBuffers(offload, nodes.length, rigidCount, pairCount);
  const device = offload.device;

  if (!state.rigidSoftEdgeBroadphasePipelinePromise) {
    const module = device.createShaderModule({ code: rigidSoftEdgeBroadphaseWgsl });
    state.rigidSoftEdgeBroadphasePipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    }).then((pipeline) => {
      state.rigidSoftEdgeBroadphasePipeline = pipeline;
      state.rigidSoftEdgeBroadphaseBindGroup = null;
      return pipeline;
    });
  }

  let pipeline = state.rigidSoftEdgeBroadphasePipeline;
  if (!pipeline && state.rigidSoftEdgeBroadphasePipelinePromise) {
    try {
      pipeline = await state.rigidSoftEdgeBroadphasePipelinePromise;
      state.rigidSoftEdgeBroadphasePipeline = pipeline || null;
    } catch (err) {
      const reason = `pipeline-error:${String(err?.message || err || 'unknown')}`;
      state.lastEdgeBroadphaseDispatched = false;
      state.lastEdgeBroadphaseReason = reason;
      state.lastError = reason;
      state.rigidSoftEdgeBroadphasePipelinePromise = null;
      return null;
    }
  }
  if (!pipeline) {
    state.lastEdgeBroadphaseDispatched = false;
    state.lastEdgeBroadphaseReason = 'pipeline-pending';
    return null;
  }

  if (!state.rigidSoftEdgeBroadphaseBindGroup) {
    state.rigidSoftEdgeBroadphaseBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.rigidSoftEdgeBroadphaseParams } },
        { binding: 1, resource: { buffer: state.rigidSoftEdgeBroadphasePairRigidIndex } },
        { binding: 2, resource: { buffer: state.rigidSoftEdgeBroadphasePairNodeAIndex } },
        { binding: 3, resource: { buffer: state.rigidSoftEdgeBroadphasePairNodeBIndex } },
        { binding: 4, resource: { buffer: state.rigidSoftEdgeBroadphaseNodeX } },
        { binding: 5, resource: { buffer: state.rigidSoftEdgeBroadphaseNodeY } },
        { binding: 6, resource: { buffer: state.rigidSoftEdgeBroadphaseMinX } },
        { binding: 7, resource: { buffer: state.rigidSoftEdgeBroadphaseMinY } },
        { binding: 8, resource: { buffer: state.rigidSoftEdgeBroadphaseMaxX } },
        { binding: 9, resource: { buffer: state.rigidSoftEdgeBroadphaseMaxY } },
        { binding: 10, resource: { buffer: state.rigidSoftEdgeBroadphaseActiveMaskOut } },
      ],
    });
  }

  return scheduleSerializedDispatch(state, 'pendingRigidSoftEdgeBroadphaseDispatchPromise', async () => {
    const timing = {
      uploadMs: 0,
      dispatchSubmitMs: 0,
      countReadbackMs: 0,
      indexReadbackMs: 0,
      compactBuildMs: 0,
      totalMs: 0,
    };
    const totalStartMs = performance.now();
    state.lastEdgeBroadphaseTiming = timing;

    const nodeX = new Float32Array(nodes.length);
    const nodeY = new Float32Array(nodes.length);
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i] || {};
      nodeX[i] = Number(n.x) || 0;
      nodeY[i] = Number(n.y) || 0;
    }
    const computedRigidAabb = rigidAabb || computeRigidBodyAabbs(rigidBodies);
    const safeEdgeSlop = Number.isFinite(Number(edgeSlop)) ? Math.max(0, Number(edgeSlop)) : 0;

    const params = new Float32Array(4);
    new Uint32Array(params.buffer)[0] = pairCount >>> 0;
    params[3] = safeEdgeSlop;
    const uploadStartMs = performance.now();
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphaseParams, 0, params);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphasePairRigidIndex, 0, layout.layout.edgePairRigidIndex);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphasePairNodeAIndex, 0, layout.layout.edgePairNodeAIndex);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphasePairNodeBIndex, 0, layout.layout.edgePairNodeBIndex);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphaseNodeX, 0, nodeX);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphaseNodeY, 0, nodeY);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphaseMinX, 0, computedRigidAabb.minX);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphaseMinY, 0, computedRigidAabb.minY);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphaseMaxX, 0, computedRigidAabb.maxX);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphaseMaxY, 0, computedRigidAabb.maxY);
    device.queue.writeBuffer(state.rigidSoftEdgeBroadphaseActiveMaskOut, 0, new Uint32Array([0]));
    timing.uploadMs += Math.max(0, performance.now() - uploadStartMs);

    const dispatchCount = Math.ceil(pairCount / WGSL_WORKGROUP_SIZE);
    const packedReadbackBytes = (pairCount + 1) * 4;
    const dispatchSubmitStartMs = performance.now();
    const countEncoder = device.createCommandEncoder();
    const pass = countEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, state.rigidSoftEdgeBroadphaseBindGroup);
    pass.dispatchWorkgroups(dispatchCount);
    pass.end();
    countEncoder.copyBufferToBuffer(
      state.rigidSoftEdgeBroadphaseActiveMaskOut,
      0,
      state.rigidSoftEdgeBroadphaseActiveMaskReadback,
      0,
      packedReadbackBytes,
    );
    device.queue.submit([countEncoder.finish()]);
    timing.dispatchSubmitMs += Math.max(0, performance.now() - dispatchSubmitStartMs);

    state.lastEdgeBroadphaseDispatched = true;
    state.lastEdgeBroadphaseReason = null;
    state.lastEdgeBroadphaseDispatch = dispatchCount;
    state.lastEdgeBroadphasePairCount = pairCount;
    state.lastEdgeBroadphaseCandidatePairCount = pairCount;
    state.lastEdgeBroadphaseSourceRoute = 'wgsl-rigid-soft-edge-broadphase-proposal';

    safeUnmapBuffer(state.rigidSoftEdgeBroadphaseActiveMaskReadback);

    try {
      const readbackStartMs = performance.now();
      await state.rigidSoftEdgeBroadphaseActiveMaskReadback.mapAsync(globalThis.GPUMapMode.READ, 0, packedReadbackBytes);
      const mappedPacked = state.rigidSoftEdgeBroadphaseActiveMaskReadback.getMappedRange(0, packedReadbackBytes);
      const packedU32 = new Uint32Array(mappedPacked.slice(0));
      timing.countReadbackMs += Math.max(0, performance.now() - readbackStartMs);

      const activeCountRaw = packedU32[0] >>> 0;
      const activeCount = Math.min(pairCount >>> 0, activeCountRaw >>> 0);
      state.lastEdgeBroadphaseActivePairCount = activeCount;
      state.lastEdgeBroadphaseReadbackCount = activeCount;
      state.lastEdgeBroadphaseReadbackSourceRoute = 'wgsl-rigid-soft-edge-broadphase-readback';

      if (activeCount === 0) {
        timing.totalMs = Math.max(0, performance.now() - totalStartMs);
        return {
          candidatePairCount: pairCount,
          activePairCount: 0,
          compactPairs: {
            compactRigidIndex: new Uint32Array(0),
            compactSpringIndex: new Uint32Array(0),
            compactNodeAIndex: new Uint32Array(0),
            compactNodeBIndex: new Uint32Array(0),
            pairCount: 0,
            byteLength: 0,
            signature: 0x811c9dc5,
          },
        };
      }

      const indexExtractStartMs = performance.now();
      const activePairIndices = packedU32.slice(1, 1 + activeCount);
      timing.indexReadbackMs += Math.max(0, performance.now() - indexExtractStartMs);

      const compactStartMs = performance.now();
      const compactPairs = buildCompactRigidSoftEdgePairsFromActiveIndices({
        prep: layout,
        activePairIndices,
      });
      timing.compactBuildMs += Math.max(0, performance.now() - compactStartMs);
      if (!compactPairs) {
        timing.totalMs = Math.max(0, performance.now() - totalStartMs);
        return null;
      }
      timing.totalMs = Math.max(0, performance.now() - totalStartMs);
      return {
        candidatePairCount: pairCount,
        activePairCount: compactPairs.pairCount | 0,
        compactPairs,
      };
    } catch (err) {
      state.lastEdgeBroadphaseReason = String(err?.message || err || 'readback-failed');
      state.lastEdgeBroadphaseReadbackSourceRoute = 'cpu-rigid-soft-edge-broadphase-readback-fallback';
      timing.totalMs = Math.max(0, performance.now() - totalStartMs);
      return null;
    } finally {
      safeUnmapBuffer(state.rigidSoftEdgeBroadphaseActiveMaskReadback);
    }
  });
}


async function dispatchRigidSoftNodeNarrowphaseAabbProbeWgsl({ rigidBodies, soft, offload, nodePairs, rigidAabb = null }) {
  if (!canUseWgslOffload(offload)) return false;
  const pairCount = Number(nodePairs?.pairCount) || 0;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const rigidCount = Array.isArray(rigidBodies) ? rigidBodies.length : 0;
  if (pairCount <= 0 || nodes.length <= 0 || rigidCount <= 0) return false;
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const timing = {
    inputBuildMs: 0,
    uploadMs: 0,
    dispatchSubmitMs: 0,
    readbackMapMs: 0,
    decodeMs: 0,
    totalMs: 0,
  };
  const totalStartMs = performance.now();
  state.lastNodeNarrowphaseAabbProbeTiming = timing;

  if (!state.rigidSoftNodeNarrowphaseAabbProbePipelinePromise) {
    const module = device.createShaderModule({ code: rigidSoftNodeNarrowphaseAabbProbeWgsl });
    state.rigidSoftNodeNarrowphaseAabbProbePipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    }).then((pipeline) => {
      state.rigidSoftNodeNarrowphaseAabbProbePipeline = pipeline;
      return pipeline;
    });
  }
  let pipeline = state.rigidSoftNodeNarrowphaseAabbProbePipeline;
  if (!pipeline && state.rigidSoftNodeNarrowphaseAabbProbePipelinePromise) {
    try {
      pipeline = await state.rigidSoftNodeNarrowphaseAabbProbePipelinePromise;
      state.rigidSoftNodeNarrowphaseAabbProbePipeline = pipeline || null;
    } catch (err) {
      state.lastNodeNarrowphaseAabbProbeSource = `cpu-rigid-soft-node-aabb-probe-pipeline-error:${String(err?.message || err || 'unknown')}`;
      state.lastError = state.lastNodeNarrowphaseAabbProbeSource;
      state.rigidSoftNodeNarrowphaseAabbProbePipelinePromise = null;
      timing.totalMs = Math.max(0, performance.now() - totalStartMs);
      return false;
    }
  }
  if (!pipeline) {
    timing.totalMs = Math.max(0, performance.now() - totalStartMs);
    return false;
  }

  const inputBuildStartMs = performance.now();
  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  const nodeR = new Float32Array(nodes.length);
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i] || {};
    nodeX[i] = Number(n.x) || 0;
    nodeY[i] = Number(n.y) || 0;
    nodeR[i] = Math.max(0, Number(n.r) || 0);
  }
  const computedRigidAabb = rigidAabb || computeRigidBodyAabbs(rigidBodies);
  timing.inputBuildMs += Math.max(0, performance.now() - inputBuildStartMs);
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const pairBytes = pairCount * 4;
  const pairVec2Bytes = pairCount * 8;

  const params = device.createBuffer({ size: 16, usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST });
  const pairRigidIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
  const pairNodeIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
  const nodeXBuf = device.createBuffer({ size: nodes.length * 4, usage: storageUsage });
  const nodeYBuf = device.createBuffer({ size: nodes.length * 4, usage: storageUsage });
  const nodeRBuf = device.createBuffer({ size: nodes.length * 4, usage: storageUsage });
  const minXBuf = device.createBuffer({ size: rigidCount * 4, usage: storageUsage });
  const minYBuf = device.createBuffer({ size: rigidCount * 4, usage: storageUsage });
  const maxXBuf = device.createBuffer({ size: rigidCount * 4, usage: storageUsage });
  const maxYBuf = device.createBuffer({ size: rigidCount * 4, usage: storageUsage });
  const outBuf = device.createBuffer({ size: pairVec2Bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: pairVec2Bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: pairRigidIndex } },
      { binding: 2, resource: { buffer: pairNodeIndex } },
      { binding: 3, resource: { buffer: nodeXBuf } },
      { binding: 4, resource: { buffer: nodeYBuf } },
      { binding: 5, resource: { buffer: nodeRBuf } },
      { binding: 6, resource: { buffer: minXBuf } },
      { binding: 7, resource: { buffer: minYBuf } },
      { binding: 8, resource: { buffer: maxXBuf } },
      { binding: 9, resource: { buffer: maxYBuf } },
      { binding: 10, resource: { buffer: outBuf } },
    ],
  });

  const paramBytes = new ArrayBuffer(16);
  new Uint32Array(paramBytes)[0] = pairCount >>> 0;
  const uploadStartMs = performance.now();
  device.queue.writeBuffer(params, 0, paramBytes);
  device.queue.writeBuffer(pairRigidIndex, 0, nodePairs.compactRigidIndex);
  device.queue.writeBuffer(pairNodeIndex, 0, nodePairs.compactNodeIndex);
  device.queue.writeBuffer(nodeXBuf, 0, nodeX);
  device.queue.writeBuffer(nodeYBuf, 0, nodeY);
  device.queue.writeBuffer(nodeRBuf, 0, nodeR);
  device.queue.writeBuffer(minXBuf, 0, computedRigidAabb.minX);
  device.queue.writeBuffer(minYBuf, 0, computedRigidAabb.minY);
  device.queue.writeBuffer(maxXBuf, 0, computedRigidAabb.maxX);
  device.queue.writeBuffer(maxYBuf, 0, computedRigidAabb.maxY);
  timing.uploadMs += Math.max(0, performance.now() - uploadStartMs);

  const dispatchSubmitStartMs = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(pairCount / WGSL_WORKGROUP_SIZE));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readback, 0, pairVec2Bytes);
  device.queue.submit([encoder.finish()]);
  timing.dispatchSubmitMs += Math.max(0, performance.now() - dispatchSubmitStartMs);

  const readbackStartMs = performance.now();
  await readback.mapAsync(globalThis.GPUMapMode.READ, 0, pairVec2Bytes);
  const packed = new Float32Array(readback.getMappedRange(0, pairVec2Bytes).slice(0));
  timing.readbackMapMs += Math.max(0, performance.now() - readbackStartMs);
  readback.unmap();

  const decodeStartMs = performance.now();
  const separation = new Float32Array(pairCount);
  const insideMask = new Uint32Array(pairCount);
  let finite = true;
  for (let i = 0; i < pairCount; i++) {
    const sep = Number(packed[i * 2]) || 0;
    const inside = Number(packed[i * 2 + 1]) || 0;
    if (!Number.isFinite(sep) || !Number.isFinite(inside)) finite = false;
    separation[i] = Number.isFinite(sep) ? sep : 0;
    insideMask[i] = inside >= 0.5 ? 1 : 0;
  }

  state.lastNodeNarrowphaseAabbProbePairCount = pairCount;
  state.lastNodeNarrowphaseAabbProbeSeparation = separation;
  state.lastNodeNarrowphaseAabbProbeInsideMask = insideMask;
  state.lastNodeNarrowphaseAabbProbeFinite = finite;
  state.lastNodeNarrowphaseAabbProbeSource = finite
    ? 'wgsl-rigid-soft-node-aabb-probe'
    : 'cpu-rigid-soft-node-aabb-probe-fallback-nonfinite';
  timing.decodeMs += Math.max(0, performance.now() - decodeStartMs);
  timing.totalMs = Math.max(0, performance.now() - totalStartMs);

  [params, pairRigidIndex, pairNodeIndex, nodeXBuf, nodeYBuf, nodeRBuf, minXBuf, minYBuf, maxXBuf, maxYBuf, outBuf, readback].forEach((b) => b?.destroy?.());
  return finite;
}

async function dispatchRigidSoftEdgeNarrowphaseAabbProbeWgsl({ rigidBodies, soft, offload, edgePairs, edgeSlop = 0, rigidAabb = null }) {
  if (!canUseWgslOffload(offload)) return false;
  const pairCount = Number(edgePairs?.pairCount) || 0;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const rigidCount = Array.isArray(rigidBodies) ? rigidBodies.length : 0;
  if (pairCount <= 0 || nodes.length <= 0 || rigidCount <= 0) return false;
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const timing = {
    inputBuildMs: 0,
    uploadMs: 0,
    dispatchSubmitMs: 0,
    readbackMapMs: 0,
    decodeMs: 0,
    totalMs: 0,
  };
  const totalStartMs = performance.now();
  state.lastEdgeNarrowphaseAabbProbeTiming = timing;

  if (!state.rigidSoftEdgeNarrowphaseAabbProbePipelinePromise) {
    const module = device.createShaderModule({ code: rigidSoftEdgeNarrowphaseAabbProbeWgsl });
    state.rigidSoftEdgeNarrowphaseAabbProbePipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    }).then((pipeline) => {
      state.rigidSoftEdgeNarrowphaseAabbProbePipeline = pipeline;
      return pipeline;
    });
  }
  let pipeline = state.rigidSoftEdgeNarrowphaseAabbProbePipeline;
  if (!pipeline && state.rigidSoftEdgeNarrowphaseAabbProbePipelinePromise) {
    try {
      pipeline = await state.rigidSoftEdgeNarrowphaseAabbProbePipelinePromise;
      state.rigidSoftEdgeNarrowphaseAabbProbePipeline = pipeline || null;
    } catch (err) {
      state.lastEdgeNarrowphaseAabbProbeSource = `cpu-rigid-soft-edge-aabb-probe-pipeline-error:${String(err?.message || err || 'unknown')}`;
      state.lastError = state.lastEdgeNarrowphaseAabbProbeSource;
      state.rigidSoftEdgeNarrowphaseAabbProbePipelinePromise = null;
      timing.totalMs = Math.max(0, performance.now() - totalStartMs);
      return false;
    }
  }
  if (!pipeline) {
    timing.totalMs = Math.max(0, performance.now() - totalStartMs);
    return false;
  }

  const inputBuildStartMs = performance.now();
  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i] || {};
    nodeX[i] = Number(n.x) || 0;
    nodeY[i] = Number(n.y) || 0;
  }

  const computedRigidAabb = rigidAabb || computeRigidBodyAabbs(rigidBodies);
  timing.inputBuildMs += Math.max(0, performance.now() - inputBuildStartMs);
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const pairBytes = pairCount * 4;
  const pairVec2Bytes = pairCount * 8;

  const params = device.createBuffer({ size: 16, usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST });
  const pairRigidIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
  const pairNodeAIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
  const pairNodeBIndex = device.createBuffer({ size: pairBytes, usage: storageUsage });
  const nodeXBuf = device.createBuffer({ size: nodes.length * 4, usage: storageUsage });
  const nodeYBuf = device.createBuffer({ size: nodes.length * 4, usage: storageUsage });
  const minXBuf = device.createBuffer({ size: rigidCount * 4, usage: storageUsage });
  const minYBuf = device.createBuffer({ size: rigidCount * 4, usage: storageUsage });
  const maxXBuf = device.createBuffer({ size: rigidCount * 4, usage: storageUsage });
  const maxYBuf = device.createBuffer({ size: rigidCount * 4, usage: storageUsage });
  const outBuf = device.createBuffer({ size: pairVec2Bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: pairVec2Bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: pairRigidIndex } },
      { binding: 2, resource: { buffer: pairNodeAIndex } },
      { binding: 3, resource: { buffer: pairNodeBIndex } },
      { binding: 4, resource: { buffer: nodeXBuf } },
      { binding: 5, resource: { buffer: nodeYBuf } },
      { binding: 6, resource: { buffer: minXBuf } },
      { binding: 7, resource: { buffer: minYBuf } },
      { binding: 8, resource: { buffer: maxXBuf } },
      { binding: 9, resource: { buffer: maxYBuf } },
      { binding: 10, resource: { buffer: outBuf } },
    ],
  });

  const paramFloats = new Float32Array(4);
  new Uint32Array(paramFloats.buffer)[0] = pairCount >>> 0;
  paramFloats[1] = Number.isFinite(Number(edgeSlop)) ? Math.max(0, Number(edgeSlop)) : 0;
  const uploadStartMs = performance.now();
  device.queue.writeBuffer(params, 0, paramFloats);
  device.queue.writeBuffer(pairRigidIndex, 0, edgePairs.compactRigidIndex);
  device.queue.writeBuffer(pairNodeAIndex, 0, edgePairs.compactNodeAIndex);
  device.queue.writeBuffer(pairNodeBIndex, 0, edgePairs.compactNodeBIndex);
  device.queue.writeBuffer(nodeXBuf, 0, nodeX);
  device.queue.writeBuffer(nodeYBuf, 0, nodeY);
  device.queue.writeBuffer(minXBuf, 0, computedRigidAabb.minX);
  device.queue.writeBuffer(minYBuf, 0, computedRigidAabb.minY);
  device.queue.writeBuffer(maxXBuf, 0, computedRigidAabb.maxX);
  device.queue.writeBuffer(maxYBuf, 0, computedRigidAabb.maxY);
  timing.uploadMs += Math.max(0, performance.now() - uploadStartMs);

  const dispatchSubmitStartMs = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(pairCount / WGSL_WORKGROUP_SIZE));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readback, 0, pairVec2Bytes);
  device.queue.submit([encoder.finish()]);
  timing.dispatchSubmitMs += Math.max(0, performance.now() - dispatchSubmitStartMs);

  const readbackStartMs = performance.now();
  await readback.mapAsync(globalThis.GPUMapMode.READ, 0, pairVec2Bytes);
  const packed = new Float32Array(readback.getMappedRange(0, pairVec2Bytes).slice(0));
  timing.readbackMapMs += Math.max(0, performance.now() - readbackStartMs);
  readback.unmap();

  const decodeStartMs = performance.now();
  const separation = new Float32Array(pairCount);
  const insideMask = new Uint32Array(pairCount);
  let finite = true;
  for (let i = 0; i < pairCount; i++) {
    const sep = Number(packed[i * 2]) || 0;
    const inside = Number(packed[i * 2 + 1]) || 0;
    if (!Number.isFinite(sep) || !Number.isFinite(inside)) finite = false;
    separation[i] = Number.isFinite(sep) ? sep : 0;
    insideMask[i] = inside >= 0.5 ? 1 : 0;
  }

  state.lastEdgeNarrowphaseAabbProbePairCount = pairCount;
  state.lastEdgeNarrowphaseAabbProbeSeparation = separation;
  state.lastEdgeNarrowphaseAabbProbeInsideMask = insideMask;
  state.lastEdgeNarrowphaseAabbProbeFinite = finite;
  state.lastEdgeNarrowphaseAabbProbeSource = finite
    ? 'wgsl-rigid-soft-edge-aabb-probe'
    : 'cpu-rigid-soft-edge-aabb-probe-fallback-nonfinite';
  timing.decodeMs += Math.max(0, performance.now() - decodeStartMs);
  timing.totalMs = Math.max(0, performance.now() - totalStartMs);

  [params, pairRigidIndex, pairNodeAIndex, pairNodeBIndex, nodeXBuf, nodeYBuf, minXBuf, minYBuf, maxXBuf, maxYBuf, outBuf, readback].forEach((b) => b?.destroy?.());
  return finite;
}

export function buildAuthoritativeRigidSoftNodePairsFromAabbProbe({
  nodePairs,
  separation,
  insideMask,
  nodeSlop = 0,
}) {
  const rigidIndex = nodePairs?.compactRigidIndex;
  const nodeIndex = nodePairs?.compactNodeIndex;
  const pairCount = Number(nodePairs?.pairCount) || 0;
  if (!(rigidIndex instanceof Uint32Array) || !(nodeIndex instanceof Uint32Array)) return null;
  if (!(separation instanceof Float32Array) || !(insideMask instanceof Uint32Array)) return null;
  if (separation.length !== pairCount || insideMask.length !== pairCount) return null;

  const activeRigid = [];
  const activeNode = [];
  const slop = Number.isFinite(Number(nodeSlop)) ? Math.max(0, Number(nodeSlop)) : 0;
  let signature = 0x811c9dc5;

  for (let i = 0; i < pairCount; i++) {
    const sep = Number(separation[i]);
    const inside = (insideMask[i] >>> 0) === 1;
    if (!Number.isFinite(sep)) return null;
    if (!inside && sep > slop) continue;

    const rbi = rigidIndex[i] >>> 0;
    const ni = nodeIndex[i] >>> 0;
    activeRigid.push(rbi);
    activeNode.push(ni);
    signature = fnv1aMix(signature, rbi);
    signature = fnv1aMix(signature, ni);
  }

  const compactRigidIndex = Uint32Array.from(activeRigid);
  const compactNodeIndex = Uint32Array.from(activeNode);
  return {
    compactRigidIndex,
    compactNodeIndex,
    pairCount: compactRigidIndex.length,
    byteLength: compactRigidIndex.byteLength + compactNodeIndex.byteLength,
    signature: signature >>> 0,
  };
}

export function buildRigidSoftNarrowphaseImpulseSeedLayout({
  narrowphaseLayout,
  narrowphaseSceneLayout,
}) {
  const nodePairRigidIndex = narrowphaseLayout?.nodePairRigidIndex;
  const nodePairNodeIndex = narrowphaseLayout?.nodePairNodeIndex;
  const edgePairRigidIndex = narrowphaseLayout?.edgePairRigidIndex;
  const edgePairSpringIndex = narrowphaseLayout?.edgePairSpringIndex;
  const edgePairNodeAIndex = narrowphaseLayout?.edgePairNodeAIndex;
  const edgePairNodeBIndex = narrowphaseLayout?.edgePairNodeBIndex;

  const rigidX = narrowphaseSceneLayout?.rigidX;
  const rigidY = narrowphaseSceneLayout?.rigidY;
  const rigidMinX = narrowphaseSceneLayout?.rigidMinX;
  const rigidMinY = narrowphaseSceneLayout?.rigidMinY;
  const rigidMaxX = narrowphaseSceneLayout?.rigidMaxX;
  const rigidMaxY = narrowphaseSceneLayout?.rigidMaxY;
  const rigidInvMass = narrowphaseSceneLayout?.rigidInvMass;
  const rigidInvInertia = narrowphaseSceneLayout?.rigidInvInertia;
  const nodeX = narrowphaseSceneLayout?.nodeX;
  const nodeY = narrowphaseSceneLayout?.nodeY;
  const nodeVx = narrowphaseSceneLayout?.nodeVx;
  const nodeVy = narrowphaseSceneLayout?.nodeVy;
  const nodeR = narrowphaseSceneLayout?.nodeR;
  const nodeInvMass = narrowphaseSceneLayout?.nodeInvMass;

  if (!(nodePairRigidIndex instanceof Uint32Array)
    || !(nodePairNodeIndex instanceof Uint32Array)
    || !(edgePairRigidIndex instanceof Uint32Array)
    || !(edgePairSpringIndex instanceof Uint32Array)
    || !(edgePairNodeAIndex instanceof Uint32Array)
    || !(edgePairNodeBIndex instanceof Uint32Array)
    || !(rigidX instanceof Float32Array)
    || !(rigidY instanceof Float32Array)
    || !(rigidMinX instanceof Float32Array)
    || !(rigidMinY instanceof Float32Array)
    || !(rigidMaxX instanceof Float32Array)
    || !(rigidMaxY instanceof Float32Array)
    || !(rigidInvMass instanceof Float32Array)
    || !(rigidInvInertia instanceof Float32Array)
    || !(nodeX instanceof Float32Array)
    || !(nodeY instanceof Float32Array)
    || !(nodeVx instanceof Float32Array)
    || !(nodeVy instanceof Float32Array)
    || !(nodeR instanceof Float32Array)
    || !(nodeInvMass instanceof Float32Array)) {
    return null;
  }

  let signature = 0x811c9dc5;
  signature = fnv1aMix(signature, nodePairRigidIndex.length >>> 0);
  signature = fnv1aMix(signature, edgePairRigidIndex.length >>> 0);
  signature = fnv1aMix(signature, rigidX.length >>> 0);
  signature = fnv1aMix(signature, nodeX.length >>> 0);
  signature = fnv1aMix(signature, narrowphaseLayout?.signature >>> 0);
  signature = fnv1aMix(signature, narrowphaseSceneLayout?.signature >>> 0);

  const sampleNodeCount = Math.min(nodePairNodeIndex.length, 64);
  for (let i = 0; i < sampleNodeCount; i++) {
    signature = fnv1aMix(signature, nodePairRigidIndex[i] >>> 0);
    signature = fnv1aMix(signature, nodePairNodeIndex[i] >>> 0);
  }

  return {
    nodePairRigidIndex,
    nodePairNodeIndex,
    edgePairRigidIndex,
    edgePairSpringIndex,
    edgePairNodeAIndex,
    edgePairNodeBIndex,
    rigidX,
    rigidY,
    rigidMinX,
    rigidMinY,
    rigidMaxX,
    rigidMaxY,
    rigidInvMass,
    rigidInvInertia,
    nodeX,
    nodeY,
    nodeVx,
    nodeVy,
    nodeR,
    nodeInvMass,
    nodePairCount: nodePairRigidIndex.length,
    edgePairCount: edgePairRigidIndex.length,
    byteLength:
      nodePairRigidIndex.byteLength
      + nodePairNodeIndex.byteLength
      + edgePairRigidIndex.byteLength
      + edgePairSpringIndex.byteLength
      + edgePairNodeAIndex.byteLength
      + edgePairNodeBIndex.byteLength
      + rigidX.byteLength
      + rigidY.byteLength
      + rigidMinX.byteLength
      + rigidMinY.byteLength
      + rigidMaxX.byteLength
      + rigidMaxY.byteLength
      + rigidInvMass.byteLength
      + rigidInvInertia.byteLength
      + nodeX.byteLength
      + nodeY.byteLength
      + nodeVx.byteLength
      + nodeVy.byteLength
      + nodeR.byteLength
      + nodeInvMass.byteLength,
    signature: signature >>> 0,
  };
}

const rigidSoftNodeResponseWgsl = /* wgsl */`
struct Params {
  pairCount: u32,
  restitution: f32,
  contactSlop: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pairIndex: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> rigidStateA: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rigidStateB: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> nodeStateA: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> nodeStateB: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> rigidVertexRange: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> rigidVertices: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> outNodeDelta: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> outRigidDeltaA: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> outRigidDeltaB: array<vec2<f32>>;

fn finiteOr(v: f32, fallback: f32) -> f32 {
  return select(fallback, v, v == v && abs(v) < 1e20);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.pairCount) { return; }

  outNodeDelta[i] = vec4<f32>(0.0);
  outRigidDeltaA[i] = vec4<f32>(0.0);
  outRigidDeltaB[i] = vec2<f32>(0.0);

  let pair = pairIndex[i];
  let rbi = pair.x;
  let ni = pair.y;

  let nA = nodeStateA[ni];
  let nB = nodeStateB[ni];
  let x = finiteOr(nA.x, 0.0);
  let y = finiteOr(nA.y, 0.0);
  let r = max(0.4, finiteOr(nB.x, 1.0));

  let vr = rigidVertexRange[rbi];
  let start = vr.x;
  let endv = vr.y;
  if (endv <= start + 1u) { return; }

  var bestD2 = 1e30;
  var cpX = 0.0;
  var cpY = 0.0;
  var nx = 0.0;
  var ny = 1.0;

  var vi = start;
  loop {
    if (vi >= endv) { break; }
    let vj = select(start, vi + 1u, vi + 1u < endv);
    let a = rigidVertices[vi];
    let b = rigidVertices[vj];
    let abx = b.x - a.x;
    let aby = b.y - a.y;
    let ab2 = max(1e-6, abx * abx + aby * aby);
    let t = clamp(((x - a.x) * abx + (y - a.y) * aby) / ab2, 0.0, 1.0);
    let cpx = a.x + abx * t;
    let cpy = a.y + aby * t;
    let dx = x - cpx;
    let dy = y - cpy;
    let d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      cpX = cpx;
      cpY = cpy;
      let el = max(1e-6, sqrt(ab2));
      nx = -aby / el;
      ny = abx / el;
    }
    vi = vi + 1u;
  }

  let dist = max(1e-6, sqrt(bestD2));
  let penetration = r - dist;
  if (penetration <= params.contactSlop) { return; }

  let tx = x - cpX;
  let ty = y - cpY;
  if (tx * nx + ty * ny < 0.0) { nx = -nx; ny = -ny; }

  let invNode = max(1e-4, finiteOr(nB.y, 1.0));
  let rA = rigidStateA[rbi];
  let rB = rigidStateB[rbi];
  let invRigid = max(1e-4, finiteOr(rB.y, 1.0));
  let invSum = invNode + invRigid;
  let corr = (penetration / max(1e-6, invSum)) * 0.68;
  outNodeDelta[i].x = nx * corr * invNode;
  outNodeDelta[i].y = ny * corr * invNode;
  outRigidDeltaA[i].x = -nx * corr * invRigid;
  outRigidDeltaA[i].y = -ny * corr * invRigid;

  let rx = cpX - finiteOr(rA.x, 0.0);
  let ry = cpY - finiteOr(rA.y, 0.0);
  let omega = finiteOr(rB.x, 0.0);
  let rpvx = finiteOr(rA.z, 0.0) - omega * ry;
  let rpvy = finiteOr(rA.w, 0.0) + omega * rx;
  let rvx = finiteOr(nA.z, 0.0) - rpvx;
  let rvy = finiteOr(nA.w, 0.0) - rpvy;
  let vn = rvx * nx + rvy * ny;
  if (vn >= -0.02) {
    outRigidDeltaB[i].y = 1.0;
    return;
  }

  let invInertia = max(1e-4, finiteOr(rB.z, 1.0));
  let rn = rx * ny - ry * nx;
  let denom = max(1e-6, invNode + invRigid + (rn * rn) * invInertia);
  let rawJ = (-(1.0 + params.restitution) * vn) / denom;
  let maxImpactSpeed = 24.0;
  let maxJ = ((1.0 + params.restitution) * min(maxImpactSpeed, abs(vn))) / denom;
  let j = clamp(rawJ, -maxJ, maxJ);
  let jx = j * nx;
  let jy = j * ny;
  outNodeDelta[i].z = jx * invNode;
  outNodeDelta[i].w = jy * invNode;
  outRigidDeltaA[i].z = -jx * invRigid;
  outRigidDeltaA[i].w = -jy * invRigid;
  outRigidDeltaB[i].x = -(rx * jy - ry * jx) * invInertia;
  outRigidDeltaB[i].y = 1.0;
}
`;

const rigidSoftEdgeResponseWgsl = /* wgsl */`
struct Params {
  pairCount: u32,
  restitution: f32,
  edgeSlop: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pairIndex: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> rigidStateA: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> nodeStateA: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outRigidDeltaA: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> outRigidDeltaB: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.pairCount) { return; }

  outRigidDeltaA[i] = vec4<f32>(0.0);
  outRigidDeltaB[i] = vec2<f32>(0.0);

  let pair = pairIndex[i];
  let rbi = pair.x;
  let ai = pair.y;
  let bi = pair.z;

  let rA = rigidStateA[rbi];
  let nA = nodeStateA[ai];
  let nB = nodeStateA[bi];

  let rx = rA.x;
  let ry = rA.y;
  let ax = nA.x;
  let ay = nA.y;
  let bx = nB.x;
  let by = nB.y;

  let abx = bx - ax;
  let aby = by - ay;
  let ab2 = max(1e-6, abx * abx + aby * aby);
  let t = clamp(((rx - ax) * abx + (ry - ay) * aby) / ab2, 0.0, 1.0);
  let cpx = ax + abx * t;
  let cpy = ay + aby * t;

  var nx = rx - cpx;
  var ny = ry - cpy;
  var dist = sqrt(nx * nx + ny * ny);
  let minDist = max(0.8, params.edgeSlop);
  if (dist >= minDist) { return; }
  if (dist < 1e-6) {
    let invLen = 1.0 / max(1e-6, sqrt((-aby) * (-aby) + abx * abx));
    nx = -aby * invLen;
    ny = abx * invLen;
    dist = 1e-6;
  } else {
    nx = nx / dist;
    ny = ny / dist;
  }

  let penetration = minDist - dist;
  outRigidDeltaA[i].x = nx * penetration * 0.92;
  outRigidDeltaA[i].y = ny * penetration * 0.92;

  let edgeVx = (nA.z + nB.z) * 0.5;
  let edgeVy = (nA.w + nB.w) * 0.5;
  let rvx = rA.z - edgeVx;
  let rvy = rA.w - edgeVy;
  let vn = rvx * nx + rvy * ny;
  if (vn < 0.0) {
    let j = -(1.0 + params.restitution) * vn;
    outRigidDeltaA[i].z = nx * j;
    outRigidDeltaA[i].w = ny * j;
  }
  outRigidDeltaB[i].y = 1.0;
}
`;

async function dispatchRigidSoftResponseWgsl({
  offload,
  impulseSeed,
  edgeSlop,
}) {
  if (!canUseWgslOffload(offload)) return null;
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const nodePairCount = Number(impulseSeed?.nodePairCount) || 0;
  const edgePairCount = Number(impulseSeed?.edgePairCount) || 0;
  const timing = {
    nodeBuildMs: 0,
    nodeDispatchSubmitMs: 0,
    nodeReadbackMs: 0,
    nodeDecodeMs: 0,
    edgeBuildMs: 0,
    edgeDispatchSubmitMs: 0,
    edgeReadbackMs: 0,
    edgeDecodeMs: 0,
    totalMs: 0,
  };
  const totalStartMs = performance.now();
  state.lastRigidSoftResponseTiming = timing;
  if (nodePairCount <= 0 && edgePairCount <= 0) {
    timing.totalMs = Math.max(0, performance.now() - totalStartMs);
    return null;
  }

  const mkReadF32 = async (buf, bytes) => {
    safeUnmapBuffer(buf);
    await buf.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(buf.getMappedRange(0, bytes).slice(0));
    safeUnmapBuffer(buf);
    return out;
  };
  const mkReadU32 = async (buf, bytes) => {
    safeUnmapBuffer(buf);
    await buf.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
    const out = new Uint32Array(buf.getMappedRange(0, bytes).slice(0));
    safeUnmapBuffer(buf);
    return out;
  };

  const storage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const mkStorage = (arr) => {
    const b = device.createBuffer({ size: Math.max(4, arr.byteLength), usage: storage });
    if (arr.byteLength > 0) device.queue.writeBuffer(b, 0, arr);
    return b;
  };

  const mkOut = (count, isU32 = false) => {
    const bytes = Math.max(4, count * 4);
    return {
      gpu: device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC }),
      read: device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ }),
      bytes,
      isU32,
    };
  };

  const resources = [];
  const keep = (b) => (resources.push(b), b);

  try {
    if (!state.rigidSoftNodeResponsePipelinePromise) {
      const module = device.createShaderModule({ code: rigidSoftNodeResponseWgsl });
      state.rigidSoftNodeResponsePipelinePromise = device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })
        .then((p) => { state.rigidSoftNodeResponsePipeline = p; return p; });
    }
    if (!state.rigidSoftEdgeResponsePipelinePromise) {
      const module = device.createShaderModule({ code: rigidSoftEdgeResponseWgsl });
      state.rigidSoftEdgeResponsePipelinePromise = device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })
        .then((p) => { state.rigidSoftEdgeResponsePipeline = p; return p; });
    }

    const nodePipeline = state.rigidSoftNodeResponsePipeline || await state.rigidSoftNodeResponsePipelinePromise;
    const edgePipeline = state.rigidSoftEdgeResponsePipeline || await state.rigidSoftEdgeResponsePipelinePromise;
    if ((nodePairCount > 0 && !nodePipeline) || (edgePairCount > 0 && !edgePipeline)) return null;

    const proposal = { nodePairCount, edgePairCount };

    if (nodePairCount > 0) {
      const nodeBuildStartMs = performance.now();
      const params = keep(device.createBuffer({ size: 16, usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST }));
      const paramFloats = new Float32Array(4);
      new Uint32Array(paramFloats.buffer)[0] = nodePairCount >>> 0;
      paramFloats[1] = 0.28;
      paramFloats[2] = Math.max(0.015, Number(edgeSlop) || 0.16);
      device.queue.writeBuffer(params, 0, paramFloats);

      const rigidCount = Number(impulseSeed?.rigidX?.length) || 0;
      const nodeCount = Number(impulseSeed?.nodeX?.length) || 0;
      const pairPacked = new Uint32Array(nodePairCount * 2);
      for (let i = 0; i < nodePairCount; i++) {
        pairPacked[i * 2] = impulseSeed.nodePairRigidIndex[i] >>> 0;
        pairPacked[i * 2 + 1] = impulseSeed.nodePairNodeIndex[i] >>> 0;
      }

      const rigidStateA = new Float32Array(rigidCount * 4);
      const rigidStateB = new Float32Array(rigidCount * 4);
      for (let i = 0; i < rigidCount; i++) {
        const b4 = i * 4;
        rigidStateA[b4] = impulseSeed.rigidX[i] || 0;
        rigidStateA[b4 + 1] = impulseSeed.rigidY[i] || 0;
        rigidStateA[b4 + 2] = impulseSeed.rigidVx[i] || 0;
        rigidStateA[b4 + 3] = impulseSeed.rigidVy[i] || 0;
        rigidStateB[b4] = impulseSeed.rigidOmega[i] || 0;
        rigidStateB[b4 + 1] = impulseSeed.rigidInvMass[i] || 0;
        rigidStateB[b4 + 2] = impulseSeed.rigidInvInertia[i] || 0;
      }

      const nodeStateA = new Float32Array(nodeCount * 4);
      const nodeStateB = new Float32Array(nodeCount * 4);
      for (let i = 0; i < nodeCount; i++) {
        const b4 = i * 4;
        nodeStateA[b4] = impulseSeed.nodeX[i] || 0;
        nodeStateA[b4 + 1] = impulseSeed.nodeY[i] || 0;
        nodeStateA[b4 + 2] = impulseSeed.nodeVx[i] || 0;
        nodeStateA[b4 + 3] = impulseSeed.nodeVy[i] || 0;
        nodeStateB[b4] = impulseSeed.nodeR[i] || 0;
        nodeStateB[b4 + 1] = impulseSeed.nodeInvMass[i] || 0;
      }

      const rigidVertexStart = impulseSeed.rigidVertexStart instanceof Uint32Array
        ? impulseSeed.rigidVertexStart
        : new Uint32Array(0);
      const rigidVertexRange = new Uint32Array(rigidCount * 2);
      for (let rbi = 0; rbi < rigidCount; rbi++) {
        const start = rigidVertexStart[rbi] >>> 0;
        const end = (rbi + 1) < rigidVertexStart.length
          ? (rigidVertexStart[rbi + 1] >>> 0)
          : start;
        const b2 = rbi * 2;
        rigidVertexRange[b2] = start;
        rigidVertexRange[b2 + 1] = end;
      }

      const rigidVertexX = impulseSeed.rigidVertexX instanceof Float32Array
        ? impulseSeed.rigidVertexX
        : new Float32Array(0);
      const rigidVertexY = impulseSeed.rigidVertexY instanceof Float32Array
        ? impulseSeed.rigidVertexY
        : new Float32Array(0);
      const vertexCount = Math.min(rigidVertexX.length, rigidVertexY.length);
      const rigidVertices = new Float32Array(vertexCount * 2);
      for (let vi = 0; vi < vertexCount; vi++) {
        const b2 = vi * 2;
        rigidVertices[b2] = rigidVertexX[vi] || 0;
        rigidVertices[b2 + 1] = rigidVertexY[vi] || 0;
      }

      const inBuffers = [
        params,
        keep(mkStorage(pairPacked)),
        keep(mkStorage(rigidStateA)),
        keep(mkStorage(rigidStateB)),
        keep(mkStorage(nodeStateA)),
        keep(mkStorage(nodeStateB)),
        keep(mkStorage(rigidVertexRange)),
        keep(mkStorage(rigidVertices)),
      ];
      const outs = {
        nodeDelta: mkOut(nodePairCount * 4),
        rigidDeltaA: mkOut(nodePairCount * 4),
        rigidDeltaB: mkOut(nodePairCount * 2),
      };
      Object.values(outs).forEach((o) => { keep(o.gpu); keep(o.read); });

      const bindGroup = device.createBindGroup({
        layout: nodePipeline.getBindGroupLayout(0),
        entries: [...inBuffers, outs.nodeDelta.gpu, outs.rigidDeltaA.gpu, outs.rigidDeltaB.gpu]
          .map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      timing.nodeBuildMs += Math.max(0, performance.now() - nodeBuildStartMs);

      const nodeDispatchSubmitStartMs = performance.now();
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(nodePipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(nodePairCount / WGSL_WORKGROUP_SIZE)));
      pass.end();
      Object.values(outs).forEach((o) => encoder.copyBufferToBuffer(o.gpu, 0, o.read, 0, o.bytes));
      device.queue.submit([encoder.finish()]);
      timing.nodeDispatchSubmitMs += Math.max(0, performance.now() - nodeDispatchSubmitStartMs);

      const nodeReadbackStartMs = performance.now();
      const [nodeDelta, rigidDeltaA, rigidDeltaB] = await Promise.all([
        mkReadF32(outs.nodeDelta.read, outs.nodeDelta.bytes),
        mkReadF32(outs.rigidDeltaA.read, outs.rigidDeltaA.bytes),
        mkReadF32(outs.rigidDeltaB.read, outs.rigidDeltaB.bytes),
      ]);
      timing.nodeReadbackMs += Math.max(0, performance.now() - nodeReadbackStartMs);

      const nodeDecodeStartMs = performance.now();
      proposal.nodeDx = new Float32Array(nodePairCount);
      proposal.nodeDy = new Float32Array(nodePairCount);
      proposal.nodeDVx = new Float32Array(nodePairCount);
      proposal.nodeDVy = new Float32Array(nodePairCount);
      proposal.nodeRigidDx = new Float32Array(nodePairCount);
      proposal.nodeRigidDy = new Float32Array(nodePairCount);
      proposal.nodeRigidDVx = new Float32Array(nodePairCount);
      proposal.nodeRigidDVy = new Float32Array(nodePairCount);
      proposal.nodeRigidDOmega = new Float32Array(nodePairCount);
      proposal.nodeContactMask = new Uint32Array(nodePairCount);

      for (let i = 0; i < nodePairCount; i++) {
        const b4 = i * 4;
        const b2 = i * 2;
        proposal.nodeDx[i] = nodeDelta[b4] || 0;
        proposal.nodeDy[i] = nodeDelta[b4 + 1] || 0;
        proposal.nodeDVx[i] = nodeDelta[b4 + 2] || 0;
        proposal.nodeDVy[i] = nodeDelta[b4 + 3] || 0;
        proposal.nodeRigidDx[i] = rigidDeltaA[b4] || 0;
        proposal.nodeRigidDy[i] = rigidDeltaA[b4 + 1] || 0;
        proposal.nodeRigidDVx[i] = rigidDeltaA[b4 + 2] || 0;
        proposal.nodeRigidDVy[i] = rigidDeltaA[b4 + 3] || 0;
        proposal.nodeRigidDOmega[i] = rigidDeltaB[b2] || 0;
        proposal.nodeContactMask[i] = (rigidDeltaB[b2 + 1] || 0) > 0.5 ? 1 : 0;
      }
      timing.nodeDecodeMs += Math.max(0, performance.now() - nodeDecodeStartMs);
    }

    if (edgePairCount > 0) {
      const edgeBuildStartMs = performance.now();
      const params = keep(device.createBuffer({ size: 16, usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST }));
      const paramFloats = new Float32Array(4);
      new Uint32Array(paramFloats.buffer)[0] = edgePairCount >>> 0;
      paramFloats[1] = 0.28;
      paramFloats[2] = Math.max(0.8, Number(edgeSlop) || 1.0);
      device.queue.writeBuffer(params, 0, paramFloats);

      const pairPacked = new Uint32Array(edgePairCount * 4);
      for (let i = 0; i < edgePairCount; i++) {
        const b4 = i * 4;
        pairPacked[b4] = impulseSeed.edgePairRigidIndex[i] >>> 0;
        pairPacked[b4 + 1] = impulseSeed.edgePairNodeAIndex[i] >>> 0;
        pairPacked[b4 + 2] = impulseSeed.edgePairNodeBIndex[i] >>> 0;
      }

      const rigidCount = Number(impulseSeed?.rigidX?.length) || 0;
      const rigidStateA = new Float32Array(rigidCount * 4);
      for (let i = 0; i < rigidCount; i++) {
        const b4 = i * 4;
        rigidStateA[b4] = impulseSeed.rigidX[i] || 0;
        rigidStateA[b4 + 1] = impulseSeed.rigidY[i] || 0;
        rigidStateA[b4 + 2] = impulseSeed.rigidVx[i] || 0;
        rigidStateA[b4 + 3] = impulseSeed.rigidVy[i] || 0;
      }

      const nodeCount = Number(impulseSeed?.nodeX?.length) || 0;
      const nodeStateA = new Float32Array(nodeCount * 4);
      for (let i = 0; i < nodeCount; i++) {
        const b4 = i * 4;
        nodeStateA[b4] = impulseSeed.nodeX[i] || 0;
        nodeStateA[b4 + 1] = impulseSeed.nodeY[i] || 0;
        nodeStateA[b4 + 2] = impulseSeed.nodeVx[i] || 0;
        nodeStateA[b4 + 3] = impulseSeed.nodeVy[i] || 0;
      }

      const inBuffers = [
        params,
        keep(mkStorage(pairPacked)),
        keep(mkStorage(rigidStateA)),
        keep(mkStorage(nodeStateA)),
      ];
      const outs = {
        rigidDeltaA: mkOut(edgePairCount * 4),
        rigidDeltaB: mkOut(edgePairCount * 2),
      };
      Object.values(outs).forEach((o) => { keep(o.gpu); keep(o.read); });

      const bindGroup = device.createBindGroup({
        layout: edgePipeline.getBindGroupLayout(0),
        entries: [...inBuffers, outs.rigidDeltaA.gpu, outs.rigidDeltaB.gpu]
          .map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      timing.edgeBuildMs += Math.max(0, performance.now() - edgeBuildStartMs);
      const edgeDispatchSubmitStartMs = performance.now();
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(edgePipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(edgePairCount / WGSL_WORKGROUP_SIZE)));
      pass.end();
      Object.values(outs).forEach((o) => encoder.copyBufferToBuffer(o.gpu, 0, o.read, 0, o.bytes));
      device.queue.submit([encoder.finish()]);
      timing.edgeDispatchSubmitMs += Math.max(0, performance.now() - edgeDispatchSubmitStartMs);

      const edgeReadbackStartMs = performance.now();
      const [rigidDeltaA, rigidDeltaB] = await Promise.all([
        mkReadF32(outs.rigidDeltaA.read, outs.rigidDeltaA.bytes),
        mkReadF32(outs.rigidDeltaB.read, outs.rigidDeltaB.bytes),
      ]);
      timing.edgeReadbackMs += Math.max(0, performance.now() - edgeReadbackStartMs);

      const edgeDecodeStartMs = performance.now();
      proposal.edgeRigidDx = new Float32Array(edgePairCount);
      proposal.edgeRigidDy = new Float32Array(edgePairCount);
      proposal.edgeRigidDVx = new Float32Array(edgePairCount);
      proposal.edgeRigidDVy = new Float32Array(edgePairCount);
      proposal.edgeContactMask = new Uint32Array(edgePairCount);

      for (let i = 0; i < edgePairCount; i++) {
        const b4 = i * 4;
        const b2 = i * 2;
        proposal.edgeRigidDx[i] = rigidDeltaA[b4] || 0;
        proposal.edgeRigidDy[i] = rigidDeltaA[b4 + 1] || 0;
        proposal.edgeRigidDVx[i] = rigidDeltaA[b4 + 2] || 0;
        proposal.edgeRigidDVy[i] = rigidDeltaA[b4 + 3] || 0;
        proposal.edgeContactMask[i] = (rigidDeltaB[b2 + 1] || 0) > 0.5 ? 1 : 0;
      }
      timing.edgeDecodeMs += Math.max(0, performance.now() - edgeDecodeStartMs);
    }

    timing.totalMs = Math.max(0, performance.now() - totalStartMs);
    return proposal;
  } finally {
    if (!(Number(timing.totalMs) > 0)) timing.totalMs = Math.max(0, performance.now() - totalStartMs);
    resources.forEach((b) => b?.destroy?.());
  }
}

function isFiniteArray(arr, expectedLength) {
  if (!(arr instanceof Float32Array) || arr.length !== expectedLength) return false;
  for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) return false;
  return true;
}

function isBinaryMask(arr, expectedLength) {
  if ((expectedLength | 0) === 0) {
    return arr == null || (arr instanceof Uint32Array && arr.length === 0);
  }
  if (!(arr instanceof Uint32Array) || arr.length !== expectedLength) return false;
  for (let i = 0; i < arr.length; i++) {
    const bit = arr[i] >>> 0;
    if (bit !== 0 && bit !== 1) return false;
  }
  return true;
}

function isFiniteArrayWithin(arr, expectedLength, absLimit) {
  if (!isFiniteArray(arr, expectedLength)) return false;
  for (let i = 0; i < arr.length; i++) {
    if (Math.abs(arr[i]) > absLimit) return false;
  }
  return true;
}

function validateRigidSoftAuthoritativeProposal({ proposal, impulseSeed, signature }) {
  if (!proposal || !impulseSeed) return { ok: false, reason: 'missing-proposal-or-seed' };
  if ((signature >>> 0) !== ((impulseSeed.signature || 0) >>> 0)) return { ok: false, reason: 'signature-mismatch' };
  const np = Number(impulseSeed.nodePairCount) || 0;
  const ep = Number(impulseSeed.edgePairCount) || 0;
  const posLimit = 2048;
  const velLimit = 4096;
  const omegaLimit = 4096;

  if (!isFiniteArrayWithin(proposal.nodeDx || new Float32Array(0), np, posLimit)) return { ok: false, reason: 'nodeDx-invalid' };
  if (!isFiniteArrayWithin(proposal.nodeDy || new Float32Array(0), np, posLimit)) return { ok: false, reason: 'nodeDy-invalid' };
  if (!isFiniteArrayWithin(proposal.nodeDVx || new Float32Array(0), np, velLimit)) return { ok: false, reason: 'nodeDVx-invalid' };
  if (!isFiniteArrayWithin(proposal.nodeDVy || new Float32Array(0), np, velLimit)) return { ok: false, reason: 'nodeDVy-invalid' };
  if (!isFiniteArrayWithin(proposal.nodeRigidDx || new Float32Array(0), np, posLimit)) return { ok: false, reason: 'nodeRigidDx-invalid' };
  if (!isFiniteArrayWithin(proposal.nodeRigidDy || new Float32Array(0), np, posLimit)) return { ok: false, reason: 'nodeRigidDy-invalid' };
  if (!isFiniteArrayWithin(proposal.nodeRigidDVx || new Float32Array(0), np, velLimit)) return { ok: false, reason: 'nodeRigidDVx-invalid' };
  if (!isFiniteArrayWithin(proposal.nodeRigidDVy || new Float32Array(0), np, velLimit)) return { ok: false, reason: 'nodeRigidDVy-invalid' };
  if (!isFiniteArrayWithin(proposal.nodeRigidDOmega || new Float32Array(0), np, omegaLimit)) return { ok: false, reason: 'nodeRigidDOmega-invalid' };
  if (!isBinaryMask(proposal.nodeContactMask, np)) return { ok: false, reason: 'nodeContactMask-invalid' };
  if (!isFiniteArrayWithin(proposal.edgeRigidDx || new Float32Array(0), ep, posLimit)) return { ok: false, reason: 'edgeRigidDx-invalid' };
  if (!isFiniteArrayWithin(proposal.edgeRigidDy || new Float32Array(0), ep, posLimit)) return { ok: false, reason: 'edgeRigidDy-invalid' };
  if (!isFiniteArrayWithin(proposal.edgeRigidDVx || new Float32Array(0), ep, velLimit)) return { ok: false, reason: 'edgeRigidDVx-invalid' };
  if (!isFiniteArrayWithin(proposal.edgeRigidDVy || new Float32Array(0), ep, velLimit)) return { ok: false, reason: 'edgeRigidDVy-invalid' };
  if (!isBinaryMask(proposal.edgeContactMask, ep)) return { ok: false, reason: 'edgeContactMask-invalid' };

  return { ok: true, reason: null };
}

function canApplyRigidSoftAuthoritativeProposal({ proposal, impulseSeed, signature, out = null }) {
  const validation = validateRigidSoftAuthoritativeProposal({ proposal, impulseSeed, signature });
  if (out && typeof out === 'object') out.reason = validation.reason;
  return validation.ok;
}

function applyRigidSoftAuthoritativeProposal({ rigidBodies, soft, impulseSeed, proposal }) {
  for (let i = 0; i < impulseSeed.nodePairCount; i++) {
    if ((proposal.nodeContactMask[i] >>> 0) === 0) continue;
    const rbi = impulseSeed.nodePairRigidIndex[i] >>> 0;
    const ni = impulseSeed.nodePairNodeIndex[i] >>> 0;
    const rb = rigidBodies[rbi];
    const sn = soft.nodes[ni];
    if (!rb || !sn) continue;
    sn.x += proposal.nodeDx[i];
    sn.y += proposal.nodeDy[i];
    sn.vx += proposal.nodeDVx[i];
    sn.vy += proposal.nodeDVy[i];
    rb.x += proposal.nodeRigidDx[i];
    rb.y += proposal.nodeRigidDy[i];
    rb.vx += proposal.nodeRigidDVx[i];
    rb.vy += proposal.nodeRigidDVy[i];
    rb.omega = (rb.omega || 0) + proposal.nodeRigidDOmega[i];
  }

  for (let i = 0; i < impulseSeed.edgePairCount; i++) {
    if ((proposal.edgeContactMask[i] >>> 0) === 0) continue;
    const rbi = impulseSeed.edgePairRigidIndex[i] >>> 0;
    const rb = rigidBodies[rbi];
    if (!rb) continue;
    rb.x += proposal.edgeRigidDx[i];
    rb.y += proposal.edgeRigidDy[i];
    rb.vx += proposal.edgeRigidDVx[i];
    rb.vy += proposal.edgeRigidDVy[i];
  }
}

function resolveRigidSoftNodeCollisionCpuFallback(rb, sn, nodeSolverInput, nodeSlop) {
  return resolveRigidVsSoftNodeCollisionGpuOnly(rb, sn, nodeSolverInput, nodeSlop);
}

function resolveRigidSoftEdgeCollisionCpuFallback(rb, a, b, edgeSlop) {
  return resolveRigidVsSoftEdgeCollisionGpuOnly(rb, a, b, edgeSlop);
}

function addRigidSoftProfileMs(profile, key, deltaMs) {
  if (!profile || !key) return;
  const ms = Number(deltaMs);
  if (!Number.isFinite(ms) || ms <= 0) return;
  if (!profile.substageMs || typeof profile.substageMs !== 'object') {
    profile.substageMs = {};
  }
  profile.substageMs[key] = (Number(profile.substageMs[key]) || 0) + ms;
}

function applyCpuRigidSoftResponseFallback({
  rigidBodies,
  soft,
  compactNodePairs,
  compactEdgePairs,
  wgslNodeBroadphaseMask,
  wgslEdgeBroadphaseMask,
  edgeBodyModeBlock,
  nodeSlop,
  edgeSlop,
  wgslOffload,
  profile,
}) {
  const solveNodeCollision = (rb, sn) => resolveRigidSoftNodeCollisionCpuFallback(rb, sn, null, nodeSlop);
  const solveEdgeCollision = (rb, a, b) => resolveRigidSoftEdgeCollisionCpuFallback(rb, a, b, edgeSlop);

  const nodeStartMs = profile ? performance.now() : 0;
  const compactNodeRigidIndex = compactNodePairs?.compactRigidIndex;
  const compactNodeNodeIndex = compactNodePairs?.compactNodeIndex;
  if (compactNodeRigidIndex instanceof Uint32Array && compactNodeNodeIndex instanceof Uint32Array) {
    if (wgslOffload?.state) {
      wgslOffload.state.lastNodeCollisionResponseSource = 'cpu-rigid-soft-compact-node-hard-fallback-response';
      wgslOffload.state.lastNodeCollisionResponsePairCount = compactNodePairs.pairCount | 0;
      if (canUseWgslOffload(wgslOffload)) {
        wgslOffload.state.lastSourceRoute = 'cpu-rigid-soft-response-hard-fallback';
        wgslOffload.state.lastMode = 'cpu-rigid-soft-response-hard-fallback';
      }
    }
    for (let pi = 0; pi < compactNodePairs.pairCount; pi++) {
      const rbi = compactNodeRigidIndex[pi] | 0;
      const ni = compactNodeNodeIndex[pi] | 0;
      const rb = rigidBodies[rbi];
      const sn = soft.nodes[ni];
      if (!rb || !sn) continue;
      solveNodeCollision(rb, sn);
    }
  } else {
    if (wgslOffload?.state) {
      wgslOffload.state.lastNodeCollisionResponseSource = 'cpu-rigid-soft-full-scan-hard-fallback-response';
      if (canUseWgslOffload(wgslOffload)) {
        wgslOffload.state.lastSourceRoute = 'cpu-rigid-soft-response-hard-fallback';
        wgslOffload.state.lastMode = 'cpu-rigid-soft-response-hard-fallback';
      }
    }
    let nodePairCursor = 0;
    for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
      const rb = rigidBodies[rbi];
      for (let ni = 0; ni < soft.nodes.length; ni++) {
        const broadphaseActive = wgslNodeBroadphaseMask
          ? (wgslNodeBroadphaseMask[nodePairCursor] === 1)
          : true;
        nodePairCursor++;
        if (!broadphaseActive) continue;
        const sn = soft.nodes[ni];
        solveNodeCollision(rb, sn);
      }
    }
  }

  if (profile) {
    profile.cpuFallbackUsedCompactNodePairs = Boolean(compactNodeRigidIndex instanceof Uint32Array && compactNodeNodeIndex instanceof Uint32Array);
    profile.cpuFallbackNodePairCount = profile.cpuFallbackUsedCompactNodePairs
      ? (compactNodePairs?.pairCount | 0)
      : ((rigidBodies.length | 0) * (soft.nodes.length | 0));
    const nodeWallMs = Math.max(0, performance.now() - nodeStartMs);
    profile.cpuFallbackNodeWallMs = (profile.cpuFallbackNodeWallMs || 0) + nodeWallMs;
    addRigidSoftProfileMs(profile, 'cpuFallbackNodeApplyMs', nodeWallMs);
  }

  const edgeStartMs = profile ? performance.now() : 0;
  const compactEdgeRigidIndex = compactEdgePairs?.compactRigidIndex;
  const compactEdgeNodeAIndex = compactEdgePairs?.compactNodeAIndex;
  const compactEdgeNodeBIndex = compactEdgePairs?.compactNodeBIndex;
  if (
    compactEdgeRigidIndex instanceof Uint32Array
    && compactEdgeNodeAIndex instanceof Uint32Array
    && compactEdgeNodeBIndex instanceof Uint32Array
  ) {
    if (wgslOffload?.state) {
      wgslOffload.state.lastEdgeCollisionResponseSource = 'cpu-rigid-soft-compact-edge-hard-fallback-response';
      wgslOffload.state.lastEdgeCollisionResponsePairCount = compactEdgePairs.pairCount | 0;
    }
    for (let pi = 0; pi < compactEdgePairs.pairCount; pi++) {
      const rbi = compactEdgeRigidIndex[pi] | 0;
      const ni = compactEdgeNodeAIndex[pi] | 0;
      const nj = compactEdgeNodeBIndex[pi] | 0;
      const rb = rigidBodies[rbi];
      const a = soft.nodes[ni];
      const b = soft.nodes[nj];
      if (!rb || !a || !b) continue;
      solveEdgeCollision(rb, a, b);
    }
  } else {
    if (wgslOffload?.state) {
      wgslOffload.state.lastEdgeCollisionResponseSource = 'cpu-rigid-soft-full-scan-edge-hard-fallback-response';
    }
    let edgePairCursor = 0;
    for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
      const rb = rigidBodies[rbi];
      for (let si = 0; si < soft.springs.length; si++) {
        const spring = soft.springs[si];
        if (!Array.isArray(spring) || spring.length < 4) continue;
        const i = Number(spring[0]) | 0;
        const j = Number(spring[1]) | 0;
        const edgeBodyMode = spring[3];
        if (edgeBodyMode !== edgeBodyModeBlock) continue;
        if (i < 0 || j < 0 || i >= soft.nodes.length || j >= soft.nodes.length) continue;
        const broadphaseActive = wgslEdgeBroadphaseMask
          ? (wgslEdgeBroadphaseMask[edgePairCursor] === 1)
          : true;
        edgePairCursor++;
        if (!broadphaseActive) continue;
        solveEdgeCollision(rb, soft.nodes[i], soft.nodes[j]);
      }
    }
  }

  if (profile) {
    profile.cpuFallbackUsedCompactEdgePairs = Boolean(
      compactEdgeRigidIndex instanceof Uint32Array
      && compactEdgeNodeAIndex instanceof Uint32Array
      && compactEdgeNodeBIndex instanceof Uint32Array,
    );
    profile.cpuFallbackEdgePairCount = profile.cpuFallbackUsedCompactEdgePairs
      ? (compactEdgePairs?.pairCount | 0)
      : ((rigidBodies.length | 0) * Math.max(0, Array.isArray(soft.springs) ? soft.springs.length : 0));
    const edgeWallMs = Math.max(0, performance.now() - edgeStartMs);
    profile.cpuFallbackEdgeWallMs = (profile.cpuFallbackEdgeWallMs || 0) + edgeWallMs;
    addRigidSoftProfileMs(profile, 'cpuFallbackEdgeApplyMs', edgeWallMs);
  }
}

export async function resolveRigidSoftCollisionPassGpuOnly({
  rigidBodies,
  soft,
  edgeBodyModeBlock,
  nodeSlop = 0.18,
  edgeSlop = 0.16,
  wgslOffload,
  profile,
}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return;
  if (!soft || !Array.isArray(soft.nodes) || !Array.isArray(soft.springs)) return;

  if (profile) {
    profile.route = 'cpu-fallback';
    profile.responseOwnership = 'cpu-hard-fallback';
    profile.fallbackReason = null;
    profile.usedWgslAuthoritativeResponse = false;
    profile.cpuFallbackNodeWallMs = 0;
    profile.cpuFallbackEdgeWallMs = 0;
    profile.cpuFallbackNodePairCount = 0;
    profile.cpuFallbackEdgePairCount = 0;
    profile.substageMs = {};
  }

  if (soft.nodes.length === 0) {
    if (profile) {
      profile.route = 'cpu-fallback';
      profile.fallbackReason = 'empty-soft-scene';
    }
    if (wgslOffload?.enabled === true && wgslOffload?.state) {
      wgslOffload.state.lastPreparedNodePairCount = 0;
      wgslOffload.state.lastPreparedEdgePairCount = 0;
      wgslOffload.state.lastPreparedLayoutBytes = 0;
      wgslOffload.state.lastSourceRoute = 'cpu-rigid-soft-empty-scene';
      wgslOffload.state.lastMode = 'cpu-empty-scene';
      wgslOffload.state.lastError = null;
    }
    return;
  }

  let wgslPrep = null;
  let wgslNodeBroadphase = null;
  let wgslEdgeBroadphase = null;
  let compactNodePairs = null;
  let compactEdgePairs = null;
  const wgslUnavailableReason = getWgslOffloadUnavailableReason(wgslOffload);
  const wgslOffloadAvailable = wgslUnavailableReason === null;
  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    if (!wgslOffloadAvailable) {
      const layoutStartMs = profile ? performance.now() : 0;
      wgslPrep = buildRigidSoftCollisionWgslLayout({
        rigidBodies,
        soft,
        edgeBodyModeBlock,
      });
      if (profile) addRigidSoftProfileMs(profile, 'candidateLayoutBuildMs', performance.now() - layoutStartMs);
      const hasAnyCandidatePairs = (wgslPrep.nodePairCount | 0) > 0 || (wgslPrep.edgePairCount | 0) > 0;
      wgslOffload.state.preparedLayout = wgslPrep.layout;
      wgslOffload.state.lastPreparedNodePairCount = wgslPrep.nodePairCount;
      wgslOffload.state.lastPreparedEdgePairCount = wgslPrep.edgePairCount;
      wgslOffload.state.lastPreparedLayoutBytes = wgslPrep.byteLength;
      wgslOffload.state.lastPreparedLayoutSignature = wgslPrep.signature;
      wgslOffload.state.lastSourceRoute = hasAnyCandidatePairs
        ? 'cpu-rigid-soft-candidate-layout'
        : 'cpu-rigid-soft-empty-candidates';
      wgslOffload.state.lastMode = hasAnyCandidatePairs ? 'cpu-prepared' : 'cpu-empty-candidates';
      wgslOffload.state.lastError = wgslUnavailableReason;
      wgslOffload.state.lastWgslUnavailableReason = wgslUnavailableReason;
      wgslOffload.state.lastRigidSoftResponseAuthoritativeSource = 'cpu-rigid-soft-response-fallback-unavailable';
      wgslOffload.state.lastRigidSoftResponseOwnership = 'cpu-fallback';
      wgslOffload.state.lastRigidSoftResponseRoute = 'cpu-fallback';
      wgslOffload.state.lastRigidSoftResponseFallbackReason = wgslUnavailableReason;

      const geometryStartMs = profile ? performance.now() : 0;
      const wgslNarrowphaseGeometry = buildRigidSoftNarrowphaseGeometryLayout(rigidBodies);
      if (profile) addRigidSoftProfileMs(profile, 'narrowphaseGeometryLayoutBuildMs', performance.now() - geometryStartMs);
      wgslOffload.state.lastPreparedNarrowphaseGeometry = wgslNarrowphaseGeometry.layout;
      wgslOffload.state.lastPreparedNarrowphaseRigidCount = wgslNarrowphaseGeometry.rigidCount;
      wgslOffload.state.lastPreparedNarrowphaseEdgeCount = wgslNarrowphaseGeometry.edgeCount;
      wgslOffload.state.lastPreparedNarrowphaseBytes = wgslNarrowphaseGeometry.byteLength;
      wgslOffload.state.lastPreparedNarrowphaseSignature = wgslNarrowphaseGeometry.signature;
      if (!hasAnyCandidatePairs) return;
    } else {
      const layoutStartMs = profile ? performance.now() : 0;
      wgslPrep = buildRigidSoftCollisionWgslLayout({
        rigidBodies,
        soft,
        edgeBodyModeBlock,
      });
      if (profile) addRigidSoftProfileMs(profile, 'candidateLayoutBuildMs', performance.now() - layoutStartMs);
    const hasAnyCandidatePairs = (wgslPrep.nodePairCount | 0) > 0 || (wgslPrep.edgePairCount | 0) > 0;
    wgslOffload.state.preparedLayout = wgslPrep.layout;
    wgslOffload.state.lastPreparedNodePairCount = wgslPrep.nodePairCount;
    wgslOffload.state.lastPreparedEdgePairCount = wgslPrep.edgePairCount;
    wgslOffload.state.lastPreparedLayoutBytes = wgslPrep.byteLength;
    wgslOffload.state.lastPreparedLayoutSignature = wgslPrep.signature;
    wgslOffload.state.lastSourceRoute = hasAnyCandidatePairs
      ? 'cpu-rigid-soft-candidate-layout'
      : 'cpu-rigid-soft-empty-candidates';
    wgslOffload.state.lastMode = hasAnyCandidatePairs ? 'cpu-prepared' : 'cpu-empty-candidates';
    wgslOffload.state.lastError = null;

    if (!hasAnyCandidatePairs) {
      return;
    }

    const geometryStartMs = profile ? performance.now() : 0;
    const wgslNarrowphaseGeometry = buildRigidSoftNarrowphaseGeometryLayout(rigidBodies);
    if (profile) addRigidSoftProfileMs(profile, 'narrowphaseGeometryLayoutBuildMs', performance.now() - geometryStartMs);
    wgslOffload.state.lastPreparedNarrowphaseGeometry = wgslNarrowphaseGeometry.layout;
    wgslOffload.state.lastPreparedNarrowphaseRigidCount = wgslNarrowphaseGeometry.rigidCount;
    wgslOffload.state.lastPreparedNarrowphaseEdgeCount = wgslNarrowphaseGeometry.edgeCount;
    wgslOffload.state.lastPreparedNarrowphaseBytes = wgslNarrowphaseGeometry.byteLength;
    wgslOffload.state.lastPreparedNarrowphaseSignature = wgslNarrowphaseGeometry.signature;

    const rigidAabbStartMs = profile ? performance.now() : 0;
    const rigidAabb = computeRigidBodyAabbs(rigidBodies);
    if (profile) addRigidSoftProfileMs(profile, 'rigidAabbBuildMs', performance.now() - rigidAabbStartMs);
    const fastBroadphaseMode = isGpuOnlyFastMode(wgslOffload);
    const broadphaseEpoch = (Number(wgslOffload.state.lastRigidSoftBroadphaseEpoch) || 0) + 1;
    wgslOffload.state.lastRigidSoftBroadphaseEpoch = broadphaseEpoch;
    const fastBroadphaseInterval = normalizeFastRigidSoftBroadphaseInterval(
      wgslOffload?.fastRigidSoftBroadphaseInterval ?? wgslOffload.state?.fastRigidSoftBroadphaseInterval,
    );
    const cadenceDue = !fastBroadphaseMode || (broadphaseEpoch % fastBroadphaseInterval) === 0;
    wgslOffload.state.lastRigidSoftBroadphaseCadenceDue = cadenceDue;
    wgslOffload.state.lastRigidSoftBroadphaseInterval = fastBroadphaseInterval;

    const nodeReplayEpochDelta = broadphaseEpoch - (Number(wgslOffload.state.lastNodeBroadphaseDispatchEpoch) || 0);
    const nodeReplayAvailable = fastBroadphaseMode
      && !cadenceDue
      && nodeReplayEpochDelta > 0
      && nodeReplayEpochDelta <= fastBroadphaseInterval
      && Number(wgslOffload.state.lastPreparedLayoutSignature) === (Number(wgslPrep.signature) || 0)
      && wgslOffload.state.lastPreparedNodeNarrowphasePairRigidIndex instanceof Uint32Array
      && wgslOffload.state.lastPreparedNodeNarrowphasePairNodeIndex instanceof Uint32Array;

    if (nodeReplayAvailable) {
      compactNodePairs = {
        compactRigidIndex: wgslOffload.state.lastPreparedNodeNarrowphasePairRigidIndex,
        compactNodeIndex: wgslOffload.state.lastPreparedNodeNarrowphasePairNodeIndex,
        pairCount: Number(wgslOffload.state.lastPreparedNodeNarrowphasePairCount) || 0,
        byteLength: Number(wgslOffload.state.lastPreparedNodeNarrowphaseBytes) || 0,
        signature: Number(wgslOffload.state.lastPreparedNodeNarrowphaseSignature) || 0,
      };
      wgslOffload.state.lastNodeBroadphaseAuthoritativeSource = 'wgsl-rigid-soft-node-broadphase-authoritative-replay-fast';
      wgslOffload.state.lastNodeBroadphaseReadbackSourceRoute = 'wgsl-rigid-soft-node-broadphase-replay-fast';
      wgslOffload.state.lastNodeBroadphaseCandidatePairCount = Number(wgslPrep.nodePairCount) || 0;
      wgslOffload.state.lastNodeBroadphaseActivePairCount = compactNodePairs.pairCount | 0;
      wgslOffload.state.lastNodeBroadphaseReadbackCount = compactNodePairs.pairCount | 0;
      wgslOffload.state.lastNodeBroadphaseReplayEpochDelta = nodeReplayEpochDelta;
      wgslOffload.state.lastNodeBroadphaseReplayHitCount = (Number(wgslOffload.state.lastNodeBroadphaseReplayHitCount) || 0) + 1;
    } else {
      const nodeBroadphaseStartMs = profile ? performance.now() : 0;
      wgslNodeBroadphase = await dispatchRigidSoftNodeBroadphaseWgsl({
        rigidBodies,
        soft,
        offload: wgslOffload,
        layout: wgslPrep,
        rigidAabb,
      });
      if (profile) {
        addRigidSoftProfileMs(profile, 'nodeBroadphaseDispatchReadbackMs', performance.now() - nodeBroadphaseStartMs);
        const t = wgslOffload?.state?.lastNodeBroadphaseTiming;
        if (t && typeof t === 'object') {
          addRigidSoftProfileMs(profile, 'nodeBroadphaseUploadMs', t.uploadMs);
          addRigidSoftProfileMs(profile, 'nodeBroadphaseDispatchSubmitMs', t.dispatchSubmitMs);
          addRigidSoftProfileMs(profile, 'nodeBroadphaseCountReadbackMs', t.countReadbackMs);
          addRigidSoftProfileMs(profile, 'nodeBroadphaseIndexReadbackMs', t.indexReadbackMs);
          addRigidSoftProfileMs(profile, 'nodeBroadphaseCompactBuildMs', t.compactBuildMs);
          addRigidSoftProfileMs(profile, 'nodeBroadphaseDetailedTotalMs', t.totalMs);
        }
      }
      if (wgslNodeBroadphase?.compactPairs) {
        wgslOffload.state.lastSourceRoute = 'wgsl-rigid-soft-node-broadphase-authoritative-filter';
        wgslOffload.state.lastMode = 'wgsl-broadphase-authoritative-filter';
        wgslOffload.state.lastNodeBroadphaseReadbackCount = Number(wgslNodeBroadphase.activePairCount) || 0;
        wgslOffload.state.lastNodeBroadphaseCandidatePairCount = Number(wgslNodeBroadphase.candidatePairCount) || (wgslPrep.nodePairCount >>> 0);
        wgslOffload.state.lastNodeBroadphaseDispatchEpoch = broadphaseEpoch;
        wgslOffload.state.lastNodeBroadphaseDispatchHitCount = (Number(wgslOffload.state.lastNodeBroadphaseDispatchHitCount) || 0) + 1;
        compactNodePairs = wgslNodeBroadphase.compactPairs;
        wgslOffload.state.lastPreparedNodeNarrowphasePairRigidIndex = compactNodePairs.compactRigidIndex;
        wgslOffload.state.lastPreparedNodeNarrowphasePairNodeIndex = compactNodePairs.compactNodeIndex;
        wgslOffload.state.lastPreparedNodeNarrowphasePairCount = compactNodePairs.pairCount;
        wgslOffload.state.lastPreparedNodeNarrowphaseBytes = compactNodePairs.byteLength;
        wgslOffload.state.lastPreparedNodeNarrowphaseSignature = compactNodePairs.signature;
      } else {
        wgslOffload.state.lastSourceRoute = 'cpu-rigid-soft-node-broadphase-compact-fallback';
        wgslOffload.state.lastMode = 'cpu-authoritative-mask-fallback';
        wgslOffload.state.lastError = wgslOffload.state.lastNodeBroadphaseReason || 'rigid-soft-broadphase-readback-failed';
      }
    }

    if (profile) {
      profile.nodeBroadphaseCandidatePairs = Number(wgslOffload.state.lastNodeBroadphaseCandidatePairCount) || 0;
      profile.nodeBroadphaseActivePairs = Number(wgslOffload.state.lastNodeBroadphaseActivePairCount) || 0;
    }

    if (compactNodePairs && compactNodePairs.pairCount > 0) {
      const nodeProbeInputCount = compactNodePairs.pairCount | 0;
      const nodeProbeStartMs = profile ? performance.now() : 0;
      const nodeProbeFinite = await dispatchRigidSoftNodeNarrowphaseAabbProbeWgsl({
        rigidBodies,
        soft,
        offload: wgslOffload,
        nodePairs: compactNodePairs,
        rigidAabb,
      });
      if (profile) {
        addRigidSoftProfileMs(profile, 'nodeNarrowphaseAabbProbeDispatchReadbackMs', performance.now() - nodeProbeStartMs);
        const t = wgslOffload?.state?.lastNodeNarrowphaseAabbProbeTiming;
        if (t && typeof t === 'object') {
          addRigidSoftProfileMs(profile, 'nodeNarrowphaseAabbProbeInputBuildMs', t.inputBuildMs);
          addRigidSoftProfileMs(profile, 'nodeNarrowphaseAabbProbeUploadMs', t.uploadMs);
          addRigidSoftProfileMs(profile, 'nodeNarrowphaseAabbProbeDispatchSubmitMs', t.dispatchSubmitMs);
          addRigidSoftProfileMs(profile, 'nodeNarrowphaseAabbProbeReadbackMapMs', t.readbackMapMs);
          addRigidSoftProfileMs(profile, 'nodeNarrowphaseAabbProbeDecodeMs', t.decodeMs);
          addRigidSoftProfileMs(profile, 'nodeNarrowphaseAabbProbeDetailedTotalMs', t.totalMs);
        }
      }

      const nodeProbeFilterStartMs = profile ? performance.now() : 0;
      const nodeProbeFilteredPairs = nodeProbeFinite
        ? buildAuthoritativeRigidSoftNodePairsFromAabbProbe({
          nodePairs: compactNodePairs,
          separation: wgslOffload.state.lastNodeNarrowphaseAabbProbeSeparation,
          insideMask: wgslOffload.state.lastNodeNarrowphaseAabbProbeInsideMask,
          nodeSlop,
        })
        : null;
      if (profile) addRigidSoftProfileMs(profile, 'nodeNarrowphaseAabbProbeFilterMs', performance.now() - nodeProbeFilterStartMs);

      if (nodeProbeFilteredPairs) {
        compactNodePairs = nodeProbeFilteredPairs;
        wgslOffload.state.lastNodeNarrowphaseDroppedPairCount = Math.max(0, nodeProbeInputCount - (compactNodePairs.pairCount | 0));
        wgslOffload.state.lastNodeNarrowphaseAuthoritativeSource = 'wgsl-rigid-soft-node-aabb-probe-authoritative-filter';
        wgslOffload.state.lastPreparedNodeNarrowphasePairRigidIndex = compactNodePairs.compactRigidIndex;
        wgslOffload.state.lastPreparedNodeNarrowphasePairNodeIndex = compactNodePairs.compactNodeIndex;
        wgslOffload.state.lastPreparedNodeNarrowphasePairCount = compactNodePairs.pairCount;
        wgslOffload.state.lastPreparedNodeNarrowphaseBytes = compactNodePairs.byteLength;
        wgslOffload.state.lastPreparedNodeNarrowphaseSignature = compactNodePairs.signature;
      } else {
        wgslOffload.state.lastNodeNarrowphaseAuthoritativeSource = nodeProbeFinite
          ? 'cpu-rigid-soft-node-aabb-probe-filter-fallback'
          : 'cpu-rigid-soft-node-aabb-probe-nonfinite-fallback';
      }
    } else if (compactNodePairs) {
      wgslOffload.state.lastNodeNarrowphaseAuthoritativeSource = 'wgsl-rigid-soft-node-broadphase-zero-active';
    }

    const edgeReplayEpochDelta = broadphaseEpoch - (Number(wgslOffload.state.lastEdgeBroadphaseDispatchEpoch) || 0);
    const edgeReplayAvailable = fastBroadphaseMode
      && !cadenceDue
      && edgeReplayEpochDelta > 0
      && edgeReplayEpochDelta <= fastBroadphaseInterval
      && Number(wgslOffload.state.lastPreparedLayoutSignature) === (Number(wgslPrep.signature) || 0)
      && wgslOffload.state.lastPreparedEdgeNarrowphasePairRigidIndex instanceof Uint32Array
      && wgslOffload.state.lastPreparedEdgeNarrowphasePairSpringIndex instanceof Uint32Array
      && wgslOffload.state.lastPreparedEdgeNarrowphasePairNodeAIndex instanceof Uint32Array
      && wgslOffload.state.lastPreparedEdgeNarrowphasePairNodeBIndex instanceof Uint32Array;

    if (edgeReplayAvailable) {
      compactEdgePairs = {
        compactRigidIndex: wgslOffload.state.lastPreparedEdgeNarrowphasePairRigidIndex,
        compactSpringIndex: wgslOffload.state.lastPreparedEdgeNarrowphasePairSpringIndex,
        compactNodeAIndex: wgslOffload.state.lastPreparedEdgeNarrowphasePairNodeAIndex,
        compactNodeBIndex: wgslOffload.state.lastPreparedEdgeNarrowphasePairNodeBIndex,
        pairCount: Number(wgslOffload.state.lastPreparedEdgeNarrowphasePairCount) || 0,
        byteLength: Number(wgslOffload.state.lastPreparedEdgeNarrowphaseBytes) || 0,
        signature: Number(wgslOffload.state.lastPreparedEdgeNarrowphaseSignature) || 0,
      };
      wgslOffload.state.lastEdgeBroadphaseAuthoritativeSource = 'wgsl-rigid-soft-edge-broadphase-authoritative-replay-fast';
      wgslOffload.state.lastEdgeBroadphaseReadbackSourceRoute = 'wgsl-rigid-soft-edge-broadphase-replay-fast';
      wgslOffload.state.lastEdgeBroadphaseCandidatePairCount = Number(wgslPrep.edgePairCount) || 0;
      wgslOffload.state.lastEdgeBroadphaseActivePairCount = compactEdgePairs.pairCount | 0;
      wgslOffload.state.lastEdgeBroadphaseReadbackCount = compactEdgePairs.pairCount | 0;
      wgslOffload.state.lastEdgeBroadphaseReplayEpochDelta = edgeReplayEpochDelta;
      wgslOffload.state.lastEdgeBroadphaseReplayHitCount = (Number(wgslOffload.state.lastEdgeBroadphaseReplayHitCount) || 0) + 1;
    } else {
      const edgeBroadphaseStartMs = profile ? performance.now() : 0;
      wgslEdgeBroadphase = await dispatchRigidSoftEdgeBroadphaseWgsl({
        rigidBodies,
        soft,
        offload: wgslOffload,
        layout: wgslPrep,
        edgeSlop,
        rigidAabb,
      });
      if (profile) {
        addRigidSoftProfileMs(profile, 'edgeBroadphaseDispatchReadbackMs', performance.now() - edgeBroadphaseStartMs);
        const t = wgslOffload?.state?.lastEdgeBroadphaseTiming;
        if (t && typeof t === 'object') {
          addRigidSoftProfileMs(profile, 'edgeBroadphaseUploadMs', t.uploadMs);
          addRigidSoftProfileMs(profile, 'edgeBroadphaseDispatchSubmitMs', t.dispatchSubmitMs);
          addRigidSoftProfileMs(profile, 'edgeBroadphaseCountReadbackMs', t.countReadbackMs);
          addRigidSoftProfileMs(profile, 'edgeBroadphaseIndexReadbackMs', t.indexReadbackMs);
          addRigidSoftProfileMs(profile, 'edgeBroadphaseCompactBuildMs', t.compactBuildMs);
          addRigidSoftProfileMs(profile, 'edgeBroadphaseDetailedTotalMs', t.totalMs);
        }
      }
      if (wgslEdgeBroadphase?.compactPairs) {
        wgslOffload.state.lastEdgeBroadphaseAuthoritativeSource = 'wgsl-rigid-soft-edge-broadphase-authoritative-filter';
        wgslOffload.state.lastEdgeBroadphaseReadbackCount = Number(wgslEdgeBroadphase.activePairCount) || 0;
        wgslOffload.state.lastEdgeBroadphaseCandidatePairCount = Number(wgslEdgeBroadphase.candidatePairCount) || (wgslPrep.edgePairCount >>> 0);
        wgslOffload.state.lastEdgeBroadphaseDispatchEpoch = broadphaseEpoch;
        wgslOffload.state.lastEdgeBroadphaseDispatchHitCount = (Number(wgslOffload.state.lastEdgeBroadphaseDispatchHitCount) || 0) + 1;
        compactEdgePairs = wgslEdgeBroadphase.compactPairs;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairRigidIndex = compactEdgePairs.compactRigidIndex;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairSpringIndex = compactEdgePairs.compactSpringIndex;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairNodeAIndex = compactEdgePairs.compactNodeAIndex;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairNodeBIndex = compactEdgePairs.compactNodeBIndex;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairCount = compactEdgePairs.pairCount;
        wgslOffload.state.lastPreparedEdgeNarrowphaseBytes = compactEdgePairs.byteLength;
        wgslOffload.state.lastPreparedEdgeNarrowphaseSignature = compactEdgePairs.signature;
      } else {
        wgslOffload.state.lastEdgeBroadphaseAuthoritativeSource = 'cpu-rigid-soft-edge-broadphase-compact-fallback';
        wgslOffload.state.lastError = wgslOffload.state.lastEdgeBroadphaseReason || 'rigid-soft-edge-broadphase-readback-failed';
      }
    }

    if (profile) {
      profile.edgeBroadphaseCandidatePairs = Number(wgslOffload.state.lastEdgeBroadphaseCandidatePairCount) || 0;
      profile.edgeBroadphaseActivePairs = Number(wgslOffload.state.lastEdgeBroadphaseActivePairCount) || 0;
    }

    if (compactEdgePairs && compactEdgePairs.pairCount > 0) {
      const edgeProbeStartMs = profile ? performance.now() : 0;
      await dispatchRigidSoftEdgeNarrowphaseAabbProbeWgsl({
        rigidBodies,
        soft,
        offload: wgslOffload,
        edgePairs: compactEdgePairs,
        edgeSlop,
        rigidAabb,
      });
      if (profile) {
        addRigidSoftProfileMs(profile, 'edgeNarrowphaseAabbProbeDispatchReadbackMs', performance.now() - edgeProbeStartMs);
        const t = wgslOffload?.state?.lastEdgeNarrowphaseAabbProbeTiming;
        if (t && typeof t === 'object') {
          addRigidSoftProfileMs(profile, 'edgeNarrowphaseAabbProbeInputBuildMs', t.inputBuildMs);
          addRigidSoftProfileMs(profile, 'edgeNarrowphaseAabbProbeUploadMs', t.uploadMs);
          addRigidSoftProfileMs(profile, 'edgeNarrowphaseAabbProbeDispatchSubmitMs', t.dispatchSubmitMs);
          addRigidSoftProfileMs(profile, 'edgeNarrowphaseAabbProbeReadbackMapMs', t.readbackMapMs);
          addRigidSoftProfileMs(profile, 'edgeNarrowphaseAabbProbeDecodeMs', t.decodeMs);
          addRigidSoftProfileMs(profile, 'edgeNarrowphaseAabbProbeDetailedTotalMs', t.totalMs);
        }
      }
    } else if (compactEdgePairs) {
      wgslOffload.state.lastEdgeNarrowphaseAabbProbeSource = 'wgsl-rigid-soft-edge-broadphase-zero-active';
    }

    const narrowphaseLayoutStartMs = profile ? performance.now() : 0;
    const narrowphaseLayout = buildRigidSoftNarrowphaseWgslLayout({
      nodePairs: compactNodePairs || {
        compactRigidIndex: new Uint32Array(0),
        compactNodeIndex: new Uint32Array(0),
      },
      edgePairs: compactEdgePairs || {
        compactRigidIndex: new Uint32Array(0),
        compactSpringIndex: new Uint32Array(0),
        compactNodeAIndex: new Uint32Array(0),
        compactNodeBIndex: new Uint32Array(0),
      },
    });
    if (profile) addRigidSoftProfileMs(profile, 'narrowphasePairLayoutBuildMs', performance.now() - narrowphaseLayoutStartMs);
    if (narrowphaseLayout) {
      wgslOffload.state.lastPreparedNarrowphasePairCount =
        (Number(narrowphaseLayout.nodePairCount) || 0) + (Number(narrowphaseLayout.edgePairCount) || 0);
      wgslOffload.state.lastPreparedNarrowphaseLayoutBytes = narrowphaseLayout.byteLength;
      wgslOffload.state.lastPreparedNarrowphaseCombinedSignature = narrowphaseLayout.signature;
    }

    const narrowphaseSceneStartMs = profile ? performance.now() : 0;
    const narrowphaseSceneLayout = buildRigidSoftNarrowphaseSceneWgslLayout({ rigidBodies, soft });
    if (profile) addRigidSoftProfileMs(profile, 'narrowphaseSceneLayoutBuildMs', performance.now() - narrowphaseSceneStartMs);
    if (narrowphaseSceneLayout) {
      wgslOffload.state.lastPreparedNarrowphaseSceneLayoutBytes = narrowphaseSceneLayout.byteLength;
      wgslOffload.state.lastPreparedNarrowphaseSceneSignature = narrowphaseSceneLayout.signature;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidX = narrowphaseSceneLayout.rigidX;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidY = narrowphaseSceneLayout.rigidY;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidTheta = narrowphaseSceneLayout.rigidTheta;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidVx = narrowphaseSceneLayout.rigidVx;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidVy = narrowphaseSceneLayout.rigidVy;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidOmega = narrowphaseSceneLayout.rigidOmega;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidInvMass = narrowphaseSceneLayout.rigidInvMass;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidInvInertia = narrowphaseSceneLayout.rigidInvInertia;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidMinX = narrowphaseSceneLayout.rigidMinX;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidMinY = narrowphaseSceneLayout.rigidMinY;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidMaxX = narrowphaseSceneLayout.rigidMaxX;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidMaxY = narrowphaseSceneLayout.rigidMaxY;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidVertexStart = narrowphaseSceneLayout.rigidVertexStart;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidVertexX = narrowphaseSceneLayout.rigidVertexX;
      wgslOffload.state.lastPreparedNarrowphaseSceneRigidVertexY = narrowphaseSceneLayout.rigidVertexY;
      wgslOffload.state.lastPreparedNarrowphaseSceneNodeX = narrowphaseSceneLayout.nodeX;
      wgslOffload.state.lastPreparedNarrowphaseSceneNodeY = narrowphaseSceneLayout.nodeY;
      wgslOffload.state.lastPreparedNarrowphaseSceneNodeVx = narrowphaseSceneLayout.nodeVx;
      wgslOffload.state.lastPreparedNarrowphaseSceneNodeVy = narrowphaseSceneLayout.nodeVy;
      wgslOffload.state.lastPreparedNarrowphaseSceneNodeR = narrowphaseSceneLayout.nodeR;
      wgslOffload.state.lastPreparedNarrowphaseSceneNodeInvMass = narrowphaseSceneLayout.nodeInvMass;
      wgslOffload.state.lastPreparedNarrowphaseSceneSpringNodeA = narrowphaseSceneLayout.springNodeA;
      wgslOffload.state.lastPreparedNarrowphaseSceneSpringNodeB = narrowphaseSceneLayout.springNodeB;
      wgslOffload.state.lastPreparedNarrowphaseSceneSpringRestLen = narrowphaseSceneLayout.springRestLen;
      wgslOffload.state.lastPreparedNarrowphaseSceneSource = 'cpu-rigid-soft-narrowphase-scene-layout';
    }

    const impulseSeedLayoutStartMs = profile ? performance.now() : 0;
    const narrowphaseImpulseSeedLayout = buildRigidSoftNarrowphaseImpulseSeedLayout({
      narrowphaseLayout,
      narrowphaseSceneLayout,
    });
    if (profile) addRigidSoftProfileMs(profile, 'impulseSeedLayoutBuildMs', performance.now() - impulseSeedLayoutStartMs);
    if (narrowphaseImpulseSeedLayout) {
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedLayoutBytes = narrowphaseImpulseSeedLayout.byteLength;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedSignature = narrowphaseImpulseSeedLayout.signature;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodePairCount = narrowphaseImpulseSeedLayout.nodePairCount;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairCount = narrowphaseImpulseSeedLayout.edgePairCount;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodePairRigidIndex = narrowphaseImpulseSeedLayout.nodePairRigidIndex;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodePairNodeIndex = narrowphaseImpulseSeedLayout.nodePairNodeIndex;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairRigidIndex = narrowphaseImpulseSeedLayout.edgePairRigidIndex;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairSpringIndex = narrowphaseImpulseSeedLayout.edgePairSpringIndex;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairNodeAIndex = narrowphaseImpulseSeedLayout.edgePairNodeAIndex;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairNodeBIndex = narrowphaseImpulseSeedLayout.edgePairNodeBIndex;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedRigidInvMass = narrowphaseImpulseSeedLayout.rigidInvMass;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedRigidInvInertia = narrowphaseImpulseSeedLayout.rigidInvInertia;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeX = narrowphaseImpulseSeedLayout.nodeX;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeY = narrowphaseImpulseSeedLayout.nodeY;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeVx = narrowphaseImpulseSeedLayout.nodeVx;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeVy = narrowphaseImpulseSeedLayout.nodeVy;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeR = narrowphaseImpulseSeedLayout.nodeR;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeInvMass = narrowphaseImpulseSeedLayout.nodeInvMass;
      wgslOffload.state.lastPreparedNarrowphaseImpulseSeedSource = 'cpu-rigid-soft-narrowphase-impulse-seed-layout';
    }
    }
  }

  const impulseSeed = wgslOffload?.state
    ? {
      nodePairRigidIndex: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodePairRigidIndex || new Uint32Array(0),
      nodePairNodeIndex: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodePairNodeIndex || new Uint32Array(0),
      edgePairRigidIndex: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairRigidIndex || new Uint32Array(0),
      edgePairNodeAIndex: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairNodeAIndex || new Uint32Array(0),
      edgePairNodeBIndex: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairNodeBIndex || new Uint32Array(0),
      rigidX: wgslOffload.state.lastPreparedNarrowphaseSceneRigidX || new Float32Array(0),
      rigidY: wgslOffload.state.lastPreparedNarrowphaseSceneRigidY || new Float32Array(0),
      rigidVx: wgslOffload.state.lastPreparedNarrowphaseSceneRigidVx || new Float32Array(0),
      rigidVy: wgslOffload.state.lastPreparedNarrowphaseSceneRigidVy || new Float32Array(0),
      rigidOmega: wgslOffload.state.lastPreparedNarrowphaseSceneRigidOmega || new Float32Array(0),
      rigidInvMass: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedRigidInvMass || new Float32Array(0),
      rigidInvInertia: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedRigidInvInertia || new Float32Array(0),
      rigidVertexStart: wgslOffload.state.lastPreparedNarrowphaseSceneRigidVertexStart || new Uint32Array(0),
      rigidVertexX: wgslOffload.state.lastPreparedNarrowphaseSceneRigidVertexX || new Float32Array(0),
      rigidVertexY: wgslOffload.state.lastPreparedNarrowphaseSceneRigidVertexY || new Float32Array(0),
      nodeX: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeX || new Float32Array(0),
      nodeY: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeY || new Float32Array(0),
      nodeVx: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeVx || new Float32Array(0),
      nodeVy: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeVy || new Float32Array(0),
      nodeR: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeR || new Float32Array(0),
      nodeInvMass: wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodeInvMass || new Float32Array(0),
      nodePairCount: Number(wgslOffload.state.lastPreparedNarrowphaseImpulseSeedNodePairCount) || 0,
      edgePairCount: Number(wgslOffload.state.lastPreparedNarrowphaseImpulseSeedEdgePairCount) || 0,
      signature: Number(wgslOffload.state.lastPreparedNarrowphaseImpulseSeedSignature) || 0,
    }
    : null;

  const totalImpulsePairCount = (Number(impulseSeed?.nodePairCount) || 0) + (Number(impulseSeed?.edgePairCount) || 0);
  if (profile) {
    profile.impulseNodePairCount = Number(impulseSeed?.nodePairCount) || 0;
    profile.impulseEdgePairCount = Number(impulseSeed?.edgePairCount) || 0;
    profile.totalImpulsePairCount = totalImpulsePairCount;
    if (wgslOffload?.state) {
      profile.lastNodeBroadphaseReadbackSourceRoute = wgslOffload.state.lastNodeBroadphaseReadbackSourceRoute || null;
      profile.lastEdgeBroadphaseReadbackSourceRoute = wgslOffload.state.lastEdgeBroadphaseReadbackSourceRoute || null;
      profile.lastRigidSoftResponseRoute = wgslOffload.state.lastRigidSoftResponseRoute || null;
      profile.lastRigidSoftResponseFallbackReason = wgslOffload.state.lastRigidSoftResponseFallbackReason || null;
    }
  }
  if (wgslOffload?.state && wgslOffloadAvailable && totalImpulsePairCount === 0) {
    wgslOffload.state.lastNodeCollisionResponseSource = 'wgsl-rigid-soft-node-response-authoritative-noop';
    wgslOffload.state.lastEdgeCollisionResponseSource = 'wgsl-rigid-soft-edge-response-authoritative-noop';
    wgslOffload.state.lastNodeCollisionResponsePairCount = 0;
    wgslOffload.state.lastEdgeCollisionResponsePairCount = 0;
    wgslOffload.state.lastRigidSoftResponseAuthoritativeSource = 'wgsl-rigid-soft-response-authoritative-noop';
    wgslOffload.state.lastRigidSoftResponseOwnership = 'wgsl-authoritative';
    wgslOffload.state.lastRigidSoftResponseOwnershipDetailed = 'wgsl-authoritative:noop-zero-pairs';
    wgslOffload.state.lastRigidSoftResponseRoute = 'wgsl-authoritative-noop-zero-pairs';
    wgslOffload.state.lastRigidSoftResponseFallbackReason = null;
    wgslOffload.state.lastSourceRoute = 'wgsl-rigid-soft-response-authoritative-noop';
    wgslOffload.state.lastMode = 'wgsl-rigid-soft-response-authoritative-noop';
    if (profile) {
      profile.route = 'wgsl-authoritative';
      profile.responseOwnership = 'wgsl-authoritative';
      profile.fallbackReason = null;
      profile.usedWgslAuthoritativeResponse = true;
    }
    return;
  }

  let usedWgslAuthoritativeResponse = false;
  const applyAuthoritativeResponseProposal = ({ proposal, sourceTag, routeTag }) => {
    applyRigidSoftAuthoritativeProposal({ rigidBodies, soft, impulseSeed, proposal });
    if (sourceTag === 'authoritative') {
      wgslOffload.state.lastNodeCollisionResponseSource = 'wgsl-rigid-soft-node-response-authoritative';
      wgslOffload.state.lastEdgeCollisionResponseSource = 'wgsl-rigid-soft-edge-response-authoritative';
      wgslOffload.state.lastRigidSoftResponseAuthoritativeSource = 'wgsl-rigid-soft-response-authoritative';
    } else {
      wgslOffload.state.lastNodeCollisionResponseSource = `wgsl-rigid-soft-node-response-${sourceTag}`;
      wgslOffload.state.lastEdgeCollisionResponseSource = `wgsl-rigid-soft-edge-response-${sourceTag}`;
      wgslOffload.state.lastRigidSoftResponseAuthoritativeSource = `wgsl-rigid-soft-response-${sourceTag}`;
    }
    wgslOffload.state.lastNodeCollisionResponsePairCount = impulseSeed.nodePairCount;
    wgslOffload.state.lastEdgeCollisionResponsePairCount = impulseSeed.edgePairCount;
    wgslOffload.state.lastRigidSoftResponseOwnership = 'wgsl-authoritative';
    wgslOffload.state.lastRigidSoftResponseOwnershipDetailed = `wgsl-authoritative:${sourceTag}`;
    wgslOffload.state.lastRigidSoftResponseRoute = routeTag;
    wgslOffload.state.lastRigidSoftResponseFallbackReason = null;
    wgslOffload.state.lastSourceRoute = `wgsl-rigid-soft-response-${sourceTag}`;
    wgslOffload.state.lastMode = `wgsl-rigid-soft-response-${sourceTag}`;
    if (profile) {
      profile.route = 'wgsl-authoritative';
      profile.responseOwnership = 'wgsl-authoritative';
      profile.fallbackReason = null;
      profile.usedWgslAuthoritativeResponse = true;
    }
    usedWgslAuthoritativeResponse = true;
  };

  if (wgslOffload?.state && impulseSeed && wgslOffloadAvailable) {
    const signature = Number(wgslOffload.state.lastPreparedNarrowphaseImpulseSeedSignature) || 0;
    try {
      const responseDispatchStartMs = profile ? performance.now() : 0;
      const proposal = await dispatchRigidSoftResponseWgsl({
        offload: wgslOffload,
        impulseSeed,
        edgeSlop,
      });
      if (profile) {
        addRigidSoftProfileMs(profile, 'responseDispatchReadbackMs', performance.now() - responseDispatchStartMs);
        const t = wgslOffload?.state?.lastRigidSoftResponseTiming;
        if (t && typeof t === 'object') {
          addRigidSoftProfileMs(profile, 'responseNodeBuildMs', t.nodeBuildMs);
          addRigidSoftProfileMs(profile, 'responseNodeDispatchSubmitMs', t.nodeDispatchSubmitMs);
          addRigidSoftProfileMs(profile, 'responseNodeReadbackMs', t.nodeReadbackMs);
          addRigidSoftProfileMs(profile, 'responseNodeDecodeMs', t.nodeDecodeMs);
          addRigidSoftProfileMs(profile, 'responseEdgeBuildMs', t.edgeBuildMs);
          addRigidSoftProfileMs(profile, 'responseEdgeDispatchSubmitMs', t.edgeDispatchSubmitMs);
          addRigidSoftProfileMs(profile, 'responseEdgeReadbackMs', t.edgeReadbackMs);
          addRigidSoftProfileMs(profile, 'responseEdgeDecodeMs', t.edgeDecodeMs);
          addRigidSoftProfileMs(profile, 'responseDetailedTotalMs', t.totalMs);
        }
      }
      const validationStartMs = profile ? performance.now() : 0;
      const validationOut = { reason: null };
      if (canApplyRigidSoftAuthoritativeProposal({ proposal, impulseSeed, signature, out: validationOut })) {
        applyAuthoritativeResponseProposal({
          proposal,
          sourceTag: 'authoritative',
          routeTag: 'wgsl-authoritative',
        });
        wgslOffload.state.lastGoodRigidSoftResponseProposal = proposal;
        wgslOffload.state.lastGoodRigidSoftResponseSignature = signature >>> 0;
      } else {
        const fallbackReason = validationOut.reason || 'proposal-validation-failed';
        const cachedProposal = wgslOffload.state.lastGoodRigidSoftResponseProposal;
        const cachedSignature = Number(wgslOffload.state.lastGoodRigidSoftResponseSignature) >>> 0;
        const replayValidationOut = { reason: null };
        if (
          cachedProposal
          && cachedSignature === (signature >>> 0)
          && canApplyRigidSoftAuthoritativeProposal({
            proposal: cachedProposal,
            impulseSeed,
            signature,
            out: replayValidationOut,
          })
        ) {
          applyAuthoritativeResponseProposal({
            proposal: cachedProposal,
            sourceTag: 'authoritative-replay',
            routeTag: 'wgsl-authoritative-replay',
          });
          wgslOffload.state.lastRigidSoftResponseReplayReason = fallbackReason;
        } else {
          const replayReason = replayValidationOut.reason || 'cached-proposal-invalid';
          const combinedReason = `${fallbackReason}|${replayReason}`;
          wgslOffload.state.lastRigidSoftResponseAuthoritativeSource = 'cpu-rigid-soft-response-fallback-nonfinite';
          wgslOffload.state.lastRigidSoftResponseOwnership = 'cpu-fallback';
          wgslOffload.state.lastRigidSoftResponseOwnershipDetailed = `cpu-hard-fallback:${combinedReason}`;
          wgslOffload.state.lastRigidSoftResponseRoute = 'cpu-fallback';
          wgslOffload.state.lastRigidSoftResponseFallbackReason = combinedReason;
          if (profile) {
            profile.route = 'cpu-fallback';
            profile.responseOwnership = 'cpu-hard-fallback';
            profile.fallbackReason = combinedReason;
          }
        }
      }
      if (profile) addRigidSoftProfileMs(profile, 'responseValidationApplyMs', performance.now() - validationStartMs);
    } catch (err) {
      wgslOffload.state.lastError = String(err?.message || err || 'rigid-soft-response-wgsl-error');
      wgslOffload.state.lastRigidSoftResponseAuthoritativeSource = 'cpu-rigid-soft-response-fallback-error';
      wgslOffload.state.lastRigidSoftResponseOwnership = 'cpu-fallback';
      wgslOffload.state.lastRigidSoftResponseOwnershipDetailed = 'cpu-hard-fallback:wgsl-dispatch-error';
      wgslOffload.state.lastRigidSoftResponseRoute = 'cpu-fallback';
      wgslOffload.state.lastRigidSoftResponseFallbackReason = 'wgsl-dispatch-error';
      if (profile) {
        profile.route = 'cpu-fallback';
        profile.responseOwnership = 'cpu-hard-fallback';
        profile.fallbackReason = 'wgsl-dispatch-error';
      }
    }
  }

  if (usedWgslAuthoritativeResponse) return;

  if (wgslOffload?.state && wgslOffloadAvailable) {
    if (!wgslOffload.state.lastRigidSoftResponseAuthoritativeSource) {
      wgslOffload.state.lastRigidSoftResponseAuthoritativeSource = 'cpu-rigid-soft-response-hard-fallback-prep';
    }
    if (!wgslOffload.state.lastRigidSoftResponseFallbackReason) {
      wgslOffload.state.lastRigidSoftResponseFallbackReason = 'hard-fallback';
    }
    wgslOffload.state.lastRigidSoftResponseOwnership = 'cpu-fallback';
    wgslOffload.state.lastRigidSoftResponseOwnershipDetailed = `cpu-hard-fallback:${wgslOffload.state.lastRigidSoftResponseFallbackReason || 'hard-fallback'}`;
    wgslOffload.state.lastRigidSoftResponseRoute = 'cpu-fallback';
    wgslOffload.state.lastSourceRoute = 'cpu-rigid-soft-response-hard-fallback';
    wgslOffload.state.lastMode = 'cpu-rigid-soft-response-hard-fallback';
    if (profile && !profile.fallbackReason) {
      profile.route = 'cpu-fallback';
      profile.responseOwnership = 'cpu-hard-fallback';
      profile.fallbackReason = wgslOffload.state.lastRigidSoftResponseFallbackReason || 'hard-fallback';
    }
  }

  if (profile && !profile.fallbackReason) {
    profile.route = 'cpu-fallback';
    profile.responseOwnership = 'cpu-hard-fallback';
    profile.fallbackReason = wgslOffloadAvailable
      ? 'hard-fallback'
      : (wgslUnavailableReason || 'wgsl-offload-unavailable');
  }

  applyCpuRigidSoftResponseFallback({
    rigidBodies,
    soft,
    compactNodePairs,
    compactEdgePairs,
    wgslNodeBroadphaseMask: null,
    wgslEdgeBroadphaseMask: null,
    edgeBodyModeBlock,
    nodeSlop,
    edgeSlop,
    wgslOffload,
    profile,
  });
}
