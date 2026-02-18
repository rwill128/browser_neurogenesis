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
@group(0) @binding(10) var<storage, read_write> activeMaskOut: array<u32>;

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

  let active = select(0u, 1u, x >= minX && x <= maxX && y >= minY && y <= maxY);
  activeMaskOut[pairIndex] = active;
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
@group(0) @binding(10) var<storage, read_write> activeMaskOut: array<u32>;

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
  activeMaskOut[pairIndex] = select(0u, 1u, overlaps);
}
`;

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined') return false;
  return true;
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
    const bytes = cap * 4;
    state.rigidSoftNodeBroadphasePairRigidIndex?.destroy?.();
    state.rigidSoftNodeBroadphasePairNodeIndex?.destroy?.();
    state.rigidSoftNodeBroadphaseActiveMaskOut?.destroy?.();
    state.rigidSoftNodeBroadphaseActiveMaskReadback?.destroy?.();
    state.rigidSoftNodeBroadphasePairRigidIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphasePairNodeIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftNodeBroadphaseActiveMaskOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.rigidSoftNodeBroadphaseActiveMaskReadback = device.createBuffer({
      size: bytes,
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
    const bytes = cap * 4;
    state.rigidSoftEdgeBroadphasePairRigidIndex?.destroy?.();
    state.rigidSoftEdgeBroadphasePairNodeAIndex?.destroy?.();
    state.rigidSoftEdgeBroadphasePairNodeBIndex?.destroy?.();
    state.rigidSoftEdgeBroadphaseActiveMaskOut?.destroy?.();
    state.rigidSoftEdgeBroadphaseActiveMaskReadback?.destroy?.();
    state.rigidSoftEdgeBroadphasePairRigidIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphasePairNodeAIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphasePairNodeBIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.rigidSoftEdgeBroadphaseActiveMaskOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.rigidSoftEdgeBroadphaseActiveMaskReadback = device.createBuffer({
      size: bytes,
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

  const pipeline = state.rigidSoftNodeBroadphasePipeline;
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

    const dispatchCount = Math.ceil(pairCount / WGSL_WORKGROUP_SIZE);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, state.rigidSoftNodeBroadphaseBindGroup);
    pass.dispatchWorkgroups(dispatchCount);
    pass.end();
    encoder.copyBufferToBuffer(
      state.rigidSoftNodeBroadphaseActiveMaskOut,
      0,
      state.rigidSoftNodeBroadphaseActiveMaskReadback,
      0,
      pairCount * 4,
    );
    device.queue.submit([encoder.finish()]);

    state.lastNodeBroadphaseDispatched = true;
    state.lastNodeBroadphaseReason = null;
    state.lastNodeBroadphaseDispatch = dispatchCount;
    state.lastNodeBroadphasePairCount = pairCount;
    state.lastNodeBroadphaseSourceRoute = 'wgsl-rigid-soft-node-broadphase-proposal';

    const readbackBytes = pairCount * 4;
    safeUnmapBuffer(state.rigidSoftNodeBroadphaseActiveMaskReadback);

    try {
      await state.rigidSoftNodeBroadphaseActiveMaskReadback.mapAsync(globalThis.GPUMapMode.READ, 0, readbackBytes);
      const mapped = state.rigidSoftNodeBroadphaseActiveMaskReadback.getMappedRange(0, readbackBytes);
      const activeMask = new Uint32Array(mapped.slice(0));
      state.lastNodeBroadphaseReadbackSourceRoute = 'wgsl-rigid-soft-node-broadphase-readback';
      return activeMask;
    } catch (err) {
      state.lastNodeBroadphaseReason = String(err?.message || err || 'readback-failed');
      state.lastNodeBroadphaseReadbackSourceRoute = 'cpu-rigid-soft-node-broadphase-readback-fallback';
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

  const pipeline = state.rigidSoftEdgeBroadphasePipeline;
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

    const dispatchCount = Math.ceil(pairCount / WGSL_WORKGROUP_SIZE);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, state.rigidSoftEdgeBroadphaseBindGroup);
    pass.dispatchWorkgroups(dispatchCount);
    pass.end();
    encoder.copyBufferToBuffer(
      state.rigidSoftEdgeBroadphaseActiveMaskOut,
      0,
      state.rigidSoftEdgeBroadphaseActiveMaskReadback,
      0,
      pairCount * 4,
    );
    device.queue.submit([encoder.finish()]);

    state.lastEdgeBroadphaseDispatched = true;
    state.lastEdgeBroadphaseReason = null;
    state.lastEdgeBroadphaseDispatch = dispatchCount;
    state.lastEdgeBroadphasePairCount = pairCount;
    state.lastEdgeBroadphaseSourceRoute = 'wgsl-rigid-soft-edge-broadphase-proposal';

    const readbackBytes = pairCount * 4;
    safeUnmapBuffer(state.rigidSoftEdgeBroadphaseActiveMaskReadback);

    try {
      await state.rigidSoftEdgeBroadphaseActiveMaskReadback.mapAsync(globalThis.GPUMapMode.READ, 0, readbackBytes);
      const mapped = state.rigidSoftEdgeBroadphaseActiveMaskReadback.getMappedRange(0, readbackBytes);
      const activeMask = new Uint32Array(mapped.slice(0));
      state.lastEdgeBroadphaseReadbackSourceRoute = 'wgsl-rigid-soft-edge-broadphase-readback';
      return activeMask;
    } catch (err) {
      state.lastEdgeBroadphaseReason = String(err?.message || err || 'readback-failed');
      state.lastEdgeBroadphaseReadbackSourceRoute = 'cpu-rigid-soft-edge-broadphase-readback-fallback';
      return null;
    } finally {
      safeUnmapBuffer(state.rigidSoftEdgeBroadphaseActiveMaskReadback);
    }
  });
}

export async function resolveRigidSoftCollisionPassGpuOnly({
  rigidBodies,
  soft,
  resolveRigidVsSoftNodeCollision = resolveRigidVsSoftNodeCollisionGpuOnly,
  resolveRigidVsSoftEdgeCollision = resolveRigidVsSoftEdgeCollisionGpuOnly,
  edgeBodyModeBlock,
  nodeSlop = 0.18,
  edgeSlop = 0.16,
  wgslOffload,
}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return;
  if (!soft || !Array.isArray(soft.nodes) || !Array.isArray(soft.springs)) return;

  let wgslPrep = null;
  let wgslNodeBroadphaseMask = null;
  let wgslEdgeBroadphaseMask = null;
  let compactNodePairs = null;
  let compactEdgePairs = null;
  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    wgslPrep = buildRigidSoftCollisionWgslLayout({
      rigidBodies,
      soft,
      edgeBodyModeBlock,
    });
    const wgslNarrowphaseGeometry = buildRigidSoftNarrowphaseGeometryLayout(rigidBodies);
    wgslOffload.state.preparedLayout = wgslPrep.layout;
    wgslOffload.state.lastPreparedNodePairCount = wgslPrep.nodePairCount;
    wgslOffload.state.lastPreparedEdgePairCount = wgslPrep.edgePairCount;
    wgslOffload.state.lastPreparedLayoutBytes = wgslPrep.byteLength;
    wgslOffload.state.lastPreparedLayoutSignature = wgslPrep.signature;
    wgslOffload.state.lastPreparedNarrowphaseGeometry = wgslNarrowphaseGeometry.layout;
    wgslOffload.state.lastPreparedNarrowphaseRigidCount = wgslNarrowphaseGeometry.rigidCount;
    wgslOffload.state.lastPreparedNarrowphaseEdgeCount = wgslNarrowphaseGeometry.edgeCount;
    wgslOffload.state.lastPreparedNarrowphaseBytes = wgslNarrowphaseGeometry.byteLength;
    wgslOffload.state.lastPreparedNarrowphaseSignature = wgslNarrowphaseGeometry.signature;
    wgslOffload.state.lastSourceRoute = 'cpu-rigid-soft-candidate-layout';
    wgslOffload.state.lastMode = 'cpu-prepared';
    wgslOffload.state.lastError = null;

    const rigidAabb = computeRigidBodyAabbs(rigidBodies);

    wgslNodeBroadphaseMask = await dispatchRigidSoftNodeBroadphaseWgsl({
      rigidBodies,
      soft,
      offload: wgslOffload,
      layout: wgslPrep,
      rigidAabb,
    });
    if (wgslNodeBroadphaseMask && wgslNodeBroadphaseMask.length === (wgslPrep.nodePairCount >>> 0)) {
      wgslOffload.state.lastSourceRoute = 'wgsl-rigid-soft-node-broadphase-authoritative-filter';
      wgslOffload.state.lastMode = 'wgsl-broadphase-authoritative-filter';
      wgslOffload.state.lastNodeBroadphaseReadbackCount = wgslNodeBroadphaseMask.length;
      compactNodePairs = buildActiveRigidSoftNodeNarrowphasePairs({ prep: wgslPrep, activeMask: wgslNodeBroadphaseMask });
      if (compactNodePairs) {
        wgslOffload.state.lastPreparedNodeNarrowphasePairRigidIndex = compactNodePairs.compactRigidIndex;
        wgslOffload.state.lastPreparedNodeNarrowphasePairNodeIndex = compactNodePairs.compactNodeIndex;
        wgslOffload.state.lastPreparedNodeNarrowphasePairCount = compactNodePairs.pairCount;
        wgslOffload.state.lastPreparedNodeNarrowphaseBytes = compactNodePairs.byteLength;
        wgslOffload.state.lastPreparedNodeNarrowphaseSignature = compactNodePairs.signature;
      }
    } else if (wgslNodeBroadphaseMask) {
      wgslOffload.state.lastSourceRoute = 'cpu-rigid-soft-node-broadphase-mask-fallback';
      wgslOffload.state.lastMode = 'cpu-authoritative-mask-fallback';
      wgslOffload.state.lastError = 'rigid-soft-broadphase-mask-size-mismatch';
      wgslNodeBroadphaseMask = null;
    }

    wgslEdgeBroadphaseMask = await dispatchRigidSoftEdgeBroadphaseWgsl({
      rigidBodies,
      soft,
      offload: wgslOffload,
      layout: wgslPrep,
      edgeSlop,
      rigidAabb,
    });
    if (wgslEdgeBroadphaseMask && wgslEdgeBroadphaseMask.length === (wgslPrep.edgePairCount >>> 0)) {
      wgslOffload.state.lastEdgeBroadphaseAuthoritativeSource = 'wgsl-rigid-soft-edge-broadphase-authoritative-filter';
      wgslOffload.state.lastEdgeBroadphaseReadbackCount = wgslEdgeBroadphaseMask.length;
      compactEdgePairs = buildActiveRigidSoftEdgeNarrowphasePairs({ prep: wgslPrep, activeMask: wgslEdgeBroadphaseMask });
      if (compactEdgePairs) {
        wgslOffload.state.lastPreparedEdgeNarrowphasePairRigidIndex = compactEdgePairs.compactRigidIndex;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairSpringIndex = compactEdgePairs.compactSpringIndex;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairNodeAIndex = compactEdgePairs.compactNodeAIndex;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairNodeBIndex = compactEdgePairs.compactNodeBIndex;
        wgslOffload.state.lastPreparedEdgeNarrowphasePairCount = compactEdgePairs.pairCount;
        wgslOffload.state.lastPreparedEdgeNarrowphaseBytes = compactEdgePairs.byteLength;
        wgslOffload.state.lastPreparedEdgeNarrowphaseSignature = compactEdgePairs.signature;
      }
    } else if (wgslEdgeBroadphaseMask) {
      wgslOffload.state.lastEdgeBroadphaseAuthoritativeSource = 'cpu-rigid-soft-edge-broadphase-mask-fallback';
      wgslOffload.state.lastError = 'rigid-soft-edge-broadphase-mask-size-mismatch';
      wgslEdgeBroadphaseMask = null;
    }

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
    if (narrowphaseLayout) {
      wgslOffload.state.lastPreparedNarrowphasePairCount =
        (Number(narrowphaseLayout.nodePairCount) || 0) + (Number(narrowphaseLayout.edgePairCount) || 0);
      wgslOffload.state.lastPreparedNarrowphaseLayoutBytes = narrowphaseLayout.byteLength;
      wgslOffload.state.lastPreparedNarrowphaseCombinedSignature = narrowphaseLayout.signature;
    }
  }

  let nodePairCursor = 0;
  let edgePairCursor = 0;
  for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
    const rb = rigidBodies[rbi];

    for (let ni = 0; ni < soft.nodes.length; ni++) {
      const broadphaseActive = wgslNodeBroadphaseMask
        ? (wgslNodeBroadphaseMask[nodePairCursor] === 1)
        : true;
      nodePairCursor++;
      if (!broadphaseActive) continue;
      const sn = soft.nodes[ni];
      resolveRigidVsSoftNodeCollision(rb, sn, null, nodeSlop);
    }

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
      resolveRigidVsSoftEdgeCollision(rb, soft.nodes[i], soft.nodes[j], edgeSlop);
    }
  }
}
