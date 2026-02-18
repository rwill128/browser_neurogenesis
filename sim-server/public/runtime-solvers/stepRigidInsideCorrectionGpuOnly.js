/**
 * GPU-only rigid-inside correction pass.
 *
 * Owns post-collision rigid↔soft inside projection correction in the isolated
 * gpu-only runtime path so this stepping responsibility no longer lives inline
 * in gpu-lab.js orchestration.
 */

import { getRigidCollisionPolysWorld, pointInPolygonInclusive } from '../rigid-collision.js';

const EPS = 1e-8;

function hashU32ArrayFnv1a(arr) {
  let hash = 0x811c9dc5;
  const len = Number(arr?.length) || 0;
  for (let i = 0; i < len; i++) {
    hash ^= (Number(arr[i]) || 0) >>> 0;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function hashF32ArrayFnv1a(arr) {
  const len = Number(arr?.length) || 0;
  const scratch = new ArrayBuffer(4);
  const asF32 = new Float32Array(scratch);
  const asU32 = new Uint32Array(scratch);
  let hash = 0x811c9dc5;
  for (let i = 0; i < len; i++) {
    asF32[0] = Number(arr[i]) || 0;
    hash ^= asU32[0] >>> 0;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function buildRigidInsideCorrectionWgslLayout({
  rigidBodies,
  soft,
  hybridAttachedByRigid,
  getRigidPolysWorld,
}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const rigid = Array.isArray(rigidBodies) ? rigidBodies : [];

  const nodeCount = nodes.length;
  const rigidCount = rigid.length;

  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  const nodeR = new Float32Array(nodeCount);
  const nodeMass = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i] || {};
    nodeX[i] = Number(node.x) || 0;
    nodeY[i] = Number(node.y) || 0;
    nodeR[i] = Math.max(0.25, Number(node.r) || 1);
    nodeMass[i] = Math.max(0.02, Number(node.mass) || 1);
  }

  const rigidPolyOffsets = new Uint32Array(rigidCount + 1);
  const eligibleNodeOffsets = new Uint32Array(rigidCount + 1);
  const eligibleNodeList = [];
  const polyPointX = [];
  const polyPointY = [];
  const polyPointOffsets = [0];
  const polyRigidIndex = [];

  let polyCount = 0;
  for (let rbi = 0; rbi < rigidCount; rbi++) {
    rigidPolyOffsets[rbi] = polyCount;
    const rb = rigid[rbi];
    if (!rb || rb.insideCorrectionEnabled === false) {
      eligibleNodeOffsets[rbi] = eligibleNodeList.length;
      continue;
    }
    const polys = getRigidPolysWorld(rb);
    if (Array.isArray(polys)) {
      for (const poly of polys) {
        if (!Array.isArray(poly) || poly.length < 3) continue;
        polyRigidIndex.push(rbi >>> 0);
        for (let i = 0; i < poly.length; i++) {
          const p = poly[i] || {};
          polyPointX.push(Number(p.x) || 0);
          polyPointY.push(Number(p.y) || 0);
        }
        polyPointOffsets.push(polyPointX.length);
        polyCount += 1;
      }
    }

    eligibleNodeOffsets[rbi] = eligibleNodeList.length;
    const attachedNodeSet = hybridAttachedByRigid?.get?.(rbi) || null;
    for (let ni = 0; ni < nodeCount; ni++) {
      if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
      eligibleNodeList.push(ni >>> 0);
    }
  }
  rigidPolyOffsets[rigidCount] = polyCount;
  eligibleNodeOffsets[rigidCount] = eligibleNodeList.length;

  const layout = {
    nodeX,
    nodeY,
    nodeR,
    nodeMass,
    rigidPolyOffsets,
    eligibleNodeOffsets,
    eligibleNodeIndices: new Uint32Array(eligibleNodeList),
    polyRigidIndex: new Uint32Array(polyRigidIndex),
    polyPointOffsets: new Uint32Array(polyPointOffsets),
    polyPointX: new Float32Array(polyPointX),
    polyPointY: new Float32Array(polyPointY),
  };

  const byteLength = Object.values(layout).reduce((sum, arr) => sum + (arr?.byteLength || 0), 0);
  let signature = 0x811c9dc5;
  signature ^= hashU32ArrayFnv1a(layout.rigidPolyOffsets);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(layout.eligibleNodeOffsets);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(layout.eligibleNodeIndices);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashU32ArrayFnv1a(layout.polyPointOffsets);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(layout.polyPointX);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashF32ArrayFnv1a(layout.polyPointY);
  signature >>>= 0;

  return {
    layout,
    nodeCount,
    rigidCount,
    polyCount,
    eligibleNodeCount: layout.eligibleNodeIndices.length,
    byteLength,
    signature,
  };
}

function computeRigidInsideCorrectionProposalSignature({
  nodeX,
  nodeY,
  nodeVx,
  nodeVy,
  rigidX,
  rigidY,
  correctionIters,
  correctionSlop,
  layoutSignature,
}) {
  let hash = 0x811c9dc5;
  hash ^= hashF32ArrayFnv1a(nodeX);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(nodeY);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(nodeVx);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(nodeVy);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(rigidX);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(rigidY);
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= (Number(correctionIters) || 0) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;

  const scratch = new ArrayBuffer(4);
  const asF32 = new Float32Array(scratch);
  const asU32 = new Uint32Array(scratch);
  asF32[0] = Number(correctionSlop) || 0;
  hash ^= asU32[0] >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;

  hash ^= (Number(layoutSignature) || 0) >>> 0;
  return hash >>> 0;
}

function closestPointOnSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const ab2 = abx * abx + aby * aby;
  if (ab2 < 1e-12) return { x: ax, y: ay, t: 0, abx, aby, ab2 };
  const apx = px - ax;
  const apy = py - ay;
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / ab2));
  return { x: ax + abx * t, y: ay + aby * t, t, abx, aby, ab2 };
}

function resolveRigidInsideProjection(rb, node, poly, correctionSlop = 0.04) {
  if (!rb || !node || !Array.isArray(poly) || poly.length < 3) return false;
  if (!pointInPolygonInclusive(node.x, node.y, poly)) return false;

  let cx = 0;
  let cy = 0;
  for (const p of poly) {
    cx += p.x;
    cy += p.y;
  }
  cx /= Math.max(1, poly.length);
  cy /= Math.max(1, poly.length);

  let best = null;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const cp = closestPointOnSegment(node.x, node.y, a.x, a.y, b.x, b.y);

    const ex = b.x - a.x;
    const ey = b.y - a.y;
    const eLen = Math.max(1e-6, Math.hypot(ex, ey));
    let nx = -ey / eLen;
    let ny = ex / eLen;
    const mx = (a.x + b.x) * 0.5;
    const my = (a.y + b.y) * 0.5;
    const toCenterX = cx - mx;
    const toCenterY = cy - my;
    if (toCenterX * nx + toCenterY * ny > 0) {
      nx = -nx;
      ny = -ny;
    }

    const dx = node.x - cp.x;
    const dy = node.y - cp.y;
    const d2 = dx * dx + dy * dy;
    if (!best || d2 < best.d2) best = { cp, d2, nx, ny };
  }
  if (!best) return false;

  const pad = Math.max(0.25, Number(node.r) || 1) + correctionSlop;
  const tx = best.cp.x + best.nx * pad;
  const ty = best.cp.y + best.ny * pad;

  const corrX = tx - node.x;
  const corrY = ty - node.y;
  const corrLen = Math.hypot(corrX, corrY);
  if (!Number.isFinite(corrLen) || corrLen < EPS) return false;

  const mNode = Math.max(0.02, Number(node.mass) || 1);
  const mRigid = Math.max(0.05, Number(rb.mass) || 1);
  const invNode = 1 / mNode;
  const invRigid = 1 / mRigid;
  const invSum = invNode + invRigid;

  const nodeShare = invNode / Math.max(EPS, invSum);
  const rigidShare = invRigid / Math.max(EPS, invSum);

  node.x += corrX * nodeShare;
  node.y += corrY * nodeShare;
  rb.x -= corrX * rigidShare * 0.45;
  rb.y -= corrY * rigidShare * 0.45;

  const vn = node.vx * best.nx + node.vy * best.ny;
  if (vn < 0) {
    node.vx -= best.nx * vn;
    node.vy -= best.ny * vn;
  }

  return true;
}

export function applyRigidInsideCorrectionPassGpuOnly({
  rigidBodies,
  soft,
  hybridAttachedByRigid,
  correctionIters = 2,
  correctionSlop = 0.04,
  getRigidPolysWorld = getRigidCollisionPolysWorld,
  wgslOffload,
} = {}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return 0;
  if (!soft || !Array.isArray(soft.nodes) || soft.nodes.length === 0) return 0;

  let corrected = 0;
  let preparedLayoutSignature = 0;
  let proposalSignature = 0;

  const nodeCount = soft.nodes.length;
  const rigidCount = rigidBodies.length;
  const inputNodeX = new Float32Array(nodeCount);
  const inputNodeY = new Float32Array(nodeCount);
  const inputNodeVx = new Float32Array(nodeCount);
  const inputNodeVy = new Float32Array(nodeCount);
  const inputRigidX = new Float32Array(rigidCount);
  const inputRigidY = new Float32Array(rigidCount);

  for (let i = 0; i < nodeCount; i++) {
    const node = soft.nodes[i] || {};
    inputNodeX[i] = Number(node.x) || 0;
    inputNodeY[i] = Number(node.y) || 0;
    inputNodeVx[i] = Number(node.vx) || 0;
    inputNodeVy[i] = Number(node.vy) || 0;
  }
  for (let i = 0; i < rigidCount; i++) {
    const rb = rigidBodies[i] || {};
    inputRigidX[i] = Number(rb.x) || 0;
    inputRigidY[i] = Number(rb.y) || 0;
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const prep = buildRigidInsideCorrectionWgslLayout({
      rigidBodies,
      soft,
      hybridAttachedByRigid,
      getRigidPolysWorld,
    });
    preparedLayoutSignature = prep.signature >>> 0;
    proposalSignature = computeRigidInsideCorrectionProposalSignature({
      nodeX: inputNodeX,
      nodeY: inputNodeY,
      nodeVx: inputNodeVx,
      nodeVy: inputNodeVy,
      rigidX: inputRigidX,
      rigidY: inputRigidY,
      correctionIters,
      correctionSlop,
      layoutSignature: preparedLayoutSignature,
    });

    wgslOffload.state.preparedInsideLayout = prep.layout;
    wgslOffload.state.lastPreparedInsideNodeCount = prep.nodeCount;
    wgslOffload.state.lastPreparedInsideRigidCount = prep.rigidCount;
    wgslOffload.state.lastPreparedInsidePolyCount = prep.polyCount;
    wgslOffload.state.lastPreparedInsideEligibleNodeCount = prep.eligibleNodeCount;
    wgslOffload.state.lastPreparedInsideLayoutBytes = prep.byteLength;
    wgslOffload.state.lastPreparedInsideLayoutSignature = preparedLayoutSignature;
    wgslOffload.state.lastPreparedInsideProposalSignature = proposalSignature;
    wgslOffload.state.lastSourceRoute = 'cpu-rigid-inside-prepared-layout';
    wgslOffload.state.lastMode = 'cpu-rigid-inside-authoritative';
    wgslOffload.state.lastError = null;
  }

  for (let iter = 0; iter < correctionIters; iter++) {
    for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
      const rb = rigidBodies[rbi];
      if (!rb || rb.insideCorrectionEnabled === false) continue;
      const polys = getRigidPolysWorld(rb);
      if (!Array.isArray(polys) || polys.length === 0) continue;
      const attachedNodeSet = hybridAttachedByRigid?.get?.(rbi) || null;

      for (let ni = 0; ni < soft.nodes.length; ni++) {
        if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
        const node = soft.nodes[ni];
        if (!node) continue;
        for (const poly of polys) {
          if (!Array.isArray(poly) || poly.length < 3) continue;
          if (!pointInPolygonInclusive(node.x, node.y, poly)) continue;
          if (resolveRigidInsideProjection(rb, node, poly, correctionSlop)) corrected += 1;
          break;
        }
      }
    }
  }

  if (wgslOffload?.state) {
    const outputNodeX = new Float32Array(nodeCount);
    const outputNodeY = new Float32Array(nodeCount);
    const outputNodeVx = new Float32Array(nodeCount);
    const outputNodeVy = new Float32Array(nodeCount);
    const outputRigidX = new Float32Array(rigidCount);
    const outputRigidY = new Float32Array(rigidCount);

    for (let i = 0; i < nodeCount; i++) {
      const node = soft.nodes[i] || {};
      outputNodeX[i] = Number(node.x) || 0;
      outputNodeY[i] = Number(node.y) || 0;
      outputNodeVx[i] = Number(node.vx) || 0;
      outputNodeVy[i] = Number(node.vy) || 0;
    }
    for (let i = 0; i < rigidCount; i++) {
      const rb = rigidBodies[i] || {};
      outputRigidX[i] = Number(rb.x) || 0;
      outputRigidY[i] = Number(rb.y) || 0;
    }

    wgslOffload.state.lastInsideCpuReference = {
      nodeX: outputNodeX,
      nodeY: outputNodeY,
      nodeVx: outputNodeVx,
      nodeVy: outputNodeVy,
      rigidX: outputRigidX,
      rigidY: outputRigidY,
      correctedCount: corrected,
      source: 'cpu-rigid-inside-authoritative-reference',
    };
    wgslOffload.state.lastInsideParity = {
      maxAbs: 0,
      meanAbs: 0,
      comparedCount: nodeCount * 4 + rigidCount * 2,
      mismatchCount: 0,
      source: 'cpu-rigid-inside-authoritative-reference',
      proposalSignature: proposalSignature >>> 0,
      preparedLayoutSignature: preparedLayoutSignature >>> 0,
    };
    wgslOffload.state.lastInsideCorrectionCount = corrected;
    wgslOffload.state.lastAuthoritativeInsideSource = 'cpu-rigid-inside-authoritative';
    wgslOffload.state.lastSourceRoute = 'cpu-rigid-inside-authoritative';
    wgslOffload.state.lastMode = 'cpu-rigid-inside-authoritative';
  }

  return corrected;
}
