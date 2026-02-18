/**
 * GPU-only rigid-inside correction pass.
 *
 * Owns post-collision rigid↔soft inside projection correction in the isolated
 * gpu-only runtime path so this stepping responsibility no longer lives inline
 * in gpu-lab.js orchestration.
 */

import { getRigidCollisionPolysWorld, pointInPolygonInclusive } from '../rigid-collision.js';

const EPS = 1e-8;

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
} = {}) {
  if (!Array.isArray(rigidBodies) || rigidBodies.length === 0) return 0;
  if (!soft || !Array.isArray(soft.nodes) || soft.nodes.length === 0) return 0;

  let corrected = 0;

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

  return corrected;
}
