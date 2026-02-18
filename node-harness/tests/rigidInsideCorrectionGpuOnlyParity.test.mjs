import test from 'node:test';
import assert from 'node:assert/strict';
import { getRigidCollisionPolysWorld, pointInPolygonInclusive } from '../../sim-server/public/rigid-collision.js';
import { applyRigidInsideCorrectionPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidInsideCorrectionGpuOnly.js';

const RIGID_INSIDE_CORRECTION_ITERS = 2;
const RIGID_INSIDE_CORRECTION_SLOP = 0.04;

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

function resolveRigidInsideProjection(rb, node, poly) {
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

  const pad = Math.max(0.25, Number(node.r) || 1) + RIGID_INSIDE_CORRECTION_SLOP;
  const tx = best.cp.x + best.nx * pad;
  const ty = best.cp.y + best.ny * pad;

  const corrX = tx - node.x;
  const corrY = ty - node.y;
  const corrLen = Math.hypot(corrX, corrY);
  if (!Number.isFinite(corrLen) || corrLen < 1e-8) return false;

  const mNode = Math.max(0.02, Number(node.mass) || 1);
  const mRigid = Math.max(0.05, Number(rb.mass) || 1);
  const invNode = 1 / mNode;
  const invRigid = 1 / mRigid;
  const invSum = invNode + invRigid;

  const nodeShare = invNode / Math.max(1e-8, invSum);
  const rigidShare = invRigid / Math.max(1e-8, invSum);

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

function runBaselineInlineRigidInsidePass(rigidBodies, soft, hybridAttachedByRigid) {
  if (!rigidBodies?.length || !soft?.nodes?.length) return 0;
  let corrected = 0;

  for (let iter = 0; iter < RIGID_INSIDE_CORRECTION_ITERS; iter++) {
    for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
      const rb = rigidBodies[rbi];
      if (!rb || rb.insideCorrectionEnabled === false) continue;
      const polys = getRigidCollisionPolysWorld(rb);
      if (!Array.isArray(polys) || polys.length === 0) continue;
      const attachedNodeSet = hybridAttachedByRigid?.get?.(rbi) || null;

      for (let ni = 0; ni < soft.nodes.length; ni++) {
        if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
        const node = soft.nodes[ni];
        if (!node) continue;
        for (const poly of polys) {
          if (!Array.isArray(poly) || poly.length < 3) continue;
          if (!pointInPolygonInclusive(node.x, node.y, poly)) continue;
          if (resolveRigidInsideProjection(rb, node, poly)) corrected += 1;
          break;
        }
      }
    }
  }

  return corrected;
}

function makeFixture() {
  const rigidBodies = [
    {
      x: 10,
      y: 10,
      r: 3.2,
      sides: 4,
      theta: 0.2,
      mass: 1.8,
      vx: 0.2,
      vy: -0.1,
      omega: 0.03,
      insideCorrectionEnabled: true,
    },
  ];

  const soft = {
    nodes: [
      { x: 10.1, y: 10.2, vx: -0.4, vy: 0.3, r: 0.7, mass: 0.6, clusterId: 2 },
      { x: 12.9, y: 10.4, vx: -0.2, vy: -0.1, r: 0.7, mass: 0.5, clusterId: 2 },
      { x: 6.5, y: 5.5, vx: 0.1, vy: 0.2, r: 0.7, mass: 0.5, clusterId: 3 },
    ],
  };

  const hybridAttachedByRigid = new Map([[0, new Set([1])]]);
  return { rigidBodies, soft, hybridAttachedByRigid };
}

test('rigid inside-correction parity: gpu-only runtime pass matches previous inline behavior', () => {
  const baseline = makeFixture();
  const gpuOnly = makeFixture();

  const baselineCorrections = runBaselineInlineRigidInsidePass(
    baseline.rigidBodies,
    baseline.soft,
    baseline.hybridAttachedByRigid,
  );

  const gpuCorrections = applyRigidInsideCorrectionPassGpuOnly({
    rigidBodies: gpuOnly.rigidBodies,
    soft: gpuOnly.soft,
    hybridAttachedByRigid: gpuOnly.hybridAttachedByRigid,
  });

  assert.equal(gpuCorrections, baselineCorrections);
  assert.deepEqual(gpuOnly, baseline);
});
