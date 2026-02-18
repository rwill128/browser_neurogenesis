/**
 * GPU-only soft membrane inside-correction pass.
 *
 * Owns post-collision membrane containment recovery in the isolated gpu-only
 * runtime path so this stepping responsibility no longer depends on baseline
 * inline logic from gpu-lab.js.
 */

import { pointInPolygonInclusive } from '../rigid-collision.js';

const EPS = 1e-8;

function closestPointOnSegment(px, py, ax, ay, bx, by) {
  const ex = bx - ax;
  const ey = by - ay;
  const segLenSq = ex * ex + ey * ey;
  if (!Number.isFinite(segLenSq) || segLenSq < 1e-12) return { x: ax, y: ay };
  const apx = px - ax;
  const apy = py - ay;
  const t = Math.max(0, Math.min(1, (apx * ex + apy * ey) / segLenSq));
  return { x: ax + ex * t, y: ay + ey * t };
}

function buildMembraneInsideCandidates({ soft, loops, clusterMap } = {}) {
  if (!soft || !Array.isArray(soft.nodes) || !Array.isArray(loops) || !(clusterMap instanceof Map)) return [];

  const candidates = [];
  for (const loop of loops) {
    const cid = Number(loop?.clusterId);
    if (!Number.isInteger(cid)) continue;

    const membrane = clusterMap.get(cid);
    if (!membrane || Number(membrane.insideCorrectionEnabled) <= 0) continue;

    const ids = Array.isArray(loop?.indices) ? loop.indices : [];
    if (ids.length < 3) continue;

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let area = 0;
    const poly = [];

    for (let i = 0; i < ids.length; i++) {
      const node = soft.nodes[ids[i]];
      if (!node) continue;
      poly.push({ x: node.x, y: node.y });
      minX = Math.min(minX, node.x);
      minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x);
      maxY = Math.max(maxY, node.y);

      const next = soft.nodes[ids[(i + 1) % ids.length]];
      if (next) area += node.x * next.y - next.x * node.y;
    }

    if (poly.length < 3 || !Number.isFinite(area) || Math.abs(area) < EPS) continue;

    candidates.push({
      cid,
      ids,
      poly,
      minX,
      minY,
      maxX,
      maxY,
      areaSign: area >= 0 ? 1 : -1,
    });
  }

  return candidates;
}

export function applySoftMembraneInsideCorrectionPassGpuOnly({
  sim,
  soft,
  loops,
  correctionIters = 2,
  correctionSlop = 0.04,
} = {}) {
  if (!soft?.nodes?.length || !Array.isArray(loops) || loops.length === 0) return 0;
  const clusterMap = sim?.softMembraneClusterMap;
  if (!(clusterMap instanceof Map) || clusterMap.size === 0) return 0;

  const candidates = buildMembraneInsideCandidates({ soft, loops, clusterMap });
  if (candidates.length === 0) return 0;

  let corrected = 0;

  for (let iter = 0; iter < correctionIters; iter++) {
    for (let ni = 0; ni < soft.nodes.length; ni++) {
      const node = soft.nodes[ni];
      if (!node) continue;

      for (const candidate of candidates) {
        if (Number(node.clusterId) === candidate.cid) continue;

        const pad = Math.max(0.2, Number(node.r) || 1) + correctionSlop;
        if (
          node.x < candidate.minX - pad ||
          node.x > candidate.maxX + pad ||
          node.y < candidate.minY - pad ||
          node.y > candidate.maxY + pad
        ) continue;

        if (!pointInPolygonInclusive(node.x, node.y, candidate.poly)) continue;

        let best = null;
        for (let i = 0; i < candidate.ids.length; i++) {
          const ai = candidate.ids[i];
          const bi = candidate.ids[(i + 1) % candidate.ids.length];
          const a = soft.nodes[ai];
          const b = soft.nodes[bi];
          if (!a || !b) continue;

          const cp = closestPointOnSegment(node.x, node.y, a.x, a.y, b.x, b.y);
          const dx = node.x - cp.x;
          const dy = node.y - cp.y;
          const d2 = dx * dx + dy * dy;
          if (!best || d2 < best.d2) {
            const ex = b.x - a.x;
            const ey = b.y - a.y;
            const len = Math.max(1e-6, Math.hypot(ex, ey));
            const nx = candidate.areaSign >= 0 ? (ey / len) : (-ey / len);
            const ny = candidate.areaSign >= 0 ? (-ex / len) : (ex / len);
            best = { d2, cp, nx, ny, ai, bi };
          }
        }

        if (!best) continue;

        const tx = best.cp.x + best.nx * pad;
        const ty = best.cp.y + best.ny * pad;
        const corrX = tx - node.x;
        const corrY = ty - node.y;
        const corrLen = Math.hypot(corrX, corrY);
        if (!Number.isFinite(corrLen) || corrLen < EPS) continue;

        node.x += corrX * 0.95;
        node.y += corrY * 0.95;

        const a = soft.nodes[best.ai];
        const b = soft.nodes[best.bi];
        if (a && b) {
          a.x -= corrX * 0.02;
          a.y -= corrY * 0.02;
          b.x -= corrX * 0.02;
          b.y -= corrY * 0.02;
        }

        const vn = node.vx * best.nx + node.vy * best.ny;
        if (vn < 0) {
          node.vx -= best.nx * vn;
          node.vy -= best.ny * vn;
        }

        corrected += 1;
        break;
      }
    }
  }

  return corrected;
}

export function _buildMembraneInsideCandidatesGpuOnlyForTest(args = {}) {
  return buildMembraneInsideCandidates(args);
}
