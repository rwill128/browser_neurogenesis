/**
 * GPU-only soft membrane inside-correction pass.
 *
 * Owns post-collision membrane containment recovery in the isolated gpu-only
 * runtime path so this stepping responsibility no longer depends on baseline
 * inline logic from gpu-lab.js.
 */

import { pointInPolygonInclusive } from '../rigid-collision.js';

const EPS = 1e-8;
const WGSL_LAYOUT_SIG_SEED = 0x811c9dc5;

function getGpuOnlyPipelineModeProfile(offload) {
  const modeProfile = String(offload?.pipelineMode || '').toLowerCase();
  if (modeProfile === 'gpu-only-fast') return 'gpu-only-fast';
  if (modeProfile === 'gpu-only-validated') return 'gpu-only-validated';
  return 'standard';
}

function hashUint(value, hash) {
  let h = hash >>> 0;
  h ^= (Number(value) >>> 0);
  h = Math.imul(h, 0x01000193) >>> 0;
  return h >>> 0;
}

function hashFloat(value, hash) {
  if (!Number.isFinite(value)) return hashUint(0x7fc00000, hash);
  const buf = new ArrayBuffer(4);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = value;
  return hashUint(u[0], hash);
}

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

function buildSoftMembraneInsideCorrectionWgslLayout({ soft, candidates, correctionSlop = 0.04 } = {}) {
  if (!soft || !Array.isArray(soft.nodes) || !Array.isArray(candidates) || candidates.length === 0) return null;

  const nodeCount = soft.nodes.length;
  const loopCount = candidates.length;

  const nodeData = new Float32Array(nodeCount * 4);
  const nodeClusterId = new Int32Array(nodeCount);
  const loopBounds = new Float32Array(loopCount * 4);
  const loopMeta = new Float32Array(loopCount * 3);
  const loopClusterId = new Int32Array(loopCount);
  const loopPointOffsets = new Uint32Array(loopCount + 1);

  const flatPoints = [];
  for (let i = 0; i < nodeCount; i++) {
    const n = soft.nodes[i];
    nodeData[i * 4] = Number(n?.x) || 0;
    nodeData[i * 4 + 1] = Number(n?.y) || 0;
    nodeData[i * 4 + 2] = Number(n?.r) || 0;
    nodeData[i * 4 + 3] = Math.max(0.2, Number(n?.r) || 1) + Math.max(0, Number(correctionSlop) || 0.04);
    nodeClusterId[i] = Number.isFinite(Number(n?.clusterId)) ? (Number(n.clusterId) | 0) : -1;
  }

  let pointOffset = 0;
  for (let li = 0; li < loopCount; li++) {
    const c = candidates[li];
    loopPointOffsets[li] = pointOffset >>> 0;
    loopBounds[li * 4] = Number(c.minX) || 0;
    loopBounds[li * 4 + 1] = Number(c.minY) || 0;
    loopBounds[li * 4 + 2] = Number(c.maxX) || 0;
    loopBounds[li * 4 + 3] = Number(c.maxY) || 0;
    loopMeta[li * 3] = Number(c.areaSign) || 1;
    loopMeta[li * 3 + 1] = Number(c.cid) || 0;
    loopMeta[li * 3 + 2] = Number(c.ids?.length || 0);
    loopClusterId[li] = Number(c.cid) | 0;

    const poly = Array.isArray(c.poly) ? c.poly : [];
    for (let pi = 0; pi < poly.length; pi++) {
      const p = poly[pi];
      flatPoints.push(Number(p?.x) || 0, Number(p?.y) || 0);
    }
    pointOffset += poly.length;
  }
  loopPointOffsets[loopCount] = pointOffset >>> 0;

  return {
    nodeCount,
    loopCount,
    flatPointCount: pointOffset,
    nodeData,
    nodeClusterId,
    loopBounds,
    loopMeta,
    loopClusterId,
    loopPointOffsets,
    flatPoints: Float32Array.from(flatPoints),
  };
}

function computeSoftMembraneInsideLayoutSignature(layout) {
  if (!layout) return 0;
  let hash = WGSL_LAYOUT_SIG_SEED;
  hash = hashUint(layout.nodeCount || 0, hash);
  hash = hashUint(layout.loopCount || 0, hash);
  hash = hashUint(layout.flatPointCount || 0, hash);

  const hashArray = (arr, isFloat = false) => {
    if (!arr) return;
    const limit = Math.min(arr.length, 4096);
    for (let i = 0; i < limit; i++) {
      hash = isFloat ? hashFloat(Number(arr[i]), hash) : hashUint(Number(arr[i]), hash);
    }
  };

  hashArray(layout.nodeData, true);
  hashArray(layout.nodeClusterId, false);
  hashArray(layout.loopBounds, true);
  hashArray(layout.loopMeta, true);
  hashArray(layout.loopClusterId, false);
  hashArray(layout.loopPointOffsets, false);
  hashArray(layout.flatPoints, true);
  return hash >>> 0;
}

export function applySoftMembraneInsideCorrectionPassGpuOnly({
  sim,
  soft,
  loops,
  correctionIters = 2,
  correctionSlop = 0.04,
  wgslOffload = null,
} = {}) {
  if (!soft?.nodes?.length || !Array.isArray(loops) || loops.length === 0) return 0;
  const clusterMap = sim?.softMembraneClusterMap;
  if (!(clusterMap instanceof Map) || clusterMap.size === 0) return 0;

  const candidates = buildMembraneInsideCandidates({ soft, loops, clusterMap });
  if (candidates.length === 0) return 0;

  const pipelineMode = getGpuOnlyPipelineModeProfile(wgslOffload);
  const standardMode = pipelineMode === 'standard';
  const stagedWgslLayout = !standardMode
    ? buildSoftMembraneInsideCorrectionWgslLayout({ soft, candidates, correctionSlop })
    : null;
  const stagedWgslLayoutSignature = stagedWgslLayout
    ? computeSoftMembraneInsideLayoutSignature(stagedWgslLayout)
    : 0;

  if (wgslOffload?.state) {
    wgslOffload.state.lastPreparedMembraneInsideLayout = stagedWgslLayout;
    wgslOffload.state.lastPreparedMembraneInsideLayoutSignature = stagedWgslLayoutSignature >>> 0;
    wgslOffload.state.lastPreparedMembraneInsideLayoutBytes = stagedWgslLayout
      ? (
          stagedWgslLayout.nodeData.byteLength
          + stagedWgslLayout.nodeClusterId.byteLength
          + stagedWgslLayout.loopBounds.byteLength
          + stagedWgslLayout.loopMeta.byteLength
          + stagedWgslLayout.loopClusterId.byteLength
          + stagedWgslLayout.loopPointOffsets.byteLength
          + stagedWgslLayout.flatPoints.byteLength
        )
      : 0;
    wgslOffload.state.lastPreparedMembraneInsideLayoutNodeCount = stagedWgslLayout?.nodeCount || 0;
    wgslOffload.state.lastPreparedMembraneInsideLayoutLoopCount = stagedWgslLayout?.loopCount || 0;
    wgslOffload.state.lastPreparedMembraneInsideLayoutPointCount = stagedWgslLayout?.flatPointCount || 0;
    wgslOffload.state.lastMembraneInsideModeProfile = pipelineMode;
    if (!standardMode) {
      wgslOffload.state.lastMembraneInsideSourceRoute = 'cpu-membrane-inside-prepared-layout';
      wgslOffload.state.lastSourceRoute = 'cpu-membrane-inside-prepared-layout';
    }
  }

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

  if (wgslOffload?.state) {
    wgslOffload.state.lastMembraneInsideAuthoritativeSource = 'cpu-membrane-inside-authoritative';
    if (!wgslOffload.state.lastMembraneInsideSourceRoute) {
      wgslOffload.state.lastMembraneInsideSourceRoute = 'cpu-membrane-inside-authoritative';
    }
  }

  return corrected;
}

export function _buildMembraneInsideCandidatesGpuOnlyForTest(args = {}) {
  return buildMembraneInsideCandidates(args);
}
