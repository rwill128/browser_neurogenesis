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

const WGSL_WORKGROUP_SIZE = 64;

const softMembraneInsideCorrectionProposalWgsl = /* wgsl */`
struct Params {
  node_count: u32,
  loop_count: u32,
  correction_slop: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> node_data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> node_cluster_id: array<i32>;
@group(0) @binding(3) var<storage, read> loop_bounds: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> loop_meta: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read> loop_cluster_id: array<i32>;
@group(0) @binding(6) var<storage, read> loop_point_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> flat_points: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_corr_x: array<f32>;
@group(0) @binding(9) var<storage, read_write> out_corr_y: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_nx: array<f32>;
@group(0) @binding(11) var<storage, read_write> out_ny: array<f32>;
@group(0) @binding(12) var<storage, read_write> out_edge_a: array<i32>;
@group(0) @binding(13) var<storage, read_write> out_edge_b: array<i32>;
@group(0) @binding(14) var<storage, read_write> out_hit: array<u32>;

fn point_xy(index: u32) -> vec2<f32> {
  let bi = index * 2u;
  return vec2<f32>(flat_points[bi], flat_points[bi + 1u]);
}

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ni = gid.x;
  if (ni >= params.node_count) { return; }

  let nd = node_data[ni];
  let nx = nd.x;
  let ny = nd.y;
  let pad = max(0.2, nd.z) + max(0.0, params.correction_slop);
  let cluster_id = node_cluster_id[ni];

  var best_d2 = 1e30;
  var best_corr = vec2<f32>(0.0, 0.0);
  var best_n = vec2<f32>(0.0, 0.0);
  var best_ai: i32 = -1;
  var best_bi: i32 = -1;
  var hit: u32 = 0u;

  for (var li: u32 = 0u; li < params.loop_count; li = li + 1u) {
    if (cluster_id == loop_cluster_id[li]) { continue; }
    let b = loop_bounds[li];
    if (nx < b.x - pad || nx > b.z + pad || ny < b.y - pad || ny > b.w + pad) { continue; }

    let start = loop_point_offsets[li];
    let stop = loop_point_offsets[li + 1u];
    if (stop <= start + 2u) { continue; }

    var inside = false;
    var j = stop - 1u;
    for (var i = start; i < stop; i = i + 1u) {
      let pi = point_xy(i);
      let pj = point_xy(j);
      let yi = pi.y;
      let yj = pj.y;
      let intersects = ((yi > ny) != (yj > ny))
        && (nx < (pj.x - pi.x) * (ny - yi) / max(1e-6, (yj - yi)) + pi.x);
      if (intersects) { inside = !inside; }
      j = i;
    }
    if (!inside) { continue; }

    let area_sign = select(-1.0, 1.0, loop_meta[li].x >= 0.0);
    for (var i = start; i < stop; i = i + 1u) {
      let next = select(i + 1u, start, i + 1u >= stop);
      let a = point_xy(i);
      let bpt = point_xy(next);
      let e = bpt - a;
      let el2 = max(1e-6, dot(e, e));
      let t = clamp(dot(vec2<f32>(nx, ny) - a, e) / el2, 0.0, 1.0);
      let cp = a + e * t;
      let d = vec2<f32>(nx, ny) - cp;
      let d2 = dot(d, d);
      if (d2 < best_d2) {
        let len = max(1e-6, sqrt(el2));
        let n = area_sign >= 0.0
          ? vec2<f32>(e.y / len, -e.x / len)
          : vec2<f32>(-e.y / len, e.x / len);
        let target = cp + n * pad;
        best_corr = target - vec2<f32>(nx, ny);
        best_n = n;
        best_d2 = d2;
        best_ai = i32(i - start);
        best_bi = i32(next - start);
        hit = 1u;
      }
    }
  }

  out_corr_x[ni] = best_corr.x;
  out_corr_y[ni] = best_corr.y;
  out_nx[ni] = best_n.x;
  out_ny[ni] = best_n.y;
  out_edge_a[ni] = best_ai;
  out_edge_b[ni] = best_bi;
  out_hit[ni] = hit;
}
`;

function ensureSoftMembraneInsideProposalPipeline(wgslOffload) {
  if (!wgslOffload?.device) return null;
  if (wgslOffload.softMembraneInsideProposalPipeline) return wgslOffload.softMembraneInsideProposalPipeline;
  const device = wgslOffload.device;
  const module = device.createShaderModule({ code: softMembraneInsideCorrectionProposalWgsl });
  wgslOffload.softMembraneInsideProposalPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  return wgslOffload.softMembraneInsideProposalPipeline;
}

function canApplyAuthoritativeMembraneInsideProposal({ wgslOffload, signature, expectedNodeCount }) {
  const state = wgslOffload?.state;
  if (!state || state.enableAuthoritativeMembraneInsideCorrection !== true) return false;
  if ((state.lastMembraneInsideProposalSignature >>> 0) !== (signature >>> 0)) return false;
  if (!(state.lastMembraneInsideProposalCorrX instanceof Float32Array)) return false;
  if (!(state.lastMembraneInsideProposalCorrY instanceof Float32Array)) return false;
  if (!(state.lastMembraneInsideProposalNx instanceof Float32Array)) return false;
  if (!(state.lastMembraneInsideProposalNy instanceof Float32Array)) return false;
  if (!(state.lastMembraneInsideProposalHit instanceof Uint32Array)) return false;
  if (state.lastMembraneInsideProposalCorrX.length !== expectedNodeCount) return false;
  if (state.lastMembraneInsideProposalCorrY.length !== expectedNodeCount) return false;
  if (state.lastMembraneInsideProposalNx.length !== expectedNodeCount) return false;
  if (state.lastMembraneInsideProposalNy.length !== expectedNodeCount) return false;
  if (state.lastMembraneInsideProposalHit.length !== expectedNodeCount) return false;
  return true;
}

function applyAuthoritativeMembraneInsideProposal({ soft, proposal, correctionIters = 1 }) {
  let corrected = 0;
  for (let iter = 0; iter < correctionIters; iter++) {
    for (let ni = 0; ni < soft.nodes.length; ni++) {
      if ((proposal.hit[ni] >>> 0) === 0) continue;
      const node = soft.nodes[ni];
      if (!node) continue;
      const corrX = Number(proposal.corrX[ni]) || 0;
      const corrY = Number(proposal.corrY[ni]) || 0;
      const corrLen = Math.hypot(corrX, corrY);
      if (!Number.isFinite(corrLen) || corrLen < EPS) continue;
      node.x += corrX * 0.95;
      node.y += corrY * 0.95;
      const nx = Number(proposal.nx[ni]) || 0;
      const ny = Number(proposal.ny[ni]) || 0;
      const vn = node.vx * nx + node.vy * ny;
      if (vn < 0) {
        node.vx -= nx * vn;
        node.vy -= ny * vn;
      }
      corrected += 1;
    }
  }
  return corrected;
}

async function readBackTypedArray(device, buffer, Type, count) {
  const byteLength = Math.max(4, count * Type.BYTES_PER_ELEMENT);
  const alignedSize = Math.max(4, Math.ceil(byteLength / 4) * 4);
  const readback = device.createBuffer({ size: alignedSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readback, 0, alignedSize);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const slice = readback.getMappedRange(0, byteLength).slice(0);
  readback.unmap();
  readback.destroy();
  return new Type(slice);
}

function dispatchSoftMembraneInsideCorrectionProposal(wgslOffload, layout, signature, correctionSlop) {
  const device = wgslOffload?.device;
  const state = wgslOffload?.state;
  if (!device || !state || !layout) return Promise.resolve();
  const pipeline = ensureSoftMembraneInsideProposalPipeline(wgslOffload);
  if (!pipeline) return Promise.resolve();

  const nodeCount = layout.nodeCount >>> 0;
  const loopCount = layout.loopCount >>> 0;
  const createStorageBuffer = (typedArray, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => {
    const size = Math.max(4, Math.ceil(typedArray.byteLength / 4) * 4);
    const buf = device.createBuffer({ size, usage });
    device.queue.writeBuffer(buf, 0, typedArray.buffer, typedArray.byteOffset, typedArray.byteLength);
    return buf;
  };

  const params = new Float32Array([nodeCount, loopCount, Number(correctionSlop) || 0, 0]);
  const paramsBuffer = createStorageBuffer(params, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const nodeDataBuffer = createStorageBuffer(layout.nodeData);
  const nodeClusterIdBuffer = createStorageBuffer(layout.nodeClusterId);
  const loopBoundsBuffer = createStorageBuffer(layout.loopBounds);
  const loopMetaBuffer = createStorageBuffer(layout.loopMeta);
  const loopClusterIdBuffer = createStorageBuffer(layout.loopClusterId);
  const loopPointOffsetsBuffer = createStorageBuffer(layout.loopPointOffsets);
  const flatPointsBuffer = createStorageBuffer(layout.flatPoints);

  const makeOut = () => device.createBuffer({
    size: Math.max(4, nodeCount * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const outCorrX = makeOut();
  const outCorrY = makeOut();
  const outNx = makeOut();
  const outNy = makeOut();
  const outEdgeA = makeOut();
  const outEdgeB = makeOut();
  const outHit = makeOut();

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: nodeDataBuffer } },
      { binding: 2, resource: { buffer: nodeClusterIdBuffer } },
      { binding: 3, resource: { buffer: loopBoundsBuffer } },
      { binding: 4, resource: { buffer: loopMetaBuffer } },
      { binding: 5, resource: { buffer: loopClusterIdBuffer } },
      { binding: 6, resource: { buffer: loopPointOffsetsBuffer } },
      { binding: 7, resource: { buffer: flatPointsBuffer } },
      { binding: 8, resource: { buffer: outCorrX } },
      { binding: 9, resource: { buffer: outCorrY } },
      { binding: 10, resource: { buffer: outNx } },
      { binding: 11, resource: { buffer: outNy } },
      { binding: 12, resource: { buffer: outEdgeA } },
      { binding: 13, resource: { buffer: outEdgeB } },
      { binding: 14, resource: { buffer: outHit } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(nodeCount / WGSL_WORKGROUP_SIZE)));
  pass.end();
  device.queue.submit([encoder.finish()]);

  return Promise.all([
    readBackTypedArray(device, outCorrX, Float32Array, nodeCount),
    readBackTypedArray(device, outCorrY, Float32Array, nodeCount),
    readBackTypedArray(device, outNx, Float32Array, nodeCount),
    readBackTypedArray(device, outNy, Float32Array, nodeCount),
    readBackTypedArray(device, outEdgeA, Int32Array, nodeCount),
    readBackTypedArray(device, outEdgeB, Int32Array, nodeCount),
    readBackTypedArray(device, outHit, Uint32Array, nodeCount),
  ]).then(([corrX, corrY, nx, ny, edgeA, edgeB, hit]) => {
    state.lastMembraneInsideProposalCorrX = corrX;
    state.lastMembraneInsideProposalCorrY = corrY;
    state.lastMembraneInsideProposalNx = nx;
    state.lastMembraneInsideProposalNy = ny;
    state.lastMembraneInsideProposalEdgeA = edgeA;
    state.lastMembraneInsideProposalEdgeB = edgeB;
    state.lastMembraneInsideProposalHit = hit;
    state.lastMembraneInsideProposalSource = 'wgsl-membrane-inside-correction-proposal';
    state.lastMembraneInsideProposalSignature = signature >>> 0;
    state.lastMembraneInsideSourceRoute = 'wgsl-membrane-inside-correction-proposal';
    state.lastMode = 'wgsl-membrane-inside-correction-proposal';
  }).catch((err) => {
    state.lastMembraneInsideProposalError = String(err?.message || err || 'unknown-error');
    state.lastMembraneInsideProposalSource = 'cpu-membrane-inside-authoritative';
  });
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

  if (canApplyAuthoritativeMembraneInsideProposal({
    wgslOffload,
    signature: stagedWgslLayoutSignature,
    expectedNodeCount: soft.nodes.length,
  })) {
    const proposal = {
      corrX: wgslOffload.state.lastMembraneInsideProposalCorrX,
      corrY: wgslOffload.state.lastMembraneInsideProposalCorrY,
      nx: wgslOffload.state.lastMembraneInsideProposalNx,
      ny: wgslOffload.state.lastMembraneInsideProposalNy,
      hit: wgslOffload.state.lastMembraneInsideProposalHit,
    };
    const correctedByWgsl = applyAuthoritativeMembraneInsideProposal({ soft, proposal, correctionIters });
    wgslOffload.state.lastMembraneInsideAuthoritativeSource = pipelineMode === 'gpu-only-fast'
      ? 'wgsl-membrane-inside-authoritative-fast'
      : 'wgsl-membrane-inside-authoritative';
    wgslOffload.state.lastMembraneInsideSourceRoute = wgslOffload.state.lastMembraneInsideAuthoritativeSource;
    wgslOffload.state.lastSourceRoute = wgslOffload.state.lastMembraneInsideAuthoritativeSource;
    wgslOffload.state.lastMode = wgslOffload.state.lastMembraneInsideAuthoritativeSource;
    return correctedByWgsl;
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

  if (!standardMode && wgslOffload?.state && wgslOffload?.enabled === true && wgslOffload?.device && stagedWgslLayout) {
    wgslOffload.state.lastMembraneInsideProposalLayoutBytes =
      stagedWgslLayout.nodeData.byteLength
      + stagedWgslLayout.nodeClusterId.byteLength
      + stagedWgslLayout.loopBounds.byteLength
      + stagedWgslLayout.loopMeta.byteLength
      + stagedWgslLayout.loopClusterId.byteLength
      + stagedWgslLayout.loopPointOffsets.byteLength
      + stagedWgslLayout.flatPoints.byteLength;
    wgslOffload.state.lastMembraneInsideProposalSignaturePrepared = stagedWgslLayoutSignature >>> 0;
    const serializedDispatch = (wgslOffload.state.pendingWgslMembraneInsideProposalPromise || Promise.resolve())
      .catch(() => {})
      .then(() => dispatchSoftMembraneInsideCorrectionProposal(
        wgslOffload,
        stagedWgslLayout,
        stagedWgslLayoutSignature,
        correctionSlop,
      ));
    wgslOffload.state.pendingWgslMembraneInsideProposalPromise = serializedDispatch;
  }

  return corrected;
}

export function _buildMembraneInsideCandidatesGpuOnlyForTest(args = {}) {
  return buildMembraneInsideCandidates(args);
}
