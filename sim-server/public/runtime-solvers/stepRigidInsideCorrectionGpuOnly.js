/**
 * GPU-only rigid-inside correction pass.
 *
 * Owns post-collision rigid↔soft inside projection correction in the isolated
 * gpu-only runtime path so this stepping responsibility no longer lives inline
 * in gpu-lab.js orchestration.
 */

import { getRigidCollisionPolysWorld, pointInPolygonInclusive } from '../rigid-collision.js';

const EPS = 1e-8;
const WGSL_WORKGROUP_SIZE = 64;

const rigidInsidePolyBoundsProbeWgsl = /* wgsl */`
@group(0) @binding(0) var<storage, read> poly_point_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> poly_point_x: array<f32>;
@group(0) @binding(2) var<storage, read> poly_point_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> poly_min_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> poly_min_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> poly_max_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> poly_max_y: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let poly_index = gid.x;
  let poly_count = arrayLength(&poly_min_x);
  if (poly_index >= poly_count) { return; }

  let start = poly_point_offsets[poly_index];
  let stop = poly_point_offsets[poly_index + 1u];
  if (stop <= start) {
    poly_min_x[poly_index] = 0.0;
    poly_min_y[poly_index] = 0.0;
    poly_max_x[poly_index] = 0.0;
    poly_max_y[poly_index] = 0.0;
    return;
  }

  var min_x = poly_point_x[start];
  var min_y = poly_point_y[start];
  var max_x = min_x;
  var max_y = min_y;

  for (var i = start + 1u; i < stop; i = i + 1u) {
    let px = poly_point_x[i];
    let py = poly_point_y[i];
    min_x = min(min_x, px);
    min_y = min(min_y, py);
    max_x = max(max_x, px);
    max_y = max(max_y, py);
  }

  poly_min_x[poly_index] = min_x;
  poly_min_y[poly_index] = min_y;
  poly_max_x[poly_index] = max_x;
  poly_max_y[poly_index] = max_y;
}
`;

const rigidInsideNodePolyCandidateWgsl = /* wgsl */`
@group(0) @binding(0) var<storage, read> poly_rigid_index: array<u32>;
@group(0) @binding(1) var<storage, read> poly_min_x: array<f32>;
@group(0) @binding(2) var<storage, read> poly_min_y: array<f32>;
@group(0) @binding(3) var<storage, read> poly_max_x: array<f32>;
@group(0) @binding(4) var<storage, read> poly_max_y: array<f32>;
@group(0) @binding(5) var<storage, read> node_x: array<f32>;
@group(0) @binding(6) var<storage, read> node_y: array<f32>;
@group(0) @binding(7) var<storage, read> node_r: array<f32>;
@group(0) @binding(8) var<storage, read_write> node_candidate_count_out: array<u32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node_index = gid.x;
  let node_count = arrayLength(&node_candidate_count_out);
  if (node_index >= node_count) { return; }

  let nx = node_x[node_index];
  let ny = node_y[node_index];
  let nr = max(node_r[node_index], 0.25);
  var candidate_count = 0u;
  let poly_count = arrayLength(&poly_rigid_index);

  for (var pi = 0u; pi < poly_count; pi = pi + 1u) {
    let min_x = poly_min_x[pi] - nr;
    let min_y = poly_min_y[pi] - nr;
    let max_x = poly_max_x[pi] + nr;
    let max_y = poly_max_y[pi] + nr;
    if (nx >= min_x && nx <= max_x && ny >= min_y && ny <= max_y) {
      candidate_count = candidate_count + 1u;
    }
  }

  node_candidate_count_out[node_index] = candidate_count;
}
`;

const rigidInsideCorrectionProposalWgsl = /* wgsl */`
struct Params {
  node_count: u32,
  poly_count: u32,
  _pad0: u32,
  _pad1: u32,
  correction_slop: f32,
  _pad2: f32,
  _pad3: f32,
  _pad4: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> poly_rigid_index: array<u32>;
@group(0) @binding(2) var<storage, read> poly_min_x: array<f32>;
@group(0) @binding(3) var<storage, read> poly_min_y: array<f32>;
@group(0) @binding(4) var<storage, read> poly_max_x: array<f32>;
@group(0) @binding(5) var<storage, read> poly_max_y: array<f32>;
@group(0) @binding(6) var<storage, read> node_x: array<f32>;
@group(0) @binding(7) var<storage, read> node_y: array<f32>;
@group(0) @binding(8) var<storage, read> node_r: array<f32>;
@group(0) @binding(9) var<storage, read_write> out_corr_x: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_corr_y: array<f32>;
@group(0) @binding(11) var<storage, read_write> out_rigid_index: array<u32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node_index = gid.x;
  if (node_index >= params.node_count) { return; }

  let nx = node_x[node_index];
  let ny = node_y[node_index];
  let pad = max(node_r[node_index], 0.25) + max(params.correction_slop, 0.0);

  var best_pen = -1.0;
  var best_dx = 0.0;
  var best_dy = 0.0;
  var best_rigid = 0u;

  for (var pi = 0u; pi < params.poly_count; pi = pi + 1u) {
    let min_x = poly_min_x[pi] + pad;
    let min_y = poly_min_y[pi] + pad;
    let max_x = poly_max_x[pi] - pad;
    let max_y = poly_max_y[pi] - pad;

    if (max_x <= min_x || max_y <= min_y) { continue; }
    if (nx < min_x || nx > max_x || ny < min_y || ny > max_y) { continue; }

    let dx_left = nx - min_x;
    let dx_right = max_x - nx;
    let dy_bottom = ny - min_y;
    let dy_top = max_y - ny;

    var push_x = 0.0;
    var push_y = 0.0;
    var local_pen = dx_left;
    push_x = -dx_left;

    if (dx_right < local_pen) {
      local_pen = dx_right;
      push_x = dx_right;
      push_y = 0.0;
    }
    if (dy_bottom < local_pen) {
      local_pen = dy_bottom;
      push_x = 0.0;
      push_y = -dy_bottom;
    }
    if (dy_top < local_pen) {
      local_pen = dy_top;
      push_x = 0.0;
      push_y = dy_top;
    }

    if (local_pen > best_pen) {
      best_pen = local_pen;
      best_dx = push_x;
      best_dy = push_y;
      best_rigid = poly_rigid_index[pi];
    }
  }

  out_corr_x[node_index] = best_dx;
  out_corr_y[node_index] = best_dy;
  out_rigid_index[node_index] = best_rigid;
}
`;

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
    for (let ni = 0; ni < nodeCount; ni++) {
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

function canUseWgslOffload(offload) {
  return Boolean(
    offload
    && offload.enabled === true
    && offload.device
    && typeof offload.device.createBuffer === 'function'
    && typeof offload.device.createCommandEncoder === 'function'
    && offload.state
    && globalThis.GPUBufferUsage,
  );
}

function getGpuOnlyPipelineModeProfile(offload) {
  const modeProfile = String(offload?.modeProfile || '').toLowerCase();
  if (modeProfile === 'gpu-only-fast') return 'gpu-only-fast';
  if (modeProfile === 'gpu-only-validated') return 'gpu-only-validated';
  return 'standard';
}

function isGpuOnlyFastMode(offload) {
  return getGpuOnlyPipelineModeProfile(offload) === 'gpu-only-fast';
}

function isGpuOnlyValidatedMode(offload) {
  return getGpuOnlyPipelineModeProfile(offload) === 'gpu-only-validated';
}

function ensureRigidInsidePolyBoundsProbeState({ offload, prep }) {
  const state = offload.state;
  const device = offload.device;
  const polyCount = Math.max(1, Number(prep?.polyCount) || 0);
  const polyPointCount = Math.max(1, Number(prep?.layout?.polyPointX?.length) || 0);

  if (!state.insidePolyBoundsProbePipeline) {
    const module = device.createShaderModule({ code: rigidInsidePolyBoundsProbeWgsl });
    state.insidePolyBoundsProbePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;

  if ((state.insidePolyBoundsProbePolyCapacity || 0) < polyCount) {
    const bytes = polyCount * 4;
    state.insidePolyBoundsMinX?.destroy?.();
    state.insidePolyBoundsMinY?.destroy?.();
    state.insidePolyBoundsMaxX?.destroy?.();
    state.insidePolyBoundsMaxY?.destroy?.();
    state.insidePolyBoundsMinX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insidePolyBoundsMinY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insidePolyBoundsMaxX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insidePolyBoundsMaxY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insidePolyBoundsProbePolyCapacity = polyCount;
    state.insidePolyBoundsProbeBindGroup = null;
  }

  if ((state.insidePolyBoundsProbePointCapacity || 0) < polyPointCount) {
    const bytes = polyPointCount * 4;
    state.insidePolyBoundsPointX?.destroy?.();
    state.insidePolyBoundsPointY?.destroy?.();
    state.insidePolyBoundsPointX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insidePolyBoundsPointY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insidePolyBoundsProbePointCapacity = polyPointCount;
    state.insidePolyBoundsProbeBindGroup = null;
  }

  if ((state.insidePolyBoundsProbeOffsetCapacity || 0) < (polyCount + 1)) {
    const bytes = (polyCount + 1) * 4;
    state.insidePolyBoundsPointOffsets?.destroy?.();
    state.insidePolyBoundsPointOffsets = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insidePolyBoundsProbeOffsetCapacity = polyCount + 1;
    state.insidePolyBoundsProbeBindGroup = null;
  }

  if (!state.insidePolyBoundsProbeBindGroup) {
    state.insidePolyBoundsProbeBindGroup = device.createBindGroup({
      layout: state.insidePolyBoundsProbePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.insidePolyBoundsPointOffsets } },
        { binding: 1, resource: { buffer: state.insidePolyBoundsPointX } },
        { binding: 2, resource: { buffer: state.insidePolyBoundsPointY } },
        { binding: 3, resource: { buffer: state.insidePolyBoundsMinX } },
        { binding: 4, resource: { buffer: state.insidePolyBoundsMinY } },
        { binding: 5, resource: { buffer: state.insidePolyBoundsMaxX } },
        { binding: 6, resource: { buffer: state.insidePolyBoundsMaxY } },
      ],
    });
  }

  return state;
}

function ensureRigidInsideNodeCandidateState({ offload, prep }) {
  const state = offload.state;
  const device = offload.device;
  const polyCount = Math.max(1, Number(prep?.polyCount) || 0);
  const nodeCount = Math.max(1, Number(prep?.nodeCount) || 0);

  if (!state.insideNodeCandidatePipeline) {
    const module = device.createShaderModule({ code: rigidInsideNodePolyCandidateWgsl });
    state.insideNodeCandidatePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const readbackUsage = globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ;

  if ((state.insideNodeCandidatePolyCapacity || 0) < polyCount) {
    const bytes = polyCount * 4;
    state.insideNodeCandidatePolyRigidIndex?.destroy?.();
    state.insideNodeCandidatePolyRigidIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insideNodeCandidatePolyCapacity = polyCount;
    state.insideNodeCandidateBindGroup = null;
  }

  if ((state.insideNodeCandidateNodeCapacity || 0) < nodeCount) {
    const bytes = nodeCount * 4;
    state.insideNodeCandidateNodeX?.destroy?.();
    state.insideNodeCandidateNodeY?.destroy?.();
    state.insideNodeCandidateNodeR?.destroy?.();
    state.insideNodeCandidateCountOut?.destroy?.();
    state.insideNodeCandidateCountReadback?.destroy?.();
    state.insideNodeCandidateNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insideNodeCandidateNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insideNodeCandidateNodeR = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insideNodeCandidateCountOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.insideNodeCandidateCountReadback = device.createBuffer({ size: bytes, usage: readbackUsage });
    state.insideNodeCandidateNodeCapacity = nodeCount;
    state.insideNodeCandidateBindGroup = null;
  }

  if (!state.insideNodeCandidateBindGroup) {
    state.insideNodeCandidateBindGroup = device.createBindGroup({
      layout: state.insideNodeCandidatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.insideNodeCandidatePolyRigidIndex } },
        { binding: 1, resource: { buffer: state.insidePolyBoundsMinX } },
        { binding: 2, resource: { buffer: state.insidePolyBoundsMinY } },
        { binding: 3, resource: { buffer: state.insidePolyBoundsMaxX } },
        { binding: 4, resource: { buffer: state.insidePolyBoundsMaxY } },
        { binding: 5, resource: { buffer: state.insideNodeCandidateNodeX } },
        { binding: 6, resource: { buffer: state.insideNodeCandidateNodeY } },
        { binding: 7, resource: { buffer: state.insideNodeCandidateNodeR } },
        { binding: 8, resource: { buffer: state.insideNodeCandidateCountOut } },
      ],
    });
  }

  return state;
}

function ensureRigidInsideCorrectionProposalState({ offload, prep }) {
  const state = offload.state;
  const device = offload.device;
  const polyCount = Math.max(1, Number(prep?.polyCount) || 0);
  const nodeCount = Math.max(1, Number(prep?.nodeCount) || 0);

  if (!state.insideCorrectionProposalPipeline) {
    const module = device.createShaderModule({ code: rigidInsideCorrectionProposalWgsl });
    state.insideCorrectionProposalPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const writeReadUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC;
  const readbackUsage = globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ;

  if ((state.insideCorrectionProposalPolyCapacity || 0) < polyCount) {
    const bytes = polyCount * 4;
    state.insideCorrectionProposalPolyRigidIndex?.destroy?.();
    state.insideCorrectionProposalPolyRigidIndex = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insideCorrectionProposalPolyCapacity = polyCount;
    state.insideCorrectionProposalBindGroup = null;
  }

  if ((state.insideCorrectionProposalNodeCapacity || 0) < nodeCount) {
    const bytes = nodeCount * 4;
    state.insideCorrectionProposalNodeX?.destroy?.();
    state.insideCorrectionProposalNodeY?.destroy?.();
    state.insideCorrectionProposalNodeR?.destroy?.();
    state.insideCorrectionProposalCorrXOut?.destroy?.();
    state.insideCorrectionProposalCorrYOut?.destroy?.();
    state.insideCorrectionProposalRigidIndexOut?.destroy?.();
    state.insideCorrectionProposalCorrXReadback?.destroy?.();
    state.insideCorrectionProposalCorrYReadback?.destroy?.();
    state.insideCorrectionProposalRigidIndexReadback?.destroy?.();
    state.insideCorrectionProposalNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insideCorrectionProposalNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insideCorrectionProposalNodeR = device.createBuffer({ size: bytes, usage: storageUsage });
    state.insideCorrectionProposalCorrXOut = device.createBuffer({ size: bytes, usage: writeReadUsage });
    state.insideCorrectionProposalCorrYOut = device.createBuffer({ size: bytes, usage: writeReadUsage });
    state.insideCorrectionProposalRigidIndexOut = device.createBuffer({ size: bytes, usage: writeReadUsage });
    state.insideCorrectionProposalCorrXReadback = device.createBuffer({ size: bytes, usage: readbackUsage });
    state.insideCorrectionProposalCorrYReadback = device.createBuffer({ size: bytes, usage: readbackUsage });
    state.insideCorrectionProposalRigidIndexReadback = device.createBuffer({ size: bytes, usage: readbackUsage });
    state.insideCorrectionProposalNodeCapacity = nodeCount;
    state.insideCorrectionProposalBindGroup = null;
  }

  if (!state.insideCorrectionProposalParams) {
    state.insideCorrectionProposalParams = device.createBuffer({
      size: 32,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
  }

  if (!state.insideCorrectionProposalBindGroup) {
    state.insideCorrectionProposalBindGroup = device.createBindGroup({
      layout: state.insideCorrectionProposalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.insideCorrectionProposalParams } },
        { binding: 1, resource: { buffer: state.insideCorrectionProposalPolyRigidIndex } },
        { binding: 2, resource: { buffer: state.insidePolyBoundsMinX } },
        { binding: 3, resource: { buffer: state.insidePolyBoundsMinY } },
        { binding: 4, resource: { buffer: state.insidePolyBoundsMaxX } },
        { binding: 5, resource: { buffer: state.insidePolyBoundsMaxY } },
        { binding: 6, resource: { buffer: state.insideCorrectionProposalNodeX } },
        { binding: 7, resource: { buffer: state.insideCorrectionProposalNodeY } },
        { binding: 8, resource: { buffer: state.insideCorrectionProposalNodeR } },
        { binding: 9, resource: { buffer: state.insideCorrectionProposalCorrXOut } },
        { binding: 10, resource: { buffer: state.insideCorrectionProposalCorrYOut } },
        { binding: 11, resource: { buffer: state.insideCorrectionProposalRigidIndexOut } },
      ],
    });
  }

  return state;
}


function dispatchRigidInsidePolyBoundsProbe({ offload, prep, proposalSignature }) {
  if (!canUseWgslOffload(offload)) return false;
  if ((Number(prep?.polyCount) || 0) <= 0) return false;

  const state = ensureRigidInsidePolyBoundsProbeState({ offload, prep });
  const device = offload.device;

  device.queue.writeBuffer(state.insidePolyBoundsPointOffsets, 0, prep.layout.polyPointOffsets);
  device.queue.writeBuffer(state.insidePolyBoundsPointX, 0, prep.layout.polyPointX);
  device.queue.writeBuffer(state.insidePolyBoundsPointY, 0, prep.layout.polyPointY);

  const dispatchCount = Math.max(1, Math.ceil(prep.polyCount / WGSL_WORKGROUP_SIZE));
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.insidePolyBoundsProbePipeline);
  pass.setBindGroup(0, state.insidePolyBoundsProbeBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  device.queue.submit([encoder.finish()]);

  state.lastInsidePolyBoundsProbeSource = 'wgsl-rigid-inside-poly-bounds-probe';
  state.lastInsidePolyBoundsProbeDispatchCount = dispatchCount;
  state.lastInsidePolyBoundsProbePolyCount = prep.polyCount;
  state.lastInsidePolyBoundsProbeSignature = proposalSignature >>> 0;
  return true;
}



function dispatchRigidInsideNodeCandidateProposal({ offload, prep, proposalSignature, includeReadbackTelemetry = true }) {
  if (!canUseWgslOffload(offload)) return false;
  if ((Number(prep?.polyCount) || 0) <= 0) return false;
  if ((Number(prep?.nodeCount) || 0) <= 0) return false;

  const state = ensureRigidInsideNodeCandidateState({ offload, prep });
  const device = offload.device;
  const nodeCount = prep.nodeCount;

  device.queue.writeBuffer(state.insideNodeCandidatePolyRigidIndex, 0, prep.layout.polyRigidIndex);
  device.queue.writeBuffer(state.insideNodeCandidateNodeX, 0, prep.layout.nodeX);
  device.queue.writeBuffer(state.insideNodeCandidateNodeY, 0, prep.layout.nodeY);
  device.queue.writeBuffer(state.insideNodeCandidateNodeR, 0, prep.layout.nodeR);

  const dispatchCount = Math.max(1, Math.ceil(nodeCount / WGSL_WORKGROUP_SIZE));
  const nodeBytes = nodeCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.insideNodeCandidatePipeline);
  pass.setBindGroup(0, state.insideNodeCandidateBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  if (includeReadbackTelemetry) {
    encoder.copyBufferToBuffer(
      state.insideNodeCandidateCountOut,
      0,
      state.insideNodeCandidateCountReadback,
      0,
      nodeBytes,
    );
  }
  device.queue.submit([encoder.finish()]);

  if (includeReadbackTelemetry) {
    const readbackPromise = state.insideNodeCandidateCountReadback
      .mapAsync(globalThis.GPUMapMode.READ)
      .then(() => {
        const mapped = state.insideNodeCandidateCountReadback.getMappedRange(0, nodeBytes);
        const nodeCandidateCount = new Uint32Array(mapped.slice(0));
        state.insideNodeCandidateCountReadback.unmap();
        state.lastInsideNodeCandidateCountByNode = nodeCandidateCount;
        state.lastInsideNodeCandidateCountSource = 'wgsl-rigid-inside-node-candidate-proposal';
        state.lastInsideNodeCandidateProposalSignature = proposalSignature >>> 0;
        state.lastInsideNodeCandidateDispatchCount = dispatchCount;
        state.lastInsideNodeCandidateComparedNodeCount = nodeCount;
        state.lastInsideNodeCandidateTotal = nodeCandidateCount.reduce((sum, value) => sum + (Number(value) || 0), 0);
        state.lastInsideNodeCandidateMax = nodeCandidateCount.reduce((max, value) => Math.max(max, Number(value) || 0), 0);
        state.lastInsideNodeCandidateError = null;
      })
      .catch((err) => {
        state.lastInsideNodeCandidateError = String(err?.message || err || 'unknown-error');
      });

    state.pendingInsideNodeCandidatePromise = readbackPromise;
    state.lastInsideNodeCandidateSourceRoute = 'wgsl-rigid-inside-node-candidate-proposal';
  } else {
    state.pendingInsideNodeCandidatePromise = null;
    state.lastInsideNodeCandidateCountByNode = null;
    state.lastInsideNodeCandidateCountSource = 'wgsl-rigid-inside-node-candidate-proposal-fast';
    state.lastInsideNodeCandidateComparedNodeCount = nodeCount;
    state.lastInsideNodeCandidateTotal = null;
    state.lastInsideNodeCandidateMax = null;
    state.lastInsideNodeCandidateError = null;
    state.lastInsideNodeCandidateSourceRoute = 'wgsl-rigid-inside-node-candidate-proposal-fast';
    state.lastInsideNodeCandidateValidation = 'skipped-readback-telemetry';
  }

  state.lastInsideNodeCandidateDispatchCount = dispatchCount;
  state.lastInsideNodeCandidateProposalSignature = proposalSignature >>> 0;
  return true;
}

function dispatchRigidInsideCorrectionProposal({
  offload,
  prep,
  proposalSignature,
  correctionSlop,
  includeReadbackTelemetry = true,
}) {
  if (!canUseWgslOffload(offload)) return false;
  if ((Number(prep?.polyCount) || 0) <= 0) return false;
  if ((Number(prep?.nodeCount) || 0) <= 0) return false;

  const state = ensureRigidInsideCorrectionProposalState({ offload, prep });
  const device = offload.device;
  const nodeCount = prep.nodeCount;
  const nodeBytes = nodeCount * 4;

  device.queue.writeBuffer(state.insideCorrectionProposalPolyRigidIndex, 0, prep.layout.polyRigidIndex);
  device.queue.writeBuffer(state.insideCorrectionProposalNodeX, 0, prep.layout.nodeX);
  device.queue.writeBuffer(state.insideCorrectionProposalNodeY, 0, prep.layout.nodeY);
  device.queue.writeBuffer(state.insideCorrectionProposalNodeR, 0, prep.layout.nodeR);

  const paramsBuffer = new ArrayBuffer(32);
  const view = new DataView(paramsBuffer);
  view.setUint32(0, nodeCount >>> 0, true);
  view.setUint32(4, (prep.polyCount || 0) >>> 0, true);
  view.setFloat32(16, Number(correctionSlop) || 0, true);
  device.queue.writeBuffer(state.insideCorrectionProposalParams, 0, paramsBuffer);

  const dispatchCount = Math.max(1, Math.ceil(nodeCount / WGSL_WORKGROUP_SIZE));
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.insideCorrectionProposalPipeline);
  pass.setBindGroup(0, state.insideCorrectionProposalBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();

  if (includeReadbackTelemetry) {
    encoder.copyBufferToBuffer(state.insideCorrectionProposalCorrXOut, 0, state.insideCorrectionProposalCorrXReadback, 0, nodeBytes);
    encoder.copyBufferToBuffer(state.insideCorrectionProposalCorrYOut, 0, state.insideCorrectionProposalCorrYReadback, 0, nodeBytes);
    encoder.copyBufferToBuffer(state.insideCorrectionProposalRigidIndexOut, 0, state.insideCorrectionProposalRigidIndexReadback, 0, nodeBytes);
  }
  device.queue.submit([encoder.finish()]);

  if (includeReadbackTelemetry) {
    const readbackPromise = Promise.all([
      state.insideCorrectionProposalCorrXReadback.mapAsync(globalThis.GPUMapMode.READ),
      state.insideCorrectionProposalCorrYReadback.mapAsync(globalThis.GPUMapMode.READ),
      state.insideCorrectionProposalRigidIndexReadback.mapAsync(globalThis.GPUMapMode.READ),
    ]).then(() => {
      const corrX = new Float32Array(state.insideCorrectionProposalCorrXReadback.getMappedRange(0, nodeBytes).slice(0));
      const corrY = new Float32Array(state.insideCorrectionProposalCorrYReadback.getMappedRange(0, nodeBytes).slice(0));
      const rigidIndex = new Uint32Array(state.insideCorrectionProposalRigidIndexReadback.getMappedRange(0, nodeBytes).slice(0));
      state.insideCorrectionProposalCorrXReadback.unmap();
      state.insideCorrectionProposalCorrYReadback.unmap();
      state.insideCorrectionProposalRigidIndexReadback.unmap();
      state.lastInsideCorrectionProposalCorrX = corrX;
      state.lastInsideCorrectionProposalCorrY = corrY;
      state.lastInsideCorrectionProposalRigidIndex = rigidIndex;
      state.lastInsideCorrectionProposalSource = 'wgsl-rigid-inside-correction-proposal';
      state.lastInsideCorrectionProposalSignature = proposalSignature >>> 0;
      state.lastInsideCorrectionProposalDispatchCount = dispatchCount;
      state.lastInsideCorrectionProposalNodeCount = nodeCount;
      state.lastInsideCorrectionProposalError = null;
    }).catch((err) => {
      state.lastInsideCorrectionProposalError = String(err?.message || err || 'unknown-error');
    });
    state.pendingInsideCorrectionProposalPromise = readbackPromise;
  } else {
    state.pendingInsideCorrectionProposalPromise = null;
    state.lastInsideCorrectionProposalCorrX = null;
    state.lastInsideCorrectionProposalCorrY = null;
    state.lastInsideCorrectionProposalRigidIndex = null;
    state.lastInsideCorrectionProposalSource = 'wgsl-rigid-inside-correction-proposal-fast';
    state.lastInsideCorrectionProposalNodeCount = nodeCount;
    state.lastInsideCorrectionProposalError = null;
    state.lastInsideCorrectionProposalValidation = 'skipped-readback-telemetry';
  }

  state.lastInsideCorrectionProposalDispatchCount = dispatchCount;
  state.lastInsideCorrectionProposalSignature = proposalSignature >>> 0;
  return true;
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

function canApplyAuthoritativeRigidInsideProposal({
  wgslOffload,
  proposalSignature,
  nodeCount,
  rigidCount,
}) {
  const state = wgslOffload?.state;
  if (!state) return false;
  const fastMode = isGpuOnlyFastMode(wgslOffload);
  const source = String(state.lastInsideCorrectionProposalSource || '');
  if (fastMode) {
    const fastAccepted = source === 'wgsl-rigid-inside-correction-proposal-fast'
      || source === 'wgsl-rigid-inside-correction-proposal';
    if (!fastAccepted) return false;
  } else if (source !== 'wgsl-rigid-inside-correction-proposal') {
    return false;
  }
  if ((state.lastInsideCorrectionProposalSignature >>> 0) !== (proposalSignature >>> 0)) return false;
  if (!(state.lastInsideCorrectionProposalCorrX instanceof Float32Array)) return false;
  if (!(state.lastInsideCorrectionProposalCorrY instanceof Float32Array)) return false;
  if (!(state.lastInsideCorrectionProposalRigidIndex instanceof Uint32Array)) return false;
  if (state.lastInsideCorrectionProposalCorrX.length !== nodeCount) return false;
  if (state.lastInsideCorrectionProposalCorrY.length !== nodeCount) return false;
  if (state.lastInsideCorrectionProposalRigidIndex.length !== nodeCount) return false;
  if (!fastMode) {
    const err = state.lastInsideCorrectionProposalError;
    if (typeof err === 'string' && err.length > 0) return false;
  }
  for (let i = 0; i < nodeCount; i++) {
    const cx = Number(state.lastInsideCorrectionProposalCorrX[i]);
    const cy = Number(state.lastInsideCorrectionProposalCorrY[i]);
    const rbi = Number(state.lastInsideCorrectionProposalRigidIndex[i]);
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(rbi)) return false;
    if (rbi < 0 || rbi >= rigidCount) return false;
  }
  return true;
}

function applyAuthoritativeRigidInsideProposal({
  rigidBodies,
  soft,
  proposalCorrX,
  proposalCorrY,
  proposalRigidIndex,
  correctionIters,
}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const rigids = Array.isArray(rigidBodies) ? rigidBodies : [];
  const nodeCount = Math.min(
    nodes.length,
    proposalCorrX.length,
    proposalCorrY.length,
    proposalRigidIndex.length,
  );

  let corrected = 0;
  const iters = Math.max(1, Number(correctionIters) || 1);
  for (let iter = 0; iter < iters; iter++) {
    for (let ni = 0; ni < nodeCount; ni++) {
      const node = nodes[ni];
      if (!node) continue;
      const corrX = Number(proposalCorrX[ni]);
      const corrY = Number(proposalCorrY[ni]);
      const corrLen = Math.hypot(corrX, corrY);
      if (!Number.isFinite(corrLen) || corrLen < EPS) continue;

      const rbi = Number(proposalRigidIndex[ni]);
      if (!Number.isFinite(rbi) || rbi < 0 || rbi >= rigids.length) continue;
      const rb = rigids[rbi];
      if (!rb) continue;

      const mNode = Math.max(0.02, Number(node.mass) || 1);
      const mRigid = Math.max(0.05, Number(rb.mass) || 1);
      const invNode = 1 / mNode;
      const invRigid = 1 / mRigid;
      const invSum = Math.max(EPS, invNode + invRigid);
      const nodeShare = invNode / invSum;
      const rigidShare = invRigid / invSum;

      node.x = (Number(node.x) || 0) + corrX * nodeShare;
      node.y = (Number(node.y) || 0) + corrY * nodeShare;
      rb.x = (Number(rb.x) || 0) - corrX * rigidShare * 0.45;
      rb.y = (Number(rb.y) || 0) - corrY * rigidShare * 0.45;

      const nx = corrX / corrLen;
      const ny = corrY / corrLen;
      const vn = (Number(node.vx) || 0) * nx + (Number(node.vy) || 0) * ny;
      if (vn < 0) {
        node.vx = (Number(node.vx) || 0) - nx * vn;
        node.vy = (Number(node.vy) || 0) - ny * vn;
      }

      corrected += 1;
    }
  }

  return corrected;
}


export function applyRigidInsideCorrectionPassGpuOnly({
  rigidBodies,
  soft,
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
    const pipelineMode = getGpuOnlyPipelineModeProfile(wgslOffload);
    const fastMode = isGpuOnlyFastMode(wgslOffload);
    const validatedMode = isGpuOnlyValidatedMode(wgslOffload);
    const standardMode = pipelineMode === 'standard';

    const prep = buildRigidInsideCorrectionWgslLayout({
      rigidBodies,
      soft,
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
    wgslOffload.state.lastPipelineModeProfile = pipelineMode;
    wgslOffload.state.lastSourceRoute = 'cpu-rigid-inside-prepared-layout';
    wgslOffload.state.lastMode = 'cpu-rigid-inside-authoritative';
    wgslOffload.state.lastError = null;

    try {
      if (!standardMode) {
        const probeRan = dispatchRigidInsidePolyBoundsProbe({
          offload: wgslOffload,
          prep,
          proposalSignature,
        });
        if (probeRan) {
          wgslOffload.state.lastSourceRoute = 'wgsl-rigid-inside-poly-bounds-probe';
          wgslOffload.state.lastMode = fastMode
            ? 'cpu-rigid-inside-authoritative-wgsl-probe-fast'
            : 'cpu-rigid-inside-authoritative-wgsl-probe';
          const candidateRan = dispatchRigidInsideNodeCandidateProposal({
            offload: wgslOffload,
            prep,
            proposalSignature,
            includeReadbackTelemetry: !fastMode,
          });
          if (candidateRan) {
            wgslOffload.state.lastSourceRoute = fastMode
              ? 'wgsl-rigid-inside-node-candidate-proposal-fast'
              : 'wgsl-rigid-inside-node-candidate-proposal';
            wgslOffload.state.lastMode = fastMode
              ? 'cpu-rigid-inside-authoritative-wgsl-candidate-proposal-fast'
              : 'cpu-rigid-inside-authoritative-wgsl-candidate-proposal';
            wgslOffload.state.lastInsideNodeCandidateValidation = fastMode
              ? 'skipped-readback-telemetry'
              : (validatedMode ? 'readback-telemetry-validated' : 'readback-telemetry');

            const correctionProposalRan = dispatchRigidInsideCorrectionProposal({
              offload: wgslOffload,
              prep,
              proposalSignature,
              correctionSlop,
              includeReadbackTelemetry: true,
            });
            if (correctionProposalRan) {
              wgslOffload.state.lastSourceRoute = fastMode
                ? 'wgsl-rigid-inside-correction-proposal-fast'
                : 'wgsl-rigid-inside-correction-proposal';
              wgslOffload.state.lastMode = fastMode
                ? 'cpu-rigid-inside-authoritative-wgsl-correction-proposal-fast'
                : 'cpu-rigid-inside-authoritative-wgsl-correction-proposal';
            }
          }
        }
      }
    } catch (err) {
      wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
      wgslOffload.state.lastMode = 'cpu-rigid-inside-authoritative';
      wgslOffload.state.lastSourceRoute = 'cpu-rigid-inside-prepared-layout';
    }
  }

  let usedAuthoritativeWgslProposal = false;
  const canUseAuthoritativeWgsl = canApplyAuthoritativeRigidInsideProposal({
    wgslOffload,
    proposalSignature,
    nodeCount,
    rigidCount,
  });

  if (canUseAuthoritativeWgsl) {
    corrected = applyAuthoritativeRigidInsideProposal({
      rigidBodies,
      soft,
      proposalCorrX: wgslOffload.state.lastInsideCorrectionProposalCorrX,
      proposalCorrY: wgslOffload.state.lastInsideCorrectionProposalCorrY,
      proposalRigidIndex: wgslOffload.state.lastInsideCorrectionProposalRigidIndex,
      correctionIters,
    });
    usedAuthoritativeWgslProposal = true;
  } else {
    for (let iter = 0; iter < correctionIters; iter++) {
      for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
        const rb = rigidBodies[rbi];
        if (!rb || rb.insideCorrectionEnabled === false) continue;
        const polys = getRigidPolysWorld(rb);
        if (!Array.isArray(polys) || polys.length === 0) continue;

        for (let ni = 0; ni < soft.nodes.length; ni++) {
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
  }

  if (wgslOffload?.state) {
    const fastMode = isGpuOnlyFastMode(wgslOffload);

    if (!fastMode) {
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
      wgslOffload.state.lastInsideValidation = 'cpu-reference';
    } else {
      wgslOffload.state.lastInsideCpuReference = null;
      wgslOffload.state.lastInsideParity = {
        maxAbs: null,
        meanAbs: null,
        comparedCount: 0,
        mismatchCount: 0,
        source: 'cpu-rigid-inside-authoritative-fast',
        proposalSignature: proposalSignature >>> 0,
        preparedLayoutSignature: preparedLayoutSignature >>> 0,
        validation: 'skipped-cpu-reference',
      };
      wgslOffload.state.lastInsideValidation = 'skipped-cpu-reference';
    }

    wgslOffload.state.lastInsideCorrectionCount = corrected;
    wgslOffload.state.lastAuthoritativeInsideSource = usedAuthoritativeWgslProposal
      ? (fastMode ? 'wgsl-rigid-inside-authoritative-fast' : 'wgsl-rigid-inside-authoritative')
      : 'cpu-rigid-inside-authoritative';
    wgslOffload.state.lastSourceRoute = usedAuthoritativeWgslProposal
      ? (fastMode ? 'wgsl-rigid-inside-authoritative-fast' : 'wgsl-rigid-inside-authoritative')
      : 'cpu-rigid-inside-authoritative';
    wgslOffload.state.lastMode = usedAuthoritativeWgslProposal
      ? (fastMode ? 'wgsl-rigid-inside-authoritative-fast' : 'wgsl-rigid-inside-authoritative')
      : (fastMode ? 'cpu-rigid-inside-authoritative-fast' : 'cpu-rigid-inside-authoritative');
  }

  return corrected;
}
