function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function finiteOr(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function nodeMass(node, minMass) {
  return Math.max(minMass, finiteOr(node?.mass, minMass));
}

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

export function computeSoftClusterKinematicsPrepSignature(layout) {
  const nodeCount = Number(layout?.nodeIndex?.length) || 0;
  const clusterCount = Math.max(0, (Number(layout?.clusterOffsets?.length) || 1) - 1);
  let hash = 0x811c9dc5;
  hash ^= nodeCount >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= clusterCount >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashU32ArrayFnv1a(layout?.clusterOriginalId) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashU32ArrayFnv1a(layout?.clusterOffsets) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashU32ArrayFnv1a(layout?.nodeIndex) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(layout?.nodeX) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(layout?.nodeY) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(layout?.nodeVx) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(layout?.nodeVy) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  hash ^= hashF32ArrayFnv1a(layout?.nodeMass) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  return hash >>> 0;
}

export function buildSoftClusterKinematicsWgslPrep(nodes, {
  minMass = 0.02,
} = {}) {
  const accepted = [];
  const clusterBuckets = new Map();

  if (!Array.isArray(nodes) || nodes.length === 0) {
    const emptyLayout = {
      clusterOriginalId: new Uint32Array(0),
      clusterOffsets: new Uint32Array(1),
      nodeIndex: new Uint32Array(0),
      nodeX: new Float32Array(0),
      nodeY: new Float32Array(0),
      nodeVx: new Float32Array(0),
      nodeVy: new Float32Array(0),
      nodeMass: new Float32Array(0),
      byteLength: 4,
    };
    return {
      plan: {
        nodeCount: 0,
        clusterCount: 0,
      },
      layout: emptyLayout,
      signature: computeSoftClusterKinematicsPrepSignature(emptyLayout),
    };
  }

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const x = Number(node?.x);
    const y = Number(node?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    const cidRaw = Number.isFinite(Number(node?.clusterId)) ? Number(node.clusterId) : 0;
    const cid = Math.max(0, Math.floor(cidRaw)) >>> 0;
    if (!clusterBuckets.has(cid)) clusterBuckets.set(cid, []);

    const acceptedIndex = accepted.length;
    accepted.push({
      nodeIndex: i,
      x,
      y,
      vx: finiteOr(node?.vx, 0),
      vy: finiteOr(node?.vy, 0),
      mass: nodeMass(node, minMass),
      clusterId: cid,
    });
    clusterBuckets.get(cid).push(acceptedIndex);
  }

  const clusterOriginalId = Uint32Array.from([...clusterBuckets.keys()].sort((a, b) => a - b));
  const clusterCount = clusterOriginalId.length;
  const nodeCount = accepted.length;
  const clusterOffsets = new Uint32Array(clusterCount + 1);
  const nodeIndex = new Uint32Array(nodeCount);
  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  const nodeVx = new Float32Array(nodeCount);
  const nodeVy = new Float32Array(nodeCount);
  const nodeMassArray = new Float32Array(nodeCount);

  let write = 0;
  for (let ci = 0; ci < clusterCount; ci++) {
    const cid = clusterOriginalId[ci];
    const bucket = clusterBuckets.get(cid) || [];
    clusterOffsets[ci] = write;
    for (let bi = 0; bi < bucket.length; bi++) {
      const a = accepted[bucket[bi]];
      nodeIndex[write] = a.nodeIndex >>> 0;
      nodeX[write] = a.x;
      nodeY[write] = a.y;
      nodeVx[write] = a.vx;
      nodeVy[write] = a.vy;
      nodeMassArray[write] = a.mass;
      write += 1;
    }
  }
  clusterOffsets[clusterCount] = write;

  const layout = {
    clusterOriginalId,
    clusterOffsets,
    nodeIndex,
    nodeX,
    nodeY,
    nodeVx,
    nodeVy,
    nodeMass: nodeMassArray,
  };
  layout.byteLength =
    layout.clusterOriginalId.byteLength
    + layout.clusterOffsets.byteLength
    + layout.nodeIndex.byteLength
    + layout.nodeX.byteLength
    + layout.nodeY.byteLength
    + layout.nodeVx.byteLength
    + layout.nodeVy.byteLength
    + layout.nodeMass.byteLength;

  return {
    plan: {
      nodeCount,
      clusterCount,
    },
    layout,
    signature: computeSoftClusterKinematicsPrepSignature(layout),
  };
}

export function computeSoftClusterKinematicsGpuOnly(nodes, {
  minMass = 0.02,
  minInertia = 1e-4,
} = {}) {
  const clusters = new Map();
  if (!Array.isArray(nodes) || nodes.length === 0) return clusters;

  const ensure = (cid) => {
    if (!clusters.has(cid)) {
      clusters.set(cid, {
        clusterId: cid,
        mass: 0,
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        inertia: minInertia,
        angularMomentum: 0,
        omega: 0,
        meanRadius: 0,
        nodeIndices: [],
      });
    }
    return clusters.get(cid);
  };

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const x = Number(node?.x);
    const y = Number(node?.y);
    const vx = finiteOr(node?.vx, 0);
    const vy = finiteOr(node?.vy, 0);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    const cid = Number.isFinite(Number(node?.clusterId)) ? Number(node.clusterId) : 0;
    const m = nodeMass(node, minMass);
    const st = ensure(cid);
    st.mass += m;
    st.x += x * m;
    st.y += y * m;
    st.vx += vx * m;
    st.vy += vy * m;
    st.nodeIndices.push(i);
  }

  for (const st of clusters.values()) {
    const invMass = 1 / Math.max(minMass, st.mass);
    st.x *= invMass;
    st.y *= invMass;
    st.vx *= invMass;
    st.vy *= invMass;
  }

  for (const st of clusters.values()) {
    let inertia = 0;
    let angularMomentum = 0;
    let radiusSum = 0;
    let radiusCount = 0;

    for (const i of st.nodeIndices) {
      const node = nodes[i];
      const x = Number(node?.x);
      const y = Number(node?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const m = nodeMass(node, minMass);
      const rx = x - st.x;
      const ry = y - st.y;
      const dvx = finiteOr(node?.vx, 0) - st.vx;
      const dvy = finiteOr(node?.vy, 0) - st.vy;
      const r2 = rx * rx + ry * ry;
      inertia += m * r2;
      angularMomentum += m * (rx * dvy - ry * dvx);
      radiusSum += Math.sqrt(r2);
      radiusCount += 1;
    }

    st.inertia = Math.max(minInertia, inertia);
    st.angularMomentum = angularMomentum;
    st.omega = Number.isFinite(angularMomentum) ? (angularMomentum / st.inertia) : 0;
    st.meanRadius = radiusCount > 0 ? (radiusSum / radiusCount) : 0;
  }

  return clusters;
}

export function computeSoftClusterKinematicsFromMassMomentsGpuOnly(nodes, layout, probe, {
  minMass = 0.02,
  minInertia = 1e-4,
} = {}) {
  const clusters = new Map();
  if (!Array.isArray(nodes) || !layout || !probe) return clusters;

  const clusterIds = layout.clusterOriginalId;
  const clusterOffsets = layout.clusterOffsets;
  const mass = probe.mass;
  const xMass = probe.xMass;
  const yMass = probe.yMass;
  const vxMass = probe.vxMass;
  const vyMass = probe.vyMass;
  const x2Mass = probe.x2Mass;
  const y2Mass = probe.y2Mass;
  const xVyMass = probe.xVyMass;
  const yVxMass = probe.yVxMass;

  const clusterCount = Math.max(0, Number(clusterIds?.length) || 0);
  if (clusterCount === 0) return clusters;

  for (let ci = 0; ci < clusterCount; ci++) {
    const cid = Number(clusterIds[ci]) || 0;
    const totalMass = Math.max(minMass, Number(mass?.[ci]) || 0);
    const invMass = 1 / totalMass;
    clusters.set(cid, {
      clusterId: cid,
      mass: totalMass,
      x: (Number(xMass?.[ci]) || 0) * invMass,
      y: (Number(yMass?.[ci]) || 0) * invMass,
      vx: (Number(vxMass?.[ci]) || 0) * invMass,
      vy: (Number(vyMass?.[ci]) || 0) * invMass,
      inertia: 0,
      angularMomentum: 0,
      omega: 0,
      meanRadius: 0,
      nodeIndices: [],
    });
  }

  const hasWgslSecondMomentProbe = (
    x2Mass instanceof Float32Array
    && y2Mass instanceof Float32Array
    && xVyMass instanceof Float32Array
    && yVxMass instanceof Float32Array
    && x2Mass.length >= clusterCount
    && y2Mass.length >= clusterCount
    && xVyMass.length >= clusterCount
    && yVxMass.length >= clusterCount
  );

  if (hasWgslSecondMomentProbe) {
    for (let ci = 0; ci < clusterCount; ci++) {
      const cid = Number(clusterIds[ci]) || 0;
      const st = clusters.get(cid);
      if (!st) continue;
      const m = Math.max(minMass, Number(st.mass) || 0);
      const cx = Number(st.x) || 0;
      const cy = Number(st.y) || 0;
      const cvx = Number(st.vx) || 0;
      const cvy = Number(st.vy) || 0;

      const sumX2Mass = Number(x2Mass[ci]) || 0;
      const sumY2Mass = Number(y2Mass[ci]) || 0;
      const sumXVyMass = Number(xVyMass[ci]) || 0;
      const sumYVxMass = Number(yVxMass[ci]) || 0;

      const inertia = Math.max(minInertia, (sumX2Mass + sumY2Mass) - m * (cx * cx + cy * cy));
      const angularMomentum = (sumXVyMass - sumYVxMass) - m * (cx * cvy - cy * cvx);
      st.inertia = Number.isFinite(inertia) ? inertia : minInertia;
      st.angularMomentum = Number.isFinite(angularMomentum) ? angularMomentum : 0;
      st.omega = Number.isFinite(st.angularMomentum) ? (st.angularMomentum / st.inertia) : 0;
      st.meanRadius = Math.sqrt(Math.max(0, st.inertia / Math.max(minMass, m)));
    }
    return clusters;
  }

  const nodeCount = Math.max(0, Number(layout.nodeIndex?.length) || 0);
  let clusterListIndex = 0;
  for (let li = 0; li < nodeCount; li++) {
    const nodeIdx = Number(layout.nodeIndex[li]);
    if (!Number.isFinite(nodeIdx) || nodeIdx < 0 || nodeIdx >= nodes.length) continue;
    const node = nodes[nodeIdx];
    if (!node) continue;

    while (clusterListIndex + 1 < clusterOffsets.length && li >= (clusterOffsets[clusterListIndex + 1] >>> 0)) {
      clusterListIndex += 1;
    }
    const cid = Number(clusterIds[clusterListIndex]) || 0;
    const st = clusters.get(cid);
    if (!st) continue;

    const x = Number(node.x);
    const y = Number(node.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    st.nodeIndices.push(nodeIdx);
    const m = nodeMass(node, minMass);
    const rx = x - st.x;
    const ry = y - st.y;
    const dvx = finiteOr(node.vx, 0) - st.vx;
    const dvy = finiteOr(node.vy, 0) - st.vy;
    const r2 = rx * rx + ry * ry;
    st.inertia += m * r2;
    st.angularMomentum += m * (rx * dvy - ry * dvx);
    st.meanRadius += Math.sqrt(r2);
  }

  for (const st of clusters.values()) {
    const count = st.nodeIndices.length;
    st.inertia = Math.max(minInertia, st.inertia);
    st.omega = Number.isFinite(st.angularMomentum) ? (st.angularMomentum / st.inertia) : 0;
    st.meanRadius = count > 0 ? (st.meanRadius / count) : 0;
  }

  return clusters;
}

export function projectNodesTowardClusterRigidMotionGpuOnly(nodes, clusterKinematics, {
  linearGain = 0.08,
  angularGain = 0.18,
  membraneClusterSet = null,
  membraneGainScale = 0.72,
} = {}) {
  if (!Array.isArray(nodes) || !(clusterKinematics instanceof Map) || clusterKinematics.size === 0) {
    return { projectedNodes: 0, meanDelta: 0 };
  }

  let projectedNodes = 0;
  let sumDelta = 0;

  for (const node of nodes) {
    if (!node) continue;
    const cid = Number.isFinite(Number(node.clusterId)) ? Number(node.clusterId) : 0;
    const st = clusterKinematics.get(cid);
    if (!st) continue;

    const x = Number(node.x);
    const y = Number(node.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    const membraneScale = (membraneClusterSet instanceof Set && membraneClusterSet.has(cid))
      ? membraneGainScale
      : 1;

    const lg = clamp(Number(linearGain) * membraneScale, 0, 1);
    const ag = clamp(Number(angularGain) * membraneScale, 0, 1);
    if (lg <= 0 && ag <= 0) continue;

    const vx = finiteOr(node.vx, 0);
    const vy = finiteOr(node.vy, 0);

    const rx = x - st.x;
    const ry = y - st.y;
    const targetRelVx = -st.omega * ry;
    const targetRelVy = st.omega * rx;

    const curRelVx = vx - st.vx;
    const curRelVy = vy - st.vy;

    const deltaVx = (st.vx - vx) * lg + (targetRelVx - curRelVx) * ag;
    const deltaVy = (st.vy - vy) * lg + (targetRelVy - curRelVy) * ag;

    node.vx = vx + deltaVx;
    node.vy = vy + deltaVy;

    projectedNodes += 1;
    sumDelta += Math.hypot(deltaVx, deltaVy);
  }

  return {
    projectedNodes,
    meanDelta: projectedNodes > 0 ? (sumDelta / projectedNodes) : 0,
  };
}

const WGSL_WORKGROUP_SIZE = 64;
const SOFT_CLUSTER_PROJECTION_LAYOUT_STRIDE_FLOATS = 13;

const softClusterProjectionProposalWgsl = /* wgsl */ `
struct Params {
  count : f32,
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
};

@group(0) @binding(0) var<storage, read> layout : array<f32>;
@group(0) @binding(1) var<storage, read_write> deltaVxOut : array<f32>;
@group(0) @binding(2) var<storage, read_write> deltaVyOut : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

fn finiteOrZero(v : f32) -> f32 {
  if (v == v && abs(v) < 1e20) {
    return v;
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (f32(i) >= params.count) {
    return;
  }

  let base = i * ${SOFT_CLUSTER_PROJECTION_LAYOUT_STRIDE_FLOATS}u;
  let x = finiteOrZero(layout[base + 1u]);
  let y = finiteOrZero(layout[base + 2u]);
  let vx = finiteOrZero(layout[base + 3u]);
  let vy = finiteOrZero(layout[base + 4u]);
  let cx = finiteOrZero(layout[base + 5u]);
  let cy = finiteOrZero(layout[base + 6u]);
  let cvx = finiteOrZero(layout[base + 7u]);
  let cvy = finiteOrZero(layout[base + 8u]);
  let omega = finiteOrZero(layout[base + 9u]);
  let lg = clamp(max(0.0, finiteOrZero(layout[base + 10u])), 0.0, 1.0);
  let ag = clamp(max(0.0, finiteOrZero(layout[base + 11u])), 0.0, 1.0);
  let membraneScale = clamp(max(0.0, finiteOrZero(layout[base + 12u])), 0.0, 1.0);

  lg = clamp(lg * membraneScale, 0.0, 1.0);
  ag = clamp(ag * membraneScale, 0.0, 1.0);

  let rx = x - cx;
  let ry = y - cy;
  let targetRelVx = -omega * ry;
  let targetRelVy = omega * rx;

  let curRelVx = vx - cvx;
  let curRelVy = vy - cvy;

  deltaVxOut[i] = (cvx - vx) * lg + (targetRelVx - curRelVx) * ag;
  deltaVyOut[i] = (cvy - vy) * lg + (targetRelVy - curRelVy) * ag;
}
`;

function checkFiniteFloat32Array(values) {
  if (!(values instanceof Float32Array)) return false;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) return false;
  }
  return true;
}

function computeSoftClusterProjectionSignature(layout, linearGain, angularGain, membraneGainScale) {
  if (!(layout instanceof Float32Array) || layout.length === 0) {
    return `0|${Math.fround(linearGain)}|${Math.fround(angularGain)}|${Math.fround(membraneGainScale)}|0`;
  }
  let sum = 0;
  for (let i = 0; i < layout.length; i += SOFT_CLUSTER_PROJECTION_LAYOUT_STRIDE_FLOATS) {
    sum += Math.fround(layout[i + 1] || 0) * 0.13;
    sum += Math.fround(layout[i + 2] || 0) * 0.11;
    sum += Math.fround(layout[i + 9] || 0) * 0.17;
    sum += Math.fround(layout[i + 12] || 0) * 0.19;
  }
  return `${Math.floor(layout.length / SOFT_CLUSTER_PROJECTION_LAYOUT_STRIDE_FLOATS)}|${Math.fround(linearGain)}|${Math.fround(angularGain)}|${Math.fround(membraneGainScale)}|${Math.fround(sum)}`;
}

function ensureSoftClusterProjectionProposalPipeline(offload) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device) return null;
  if (!state.softClusterProjectionProposalPipelinePromise) {
    state.softClusterProjectionProposalPipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: softClusterProjectionProposalWgsl }),
        entryPoint: 'main',
      },
    }).catch((err) => {
      state.softClusterProjectionProposalPipelinePromise = null;
      throw err;
    });
  }
  return state.softClusterProjectionProposalPipelinePromise;
}

function getGpuOnlyPipelineModeProfile(offload) {
  const modeProfile = String(offload?.modeProfile || '').trim().toLowerCase();
  if (modeProfile === 'gpu-only-fast') return 'gpu-only-fast';
  if (modeProfile === 'gpu-only-validated') return 'gpu-only-validated';
  return 'standard';
}

async function dispatchSoftClusterProjectionProposal(offload, layout, signature) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device || !(layout instanceof Float32Array) || layout.length === 0) return false;
  const count = Math.floor(layout.length / SOFT_CLUSTER_PROJECTION_LAYOUT_STRIDE_FLOATS);
  if (count <= 0) return false;

  const pipeline = await ensureSoftClusterProjectionProposalPipeline(offload);
  if (!pipeline) return false;

  const layoutBuffer = device.createBuffer({ size: layout.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBytes = count * Float32Array.BYTES_PER_ELEMENT;
  const deltaVxBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const deltaVyBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readDeltaVxBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readDeltaVyBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const params = new Float32Array([count, 0, 0, 0]);
  const paramBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(layoutBuffer, 0, layout);
  device.queue.writeBuffer(paramBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: layoutBuffer } },
      { binding: 1, resource: { buffer: deltaVxBuffer } },
      { binding: 2, resource: { buffer: deltaVyBuffer } },
      { binding: 3, resource: { buffer: paramBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / WGSL_WORKGROUP_SIZE)));
  pass.end();
  encoder.copyBufferToBuffer(deltaVxBuffer, 0, readDeltaVxBuffer, 0, outBytes);
  encoder.copyBufferToBuffer(deltaVyBuffer, 0, readDeltaVyBuffer, 0, outBytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([readDeltaVxBuffer.mapAsync(GPUMapMode.READ), readDeltaVyBuffer.mapAsync(GPUMapMode.READ)]);
  const deltaVx = new Float32Array(readDeltaVxBuffer.getMappedRange().slice(0));
  const deltaVy = new Float32Array(readDeltaVyBuffer.getMappedRange().slice(0));
  readDeltaVxBuffer.unmap();
  readDeltaVyBuffer.unmap();

  const allFinite = checkFiniteFloat32Array(deltaVx) && checkFiniteFloat32Array(deltaVy);
  state.lastSoftClusterProjectionProposalSignature = String(signature || '');
  state.lastSoftClusterProjectionProposalSource = allFinite
    ? 'wgsl-soft-cluster-rigid-motion-projection-proposal'
    : 'cpu-soft-cluster-rigid-motion-projection-authoritative-nonfinite';
  state.lastSoftClusterProjectionProposalFinite = { allFinite };
  if (allFinite) {
    state.lastSoftClusterProjectionProposalDeltaVx = deltaVx;
    state.lastSoftClusterProjectionProposalDeltaVy = deltaVy;
  } else {
    state.lastSoftClusterProjectionProposalDeltaVx = null;
    state.lastSoftClusterProjectionProposalDeltaVy = null;
  }

  layoutBuffer.destroy();
  deltaVxBuffer.destroy();
  deltaVyBuffer.destroy();
  readDeltaVxBuffer.destroy();
  readDeltaVyBuffer.destroy();
  paramBuffer.destroy();
  return allFinite;
}

export async function projectNodesTowardClusterRigidMotionWithWgslGpuOnly(nodes, clusterKinematics, {
  linearGain = 0.08,
  angularGain = 0.18,
  membraneClusterSet = null,
  membraneGainScale = 0.72,
  wgslOffload = null,
} = {}) {
  if (!Array.isArray(nodes) || !(clusterKinematics instanceof Map) || clusterKinematics.size === 0) {
    return { projectedNodes: 0, meanDelta: 0 };
  }

  const runWgslProposal = wgslOffload?.enabled === true
    && wgslOffload?.state
    && wgslOffload?.device
    && getGpuOnlyPipelineModeProfile(wgslOffload) !== 'standard';

  const layoutEntries = [];
  const nodeRefs = [];
  const cpuDeltaVx = [];
  const cpuDeltaVy = [];

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (!node) continue;
    const cid = Number.isFinite(Number(node.clusterId)) ? Number(node.clusterId) : 0;
    const st = clusterKinematics.get(cid);
    if (!st) continue;

    const x = Number(node.x);
    const y = Number(node.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    const membraneScale = (membraneClusterSet instanceof Set && membraneClusterSet.has(cid))
      ? Number(membraneGainScale) || 0.72
      : 1;

    const lg = clamp(Number(linearGain) * membraneScale, 0, 1);
    const ag = clamp(Number(angularGain) * membraneScale, 0, 1);
    if (lg <= 0 && ag <= 0) continue;

    const vx = finiteOr(node.vx, 0);
    const vy = finiteOr(node.vy, 0);
    const rx = x - st.x;
    const ry = y - st.y;
    const targetRelVx = -st.omega * ry;
    const targetRelVy = st.omega * rx;
    const curRelVx = vx - st.vx;
    const curRelVy = vy - st.vy;
    const dvx = (st.vx - vx) * lg + (targetRelVx - curRelVx) * ag;
    const dvy = (st.vy - vy) * lg + (targetRelVy - curRelVy) * ag;

    nodeRefs.push(node);
    cpuDeltaVx.push(dvx);
    cpuDeltaVy.push(dvy);

    if (runWgslProposal) {
      layoutEntries.push(
        i,
        x,
        y,
        vx,
        vy,
        Number(st.x) || 0,
        Number(st.y) || 0,
        Number(st.vx) || 0,
        Number(st.vy) || 0,
        Number(st.omega) || 0,
        Number(linearGain) || 0,
        Number(angularGain) || 0,
        membraneScale,
      );
    }
  }

  let useWgslAuthoritative = false;
  let wgslDeltaVx = null;
  let wgslDeltaVy = null;
  let signature = '';

  if (runWgslProposal && layoutEntries.length > 0) {
    const layout = Float32Array.from(layoutEntries);
    signature = computeSoftClusterProjectionSignature(layout, linearGain, angularGain, membraneGainScale);
    wgslOffload.state.lastSoftClusterProjectionProposalLayoutBytes = layout.byteLength;
    wgslOffload.state.lastSoftClusterProjectionProposalSignaturePrepared = signature;
    if (!wgslOffload.state.lastSoftClusterProjectionProposalSource) {
      wgslOffload.state.lastSoftClusterProjectionProposalSource = 'cpu-soft-cluster-rigid-motion-projection-authoritative';
    }

    const serializedDispatch = (wgslOffload.state.pendingSoftClusterProjectionProposalPromise || Promise.resolve())
      .catch(() => {})
      .then(() => dispatchSoftClusterProjectionProposal(wgslOffload, layout, signature))
      .catch((err) => {
        wgslOffload.state.lastSoftClusterProjectionProposalError = String(err?.message || err || 'unknown-error');
        wgslOffload.state.lastSoftClusterProjectionProposalSource = 'cpu-soft-cluster-rigid-motion-projection-authoritative';
        return false;
      });
    wgslOffload.state.pendingSoftClusterProjectionProposalPromise = serializedDispatch;

    const proposalReady = await serializedDispatch;
    useWgslAuthoritative = wgslOffload?.state?.enableAuthoritativeSoftClusterProjection === true
      && proposalReady === true
      && String(wgslOffload?.state?.lastSoftClusterProjectionProposalSignature || '') === signature
      && wgslOffload?.state?.lastSoftClusterProjectionProposalDeltaVx instanceof Float32Array
      && wgslOffload?.state?.lastSoftClusterProjectionProposalDeltaVy instanceof Float32Array
      && wgslOffload.state.lastSoftClusterProjectionProposalDeltaVx.length === nodeRefs.length
      && wgslOffload.state.lastSoftClusterProjectionProposalDeltaVy.length === nodeRefs.length
      && wgslOffload.state.lastSoftClusterProjectionProposalFinite?.allFinite === true;

    if (useWgslAuthoritative) {
      wgslDeltaVx = wgslOffload.state.lastSoftClusterProjectionProposalDeltaVx;
      wgslDeltaVy = wgslOffload.state.lastSoftClusterProjectionProposalDeltaVy;
    }
  }

  let projectedNodes = 0;
  let sumDelta = 0;
  for (let i = 0; i < nodeRefs.length; i++) {
    const node = nodeRefs[i];
    const dvx = useWgslAuthoritative ? Number(wgslDeltaVx?.[i]) : Number(cpuDeltaVx[i]);
    const dvy = useWgslAuthoritative ? Number(wgslDeltaVy?.[i]) : Number(cpuDeltaVy[i]);
    const safeDvx = Number.isFinite(dvx) ? dvx : 0;
    const safeDvy = Number.isFinite(dvy) ? dvy : 0;
    node.vx = finiteOr(node.vx, 0) + safeDvx;
    node.vy = finiteOr(node.vy, 0) + safeDvy;
    projectedNodes += 1;
    sumDelta += Math.hypot(safeDvx, safeDvy);
  }

  if (wgslOffload?.state) {
    wgslOffload.state.lastSoftClusterProjectionAuthoritativeSource = useWgslAuthoritative
      ? 'wgsl-soft-cluster-rigid-motion-projection-authoritative'
      : 'cpu-soft-cluster-rigid-motion-projection-authoritative';
    wgslOffload.state.lastSoftClusterProjectionAuthoritativeSignature = signature;
  }

  return {
    projectedNodes,
    meanDelta: projectedNodes > 0 ? (sumDelta / projectedNodes) : 0,
  };
}
