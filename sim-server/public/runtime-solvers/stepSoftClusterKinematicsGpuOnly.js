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
