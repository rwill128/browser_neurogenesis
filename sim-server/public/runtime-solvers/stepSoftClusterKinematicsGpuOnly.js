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
