import test from 'node:test';
import assert from 'node:assert/strict';
import { applySoftFluidCouplingGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftFluidCouplingGpuOnly.js';

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function computeSoftCentroid(nodes) {
  if (!nodes.length) return { x: 0, y: 0 };
  let sx = 0;
  let sy = 0;
  for (const n of nodes) { sx += n.x; sy += n.y; }
  return { x: sx / nodes.length, y: sy / nodes.length };
}

function computeSoftClusterKinematics(nodes) {
  const byCluster = new Map();
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    let st = byCluster.get(cid);
    if (!st) {
      st = { mass: 0, x: 0, y: 0, vx: 0, vy: 0, inertia: 0, omegaNum: 0 };
      byCluster.set(cid, st);
    }
    const m = Math.max(0.02, Number(node.mass) || 1);
    st.mass += m;
    st.x += node.x * m;
    st.y += node.y * m;
    st.vx += node.vx * m;
    st.vy += node.vy * m;
  }
  for (const st of byCluster.values()) {
    const inv = st.mass > 0 ? (1 / st.mass) : 0;
    st.x *= inv;
    st.y *= inv;
    st.vx *= inv;
    st.vy *= inv;
  }
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const st = byCluster.get(cid);
    const m = Math.max(0.02, Number(node.mass) || 1);
    const rx = node.x - st.x;
    const ry = node.y - st.y;
    st.inertia += m * (rx * rx + ry * ry);
    st.omegaNum += m * (rx * (node.vy - st.vy) - ry * (node.vx - st.vx));
  }
  for (const st of byCluster.values()) {
    const I = Math.max(1e-4, st.inertia);
    st.omega = st.omegaNum / I;
    st.inertia = I;
  }
  return byCluster;
}

function projectNodesTowardClusterRigidMotion(nodes, clusterMap, opts) {
  const linearGain = Number(opts?.linearGain) || 0;
  const angularGain = Number(opts?.angularGain) || 0;
  const membraneSet = opts?.membraneClusterSet || new Set();
  const membraneScale = Number(opts?.membraneGainScale) || 1;
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const c = clusterMap.get(cid);
    if (!c) continue;
    const rx = node.x - c.x;
    const ry = node.y - c.y;
    const targetVx = c.vx - c.omega * ry;
    const targetVy = c.vy + c.omega * rx;
    const scale = membraneSet.has(cid) ? membraneScale : 1;
    node.vx += (targetVx - node.vx) * linearGain * scale;
    node.vy += (targetVy - node.vy) * linearGain * scale;
    node.vx += (-c.omega * ry) * angularGain * scale;
    node.vy += (c.omega * rx) * angularGain * scale;
  }
}

function sampleFluidForBodyCoupling(field, n, x, y) {
  const ix = clamp(Math.round(x), 0, n - 1);
  const iy = clamp(Math.round(y), 0, n - 1);
  return field[iy * n + ix] || 0;
}

function viscosityMotionResponse(honey, vmaxBase) {
  const h = Math.max(0, Number(honey) || 0);
  return {
    damp: Math.max(0.72, 0.985 - h * 0.045),
    vmax: Math.max(0.4, vmaxBase - h * 0.25),
  };
}

function runBaselineSoftFluidCoupling(args) {
  const {
    sim, soft, n, dt, dtNorm, vxField, vyField, dragK, swimGain, localHoneyDrag,
    obstacleMask, bodyFeedbackPrevVx, bodyFeedbackPrevVy, selfFeedbackSuppression,
    softMembraneClusterSet, softNodeMomentumScale, constants,
  } = args;
  const {
    SOFT_NODE_FLOW_COUPLING,
    SOFT_NODE_LOCAL_FLOW_SHARE,
    SOFT_CLUSTER_TUG_COUPLING,
    SOFT_CLUSTER_RELATIVE_DRAG,
    softClusterFluidTorqueCoupling,
    SOFT_CLUSTER_LINEAR_PROJECTION,
    softClusterAngularProjection,
  } = constants;

  const softCentroid = computeSoftCentroid(soft.nodes);
  let softClusterKinematics = computeSoftClusterKinematics(soft.nodes);
  const clusterCarryMap = new Map();
  const clusterFluidLoadMap = new Map();
  const ensureClusterLoad = (cid) => {
    if (!clusterFluidLoadMap.has(cid)) clusterFluidLoadMap.set(cid, { forceX: 0, forceY: 0, torque: 0, count: 0 });
    return clusterFluidLoadMap.get(cid);
  };
  let softCarryTransfer = 0;

  for (let i = 0; i < soft.nodes.length; i++) {
    const node = soft.nodes[i];
    const mass = Math.max(0.02, node.mass);
    const invMass = 1 / mass;
    const cx = node.x - softCentroid.x;
    const cy = node.y - softCentroid.y;
    const activeSwimPhase = sim.frame * 0.12 + i * 1.57;
    const activeSwimAmp = 0.008 * (1 + 0.2 * Math.sin(sim.frame * 0.05 + i));
    const swimX = swimGain * (-cy * activeSwimAmp + Math.cos(activeSwimPhase) * 0.004) * invMass;
    const swimY = swimGain * (cx * activeSwimAmp + Math.sin(activeSwimPhase) * 0.004) * invMass;
    const honey = localHoneyDrag(node.x, node.y);
    const cid = node.clusterId ?? 0;
    const isMembraneCluster = softMembraneClusterSet.has(cid);
    const nodeMomentum = softNodeMomentumScale(i);
    const flowCouplingBase = isMembraneCluster ? (SOFT_NODE_FLOW_COUPLING * 0.88) : SOFT_NODE_FLOW_COUPLING;
    const flowCoupling = flowCouplingBase * nodeMomentum;

    const clusterKin = softClusterKinematics.get(cid);
    const clusterX = Number.isFinite(Number(clusterKin?.x)) ? Number(clusterKin.x) : softCentroid.x;
    const clusterY = Number.isFinite(Number(clusterKin?.y)) ? Number(clusterKin.y) : softCentroid.y;
    const clusterVx = Number.isFinite(Number(clusterKin?.vx)) ? Number(clusterKin.vx) : 0;
    const clusterVy = Number.isFinite(Number(clusterKin?.vy)) ? Number(clusterKin.vy) : 0;
    const clusterOmega = Number.isFinite(Number(clusterKin?.omega)) ? Number(clusterKin.omega) : 0;

    const rx = node.x - clusterX;
    const ry = node.y - clusterY;
    const fx = sampleFluidForBodyCoupling(vxField, n, node.x, node.y, rx, ry, obstacleMask, bodyFeedbackPrevVx, selfFeedbackSuppression);
    const fy = sampleFluidForBodyCoupling(vyField, n, node.x, node.y, rx, ry, obstacleMask, bodyFeedbackPrevVy, selfFeedbackSuppression);
    const clusterLocalVx = clusterVx - clusterOmega * ry;
    const clusterLocalVy = clusterVy + clusterOmega * rx;

    const forceX = (fx - clusterLocalVx) * dragK * honey * flowCoupling * mass;
    const forceY = (fy - clusterLocalVy) * dragK * honey * flowCoupling * mass;
    const carryX = forceX * invMass;
    const carryY = forceY * invMass;

    const localCarryX = (fx - node.vx) * dragK * honey * invMass * flowCoupling * SOFT_NODE_LOCAL_FLOW_SHARE;
    const localCarryY = (fy - node.vy) * dragK * honey * invMass * flowCoupling * SOFT_NODE_LOCAL_FLOW_SHARE;

    node.vx += localCarryX * dt * 60 + swimX * dtNorm;
    node.vy += localCarryY * dt * 60 + swimY * dtNorm;

    const load = ensureClusterLoad(cid);
    load.forceX += forceX;
    load.forceY += forceY;
    load.torque += rx * forceY - ry * forceX;
    load.count += 1;

    const st = clusterCarryMap.get(cid) || { sumX: 0, sumY: 0, count: 0, maxX: 0, maxY: 0, maxMag: 0 };
    st.sumX += carryX;
    st.sumY += carryY;
    st.count += 1;
    const cmag = Math.hypot(carryX, carryY);
    if (cmag > st.maxMag) {
      st.maxMag = cmag;
      st.maxX = carryX;
      st.maxY = carryY;
    }
    clusterCarryMap.set(cid, st);

    const softVisc = viscosityMotionResponse(honey, 2.8);
    node.vx *= softVisc.damp;
    node.vy *= softVisc.damp;
    const nMax = softVisc.vmax;
    const nMag = Math.hypot(node.vx, node.vy);
    if (nMag > nMax) {
      node.vx = (node.vx / nMag) * nMax;
      node.vy = (node.vy / nMag) * nMax;
    }
    softCarryTransfer += Math.hypot(carryX, carryY);
  }

  const clusterAccelMap = new Map();
  for (const [cid, load] of clusterFluidLoadMap.entries()) {
    const clusterKin = softClusterKinematics.get(cid);
    const clusterMass = Math.max(0.02, Number(clusterKin?.mass) || (Math.max(1, load.count) * sim.controls.massSoft));
    const clusterInertia = Math.max(1e-4, Number(clusterKin?.inertia) || 1e-4);
    clusterAccelMap.set(cid, {
      x: Number.isFinite(Number(clusterKin?.x)) ? Number(clusterKin.x) : softCentroid.x,
      y: Number.isFinite(Number(clusterKin?.y)) ? Number(clusterKin.y) : softCentroid.y,
      ax: load.forceX / clusterMass,
      ay: load.forceY / clusterMass,
      alpha: load.torque / clusterInertia,
    });
  }

  for (const node of soft.nodes) {
    const cid = node.clusterId ?? 0;
    const acc = clusterAccelMap.get(cid);
    if (!acc) continue;
    const rx = node.x - acc.x;
    const ry = node.y - acc.y;
    const flowShare = softMembraneClusterSet.has(cid) ? (softClusterFluidTorqueCoupling * 0.7) : softClusterFluidTorqueCoupling;
    node.vx += (acc.ax - acc.alpha * ry) * flowShare * dt * 60;
    node.vy += (acc.ay + acc.alpha * rx) * flowShare * dt * 60;
  }

  for (const node of soft.nodes) {
    const cid = node.clusterId ?? 0;
    const st = clusterCarryMap.get(cid);
    if (!st || st.count <= 0) continue;
    const meanX = st.sumX / st.count;
    const meanY = st.sumY / st.count;
    const pullX = meanX * 0.6 + st.maxX * 0.4;
    const pullY = meanY * 0.6 + st.maxY * 0.4;
    const tugCoupling = softMembraneClusterSet.has(cid) ? (SOFT_CLUSTER_TUG_COUPLING * 0.45) : (SOFT_CLUSTER_TUG_COUPLING * 0.72);
    node.vx += pullX * tugCoupling * dt * 60;
    node.vy += pullY * tugCoupling * dt * 60;
  }

  const clusterVelMap = new Map();
  for (const node of soft.nodes) {
    const cid = node.clusterId ?? 0;
    const st = clusterVelMap.get(cid) || { sumVx: 0, sumVy: 0, count: 0 };
    st.sumVx += node.vx || 0;
    st.sumVy += node.vy || 0;
    st.count += 1;
    clusterVelMap.set(cid, st);
  }
  for (const node of soft.nodes) {
    const cid = node.clusterId ?? 0;
    const st = clusterVelMap.get(cid);
    if (!st || st.count <= 0) continue;
    const meanVx = st.sumVx / st.count;
    const meanVy = st.sumVy / st.count;
    const relDamp = softMembraneClusterSet.has(cid) ? (SOFT_CLUSTER_RELATIVE_DRAG * 0.65) : SOFT_CLUSTER_RELATIVE_DRAG;
    node.vx -= (node.vx - meanVx) * relDamp * dtNorm;
    node.vy -= (node.vy - meanVy) * relDamp * dtNorm;
  }

  softClusterKinematics = computeSoftClusterKinematics(soft.nodes);
  projectNodesTowardClusterRigidMotion(soft.nodes, softClusterKinematics, {
    linearGain: SOFT_CLUSTER_LINEAR_PROJECTION * dtNorm,
    angularGain: softClusterAngularProjection * dtNorm,
    membraneClusterSet: softMembraneClusterSet,
    membraneGainScale: 0.72,
  });

  return { softCarryTransfer };
}

test('soft fluid coupling parity: gpu-only module matches baseline cluster-coupled velocity stepping', () => {
  const n = 24;
  const cells = n * n;
  const vxField = new Float32Array(cells);
  const vyField = new Float32Array(cells);
  for (let i = 0; i < cells; i++) {
    vxField[i] = Math.sin(i * 0.13) * 0.35;
    vyField[i] = Math.cos(i * 0.09) * 0.28;
  }

  const seedSoft = {
    nodes: [
      { x: 6.2, y: 6.9, vx: 0.2, vy: -0.1, mass: 1.0, clusterId: 0 },
      { x: 7.1, y: 6.4, vx: -0.15, vy: 0.23, mass: 1.1, clusterId: 0 },
      { x: 14.0, y: 13.6, vx: 0.31, vy: -0.08, mass: 0.9, clusterId: 1 },
      { x: 14.8, y: 14.3, vx: -0.05, vy: 0.27, mass: 1.2, clusterId: 1 },
    ],
  };

  const baseSim = { frame: 37, controls: { massSoft: 1.0 } };
  const constants = {
    SOFT_NODE_FLOW_COUPLING: 0.052,
    SOFT_NODE_LOCAL_FLOW_SHARE: 0.72,
    SOFT_CLUSTER_TUG_COUPLING: 0.095,
    SOFT_CLUSTER_RELATIVE_DRAG: 0.065,
    softClusterFluidTorqueCoupling: 0.082,
    SOFT_CLUSTER_LINEAR_PROJECTION: 0.24,
    softClusterAngularProjection: 0.11,
  };

  const baselineSoft = structuredClone(seedSoft);
  const gpuSoft = structuredClone(seedSoft);

  const sharedArgs = {
    n,
    dt: 0.016,
    dtNorm: 1,
    vxField,
    vyField,
    dragK: 0.35,
    swimGain: 0.58,
    localHoneyDrag: () => 0.35,
    viscosityMotionResponse,
    obstacleMask: null,
    bodyFeedbackPrevVx: new Float32Array(cells),
    bodyFeedbackPrevVy: new Float32Array(cells),
    selfFeedbackSuppression: 0.82,
    softMembraneClusterSet: new Set([1]),
    softNodeMomentumScale: (idx) => (idx % 2 === 0 ? 0.8 : 1.0),
    constants,
    computeSoftCentroid,
    computeSoftClusterKinematics,
    projectNodesTowardClusterRigidMotion,
    sampleFluidForBodyCoupling,
  };

  const baselineResult = runBaselineSoftFluidCoupling({ sim: structuredClone(baseSim), soft: baselineSoft, ...sharedArgs });
  const gpuResult = applySoftFluidCouplingGpuOnly({ sim: structuredClone(baseSim), soft: gpuSoft, ...sharedArgs });

  assert.ok(Math.abs(gpuResult.softCarryTransfer - baselineResult.softCarryTransfer) < 1e-9);
  assert.equal(gpuSoft.nodes.length, baselineSoft.nodes.length);
  for (let i = 0; i < baselineSoft.nodes.length; i++) {
    const b = baselineSoft.nodes[i];
    const g = gpuSoft.nodes[i];
    assert.ok(Math.abs(g.vx - b.vx) < 1e-9, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-9, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    assert.ok(Math.abs(g.x - b.x) < 1e-12, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-12, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
  }
});
