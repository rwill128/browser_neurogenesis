/**
 * GPU-only runtime soft fluid-coupling path.
 * Isolates soft-node fluid carry, cluster load redistribution, and
 * cluster rigid-motion projection from baseline loops in gpu-lab.js.
 */
export function applySoftFluidCouplingGpuOnly({
  sim,
  soft,
  n,
  dt,
  dtNorm,
  vxField,
  vyField,
  dragK,
  swimGain,
  localHoneyDrag,
  viscosityMotionResponse,
  obstacleMask,
  bodyFeedbackPrevVx,
  bodyFeedbackPrevVy,
  selfFeedbackSuppression,
  softMembraneClusterSet,
  softNodeMomentumScale,
  constants,
  computeSoftCentroid,
  computeSoftClusterKinematics,
  projectNodesTowardClusterRigidMotion,
  sampleFluidForBodyCoupling,
}) {
  const {
    SOFT_NODE_FLOW_COUPLING,
    SOFT_NODE_LOCAL_FLOW_SHARE,
    SOFT_CLUSTER_TUG_COUPLING,
    SOFT_CLUSTER_RELATIVE_DRAG,
    softClusterFluidTorqueCoupling,
    SOFT_CLUSTER_LINEAR_PROJECTION,
    softClusterAngularProjection,
  } = constants;

  const nodes = soft?.nodes || [];
  const softCentroid = computeSoftCentroid(nodes);
  let softClusterKinematics = computeSoftClusterKinematics(nodes);
  const clusterCarryMap = new Map();
  const clusterFluidLoadMap = new Map();
  const ensureClusterLoad = (cid) => {
    if (!clusterFluidLoadMap.has(cid)) {
      clusterFluidLoadMap.set(cid, { forceX: 0, forceY: 0, torque: 0, count: 0 });
    }
    return clusterFluidLoadMap.get(cid);
  };

  let softCarryTransfer = 0;

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
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
    const fx = sampleFluidForBodyCoupling(
      vxField,
      n,
      node.x,
      node.y,
      rx,
      ry,
      obstacleMask,
      bodyFeedbackPrevVx,
      selfFeedbackSuppression,
    );
    const fy = sampleFluidForBodyCoupling(
      vyField,
      n,
      node.x,
      node.y,
      rx,
      ry,
      obstacleMask,
      bodyFeedbackPrevVy,
      selfFeedbackSuppression,
    );
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

  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const acc = clusterAccelMap.get(cid);
    if (!acc) continue;
    const rx = node.x - acc.x;
    const ry = node.y - acc.y;
    const flowShare = softMembraneClusterSet.has(cid)
      ? (softClusterFluidTorqueCoupling * 0.7)
      : softClusterFluidTorqueCoupling;
    node.vx += (acc.ax - acc.alpha * ry) * flowShare * dt * 60;
    node.vy += (acc.ay + acc.alpha * rx) * flowShare * dt * 60;
  }

  for (const node of nodes) {
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
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const st = clusterVelMap.get(cid) || { sumVx: 0, sumVy: 0, count: 0 };
    st.sumVx += node.vx || 0;
    st.sumVy += node.vy || 0;
    st.count += 1;
    clusterVelMap.set(cid, st);
  }
  for (const node of nodes) {
    const cid = node.clusterId ?? 0;
    const st = clusterVelMap.get(cid);
    if (!st || st.count <= 0) continue;
    const meanVx = st.sumVx / st.count;
    const meanVy = st.sumVy / st.count;
    const relDamp = softMembraneClusterSet.has(cid) ? (SOFT_CLUSTER_RELATIVE_DRAG * 0.65) : SOFT_CLUSTER_RELATIVE_DRAG;
    node.vx -= (node.vx - meanVx) * relDamp * dtNorm;
    node.vy -= (node.vy - meanVy) * relDamp * dtNorm;
  }

  softClusterKinematics = computeSoftClusterKinematics(nodes);
  projectNodesTowardClusterRigidMotion(nodes, softClusterKinematics, {
    linearGain: SOFT_CLUSTER_LINEAR_PROJECTION * dtNorm,
    angularGain: softClusterAngularProjection * dtNorm,
    membraneClusterSet: softMembraneClusterSet,
    membraneGainScale: 0.72,
  });

  return {
    softCarryTransfer,
    softCentroid,
    softClusterKinematics,
  };
}
