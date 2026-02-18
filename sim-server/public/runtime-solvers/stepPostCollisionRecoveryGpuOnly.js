import { applyRigidInsideCorrectionPassGpuOnly } from './stepRigidInsideCorrectionGpuOnly.js';
import {
  buildSoftClusterKinematicsWgslPrep,
  computeSoftClusterKinematicsGpuOnly,
  projectNodesTowardClusterRigidMotionGpuOnly,
} from './stepSoftClusterKinematicsGpuOnly.js';
import { applySoftMembraneInsideCorrectionPassGpuOnly } from './stepSoftMembraneInsideCorrectionGpuOnly.js';

export async function applyPostCollisionRecoveryGpuOnly(args = {}) {
  const {
    bodies,
    soft,
    dtNorm,
    softMembraneClusterSet,
    softClusterCollisionLinearProjection,
    softClusterCollisionAngularProjection,
    membraneGainScale = 0.72,
    computeSoftClusterKinematics,
    projectNodesTowardClusterRigidMotion,
    applyRigidInsideCorrectionPass,
    applyMembraneInsideCorrectionPass,
    applyBounceBoundary,
    applyCollisionBoundaryPassGpuOnly,
    wgslOffload,
    n,
    softClusterLoops,
  } = args;

  if (!bodies || !soft || !Array.isArray(soft.nodes)) {
    return { rigidInsideCorrections: 0, membraneInsideCorrections: 0, postCollisionClusterKinematics: new Map() };
  }

  const dtNormSafe = Math.max(0, Number(dtNorm) || 0);
  const linearGain = (Number(softClusterCollisionLinearProjection) || 0) * dtNormSafe;
  const angularGain = (Number(softClusterCollisionAngularProjection) || 0) * dtNormSafe;

  const computeKinematics =
    typeof computeSoftClusterKinematics === 'function'
      ? computeSoftClusterKinematics
      : computeSoftClusterKinematicsGpuOnly;
  const projectTowardRigidMotion =
    typeof projectNodesTowardClusterRigidMotion === 'function'
      ? projectNodesTowardClusterRigidMotion
      : projectNodesTowardClusterRigidMotionGpuOnly;

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const prep = buildSoftClusterKinematicsWgslPrep(soft.nodes);
    wgslOffload.state.preparedSoftClusterPlan = prep.plan;
    wgslOffload.state.preparedSoftClusterLayout = prep.layout;
    wgslOffload.state.preparedSoftClusterSignature = prep.signature;
    wgslOffload.state.lastPreparedSoftClusterNodeCount = prep.plan.nodeCount;
    wgslOffload.state.lastPreparedSoftClusterCount = prep.plan.clusterCount;
    wgslOffload.state.lastPreparedSoftClusterLayoutBytes = prep.layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared-soft-cluster-kinematics';
  }

  const postCollisionClusterKinematics = await Promise.resolve(computeKinematics(soft.nodes));
  await Promise.resolve(projectTowardRigidMotion(soft.nodes, postCollisionClusterKinematics, {
    linearGain,
    angularGain,
    membraneClusterSet: softMembraneClusterSet,
    membraneGainScale: Number(membraneGainScale) || 0.72,
  }));

  const rigidInsideCorrections =
    typeof applyRigidInsideCorrectionPass === 'function'
      ? applyRigidInsideCorrectionPass(bodies, soft, args.hybridAttachedByRigid)
      : applyRigidInsideCorrectionPassGpuOnly({
        rigidBodies: bodies.rigid,
        soft,
        hybridAttachedByRigid: args.hybridAttachedByRigid,
      });

  const membraneInsideCorrections =
    typeof applyMembraneInsideCorrectionPass === 'function'
      ? applyMembraneInsideCorrectionPass(args.sim, soft, softClusterLoops)
      : applySoftMembraneInsideCorrectionPassGpuOnly({
        sim: args.sim,
        soft,
        loops: softClusterLoops,
      });

  let boundaryRuntime = { mode: 'cpu-inline', reason: 'bounce-callback' };
  if (typeof applyCollisionBoundaryPassGpuOnly === 'function') {
    boundaryRuntime =
      (await Promise.resolve(applyCollisionBoundaryPassGpuOnly({
        rigidBodies: bodies.rigid,
        soft,
        n,
        rigidBounce: 0.84,
        softBounce: 0.78,
        applyBounceBoundary,
        wgslOffload,
      }))) || { mode: 'cpu-fallback', reason: 'unknown' };
  } else if (typeof applyBounceBoundary === 'function') {
    for (const rb of (bodies.rigid || [])) applyBounceBoundary(rb, n, 0.84);
    for (const sn of (soft.nodes || [])) applyBounceBoundary(sn, n, 0.78);
  } else {
    boundaryRuntime = { mode: 'cpu-inline', reason: 'no-boundary-handler' };
  }

  return {
    rigidInsideCorrections,
    membraneInsideCorrections,
    postCollisionClusterKinematics,
    boundaryRuntime,
  };
}
