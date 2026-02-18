export function applyPostCollisionRecoveryGpuOnly(args = {}) {
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
    n,
    softClusterLoops,
  } = args;

  if (!bodies || !soft || !Array.isArray(soft.nodes)) {
    return { rigidInsideCorrections: 0, membraneInsideCorrections: 0, postCollisionClusterKinematics: new Map() };
  }

  const dtNormSafe = Math.max(0, Number(dtNorm) || 0);
  const linearGain = (Number(softClusterCollisionLinearProjection) || 0) * dtNormSafe;
  const angularGain = (Number(softClusterCollisionAngularProjection) || 0) * dtNormSafe;

  const postCollisionClusterKinematics =
    typeof computeSoftClusterKinematics === 'function'
      ? computeSoftClusterKinematics(soft.nodes)
      : new Map();

  if (typeof projectNodesTowardClusterRigidMotion === 'function') {
    projectNodesTowardClusterRigidMotion(soft.nodes, postCollisionClusterKinematics, {
      linearGain,
      angularGain,
      membraneClusterSet: softMembraneClusterSet,
      membraneGainScale: Number(membraneGainScale) || 0.72,
    });
  }

  const rigidInsideCorrections =
    typeof applyRigidInsideCorrectionPass === 'function'
      ? applyRigidInsideCorrectionPass(bodies, soft, args.hybridAttachedByRigid)
      : 0;

  const membraneInsideCorrections =
    typeof applyMembraneInsideCorrectionPass === 'function'
      ? applyMembraneInsideCorrectionPass(args.sim, soft, softClusterLoops)
      : 0;

  if (typeof applyBounceBoundary === 'function') {
    for (const rb of (bodies.rigid || [])) applyBounceBoundary(rb, n, 0.84);
    for (const sn of (soft.nodes || [])) applyBounceBoundary(sn, n, 0.78);
  }

  return {
    rigidInsideCorrections,
    membraneInsideCorrections,
    postCollisionClusterKinematics,
  };
}
