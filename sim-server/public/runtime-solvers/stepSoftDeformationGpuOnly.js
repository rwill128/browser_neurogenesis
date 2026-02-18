export function applySoftDeformationInterventionsGpuOnly({
  sim,
  soft,
  softClusterLoops,
  severeInterventionsOn,
  buildSoftDeformationState,
  stabilizeSeverelyDeformedSoftClusters,
}) {
  if (!sim || !soft || typeof buildSoftDeformationState !== 'function') {
    throw new Error('applySoftDeformationInterventionsGpuOnly requires sim, soft, and buildSoftDeformationState');
  }

  let deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  if (severeInterventionsOn && deform?.severeCollapseCount > 0) {
    stabilizeSeverelyDeformedSoftClusters?.(sim, soft, softClusterLoops, deform);
    deform = buildSoftDeformationState(sim, soft, softClusterLoops);
  }

  return deform;
}
