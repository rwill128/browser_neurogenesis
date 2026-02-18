/**
 * GPU-only runtime soft spring-rest recovery pass.
 *
 * Owns soft spring-rest stabilization parameterization for the isolated gpu-only
 * runtime solver path so baseline/default stepping remains untouched.
 */

function pickProfile(profile, baseline, warning, severe) {
  return profile === 'baseline' ? baseline : (profile === 'warning' ? warning : severe);
}

export function buildSoftRestRecoveryOptionsGpuOnly({
  severeInterventionsOn,
  warningInterventionsOn,
  deform = null,
} = {}) {
  const severeProfile = severeInterventionsOn && (deform?.severeCollapseCount > 0);
  const warningProfile = warningInterventionsOn && !severeProfile && (deform?.warningCount > 0);
  const profile = severeProfile ? 'severe' : (warningProfile ? 'warning' : 'baseline');

  return {
    recoverRate: pickProfile(profile, 0.056, 0.036, 0.015),
    hardMinFactor: 0.7,
    hardMaxFactor: 1.45,
    jitterDeadband: 1e-5,
    adaptiveGainMax: pickProfile(profile, 2.4, 1.85, 1.35),
    adaptiveExponent: pickProfile(profile, 0.78, 0.9, 1.0),
    elongationBiasMax: pickProfile(profile, 1.22, 1.14, 1.08),
    compressionBiasMax: pickProfile(profile, 1.12, 1.08, 1.04),
    errorPivot: 0.16,
    outlierRecoveryCouplingMax: pickProfile(profile, 1.22, 1.12, 1.05),
    outlierErrorPivot: pickProfile(profile, 0.75, 0.85, 0.95),
    localEndpointCouplingMax: pickProfile(profile, 1.16, 1.08, 1.03),
    localDirectionalCouplingMax: pickProfile(profile, 1.1, 1.06, 1.02),
    localImbalanceCouplingMax: pickProfile(profile, 1.12, 1.07, 1.02),
    localConsensusCouplingMax: pickProfile(profile, 1.08, 1.04, 1.01),
    localOutlierCouplingMax: pickProfile(profile, 1.12, 1.07, 1.02),
    localOutlierErrorPivot: pickProfile(profile, 0.85, 0.95, 1.1),
    localErrorPivot: pickProfile(profile, 0.2, 0.25, 0.3),
    counterPolarityCouplingMax: pickProfile(profile, 1.09, 1.05, 1.02),
    smallRestRecoveryCouplingMax: pickProfile(profile, 1.14, 1.08, 1.0),
    smallRestPivot: pickProfile(profile, 0.9, 1.05, 1.2),
    lowErrorRecoveryCouplingMax: pickProfile(profile, 1.12, 1.06, 1.0),
    lowErrorRecoveryGate: pickProfile(profile, 0.08, 0.065, 0.05),
  };
}

export function applySoftRestRecoveryGpuOnly({
  springs,
  restBaseline,
  severeInterventionsOn,
  warningInterventionsOn,
  deform,
  recoverSoftSpringRests,
}) {
  if (!Array.isArray(springs) || !restBaseline || springs.length !== restBaseline.length) return;
  if (typeof recoverSoftSpringRests !== 'function') {
    throw new Error('gpu-only soft rest-recovery pass requires recoverSoftSpringRests callback');
  }

  const options = buildSoftRestRecoveryOptionsGpuOnly({
    severeInterventionsOn,
    warningInterventionsOn,
    deform,
  });
  recoverSoftSpringRests(springs, restBaseline, options);
}
