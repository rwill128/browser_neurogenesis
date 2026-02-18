import test from 'node:test';
import assert from 'node:assert/strict';
import { applySoftRestRecoveryGpuOnly, buildSoftRestRecoveryOptionsGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftRestRecoveryGpuOnly.js';

function buildBaselineOptions({ severeInterventionsOn, warningInterventionsOn, deform }) {
  const severeProfile = severeInterventionsOn && deform.severeCollapseCount > 0;
  const warningProfile = warningInterventionsOn && !severeProfile && deform.warningCount > 0;
  const profile = severeProfile ? 'severe' : (warningProfile ? 'warning' : 'baseline');
  const pick = (baseline, warning, severe) => (profile === 'baseline' ? baseline : (profile === 'warning' ? warning : severe));
  return {
    recoverRate: pick(0.056, 0.036, 0.015),
    hardMinFactor: 0.7,
    hardMaxFactor: 1.45,
    jitterDeadband: 1e-5,
    adaptiveGainMax: pick(2.4, 1.85, 1.35),
    adaptiveExponent: pick(0.78, 0.9, 1.0),
    elongationBiasMax: pick(1.22, 1.14, 1.08),
    compressionBiasMax: pick(1.12, 1.08, 1.04),
    errorPivot: 0.16,
    outlierRecoveryCouplingMax: pick(1.22, 1.12, 1.05),
    outlierErrorPivot: pick(0.75, 0.85, 0.95),
    localEndpointCouplingMax: pick(1.16, 1.08, 1.03),
    localDirectionalCouplingMax: pick(1.1, 1.06, 1.02),
    localImbalanceCouplingMax: pick(1.12, 1.07, 1.02),
    localConsensusCouplingMax: pick(1.08, 1.04, 1.01),
    localOutlierCouplingMax: pick(1.12, 1.07, 1.02),
    localOutlierErrorPivot: pick(0.85, 0.95, 1.1),
    localErrorPivot: pick(0.2, 0.25, 0.3),
    counterPolarityCouplingMax: pick(1.09, 1.05, 1.02),
    smallRestRecoveryCouplingMax: pick(1.14, 1.08, 1.0),
    smallRestPivot: pick(0.9, 1.05, 1.2),
    lowErrorRecoveryCouplingMax: pick(1.12, 1.06, 1.0),
    lowErrorRecoveryGate: pick(0.08, 0.065, 0.05),
  };
}

for (const sample of [
  { name: 'baseline profile', severeInterventionsOn: true, warningInterventionsOn: true, deform: { severeCollapseCount: 0, warningCount: 0 } },
  { name: 'warning profile', severeInterventionsOn: true, warningInterventionsOn: true, deform: { severeCollapseCount: 0, warningCount: 2 } },
  { name: 'severe profile', severeInterventionsOn: true, warningInterventionsOn: true, deform: { severeCollapseCount: 1, warningCount: 4 } },
]) {
  test(`soft rest-recovery options parity: ${sample.name}`, () => {
    const baseline = buildBaselineOptions(sample);
    const gpuOnly = buildSoftRestRecoveryOptionsGpuOnly(sample);
    assert.deepEqual(gpuOnly, baseline);
  });
}

test('soft rest-recovery gpu-only keeps CPU-authoritative options/behavior contract', () => {
  const springs = [
    [0, 1, 2.0],
    [1, 2, 1.8],
  ];
  const restBaseline = [2.1, 1.75];
  const deform = { severeCollapseCount: 0, warningCount: 3 };
  const expected = buildSoftRestRecoveryOptionsGpuOnly({
    severeInterventionsOn: true,
    warningInterventionsOn: true,
    deform,
  });

  let called = 0;
  let gotSprings = null;
  let gotBaseline = null;
  let gotOptions = null;

  applySoftRestRecoveryGpuOnly({
    springs,
    softNodes: [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
    ],
    restBaseline,
    severeInterventionsOn: true,
    warningInterventionsOn: true,
    deform,
    recoverSoftSpringRests: (s, b, o) => {
      called += 1;
      gotSprings = s;
      gotBaseline = b;
      gotOptions = o;
    },
    wgslOffload: {
      enabled: true,
      // intentionally missing device so WGSL path is unavailable
      state: {},
    },
  });

  assert.equal(called, 1, 'recoverSoftSpringRests should remain authoritative and run exactly once');
  assert.equal(gotSprings, springs, 'springs ref should pass through unchanged');
  assert.equal(gotBaseline, restBaseline, 'rest baseline ref should pass through unchanged');
  assert.deepEqual(gotOptions, expected, 'gpu-only options must match baseline-compatible profile output');
  assert.equal(
    gotOptions && expected && gotOptions.recoverRate,
    expected.recoverRate,
    'cpu-authoritative rest-recovery tuning contract should remain unchanged while WGSL stages run as telemetry/proposal only',
  );
});

test('soft rest-recovery gpu-only defaults proposal source-route telemetry to cpu when wgsl is unavailable', () => {
  const wgslOffload = { enabled: true, state: {} };
  applySoftRestRecoveryGpuOnly({
    springs: [[0, 1, 1.2]],
    softNodes: [{ x: 0, y: 0 }, { x: 1, y: 0 }],
    restBaseline: [1.1],
    severeInterventionsOn: false,
    warningInterventionsOn: false,
    deform: { severeCollapseCount: 0, warningCount: 0 },
    recoverSoftSpringRests: () => {},
    wgslOffload,
  });

  assert.equal(
    wgslOffload.state.lastProposalSource,
    'cpu-rest-recovery-authoritative',
    'expected explicit proposal source-route fallback when wgsl offload cannot run',
  );
});
