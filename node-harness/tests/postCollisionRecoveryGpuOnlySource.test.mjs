import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepPostCollisionRecoveryGpuOnly.js'), 'utf8');

test('post-collision recovery gpu-only source routes boundary pass through collision-boundary module when provided', () => {
  assert.match(
    source,
    /if \(typeof applyCollisionBoundaryPassGpuOnly === 'function'\) \{[\s\S]*applyCollisionBoundaryPassGpuOnly\(\{[\s\S]*rigidBounce: 0\.84,[\s\S]*softBounce: 0\.78,[\s\S]*wgslOffload,[\s\S]*\}\)/,
    'expected post-collision recovery to delegate boundary pass via gpu-only collision boundary module',
  );

  assert.match(
    source,
    /boundaryRuntime =\s*\{ mode: 'cpu-inline', reason: 'bounce-callback' \};[\s\S]*else if \(typeof applyBounceBoundary === 'function'\)/,
    'expected legacy CPU bounce callback path to remain available as fallback',
  );

  assert.match(
    source,
    /return \{[\s\S]*boundaryRuntime,[\s\S]*\};/,
    'expected boundary runtime source-route telemetry to be returned to gpu-only orchestrator',
  );
});

test('post-collision recovery gpu-only source publishes deterministic soft-cluster WGSL prep metadata in offload state', () => {
  assert.match(
    source,
    /buildSoftClusterKinematicsWgslPrep\(soft\.nodes\)[\s\S]*preparedSoftClusterPlan = prep\.plan;[\s\S]*preparedSoftClusterLayout = prep\.layout;[\s\S]*preparedSoftClusterSignature = prep\.signature;[\s\S]*lastPreparedSoftClusterLayoutBytes = prep\.layout\.byteLength;[\s\S]*lastMode = 'cpu-prepared-soft-cluster-kinematics';/,
    'expected gpu-only post-collision recovery path to persist deterministic soft-cluster prep ownership for next WGSL reduction stage',
  );
});

test('post-collision recovery gpu-only source dispatches concrete WGSL soft-cluster kinematics probe with source-route telemetry', () => {
  assert.match(
    source,
    /const SOFT_CLUSTER_KINEMATICS_PROBE_WGSL = \/\* wgsl \*\/[\s\S]*@compute @workgroup_size\(\$\{WGSL_WORKGROUP_SIZE\}\)[\s\S]*out_vx_mass\[ci\] = sum_vx_mass;[\s\S]*out_vy_mass\[ci\] = sum_vy_mass;/,
    'expected concrete WGSL soft-cluster kinematics probe shader for per-cluster mass-weighted sums',
  );

  assert.match(
    source,
    /if \(canUseWgslOffload\(wgslOffload\) && prep\.plan\.clusterCount > 0\) \{[\s\S]*dispatchSoftClusterKinematicsProbe\(wgslOffload, prep\)[\s\S]*lastMode = 'wgsl-soft-cluster-kinematics-probe';[\s\S]*lastSoftClusterProbeSource = 'wgsl-soft-cluster-kinematics-probe';/,
    'expected gpu-only post-collision recovery to run serialized WGSL soft-cluster probe dispatch and publish source-route ownership',
  );

  assert.match(
    source,
    /state\.lastSoftClusterProbeParity = computeSoftClusterMassParity\(prep\.layout, probe\);/,
    'expected WGSL soft-cluster probe to persist deterministic CPU-vs-WGSL parity telemetry',
  );
});
