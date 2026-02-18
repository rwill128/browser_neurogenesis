import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepPostCollisionRecoveryGpuOnly.js'), 'utf8');

test('post-collision recovery WGSL source-route supports authoritative soft-cluster mass replay with parity telemetry', () => {
  assert.match(
    source,
    /hasAuthoritativeSoftClusterProbe\([\s\S]*authoritativeSoftClusterMass !== true[\s\S]*lastSoftClusterProbeSignature[\s\S]*lastSoftClusterProbe\?\.mass instanceof Float32Array/,
    'expected authoritative soft-cluster mass probe gate to require explicit flag + deterministic signature + typed probe buffers',
  );

  assert.match(
    source,
    /computeSoftClusterKinematicsFromMassMomentsGpuOnly\([\s\S]*lastSoftClusterAuthoritativeParity = computeSoftClusterKinematicsParity\([\s\S]*lastAuthoritativeSoftClusterSource = hasAuthoritativeProbe[\s\S]*'wgsl-soft-cluster-mass-authoritative'[\s\S]*'cpu-soft-cluster-kinematics-authoritative'[\s\S]*lastSourceRoute = wgslOffload\.state\.lastAuthoritativeSoftClusterSource/,
    'expected authoritative WGSL mass replay path to publish parity and source-route ownership',
  );
});
