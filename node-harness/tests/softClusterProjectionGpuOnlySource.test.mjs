import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const kinematicsSource = fs.readFileSync(new URL('../../sim-server/public/runtime-solvers/stepSoftClusterKinematicsGpuOnly.js', import.meta.url), 'utf8');
const recoverySource = fs.readFileSync(new URL('../../sim-server/public/runtime-solvers/stepPostCollisionRecoveryGpuOnly.js', import.meta.url), 'utf8');

test('soft cluster kinematics gpu-only module lands concrete WGSL rigid-motion projection proposal + authoritative apply/fallback routing', () => {
  assert.match(
    kinematicsSource,
    /const softClusterProjectionProposalWgsl = \/\* wgsl \*\/[\s\S]*deltaVxOut\[i\][\s\S]*deltaVyOut\[i\]/,
    'expected soft-cluster kinematics module to include a concrete WGSL projection kernel that emits per-node velocity deltas',
  );

  assert.match(
    kinematicsSource,
    /export async function projectNodesTowardClusterRigidMotionWithWgslGpuOnly\([\s\S]*enableAuthoritativeSoftClusterProjection === true[\s\S]*lastSoftClusterProjectionAuthoritativeSource = useWgslAuthoritative\s*\?[\s\S]*'wgsl-soft-cluster-rigid-motion-projection-authoritative'[\s\S]*'cpu-soft-cluster-rigid-motion-projection-authoritative'/,
    'expected projection stage to gate authoritative WGSL replay behind explicit flag/signature checks while preserving explicit CPU fallback route telemetry',
  );
});

test('post-collision recovery routes projection stage through new WGSL-capable projection authority path', () => {
  assert.match(
    recoverySource,
    /projectNodesTowardClusterRigidMotionWithWgslGpuOnly/,
    'expected post-collision recovery module to wire projection stage through WGSL-capable authoritative helper',
  );

  assert.match(
    recoverySource,
    /projectTowardRigidMotion\(soft\.nodes, postCollisionClusterKinematics, \{[\s\S]*wgslOffload,[\s\S]*\}\)/,
    'expected post-collision recovery to pass wgslOffload into projection stage so source-route fallback checks stay explicit',
  );
});
