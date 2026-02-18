import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const gpuLabSource = readFileSync(resolve(ROOT, 'sim-server/public/gpu-lab.js'), 'utf8');
const moduleSource = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepRigidGpuOnly.js'), 'utf8');

test('gpu-lab wires rigid gpu-only pass with isolated WGSL offload state bag', () => {
  assert.match(
    gpuLabSource,
    /stepRigidBodiesGpuOnly\(\{[\s\S]*wgslOffload:[\s\S]*enabled: true,[\s\S]*device: sim\?\.device,[\s\S]*modeProfile: normalizeRuntimePipelineMode\(sim\?\.controls\?\.runtimePipelineMode, sim\?\.controls\?\.runtimeSolverPath\),[\s\S]*sim\.rigidStepWgslState \|\|= \{\}[\s\S]*enableAuthoritativeRigidStep !== false[\s\S]*enableAuthoritativeRigidStep = true/,
    'expected rigid gpu-only path to wire isolated WGSL offload state bag with explicit mode routing and default authoritative flag',
  );
});

test('rigid gpu-only module defines concrete WGSL rigid step proposal kernel with source-route telemetry + fallback', () => {
  assert.match(
    moduleSource,
    /const rigidStepProposalWgsl = \/\* wgsl \*\/[\s\S]*vx = vx \+ ax \* params\.dt \* 60\.0 \+ swimX \* params\.dtNorm;[\s\S]*omega = clamp\(omega, -0\.25, 0\.25\);[\s\S]*carryOut\[i\] = sqrt\(ax \* ax \+ ay \* ay\);/,
    'expected rigid step WGSL kernel to contain integration, clamping, and carry transfer math',
  );

  assert.match(
    moduleSource,
    /getGpuOnlyPipelineModeProfile\(offload\)[\s\S]*modeProfile === 'gpu-only-fast'[\s\S]*modeProfile === 'gpu-only-validated'[\s\S]*return 'standard';/,
    'expected rigid gpu-only path to normalize explicit standard/validated/fast mode routing',
  );

  assert.match(
    moduleSource,
    /dispatchRigidStepProposal\([\s\S]*lastRigidStepProposalSignature[\s\S]*lastRigidStepProposalSource = allFinite \?[\s\S]*'wgsl-rigid-step-proposal'[\s\S]*'cpu-rigid-step-authoritative-nonfinite'/,
    'expected rigid proposal dispatch to preserve explicit source-route telemetry and hard non-finite fallback',
  );

  assert.match(
    moduleSource,
    /pendingWgslRigidStepProposalPromise[\s\S]*dispatchRigidStepProposal\([\s\S]*lastRigidStepProposalSource = 'cpu-rigid-step-authoritative'[\s\S]*lastMode = 'cpu-rigid-step-authoritative'/,
    'expected serialized rigid WGSL dispatch with hard CPU fallback source-route when dispatch fails',
  );
});
