import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const gpuLabSource = readFileSync(resolve(ROOT, 'sim-server/public/gpu-lab.js'), 'utf8');
const moduleSource = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftRestRecoveryGpuOnly.js'), 'utf8');

test('gpu-lab wires soft rest-recovery gpu-only pass with WGSL offload context + isolated state bag', () => {
  assert.match(
    gpuLabSource,
    /applySoftRestRecoveryGpuOnly\(\{[\s\S]*softNodes: s\.nodes,[\s\S]*wgslOffload:[\s\S]*enabled: true,[\s\S]*device: sim\?\.device,[\s\S]*state: \(sim\.softRestRecoveryWgslState \|\|= \{\}\),[\s\S]*\}\);/,
    'expected gpu-lab rest-recovery dispatch to wire optional WGSL context in gpu-only branch while baseline stays untouched',
  );
});

test('soft rest-recovery gpu-only module includes concrete WGSL strain probe stage and source-route telemetry', () => {
  assert.match(
    moduleSource,
    /const softRestRecoveryProbeWgsl = \/\* wgsl \*\/[\s\S]*@compute @workgroup_size\([\s\S]*strainOut\[si\] = \(d - rest\) \/ rest;/,
    'expected concrete WGSL compute probe for spring strain in rest-recovery module',
  );

  assert.match(
    moduleSource,
    /buildSoftRestRecoveryWgslPlan\([\s\S]*buildSoftRestRecoveryWgslLayout\([\s\S]*computeSoftRestRecoveryProposalSignature\([\s\S]*lastPreparedProposalSignature[\s\S]*pendingWgslRestRecoveryProbePromise[\s\S]*dispatchSoftRestRecoveryWgslProbe\([\s\S]*dispatchSoftRestRecoveryWgslProposal\([\s\S]*lastMode = proposalRan[\s\S]*'wgsl-rest-recovery-proposal'/,
    'expected deterministic plan/layout prep, proposal signature routing, and serialized WGSL probe+proposal source-route reporting',
  );

  assert.match(
    moduleSource,
    /canApplyAuthoritativeRestRecoveryProposal\([\s\S]*enableAuthoritativeRestRecovery[\s\S]*applyAuthoritativeRestRecoveryProposal\([\s\S]*lastAuthoritativeSource = useAuthoritativeProposal[\s\S]*'wgsl-rest-recovery-authoritative'/,
    'expected cached WGSL proposal to gate authoritative rest-length ownership in gpu-only path when parity + signature checks pass',
  );
});


test('soft rest-recovery gpu-only module dispatches concrete WGSL rest-length proposal stage with parity telemetry', () => {
  assert.match(
    moduleSource,
    /const softRestRecoveryProposalWgsl = \/\* wgsl \*\/[\s\S]*springRestProposalOut\[si\] = clamp\(next, minRest, maxRest\);/,
    'expected concrete WGSL rest-length proposal stage in rest-recovery module',
  );

  assert.match(
    moduleSource,
    /dispatchSoftRestRecoveryWgslProposal\([\s\S]*buildCpuRestProposalBySpring\([\s\S]*computeRestProposalParity\([\s\S]*lastCpuProposalBySpring = cpuProposalBySpring;[\s\S]*source: 'wgsl-rest-recovery-proposal-vs-cpu'/,
    'expected WGSL rest proposal branch to persist deterministic CPU-vs-WGSL parity telemetry and source route',
  );
});


test('soft rest-recovery gpu-only fast mode skips cpu parity while preserving finite safety + source-route telemetry', () => {
  assert.match(
    moduleSource,
    /isGpuOnlyFastMode\(offload\)[\s\S]*checkFiniteFloat32Array\(proposalBySpring\)[\s\S]*if \(!fastMode\) \{[\s\S]*buildCpuRestProposalBySpring\([\s\S]*state\.lastProposalFinite = finite;[\s\S]*if \(fastMode\) \{[\s\S]*source: 'wgsl-rest-recovery-proposal-fast'[\s\S]*validation: 'skipped-cpu-parity'[\s\S]*lastProposalSource = 'wgsl-rest-recovery-proposal-fast'/,
    'expected rest-recovery fast mode to bypass cpu parity/reference work while retaining finite checks and explicit fast source-route ownership',
  );

  assert.match(
    moduleSource,
    /canApplyAuthoritativeRestRecoveryProposal\([\s\S]*const fastMode = isGpuOnlyFastMode\(wgslOffload\)[\s\S]*expectedSource = fastMode \? 'wgsl-rest-recovery-proposal-fast' : 'wgsl-rest-recovery-proposal'[\s\S]*if \(fastMode\) \{[\s\S]*state\.lastProposalFinite\?\.allFinite !== true/,
    'expected authoritative replay gate to switch to finite safety checks in fast mode while keeping validated mode parity gate intact',
  );
});
