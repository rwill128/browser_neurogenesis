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
    'expected deterministic plan/layout prep, proposal signature routing, and serialized WGSL probe+proposal source-route reporting before CPU-authoritative rest recovery',
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
    /dispatchSoftRestRecoveryWgslProposal\([\s\S]*computeRestProposalParity\([\s\S]*lastProposalParity = \{[\s\S]*source: 'wgsl-rest-recovery-proposal'/,
    'expected WGSL rest proposal branch to persist deterministic CPU-vs-WGSL parity telemetry and source route',
  );
});
