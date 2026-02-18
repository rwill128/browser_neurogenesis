import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftIntegrateGpuOnly.js'), 'utf8');

test('soft integrate WGSL source packs params with u32 count ABI and recreates bindGroup after buffer growth', () => {
  assert.match(
    source,
    /const paramsBuffer = new ArrayBuffer\(32\);[\s\S]*paramsView\.setUint32\(0, count, true\);[\s\S]*paramsView\.setFloat32\(16, dt, true\);[\s\S]*paramsView\.setFloat32\(20, softIntegrationScale, true\);[\s\S]*paramsView\.setFloat32\(24, hybridNodeVCap, true\);/,
    'expected WGSL params packing to preserve u32+f32 ABI alignment',
  );

  assert.match(
    source,
    /if \(\(state\.capacity \|\| 0\) < requiredCapacity\) \{[\s\S]*state\.bindGroup = null;[\s\S]*\}/,
    'expected buffer-capacity growth path to invalidate bindGroup so new storage buffers are routed',
  );
});

test('soft integrate gpu-only WGSL path records deterministic proposal signature + parity/source-route telemetry for authoritative bring-up', () => {
  assert.match(
    source,
    /computeSoftIntegrateProposalSignature\([\s\S]*state\.lastProposalSignature = proposalSignature >>> 0;[\s\S]*state\.lastSourceRoute = fastMode \? 'wgsl-integrate-proposal-fast' : 'wgsl-integrate-proposal';[\s\S]*buildCpuReferenceIntegration\([\s\S]*computeSoftIntegrateParity\([\s\S]*state\.lastParity = \{[\s\S]*source: 'wgsl-integrate-proposal-vs-cpu'/,
    'expected soft integrate WGSL validated branch to persist deterministic signature + parity metrics with proposal source-route ownership',
  );

  assert.match(
    source,
    /isGpuOnlyFastMode\(offload\)[\s\S]*checkFiniteFloat32Array\(outX\)[\s\S]*if \(fastMode\) \{[\s\S]*source: 'wgsl-integrate-proposal-fast'[\s\S]*validation: 'skipped-cpu-parity'/,
    'expected fast mode branch to skip cpu parity while retaining finite checks and explicit fast source-route telemetry',
  );

  assert.match(
    source,
    /wgslOffload\.state\.lastMode = isGpuOnlyFastMode\(wgslOffload\) \? 'wgsl-fast' : 'wgsl-validated';[\s\S]*wgslOffload\.state\.lastSourceRoute = isGpuOnlyFastMode\(wgslOffload\)[\s\S]*\? 'wgsl-integrate-authoritative-fast'[\s\S]*: 'wgsl-integrate-authoritative';/,
    'expected authoritative WGSL success path to publish validated-vs-fast source-route ownership',
  );
});

test('soft integrate gpu-only pipeline mode exposes explicit standard fallback semantics', () => {
  assert.match(
    source,
    /const pipelineMode = normalizeGpuOnlyPipelineMode\(wgslOffload\);[\s\S]*if \(pipelineMode === 'standard'\) \{[\s\S]*integrateSoftBodiesCpu\([\s\S]*lastMode = 'cpu-standard'[\s\S]*lastSourceRoute = 'cpu-standard-authoritative'/,
    'expected gpu-only integrate path to preserve explicit standard baseline-reference mode before validated/fast WGSL routes',
  );
});
