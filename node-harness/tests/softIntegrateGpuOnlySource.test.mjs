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
    /computeSoftIntegrateProposalSignature\([\s\S]*buildCpuReferenceIntegration\([\s\S]*computeSoftIntegrateParity\([\s\S]*state\.lastProposalSignature = proposalSignature >>> 0;[\s\S]*state\.lastSourceRoute = 'wgsl-integrate-proposal';[\s\S]*state\.lastParity = \{[\s\S]*source: 'wgsl-integrate-proposal-vs-cpu'/,
    'expected soft integrate WGSL branch to persist deterministic signature + parity metrics and proposal source-route ownership',
  );

  assert.match(
    source,
    /wgslOffload\.state\.lastMode = 'wgsl';[\s\S]*wgslOffload\.state\.lastSourceRoute = 'wgsl-integrate-authoritative';/,
    'expected authoritative WGSL success path to publish source-route ownership',
  );
});
