import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepBodyFluidInjectionGpuOnly.js'), 'utf8');

test('body-fluid injection gpu-only keeps deterministic WGSL gather layout + validated signature routing', () => {
  assert.match(
    source,
    /function buildBodyFluidInjectionGatherLayout\(\{[\s\S]*cellOffsets = new Uint32Array\(cellCount \+ 1\);[\s\S]*contribPointIndex = new Uint32Array\(totalContrib\);[\s\S]*contribWeight = new Float32Array\(totalContrib\);/,
    'expected gather layout helper with deterministic CSR-style cell offsets and contribution streams',
  );

  assert.match(
    source,
    /const gatherSignature = fastMode \? 0 : buildBodyFluidInjectionGatherSignature\(gatherLayout\);/,
    'expected validated mode to retain deterministic gather signatures while fast mode skips signature/parity overhead',
  );

  assert.match(
    source,
    /const hasMatchingWgslGather = fastMode[\s\S]*lastGatherProposalFrame[\s\S]*: wgslOffload\.state\.lastGatherProposalSignature === gatherSignature/,
    'expected validated mode to gate authoritative WGSL gather replay on deterministic signatures',
  );
});

test('body-fluid injection gpu-only fast mode skips shadow cpu parity while preserving finite checks + fallback route visibility', () => {
  assert.match(
    source,
    /function isGpuOnlyFastMode\(offload\) \{[\s\S]*modeProfile[\s\S]*gpu-only-fast[\s\S]*\}/,
    'expected body-fluid injection path to detect explicit gpu-only-fast mode profile',
  );

  assert.match(
    source,
    /let cpuGatherDelta = null;[\s\S]*if \(!fastMode\) \{[\s\S]*computeBodyFluidInjectionCellDeltasFromGatherLayout\(/,
    'expected fast mode to skip default shadow CPU gather/parity work before WGSL proposal dispatch',
  );

  assert.match(
    source,
    /checkFiniteFloat32Array\(wgslDeltaVx\)[\s\S]*checkFiniteFloat32Array\(wgslDeltaVy\)[\s\S]*gatherSource = fastMode \? 'wgsl-gather-authoritative-fast' : 'wgsl-gather-authoritative';/,
    'expected fast mode WGSL gather route to retain hard non-finite safety rails before authoritative apply',
  );

  assert.match(
    source,
    /if \(!gatherDeltaToApply\) \{[\s\S]*gatherSource = fastMode \? 'cpu-gather-fallback-fast' : 'cpu-gather-authoritative';/,
    'expected fast mode to keep explicit CPU fallback source-route telemetry when WGSL gather is unavailable/non-finite',
  );
});
