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
    /const gatherSignature = validatedMode \? buildBodyFluidInjectionGatherSignature\(gatherLayout\) : 0;/,
    'expected validated mode to retain deterministic gather signatures while standard/fast modes skip signature overhead',
  );

  assert.match(
    source,
    /const hasMatchingWgslGather = fastMode[\s\S]*lastGatherProposalFrame[\s\S]*: validatedMode[\s\S]*lastGatherProposalSignature === gatherSignature/,
    'expected validated mode to gate authoritative WGSL gather replay on deterministic signatures while non-validated modes avoid signature gating',
  );
});

test('body-fluid injection gpu-only pipeline mode exposes explicit standard vs validated vs fast runtime behavior', () => {
  assert.match(
    source,
    /function getGpuOnlyPipelineModeProfile\(offload\) \{[\s\S]*modeProfile === 'gpu-only-fast'[\s\S]*modeProfile === 'gpu-only-validated'[\s\S]*return 'standard';[\s\S]*\}/,
    'expected body-fluid injection path to normalize explicit standard/validated/fast mode profiles',
  );

  assert.match(
    source,
    /const pipelineMode = getGpuOnlyPipelineModeProfile\(wgslOffload\);[\s\S]*const standardMode = pipelineMode === 'standard';[\s\S]*if \(!standardMode && canUseWgslOffload\(wgslOffload\)\) \{/,
    'expected standard mode to skip WGSL dispatch while validated/fast modes keep WGSL execution path',
  );

  assert.match(
    source,
    /let gatherSource = standardMode \? 'cpu-standard-authoritative' : 'cpu-gather-authoritative';/,
    'expected standard mode to publish explicit cpu-standard source route ownership',
  );
});

test('body-fluid injection gpu-only fast mode skips shadow cpu parity while preserving finite checks + fallback route visibility', () => {
  assert.match(
    source,
    /function isGpuOnlyFastMode\(offload\) \{[\s\S]*getGpuOnlyPipelineModeProfile\(offload\) === 'gpu-only-fast';[\s\S]*\}/,
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
    /if \(!gatherDeltaToApply\) \{[\s\S]*gatherSource = fastMode[\s\S]*'cpu-gather-fallback-fast'[\s\S]*'cpu-gather-authoritative'/,
    'expected fast mode to keep explicit CPU fallback source-route telemetry when WGSL gather is unavailable/non-finite',
  );
});
