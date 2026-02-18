import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const gpuLabSource = readFileSync(resolve(ROOT, 'sim-server/public/gpu-lab.js'), 'utf8');
const moduleSource = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftMembraneConstraintsGpuOnly.js'), 'utf8');

test('gpu-lab wires membrane boundary xpbd gpu-only pass with WGSL offload state bag', () => {
  assert.match(
    gpuLabSource,
    /applySoftMembraneBoundaryXPBDVelocityGpuOnly\(\{[\s\S]*wgslOffload:[\s\S]*enabled: true,[\s\S]*device: sim\?\.device,[\s\S]*modeProfile: normalizeRuntimePipelineMode\(sim\?\.controls\?\.runtimePipelineMode, sim\?\.controls\?\.runtimeSolverPath\),[\s\S]*state: \(\(\) => \{[\s\S]*sim\.softMembraneBoundaryWgslState \|\|= \{\}[\s\S]*enableAuthoritativeMembraneBoundaryEdge !== false[\s\S]*enableAuthoritativeMembraneBoundaryEdge = true;[\s\S]*return st;[\s\S]*\}\)\(\),[\s\S]*\}\)/,
    'expected gpu-lab membrane boundary gpu-only branch to wire isolated WGSL offload state bag with authoritative WGSL replay enabled by default (unless explicitly disabled)',
  );
});

test('gpu-lab wires membrane shape-memory gpu-only pass with WGSL offload state bag', () => {
  assert.match(
    gpuLabSource,
    /applySoftMembraneShapeMemoryVelocityGpuOnly\(\{[\s\S]*wgslOffload:[\s\S]*enabled: true,[\s\S]*device: sim\?\.device,[\s\S]*state: \(\(\) => \{[\s\S]*sim\.softMembraneShapeMemoryWgslState \|\|= \{\}[\s\S]*enableAuthoritativeShapeMemory !== false[\s\S]*enableAuthoritativeShapeMemory = true;[\s\S]*return st;[\s\S]*\}\)\(\),[\s\S]*\}\)/,
    'expected gpu-lab membrane shape-memory gpu-only branch to wire isolated WGSL offload context with authoritative WGSL replay enabled by default (unless explicitly disabled)',
  );
});

test('membrane constraints gpu-only module defines concrete WGSL membrane-boundary edge proposal kernel with source-route telemetry', () => {
  assert.match(
    moduleSource,
    /const softMembraneBoundaryEdgeProposalWgsl = \/\* wgsl \*\/[\s\S]*let C = clamp\(d - rest, -cLimit, cLimit\);[\s\S]*deltaVxAOut\[i\] = \(-wA \* dl \* nx\) \* invDt;[\s\S]*lambdaNextOut\[i\] = lambdaNext;/,
    'expected concrete WGSL kernel math for membrane boundary edge XPBD velocity/lambda proposal deltas',
  );

  assert.match(
    moduleSource,
    /getGpuOnlyPipelineModeProfile\(offload\)[\s\S]*modeProfile === 'gpu-only-fast'[\s\S]*modeProfile === 'gpu-only-validated'[\s\S]*return 'standard';/,
    'expected membrane boundary gpu-only path to normalize explicit standard/validated/fast pipeline mode routing',
  );

  assert.match(
    moduleSource,
    /dispatchSoftMembraneBoundaryEdgeProposal\([\s\S]*lastMembraneBoundaryEdgeProposalSignature[\s\S]*lastMembraneBoundaryEdgeProposalSource = finite\.allFinite\s*\?[\s\S]*'wgsl-membrane-boundary-edge-proposal'[\s\S]*'cpu-membrane-boundary-edge-authoritative-nonfinite'/,
    'expected membrane boundary proposal dispatch to preserve explicit source-route and hard non-finite fallback telemetry',
  );

  assert.match(
    moduleSource,
    /pendingWgslMembraneBoundaryEdgeProposalPromise[\s\S]*dispatchSoftMembraneBoundaryEdgeProposal\([\s\S]*lastMembraneBoundaryEdgeProposalSource = 'cpu-membrane-boundary-edge-authoritative'[\s\S]*lastMode = 'cpu-membrane-boundary-edge-authoritative'/,
    'expected serialized membrane boundary WGSL proposal dispatch with hard CPU fallback source-route when dispatch fails',
  );

  assert.match(
    moduleSource,
    /canApplyAuthoritativeMembraneBoundaryEdgeProposal\([\s\S]*enableAuthoritativeMembraneBoundaryEdge !== true[\s\S]*applyMembraneBoundaryVelocityDeltasAuthoritative\([\s\S]*lastMembraneBoundaryEdgeAuthoritativeSource = authoritativeBoundaryFromWgsl\s*\?[\s\S]*'wgsl-membrane-boundary-edge-authoritative'\s*:[\s\S]*'cpu-membrane-boundary-edge-authoritative'/,
    'expected membrane boundary path to support signature-gated authoritative WGSL edge replay with explicit source-route ownership telemetry',
  );
});

test('membrane constraints gpu-only module defines concrete WGSL shape-memory proposal kernel with source-route telemetry', () => {
  assert.match(
    moduleSource,
    /const softMembraneShapeMemoryProposalWgsl = \/\* wgsl \*\/[\s\S]*deltaVxOut\[i\] = ex \* corrPos \* invDt;[\s\S]*deltaVyOut\[i\] = ey \* corrPos \* invDt;/,
    'expected concrete WGSL kernel math for membrane shape-memory velocity proposal deltas',
  );

  assert.match(
    moduleSource,
    /dispatchSoftMembraneShapeMemoryProposal\([\s\S]*lastShapeMemoryProposalFinite[\s\S]*lastShapeMemoryProposalSignature[\s\S]*lastShapeMemoryProposalSource = finite\.allFinite \? 'wgsl-shape-memory-proposal' : 'cpu-shape-memory-authoritative-nonfinite'/,
    'expected proposal dispatch to preserve explicit source-route and finite fallback telemetry',
  );

  assert.match(
    moduleSource,
    /canApplyAuthoritativeShapeMemoryProposal\([\s\S]*enableAuthoritativeShapeMemory !== true[\s\S]*lastShapeMemoryProposalSource !== 'wgsl-shape-memory-proposal'/,
    'expected authoritative WGSL shape-memory gating with explicit source-route + hard opt-in flag',
  );

  assert.match(
    moduleSource,
    /lastShapeMemoryAuthoritativeSource = authoritativeFromWgsl\s*\? 'wgsl-shape-memory-authoritative'\s*:\s*'cpu-shape-memory-authoritative'/,
    'expected explicit authoritative source telemetry for WGSL vs CPU membrane shape-memory apply path',
  );

  assert.match(
    moduleSource,
    /pendingWgslShapeMemoryProposalPromise[\s\S]*dispatchSoftMembraneShapeMemoryProposal\([\s\S]*lastShapeMemoryProposalSource = 'cpu-shape-memory-authoritative'[\s\S]*lastMode = 'cpu-shape-memory-authoritative'/,
    'expected serialized WGSL proposal dispatch with hard CPU fallback source-route when dispatch fails',
  );
});
