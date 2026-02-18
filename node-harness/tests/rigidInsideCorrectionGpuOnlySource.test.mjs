import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepRigidInsideCorrectionGpuOnly.js'), 'utf8');

test('rigid-inside gpu-only path publishes deterministic WGSL prep layout + source-route telemetry while remaining CPU-authoritative', () => {
  assert.match(
    source,
    /function buildRigidInsideCorrectionWgslLayout\(\{[\s\S]*rigidPolyOffsets = new Uint32Array\(rigidCount \+ 1\);[\s\S]*eligibleNodeOffsets = new Uint32Array\(rigidCount \+ 1\);[\s\S]*polyPointOffsets = \[0\];/,
    'expected deterministic rigid/poly/node ownership layout builder for upcoming WGSL inside-correction dispatch',
  );

  assert.match(
    source,
    /computeRigidInsideCorrectionProposalSignature\([\s\S]*lastPreparedInsideLayoutSignature = preparedLayoutSignature;[\s\S]*lastPreparedInsideProposalSignature = proposalSignature;[\s\S]*lastSourceRoute = 'cpu-rigid-inside-prepared-layout';/,
    'expected gpu-only inside-correction path to publish deterministic proposal+layout signatures and source-route telemetry for WGSL staging',
  );

  assert.match(
    source,
    /const rigidInsidePolyBoundsProbeWgsl = \/\* wgsl \*\/[\s\S]*@compute @workgroup_size\(\$\{WGSL_WORKGROUP_SIZE\}\)[\s\S]*dispatchRigidInsidePolyBoundsProbe\([\s\S]*lastInsidePolyBoundsProbeSource = 'wgsl-rigid-inside-poly-bounds-probe';[\s\S]*lastSourceRoute = 'wgsl-rigid-inside-poly-bounds-probe';/,
    'expected gpu-only inside-correction path to run a concrete WGSL polygon-bounds probe stage and publish probe source-route telemetry for upcoming point-in-polygon offload',
  );

  assert.match(
    source,
    /const rigidInsideNodePolyCandidateWgsl = \/\* wgsl \*\/[\s\S]*node_candidate_count_out\[node_index\] = candidate_count;[\s\S]*dispatchRigidInsideNodeCandidateProposal\([\s\S]*if \(includeReadbackTelemetry\) \{[\s\S]*copyBufferToBuffer\([\s\S]*insideNodeCandidateCountOut[\s\S]*insideNodeCandidateCountReadback[\s\S]*lastInsideNodeCandidateCountSource = 'wgsl-rigid-inside-node-candidate-proposal';[\s\S]*lastInsideNodeCandidateCountSource = 'wgsl-rigid-inside-node-candidate-proposal-fast';/,
    'expected gpu-only inside-correction path to run a concrete WGSL node/poly candidate proposal stage with validated readback telemetry and explicit fast-mode no-readback route',
  );

  assert.match(
    source,
    /const rigidInsideCorrectionProposalWgsl = \/\* wgsl \*\/[\s\S]*out_corr_x\[node_index\] = best_dx;[\s\S]*out_corr_y\[node_index\] = best_dy;[\s\S]*dispatchRigidInsideCorrectionProposal\([\s\S]*lastInsideCorrectionProposalSource = 'wgsl-rigid-inside-correction-proposal';[\s\S]*lastInsideCorrectionProposalSource = 'wgsl-rigid-inside-correction-proposal-fast';[\s\S]*lastSourceRoute = fastMode[\s\S]*'wgsl-rigid-inside-correction-proposal-fast'[\s\S]*'wgsl-rigid-inside-correction-proposal'/,
    'expected gpu-only inside-correction path to dispatch concrete WGSL correction vector proposal math with validated readback telemetry and explicit fast-mode route visibility',
  );

  assert.match(
    source,
    /wgslOffload\.state\.lastInsideCpuReference = \{[\s\S]*source: 'cpu-rigid-inside-authoritative-reference',[\s\S]*\};[\s\S]*wgslOffload\.state\.lastInsideParity = \{[\s\S]*source: 'cpu-rigid-inside-authoritative-reference',[\s\S]*proposalSignature: proposalSignature >>> 0,[\s\S]*preparedLayoutSignature: preparedLayoutSignature >>> 0/,
    'expected deterministic CPU reference/parity payload to be emitted for next WGSL inside-correction stage parity bring-up',
  );

  assert.match(
    source,
    /function canApplyAuthoritativeRigidInsideProposal\([\s\S]*enableAuthoritativeInsideCorrection !== true[\s\S]*lastInsideCorrectionProposalSource !== expectedSource[\s\S]*if \(!Number\.isFinite\(cx\) \|\| !Number\.isFinite\(cy\) \|\| !Number\.isFinite\(rbi\)\) return false;/,
    'expected rigid-inside stage to gate WGSL authoritative apply behind explicit opt-in, signature/source-route match, and hard finite validation fallback',
  );

  assert.match(
    source,
    /applyAuthoritativeRigidInsideProposal\([\s\S]*usedAuthoritativeWgslProposal = true;[\s\S]*lastAuthoritativeInsideSource = usedAuthoritativeWgslProposal[\s\S]*'wgsl-rigid-inside-authoritative'[\s\S]*'cpu-rigid-inside-authoritative'/,
    'expected rigid-inside stage to support WGSL authoritative apply path with explicit authoritative source-route visibility and CPU fallback route retention',
  );
});

test('rigid-inside gpu-only pipeline mode exposes explicit standard vs validated vs fast runtime behavior', () => {
  assert.match(
    source,
    /function getGpuOnlyPipelineModeProfile\(offload\) \{[\s\S]*modeProfile === 'gpu-only-fast'[\s\S]*modeProfile === 'gpu-only-validated'[\s\S]*return 'standard';[\s\S]*\}/,
    'expected rigid-inside path to normalize explicit standard/validated/fast mode profiles',
  );

  assert.match(
    source,
    /const pipelineMode = getGpuOnlyPipelineModeProfile\(wgslOffload\);[\s\S]*const fastMode = isGpuOnlyFastMode\(wgslOffload\);[\s\S]*const validatedMode = isGpuOnlyValidatedMode\(wgslOffload\);[\s\S]*const standardMode = pipelineMode === 'standard';[\s\S]*if \(!standardMode\) \{[\s\S]*dispatchRigidInsidePolyBoundsProbe\(/,
    'expected standard mode to retain CPU-authoritative baseline reference behavior while validated/fast modes keep WGSL staging dispatches enabled',
  );
});

test('rigid-inside gpu-only fast mode skips node-candidate readback and cpu parity snapshots while preserving source-route visibility', () => {
  assert.match(
    source,
    /dispatchRigidInsideNodeCandidateProposal\(\{ offload, prep, proposalSignature, includeReadbackTelemetry = true \}\)[\s\S]*if \(includeReadbackTelemetry\) \{[\s\S]*copyBufferToBuffer\([\s\S]*\}[\s\S]*else \{[\s\S]*lastInsideNodeCandidateCountSource = 'wgsl-rigid-inside-node-candidate-proposal-fast';[\s\S]*lastInsideNodeCandidateValidation = 'skipped-readback-telemetry';/,
    'expected fast mode candidate branch to skip readback copy/map overhead while keeping explicit fast source-route telemetry',
  );

  assert.match(
    source,
    /if \(!fastMode\) \{[\s\S]*lastInsideCpuReference = \{[\s\S]*source: 'cpu-rigid-inside-authoritative-reference'[\s\S]*\}[\s\S]*\} else \{[\s\S]*lastInsideCpuReference = null;[\s\S]*validation: 'skipped-cpu-reference'[\s\S]*lastInsideValidation = 'skipped-cpu-reference';/,
    'expected fast mode to bypass CPU reference/parity snapshots while preserving explicit validation + fallback source ownership',
  );
});
