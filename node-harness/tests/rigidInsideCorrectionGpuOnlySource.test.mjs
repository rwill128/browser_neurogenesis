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
    /const rigidInsideNodePolyCandidateWgsl = \/\* wgsl \*\/[\s\S]*node_candidate_count_out\[node_index\] = candidate_count;[\s\S]*dispatchRigidInsideNodeCandidateProposal\([\s\S]*copyBufferToBuffer\([\s\S]*insideNodeCandidateCountOut[\s\S]*insideNodeCandidateCountReadback[\s\S]*lastInsideNodeCandidateCountSource = 'wgsl-rigid-inside-node-candidate-proposal';[\s\S]*lastSourceRoute = 'wgsl-rigid-inside-node-candidate-proposal';/,
    'expected gpu-only inside-correction path to run a concrete WGSL node/poly candidate proposal stage with readback telemetry and source-route ownership',
  );

  assert.match(
    source,
    /wgslOffload\.state\.lastInsideCpuReference = \{[\s\S]*source: 'cpu-rigid-inside-authoritative-reference',[\s\S]*\};[\s\S]*wgslOffload\.state\.lastInsideParity = \{[\s\S]*source: 'cpu-rigid-inside-authoritative-reference',[\s\S]*proposalSignature: proposalSignature >>> 0,[\s\S]*preparedLayoutSignature: preparedLayoutSignature >>> 0/,
    'expected deterministic CPU reference/parity payload to be emitted for next WGSL inside-correction stage parity bring-up',
  );

  assert.match(
    source,
    /wgslOffload\.state\.lastInsideCorrectionCount = corrected;[\s\S]*lastAuthoritativeInsideSource = 'cpu-rigid-inside-authoritative';[\s\S]*lastSourceRoute = 'cpu-rigid-inside-authoritative';/,
    'expected explicit authoritative source route to remain CPU while WGSL prep ownership becomes available',
  );
});
