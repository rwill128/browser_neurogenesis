import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftFluidCouplingGpuOnly.js'), 'utf8');

test('soft fluid-coupling gpu-only publishes deterministic WGSL prep layout for next-stage offload', () => {
  assert.match(
    source,
    /function buildSoftFluidCouplingWgslLayout\(\{ nodes, softNodeMomentumScale, softMembraneClusterSet \}\) \{[\s\S]*clusterNodeOffsets = new Uint32Array\(clusterCount \+ 1\);[\s\S]*clusterNodeCount = new Uint32Array\(clusterCount\);[\s\S]*nodeClusterSlot = new Uint32Array\(nodeCount\);[\s\S]*clusterNodeIndices = new Uint32Array\(nodeCount\);/,
    'expected deterministic node/cluster packed layout builder for upcoming WGSL stage',
  );

  assert.match(
    source,
    /function buildSoftFluidCouplingCpuSampleLayout\(\{[\s\S]*fluidSampleVx = new Float32Array\(nodeCount\);[\s\S]*sampleDeltaVy = new Float32Array\(nodeCount\);/,
    'expected deterministic sampled-fluid layout builder that removes callback ownership from the next WGSL stage',
  );

  assert.match(
    source,
    /if \(wgslOffload\?\.enabled === true && wgslOffload\?\.state\) \{[\s\S]*preparedLayout = prep\.layout;[\s\S]*preparedSampleLayout = samplePrep\.layout;[\s\S]*lastPreparedLayoutBytes = prep\.byteLength;[\s\S]*lastPreparedOwnershipCount = prep\.layout\.clusterNodeIndices\.length;[\s\S]*lastPreparedSampleLayoutBytes = samplePrep\.byteLength;[\s\S]*lastPreparedSampleLayoutSignature = samplePrep\.signature;/,
    'expected gpu-only path to publish both structural and sampled-fluid WGSL prep telemetry into offload state',
  );

  assert.match(
    source,
    /lastPreparedProposalSignature = carryProposalSignature;[\s\S]*lastSourceRoute = 'cpu-sampled-layout\+cluster-ownership';[\s\S]*lastMode = 'cpu-prepared';[\s\S]*synchronous,[\s\S]*real WGSL readback[\s\S]*authoritative-routing[\s\S]*signature\/length checks pass/,
    'expected deterministic proposal-signature telemetry + explicit async WGSL blocker note for authoritative routing handoff',
  );

  assert.match(
    source,
    /hasAuthoritativeWgslCarryProposal\([\s\S]*carrySource = authoritativeCarryProposal \? 'wgsl-carry-authoritative' : 'cpu-carry-authoritative';[\s\S]*lastCpuCarryProposalForceX = cpuProposalForceX;[\s\S]*lastAuthoritativeCarrySource = carrySource;[\s\S]*lastMode = carrySource === 'wgsl-carry-authoritative'/,
    'expected gpu-only soft-fluid branch to publish deterministic CPU carry proposal arrays and route authoritative ownership to matching WGSL proposals',
  );
});
