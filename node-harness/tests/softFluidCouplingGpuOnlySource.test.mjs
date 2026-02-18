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
    /lastPreparedProposalSignature = carryProposalSignature;[\s\S]*lastSourceRoute = 'cpu-sampled-layout\+cluster-ownership';[\s\S]*dispatchSoftFluidCarryProposalWgsl\([\s\S]*proposalSignature: carryProposalSignature[\s\S]*lastCarryProposalDispatched = wgslProposalDispatched;/,
    'expected gpu-only prep branch to dispatch a concrete WGSL carry proposal stage using deterministic layout/signature telemetry',
  );

  assert.match(
    source,
    /const SOFT_FLUID_CARRY_PROPOSAL_WGSL = \/\* wgsl \*\/[\s\S]*@compute @workgroup_size\(64\)[\s\S]*outCarryX\[i\] = fx \* invM;[\s\S]*outLocalCarryY\[i\] = \(sampleVy\[i\] - nodeVy\[i\]\) \* dragHoney \* invM \* params\.localFlowShare;/,
    'expected concrete WGSL carry proposal shader to compute force/carry/local-carry outputs from deterministic prepared buffers',
  );

  assert.match(
    source,
    /const SOFT_FLUID_CLUSTER_LOAD_REDUCTION_WGSL = \/\* wgsl \*\/[\s\S]*clusterNodeOffsets: array<u32>;[\s\S]*outClusterTorque\[ci\] = sumTorque;[\s\S]*outClusterCount\[ci\] = count;/,
    'expected gpu-only soft-fluid module to define a concrete WGSL cluster-load reduction stage over deterministic ownership offsets',
  );

  assert.match(
    source,
    /dispatchSoftFluidClusterLoadReductionWgsl\([\s\S]*proposalSignature: clusterLoadProposalSignature,[\s\S]*forceX: cpuProposalForceX,[\s\S]*forceY: cpuProposalForceY,[\s\S]*lastClusterLoadProposalDispatched = wgslClusterLoadDispatched;/,
    'expected gpu-only soft-fluid branch to dispatch concrete WGSL cluster-load reduction proposals using deterministic force arrays',
  );

  assert.match(
    source,
    /hasAuthoritativeWgslClusterLoadProposal\([\s\S]*authoritativeClusterLoad === true[\s\S]*clusterLoadSource = canUseAuthoritativeClusterLoad[\s\S]*'wgsl-cluster-load-authoritative'[\s\S]*'cpu-cluster-load-authoritative';/,
    'expected gpu-only soft-fluid branch to gate authoritative WGSL cluster-load acceleration replay on deterministic signature+parity checks',
  );

  assert.match(
    source,
    /lastClusterLoadParity = \{[\s\S]*source: 'wgsl-cluster-load-proposal-vs-cpu',[\s\S]*mismatchCount: clusterLoadMismatchCount,[\s\S]*signature: clusterLoadProposalSignature >>> 0,[\s\S]*\};/,
    'expected gpu-only soft-fluid branch to publish deterministic WGSL-vs-CPU cluster-load parity telemetry before authoritative cutover',
  );

  assert.match(
    source,
    /hasAuthoritativeWgslCarryProposal\([\s\S]*carrySource = authoritativeCarryProposal \? 'wgsl-carry-authoritative' : 'cpu-carry-authoritative';[\s\S]*lastCpuCarryProposalForceX = cpuProposalForceX;[\s\S]*lastAuthoritativeCarrySource = carrySource;[\s\S]*lastAuthoritativeClusterLoadSource = clusterLoadSource;[\s\S]*lastSourceRoute = `\$\{carrySource\}\+\$\{clusterLoadSource\}`;[\s\S]*lastMode = carrySource === 'wgsl-carry-authoritative' \|\| clusterLoadSource === 'wgsl-cluster-load-authoritative'/,
    'expected gpu-only soft-fluid branch to publish deterministic CPU carry proposal arrays and route authoritative carry+cluster ownership to matching WGSL proposals',
  );
});
