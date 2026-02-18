import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';
const gpuLabSource = readFileSync(resolve(ROOT, 'sim-server/public/gpu-lab.js'), 'utf8');
const areaSource = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftAreaXpbdGpuOnly.js'), 'utf8');

test('gpu-lab routes soft area XPBD gpu-only pass with WGSL offload state wiring', () => {
  assert.match(
    gpuLabSource,
    /applySoftAreaXPBDVelocityGpuOnly\(\{[\s\S]*wgslOffload:[\s\S]*enabled: true,[\s\S]*device: sim\?\.device,[\s\S]*state: \(sim\.softAreaXpbdWgslState \|\|= \{\}\),[\s\S]*\}\);/,
    'expected soft area XPBD gpu-only dispatch to publish optional WGSL offload context',
  );
});

test('soft area XPBD gpu-only module publishes deterministic WGSL-prepared layout ownership', () => {
  assert.match(
    areaSource,
    /function buildSoftAreaXpbdWgslPlan\(\{ sim, soft, loops \}\)/,
    'expected deterministic WGSL prep plan builder for area XPBD loops',
  );

  assert.match(
    areaSource,
    /clusterOffsets = new Uint32Array\(loopList\.length \+ 1\)/,
    'expected CSR-style cluster offsets for ragged loop ownership',
  );

  assert.match(
    areaSource,
    /wgslOffload\.state\.lastPreparedClusterCount = plan\.clusterCount;[\s\S]*wgslOffload\.state\.lastPreparedEndpointCount = plan\.endpointCount;[\s\S]*wgslOffload\.state\.lastPreparedLayoutBytes = layout\.byteLength;/,
    'expected prepared WGSL layout telemetry counters for next-stage offload bring-up',
  );

  assert.match(
    areaSource,
    /if \(canUseWgslOffload\(wgslOffload\)\) \{[\s\S]*Promise\.all\(\[[\s\S]*dispatchSoftAreaWgslProbe\(\{ sim, soft, offload: wgslOffload, plan, dtPos \}\)[\s\S]*dispatchSoftAreaWgslLambdaProposal\(\{ soft, offload: wgslOffload, plan, dtPos, alpha \}\)[\s\S]*lastMode = velocityProposalRan[\s\S]*'wgsl-velocity-proposal'[\s\S]*'wgsl-proposal'[\s\S]*'wgsl-probe'/,
    'expected gpu-only soft area pass to dispatch WGSL probe + lambda proposal stages while retaining CPU-authoritative fallback',
  );

  assert.match(
    areaSource,
    /copyBufferToBuffer\(state\.areaLambdaDeltaOut, 0, state\.areaLambdaDeltaReadback, 0, bytes\)[\s\S]*copyBufferToBuffer\(state\.areaLambdaNextOut, 0, state\.areaLambdaNextReadback, 0, bytes\)[\s\S]*lastAreaProposalDeltaLambdaByCluster[\s\S]*lastAreaProposalLambdaNextByCluster[\s\S]*lastAreaProposalAbsDeltaMean[\s\S]*lastAreaProposalAbsDeltaMax/,
    'expected WGSL area lambda proposal stage to read back deterministic per-cluster lambda telemetry for upcoming reduction offload',
  );

  assert.match(
    areaSource,
    /dispatchSoftAreaWgslVelocityDeltaProposal\([\s\S]*deltaByCluster: wgslOffload\.state\.lastAreaProposalDeltaLambdaByCluster[\s\S]*lastMode = velocityProposalRan[\s\S]*'wgsl-velocity-proposal'/,
    'expected gpu-only area pass to run WGSL endpoint velocity-delta proposal stage after lambda proposal for reduction-ready telemetry',
  );

  assert.match(
    areaSource,
    /endpointClusterIndex = new Uint32Array\(endpointCount\)[\s\S]*endpointClusterIndex\[write\] = li/,
    'expected WGSL area plan builder to materialize deterministic endpoint->cluster ownership lookup',
  );

  assert.match(
    areaSource,
    /state\.areaVelocityEndpointClusterIndex[\s\S]*device\.queue\.writeBuffer\(state\.areaVelocityEndpointClusterIndex, 0, plan\.endpointClusterIndex\)/,
    'expected WGSL area velocity proposal stage to upload deterministic endpoint->cluster ownership lookup',
  );

  assert.match(
    areaSource,
    /let ci = endpointClusterIndex\[ei\];/,
    'expected WGSL velocity proposal shader to use direct endpoint->cluster lookup instead of scanning offsets per endpoint',
  );

  assert.match(
    areaSource,
    /copyBufferToBuffer\(state\.areaVelocityDeltaVXOut, 0, state\.areaVelocityDeltaVXReadback, 0, bytes\)[\s\S]*copyBufferToBuffer\(state\.areaVelocityDeltaVYOut, 0, state\.areaVelocityDeltaVYReadback, 0, bytes\)[\s\S]*lastAreaVelocityProposalDeltaVxByEndpoint[\s\S]*lastAreaVelocityProposalDeltaVyByEndpoint[\s\S]*lastAreaVelocityProposalAbsDeltaMean[\s\S]*lastAreaVelocityProposalAbsDeltaMax/,
    'expected WGSL area velocity proposal stage to read back deterministic per-endpoint node delta telemetry for upcoming reduction offload',
  );

  assert.match(
    areaSource,
    /const softAreaVelocityNodeReductionWgsl = \/\* wgsl \*\/[\s\S]*nodeContributionCountOut: array<u32>[\s\S]*for \(var ei = 0u; ei < params\.endpointCount; ei = ei \+ 1u\)[\s\S]*nodeDeltaVXOut\[ni\] = sumX;[\s\S]*nodeContributionCountOut\[ni\] = count;/,
    'expected gpu-only area module to define deterministic WGSL node-reduction shader for endpoint->node aggregation ownership',
  );

  assert.match(
    areaSource,
    /dispatchSoftAreaWgslVelocityNodeReduction\([\s\S]*state\.lastAreaVelocityProposalNodeDeltaVx[\s\S]*state\.lastAreaVelocityProposalNodeDeltaVy[\s\S]*state\.lastAreaVelocityProposalNodeContributionCount/,
    'expected WGSL node-reduction stage to publish deterministic per-node deltas and contribution counts',
  );

  assert.match(
    areaSource,
    /velocityProposalRan = await dispatchSoftAreaWgslVelocityDeltaProposal\([\s\S]*const nodeReductionRan = await dispatchSoftAreaWgslVelocityNodeReduction\([\s\S]*if \(!nodeReductionRan\) \{[\s\S]*reduceSoftAreaVelocityProposalToNodeDeltas\([\s\S]*buildSoftAreaVelocityNodeReference\([\s\S]*publishSoftAreaVelocityNodeParity\(/,
    'expected gpu-only area branch to run WGSL node-reduction after endpoint proposal, preserve CPU reduction fallback, and publish deterministic node-parity telemetry for authoritative offload bring-up',
  );

  assert.match(
    areaSource,
    /state\.lastAreaVelocityProposalNodeParityMaxError = maxNodeDeltaError;[\s\S]*state\.lastAreaVelocityProposalNodeParityMeanError = actualVx\.length > 0 \? \(sumNodeDeltaError \/ actualVx\.length\) : 0;[\s\S]*state\.lastAreaVelocityProposalNodeContributionParityMaxError = maxContributionError;/,
    'expected WGSL area node-reduction parity helper to publish deterministic per-node delta/count error telemetry against CPU reference',
  );

  assert.match(
    areaSource,
    /isGpuOnlyFastMode\(wgslOffload\)[\s\S]*dispatchSoftAreaWgslVelocityDeltaProposal\([\s\S]*includeEndpointTelemetry: !fastMode[\s\S]*lastAreaVelocityProposalSource = fastMode[\s\S]*'wgsl-node-reduction-fast'[\s\S]*'wgsl-node-reduction'[\s\S]*if \(fastMode\) \{[\s\S]*validation: 'skipped-cpu-parity'/,
    'expected gpu-only area fast mode branch to skip endpoint telemetry/parity while retaining source-route visibility',
  );

  assert.match(
    areaSource,
    /checkFiniteFloat32Array\(wgslOffload\.state\.lastAreaVelocityProposalNodeDeltaVx\)[\s\S]*lastAreaVelocityProposalFinite[\s\S]*allFinite !== true[\s\S]*lastMode = 'cpu-fallback'/,
    'expected gpu-only area fast path to keep non-finite guardrails and explicit fallback mode',
  );
});
