import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const source = readFileSync(resolve(ROOT, 'sim-server/public/runtime-solvers/stepSoftSpringsXpbdGpuOnly.js'), 'utf8');

test('soft spring gpu-only module wires WGSL prep layout alongside plan in offload branch', () => {
  assert.match(
    source,
    /if \(wgslOffload\?\.enabled === true && wgslOffload\?\.state\) \{[\s\S]*buildSoftSpringXpbdWgslPlan\([\s\S]*buildSoftSpringXpbdWgslLayout\([\s\S]*wgslOffload\.state\.preparedLayout = layout;[\s\S]*lastPreparedLayoutBytes/,
    'expected WGSL prep branch to build/store deterministic layout buffers for next XPBD offload stage',
  );
});


test('soft spring gpu-only WGSL prep branch stores deterministic spring color batches for conflict-free dispatch planning', () => {
  assert.match(
    source,
    /buildSoftSpringXpbdWgslPlan\([\s\S]*springColorOffsets[\s\S]*springColorOrderedIndices[\s\S]*wgslOffload\.state\.preparedPlan = plan/,
    'expected WGSL prep branch to retain spring color batches for no-atomic per-color dispatch staging',
  );
});


test('soft spring gpu-only WGSL layout publishes color-batched endpoint ownership buffers for upcoming node-delta reduction stage', () => {
  assert.match(
    source,
    /buildSoftSpringXpbdWgslLayout\([\s\S]*colorEndpointOffsets[\s\S]*endpointNodeIndicesByColor[\s\S]*endpointSpringIndicesByColor[\s\S]*endpointSignsI32ByColor/,
    'expected WGSL layout builder to publish deterministic per-color endpoint ownership buffers for the next reduction offload stage',
  );
});

test('soft spring gpu-only path dispatches real WGSL probe + lambda proposal stages when offload device is available', () => {
  assert.match(
    source,
    /canUseWgslOffload\(wgslOffload\)[\s\S]*Promise\.all\(\[[\s\S]*dispatchSoftSpringWgslProbe\(\{ soft, offload: wgslOffload, layout \}\)[\s\S]*dispatchSoftSpringWgslLambdaProposal\([\s\S]*lastMode = proposalRan \? 'wgsl-velocity-proposal' : 'wgsl-probe'/,
    'expected gpu-only soft spring branch to execute WGSL probe + lambda proposal dispatches (with cpu fallback semantics preserved)',
  );
});

test('soft spring gpu-only WGSL dispatch path guards against overlapping map/readback passes', () => {
  assert.match(
    source,
    /canUseWgslOffload\(wgslOffload\)[\s\S]*if \(wgslOffload\.state\.wgslInFlight\) \{[\s\S]*wgslSkippedWhileBusy[\s\S]*\} else \{[\s\S]*wgslInFlight = true;[\s\S]*Promise\.all\([\s\S]*\.finally\(\(\) => \{[\s\S]*wgslInFlight = false;[\s\S]*lastCompletedWgslRunId/,
    'expected gpu-only soft spring branch to serialize WGSL readback passes so overlapping mapAsync work does not race the next frame',
  );
});

test('soft spring gpu-only WGSL probe copies stretch output into readback buffer for deterministic parity telemetry', () => {
  assert.match(
    source,
    /copyBufferToBuffer\(state\.probeStretchOut, 0, state\.probeStretchReadback, 0, bytes\)[\s\S]*mapAsync\(globalThis\.GPUMapMode\.READ, 0, bytes\)[\s\S]*lastProbeAbsMean[\s\S]*lastProbeAbsMax[\s\S]*lastProbeStretchByColor/,
    'expected WGSL probe stage to read back stretch metrics for deterministic CPU-vs-WGSL parity harness checks',
  );
});

test('soft spring gpu-only WGSL lambda proposal stage reads back per-spring delta telemetry for next reduction offload', () => {
  assert.match(
    source,
    /copyBufferToBuffer\(state\.deltaLambdaOut, 0, state\.deltaLambdaReadback, 0, bytes\)[\s\S]*copyBufferToBuffer\(state\.lambdaNextOut, 0, state\.lambdaNextReadback, 0, bytes\)[\s\S]*lastProposalAbsDeltaMean[\s\S]*lastProposalAbsDeltaMax[\s\S]*lastProposalDeltaLambdaByColor[\s\S]*lastProposalLambdaNextByColor[\s\S]*lastProposalDeltaLambdaBySpring[\s\S]*lastProposalLambdaNextBySpring/,
    'expected WGSL lambda proposal stage to read back deterministic per-spring lambda telemetry in both color and original spring order',
  );
});


test('soft spring gpu-only WGSL proposal branch dispatches velocity-delta proposal stage and publishes endpoint+node telemetry', () => {
  assert.match(
    source,
    /proposalRan\) \{[\s\S]*dispatchSoftSpringWgslVelocityDeltaProposal\([\s\S]*lastVelocityDeltaProposalEndpointVxByColor[\s\S]*lastVelocityDeltaProposalEndpointVyByColor[\s\S]*lastVelocityDeltaProposalNodeVxByColor[\s\S]*lastVelocityDeltaProposalNodeVyByColor/,
    'expected WGSL proposal branch to dispatch deterministic velocity-delta WGSL stage and publish endpoint/node buffers',
  );
});

test('soft spring gpu-only path can promote cached WGSL velocity proposal to authoritative node/lambda apply when signature matches', () => {
  assert.match(
    source,
    /computeSoftSpringVelocityProposalSignature\([\s\S]*lastPreparedProposalSignature = proposalSignature[\s\S]*enableAuthoritativeVelocityDelta === true[\s\S]*lastVelocityDeltaProposalSignature === proposalSignature[\s\S]*applySoftSpringWgslAuthoritativeProposal\([\s\S]*lastMode = 'wgsl-velocity-authoritative'/,
    'expected soft spring gpu-only branch to gate authoritative WGSL node/lambda replay on deterministic input signature parity',
  );
});


test('soft spring gpu-only WGSL velocity proposal stage runs WGSL node-reduction dispatch before readback', () => {
  assert.match(
    source,
    /const softSpringVelocityNodeReductionWgsl = \/\* wgsl \*\/[\s\S]*velocityReductionPipeline[\s\S]*nodeReductionPass\.setPipeline\(state\.velocityReductionPipeline\)[\s\S]*nodeReductionPass\.dispatchWorkgroups\([\s\S]*copyBufferToBuffer\(state\.velocityNodeDeltaVXOut, 0, state\.velocityNodeDeltaVXReadback, 0, nodeBytes\)[\s\S]*lastVelocityDeltaReductionDispatch/,
    'expected velocity proposal path to include a concrete WGSL node-reduction stage (not CPU-only reduction loop)',
  );
});


test('soft spring gpu-only WGSL velocity proposal branch records source route + deterministic parity metrics against CPU ownership reduction', () => {
  assert.match(
    source,
    /reduceSoftSpringVelocityDeltasDeterministic\([\s\S]*lastVelocityDeltaExpectedNodeVxByColor[\s\S]*lastVelocityDeltaExpectedNodeVyByColor[\s\S]*lastVelocityDeltaProposalSource = 'wgsl-node-reduction'[\s\S]*lastVelocityDeltaProposalSource = 'cpu-deterministic-reduction'[\s\S]*computeVelocityDeltaParityStats[\s\S]*lastVelocityDeltaParity = \{[\s\S]*source: wgslOffload\.state\.lastVelocityDeltaProposalSource/,
    'expected WGSL velocity proposal branch to persist source-route ownership and CPU parity metrics required before switching to authoritative WGSL node deltas',
  );
});
