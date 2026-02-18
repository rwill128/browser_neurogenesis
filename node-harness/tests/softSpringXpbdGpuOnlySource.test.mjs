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

test('soft spring gpu-only path dispatches real WGSL probe + lambda proposal stages when offload device is available', () => {
  assert.match(
    source,
    /if \(canUseWgslOffload\(wgslOffload\)\) \{[\s\S]*Promise\.all\(\[[\s\S]*dispatchSoftSpringWgslProbe\(\{ soft, offload: wgslOffload, layout \}\)[\s\S]*dispatchSoftSpringWgslLambdaProposal\([\s\S]*lastMode = proposalRan \? 'wgsl-proposal' : 'wgsl-probe'/,
    'expected gpu-only soft spring branch to execute WGSL probe + lambda proposal dispatches (with cpu fallback semantics preserved)',
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
    /copyBufferToBuffer\(state\.deltaLambdaOut, 0, state\.deltaLambdaReadback, 0, bytes\)[\s\S]*copyBufferToBuffer\(state\.lambdaNextOut, 0, state\.lambdaNextReadback, 0, bytes\)[\s\S]*lastProposalAbsDeltaMean[\s\S]*lastProposalAbsDeltaMax[\s\S]*lastProposalDeltaLambdaByColor[\s\S]*lastProposalLambdaNextByColor/,
    'expected WGSL lambda proposal stage to read back deterministic per-spring lambda telemetry',
  );
});
