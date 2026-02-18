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
    /if \(canUseWgslOffload\(wgslOffload\)\) \{[\s\S]*Promise\.all\(\[[\s\S]*dispatchSoftAreaWgslProbe\(\{ sim, soft, offload: wgslOffload, plan, dtPos \}\)[\s\S]*dispatchSoftAreaWgslLambdaProposal\(\{ soft, offload: wgslOffload, plan, dtPos, alpha \}\)[\s\S]*lastMode = proposalRan \? 'wgsl-proposal' : 'wgsl-probe'/,
    'expected gpu-only soft area pass to dispatch WGSL probe + lambda proposal stages while retaining CPU-authoritative fallback',
  );

  assert.match(
    areaSource,
    /copyBufferToBuffer\(state\.areaLambdaDeltaOut, 0, state\.areaLambdaDeltaReadback, 0, bytes\)[\s\S]*copyBufferToBuffer\(state\.areaLambdaNextOut, 0, state\.areaLambdaNextReadback, 0, bytes\)[\s\S]*lastAreaProposalDeltaLambdaByCluster[\s\S]*lastAreaProposalLambdaNextByCluster[\s\S]*lastAreaProposalAbsDeltaMean[\s\S]*lastAreaProposalAbsDeltaMax/,
    'expected WGSL area lambda proposal stage to read back deterministic per-cluster lambda telemetry for upcoming reduction offload',
  );
});
