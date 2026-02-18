import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const sourcePath = resolve(process.cwd(), 'sim-server/public/runtime-solvers/stepSoftMembranePressureGpuOnly.js');
const source = readFileSync(sourcePath, 'utf8');

const gpuLabPath = resolve(process.cwd(), 'sim-server/public/gpu-lab.js');
const gpuLabSource = readFileSync(gpuLabPath, 'utf8');

test('soft membrane pressure gpu-only module publishes deterministic WGSL prep layout ownership', () => {
  assert.match(
    source,
    /function buildSoftMembranePressureWgslPrep\([\s\S]*const membraneOffsets = new Uint32Array\(membraneCount \+ 1\);[\s\S]*const loopIndices = new Uint32Array\(indexCount\);[\s\S]*const loopMembraneIndex = new Uint32Array\(indexCount\);[\s\S]*const areaBase = new Float32Array\(membraneCount\);[\s\S]*const pressureGain = new Float32Array\(membraneCount\);[\s\S]*const nodeX = new Float32Array\(nodeCount\);[\s\S]*const nodeMass = new Float32Array\(nodeCount\);/,
    'expected gpu-only membrane pressure pass to build deterministic typed-array layout for membrane loops and node ownership/mass proposal buffers',
  );

  assert.match(
    source,
    /wgslOffload\.state\.preparedPlan = prep\.plan;[\s\S]*wgslOffload\.state\.preparedLayout = prep\.layout;[\s\S]*lastPreparedMembraneCount[\s\S]*lastPreparedLayoutBytes[\s\S]*lastMode = 'cpu-prepared';/,
    'expected membrane pressure pass to persist WGSL prep ownership while keeping CPU-authoritative stepping',
  );
});

test('soft membrane pressure gpu-only path dispatches WGSL area-probe + velocity-proposal stages with serialized readback', () => {
  assert.match(
    source,
    /const softMembranePressureAreaProbeWgsl = \/\* wgsl \*\/[\s\S]*@compute @workgroup_size\([\s\S]*area_now_out\[mi\] = now;[\s\S]*function dispatchSoftMembranePressureAreaProbe\([\s\S]*pass\.dispatchWorkgroups\([\s\S]*copyBufferToBuffer\([\s\S]*mapAsync\(globalThis\.GPUMapMode\.READ/,
    'expected membrane pressure module to run a concrete WGSL area-probe dispatch with readback telemetry',
  );

  assert.match(
    source,
    /const softMembranePressureVelocityProposalWgsl = \/\* wgsl \*\/[\s\S]*delta_vx_out\[li\] = nx \* impulse;[\s\S]*delta_vy_out\[li\] = ny \* impulse;[\s\S]*function dispatchSoftMembranePressureVelocityDeltaProposal\([\s\S]*pass\.dispatchWorkgroups\([\s\S]*copyBufferToBuffer\(state\.velocityProposalDeltaVx,[\s\S]*copyBufferToBuffer\(state\.velocityProposalDeltaVy,[\s\S]*mapAsync\(globalThis\.GPUMapMode\.READ/,
    'expected membrane pressure module to run a concrete WGSL velocity-delta proposal dispatch with readback telemetry',
  );

  assert.match(
    source,
    /const serializedDispatch = \(wgslOffload\.state\.pendingWgslAreaProbePromise \|\| Promise\.resolve\(\)\)[\s\S]*dispatchSoftMembranePressureAreaProbe\(wgslOffload, prep\)[\s\S]*dispatchSoftMembranePressureVelocityDeltaProposal\(wgslOffload, prep, dtPos, proposalSignature, \{ fastMode \}\)[\s\S]*lastMode = proposalRan[\s\S]*'wgsl-velocity-proposal'[\s\S]*'wgsl-area-probe'[\s\S]*pendingWgslAreaProbePromise = serializedDispatch;/,
    'expected membrane pressure WGSL dispatch chain to serialize probe+proposal readbacks and publish staged mode ownership',
  );
});

test('soft membrane pressure gpu-only module can route deterministic signature-matched WGSL proposal readback as authoritative deltas', () => {
  assert.match(
    source,
    /const proposalSignature = computeMembraneProposalSignature\(prep, dtPos\);[\s\S]*const canUseAuthoritativeWgsl = prep\.plan\.indexCount > 0[\s\S]*lastVelocityProposalSignature === proposalSignature[\s\S]*lastVelocityProposalFinite\?\.allFinite === true[\s\S]*lastVelocityProposalDeltaVx instanceof Float32Array[\s\S]*lastVelocityProposalDeltaVy instanceof Float32Array[\s\S]*lastVelocityProposalSource = fastMode \? 'wgsl-pressure-authoritative-fast' : 'wgsl-pressure-authoritative';/,
    'expected membrane pressure gpu-only path to guard authoritative WGSL proposal replay behind deterministic signature + finite safety checks + typed-array ownership checks',
  );

  assert.match(
    source,
    /dispatchSoftMembranePressureVelocityDeltaProposal\(wgslOffload, prep, dtPos, proposalSignature, \{ fastMode \}\)/,
    'expected membrane pressure path to thread deterministic signature and fast-mode routing into WGSL proposal dispatch',
  );

  assert.match(
    source,
    /lastVelocityProposalSignature = String\(proposalSignature \|\| ''\);/,
    'expected membrane pressure WGSL proposal dispatch to persist deterministic signature metadata for next-frame authoritative source routing',
  );
});

test('soft membrane pressure gpu-only fast mode skips CPU parity shadow while preserving finite checks + fallback source routing', () => {
  assert.match(
    source,
    /function getGpuOnlyPipelineModeProfile\(offload\) \{[\s\S]*modeProfile === 'gpu-only-fast'[\s\S]*modeProfile === 'gpu-only-validated'[\s\S]*return 'standard';[\s\S]*\}/,
    'expected membrane pressure gpu-only path to normalize standard vs validated vs fast pipeline mode profiles',
  );

  assert.match(
    source,
    /const fastMode = options\.fastMode === true;[\s\S]*if \(!finite\.allFinite\) \{[\s\S]*lastVelocityProposalSource = fastMode \? 'cpu-membrane-fallback-fast-nonfinite' : 'cpu-membrane-fallback-nonfinite';[\s\S]*return false;[\s\S]*\}/,
    'expected membrane pressure velocity proposal readback to keep hard non-finite guardrails with explicit fast-mode fallback source telemetry',
  );

  assert.match(
    source,
    /if \(fastMode\) \{[\s\S]*lastVelocityProposalExpectedDeltaVx = null;[\s\S]*lastVelocityProposalExpectedDeltaVy = null;[\s\S]*validation: 'skipped-cpu-parity'[\s\S]*lastVelocityProposalSource = 'wgsl-pressure-velocity-proposal-fast';[\s\S]*\} else \{[\s\S]*buildCpuMembranePressureVelocityDeltaProposal\(/,
    'expected fast mode to remove CPU shadow/parity recompute while standard/validated modes keep parity reference coverage',
  );
});

test('gpu-lab routes soft membrane pressure through isolated gpu-only module with WGSL offload state wiring', () => {
  assert.match(
    gpuLabSource,
    /applySoftMembraneCellPressureGpuOnly\([\s\S]*wgslOffload:\s*\{[\s\S]*enabled:\s*true,[\s\S]*device:\s*sim\?\.device,[\s\S]*state:\s*\(sim\.softMembranePressureWgslState \|\|= \{\}\),[\s\S]*\},/,
    'expected gpu-only membrane pressure dispatch to wire optional WGSL offload state in gpu-lab routing',
  );
});
