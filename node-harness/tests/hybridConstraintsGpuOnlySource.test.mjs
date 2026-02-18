import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const sourcePath = resolve(process.cwd(), 'sim-server/public/runtime-solvers/stepHybridConstraintsGpuOnly.js');
const source = readFileSync(sourcePath, 'utf8');

test('hybrid constraints gpu-only module publishes deterministic WGSL prep layout ownership for upcoming offload stage', () => {
  assert.match(
    source,
    /export function buildHybridAttachmentWgslPrep\([\s\S]*const nodeIndex = new Uint32Array\(count\);[\s\S]*const rigidIndex = new Uint32Array\(count\);[\s\S]*const anchorAX = new Float32Array\(count\);[\s\S]*const restB = new Float32Array\(count\);/,
    'expected gpu-only hybrid module to build deterministic typed-array layout for attachment anchors and rests',
  );

  assert.match(
    source,
    /wgslOffload\.state\.preparedPlan = prep\.plan;[\s\S]*wgslOffload\.state\.preparedLayout = prep\.layout;[\s\S]*lastPreparedValidAttachmentCount[\s\S]*lastPreparedLayoutBytes[\s\S]*lastMode = 'cpu-prepared';/,
    'expected gpu-only hybrid pass to persist WGSL prep ownership metadata while remaining CPU-authoritative',
  );
});

test('hybrid constraints gpu-only path dispatches a real WGSL attachment error probe stage when WebGPU offload is available', () => {
  assert.match(
    source,
    /const hybridAttachmentErrorProbeWgsl = \/\* wgsl \*\/[\s\S]*@compute @workgroup_size\([\s\S]*maxErrorOut\[ai\] = max\(errA, errB\);[\s\S]*function dispatchHybridAttachmentErrorProbe\([\s\S]*pass\.dispatchWorkgroups\([\s\S]*copyBufferToBuffer\([\s\S]*mapAsync\(globalThis\.GPUMapMode\.READ/,
    'expected gpu-only hybrid module to run a concrete WGSL attachment error probe dispatch with readback telemetry',
  );

  assert.match(
    source,
    /const serializedDispatch = \(wgslOffload\.state\.pendingWgslProbePromise \|\| Promise\.resolve\(\)\)[\s\S]*\.then\(\(\) => dispatchHybridAttachmentErrorProbe\(wgslOffload, prep, soft\)\);[\s\S]*pendingWgslProbePromise = serializedDispatch;/,
    'expected gpu-only hybrid WGSL probe dispatch to serialize map/readback work and avoid overlapping mapAsync races',
  );
});
