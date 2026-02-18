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
    /function buildSoftMembranePressureWgslPrep\([\s\S]*const membraneOffsets = new Uint32Array\(membraneCount \+ 1\);[\s\S]*const loopIndices = new Uint32Array\(indexCount\);[\s\S]*const areaBase = new Float32Array\(membraneCount\);[\s\S]*const nodeX = new Float32Array\(nodeCount\);/,
    'expected gpu-only membrane pressure pass to build deterministic typed-array layout for membrane loops and node positions',
  );

  assert.match(
    source,
    /wgslOffload\.state\.preparedPlan = prep\.plan;[\s\S]*wgslOffload\.state\.preparedLayout = prep\.layout;[\s\S]*lastPreparedMembraneCount[\s\S]*lastPreparedLayoutBytes[\s\S]*lastMode = 'cpu-prepared';/,
    'expected membrane pressure pass to persist WGSL prep ownership while keeping CPU-authoritative stepping',
  );
});

test('soft membrane pressure gpu-only path dispatches WGSL area-probe stage with serialized readback', () => {
  assert.match(
    source,
    /const softMembranePressureAreaProbeWgsl = \/\* wgsl \*\/[\s\S]*@compute @workgroup_size\([\s\S]*area_now_out\[mi\] = now;[\s\S]*function dispatchSoftMembranePressureAreaProbe\([\s\S]*pass\.dispatchWorkgroups\([\s\S]*copyBufferToBuffer\([\s\S]*mapAsync\(globalThis\.GPUMapMode\.READ/,
    'expected membrane pressure module to run a concrete WGSL area-probe dispatch with readback telemetry',
  );

  assert.match(
    source,
    /const serializedDispatch = \(wgslOffload\.state\.pendingWgslAreaProbePromise \|\| Promise\.resolve\(\)\)[\s\S]*dispatchSoftMembranePressureAreaProbe\(wgslOffload, prep\)[\s\S]*pendingWgslAreaProbePromise = serializedDispatch;/,
    'expected membrane pressure WGSL dispatch to serialize map/readback work and avoid overlapping mapAsync races',
  );
});

test('gpu-lab routes soft membrane pressure through isolated gpu-only module with WGSL offload state wiring', () => {
  assert.match(
    gpuLabSource,
    /applySoftMembraneCellPressureGpuOnly\([\s\S]*wgslOffload:\s*\{[\s\S]*enabled:\s*true,[\s\S]*device:\s*sim\?\.device,[\s\S]*state:\s*\(sim\.softMembranePressureWgslState \|\|= \{\}\),[\s\S]*\},/,
    'expected gpu-only membrane pressure dispatch to wire optional WGSL offload state in gpu-lab routing',
  );
});
