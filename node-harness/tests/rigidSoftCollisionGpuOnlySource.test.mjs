import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const SOURCE_PATH = new URL('../../sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js', import.meta.url);
const source = readFileSync(SOURCE_PATH, 'utf8');

test('rigid-soft gpu-only module prepares deterministic candidate layout ownership for upcoming WGSL collision stage', () => {
  assert.match(
    source,
    /function buildRigidSoftCollisionWgslLayout\([\s\S]*nodePairRigidIndex[\s\S]*edgePairSpringIndex[\s\S]*lastPreparedLayoutSignature/,
    'expected rigid-soft gpu-only module to build deterministic typed-array candidate ownership metadata for WGSL collision offload bring-up',
  );
});

test('rigid-soft gpu-only pass publishes cpu-prepared source route while keeping cpu-authoritative collision behavior', () => {
  assert.match(
    source,
    /if \(wgslOffload\?\.enabled === true && wgslOffload\?\.state\) \{[\s\S]*lastSourceRoute = 'cpu-rigid-soft-candidate-layout';[\s\S]*lastMode = 'cpu-prepared';/,
    'expected rigid-soft gpu-only pass to persist source-route ownership for deterministic candidate layout prep before WGSL dispatch lands',
  );
});

test('rigid-soft gpu-only module dispatches concrete WGSL node broadphase proposal stage and records source-route ownership', () => {
  assert.match(
    source,
    /const rigidSoftNodeBroadphaseWgsl = \/\* wgsl \*\/[\s\S]*activeMaskOut\[pairIndex\][\s\S]*dispatchRigidSoftNodeBroadphaseWgsl\([\s\S]*lastMode = 'cpu-authoritative-wgsl-broadphase-proposal';/,
    'expected rigid-soft gpu-only module to run a concrete WGSL node broadphase dispatch and publish source-route ownership while CPU narrowphase remains authoritative',
  );
});
