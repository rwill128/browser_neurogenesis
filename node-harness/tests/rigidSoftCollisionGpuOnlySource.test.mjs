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

test('rigid-soft gpu-only pass publishes cpu-prepared source route before WGSL broadphase bring-up', () => {
  assert.match(
    source,
    /if \(wgslOffload\?\.enabled === true && wgslOffload\?\.state\) \{[\s\S]*lastSourceRoute = 'cpu-rigid-soft-candidate-layout';[\s\S]*lastMode = 'cpu-prepared';/,
    'expected rigid-soft gpu-only pass to persist source-route ownership for deterministic candidate layout prep before WGSL broadphase dispatch/readback',
  );
});

test('rigid-soft gpu-only module dispatches WGSL broadphase with active-mask readback and can promote it to authoritative collision filtering', () => {
  assert.match(
    source,
    /encoder\.copyBufferToBuffer\([\s\S]*rigidSoftNodeBroadphaseActiveMaskReadback[\s\S]*mapAsync\(globalThis\.GPUMapMode\.READ[\s\S]*lastSourceRoute = 'wgsl-rigid-soft-node-broadphase-authoritative-filter';[\s\S]*lastMode = 'wgsl-broadphase-authoritative-filter';/,
    'expected rigid-soft gpu-only module to read back concrete WGSL broadphase masks and apply them as authoritative node-collision filtering when valid',
  );
});

test('rigid-soft gpu-only module compacts WGSL-filtered node pairs into deterministic narrowphase ownership buffers', () => {
  assert.match(
    source,
    /export function buildActiveRigidSoftNodeNarrowphasePairs\([\s\S]*compactRigidIndex[\s\S]*compactNodeIndex[\s\S]*lastPreparedNodeNarrowphasePairCount[\s\S]*lastPreparedNodeNarrowphaseSignature/,
    'expected rigid-soft gpu-only module to compact WGSL-active node broadphase pairs into deterministic typed-array ownership metadata for the next WGSL narrowphase stage',
  );
});

test('rigid-soft gpu-only module includes WGSL edge broadphase proposal + authoritative filter source telemetry', () => {
  assert.match(
    source,
    /const rigidSoftEdgeBroadphaseWgsl = \/\* wgsl \*\/[\s\S]*edgePairNodeAIndex[\s\S]*edgePairNodeBIndex[\s\S]*dispatchRigidSoftEdgeBroadphaseWgsl[\s\S]*lastEdgeBroadphaseAuthoritativeSource = 'wgsl-rigid-soft-edge-broadphase-authoritative-filter';/,
    'expected rigid-soft gpu-only module to run concrete WGSL edge broadphase math and expose authoritative source telemetry for edge filtering',
  );
});
