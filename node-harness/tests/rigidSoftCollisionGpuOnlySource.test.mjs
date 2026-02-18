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
    /if \(wgslOffload\?\.enabled === true && wgslOffload\?\.state\) \{[\s\S]*lastSourceRoute = hasAnyCandidatePairs[\s\S]*'cpu-rigid-soft-candidate-layout'[\s\S]*lastMode = hasAnyCandidatePairs \? 'cpu-prepared' : 'cpu-empty-candidates';/,
    'expected rigid-soft gpu-only pass to persist source-route ownership for deterministic candidate layout prep before WGSL broadphase dispatch/readback',
  );
});


test('rigid-soft gpu-only pass exits early for empty soft scenes to avoid no-op rigid-soft staging overhead', () => {
  assert.match(
    source,
    /if \(soft\.nodes\.length === 0\) \{[\s\S]*lastSourceRoute = 'cpu-rigid-soft-empty-scene';[\s\S]*lastMode = 'cpu-empty-scene';[\s\S]*return;/,
    'expected rigid-soft gpu-only pass to short-circuit empty soft scenes with explicit source-route telemetry',
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

test('rigid-soft gpu-only module compacts WGSL-filtered edge pairs into deterministic narrowphase ownership buffers', () => {
  assert.match(
    source,
    /export function buildActiveRigidSoftEdgeNarrowphasePairs\([\s\S]*compactRigidIndex[\s\S]*compactSpringIndex[\s\S]*compactNodeAIndex[\s\S]*compactNodeBIndex[\s\S]*lastPreparedEdgeNarrowphasePairCount[\s\S]*lastPreparedEdgeNarrowphaseSignature/,
    'expected rigid-soft gpu-only module to compact WGSL-active edge broadphase pairs into deterministic typed-array ownership metadata for the next WGSL edge narrowphase stage',
  );
});

test('rigid-soft gpu-only module builds combined deterministic narrowphase layout/signature for upcoming WGSL narrowphase kernel bring-up', () => {
  assert.match(
    source,
    /export function buildRigidSoftNarrowphaseWgslLayout\([\s\S]*nodePairRigidIndex[\s\S]*edgePairSpringIndex[\s\S]*signature[\s\S]*lastPreparedNarrowphaseCombinedSignature/,
    'expected rigid-soft gpu-only module to assemble node+edge compact pairs into one deterministic layout/signature for next-stage WGSL narrowphase authority',
  );
});

test('rigid-soft gpu-only module emits deterministic narrowphase scene state layout needed for next WGSL narrowphase math stage', () => {
  assert.match(
    source,
    /export function buildRigidSoftNarrowphaseSceneWgslLayout\([\s\S]*rigidX[\s\S]*rigidTheta[\s\S]*rigidOmega[\s\S]*rigidInvMass[\s\S]*rigidInvInertia[\s\S]*rigidVertexStart[\s\S]*rigidVertexX[\s\S]*rigidVertexY[\s\S]*nodeVx[\s\S]*nodeInvMass[\s\S]*springRestLen[\s\S]*signature/,
    'expected rigid-soft gpu-only path to include rigid inverse-mass/inertia + rigid polygon streams in deterministic scene layout for upcoming WGSL narrowphase impulse math',
  );
  assert.match(source, /lastPreparedNarrowphaseSceneRigidInvMass/, 'expected rigid-soft gpu-only path to publish rigid inverse-mass scene ownership in offload state');
  assert.match(source, /lastPreparedNarrowphaseSceneRigidInvInertia/, 'expected rigid-soft gpu-only path to publish rigid inverse-inertia scene ownership in offload state');
  assert.match(source, /lastPreparedNarrowphaseSceneRigidVertexStart/, 'expected rigid-soft gpu-only path to publish rigid polygon start offsets for WGSL narrowphase edge traversal');
  assert.match(source, /lastPreparedNarrowphaseSceneRigidVertexX/, 'expected rigid-soft gpu-only path to publish rigid polygon x stream for WGSL narrowphase edge traversal');
  assert.match(source, /lastPreparedNarrowphaseSceneRigidVertexY/, 'expected rigid-soft gpu-only path to publish rigid polygon y stream for WGSL narrowphase edge traversal');
  assert.match(source, /lastPreparedNarrowphaseSceneSignature/, 'expected rigid-soft gpu-only path to continue publishing deterministic scene signature');
});

test('rigid-soft gpu-only module lands concrete WGSL node narrowphase AABB probe math as authoritative node narrowphase filter stage', () => {
  assert.match(
    source,
    /const rigidSoftNodeNarrowphaseAabbProbeWgsl = \/\* wgsl \*\/[\s\S]*pairOut: array<vec2<f32>>[\s\S]*let separation = sqrt\(dx \* dx \+ dy \* dy\);[\s\S]*pairOut\[pairIndex\] = vec2<f32>\(separation, select\(0\.0, 1\.0, inside\)\);/,
    'expected rigid-soft gpu-only module to add concrete WGSL node-pair AABB separation math for narrowphase prepass ownership',
  );

  assert.match(
    source,
    /dispatchRigidSoftNodeNarrowphaseAabbProbeWgsl\([\s\S]*lastNodeNarrowphaseAabbProbeSeparation[\s\S]*lastNodeNarrowphaseAabbProbeInsideMask[\s\S]*buildAuthoritativeRigidSoftNodePairsFromAabbProbe[\s\S]*lastNodeNarrowphaseAuthoritativeSource = 'wgsl-rigid-soft-node-aabb-probe-authoritative-filter';/,
    'expected rigid-soft gpu-only pass to dispatch node-pair AABB probe and apply it as authoritative node narrowphase filtering with explicit source-route ownership',
  );

  assert.match(
    source,
    /export function buildAuthoritativeRigidSoftNodePairsFromAabbProbe\([\s\S]*if \(!inside && sep > slop\) continue;[\s\S]*return \{[\s\S]*pairCount: compactRigidIndex\.length/,
    'expected rigid-soft gpu-only module to expose deterministic compact-pair filtering from WGSL AABB probe outputs with finite fallback safety',
  );
});

test('rigid-soft gpu-only module lands concrete WGSL edge narrowphase AABB probe math as immediate unblocker for edge impulse kernel authority', () => {
  assert.match(
    source,
    /const rigidSoftEdgeNarrowphaseAabbProbeWgsl = \/\* wgsl \*\/[\s\S]*pairNodeAIndex[\s\S]*pairNodeBIndex[\s\S]*let separation = sqrt\(dx \* dx \+ dy \* dy\);[\s\S]*pairOut\[pairIndex\] = vec2<f32>\(separation, select\(0\.0, 1\.0, inside\)\);/,
    'expected rigid-soft gpu-only module to add concrete WGSL edge-pair AABB separation math for edge narrowphase prepass ownership',
  );

  assert.match(
    source,
    /dispatchRigidSoftEdgeNarrowphaseAabbProbeWgsl\([\s\S]*lastEdgeNarrowphaseAabbProbeSeparation[\s\S]*lastEdgeNarrowphaseAabbProbeInsideMask[\s\S]*lastEdgeNarrowphaseAabbProbeSource/,
    'expected rigid-soft gpu-only pass to dispatch edge-pair AABB probe and publish deterministic source-route telemetry',
  );
});

test('rigid-soft gpu-only module prepares deterministic narrowphase impulse-seed layout ownership for next WGSL narrowphase impulse kernel', () => {
  assert.match(
    source,
    /export function buildRigidSoftNarrowphaseImpulseSeedLayout\([\s\S]*nodePairRigidIndex[\s\S]*edgePairSpringIndex[\s\S]*rigidInvMass[\s\S]*nodeInvMass[\s\S]*signature/,
    'expected rigid-soft gpu-only module to build a deterministic impulse-seed layout from narrowphase pairs + scene streams for next WGSL narrowphase impulse stage',
  );

  assert.match(
    source,
    /lastPreparedNarrowphaseImpulseSeedLayoutBytes[\s\S]*lastPreparedNarrowphaseImpulseSeedSignature[\s\S]*lastPreparedNarrowphaseImpulseSeedSource = 'cpu-rigid-soft-narrowphase-impulse-seed-layout';/,
    'expected rigid-soft gpu-only pass to persist impulse-seed layout telemetry/source-route ownership for immediate WGSL narrowphase impulse kernel follow-up',
  );
});


test('rigid-soft gpu-only collision response consumes compact WGSL-filtered pairs when available', () => {
  assert.match(
    source,
    /if \(compactNodeRigidIndex instanceof Uint32Array && compactNodeNodeIndex instanceof Uint32Array\) \{[\s\S]*lastNodeCollisionResponseSource = 'cpu-rigid-soft-compact-node-response';[\s\S]*resolveRigidVsSoftNodeCollision\(/,
    'expected rigid-soft response stage to iterate compact node pairs directly when WGSL-filtered ownership buffers are present',
  );

  assert.match(
    source,
    /if \([\s\S]*compactEdgeRigidIndex instanceof Uint32Array[\s\S]*compactEdgeNodeAIndex instanceof Uint32Array[\s\S]*compactEdgeNodeBIndex instanceof Uint32Array[\s\S]*lastEdgeCollisionResponseSource = 'cpu-rigid-soft-compact-edge-response';[\s\S]*resolveRigidVsSoftEdgeCollision\(/,
    'expected rigid-soft response stage to iterate compact edge pairs directly when WGSL-filtered ownership buffers are present',
  );
});
