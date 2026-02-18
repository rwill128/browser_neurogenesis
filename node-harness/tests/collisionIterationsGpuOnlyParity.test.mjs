import test from 'node:test';
import assert from 'node:assert/strict';

import { runCollisionIterationsGpuOnly } from '../../sim-server/public/runtime-solvers/stepCollisionIterationsGpuOnly.js';
import { resolveRigidRigidCollisionPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidCollisionGpuOnly.js';
import { resolveRigidSoftCollisionPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js';
import { resolveSoftSoftCollisionPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftCollisionGpuOnly.js';
import { applyCollisionBoundaryPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepCollisionBoundaryGpuOnly.js';

const EDGE_BODY_MODE_BLOCK = 1;

function makeFixture() {
  return {
    n: 32,
    rigidBodies: [
      { x: 4.2, y: 8.4, vx: 0.6, vy: -0.3, theta: 0.1, omega: 0.02, r: 1.4, mass: 1.7, inertia: 0.8, sides: 4 },
      { x: 6.0, y: 8.6, vx: -0.4, vy: 0.25, theta: -0.05, omega: -0.01, r: 1.3, mass: 1.5, inertia: 0.75, sides: 5 },
    ],
    soft: {
      nodes: [
        { x: 4.8, y: 8.5, vx: 0.1, vy: 0.0, mass: 0.6, r: 0.9, clusterId: 1 },
        { x: 5.4, y: 9.1, vx: -0.08, vy: 0.04, mass: 0.6, r: 0.9, clusterId: 1 },
        { x: 6.2, y: 8.9, vx: 0.06, vy: -0.05, mass: 0.7, r: 0.95, clusterId: 2 },
      ],
      springs: [
        [0, 1, 1.0, EDGE_BODY_MODE_BLOCK],
        [1, 2, 1.1, EDGE_BODY_MODE_BLOCK],
      ],
    },
    hybrid: [
      { rigidIndex: 0, nodeIndex: 0 },
      { rigidIndex: 1, nodeIndex: 2 },
    ],
  };
}

function cloneFixture(fixture) {
  return structuredClone(fixture);
}

function resolveRigidVsRigidPolygonCollisionStub(a, b, slop, meta = {}) {
  const push = (Number(slop) || 0) * 0.01;
  a.vx += push;
  b.vx -= push;
  a.vy -= push * 0.5;
  b.vy += push * 0.5;
  if (Array.isArray(meta.contacts)) {
    meta.contacts.push({
      phase: meta.phase,
      iter: meta.iter,
      aIndex: meta.aIndex,
      bIndex: meta.bIndex,
      push,
    });
  }
}

function resolveCircleCollisionStub(a, b, slop) {
  const push = (Number(slop) || 0) * 0.005;
  a.vx -= push;
  b.vx += push;
  a.vy += push * 0.75;
  b.vy -= push * 0.75;
}

function resolveSoftNodeVsSoftEdgeCollisionStub(node, a, b, slop) {
  const push = (Number(slop) || 0) * 0.01;
  node.vx += push;
  node.vy -= push;
  a.vx -= push * 0.5;
  b.vy += push * 0.5;
}

function applyBounceBoundaryStub(body, n, bounce) {
  const min = 0.5;
  const max = n - 1.5;
  if (body.x < min) {
    body.x = min;
    if (body.vx < 0) body.vx = -body.vx * bounce;
  } else if (body.x > max) {
    body.x = max;
    if (body.vx > 0) body.vx = -body.vx * bounce;
  }
  if (body.y < min) {
    body.y = min;
    if (body.vy < 0) body.vy = -body.vy * bounce;
  } else if (body.y > max) {
    body.y = max;
    if (body.vy > 0) body.vy = -body.vy * bounce;
  }
}

async function runPreviousInlineGpuOnlyOrchestration(state) {
  const hybridAttachedByRigid = new Map();
  for (const h of state.hybrid || []) {
    const ri = Number(h?.rigidIndex) | 0;
    const ni = Number(h?.nodeIndex) | 0;
    if (ri < 0 || ri >= state.rigidBodies.length) continue;
    if (ni < 0 || ni >= state.soft.nodes.length) continue;
    if (!hybridAttachedByRigid.has(ri)) hybridAttachedByRigid.set(ri, new Set());
    hybridAttachedByRigid.get(ri).add(ni);
  }

  const rigidContactDebug = [];
  for (let iter = 0; iter < 2; iter++) {
    resolveRigidRigidCollisionPassGpuOnly({
      rigidBodies: state.rigidBodies,
      slop: 0.32,
      contacts: rigidContactDebug,
      iter,
      phase: 'pre-soft',
      resolveRigidVsRigidPolygonCollision: resolveRigidVsRigidPolygonCollisionStub,
    });

    resolveRigidSoftCollisionPassGpuOnly({
      rigidBodies: state.rigidBodies,
      soft: state.soft,
      hybridAttachedByRigid,
      edgeBodyModeBlock: EDGE_BODY_MODE_BLOCK,
      nodeSlop: 0.18,
      edgeSlop: 0.16,
    });

    resolveSoftSoftCollisionPassGpuOnly({
      soft: state.soft,
      resolveCircleCollision: resolveCircleCollisionStub,
      resolveSoftNodeVsSoftEdgeCollision: resolveSoftNodeVsSoftEdgeCollisionStub,
      edgeBodyModeBlock: EDGE_BODY_MODE_BLOCK,
      nodeNodeSlop: 0.22,
      nodeEdgeSlop: 0.12,
    });

    resolveRigidRigidCollisionPassGpuOnly({
      rigidBodies: state.rigidBodies,
      slop: 0.32,
      contacts: rigidContactDebug,
      iter,
      phase: 'post-soft',
      resolveRigidVsRigidPolygonCollision: resolveRigidVsRigidPolygonCollisionStub,
    });

    await applyCollisionBoundaryPassGpuOnly({
      rigidBodies: state.rigidBodies,
      soft: state.soft,
      n: state.n,
      rigidBounce: 0.84,
      softBounce: 0.78,
      applyBounceBoundary: applyBounceBoundaryStub,
    });
  }

  return { rigidContactDebug, hybridAttachedByRigid };
}

function serializeHybridMap(map) {
  return Array.from(map.entries())
    .map(([k, v]) => [k, Array.from(v.values()).sort((a, b) => a - b)])
    .sort((a, b) => a[0] - b[0]);
}

test('collision iteration parity: gpu-only orchestrator matches prior inline collision stepping schedule', async () => {
  const fixture = makeFixture();
  const baseline = cloneFixture(fixture);
  const gpuOnly = cloneFixture(fixture);

  const baselineResult = await runPreviousInlineGpuOnlyOrchestration(baseline);

  const rigidContactDebug = [];
  const gpuOnlyResult = await runCollisionIterationsGpuOnly({
    rigidBodies: gpuOnly.rigidBodies,
    soft: gpuOnly.soft,
    hybrid: gpuOnly.hybrid,
    rigidContactDebug,
    collisionIterations: 2,
    rigidRigidSlop: 0.32,
    rigidSoftNodeSlop: 0.18,
    rigidSoftEdgeSlop: 0.16,
    softNodeNodeSlop: 0.22,
    softNodeEdgeSlop: 0.12,
    edgeBodyModeBlock: EDGE_BODY_MODE_BLOCK,
    n: gpuOnly.n,
    rigidBounce: 0.84,
    softBounce: 0.78,
    resolveRigidRigidCollisionPassGpuOnly,
    resolveRigidSoftCollisionPassGpuOnly,
    resolveSoftSoftCollisionPassGpuOnly,
    applyCollisionBoundaryPassGpuOnly,
    resolveRigidVsRigidPolygonCollision: resolveRigidVsRigidPolygonCollisionStub,
    resolveCircleCollision: resolveCircleCollisionStub,
    resolveSoftNodeVsSoftEdgeCollision: resolveSoftNodeVsSoftEdgeCollisionStub,
    applyBounceBoundary: applyBounceBoundaryStub,
  });

  assert.deepEqual(gpuOnly, baseline);
  assert.deepEqual(rigidContactDebug, baselineResult.rigidContactDebug);
  assert.deepEqual(
    serializeHybridMap(gpuOnlyResult.hybridAttachedByRigid),
    serializeHybridMap(baselineResult.hybridAttachedByRigid),
  );
});
