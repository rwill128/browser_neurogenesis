import test from 'node:test';
import assert from 'node:assert/strict';

import { resolveRigidSoftCollisionPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js';

const EDGE_BODY_MODE = {
  PASS: 0,
  BLOCK: 1,
};

function makeState() {
  return {
    rigid: [
      { id: 'r0' },
      { id: 'r1' },
    ],
    soft: {
      nodes: [
        { id: 'n0' },
        { id: 'n1' },
        { id: 'n2' },
        { id: 'n3' },
      ],
      springs: [
        [0, 1, 1.2, EDGE_BODY_MODE.BLOCK],
        [1, 2, 1.0, EDGE_BODY_MODE.PASS],
        [2, 3, 1.1, EDGE_BODY_MODE.BLOCK],
      ],
    },
    hybridAttachedByRigid: new Map([
      [0, new Set([1])],
      [1, new Set([2])],
    ]),
  };
}

function runBaseline(state, calls) {
  for (let rbi = 0; rbi < state.rigid.length; rbi++) {
    const rb = state.rigid[rbi];
    const attachedNodeSet = state.hybridAttachedByRigid.get(rbi) || null;

    for (let ni = 0; ni < state.soft.nodes.length; ni++) {
      if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
      const sn = state.soft.nodes[ni];
      calls.nodes.push(`${rb.id}->${sn.id}`);
    }

    for (const [i, j, _rest, edgeBodyMode] of state.soft.springs) {
      if (edgeBodyMode !== EDGE_BODY_MODE.BLOCK) continue;
      if (attachedNodeSet && (attachedNodeSet.has(i) || attachedNodeSet.has(j))) continue;
      calls.edges.push(`${rb.id}->${state.soft.nodes[i].id}-${state.soft.nodes[j].id}`);
    }
  }
}

test('gpu-only rigid-soft collision pass matches baseline contact visitation and filtering', () => {
  const baselineState = makeState();
  const gpuState = makeState();

  const baselineCalls = { nodes: [], edges: [] };
  runBaseline(baselineState, baselineCalls);

  const gpuCalls = { nodes: [], edges: [] };
  resolveRigidSoftCollisionPassGpuOnly({
    rigidBodies: gpuState.rigid,
    soft: gpuState.soft,
    hybridAttachedByRigid: gpuState.hybridAttachedByRigid,
    resolveRigidVsSoftNodeCollision: (rb, sn) => {
      gpuCalls.nodes.push(`${rb.id}->${sn.id}`);
    },
    resolveRigidVsSoftEdgeCollision: (rb, a, b) => {
      gpuCalls.edges.push(`${rb.id}->${a.id}-${b.id}`);
    },
    edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
    nodeSlop: 0.18,
    edgeSlop: 0.16,
  });

  assert.deepEqual(gpuCalls, baselineCalls);
});
