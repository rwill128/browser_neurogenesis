import test from 'node:test';
import assert from 'node:assert/strict';

import { resolveSoftSoftCollisionPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftCollisionGpuOnly.js';

const EDGE_BODY_MODE = {
  PASS: 0,
  BLOCK: 1,
};

function makeState() {
  return {
    nodes: [
      { id: 'n0', clusterId: 0 },
      { id: 'n1', clusterId: 1 },
      { id: 'n2', clusterId: 0 },
      { id: 'n3', clusterId: 2 },
    ],
    springs: [
      [0, 1, 1.1, EDGE_BODY_MODE.BLOCK],
      [1, 2, 1.0, EDGE_BODY_MODE.PASS],
      [2, 3, 1.2, EDGE_BODY_MODE.BLOCK],
      [0, 2, 0.9, EDGE_BODY_MODE.BLOCK],
    ],
  };
}

function runBaseline(soft, calls) {
  for (let i = 0; i < soft.nodes.length; i++) {
    for (let j = i + 1; j < soft.nodes.length; j++) {
      calls.nodeNode.push(`${soft.nodes[i].id}-${soft.nodes[j].id}`);
    }
  }

  for (let ni = 0; ni < soft.nodes.length; ni++) {
    const node = soft.nodes[ni];
    for (const [i, j, _rest, edgeBodyMode] of soft.springs) {
      if (edgeBodyMode !== EDGE_BODY_MODE.BLOCK) continue;
      if (i === ni || j === ni) continue;
      const a = soft.nodes[i];
      const b = soft.nodes[j];
      if (a.clusterId === node.clusterId && b.clusterId === node.clusterId) continue;
      calls.nodeEdge.push(`${node.id}->${a.id}-${b.id}`);
    }
  }
}

test('gpu-only soft collision pass matches baseline soft-soft collision visitation/filtering', () => {
  const baselineSoft = makeState();
  const gpuSoft = makeState();

  const baselineCalls = { nodeNode: [], nodeEdge: [] };
  runBaseline(baselineSoft, baselineCalls);

  const gpuCalls = { nodeNode: [], nodeEdge: [] };
  resolveSoftSoftCollisionPassGpuOnly({
    soft: gpuSoft,
    resolveCircleCollision: (a, b) => {
      gpuCalls.nodeNode.push(`${a.id}-${b.id}`);
    },
    resolveSoftNodeVsSoftEdgeCollision: (node, a, b) => {
      gpuCalls.nodeEdge.push(`${node.id}->${a.id}-${b.id}`);
    },
    edgeBodyModeBlock: EDGE_BODY_MODE.BLOCK,
    nodeNodeSlop: 0.22,
    nodeEdgeSlop: 0.12,
  });

  assert.deepEqual(gpuCalls, baselineCalls);
});
