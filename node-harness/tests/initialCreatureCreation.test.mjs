import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { runtimeState } from '../../js/engine/runtimeState.js';

function withConfigPatch(patch, fn) {
  const backup = {};
  for (const [key, value] of Object.entries(patch)) {
    backup[key] = config[key];
  }

  const runtimeBackup = runtimeState.mutationStats;

  try {
    Object.assign(config, patch);
    runtimeState.mutationStats = {};
    return fn();
  } finally {
    Object.assign(config, backup);
    runtimeState.mutationStats = runtimeBackup;
  }
}

function classifyTriTemplate(body) {
  const pointCount = body.blueprintPoints.length;
  const springCount = body.blueprintSprings.length;

  if (pointCount === 3 && springCount === 3) return 'triangle';
  if (pointCount === 4 && springCount === 5) return 'diamond';
  if (pointCount === 7 && springCount === 12) return 'hexagon';
  return 'unknown';
}

function assertSharedEdgePrimitiveInvariants(body) {
  assert.equal(body.shapeType, 3);
  assert.ok(body.blueprintSprings.length > 0);

  const firstRestLength = Number(body.blueprintSprings[0].restLength);
  assert.ok(Number.isFinite(firstRestLength) && firstRestLength > 0);

  const edgeKeys = new Set();
  for (const spring of body.blueprintSprings) {
    const p1 = Number(spring.p1Index);
    const p2 = Number(spring.p2Index);
    const key = `${Math.min(p1, p2)}:${Math.max(p1, p2)}`;
    edgeKeys.add(key);

    const restLength = Number(spring.restLength);
    assert.ok(Number.isFinite(restLength) && restLength > 0);
    assert.ok(Math.abs(restLength - firstRestLength) < 1e-6);
  }

  assert.equal(edgeKeys.size, body.blueprintSprings.length);
}

test('initial creature creation uses stable shared-edge triangle primitives and preserves diversity', () => {
  const templateCases = [
    {
      name: 'triangle',
      weights: {
        INITIAL_TRI_TEMPLATE_WEIGHT_TRIANGLE: 1,
        INITIAL_TRI_TEMPLATE_WEIGHT_DIAMOND: 0,
        INITIAL_TRI_TEMPLATE_WEIGHT_HEXAGON: 0
      },
      points: 3,
      springs: 3
    },
    {
      name: 'diamond',
      weights: {
        INITIAL_TRI_TEMPLATE_WEIGHT_TRIANGLE: 0,
        INITIAL_TRI_TEMPLATE_WEIGHT_DIAMOND: 1,
        INITIAL_TRI_TEMPLATE_WEIGHT_HEXAGON: 0
      },
      points: 4,
      springs: 5
    },
    {
      name: 'hexagon',
      weights: {
        INITIAL_TRI_TEMPLATE_WEIGHT_TRIANGLE: 0,
        INITIAL_TRI_TEMPLATE_WEIGHT_DIAMOND: 0,
        INITIAL_TRI_TEMPLATE_WEIGHT_HEXAGON: 1
      },
      points: 7,
      springs: 12
    }
  ];

  for (const tc of templateCases) {
    withConfigPatch(
      {
        INITIAL_TRIANGULATED_PRIMITIVES_ENABLED: true,
        INITIAL_TRI_MESH_EDGE_RIGID_CHANCE: 0,
        ...tc.weights
      },
      () => {
        const body = new SoftBody(10_000 + tc.points, 150, 180, null, false);
        assert.equal(classifyTriTemplate(body), tc.name);
        assert.equal(body.blueprintPoints.length, tc.points);
        assert.equal(body.blueprintSprings.length, tc.springs);
        assertSharedEdgePrimitiveInvariants(body);
      }
    );
  }

  withConfigPatch(
    {
      INITIAL_TRIANGULATED_PRIMITIVES_ENABLED: true,
      INITIAL_TRI_MESH_EDGE_RIGID_CHANCE: 0,
      INITIAL_TRI_TEMPLATE_WEIGHT_TRIANGLE: 0.25,
      INITIAL_TRI_TEMPLATE_WEIGHT_DIAMOND: 0.35,
      INITIAL_TRI_TEMPLATE_WEIGHT_HEXAGON: 0.4
    },
    () => {
      const counts = { triangle: 0, diamond: 0, hexagon: 0, unknown: 0 };
      const total = 60;

      for (let i = 0; i < total; i++) {
        const body = new SoftBody(20_000 + i, 200, 200, null, false);
        const kind = classifyTriTemplate(body);
        counts[kind] = (counts[kind] || 0) + 1;
        assertSharedEdgePrimitiveInvariants(body);
      }

      assert.equal(counts.unknown, 0);
      assert.ok(counts.triangle > 0, `expected at least one triangle, got ${JSON.stringify(counts)}`);
      assert.ok(counts.diamond > 0, `expected at least one diamond, got ${JSON.stringify(counts)}`);
      assert.ok(counts.hexagon > 0, `expected at least one hexagon, got ${JSON.stringify(counts)}`);

      const maxShare = Math.max(counts.triangle, counts.diamond, counts.hexagon) / total;
      assert.ok(maxShare < 0.9, `unexpected homogeneity: ${JSON.stringify(counts)}`);
    }
  );
});
