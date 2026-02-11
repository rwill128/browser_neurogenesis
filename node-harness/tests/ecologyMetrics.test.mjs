import test from 'node:test';
import assert from 'node:assert/strict';

import {
  aggregateNodeTypeCounts,
  computeNodeDiversity,
  summarizeGrowthCohorts
} from '../../js/engine/ecologyMetrics.mjs';

test('aggregateNodeTypeCounts counts only living body points', () => {
  const bodies = [
    { isUnstable: false, massPoints: [{ nodeType: 1 }, { nodeType: 2 }] },
    { isUnstable: false, massPoints: [{ nodeType: 1 }] },
    { isUnstable: true, massPoints: [{ nodeType: 9 }] }
  ];
  const names = { 1: 'EATER', 2: 'PHOTOSYNTHETIC', 9: 'REPULSOR' };

  const out = aggregateNodeTypeCounts(bodies, names);
  assert.deepEqual(out, { EATER: 2, PHOTOSYNTHETIC: 1 });
});

test('computeNodeDiversity returns richness and bounded evenness', () => {
  const out = computeNodeDiversity({ A: 4, B: 4, C: 2 });
  assert.equal(out.totalNodes, 10);
  assert.equal(out.richness, 3);
  assert.ok(out.shannonEntropy > 0);
  assert.ok(out.shannonEvenness >= 0 && out.shannonEvenness <= 1);
});

test('summarizeGrowthCohorts computes active/high grower fractions', () => {
  const bodies = [
    { isUnstable: false, growthEventsCompleted: 0 },
    { isUnstable: false, growthEventsCompleted: 2 },
    { isUnstable: false, growthEventsCompleted: 8 },
    { isUnstable: true, growthEventsCompleted: 20 }
  ];

  const out = summarizeGrowthCohorts(bodies, { activeThreshold: 1, highThreshold: 5 });
  assert.equal(out.livingCreatures, 3);
  assert.equal(out.activeGrowers, 2);
  assert.equal(out.highGrowers, 1);
  assert.equal(Number(out.activeGrowerFraction.toFixed(6)), Number((2 / 3).toFixed(6)));
  assert.equal(Number(out.highGrowerFraction.toFixed(6)), Number((1 / 3).toFixed(6)));
});
