import test from 'node:test';
import assert from 'node:assert/strict';

import { buildRenderableSoakSnapshot } from '../soakSnapshot.mjs';

test('buildRenderableSoakSnapshot emits renderable creature vertices/springs', () => {
  const p1 = { pos: { x: 10, y: 20 }, radius: 2, nodeType: 1 };
  const p2 = { pos: { x: 30, y: 40 }, radius: 3, nodeType: 2 };
  const body = {
    id: 7,
    isUnstable: false,
    creatureEnergy: 12.5,
    massPoints: [p1, p2],
    springs: [{ p1, p2, isRigid: true }],
    getAveragePosition: () => ({ x: 20, y: 30 })
  };

  const snap = buildRenderableSoakSnapshot({
    worldState: { softBodyPopulation: [body], particles: [] },
    worldWidth: 100,
    worldHeight: 80,
    nodeTypeEnum: { EATER: 1, NEURON: 2 }
  });

  assert.equal(snap.world.width, 100);
  assert.equal(snap.world.height, 80);
  assert.equal(snap.creatures.length, 1);
  assert.equal(snap.creatures[0].vertices.length, 2);
  assert.equal(snap.creatures[0].springs.length, 1);
  assert.equal(snap.creatures[0].vertices[0].nodeTypeName, 'EATER');
  assert.equal(snap.creatures[0].vertices[1].nodeTypeName, 'NEURON');
});
