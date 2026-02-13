import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { MovementType, NodeType } from '../../js/classes/constants.js';

test('optional edge-length hard cap projects overstretched springs before instability check', () => {
  const cfgBackup = {
    EDGE_LENGTH_HARD_CAP_ENABLED: config.EDGE_LENGTH_HARD_CAP_ENABLED,
    EDGE_LENGTH_HARD_CAP_FACTOR: config.EDGE_LENGTH_HARD_CAP_FACTOR,
    MAX_SPRING_STRETCH_FACTOR: config.MAX_SPRING_STRETCH_FACTOR
  };

  try {
    config.EDGE_LENGTH_HARD_CAP_ENABLED = true;
    config.EDGE_LENGTH_HARD_CAP_FACTOR = 1.2;
    config.MAX_SPRING_STRETCH_FACTOR = 20;

    const body = new SoftBody(9001, 100, 100, null, false);
    const a = body.massPoints[0];
    const b = new SoftBody(9002, 100, 100, null, false).massPoints[0];

    a.nodeType = NodeType.SWIMMER;
    b.nodeType = NodeType.SWIMMER;
    a.movementType = MovementType.NEUTRAL;
    b.movementType = MovementType.NEUTRAL;

    a.pos.x = 100; a.pos.y = 100; a.prevPos.x = 100; a.prevPos.y = 100;
    b.pos.x = 200; b.pos.y = 100; b.prevPos.x = 200; b.prevPos.y = 100;

    body.massPoints = [a, b];
    body.springs = [new Spring(a, b, 500, 5, 10, false)];
    body.creatureEnergy = 100;
    body.currentMaxEnergy = 100;
    body.spatialGrid = [];

    body._finalizeUpdateAndCheckStability(1 / 60);

    const dx = a.pos.x - b.pos.x;
    const dy = a.pos.y - b.pos.y;
    const currentLength = Math.hypot(dx, dy);

    assert.equal(body.isUnstable, false);
    assert.ok(currentLength <= (10 * 1.2) + 1e-6);
    assert.ok(Number(body.edgeLengthClampEvents) >= 1);
  } finally {
    Object.assign(config, cfgBackup);
  }
});
