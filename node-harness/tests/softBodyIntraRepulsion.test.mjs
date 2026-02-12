import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { MovementType, NodeType } from '../../js/classes/constants.js';

function dist(a, b) {
  const dx = a.pos.x - b.pos.x;
  const dy = a.pos.y - b.pos.y;
  return Math.sqrt(dx * dx + dy * dy);
}

test('intra-body repulsion separates overlapping non-connected points', () => {
  const cfgBackup = {
    INTRA_BODY_REPULSION_ENABLED: config.INTRA_BODY_REPULSION_ENABLED,
    INTRA_BODY_REPULSION_STRENGTH: config.INTRA_BODY_REPULSION_STRENGTH,
    INTRA_BODY_REPULSION_RADIUS_FACTOR: config.INTRA_BODY_REPULSION_RADIUS_FACTOR,
    INTRA_BODY_REPULSION_SKIP_CONNECTED: config.INTRA_BODY_REPULSION_SKIP_CONNECTED,
    WORLD_WIDTH: config.WORLD_WIDTH,
    WORLD_HEIGHT: config.WORLD_HEIGHT,
    KILL_ON_OUT_OF_BOUNDS: config.KILL_ON_OUT_OF_BOUNDS
  };

  try {
    config.INTRA_BODY_REPULSION_ENABLED = true;
    config.INTRA_BODY_REPULSION_STRENGTH = 10;
    config.INTRA_BODY_REPULSION_RADIUS_FACTOR = 1.2;
    config.INTRA_BODY_REPULSION_SKIP_CONNECTED = true;
    config.WORLD_WIDTH = 1000;
    config.WORLD_HEIGHT = 1000;
    config.KILL_ON_OUT_OF_BOUNDS = false;

    const body = new SoftBody(91, 200, 200, null, false);
    const p1 = body.massPoints[0];
    const p2 = new SoftBody(92, 202, 200, null, false).massPoints[0];

    p1.nodeType = NodeType.SWIMMER;
    p2.nodeType = NodeType.SWIMMER;
    p1.movementType = MovementType.NEUTRAL;
    p2.movementType = MovementType.NEUTRAL;
    p1.radius = 5;
    p2.radius = 5;
    p1.pos.x = 200; p1.pos.y = 200; p1.prevPos.x = 200; p1.prevPos.y = 200;
    p2.pos.x = 202; p2.pos.y = 200; p2.prevPos.x = 202; p2.prevPos.y = 200;

    body.massPoints = [p1, p2];
    body.springs = [];

    const before = dist(p1, p2);
    body._performPhysicalUpdates(0.1, null);
    const after = dist(p1, p2);

    assert.equal(body.isUnstable, false);
    assert.ok(after > before, `expected separation increase, got before=${before} after=${after}`);
  } finally {
    Object.assign(config, cfgBackup);
  }
});

test('intra-body repulsion skips directly connected points when configured', () => {
  const cfgBackup = {
    INTRA_BODY_REPULSION_ENABLED: config.INTRA_BODY_REPULSION_ENABLED,
    INTRA_BODY_REPULSION_STRENGTH: config.INTRA_BODY_REPULSION_STRENGTH,
    INTRA_BODY_REPULSION_RADIUS_FACTOR: config.INTRA_BODY_REPULSION_RADIUS_FACTOR,
    INTRA_BODY_REPULSION_SKIP_CONNECTED: config.INTRA_BODY_REPULSION_SKIP_CONNECTED,
    WORLD_WIDTH: config.WORLD_WIDTH,
    WORLD_HEIGHT: config.WORLD_HEIGHT,
    KILL_ON_OUT_OF_BOUNDS: config.KILL_ON_OUT_OF_BOUNDS
  };

  try {
    config.INTRA_BODY_REPULSION_ENABLED = true;
    config.INTRA_BODY_REPULSION_STRENGTH = 20;
    config.INTRA_BODY_REPULSION_RADIUS_FACTOR = 1.2;
    config.INTRA_BODY_REPULSION_SKIP_CONNECTED = true;
    config.WORLD_WIDTH = 1000;
    config.WORLD_HEIGHT = 1000;
    config.KILL_ON_OUT_OF_BOUNDS = false;

    const body = new SoftBody(93, 300, 300, null, false);
    const p1 = body.massPoints[0];
    const p2 = new SoftBody(94, 302, 300, null, false).massPoints[0];

    p1.nodeType = NodeType.SWIMMER;
    p2.nodeType = NodeType.SWIMMER;
    p1.movementType = MovementType.NEUTRAL;
    p2.movementType = MovementType.NEUTRAL;
    p1.radius = 5;
    p2.radius = 5;
    p1.pos.x = 300; p1.pos.y = 300; p1.prevPos.x = 300; p1.prevPos.y = 300;
    p2.pos.x = 302; p2.pos.y = 300; p2.prevPos.x = 302; p2.prevPos.y = 300;

    body.massPoints = [p1, p2];
    body.springs = [new Spring(p1, p2, 500, 5, 2, false)];

    const before = dist(p1, p2);
    body._performPhysicalUpdates(0.1, null);
    const after = dist(p1, p2);

    assert.equal(body.isUnstable, false);
    assert.ok(Math.abs(after - before) < 1e-6, `expected connected pair unchanged, before=${before} after=${after}`);
  } finally {
    Object.assign(config, cfgBackup);
  }
});
