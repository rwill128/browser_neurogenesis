import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { MassPoint } from '../../js/classes/MassPoint.js';
import { MovementType, NodeType } from '../../js/classes/constants.js';

function makePoint(x, y) {
  const p = new MassPoint(x, y, 1, 5, 'rgba(255,255,255,1)');
  p.nodeType = NodeType.SWIMMER;
  p.movementType = MovementType.NEUTRAL;
  p.force.x = 0;
  p.force.y = 0;
  return p;
}

test('motion guard clamps excessive implicit velocity before invalid-motion kill', () => {
  const backup = {
    PHYSICS_MOTION_GUARD_ENABLED: config.PHYSICS_MOTION_GUARD_ENABLED,
    PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP: config.PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP,
    PHYSICS_MAX_ACCELERATION_MAGNITUDE: config.PHYSICS_MAX_ACCELERATION_MAGNITUDE,
    MAX_PIXELS_PER_FRAME_DISPLACEMENT: config.MAX_PIXELS_PER_FRAME_DISPLACEMENT
  };

  try {
    config.PHYSICS_MOTION_GUARD_ENABLED = true;
    config.PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP = 40;
    config.PHYSICS_MAX_ACCELERATION_MAGNITUDE = 1_000_000;
    config.MAX_PIXELS_PER_FRAME_DISPLACEMENT = 300;

    const body = new SoftBody(9101, 100, 100, null, false);
    const p = makePoint(100, 100);
    p.prevPos.x = 0;
    p.prevPos.y = 100;

    body.massPoints = [p];
    body.springs = [];
    body.spatialGrid = [];

    body._performPhysicalUpdates(1 / 60, null);

    const dx = p.pos.x - p.prevPos.x;
    const dy = p.pos.y - p.prevPos.y;
    const speed = Math.hypot(dx, dy);

    assert.equal(body.isUnstable, false);
    assert.ok(speed <= 40 + 1e-6);
    assert.ok(Number(body.motionGuardVelocityClampEvents) >= 1);
  } finally {
    Object.assign(config, backup);
  }
});

test('motion guard sanitizes non-finite forces before integration', () => {
  const backup = {
    PHYSICS_MOTION_GUARD_ENABLED: config.PHYSICS_MOTION_GUARD_ENABLED,
    PHYSICS_NONFINITE_FORCE_ZERO: config.PHYSICS_NONFINITE_FORCE_ZERO,
    PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP: config.PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP,
    PHYSICS_MAX_ACCELERATION_MAGNITUDE: config.PHYSICS_MAX_ACCELERATION_MAGNITUDE
  };

  try {
    config.PHYSICS_MOTION_GUARD_ENABLED = true;
    config.PHYSICS_NONFINITE_FORCE_ZERO = true;
    config.PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP = 300;
    config.PHYSICS_MAX_ACCELERATION_MAGNITUDE = 1_000_000;

    const body = new SoftBody(9102, 100, 100, null, false);
    const p = makePoint(120, 130);
    p.prevPos.x = 120;
    p.prevPos.y = 130;
    p.force.x = Number.NaN;
    p.force.y = Number.POSITIVE_INFINITY;

    body.massPoints = [p];
    body.springs = [];
    body.spatialGrid = [];

    body._performPhysicalUpdates(1 / 60, null);

    assert.equal(body.isUnstable, false);
    assert.equal(Number.isFinite(p.pos.x), true);
    assert.equal(Number.isFinite(p.pos.y), true);
    assert.ok(Number(body.motionGuardNonFiniteForceResets) >= 1);
  } finally {
    Object.assign(config, backup);
  }
});
