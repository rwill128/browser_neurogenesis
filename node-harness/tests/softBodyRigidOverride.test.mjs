import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { MassPoint } from '../../js/classes/MassPoint.js';
import { MovementType, NodeType } from '../../js/classes/constants.js';

function makePoint(x, y) {
  const p = new MassPoint(x, y, 1, 5, 'rgba(255,255,255,1)');
  p.nodeType = NodeType.SWIMMER;
  p.movementType = MovementType.NEUTRAL;
  p.prevPos.x = x;
  p.prevPos.y = y;
  p.force.x = 0;
  p.force.y = 0;
  return p;
}

test('FORCE_ALL_SPRINGS_RIGID turns non-rigid springs rigid during physics update', () => {
  const backup = {
    FORCE_ALL_SPRINGS_RIGID: config.FORCE_ALL_SPRINGS_RIGID,
    RIGID_SPRING_STIFFNESS: config.RIGID_SPRING_STIFFNESS,
    RIGID_SPRING_DAMPING: config.RIGID_SPRING_DAMPING,
    RIGID_CONSTRAINT_PROJECTION_ENABLED: config.RIGID_CONSTRAINT_PROJECTION_ENABLED,
    RIGID_CONSTRAINT_PROJECTION_ITERATIONS: config.RIGID_CONSTRAINT_PROJECTION_ITERATIONS,
    RIGID_CONSTRAINT_MAX_RELATIVE_ERROR: config.RIGID_CONSTRAINT_MAX_RELATIVE_ERROR
  };

  try {
    config.FORCE_ALL_SPRINGS_RIGID = true;
    config.RIGID_SPRING_STIFFNESS = 43210;
    config.RIGID_SPRING_DAMPING = 98;
    config.RIGID_CONSTRAINT_PROJECTION_ENABLED = true;
    config.RIGID_CONSTRAINT_PROJECTION_ITERATIONS = 8;
    config.RIGID_CONSTRAINT_MAX_RELATIVE_ERROR = 0.0001;

    const body = new SoftBody(9201, 100, 100, null, false);
    const a = makePoint(100, 100);
    const b = makePoint(110, 100);

    body.massPoints = [a, b];
    const spring = new Spring(a, b, 500, 5, 10, false);
    body.springs = [spring];
    body.spatialGrid = [];

    body._performPhysicalUpdates(1 / 60, null);

    assert.equal(spring.isRigid, true);
    assert.equal(Number(spring.stiffness), Number(config.RIGID_SPRING_STIFFNESS));
    assert.equal(Number(spring.dampingFactor), Number(config.RIGID_SPRING_DAMPING));
  } finally {
    Object.assign(config, backup);
  }
});

test('rigid-constraint projection keeps rigid spring length near rest length after integration', () => {
  const backup = {
    FORCE_ALL_SPRINGS_RIGID: config.FORCE_ALL_SPRINGS_RIGID,
    RIGID_CONSTRAINT_PROJECTION_ENABLED: config.RIGID_CONSTRAINT_PROJECTION_ENABLED,
    RIGID_CONSTRAINT_PROJECTION_ITERATIONS: config.RIGID_CONSTRAINT_PROJECTION_ITERATIONS,
    RIGID_CONSTRAINT_MAX_RELATIVE_ERROR: config.RIGID_CONSTRAINT_MAX_RELATIVE_ERROR,
    INTRA_BODY_REPULSION_ENABLED: config.INTRA_BODY_REPULSION_ENABLED
  };

  try {
    config.FORCE_ALL_SPRINGS_RIGID = false;
    config.RIGID_CONSTRAINT_PROJECTION_ENABLED = true;
    config.RIGID_CONSTRAINT_PROJECTION_ITERATIONS = 12;
    config.RIGID_CONSTRAINT_MAX_RELATIVE_ERROR = 0.0001;
    config.INTRA_BODY_REPULSION_ENABLED = false;

    const body = new SoftBody(9202, 100, 100, null, false);
    const a = makePoint(100, 100);
    const b = makePoint(140, 100); // intentionally overstretched vs restLength

    body.massPoints = [a, b];
    const spring = new Spring(a, b, 500, 5, 10, true);
    body.springs = [spring];
    body.spatialGrid = [];

    body._performPhysicalUpdates(1 / 60, null);

    const dx = spring.p1.pos.x - spring.p2.pos.x;
    const dy = spring.p1.pos.y - spring.p2.pos.y;
    const length = Math.hypot(dx, dy);
    const relErr = Math.abs(length - spring.restLength) / Math.max(1e-9, spring.restLength);

    assert.ok(relErr <= 0.002);
    assert.ok(Number(body.rigidConstraintProjectionCorrections || 0) > 0);
  } finally {
    Object.assign(config, backup);
  }
});
