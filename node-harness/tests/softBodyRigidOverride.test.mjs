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
    RIGID_SPRING_DAMPING: config.RIGID_SPRING_DAMPING
  };

  try {
    config.FORCE_ALL_SPRINGS_RIGID = true;
    config.RIGID_SPRING_STIFFNESS = 43210;
    config.RIGID_SPRING_DAMPING = 98;

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
