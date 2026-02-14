import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { NodeType, MovementType } from '../../js/classes/constants.js';

function makeGpuStyleFluid({ vx = 0, vy = 0 } = {}) {
  const calls = [];
  return {
    scaleX: 1,
    scaleY: 1,
    getVelocityAtWorld() {
      return { vx, vy };
    },
    addVelocity(x, y, amountX, amountY) {
      calls.push({ x, y, amountX, amountY });
    },
    addDensity() {},
    calls
  };
}

test('GPU-style fluid coupling works without IX/Vx arrays (sensor + swimmer + floating carry)', () => {
  const cfgBackup = {
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED,
    SWIMMER_TO_FLUID_FEEDBACK: config.SWIMMER_TO_FLUID_FEEDBACK
  };

  try {
    config.DYE_ECOLOGY_ENABLED = false;
    config.SWIMMER_TO_FLUID_FEEDBACK = 0.5;

    const fluid = makeGpuStyleFluid({ vx: 2, vy: 0.5 });
    const body = new SoftBody(9101, 40, 40, null, false);
    const p = body.massPoints[0];

    p.nodeType = NodeType.SWIMMER;
    p.movementType = MovementType.FLOATING;
    p.pos.x = 40;
    p.pos.y = 40;
    p.prevPos.x = 40;
    p.prevPos.y = 40;
    p.swimmerActuation = { magnitude: 2, angle: 0 };

    body.massPoints = [p];
    body.springs = [];

    body._updateJetAndSwimmerFluidSensor(fluid);
    assert.equal(p.sensedFluidVelocity.x, 2);
    assert.equal(p.sensedFluidVelocity.y, 0.5);

    const beforeX = p.pos.x;
    body._performPhysicalUpdates(1 / 60, fluid);

    assert.equal(fluid.calls.length > 0, true, 'expected body->fluid velocity injection');
    assert.equal(p.pos.x > beforeX, true, 'floating/swimmer coupling should advance point with flow');
    assert.equal(fluid.calls.some((c) => c.amountX < 0), true, 'swimmer impulse should push back against fluid');
  } finally {
    Object.assign(config, cfgBackup);
  }
});

test('rigid-coupled point injects stronger feedback than soft-coupled point', () => {
  const cfgBackup = {
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED,
    BODY_TO_FLUID_FEEDBACK_SOFT: config.BODY_TO_FLUID_FEEDBACK_SOFT,
    BODY_TO_FLUID_FEEDBACK_RIGID: config.BODY_TO_FLUID_FEEDBACK_RIGID,
    BODY_FLUID_DRAG_COEFF_SOFT: config.BODY_FLUID_DRAG_COEFF_SOFT,
    BODY_FLUID_DRAG_COEFF_RIGID: config.BODY_FLUID_DRAG_COEFF_RIGID
  };

  try {
    config.DYE_ECOLOGY_ENABLED = false;
    config.BODY_TO_FLUID_FEEDBACK_SOFT = 0.01;
    config.BODY_TO_FLUID_FEEDBACK_RIGID = 0.09;
    config.BODY_FLUID_DRAG_COEFF_SOFT = 0;
    config.BODY_FLUID_DRAG_COEFF_RIGID = 0;

    const fluidSoft = makeGpuStyleFluid({ vx: 0, vy: 0 });
    const softBody = new SoftBody(9102, 20, 20, null, false);
    const softPoint = softBody.massPoints[0];
    softPoint.nodeType = NodeType.EATER;
    softPoint.movementType = MovementType.NEUTRAL;
    softPoint.pos.x = 20;
    softPoint.prevPos.x = 19;
    softPoint.pos.y = 20;
    softPoint.prevPos.y = 20;
    softBody.massPoints = [softPoint];
    softBody.springs = [];

    softBody._performPhysicalUpdates(1 / 60, fluidSoft);
    const softInjection = Math.abs(fluidSoft.calls[0]?.amountX || 0);

    const fluidRigid = makeGpuStyleFluid({ vx: 0, vy: 0 });
    const rigidBody = new SoftBody(9103, 60, 60, null, false);
    const rigidPoint = rigidBody.massPoints[0];
    const anchorBody = new SoftBody(9104, 62, 60, null, false);
    const anchorPoint = anchorBody.massPoints[0];

    rigidPoint.nodeType = NodeType.EATER;
    rigidPoint.movementType = MovementType.NEUTRAL;
    rigidPoint.pos.x = 60;
    rigidPoint.prevPos.x = 59;
    rigidPoint.pos.y = 60;
    rigidPoint.prevPos.y = 60;

    anchorPoint.nodeType = NodeType.EATER;
    anchorPoint.movementType = MovementType.FIXED;

    rigidBody.massPoints = [rigidPoint, anchorPoint];
    rigidBody.springs = [new Spring(rigidPoint, anchorPoint, 1, 0.1, 2, true)];

    rigidBody._performPhysicalUpdates(1 / 60, fluidRigid);
    const rigidInjection = Math.abs(fluidRigid.calls[0]?.amountX || 0);

    assert.ok(rigidInjection > softInjection * 2, `expected rigid feedback > soft (rigid=${rigidInjection}, soft=${softInjection})`);
  } finally {
    Object.assign(config, cfgBackup);
  }
});
