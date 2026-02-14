import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { NodeType, MovementType } from '../../js/classes/constants.js';
import { GPUFluidField } from '../../js/gpuFluidField.js';

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

function makeShadowOnlyGpuFluid({ size = 48, dt = 1 / 60, scaleX = 1, scaleY = 1 } = {}) {
  const fluid = Object.create(GPUFluidField.prototype);
  fluid.size = size;
  fluid.dt = dt;
  fluid.scaleX = scaleX;
  fluid.scaleY = scaleY;
  fluid.useWrapping = false;
  fluid.maxVelComponent = 20;
  fluid.diffusion = 0.0001;
  fluid.viscosity = 0.0001;
  fluid.gpuEnabled = false;

  const cellCount = size * size;
  fluid.shadowVx = new Float32Array(cellCount).fill(0);
  fluid.shadowVy = new Float32Array(cellCount).fill(0);
  fluid.shadowDensityR = new Float32Array(cellCount).fill(0);
  fluid.shadowDensityG = new Float32Array(cellCount).fill(0);
  fluid.shadowDensityB = new Float32Array(cellCount).fill(0);
  fluid.shadowVxNext = new Float32Array(cellCount).fill(0);
  fluid.shadowVyNext = new Float32Array(cellCount).fill(0);
  fluid.shadowDensityRNext = new Float32Array(cellCount).fill(0);
  fluid.shadowDensityGNext = new Float32Array(cellCount).fill(0);
  fluid.shadowDensityBNext = new Float32Array(cellCount).fill(0);
  return fluid;
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
    assert.ok(body.fluidCouplingCarryDisplacement > 0, `expected carry displacement > 0, got ${body.fluidCouplingCarryDisplacement}`);
    assert.ok(body.fluidCouplingDragForce > 0, `expected drag force > 0, got ${body.fluidCouplingDragForce}`);
    assert.ok(body.fluidCouplingSwimToFluidImpulse > 0, `expected swimmer pushback > 0, got ${body.fluidCouplingSwimToFluidImpulse}`);
  } finally {
    Object.assign(config, cfgBackup);
  }
});

test('GPU-style coupling uses world-space fluid queries when scale != 1 (carry + drag + pushback)', () => {
  const cfgBackup = {
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED,
    FLUID_CURRENT_STRENGTH_ON_BODY: config.FLUID_CURRENT_STRENGTH_ON_BODY,
    BODY_FLUID_ENTRAINMENT_FACTOR: config.BODY_FLUID_ENTRAINMENT_FACTOR,
    SWIMMER_TO_FLUID_FEEDBACK: config.SWIMMER_TO_FLUID_FEEDBACK
  };

  try {
    config.DYE_ECOLOGY_ENABLED = false;
    config.FLUID_CURRENT_STRENGTH_ON_BODY = 1;
    config.BODY_FLUID_ENTRAINMENT_FACTOR = 0.35;
    config.SWIMMER_TO_FLUID_FEEDBACK = 0.4;

    const fluid = makeGpuStyleFluid({ vx: 1.5, vy: 0 });
    fluid.scaleX = 10;
    fluid.scaleY = 10;

    const body = new SoftBody(9105, 80, 40, null, false);
    const p = body.massPoints[0];
    p.nodeType = NodeType.SWIMMER;
    p.movementType = MovementType.FLOATING;
    p.pos.x = 80;
    p.prevPos.x = 80;
    p.pos.y = 40;
    p.prevPos.y = 40;
    p.swimmerActuation = { magnitude: 1.2, angle: Math.PI };

    body.massPoints = [p];
    body.springs = [];

    body._performPhysicalUpdates(1 / 60, fluid);

    assert.ok(body.fluidCouplingCarryDisplacement > 0.05,
      `expected measurable carry displacement from scaled world query, got ${body.fluidCouplingCarryDisplacement}`);
    assert.ok(body.fluidCouplingDragForce > 0.01,
      `expected drag from fluid/body relative velocity, got ${body.fluidCouplingDragForce}`);
    assert.ok(body.fluidCouplingSwimToFluidImpulse > 0.05,
      `expected active swimmer pushback into fluid, got ${body.fluidCouplingSwimToFluidImpulse}`);
    assert.equal(fluid.calls.some((c) => c.amountX > 0), true, 'expected body->fluid feedback term');
    assert.equal(fluid.calls.some((c) => c.amountX < 0), true, 'expected swimmer opposite-direction pushback');
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
    config.BODY_FLUID_DRAG_COEFF_SOFT = 0.2;
    config.BODY_FLUID_DRAG_COEFF_RIGID = 1.0;

    const fluidSoft = makeGpuStyleFluid({ vx: 0.4, vy: 0 });
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

    const fluidRigid = makeGpuStyleFluid({ vx: 0.4, vy: 0 });
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
    assert.ok(rigidBody.fluidCouplingRigidFeedbackImpulse > softBody.fluidCouplingSoftFeedbackImpulse * 2,
      `expected rigid-weighted telemetry > soft-weighted telemetry (rigid=${rigidBody.fluidCouplingRigidFeedbackImpulse}, soft=${softBody.fluidCouplingSoftFeedbackImpulse})`);
    assert.ok(rigidBody.fluidCouplingRigidDragForce > softBody.fluidCouplingSoftDragForce * 2,
      `expected rigid drag telemetry > soft drag telemetry (rigid=${rigidBody.fluidCouplingRigidDragForce}, soft=${softBody.fluidCouplingSoftDragForce})`);
  } finally {
    Object.assign(config, cfgBackup);
  }
});

test('GPU shadow fluid path shows two-way coupling for soft vs rigid bodies with swimmer pushback transport', () => {
  const cfgBackup = {
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED,
    FLUID_CURRENT_STRENGTH_ON_BODY: config.FLUID_CURRENT_STRENGTH_ON_BODY,
    BODY_FLUID_ENTRAINMENT_FACTOR: config.BODY_FLUID_ENTRAINMENT_FACTOR,
    BODY_TO_FLUID_FEEDBACK_SOFT: config.BODY_TO_FLUID_FEEDBACK_SOFT,
    BODY_TO_FLUID_FEEDBACK_RIGID: config.BODY_TO_FLUID_FEEDBACK_RIGID,
    BODY_FLUID_DRAG_COEFF_SOFT: config.BODY_FLUID_DRAG_COEFF_SOFT,
    BODY_FLUID_DRAG_COEFF_RIGID: config.BODY_FLUID_DRAG_COEFF_RIGID,
    SWIMMER_TO_FLUID_FEEDBACK: config.SWIMMER_TO_FLUID_FEEDBACK
  };

  try {
    config.DYE_ECOLOGY_ENABLED = false;
    config.FLUID_CURRENT_STRENGTH_ON_BODY = 1.0;
    config.BODY_FLUID_ENTRAINMENT_FACTOR = 0.45;
    config.BODY_TO_FLUID_FEEDBACK_SOFT = 0.02;
    config.BODY_TO_FLUID_FEEDBACK_RIGID = 0.12;
    config.BODY_FLUID_DRAG_COEFF_SOFT = 0.35;
    config.BODY_FLUID_DRAG_COEFF_RIGID = 1.2;
    config.SWIMMER_TO_FLUID_FEEDBACK = 0.55;

    const dt = 1 / 30;

    const fluidSoft = makeShadowOnlyGpuFluid({ size: 48, dt, scaleX: 1, scaleY: 1 });
    fluidSoft.addVelocity(20, 20, 1.8, 0, 18);

    const softBody = new SoftBody(9201, 20, 20, null, false);
    const softPoint = softBody.massPoints[0];
    softPoint.nodeType = NodeType.SWIMMER;
    softPoint.movementType = MovementType.FLOATING;
    softPoint.pos.x = 20;
    softPoint.prevPos.x = 20;
    softPoint.pos.y = 20;
    softPoint.prevPos.y = 20;
    softPoint.swimmerActuation = { magnitude: 0.25, angle: Math.PI };
    softBody.massPoints = [softPoint];
    softBody.springs = [];

    softBody._performPhysicalUpdates(dt, fluidSoft);
    fluidSoft.step();

    const softPushbackNeighbor = Math.abs(fluidSoft.getVelocityAtWorld(21, 20).vx);

    const fluidRigid = makeShadowOnlyGpuFluid({ size: 48, dt, scaleX: 1, scaleY: 1 });
    fluidRigid.addVelocity(20, 20, 1.8, 0, 18);

    const rigidBody = new SoftBody(9202, 20, 20, null, false);
    const rigidPoint = rigidBody.massPoints[0];
    const anchorBody = new SoftBody(9203, 22, 20, null, false);
    const anchorPoint = anchorBody.massPoints[0];

    rigidPoint.nodeType = NodeType.SWIMMER;
    rigidPoint.movementType = MovementType.FLOATING;
    rigidPoint.pos.x = 20;
    rigidPoint.prevPos.x = 20;
    rigidPoint.pos.y = 20;
    rigidPoint.prevPos.y = 20;
    rigidPoint.swimmerActuation = { magnitude: 0.25, angle: Math.PI };

    anchorPoint.movementType = MovementType.FIXED;

    rigidBody.massPoints = [rigidPoint, anchorPoint];
    rigidBody.springs = [new Spring(rigidPoint, anchorPoint, 1, 0.1, 2, true)];

    rigidBody._performPhysicalUpdates(dt, fluidRigid);
    fluidRigid.step();

    const rigidPushbackNeighbor = Math.abs(fluidRigid.getVelocityAtWorld(21, 20).vx);

    assert.ok(softBody.fluidCouplingCarryDisplacement > 0.01,
      `expected measurable soft carry displacement, got ${softBody.fluidCouplingCarryDisplacement}`);
    assert.ok(rigidBody.fluidCouplingCarryDisplacement > 0.01,
      `expected measurable rigid carry displacement, got ${rigidBody.fluidCouplingCarryDisplacement}`);
    assert.ok(rigidBody.fluidCouplingRigidDragForce > softBody.fluidCouplingSoftDragForce,
      `expected stronger rigid drag coupling (rigid=${rigidBody.fluidCouplingRigidDragForce}, soft=${softBody.fluidCouplingSoftDragForce})`);
    assert.ok(rigidBody.fluidCouplingRigidFeedbackImpulse > softBody.fluidCouplingSoftFeedbackImpulse,
      `expected stronger rigid body->fluid feedback (rigid=${rigidBody.fluidCouplingRigidFeedbackImpulse}, soft=${softBody.fluidCouplingSoftFeedbackImpulse})`);
    assert.ok(softBody.fluidCouplingSwimToFluidImpulse > 0.05,
      `expected soft swimmer pushback > 0, got ${softBody.fluidCouplingSwimToFluidImpulse}`);
    assert.ok(rigidBody.fluidCouplingSwimToFluidImpulse > 0.05,
      `expected rigid swimmer pushback > 0, got ${rigidBody.fluidCouplingSwimToFluidImpulse}`);
    assert.ok(softPushbackNeighbor > 1e-4,
      `expected soft pushback transport into neighboring cell, got ${softPushbackNeighbor}`);
    assert.ok(rigidPushbackNeighbor > 1e-4,
      `expected rigid pushback transport into neighboring cell, got ${rigidPushbackNeighbor}`);
  } finally {
    Object.assign(config, cfgBackup);
  }
});
