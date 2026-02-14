import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { NodeType, MovementType } from '../../js/classes/constants.js';
import { GPUFluidField } from '../../js/gpuFluidField.js';
import { createSeededRandom, withRandom } from '../seededRandomScope.mjs';

function makeShadowOnlyGpuFluid({ size = 64, dt = 1 / 60, scaleX = 1, scaleY = 1 } = {}) {
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
  fluid._initShadowBackCompatViews();
  return fluid;
}

function round(v, digits = 6) {
  const p = 10 ** digits;
  return Math.round((Number(v) || 0) * p) / p;
}

function runBrowserParityScenario({ dt, steps, seed }) {
  return withRandom(createSeededRandom(seed), () => {
    const fluid = makeShadowOnlyGpuFluid({ size: 64, dt, scaleX: 1, scaleY: 1 });

    fluid.addVelocity(20, 20, 1.9, 0.1, 18);
    fluid.addDensity(20, 20, 220, 120, 80, 70);
    fluid.addVelocity(2, 2, 1.3, 0.0, 14);
    fluid.addDensity(2, 2, 170, 90, 40, 60);

    const softBody = new SoftBody(9301, 20, 20, null, false);
  const softPoint = softBody.massPoints[0];
  softPoint.nodeType = NodeType.SWIMMER;
  softPoint.movementType = MovementType.FLOATING;
  softPoint.pos.x = 20;
  softPoint.prevPos.x = 20;
  softPoint.pos.y = 20;
  softPoint.prevPos.y = 20;
  softPoint.swimmerActuation = { magnitude: 0.2, angle: Math.PI };
  softBody.massPoints = [softPoint];
  softBody.springs = [];

  const rigidBody = new SoftBody(9302, 20, 22, null, false);
  const rigidPoint = rigidBody.massPoints[0];
  const anchorBody = new SoftBody(9303, 22, 22, null, false);
  const anchorPoint = anchorBody.massPoints[0];

  rigidPoint.nodeType = NodeType.SWIMMER;
  rigidPoint.movementType = MovementType.FLOATING;
  rigidPoint.pos.x = 20;
  rigidPoint.prevPos.x = 20;
  rigidPoint.pos.y = 22;
  rigidPoint.prevPos.y = 22;
  rigidPoint.swimmerActuation = { magnitude: 0.2, angle: Math.PI };

  anchorPoint.movementType = MovementType.FIXED;
  anchorPoint.pos.x = 22;
  anchorPoint.prevPos.x = 22;
  anchorPoint.pos.y = 22;
  anchorPoint.prevPos.y = 22;

  rigidBody.massPoints = [rigidPoint, anchorPoint];
  rigidBody.springs = [new Spring(rigidPoint, anchorPoint, 1, 0.1, 2, true)];

  const boundaryBody = new SoftBody(9304, 1, 2, null, false);
  const boundaryPoint = boundaryBody.massPoints[0];
  boundaryPoint.nodeType = NodeType.EATER;
  boundaryPoint.movementType = MovementType.FLOATING;
  boundaryPoint.pos.x = 1;
  boundaryPoint.prevPos.x = 1;
  boundaryPoint.pos.y = 2;
  boundaryPoint.prevPos.y = 2;
  boundaryBody.massPoints = [boundaryPoint];
  boundaryBody.springs = [];

  const timeline = [];

  for (let step = 0; step < steps; step++) {
    fluid.addVelocity(20, 20, 0.18, -0.02, 10);
    fluid.addDensity(20, 20, 200 - step * 8, 110, 70, 48);
    fluid.addVelocity(2, 2, 0.12, 0.0, 8);
    fluid.addDensity(2, 2, 150 - step * 6, 80, 35, 40);

    softBody._performPhysicalUpdates(dt, fluid);
    rigidBody._performPhysicalUpdates(dt, fluid);
    boundaryBody._performPhysicalUpdates(dt, fluid);

    fluid.step();

    timeline.push({
      step,
      softCarry: round(softBody.fluidCouplingCarryDisplacement),
      rigidCarry: round(rigidBody.fluidCouplingCarryDisplacement),
      softDrag: round(softBody.fluidCouplingSoftDragForce),
      rigidDrag: round(rigidBody.fluidCouplingRigidDragForce),
      softFeedback: round(softBody.fluidCouplingSoftFeedbackImpulse),
      rigidFeedback: round(rigidBody.fluidCouplingRigidFeedbackImpulse),
      softSwim: round(softBody.fluidCouplingSwimToFluidImpulse),
      rigidSwim: round(rigidBody.fluidCouplingSwimToFluidImpulse),
      wakeVx21_20: round(fluid.getVelocityAtWorld(21, 20).vx),
      wakeVy21_20: round(fluid.getVelocityAtWorld(21, 20).vy),
      dyeR21_20: round(fluid.getDensityAtWorld(21, 20)[0]),
      boundaryVx0_2: round(fluid.getVelocityAtWorld(0, 2).vx),
      boundaryDyeR0_2: round(fluid.getDensityAtWorld(0, 2)[0])
    });
  }

    return {
      timeline,
      final: {
        softPosX: round(softPoint.pos.x),
        softPosY: round(softPoint.pos.y),
        rigidPosX: round(rigidPoint.pos.x),
        rigidPosY: round(rigidPoint.pos.y),
        boundaryPosX: round(boundaryPoint.pos.x),
        boundaryPosY: round(boundaryPoint.pos.y)
      }
    };
  });
}

function groupTolerance(metric, tolerances) {
  if (metric.includes('Carry')) return tolerances.carry;
  if (metric.includes('Drag')) return tolerances.drag;
  if (metric.includes('Feedback')) return tolerances.feedback;
  if (metric.includes('Swim')) return tolerances.swim;
  if (metric.startsWith('wakeV')) return tolerances.wakeVelocity;
  if (metric.startsWith('dye')) return tolerances.wakeDye;
  if (metric.startsWith('boundary')) return tolerances.boundary;
  return tolerances.position;
}

test('node harness fluid+body coupling reproducibly matches browser reference fixture', () => {
  const fixturePath = path.resolve(process.cwd(), 'node-harness/fixtures/gpuFluidBodyCoupling.browserRef.json');
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));

  const configBackup = {
    DYE_ECOLOGY_ENABLED: config.DYE_ECOLOGY_ENABLED,
    FLUID_CURRENT_STRENGTH_ON_BODY: config.FLUID_CURRENT_STRENGTH_ON_BODY,
    BODY_FLUID_ENTRAINMENT_FACTOR: config.BODY_FLUID_ENTRAINMENT_FACTOR,
    BODY_TO_FLUID_FEEDBACK_SOFT: config.BODY_TO_FLUID_FEEDBACK_SOFT,
    BODY_TO_FLUID_FEEDBACK_RIGID: config.BODY_TO_FLUID_FEEDBACK_RIGID,
    BODY_FLUID_DRAG_COEFF_SOFT: config.BODY_FLUID_DRAG_COEFF_SOFT,
    BODY_FLUID_DRAG_COEFF_RIGID: config.BODY_FLUID_DRAG_COEFF_RIGID,
    SWIMMER_TO_FLUID_FEEDBACK: config.SWIMMER_TO_FLUID_FEEDBACK,
    BODY_FLUID_CARRY_NEUTRAL_FACTOR: config.BODY_FLUID_CARRY_NEUTRAL_FACTOR,
    BODY_FLUID_CARRY_RIGID_BOOST: config.BODY_FLUID_CARRY_RIGID_BOOST
  };

  try {
    config.DYE_ECOLOGY_ENABLED = false;
    config.FLUID_CURRENT_STRENGTH_ON_BODY = 1.0;
    config.BODY_FLUID_ENTRAINMENT_FACTOR = 0.5;
    config.BODY_TO_FLUID_FEEDBACK_SOFT = 0.025;
    config.BODY_TO_FLUID_FEEDBACK_RIGID = 0.13;
    config.BODY_FLUID_DRAG_COEFF_SOFT = 0.4;
    config.BODY_FLUID_DRAG_COEFF_RIGID = 1.3;
    config.SWIMMER_TO_FLUID_FEEDBACK = 0.6;
    config.BODY_FLUID_CARRY_NEUTRAL_FACTOR = 0.5;
    config.BODY_FLUID_CARRY_RIGID_BOOST = 0.85;

    const actual = runBrowserParityScenario({ dt: fixture.dt, steps: fixture.steps, seed: fixture.seed });

    let maxDelta = 0;
    for (let i = 0; i < fixture.timeline.length; i++) {
      const expectedStep = fixture.timeline[i];
      const actualStep = actual.timeline[i];
      const metrics = Object.keys(expectedStep).filter((k) => k !== 'step');
      for (const metric of metrics) {
        const tolerance = groupTolerance(metric, fixture.tolerances);
        const delta = Math.abs(actualStep[metric] - expectedStep[metric]);
        maxDelta = Math.max(maxDelta, delta);
        assert.ok(delta <= tolerance,
          `step=${i} metric=${metric} expected=${expectedStep[metric]} actual=${actualStep[metric]} delta=${delta} tol=${tolerance}`);
      }
    }

    for (const metric of Object.keys(fixture.final)) {
      const tolerance = groupTolerance(metric, fixture.tolerances);
      const delta = Math.abs(actual.final[metric] - fixture.final[metric]);
      maxDelta = Math.max(maxDelta, delta);
      assert.ok(delta <= tolerance,
        `final metric=${metric} expected=${fixture.final[metric]} actual=${actual.final[metric]} delta=${delta} tol=${tolerance}`);
    }

    // Coupling sanity checks in parity fixture (rigid-vs-soft + wake + boundary)
    const last = actual.timeline.at(-1);
    assert.ok(last.rigidDrag > last.softDrag * 0.8, 'rigid drag should remain comparable/higher than soft under rigid coupling');
    assert.ok(last.rigidFeedback > last.softFeedback * 1.2, 'rigid feedback should remain stronger than soft feedback');
    assert.ok(Math.abs(last.wakeVx21_20) > 0.5, 'wake velocity should remain non-trivial at local downstream sample');
    assert.ok(last.boundaryDyeR0_2 > 0.01, 'boundary dye sample should remain non-zero from near-wall transport');

    assert.ok(maxDelta <= Math.max(
      fixture.tolerances.carry,
      fixture.tolerances.drag,
      fixture.tolerances.feedback,
      fixture.tolerances.swim,
      fixture.tolerances.wakeVelocity,
      fixture.tolerances.wakeDye,
      fixture.tolerances.boundary,
      fixture.tolerances.position
    ));
  } finally {
    Object.assign(config, configBackup);
  }
});
