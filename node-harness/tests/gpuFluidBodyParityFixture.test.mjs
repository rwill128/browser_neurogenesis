import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { NodeType, MovementType } from '../../js/classes/constants.js';
import { GPUFluidField } from '../../js/gpuFluidField.js';
import { createSeededRandom, withRandom } from '../seededRandomScope.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = path.resolve(__dirname, '..', 'fixtures');

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

function setupBodies({ softPos, rigidPos, anchorPos, boundaryPos, swimmerActuation = 0.2 }) {
  const softBody = new SoftBody(9301, softPos.x, softPos.y, null, false);
  const softPoint = softBody.massPoints[0];
  softPoint.nodeType = NodeType.SWIMMER;
  softPoint.movementType = MovementType.FLOATING;
  softPoint.pos.x = softPos.x;
  softPoint.prevPos.x = softPos.x;
  softPoint.pos.y = softPos.y;
  softPoint.prevPos.y = softPos.y;
  softPoint.swimmerActuation = { magnitude: swimmerActuation, angle: Math.PI };
  softBody.massPoints = [softPoint];
  softBody.springs = [];

  const rigidBody = new SoftBody(9302, rigidPos.x, rigidPos.y, null, false);
  const rigidPoint = rigidBody.massPoints[0];
  const anchorBody = new SoftBody(9303, anchorPos.x, anchorPos.y, null, false);
  const anchorPoint = anchorBody.massPoints[0];

  rigidPoint.nodeType = NodeType.SWIMMER;
  rigidPoint.movementType = MovementType.FLOATING;
  rigidPoint.pos.x = rigidPos.x;
  rigidPoint.prevPos.x = rigidPos.x;
  rigidPoint.pos.y = rigidPos.y;
  rigidPoint.prevPos.y = rigidPos.y;
  rigidPoint.swimmerActuation = { magnitude: swimmerActuation, angle: Math.PI };

  anchorPoint.movementType = MovementType.FIXED;
  anchorPoint.pos.x = anchorPos.x;
  anchorPoint.prevPos.x = anchorPos.x;
  anchorPoint.pos.y = anchorPos.y;
  anchorPoint.prevPos.y = anchorPos.y;

  rigidBody.massPoints = [rigidPoint, anchorPoint];
  rigidBody.springs = [new Spring(rigidPoint, anchorPoint, 1, 0.1, 2, true)];

  const boundaryBody = new SoftBody(9304, boundaryPos.x, boundaryPos.y, null, false);
  const boundaryPoint = boundaryBody.massPoints[0];
  boundaryPoint.nodeType = NodeType.EATER;
  boundaryPoint.movementType = MovementType.FLOATING;
  boundaryPoint.pos.x = boundaryPos.x;
  boundaryPoint.prevPos.x = boundaryPos.x;
  boundaryPoint.pos.y = boundaryPos.y;
  boundaryPoint.prevPos.y = boundaryPos.y;
  boundaryBody.massPoints = [boundaryPoint];
  boundaryBody.springs = [];

  return { softBody, softPoint, rigidBody, rigidPoint, boundaryBody, boundaryPoint };
}

function runParityScenarioV1({ dt, steps, seed }) {
  return withRandom(createSeededRandom(seed), () => {
    const fluid = makeShadowOnlyGpuFluid({ size: 64, dt, scaleX: 1, scaleY: 1 });

    fluid.addVelocity(20, 20, 1.9, 0.1, 18);
    fluid.addDensity(20, 20, 220, 120, 80, 70);
    fluid.addVelocity(2, 2, 1.3, 0.0, 14);
    fluid.addDensity(2, 2, 170, 90, 40, 60);

    const { softBody, softPoint, rigidBody, rigidPoint, boundaryBody, boundaryPoint } = setupBodies({
      softPos: { x: 20, y: 20 },
      rigidPos: { x: 20, y: 22 },
      anchorPos: { x: 22, y: 22 },
      boundaryPos: { x: 1, y: 2 }
    });

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

function runParityScenarioV2ScaledBoundary({ dt, steps, seed }) {
  return withRandom(createSeededRandom(seed), () => {
    const fluid = makeShadowOnlyGpuFluid({ size: 64, dt, scaleX: 2, scaleY: 1.5 });

    fluid.addVelocity(18, 16, 1.4, -0.3, 16);
    fluid.addDensity(18, 16, 190, 80, 30, 56);
    fluid.addVelocity(1, 2, 1.1, 0.0, 12);
    fluid.addDensity(1, 2, 210, 60, 20, 48);

    const { softBody, softPoint, rigidBody, rigidPoint, boundaryBody, boundaryPoint } = setupBodies({
      softPos: { x: 36, y: 24 },
      rigidPos: { x: 36, y: 27 },
      anchorPos: { x: 40, y: 27 },
      boundaryPos: { x: 2, y: 3 },
      swimmerActuation: 0.24
    });

    const timeline = [];

    for (let step = 0; step < steps; step++) {
      fluid.addVelocity(18, 16, 0.2, -0.05, 9);
      fluid.addDensity(18, 16, 188 - step * 7, 82, 32, 42);
      fluid.addVelocity(1, 2, 0.14, 0.0, 7);
      fluid.addDensity(1, 2, 180 - step * 8, 70, 25, 36);

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
        wakeVx38_24: round(fluid.getVelocityAtWorld(38, 24).vx),
        wakeVy38_24: round(fluid.getVelocityAtWorld(38, 24).vy),
        dyeR38_24: round(fluid.getDensityAtWorld(38, 24)[0]),
        boundaryVx0_3: round(fluid.getVelocityAtWorld(0, 3).vx),
        boundaryDyeR2_3: round(fluid.getDensityAtWorld(2, 3)[0])
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

const FIXTURE_RUNNERS = {
  'gpu-fluid-body-coupling-browser-reference-v1': runParityScenarioV1,
  'gpu-fluid-body-coupling-browser-reference-v2-scaled-boundary': runParityScenarioV2ScaledBoundary
};

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

function assertFixtureParity(fixtureFilename) {
  const fixturePath = path.resolve(FIXTURES_DIR, fixtureFilename);
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const runScenario = FIXTURE_RUNNERS[fixture.name];
  assert.ok(runScenario, `missing scenario runner for fixture ${fixture.name}`);

  const actual = runScenario({ dt: fixture.dt, steps: fixture.steps, seed: fixture.seed });

  let maxDelta = 0;
  let worstMetric = null;
  for (let i = 0; i < fixture.timeline.length; i++) {
    const expectedStep = fixture.timeline[i];
    const actualStep = actual.timeline[i];
    const metrics = Object.keys(expectedStep).filter((k) => k !== 'step');
    for (const metric of metrics) {
      const tolerance = groupTolerance(metric, fixture.tolerances);
      const delta = Math.abs(actualStep[metric] - expectedStep[metric]);
      if (delta > maxDelta) {
        maxDelta = delta;
        worstMetric = `step=${i} metric=${metric}`;
      }
      assert.ok(delta <= tolerance,
        `${fixture.name} ${worstMetric || ''} expected=${expectedStep[metric]} actual=${actualStep[metric]} delta=${delta} tol=${tolerance}`);
    }
  }

  for (const metric of Object.keys(fixture.final)) {
    const tolerance = groupTolerance(metric, fixture.tolerances);
    const delta = Math.abs(actual.final[metric] - fixture.final[metric]);
    if (delta > maxDelta) {
      maxDelta = delta;
      worstMetric = `final metric=${metric}`;
    }
    assert.ok(delta <= tolerance,
      `${fixture.name} ${worstMetric || ''} expected=${fixture.final[metric]} actual=${actual.final[metric]} delta=${delta} tol=${tolerance}`);
  }

  const last = actual.timeline.at(-1);
  assert.ok(last.rigidDrag > last.softDrag * 0.75,
    `${fixture.name}: rigid drag should remain comparable/higher than soft`);
  assert.ok(last.rigidFeedback > last.softFeedback * 1.15,
    `${fixture.name}: rigid feedback should remain stronger than soft`);

  const wakeVelocityMetric = Object.keys(last).find((k) => k.startsWith('wakeVx'));
  const boundaryDyeMetric = Object.keys(last).find((k) => k.startsWith('boundaryDyeR'));
  assert.ok(Math.abs(last[wakeVelocityMetric]) > 0.45,
    `${fixture.name}: wake velocity should remain non-trivial (${wakeVelocityMetric}=${last[wakeVelocityMetric]})`);
  assert.ok(last[boundaryDyeMetric] > 0.01,
    `${fixture.name}: boundary dye sample should remain non-zero (${boundaryDyeMetric}=${last[boundaryDyeMetric]})`);

  return { fixture, maxDelta, worstMetric };
}

test('node harness fluid+body coupling reproducibly matches browser reference fixtures', () => {
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

    const results = [
      assertFixtureParity('gpuFluidBodyCoupling.browserRef.json'),
      assertFixtureParity('gpuFluidBodyCoupling.scaledBoundary.browserRef.json')
    ];

    for (const { fixture, maxDelta } of results) {
      const maxTolerance = Math.max(
        fixture.tolerances.carry,
        fixture.tolerances.drag,
        fixture.tolerances.feedback,
        fixture.tolerances.swim,
        fixture.tolerances.wakeVelocity,
        fixture.tolerances.wakeDye,
        fixture.tolerances.boundary,
        fixture.tolerances.position
      );
      assert.ok(maxDelta <= maxTolerance, `${fixture.name} maxDelta=${maxDelta} maxTolerance=${maxTolerance}`);
    }
  } finally {
    Object.assign(config, configBackup);
  }
});
