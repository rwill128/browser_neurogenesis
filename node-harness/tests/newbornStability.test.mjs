import test from 'node:test';
import assert from 'node:assert/strict';

import {
  fitBodyInsideWorld,
  computeNewbornSpringCaps,
  clampNewbornSpringParameters,
  stabilizeNewbornBody
} from '../../js/engine/newbornStability.mjs';

function makeBody(points, springs = []) {
  return {
    massPoints: points,
    springs,
    _updateBlueprintRadiusFromCurrentPhenotype() {
      this._radiusUpdated = true;
    }
  };
}

function point(x, y, radius = 2) {
  return {
    pos: { x, y },
    prevPos: { x, y },
    radius
  };
}

test('fitBodyInsideWorld translates newborn fully into bounds', () => {
  const body = makeBody([
    point(-10, 10, 3),
    point(8, 20, 3)
  ]);

  const result = fitBodyInsideWorld(body, { WORLD_WIDTH: 100, WORLD_HEIGHT: 100 }, { padding: 1 });

  assert.equal(result.adjusted, true);
  assert.equal(result.scaled, false);
  assert.equal(result.translated, true);
  assert.equal(body._radiusUpdated, true);

  for (const p of body.massPoints) {
    assert.ok(p.pos.x >= p.radius);
    assert.ok(p.pos.x <= 100 - p.radius);
    assert.ok(p.pos.y >= p.radius);
    assert.ok(p.pos.y <= 100 - p.radius);
    assert.equal(p.prevPos.x, p.pos.x);
    assert.equal(p.prevPos.y, p.pos.y);
  }
});

test('fitBodyInsideWorld scales oversized newborn and spring rest lengths', () => {
  const springs = [{ isRigid: true, stiffness: 500000, dampingFactor: 150, restLength: 200 }];
  const body = makeBody([
    point(0, 0, 2),
    point(200, 0, 2)
  ], springs);

  const result = fitBodyInsideWorld(body, { WORLD_WIDTH: 100, WORLD_HEIGHT: 100 }, { padding: 1 });

  assert.equal(result.scaled, true);
  assert.ok(result.scaleApplied < 1);
  assert.ok(springs[0].restLength < 200);

  const minX = Math.min(...body.massPoints.map((p) => p.pos.x - p.radius));
  const maxX = Math.max(...body.massPoints.map((p) => p.pos.x + p.radius));
  assert.ok(minX >= 0);
  assert.ok(maxX <= 100);
});

test('computeNewbornSpringCaps shrinks caps in tiny worlds', () => {
  const config = {
    WORLD_WIDTH: 100,
    WORLD_HEIGHT: 100,
    RIGID_SPRING_STIFFNESS: 500000,
    RIGID_SPRING_DAMPING: 150,
    NEWBORN_NON_RIGID_STIFFNESS_BASE_CAP: 10000,
    NEWBORN_NON_RIGID_DAMPING_BASE_CAP: 80,
    NEWBORN_STIFFNESS_WORLD_REF_DIM: 1200,
    NEWBORN_STIFFNESS_DT_REF: 1 / 30,
    NEWBORN_STIFFNESS_DT_EXPONENT: 2,
    NEWBORN_RIGID_STIFFNESS_WORLD_EXPONENT: 2,
    NEWBORN_NON_RIGID_STIFFNESS_WORLD_EXPONENT: 1,
    NEWBORN_RIGID_STIFFNESS_MIN_SCALE: 0.005,
    NEWBORN_NON_RIGID_STIFFNESS_MIN_SCALE: 0.05
  };

  const tiny = computeNewbornSpringCaps(config, 1 / 30);
  const large = computeNewbornSpringCaps({ ...config, WORLD_WIDTH: 12000, WORLD_HEIGHT: 8000 }, 1 / 30);

  assert.ok(tiny.rigidStiffnessCap < large.rigidStiffnessCap);
  assert.ok(tiny.nonRigidStiffnessCap < large.nonRigidStiffnessCap);
});

test('stabilizeNewbornBody clamps newborn spring extremes', () => {
  const body = makeBody(
    [point(10, 10, 2), point(14, 12, 2)],
    [
      { isRigid: true, stiffness: 500000, dampingFactor: 150, restLength: 8 },
      { isRigid: false, stiffness: 9000, dampingFactor: 120, restLength: 6 }
    ]
  );

  const config = {
    WORLD_WIDTH: 100,
    WORLD_HEIGHT: 100,
    RIGID_SPRING_STIFFNESS: 500000,
    RIGID_SPRING_DAMPING: 150,
    NEWBORN_STIFFNESS_CLAMP_ENABLED: true,
    NEWBORN_NON_RIGID_STIFFNESS_BASE_CAP: 10000,
    NEWBORN_NON_RIGID_DAMPING_BASE_CAP: 80,
    NEWBORN_STIFFNESS_WORLD_REF_DIM: 1200,
    NEWBORN_STIFFNESS_DT_REF: 1 / 30,
    NEWBORN_STIFFNESS_DT_EXPONENT: 2,
    NEWBORN_RIGID_STIFFNESS_WORLD_EXPONENT: 2,
    NEWBORN_NON_RIGID_STIFFNESS_WORLD_EXPONENT: 1,
    NEWBORN_RIGID_STIFFNESS_MIN_SCALE: 0.005,
    NEWBORN_NON_RIGID_STIFFNESS_MIN_SCALE: 0.05
  };

  const result = stabilizeNewbornBody(body, { config, dt: 1 / 30, fitPadding: 0.5 });

  assert.equal(result.adjusted, true);
  assert.ok(result.springClamp.clampedStiffness >= 1);
  assert.ok(result.springClamp.clampedDamping >= 1);

  assert.ok(body.springs[0].stiffness <= result.springClamp.caps.rigidStiffnessCap + 1e-9);
  assert.ok(body.springs[1].stiffness <= result.springClamp.caps.nonRigidStiffnessCap + 1e-9);
});
