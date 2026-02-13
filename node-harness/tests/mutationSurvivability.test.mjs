import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { SoftBody } from '../../js/classes/SoftBody.js';
import { NodeType, MovementType } from '../../js/classes/constants.js';
import { runtimeState } from '../../js/engine/runtimeState.js';

function withMockedRandom(value, fn) {
  const originalRandom = Math.random;
  Math.random = () => value;
  try {
    return fn();
  } finally {
    Math.random = originalRandom;
  }
}

test('blueprint viability flags disconnected structure and low diversity', () => {
  const cfgBackup = {
    OFFSPRING_MIN_BLUEPRINT_POINTS: config.OFFSPRING_MIN_BLUEPRINT_POINTS,
    OFFSPRING_MIN_NODE_TYPE_DIVERSITY: config.OFFSPRING_MIN_NODE_TYPE_DIVERSITY,
    OFFSPRING_MIN_SPRING_TO_POINT_RATIO: config.OFFSPRING_MIN_SPRING_TO_POINT_RATIO,
    OFFSPRING_REQUIRE_HARVESTER_NODE: config.OFFSPRING_REQUIRE_HARVESTER_NODE,
    OFFSPRING_REQUIRE_ACTUATOR_NODE: config.OFFSPRING_REQUIRE_ACTUATOR_NODE
  };

  try {
    config.OFFSPRING_MIN_BLUEPRINT_POINTS = 3;
    config.OFFSPRING_MIN_NODE_TYPE_DIVERSITY = 2;
    config.OFFSPRING_MIN_SPRING_TO_POINT_RATIO = 1;
    config.OFFSPRING_REQUIRE_HARVESTER_NODE = false;
    config.OFFSPRING_REQUIRE_ACTUATOR_NODE = false;

    const body = new SoftBody(1101, 100, 100, null, false);
    body.blueprintPoints = [
      { relX: 0, relY: 0, radius: 1, mass: 1, nodeType: NodeType.NEURON, movementType: MovementType.NEUTRAL, dyeColor: [255, 0, 0], canBeGrabber: false },
      { relX: 3, relY: 0, radius: 1, mass: 1, nodeType: NodeType.NEURON, movementType: MovementType.NEUTRAL, dyeColor: [255, 0, 0], canBeGrabber: false },
      { relX: 6, relY: 0, radius: 1, mass: 1, nodeType: NodeType.NEURON, movementType: MovementType.NEUTRAL, dyeColor: [255, 0, 0], canBeGrabber: false }
    ];
    body.blueprintSprings = [];

    body._sanitizeBlueprintDataInPlace();
    const viability = body._evaluateBlueprintViability();

    assert.equal(viability.ok, false);
    assert.equal(viability.reasons.structure, true);
    assert.equal(viability.reasons.diversity, true);
  } finally {
    Object.assign(config, cfgBackup);
  }
});

test('donor module graft mutation can add donor blueprint module to offspring', () => {
  const cfgBackup = {
    HGT_GRAFT_DONOR_SEARCH_RADIUS: config.HGT_GRAFT_DONOR_SEARCH_RADIUS,
    HGT_GRAFT_MIN_POINTS: config.HGT_GRAFT_MIN_POINTS,
    HGT_GRAFT_MAX_POINTS: config.HGT_GRAFT_MAX_POINTS,
    HGT_GRAFT_MAX_TOTAL_POINTS: config.HGT_GRAFT_MAX_TOTAL_POINTS,
    HGT_GRAFT_ATTACHMENT_SPRINGS: config.HGT_GRAFT_ATTACHMENT_SPRINGS
  };

  const runtimeBackup = {
    softBodyPopulation: runtimeState.softBodyPopulation,
    mutationStats: runtimeState.mutationStats
  };

  try {
    config.HGT_GRAFT_DONOR_SEARCH_RADIUS = 1e9;
    config.HGT_GRAFT_MIN_POINTS = 2;
    config.HGT_GRAFT_MAX_POINTS = 2;
    config.HGT_GRAFT_MAX_TOTAL_POINTS = 200;
    config.HGT_GRAFT_ATTACHMENT_SPRINGS = 2;

    runtimeState.mutationStats = {};

    const parent = new SoftBody(1102, 200, 200, null, false);
    const donor = new SoftBody(1103, 230, 220, null, false);

    const child = new SoftBody(1104, 210, 210, null, false);
    child.blueprintPoints = JSON.parse(JSON.stringify(parent.blueprintPoints));
    child.blueprintSprings = JSON.parse(JSON.stringify(parent.blueprintSprings));

    runtimeState.softBodyPopulation = [parent, donor, child];

    const pointsBefore = child.blueprintPoints.length;
    const springsBefore = child.blueprintSprings.length;

    const applied = withMockedRandom(0.25, () => child._attemptDonorModuleGraftMutation(parent));

    assert.equal(applied, true);
    assert.ok(child.blueprintPoints.length > pointsBefore);
    assert.ok(child.blueprintSprings.length > springsBefore);
    assert.equal((runtimeState.mutationStats.hgtDonorGraftApplied || 0) >= 1, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
    runtimeState.mutationStats = runtimeBackup.mutationStats;
  }
});

test('triangle silo mutation mode preserves parent blueprint topology exactly', () => {
  const cfgBackup = {
    MUTATION_TRIANGLE_SILO_MODE: config.MUTATION_TRIANGLE_SILO_MODE,
    MUTATION_CHANCE_BOOL: config.MUTATION_CHANCE_BOOL,
    MUTATION_CHANCE_NODE_TYPE: config.MUTATION_CHANCE_NODE_TYPE,
    ADD_POINT_MUTATION_CHANCE: config.ADD_POINT_MUTATION_CHANCE,
    SPRING_SUBDIVISION_MUTATION_CHANCE: config.SPRING_SUBDIVISION_MUTATION_CHANCE,
    HGT_GRAFT_MUTATION_CHANCE: config.HGT_GRAFT_MUTATION_CHANCE
  };

  const runtimeBackup = {
    mutationStats: runtimeState.mutationStats,
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  try {
    config.MUTATION_TRIANGLE_SILO_MODE = true;
    config.MUTATION_CHANCE_BOOL = 1;
    config.MUTATION_CHANCE_NODE_TYPE = 1;
    config.ADD_POINT_MUTATION_CHANCE = 1;
    config.SPRING_SUBDIVISION_MUTATION_CHANCE = 1;
    config.HGT_GRAFT_MUTATION_CHANCE = 1;

    runtimeState.mutationStats = {};

    const parent = new SoftBody(1110, 200, 200, null, false);
    const parentBlueprintSnapshot = JSON.parse(JSON.stringify({
      points: parent.blueprintPoints,
      springs: parent.blueprintSprings
    }));

    runtimeState.softBodyPopulation = [parent];

    const child = withMockedRandom(0, () => new SoftBody(1111, 210, 210, parent, false));

    assert.equal(child.blueprintPoints.length, parentBlueprintSnapshot.points.length);
    assert.equal(child.blueprintSprings.length, parentBlueprintSnapshot.springs.length);

    for (let i = 0; i < child.blueprintPoints.length; i++) {
      const c = child.blueprintPoints[i];
      const p = parentBlueprintSnapshot.points[i];
      assert.equal(c.relX, p.relX);
      assert.equal(c.relY, p.relY);
      assert.equal(c.nodeType, p.nodeType);
      assert.equal(c.movementType, p.movementType);
    }

    for (let i = 0; i < child.blueprintSprings.length; i++) {
      const c = child.blueprintSprings[i];
      const p = parentBlueprintSnapshot.springs[i];
      assert.equal(c.p1Index, p.p1Index);
      assert.equal(c.p2Index, p.p2Index);
      assert.equal(c.restLength, p.restLength);
    }

    assert.equal((runtimeState.mutationStats.mutationTriangleSiloApplied || 0) >= 1, true);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.mutationStats = runtimeBackup.mutationStats;
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
  }
});

test('offspring falls back to parent blueprint when viability guardrails reject mutation result', () => {
  const cfgBackup = {
    OFFSPRING_MIN_NODE_TYPE_DIVERSITY: config.OFFSPRING_MIN_NODE_TYPE_DIVERSITY
  };

  const runtimeBackup = {
    mutationStats: runtimeState.mutationStats,
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  try {
    // Intentionally impossible to force fallback path during offspring creation.
    config.OFFSPRING_MIN_NODE_TYPE_DIVERSITY = 999;
    runtimeState.mutationStats = {};

    const parent = new SoftBody(1105, 300, 300, null, false);
    runtimeState.softBodyPopulation = [parent];

    const child = new SoftBody(1106, 320, 320, parent, false);

    assert.equal((runtimeState.mutationStats.offspringViabilityFallbackToParent || 0) >= 1, true);
    assert.equal(child.blueprintPoints.length, parent.blueprintPoints.length);
  } finally {
    Object.assign(config, cfgBackup);
    runtimeState.mutationStats = runtimeBackup.mutationStats;
    runtimeState.softBodyPopulation = runtimeBackup.softBodyPopulation;
  }
});
