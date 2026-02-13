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

test('triangle mutation path extrudes one outward triangle from a boundary edge', () => {
  const cfgBackup = {
    TRIANGLE_EXTRUSION_MUTATION_CHANCE_MULTIPLIER: config.TRIANGLE_EXTRUSION_MUTATION_CHANCE_MULTIPLIER,
    OFFSPRING_MIN_NODE_TYPE_DIVERSITY: config.OFFSPRING_MIN_NODE_TYPE_DIVERSITY,
    OFFSPRING_REQUIRE_HARVESTER_NODE: config.OFFSPRING_REQUIRE_HARVESTER_NODE,
    OFFSPRING_REQUIRE_ACTUATOR_NODE: config.OFFSPRING_REQUIRE_ACTUATOR_NODE
  };

  const runtimeBackup = {
    mutationStats: runtimeState.mutationStats,
    softBodyPopulation: runtimeState.softBodyPopulation
  };

  const edgeKey = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);

  try {
    config.TRIANGLE_EXTRUSION_MUTATION_CHANCE_MULTIPLIER = 1;
    config.OFFSPRING_MIN_NODE_TYPE_DIVERSITY = 1;
    config.OFFSPRING_REQUIRE_HARVESTER_NODE = false;
    config.OFFSPRING_REQUIRE_ACTUATOR_NODE = false;

    runtimeState.mutationStats = {};

    const parent = new SoftBody(1110, 200, 200, null, false);
    parent.blueprintPoints = [
      { relX: 0, relY: 0, radius: 1, mass: 1, nodeType: NodeType.EATER, movementType: MovementType.NEUTRAL, dyeColor: [255, 0, 0], canBeGrabber: false },
      { relX: 4, relY: 0, radius: 1, mass: 1, nodeType: NodeType.SWIMMER, movementType: MovementType.NEUTRAL, dyeColor: [0, 255, 0], canBeGrabber: false },
      { relX: 2, relY: 3.4641016151, radius: 1, mass: 1, nodeType: NodeType.NEURON, movementType: MovementType.NEUTRAL, dyeColor: [0, 0, 255], canBeGrabber: false, neuronDataBlueprint: { hiddenLayerSize: 8 } }
    ];
    parent.blueprintSprings = [
      { p1Index: 0, p2Index: 1, restLength: 4, isRigid: false, stiffness: 800, damping: 8, activationIntervalGene: 2 },
      { p1Index: 1, p2Index: 2, restLength: 4, isRigid: false, stiffness: 800, damping: 8, activationIntervalGene: 2 },
      { p1Index: 2, p2Index: 0, restLength: 4, isRigid: false, stiffness: 800, damping: 8, activationIntervalGene: 2 }
    ];
    parent._sanitizeBlueprintDataInPlace();
    parent.pointAddChance = 1;

    const parentEdgeSet = new Set(parent.blueprintSprings.map((s) => edgeKey(s.p1Index, s.p2Index)));

    runtimeState.softBodyPopulation = [parent];

    const child = withMockedRandom(0, () => new SoftBody(1111, 210, 210, parent, false));

    assert.equal(child.blueprintPoints.length, parent.blueprintPoints.length + 1);
    assert.equal(child.blueprintSprings.length, parent.blueprintSprings.length + 2);

    const childEdgeSet = new Set(child.blueprintSprings.map((s) => edgeKey(s.p1Index, s.p2Index)));
    for (const key of parentEdgeSet) {
      assert.equal(childEdgeSet.has(key), true);
    }

    const newPointIndex = child.blueprintPoints.length - 1;
    const newPointEdges = child.blueprintSprings.filter((s) => s.p1Index === newPointIndex || s.p2Index === newPointIndex);
    assert.equal(newPointEdges.length, 2);

    const endpointA = newPointEdges[0].p1Index === newPointIndex ? newPointEdges[0].p2Index : newPointEdges[0].p1Index;
    const endpointB = newPointEdges[1].p1Index === newPointIndex ? newPointEdges[1].p2Index : newPointEdges[1].p1Index;
    assert.notEqual(endpointA, endpointB);
    assert.equal(parentEdgeSet.has(edgeKey(endpointA, endpointB)), true);

    assert.equal(newPointEdges[0].restLength, newPointEdges[1].restLength);

    const thirdIdx = [0, 1, 2].find((idx) => idx !== endpointA && idx !== endpointB);
    assert.notEqual(thirdIdx, undefined);

    const pa = parent.blueprintPoints[endpointA];
    const pb = parent.blueprintPoints[endpointB];
    const pt = parent.blueprintPoints[thirdIdx];
    const pn = child.blueprintPoints[newPointIndex];

    const midX = (pa.relX + pb.relX) * 0.5;
    const midY = (pa.relY + pb.relY) * 0.5;
    const nx = -(pb.relY - pa.relY);
    const ny = (pb.relX - pa.relX);
    const sideThird = nx * (pt.relX - midX) + ny * (pt.relY - midY);
    const sideNew = nx * (pn.relX - midX) + ny * (pn.relY - midY);
    assert.ok(sideThird * sideNew < 0);

    assert.equal((runtimeState.mutationStats.mutationTriangleSiloApplied || 0) >= 1, true);
    assert.equal((runtimeState.mutationStats.triangleExtrusionApplied || 0) >= 1, true);
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
