import test from 'node:test';
import assert from 'node:assert/strict';

import { SoftBody } from '../../js/classes/SoftBody.js';
import { Spring } from '../../js/classes/Spring.js';
import { NodeType, MovementType } from '../../js/classes/constants.js';
import config from '../../js/config.js';

test('blueprint sanitize enforces rigid springs and neutral/photosynthetic neighbors for photosynth nodes', () => {
  const cfgBackup = {
    PHOTOSYNTH_FORCE_RIGID_CONNECTED_SPRINGS: config.PHOTOSYNTH_FORCE_RIGID_CONNECTED_SPRINGS,
    PHOTOSYNTH_NEUTRALIZE_NON_PHOTOSYNTH_NEIGHBORS: config.PHOTOSYNTH_NEUTRALIZE_NON_PHOTOSYNTH_NEIGHBORS
  };

  try {
    config.PHOTOSYNTH_FORCE_RIGID_CONNECTED_SPRINGS = true;
    config.PHOTOSYNTH_NEUTRALIZE_NON_PHOTOSYNTH_NEIGHBORS = true;

    const body = new SoftBody(9001, 100, 100, null, false);

    body.blueprintPoints = [
      {
        relX: 0,
        relY: 0,
        radius: 3,
        mass: 0.5,
        nodeType: NodeType.PHOTOSYNTHETIC,
        movementType: MovementType.NEUTRAL,
        dyeColor: [0, 255, 0],
        canBeGrabber: false,
        neuronDataBlueprint: null,
        activationIntervalGene: 2
      },
      {
        relX: 6,
        relY: 0,
        radius: 3,
        mass: 0.5,
        nodeType: NodeType.PREDATOR,
        movementType: MovementType.FIXED,
        dyeColor: [255, 0, 0],
        canBeGrabber: false,
        neuronDataBlueprint: null,
        activationIntervalGene: 2
      }
    ];

    body.blueprintSprings = [
      {
        p1Index: 0,
        p2Index: 1,
        restLength: 6,
        isRigid: false,
        stiffness: 500,
        damping: 5,
        activationIntervalGene: 2
      }
    ];

    body._sanitizeBlueprintDataInPlace();

    assert.equal(body.blueprintSprings.length, 1);
    assert.equal(body.blueprintSprings[0].isRigid, true);
    assert.equal(body.blueprintPoints[1].movementType, MovementType.NEUTRAL);
  } finally {
    Object.assign(config, cfgBackup);
  }
});

test('phenotype enforcement rigidifies photosynth-linked springs and neutralizes non-photosynth neighbors', () => {
  const cfgBackup = {
    PHOTOSYNTH_FORCE_RIGID_CONNECTED_SPRINGS: config.PHOTOSYNTH_FORCE_RIGID_CONNECTED_SPRINGS,
    PHOTOSYNTH_NEUTRALIZE_NON_PHOTOSYNTH_NEIGHBORS: config.PHOTOSYNTH_NEUTRALIZE_NON_PHOTOSYNTH_NEIGHBORS
  };

  try {
    config.PHOTOSYNTH_FORCE_RIGID_CONNECTED_SPRINGS = true;
    config.PHOTOSYNTH_NEUTRALIZE_NON_PHOTOSYNTH_NEIGHBORS = true;

    const body = new SoftBody(9002, 80, 80, null, false);

    const pPhoto = body.massPoints[0];
    pPhoto.nodeType = NodeType.PHOTOSYNTHETIC;
    pPhoto.movementType = MovementType.NEUTRAL;

    const pNeighbor = new SoftBody(9003, 86, 80, null, false).massPoints[0];
    pNeighbor.nodeType = NodeType.PREDATOR;
    pNeighbor.movementType = MovementType.FIXED;

    body.massPoints = [pPhoto, pNeighbor];
    const spring = new Spring(pPhoto, pNeighbor, 600, 6, 6, false);
    body.springs = [spring];

    body._enforcePhotosyntheticPhenotypeConstraints();

    assert.equal(body.springs[0].isRigid, true);
    assert.equal(body.springs[0].stiffness, config.RIGID_SPRING_STIFFNESS);
    assert.equal(body.springs[0].dampingFactor, config.RIGID_SPRING_DAMPING);
    assert.equal(pNeighbor.movementType, MovementType.NEUTRAL);
  } finally {
    Object.assign(config, cfgBackup);
  }
});
