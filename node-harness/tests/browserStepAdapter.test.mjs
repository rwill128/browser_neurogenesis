import test from 'node:test';
import assert from 'node:assert/strict';

import { runBrowserStepAdapter } from '../../js/engine/browserStepAdapter.mjs';

function createFakeBody({ id = 1, centerX = 20, centerY = 20, width = 10, height = 10 } = {}) {
  return {
    id,
    isUnstable: false,
    massPoints: [{ id: 1 }],
    getAveragePosition() {
      return { x: centerX, y: centerY };
    },
    getBoundingBox() {
      return {
        minX: centerX - width / 2,
        maxX: centerX + width / 2,
        minY: centerY - height / 2,
        maxY: centerY + height / 2
      };
    }
  };
}

test('auto-follow camera centers small worlds using shared clamp math', () => {
  const body = createFakeBody();
  const worldState = {
    softBodyPopulation: [body]
  };

  const config = {
    AUTO_FOLLOW_CREATURE: true,
    AUTO_FOLLOW_ZOOM_MIN: 0.35,
    AUTO_FOLLOW_ZOOM_MAX: 2.5,
    WORLD_WIDTH: 100,
    WORLD_HEIGHT: 100,
    selectedInspectBody: null,
    selectedInspectPoint: null,
    selectedInspectPointIndex: -1,
    viewZoom: 1,
    viewOffsetX: 0,
    viewOffsetY: 0
  };

  const viewport = { zoom: 1, offsetX: 0, offsetY: 0 };
  const canvas = { clientWidth: 400, clientHeight: 300 };

  let updateInfoPanelCalls = 0;
  let updateInstabilityIndicatorCalls = 0;
  let updatePopulationCountCalls = 0;
  let stepWorldCalls = 0;

  const stepResult = runBrowserStepAdapter({
    worldState,
    dt: 1 / 60,
    config,
    stepWorld: () => {
      stepWorldCalls += 1;
      return { ok: true };
    },
    stepOptions: {},
    viewport,
    canvas,
    updateInfoPanel: () => { updateInfoPanelCalls += 1; },
    updateInstabilityIndicator: () => { updateInstabilityIndicatorCalls += 1; },
    updatePopulationCount: () => { updatePopulationCountCalls += 1; }
  });

  assert.deepEqual(stepResult, { ok: true });
  assert.equal(stepWorldCalls, 1);
  assert.equal(updateInfoPanelCalls, 0);
  assert.equal(updateInstabilityIndicatorCalls, 1);
  assert.equal(updatePopulationCountCalls, 1);

  // desiredZoom = min(2.5, max(0.35, min(400,300)/(40*6))) = 1.25
  assert.equal(viewport.zoom, 1.25);
  assert.equal(config.viewZoom, 1.25);

  // Viewport in world units is 320x240, so a 100x100 world is centered at negative offsets.
  assert.equal(viewport.offsetX, -110);
  assert.equal(viewport.offsetY, -70);
  assert.equal(config.viewOffsetX, -110);
  assert.equal(config.viewOffsetY, -70);

  assert.equal(config.selectedInspectBody, body);
  assert.equal(config.selectedInspectPointIndex, 0);
  assert.equal(config.selectedInspectPoint, body.massPoints[0]);
});

test('auto-follow disabled leaves camera unchanged', () => {
  const body = createFakeBody({ centerX: 55, centerY: 60 });
  const worldState = {
    softBodyPopulation: [body]
  };

  const config = {
    AUTO_FOLLOW_CREATURE: false,
    AUTO_FOLLOW_ZOOM_MIN: 0.35,
    AUTO_FOLLOW_ZOOM_MAX: 2.5,
    WORLD_WIDTH: 100,
    WORLD_HEIGHT: 100,
    selectedInspectBody: null,
    selectedInspectPoint: null,
    selectedInspectPointIndex: -1,
    viewZoom: 1,
    viewOffsetX: 12,
    viewOffsetY: 34
  };

  const viewport = { zoom: 1, offsetX: 12, offsetY: 34 };
  const canvas = { clientWidth: 400, clientHeight: 300 };

  runBrowserStepAdapter({
    worldState,
    dt: 1 / 60,
    config,
    stepWorld: () => ({ ok: true }),
    stepOptions: {},
    viewport,
    canvas,
    updateInfoPanel: () => {},
    updateInstabilityIndicator: () => {},
    updatePopulationCount: () => {}
  });

  assert.equal(viewport.zoom, 1);
  assert.equal(viewport.offsetX, 12);
  assert.equal(viewport.offsetY, 34);
  assert.equal(config.viewZoom, 1);
  assert.equal(config.viewOffsetX, 12);
  assert.equal(config.viewOffsetY, 34);
});
