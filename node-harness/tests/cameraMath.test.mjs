import test from 'node:test';
import assert from 'node:assert/strict';

import {
  displayToWorld,
  worldToDisplay,
  solveViewOffsetForAnchor,
  clampCameraOffsets,
  computeZoomToFitWorld,
  buildFitWorldCamera,
  resolveCanvasRenderMetrics
} from '../../js/engine/cameraMath.mjs';

test('displayToWorld handles negative offsets (centered small world)', () => {
  const world = displayToWorld({
    displayX: 70,
    displayY: 40,
    viewZoom: 2.5,
    viewOffsetX: -28,
    viewOffsetY: -16
  });

  assert.equal(world.x, 0);
  assert.equal(world.y, 0);
});

test('solveViewOffsetForAnchor preserves world point under cursor after zoom', () => {
  const worldPoint = { x: 42, y: 13 };
  const display = { x: 240, y: 180 };

  const atZoom2 = solveViewOffsetForAnchor({
    displayX: display.x,
    displayY: display.y,
    worldX: worldPoint.x,
    worldY: worldPoint.y,
    viewZoom: 2
  });

  const resolvedWorld = displayToWorld({
    displayX: display.x,
    displayY: display.y,
    viewZoom: 2,
    viewOffsetX: atZoom2.offsetX,
    viewOffsetY: atZoom2.offsetY
  });

  assert.equal(resolvedWorld.x, worldPoint.x);
  assert.equal(resolvedWorld.y, worldPoint.y);
});

test('worldToDisplay is inverse-compatible with displayToWorld', () => {
  const world = { x: 18.5, y: 92.25 };
  const camera = {
    viewZoom: 1.75,
    viewOffsetX: -8,
    viewOffsetY: 14
  };

  const display = worldToDisplay({
    worldX: world.x,
    worldY: world.y,
    viewZoom: camera.viewZoom,
    viewOffsetX: camera.viewOffsetX,
    viewOffsetY: camera.viewOffsetY
  });

  const roundTripWorld = displayToWorld({
    displayX: display.x,
    displayY: display.y,
    viewZoom: camera.viewZoom,
    viewOffsetX: camera.viewOffsetX,
    viewOffsetY: camera.viewOffsetY
  });

  assert.equal(roundTripWorld.x, world.x);
  assert.equal(roundTripWorld.y, world.y);
});

test('clampCameraOffsets centers world when viewport is larger', () => {
  const clamped = clampCameraOffsets({
    offsetX: 999,
    offsetY: -999,
    viewZoom: 2,
    worldWidth: 100,
    worldHeight: 100,
    viewportWidth: 400,
    viewportHeight: 300
  });

  // Viewport in world units is 200x150, so centered offsets are -(extra/2).
  assert.equal(clamped.offsetX, -50);
  assert.equal(clamped.offsetY, -25);
});

test('computeZoomToFitWorld returns min axis fit zoom', () => {
  const zoom = computeZoomToFitWorld({
    worldWidth: 100,
    worldHeight: 50,
    viewportWidth: 300,
    viewportHeight: 120,
    minZoom: 0.01,
    maxZoom: 10
  });

  // min(300/100, 120/50) = min(3, 2.4) = 2.4
  assert.equal(zoom, 2.4);
});

test('buildFitWorldCamera returns centered fit camera', () => {
  const camera = buildFitWorldCamera({
    worldWidth: 100,
    worldHeight: 100,
    viewportWidth: 240,
    viewportHeight: 100,
    minZoom: 0.01,
    maxZoom: 5
  });

  // Fit zoom = min(2.4, 1) = 1; world should be centered horizontally.
  assert.equal(camera.zoom, 1);
  assert.equal(camera.offsetX, -70);
  assert.equal(camera.offsetY, 0);
});

test('resolveCanvasRenderMetrics derives DPR from backing-store size', () => {
  const metrics = resolveCanvasRenderMetrics({
    canvasWidth: 1170,
    canvasHeight: 2532,
    clientWidth: 390,
    clientHeight: 844,
    fallbackDpr: 1
  });

  assert.equal(metrics.cssWidth, 390);
  assert.equal(metrics.cssHeight, 844);
  assert.equal(metrics.dprX, 3);
  assert.equal(metrics.dprY, 3);
});
