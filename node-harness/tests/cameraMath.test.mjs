import test from 'node:test';
import assert from 'node:assert/strict';

import {
  displayToWorld,
  solveViewOffsetForAnchor,
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
