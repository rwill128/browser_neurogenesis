import {
  centerCameraOnPoint,
  clampCameraOffsets
} from './cameraMath.mjs';

let followInfoRefreshCounter = 0;

function getValidFollowTarget(worldState, config) {
  if (
    config.selectedInspectBody &&
    !config.selectedInspectBody.isUnstable &&
    worldState.softBodyPopulation.includes(config.selectedInspectBody) &&
    config.selectedInspectBody.massPoints.length > 0
  ) {
    return config.selectedInspectBody;
  }

  for (const body of worldState.softBodyPopulation) {
    if (!body.isUnstable && body.massPoints && body.massPoints.length > 0) {
      return body;
    }
  }

  return null;
}

/**
 * Keep browser camera locked on a valid creature target.
 *
 * Camera bounds/centering are delegated to shared camera math so follow mode and
 * manual UI controls use identical world/screen rules.
 */
function updateAutoFollowCamera({ worldState, config, canvas, viewport, updateInfoPanel }) {
  if (!config.AUTO_FOLLOW_CREATURE) return;

  const target = getValidFollowTarget(worldState, config);
  if (!target) {
    config.selectedInspectBody = null;
    config.selectedInspectPoint = null;
    config.selectedInspectPointIndex = -1;
    return;
  }

  if (config.selectedInspectBody !== target) {
    config.selectedInspectBody = target;
    config.selectedInspectPointIndex = 0;
    config.selectedInspectPoint = target.massPoints[0] || null;
    followInfoRefreshCounter = 0;
  } else if (!config.selectedInspectPoint || !target.massPoints.includes(config.selectedInspectPoint)) {
    config.selectedInspectPointIndex = 0;
    config.selectedInspectPoint = target.massPoints[0] || null;
  }

  const center = target.getAveragePosition();
  const bbox = target.getBoundingBox();
  const bodySize = Math.max(40, bbox.maxX - bbox.minX, bbox.maxY - bbox.minY);
  const desiredZoom = Math.min(
    config.AUTO_FOLLOW_ZOOM_MAX,
    Math.max(config.AUTO_FOLLOW_ZOOM_MIN, Math.min(canvas.clientWidth, canvas.clientHeight) / (bodySize * 6))
  );

  const centered = centerCameraOnPoint({
    worldX: center.x,
    worldY: center.y,
    viewZoom: desiredZoom,
    viewportWidth: canvas.clientWidth,
    viewportHeight: canvas.clientHeight
  });

  const clamped = clampCameraOffsets({
    offsetX: centered.offsetX,
    offsetY: centered.offsetY,
    viewZoom: desiredZoom,
    worldWidth: config.WORLD_WIDTH,
    worldHeight: config.WORLD_HEIGHT,
    viewportWidth: canvas.clientWidth,
    viewportHeight: canvas.clientHeight
  });

  viewport.zoom = desiredZoom;
  config.viewZoom = desiredZoom;
  viewport.offsetX = clamped.offsetX;
  viewport.offsetY = clamped.offsetY;
  config.viewOffsetX = clamped.offsetX;
  config.viewOffsetY = clamped.offsetY;

  followInfoRefreshCounter++;
  if (followInfoRefreshCounter >= 20) {
    updateInfoPanel();
    followInfoRefreshCounter = 0;
  }
}

export function runBrowserStepAdapter({
  worldState,
  dt,
  config,
  stepWorld,
  stepOptions,
  viewport,
  canvas,
  updateInfoPanel,
  updateInstabilityIndicator,
  updatePopulationCount
}) {
  const stepResult = stepWorld(worldState, dt, stepOptions);

  updateInstabilityIndicator();
  updateAutoFollowCamera({ worldState, config, canvas, viewport, updateInfoPanel });
  updatePopulationCount();

  return stepResult;
}
