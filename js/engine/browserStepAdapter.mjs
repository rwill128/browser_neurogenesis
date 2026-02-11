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

  viewport.zoom = desiredZoom;
  config.viewZoom = desiredZoom;

  viewport.offsetX = center.x - (canvas.clientWidth / viewport.zoom / 2);
  viewport.offsetY = center.y - (canvas.clientHeight / viewport.zoom / 2);

  const maxPanX = Math.max(0, config.WORLD_WIDTH - (canvas.clientWidth / viewport.zoom));
  const maxPanY = Math.max(0, config.WORLD_HEIGHT - (canvas.clientHeight / viewport.zoom));
  viewport.offsetX = Math.max(0, Math.min(viewport.offsetX, maxPanX));
  viewport.offsetY = Math.max(0, Math.min(viewport.offsetY, maxPanY));

  config.viewOffsetX = viewport.offsetX;
  config.viewOffsetY = viewport.offsetY;

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
  stepWorld(worldState, dt, stepOptions);

  updateInstabilityIndicator();
  updateAutoFollowCamera({ worldState, config, canvas, viewport, updateInfoPanel });
  updatePopulationCount();
}
