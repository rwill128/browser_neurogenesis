/**
 * Convert a display-space pointer position (CSS pixels) into world coordinates.
 *
 * Canonical camera transform:
 *   screen = (world - viewOffset) * viewZoom
 * therefore:
 *   world = (screen / viewZoom) + viewOffset
 */
export function displayToWorld({ displayX, displayY, viewZoom, viewOffsetX, viewOffsetY }) {
  return {
    x: (displayX / viewZoom) + viewOffsetX,
    y: (displayY / viewZoom) + viewOffsetY
  };
}

/**
 * Convert a world position into display-space (CSS pixels).
 */
export function worldToDisplay({ worldX, worldY, viewZoom, viewOffsetX, viewOffsetY }) {
  return {
    x: (worldX - viewOffsetX) * viewZoom,
    y: (worldY - viewOffsetY) * viewZoom
  };
}

/**
 * Solve camera offsets so a chosen world point remains under the same display pixel
 * after a zoom change.
 */
export function solveViewOffsetForAnchor({ displayX, displayY, worldX, worldY, viewZoom }) {
  return {
    offsetX: worldX - (displayX / viewZoom),
    offsetY: worldY - (displayY / viewZoom)
  };
}

/**
 * Clamp/center camera offsets against world bounds.
 *
 * If the viewport is larger than the world in an axis, offsets become negative so
 * the world is centred in that axis.
 */
export function clampCameraOffsets({
  offsetX,
  offsetY,
  viewZoom,
  worldWidth,
  worldHeight,
  viewportWidth,
  viewportHeight
}) {
  const viewportWorldW = viewportWidth / viewZoom;
  const viewportWorldH = viewportHeight / viewZoom;

  const extraW = viewportWorldW - worldWidth;
  const extraH = viewportWorldH - worldHeight;

  let nextOffsetX;
  let nextOffsetY;

  if (extraW > 0) {
    nextOffsetX = -extraW * 0.5;
  } else {
    const maxPanX = worldWidth - viewportWorldW;
    nextOffsetX = Math.max(0, Math.min(offsetX, maxPanX));
  }

  if (extraH > 0) {
    nextOffsetY = -extraH * 0.5;
  } else {
    const maxPanY = worldHeight - viewportWorldH;
    nextOffsetY = Math.max(0, Math.min(offsetY, maxPanY));
  }

  return {
    offsetX: nextOffsetX,
    offsetY: nextOffsetY
  };
}

/**
 * Compute zoom that fits the entire world inside the viewport.
 */
export function computeZoomToFitWorld({ worldWidth, worldHeight, viewportWidth, viewportHeight, minZoom = 0.01, maxZoom = Infinity }) {
  const fitZoom = Math.min(viewportWidth / worldWidth, viewportHeight / worldHeight);
  return Math.max(minZoom, Math.min(maxZoom, fitZoom));
}

/**
 * Center the camera on a world-space focus point for the given zoom.
 */
export function centerCameraOnPoint({ worldX, worldY, viewZoom, viewportWidth, viewportHeight }) {
  return {
    offsetX: worldX - (viewportWidth / viewZoom / 2),
    offsetY: worldY - (viewportHeight / viewZoom / 2)
  };
}

/**
 * Produce a camera state that shows the full world while preserving bounds logic.
 */
export function buildFitWorldCamera({
  worldWidth,
  worldHeight,
  viewportWidth,
  viewportHeight,
  minZoom = 0.01,
  maxZoom = Infinity
}) {
  const zoom = computeZoomToFitWorld({
    worldWidth,
    worldHeight,
    viewportWidth,
    viewportHeight,
    minZoom,
    maxZoom
  });

  const centered = centerCameraOnPoint({
    worldX: worldWidth / 2,
    worldY: worldHeight / 2,
    viewZoom: zoom,
    viewportWidth,
    viewportHeight
  });

  const clamped = clampCameraOffsets({
    offsetX: centered.offsetX,
    offsetY: centered.offsetY,
    viewZoom: zoom,
    worldWidth,
    worldHeight,
    viewportWidth,
    viewportHeight
  });

  return {
    zoom,
    offsetX: clamped.offsetX,
    offsetY: clamped.offsetY
  };
}

/**
 * Resolve render-space metrics from a canvas element.
 *
 * - cssWidth/cssHeight are camera/input units (CSS pixels).
 * - dpr maps CSS pixels to backing-store device pixels.
 *
 * We derive dpr from canvas backing size when available to keep behavior robust
 * even if devicePixelRatio is unavailable/misreported.
 */
export function resolveCanvasRenderMetrics({
  canvasWidth,
  canvasHeight,
  clientWidth,
  clientHeight,
  fallbackDpr = 1
}) {
  const safeClientWidth = Math.max(1, clientWidth || 0);
  const safeClientHeight = Math.max(1, clientHeight || 0);

  let dprX = Number.isFinite(canvasWidth) ? (canvasWidth / safeClientWidth) : NaN;
  let dprY = Number.isFinite(canvasHeight) ? (canvasHeight / safeClientHeight) : NaN;

  if (!Number.isFinite(dprX) || dprX <= 0) dprX = fallbackDpr;
  if (!Number.isFinite(dprY) || dprY <= 0) dprY = fallbackDpr;

  return {
    cssWidth: safeClientWidth,
    cssHeight: safeClientHeight,
    dprX,
    dprY
  };
}
