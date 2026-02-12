/**
 * Convert a display-space pointer position (CSS pixels) into world coordinates.
 *
 * This mapping is the canonical browser-camera transform used by the renderer:
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
