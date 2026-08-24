/**
 * Newborn spawn-stability helpers.
 *
 * Goals:
 * 1) Keep freshly created bodies fully inside the world bounds.
 * 2) Clamp newborn spring parameters in tiny worlds to reduce first-tick blowups.
 */

function asFinite(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function computeBodyBounds(points) {
  if (!Array.isArray(points) || points.length === 0) {
    return { minX: 0, minY: 0, maxX: 0, maxY: 0, width: 0, height: 0 };
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const point of points) {
    const x = asFinite(point?.pos?.x, 0);
    const y = asFinite(point?.pos?.y, 0);
    const radius = Math.max(0, asFinite(point?.radius, 0));

    minX = Math.min(minX, x - radius);
    minY = Math.min(minY, y - radius);
    maxX = Math.max(maxX, x + radius);
    maxY = Math.max(maxY, y + radius);
  }

  return {
    minX,
    minY,
    maxX,
    maxY,
    width: Math.max(0, maxX - minX),
    height: Math.max(0, maxY - minY)
  };
}

function scaleBodyAroundCenter(body, scale, centerX, centerY) {
  if (!Array.isArray(body?.massPoints) || body.massPoints.length === 0 || !Number.isFinite(scale) || scale === 1) {
    return;
  }

  for (const point of body.massPoints) {
    if (!point?.pos) continue;

    const px = asFinite(point.pos.x, centerX);
    const py = asFinite(point.pos.y, centerY);
    point.pos.x = centerX + (px - centerX) * scale;
    point.pos.y = centerY + (py - centerY) * scale;

    if (point.prevPos) {
      const prevX = asFinite(point.prevPos.x, px);
      const prevY = asFinite(point.prevPos.y, py);
      point.prevPos.x = centerX + (prevX - centerX) * scale;
      point.prevPos.y = centerY + (prevY - centerY) * scale;
    }
  }

  if (Array.isArray(body?.springs)) {
    for (const spring of body.springs) {
      if (!spring) continue;
      spring.restLength = asFinite(spring.restLength, 0) * scale;
    }
  }
}

function translateBody(body, dx, dy) {
  if (!Array.isArray(body?.massPoints) || body.massPoints.length === 0) return;

  for (const point of body.massPoints) {
    if (point?.pos) {
      point.pos.x = asFinite(point.pos.x, 0) + dx;
      point.pos.y = asFinite(point.pos.y, 0) + dy;
    }

    if (point?.prevPos) {
      point.prevPos.x = asFinite(point.prevPos.x, 0) + dx;
      point.prevPos.y = asFinite(point.prevPos.y, 0) + dy;
    }
  }
}

function clampBodyPointsToWorld(body, worldWidth, worldHeight) {
  let clampedPoints = 0;

  if (!Array.isArray(body?.massPoints)) return clampedPoints;

  for (const point of body.massPoints) {
    if (!point?.pos) continue;

    const radius = Math.max(0, asFinite(point.radius, 0));
    const minX = radius;
    const maxX = worldWidth - radius;
    const minY = radius;
    const maxY = worldHeight - radius;

    const safeX = (minX <= maxX)
      ? clamp(asFinite(point.pos.x, worldWidth * 0.5), minX, maxX)
      : worldWidth * 0.5;
    const safeY = (minY <= maxY)
      ? clamp(asFinite(point.pos.y, worldHeight * 0.5), minY, maxY)
      : worldHeight * 0.5;

    if (Math.abs(safeX - point.pos.x) > 1e-6 || Math.abs(safeY - point.pos.y) > 1e-6) {
      clampedPoints += 1;
    }

    point.pos.x = safeX;
    point.pos.y = safeY;

    // Reset implicit velocity after forced spawn correction.
    if (point.prevPos) {
      point.prevPos.x = safeX;
      point.prevPos.y = safeY;
    }
  }

  return clampedPoints;
}

/**
 * Keep a newborn body inside the world bounds.
 *
 * If the body does not fit, we scale it down uniformly (runtime phenotype only)
 * and then translate it so its AABB lives inside [padding, world-padding].
 */
export function fitBodyInsideWorld(body, config, { padding = 0.5 } = {}) {
  const worldWidth = Math.max(1, asFinite(config?.WORLD_WIDTH, 1));
  const worldHeight = Math.max(1, asFinite(config?.WORLD_HEIGHT, 1));
  const fitPadding = Math.max(0, asFinite(padding, 0));

  if (!Array.isArray(body?.massPoints) || body.massPoints.length === 0) {
    return {
      adjusted: false,
      scaled: false,
      translated: false,
      scaleApplied: 1,
      translation: { x: 0, y: 0 },
      clampedPoints: 0,
      bounds: computeBodyBounds(body?.massPoints || [])
    };
  }

  let adjusted = false;
  let scaled = false;
  let translated = false;
  let scaleApplied = 1;

  let bounds = computeBodyBounds(body.massPoints);

  const availableWidth = Math.max(1e-6, worldWidth - (fitPadding * 2));
  const availableHeight = Math.max(1e-6, worldHeight - (fitPadding * 2));

  if (bounds.width > availableWidth || bounds.height > availableHeight) {
    const scaleX = availableWidth / Math.max(1e-6, bounds.width);
    const scaleY = availableHeight / Math.max(1e-6, bounds.height);
    const scale = clamp(Math.min(scaleX, scaleY), 0.02, 1);

    const centerX = (bounds.minX + bounds.maxX) * 0.5;
    const centerY = (bounds.minY + bounds.maxY) * 0.5;
    scaleBodyAroundCenter(body, scale, centerX, centerY);

    scaleApplied = scale;
    scaled = scale < 0.999999;
    adjusted = adjusted || scaled;
    bounds = computeBodyBounds(body.massPoints);
  }

  let dx = 0;
  let dy = 0;

  if (bounds.width <= availableWidth) {
    if (bounds.minX < fitPadding) dx = fitPadding - bounds.minX;
    else if (bounds.maxX > worldWidth - fitPadding) dx = (worldWidth - fitPadding) - bounds.maxX;
  } else {
    dx = (worldWidth * 0.5) - ((bounds.minX + bounds.maxX) * 0.5);
  }

  if (bounds.height <= availableHeight) {
    if (bounds.minY < fitPadding) dy = fitPadding - bounds.minY;
    else if (bounds.maxY > worldHeight - fitPadding) dy = (worldHeight - fitPadding) - bounds.maxY;
  } else {
    dy = (worldHeight * 0.5) - ((bounds.minY + bounds.maxY) * 0.5);
  }

  if (Math.abs(dx) > 1e-9 || Math.abs(dy) > 1e-9) {
    translateBody(body, dx, dy);
    translated = true;
    adjusted = true;
  }

  const clampedPoints = clampBodyPointsToWorld(body, worldWidth, worldHeight);
  if (clampedPoints > 0) adjusted = true;

  if (typeof body?._updateBlueprintRadiusFromCurrentPhenotype === 'function') {
    body._updateBlueprintRadiusFromCurrentPhenotype();
  }

  bounds = computeBodyBounds(body.massPoints);

  return {
    adjusted,
    scaled,
    translated,
    scaleApplied,
    translation: { x: dx, y: dy },
    clampedPoints,
    bounds
  };
}

/**
 * Compute newborn spring parameter caps based on world size and dt.
 *
 * The intent is to keep tiny-world newborns from inheriting large-world rigidity.
 */
export function computeNewbornSpringCaps(config, dt) {
  const worldMinDim = Math.max(1, Math.min(asFinite(config?.WORLD_WIDTH, 1), asFinite(config?.WORLD_HEIGHT, 1)));

  const worldRefDim = Math.max(1, asFinite(config?.NEWBORN_STIFFNESS_WORLD_REF_DIM, 1200));
  const dimScale = clamp(worldMinDim / worldRefDim, 0, 1);

  const dtRef = Math.max(1e-5, asFinite(config?.NEWBORN_STIFFNESS_DT_REF, 1 / 30));
  const dtSafe = Math.max(1e-5, asFinite(dt, dtRef));
  const dtExponent = asFinite(config?.NEWBORN_STIFFNESS_DT_EXPONENT, 2);
  const dtScaleRaw = Math.pow(dtRef / dtSafe, dtExponent);
  const dtScale = clamp(dtScaleRaw, 0.25, 2.0);

  const rigidMinScale = clamp(asFinite(config?.NEWBORN_RIGID_STIFFNESS_MIN_SCALE, 0.005), 0.0001, 1);
  const rigidWorldExponent = Math.max(0, asFinite(config?.NEWBORN_RIGID_STIFFNESS_WORLD_EXPONENT, 2));
  const rigidScale = Math.max(rigidMinScale, Math.pow(dimScale, rigidWorldExponent)) * dtScale;

  const nonRigidMinScale = clamp(asFinite(config?.NEWBORN_NON_RIGID_STIFFNESS_MIN_SCALE, 0.05), 0.0001, 1);
  const nonRigidWorldExponent = Math.max(0, asFinite(config?.NEWBORN_NON_RIGID_STIFFNESS_WORLD_EXPONENT, 1));
  const nonRigidScale = Math.max(nonRigidMinScale, Math.pow(dimScale, nonRigidWorldExponent)) * dtScale;

  const rigidBaseStiffness = Math.max(1e-6, asFinite(config?.RIGID_SPRING_STIFFNESS, 500000));
  const rigidBaseDamping = Math.max(1e-6, asFinite(config?.RIGID_SPRING_DAMPING, 150));
  const nonRigidBaseStiffnessCap = Math.max(1e-6, asFinite(config?.NEWBORN_NON_RIGID_STIFFNESS_BASE_CAP, 10000));
  const nonRigidBaseDampingCap = Math.max(1e-6, asFinite(config?.NEWBORN_NON_RIGID_DAMPING_BASE_CAP, 80));

  return {
    rigidStiffnessCap: rigidBaseStiffness * rigidScale,
    rigidDampingCap: rigidBaseDamping * Math.max(0.15, Math.sqrt(rigidScale)),
    nonRigidStiffnessCap: nonRigidBaseStiffnessCap * nonRigidScale,
    nonRigidDampingCap: nonRigidBaseDampingCap * Math.max(0.25, Math.sqrt(nonRigidScale)),
    worldMinDim,
    dtScale,
    dimScale,
    rigidScale,
    nonRigidScale
  };
}

/**
 * Clamp newborn spring parameters to world/dt-aware limits.
 */
export function clampNewbornSpringParameters(body, config, dt) {
  if (!Array.isArray(body?.springs) || body.springs.length === 0) {
    return {
      totalSprings: 0,
      clampedStiffness: 0,
      clampedDamping: 0,
      caps: computeNewbornSpringCaps(config, dt)
    };
  }

  const caps = computeNewbornSpringCaps(config, dt);
  let clampedStiffness = 0;
  let clampedDamping = 0;

  for (const spring of body.springs) {
    if (!spring) continue;

    const isRigid = Boolean(spring.isRigid);
    const stiffnessCap = isRigid ? caps.rigidStiffnessCap : caps.nonRigidStiffnessCap;
    const dampingCap = isRigid ? caps.rigidDampingCap : caps.nonRigidDampingCap;

    const safeStiffness = Math.max(0, asFinite(spring.stiffness, stiffnessCap));
    const safeDamping = Math.max(0, asFinite(spring.dampingFactor, dampingCap));

    const nextStiffness = Math.min(safeStiffness, stiffnessCap);
    const nextDamping = Math.min(safeDamping, dampingCap);

    if (Math.abs(nextStiffness - safeStiffness) > 1e-12) clampedStiffness += 1;
    if (Math.abs(nextDamping - safeDamping) > 1e-12) clampedDamping += 1;

    spring.stiffness = nextStiffness;
    spring.dampingFactor = nextDamping;
  }

  return {
    totalSprings: body.springs.length,
    clampedStiffness,
    clampedDamping,
    caps
  };
}

/**
 * Combined newborn stabilization pass.
 */
export function stabilizeNewbornBody(body, { config, dt, fitPadding = 0.5 } = {}) {
  const fit = fitBodyInsideWorld(body, config, { padding: fitPadding });

  const springClamp = (config?.NEWBORN_STIFFNESS_CLAMP_ENABLED === false)
    ? {
      totalSprings: Array.isArray(body?.springs) ? body.springs.length : 0,
      clampedStiffness: 0,
      clampedDamping: 0,
      caps: computeNewbornSpringCaps(config, dt)
    }
    : clampNewbornSpringParameters(body, config, dt);

  return {
    fit,
    springClamp,
    adjusted: Boolean(fit.adjusted || springClamp.clampedStiffness > 0 || springClamp.clampedDamping > 0)
  };
}
