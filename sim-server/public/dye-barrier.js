export const EDGE_DYE_MODE = {
  PASS: 0,
  DEFLECT: 1,
  ABSORB: 2,
};

export const EDGE_BODY_MODE = {
  PASS: 0,
  BLOCK: 1,
};

export function normalizeEdgeDyeModeChannel(mode) {
  const n = Number(mode);
  if (n === EDGE_DYE_MODE.PASS || n === EDGE_DYE_MODE.DEFLECT || n === EDGE_DYE_MODE.ABSORB) return n;
  return EDGE_DYE_MODE.DEFLECT;
}

export function normalizeEdgeDyeModeRGB(mode) {
  if (!Array.isArray(mode) || mode.length < 3) {
    return [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT];
  }
  return [
    normalizeEdgeDyeModeChannel(mode[0]),
    normalizeEdgeDyeModeChannel(mode[1]),
    normalizeEdgeDyeModeChannel(mode[2]),
  ];
}

export function normalizePermeabilityRGB(mode) {
  if (!Array.isArray(mode) || mode.length < 3) {
    return [0, 0, 0];
  }
  return [
    Number(mode[0]) > 0 ? 1 : 0,
    Number(mode[1]) > 0 ? 1 : 0,
    Number(mode[2]) > 0 ? 1 : 0,
  ];
}

export function applyImpermeableSegmentFieldBarrier(n, ax, ay, bx, by, r, g, b, vx, vy, thickness = 1.4, dyeMode = EDGE_DYE_MODE.DEFLECT, bodyMode = EDGE_BODY_MODE.BLOCK, applyDye = true, applyVelocity = true) {
  const ex = bx - ax;
  const ey = by - ay;
  const segLenSq = ex * ex + ey * ey;
  // Guardrail: collapsed/near-collapsed edges should not behave like point barriers,
  // which can over-attenuate dye and velocity when a spring temporarily degenerates.
  if (!Number.isFinite(segLenSq) || segLenSq < 1e-9) return;

  const minX = Math.max(0, Math.floor(Math.min(ax, bx) - thickness - 1));
  const maxX = Math.min(n - 1, Math.ceil(Math.max(ax, bx) + thickness + 1));
  const minY = Math.max(0, Math.floor(Math.min(ay, by) - thickness - 1));
  const maxY = Math.min(n - 1, Math.ceil(Math.max(ay, by) + thickness + 1));
  const el = Math.max(1e-6, Math.hypot(ex, ey));
  const nx = -ey / el;
  const ny = ex / el;
  const tx = ex / el;
  const ty = ey / el;

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const cp = closestPointOnSegment(x + 0.5, y + 0.5, ax, ay, bx, by);
      const dx = (x + 0.5) - cp.x;
      const dy = (y + 0.5) - cp.y;
      const d = Math.hypot(dx, dy);
      if (d > thickness) continue;
      const w = 1 - (d / Math.max(1e-6, thickness));
      const i = y * n + x;

      const modeRGB = normalizeEdgeDyeModeRGB(dyeMode);
      if (applyVelocity && bodyMode === EDGE_BODY_MODE.BLOCK) {
        const vn = vx[i] * nx + vy[i] * ny;
        vx[i] -= vn * nx * w;
        vy[i] -= vn * ny * w;
      }

      if (applyDye) {
        const vt = vx[i] * tx + vy[i] * ty;
        const step = vt >= 0 ? 1 : -1;
        const txi = Math.max(0, Math.min(n - 1, Math.round(x + tx * step)));
        const tyi = Math.max(0, Math.min(n - 1, Math.round(y + ty * step)));
        const ti = tyi * n + txi;
        const channels = [r, g, b];
        for (let ci = 0; ci < 3; ci++) {
          const mode = modeRGB[ci];
          if (mode === EDGE_DYE_MODE.DEFLECT) {
            if (ti !== i) {
              const move = 0.22 * w;
              const dch = channels[ci][i] * move;
              channels[ci][i] -= dch;
              channels[ci][ti] = Math.min(255, channels[ci][ti] + dch);
            }
          } else if (mode === EDGE_DYE_MODE.ABSORB) {
            channels[ci][i] *= (1 - 0.9 * w);
          }
        }
      }
    }
  }
}

export function applyBodyEdgeFieldBarriers({ sim, r, g, b, vx, vy, rigidVerticesWorld, skipSoftEdges = false, softBodyModeOverride = null, skipDye = false }) {
  const n = sim.controls.n;
  const softThickness = Math.max(1.2, 1.1 * (n / 256));
  const rigidThickness = Math.max(1.4, 1.2 * (n / 256));

  for (const rb of sim.bodies.rigid || []) {
    const verts = rigidVerticesWorld(rb);
    const sides = verts.length;
    for (let i = 0; i < sides; i++) {
      const rawModeRGB = normalizeEdgeDyeModeRGB(resolveRigidEdgeRGBTuple(rb?.edgeDyeMode, i, [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT]));
      const permeabilityRGB = normalizePermeabilityRGB(resolveRigidEdgeRGBTuple(rb?.edgePermeabilityRGB, i, [0, 0, 0]));
      const dyeModeRGB = [0, 0, 0].map((_, ci) => {
        if (permeabilityRGB[ci] > 0) return EDGE_DYE_MODE.PASS;
        // If impermeable, preserve ABSORB semantics when explicitly requested; otherwise DEFLECT.
        return rawModeRGB[ci] === EDGE_DYE_MODE.ABSORB ? EDGE_DYE_MODE.ABSORB : EDGE_DYE_MODE.DEFLECT;
      });
      const bodyMode = resolveRigidEdgeVelocityMode(rb, i);
      if (bodyMode === EDGE_BODY_MODE.PASS
        && dyeModeRGB[0] === EDGE_DYE_MODE.PASS
        && dyeModeRGB[1] === EDGE_DYE_MODE.PASS
        && dyeModeRGB[2] === EDGE_DYE_MODE.PASS) continue;
      const a = verts[i];
      const b2 = verts[(i + 1) % sides];
      if (!isFinitePoint(a) || !isFinitePoint(b2)) continue;
      applyImpermeableSegmentFieldBarrier(n, a.x, a.y, b2.x, b2.y, r, g, b, vx, vy, rigidThickness, dyeModeRGB, bodyMode, !skipDye, true);
    }
  }

  if (!skipSoftEdges) {
    const s = sim.bodies.soft;
    const overrideRaw = softBodyModeOverride;
    const hasSoftBodyOverride = overrideRaw !== null
      && overrideRaw !== undefined
      && (Number(overrideRaw) === EDGE_BODY_MODE.PASS || Number(overrideRaw) === EDGE_BODY_MODE.BLOCK);
    const softBodyModeResolved = hasSoftBodyOverride
      ? (Number(overrideRaw) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK)
      : null;

    for (const [i, j, _rest, edgeBodyMode, edgeDyeMode, edgeVelocityMode] of (s?.springs || [])) {
      const dyeModeRGB = normalizeEdgeDyeModeRGB(edgeDyeMode);
      const bodyModeFromEdge = Number(edgeVelocityMode) === EDGE_BODY_MODE.PASS
        ? EDGE_BODY_MODE.PASS
        : (Number(edgeBodyMode) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK);
      const bodyMode = softBodyModeResolved ?? bodyModeFromEdge;
      if (bodyMode === EDGE_BODY_MODE.PASS
        && dyeModeRGB[0] === EDGE_DYE_MODE.PASS
        && dyeModeRGB[1] === EDGE_DYE_MODE.PASS
        && dyeModeRGB[2] === EDGE_DYE_MODE.PASS) continue;
      const a = s.nodes?.[i];
      const b2 = s.nodes?.[j];
      if (!isFinitePoint(a) || !isFinitePoint(b2)) continue;
      applyImpermeableSegmentFieldBarrier(n, a.x, a.y, b2.x, b2.y, r, g, b, vx, vy, softThickness, dyeModeRGB, bodyMode, !skipDye, true);
    }
  }
}

function isFinitePoint(p) {
  return Number.isFinite(Number(p?.x)) && Number.isFinite(Number(p?.y));
}

function resolveRigidEdgeScalar(spec, edgeIndex, fallback) {
  if (!Array.isArray(spec)) return fallback;
  const edgeValue = spec[edgeIndex];
  return edgeValue === undefined ? fallback : edgeValue;
}

function resolveRigidEdgeVelocityMode(rb, edgeIndex) {
  const vel = resolveRigidEdgeScalar(rb?.edgeVelocityMode, edgeIndex, null);
  if (vel !== null && vel !== undefined) {
    return Number(vel) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK;
  }
  return Number(resolveRigidEdgeScalar(rb?.edgeBodyMode, edgeIndex, EDGE_BODY_MODE.BLOCK)) === EDGE_BODY_MODE.PASS
    ? EDGE_BODY_MODE.PASS
    : EDGE_BODY_MODE.BLOCK;
}

function resolveRigidEdgeRGBTuple(spec, edgeIndex, fallback) {
  if (!Array.isArray(spec)) return fallback;
  const edgeValue = spec[edgeIndex];
  if (!Array.isArray(edgeValue) || edgeValue.length < 3) return fallback;
  return edgeValue;
}

function closestPointOnSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const den = Math.max(1e-9, abx * abx + aby * aby);
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / den));
  return { x: ax + abx * t, y: ay + aby * t, t };
}
