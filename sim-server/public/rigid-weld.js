export function buildRigidWeldPairSet(rigidWelds) {
  const set = new Set();
  for (const w of (rigidWelds || [])) {
    const aRaw = Number(w?.a);
    const bRaw = Number(w?.b);
    if (!Number.isInteger(aRaw) || !Number.isInteger(bRaw) || aRaw === bRaw) continue;
    const a = Math.min(aRaw, bRaw);
    const b = Math.max(aRaw, bRaw);
    set.add(`${a}:${b}`);
  }
  return set;
}

export function applyRigidWeldConstraints({ rigid, rigidWelds, rigidVertexWorld, stiffness = 0.06, errorScale = 0.95, torqueScale = 0.0009, maxPairError = 2.5 }) {
  if (!Array.isArray(rigid) || !Array.isArray(rigidWelds) || typeof rigidVertexWorld !== 'function') return;

  const boundedMaxPairError = Math.max(0.05, Number.isFinite(maxPairError) ? maxPairError : 2.5);

  for (const w of rigidWelds) {
    const ra = rigid[w?.a];
    const rb = rigid[w?.b];
    if (!ra || !rb) continue;

    const a0 = rigidVertexWorld(ra, w.a0);
    const a1 = rigidVertexWorld(ra, w.a1);
    const b0 = rigidVertexWorld(rb, w.b0);
    const b1 = rigidVertexWorld(rb, w.b1);

    const pairs = [[a0, b0], [a1, b1]];
    for (const [pa, pb] of pairs) {
      if (!isFinitePoint(pa) || !isFinitePoint(pb)) continue;

      const dx = pb.x - pa.x;
      const dy = pb.y - pa.y;
      const dRaw = Math.hypot(dx, dy);
      if (!Number.isFinite(dRaw)) continue;

      const d = Math.max(1e-6, dRaw);
      const err = Math.min(d * errorScale, boundedMaxPairError);
      const nx = dx / d;
      const ny = dy / d;
      if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(err)) continue;

      ra.vx = finiteOrZero(ra.vx) + nx * err * stiffness;
      ra.vy = finiteOrZero(ra.vy) + ny * err * stiffness;
      rb.vx = finiteOrZero(rb.vx) - nx * err * stiffness;
      rb.vy = finiteOrZero(rb.vy) - ny * err * stiffness;

      const rax = pa.x - finiteOrZero(ra.x);
      const ray = pa.y - finiteOrZero(ra.y);
      const rbx = pb.x - finiteOrZero(rb.x);
      const rby = pb.y - finiteOrZero(rb.y);

      ra.omega = finiteOrZero(ra.omega) + (rax * ny - ray * nx) * err * torqueScale;
      rb.omega = finiteOrZero(rb.omega) - (rbx * ny - rby * nx) * err * torqueScale;
    }
  }
}

function finiteOrZero(v) {
  return Number.isFinite(v) ? v : 0;
}

function isFinitePoint(p) {
  return !!p && Number.isFinite(p.x) && Number.isFinite(p.y);
}
