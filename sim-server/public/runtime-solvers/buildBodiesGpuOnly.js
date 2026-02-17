/**
 * GPU-only runtime solver import adapter.
 *
 * The current GPU-only solver port intentionally keeps import/output contracts
 * aligned with baseline while GPU execution internals are built out in a
 * separate, isolated code path.
 */

function cloneBodies(result) {
  return {
    rigid: Array.isArray(result?.rigid)
      ? result.rigid.map((body) => ({
        ...body,
        verticesLocal: Array.isArray(body?.verticesLocal)
          ? body.verticesLocal.map((v) => ({ x: Number(v?.x) || 0, y: Number(v?.y) || 0 }))
          : [],
        subPolysLocal: Array.isArray(body?.subPolysLocal)
          ? body.subPolysLocal.map((poly) => (Array.isArray(poly)
            ? poly.map((p) => ({ x: Number(p?.x) || 0, y: Number(p?.y) || 0 }))
            : []))
          : null,
      }))
      : [],
    soft: {
      nodes: Array.isArray(result?.soft?.nodes)
        ? result.soft.nodes.map((node) => ({ ...node }))
        : [],
      springs: Array.isArray(result?.soft?.springs)
        ? result.soft.springs.map((spring) => (Array.isArray(spring) ? [...spring] : spring))
        : [],
    },
    hybrid: Array.isArray(result?.hybrid) ? result.hybrid.map((joint) => ({ ...joint })) : [],
    softMembraneClusters: Array.isArray(result?.softMembraneClusters)
      ? result.softMembraneClusters.map((cluster) => ({ ...cluster }))
      : [],
  };
}

export function buildBodiesFromCreatureSpecGpuOnly({ spec, n, controls, buildBaselineBodies }) {
  if (typeof buildBaselineBodies !== 'function') {
    throw new Error('gpu-only build adapter requires buildBaselineBodies callback');
  }
  // For now, GPU-only import/output uses baseline geometry compilation as a
  // deterministic reference, but remains isolated in its own module.
  const baselineResult = buildBaselineBodies(spec, n, controls);
  return cloneBodies(baselineResult);
}
