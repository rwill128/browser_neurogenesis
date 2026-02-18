/**
 * GPU-only runtime solver import adapter.
 *
 * The current GPU-only solver port intentionally keeps import/output contracts
 * aligned with baseline while GPU execution internals are built out in a
 * separate, isolated code path.
 */

function cloneRgbTuple(tuple, fallback = [1, 1, 1]) {
  if (!Array.isArray(tuple)) return [...fallback];
  return [Number(tuple?.[0]) || 0, Number(tuple?.[1]) || 0, Number(tuple?.[2]) || 0];
}

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
        edgeBodyMode: Array.isArray(body?.edgeBodyMode)
          ? body.edgeBodyMode.map((v) => Number(v) || 0)
          : [],
        edgeDyeMode: Array.isArray(body?.edgeDyeMode)
          ? body.edgeDyeMode.map((rgb) => cloneRgbTuple(rgb, [1, 1, 1]))
          : [],
        edgeVelocityMode: Array.isArray(body?.edgeVelocityMode)
          ? body.edgeVelocityMode.map((v) => Number(v) || 0)
          : [],
        edgePermeabilityRGB: Array.isArray(body?.edgePermeabilityRGB)
          ? body.edgePermeabilityRGB.map((rgb) => cloneRgbTuple(rgb, [0, 0, 0]))
          : [],
      }))
      : [],
    soft: {
      nodes: Array.isArray(result?.soft?.nodes)
        ? result.soft.nodes.map((node) => ({ ...node }))
        : [],
      springs: Array.isArray(result?.soft?.springs)
        ? result.soft.springs.map((spring) => {
          if (!Array.isArray(spring)) return spring;
          const clone = [...spring];
          clone[4] = cloneRgbTuple(spring?.[4], [1, 1, 1]);
          return clone;
        })
        : [],
    },
    hybrid: [],
    softMembraneClusters: Array.isArray(result?.softMembraneClusters)
      ? result.softMembraneClusters.map((cluster) => ({
        ...cluster,
        nodeIndices: Array.isArray(cluster?.nodeIndices) ? cluster.nodeIndices.map((v) => Number(v) || 0) : [],
      }))
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
