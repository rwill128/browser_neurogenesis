export const CREATURE_SPEC_VERSION = 'creature-spec.v2';

const EDGE_BODY_BLOCK = 1;
const EDGE_DYE_PASS = 0;
const EDGE_DYE_DEFLECT = 1;
const EDGE_DYE_ABSORB = 2;
const EDGE_DYE_DEFLECT_RGB = [EDGE_DYE_DEFLECT, EDGE_DYE_DEFLECT, EDGE_DYE_DEFLECT];
const MEMBRANE_DEFAULT_PRESSURE_GAIN = 0.08;
const MEMBRANE_DEFAULT_RADIAL_DAMPING = 0.06;
const MEMBRANE_DEFAULT_SHAPE_MEMORY_GAIN = 0.045;
const MEMBRANE_DEFAULT_MIN_EDGE_LENGTH = 4.0;
const MEMBRANE_DEFAULT_MAX_NODES = 96;
const MEMBRANE_DEFAULT_SIMPLIFY_EPS = 0.8;

/**
 * Normalize solver mode strings to the stable runtime contract.
 *
 * Forward-compatibility rule: only the explicit `"membrane"` token enables
 * membrane behavior; unknown/experimental values deterministically fall back to
 * `"spring"`.
 *
 * @param {unknown} mode
 * @returns {'spring'|'membrane'}
 */
function normalizeSoftSolverMode(mode) {
  return String(mode || '').toLowerCase() === 'membrane' ? 'membrane' : 'spring';
}

/**
 * Clamp a numeric value to an inclusive range.
 *
 * Non-finite values are expected to be sanitized by callers before use.
 *
 * @param {number} v
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Build a perimeter-only spring ring from ordered membrane nodes.
 *
 * This intentionally emits exactly one cyclic edge per valid node (no interior
 * chords), with blocking body mode + deflect dye defaults on each spring.
 *
 * @param {Array<{x:number,y:number}>} nodes
 * @returns {Array<[number, number, number, number, [number, number, number]]>}
 */
function buildMembranePerimeterSpringsFromNodes(nodes) {
  const ring = [];
  for (let idx = 0; idx < (nodes || []).length; idx++) {
    const p = nodes[idx];
    const x = Number(p?.x);
    const y = Number(p?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    ring.push(idx);
  }

  if (ring.length < 3) return [];

  const springs = [];
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i];
    const b = ring[(i + 1) % ring.length];
    const pa = nodes[a];
    const pb = nodes[b];
    if (!pa || !pb || a === b) continue;
    springs.push([
      a,
      b,
      Math.max(1e-4, Math.hypot((Number(pb.x) || 0) - (Number(pa.x) || 0), (Number(pb.y) || 0) - (Number(pa.y) || 0))),
      EDGE_BODY_BLOCK,
      [...EDGE_DYE_DEFLECT_RGB],
    ]);
  }

  return springs.length >= 3 ? springs : [];
}

/**
 * Build a normalized creature-spec.v2 document from mesh/compiler output.
 *
 * The exporter keeps solver-facing geometry deterministic and optionally carries
 * authoring fields (`rigidField`, `softField`, membrane maps) for later mutation
 * or re-editing workflows.
 *
 * @param {object} mesh
 * @param {Array<object>} [mesh.nodes]
 * @param {Array<object>} [mesh.triangles]
 * @param {object} [mesh.meta]
 * @param {object} [options]
 * @returns {object}
 */
export function createCreatureSpecFromMesh(mesh, options = {}) {
  const width = mesh?.meta?.width || options.width || 128;
  const height = mesh?.meta?.height || options.height || 128;
  const nodes = (mesh?.nodes || []).map((p, i) => ({
    id: i,
    x: Number(p.x) || 0,
    y: Number(p.y) || 0,
    rigid: Number(p.rigid) || 0,
    soft: Number(p.soft) || 0,
  }));

  const triByKind = { rigid: [], soft: [] };
  for (const t of (mesh?.triangles || [])) {
    if (t?.kind === 'rigid') triByKind.rigid.push(t);
    else if (t?.kind === 'soft') triByKind.soft.push(t);
  }

  const rigidBuild = Array.isArray(mesh?.rigidPieces)
    ? buildRigidExportFromCompilerPieces(mesh.rigidPieces, nodes, options, width, height)
    : buildRigidExport(triByKind.rigid, nodes, options, width, height);

  const solverMode = normalizeSoftSolverMode(options?.softSolverMode);
  const hasAuthoringSoftField = Array.isArray(options?.fields?.softField)
    || (options?.fields?.softField instanceof Float32Array);
  const softBuild = (solverMode === 'membrane' && hasAuthoringSoftField)
    ? buildSoftMembraneExportFromField({
        width,
        height,
        softField: options?.fields?.softField,
        edgeLengthField: options?.fields?.membraneEdgeMap,
        shapeMemoryField: options?.fields?.membraneShapeMap,
        threshold: Number(mesh?.meta?.threshold) || Number(options?.threshold) || 0.35,
        options,
      })
    : buildSoftExport(triByKind.soft, nodes, options, mesh?.softCrossBeams || []);

  const hybridJoints = buildHybridExport({
    rigidComps: rigidBuild.components,
    softComps: softBuild.components,
    rigidBodies: rigidBuild.rigidBodies,
    sourceNodes: nodes,
  });

  const out = {
    schemaVersion: CREATURE_SPEC_VERSION,
    createdAt: new Date().toISOString(),
    name: options.name || 'unnamed-creature',
    space: { width, height },
    rigidBodies: rigidBuild.rigidBodies.map((rb) => compactRigidBodyEdgeDefaults(rb)),
    softBodies: softBuild.softBodies,
    hybridJoints,
  };

  // Authoring payload is optional and intentionally separate from solver contract.
  if (options.includeAuthoring !== false) {
    const rigidField = options?.fields?.rigidField;
    const softField = options?.fields?.softField;
    const softDensityField = options?.fields?.softDensityField;
    const membraneEdgeMap = options?.fields?.membraneEdgeMap;
    const membraneShapeMap = options?.fields?.membraneShapeMap;
    const rigidPermeabilityMap = options?.fields?.rigidPermeabilityMap;
    if (rigidField && softField) {
      out.authoring = {
        fields: {
          width,
          height,
          rigid: Array.from(rigidField),
          soft: Array.from(softField),
          softDensity: softDensityField ? Array.from(softDensityField) : undefined,
          membraneEdgeMap: membraneEdgeMap ? Array.from(membraneEdgeMap) : undefined,
          membraneShapeMap: membraneShapeMap ? Array.from(membraneShapeMap) : undefined,
          rigidPermeabilityMap: rigidPermeabilityMap ? Array.from(rigidPermeabilityMap) : undefined,
        },
      };
    }
  }

  return out;
}

/**
 * Remove rigid edge arrays when every entry equals deterministic solver defaults.
 *
 * This keeps exports compact without losing meaning, because importer-side
 * normalization recreates these defaults by hull edge count.
 *
 * @param {object} rb
 * @returns {object}
 */
function compactRigidBodyEdgeDefaults(rb) {
  if (!rb || typeof rb !== 'object') return rb;
  const out = { ...rb };

  if (Array.isArray(out.edgeBodyMode) && out.edgeBodyMode.length > 0) {
    const allBlock = out.edgeBodyMode.every((m) => Number(m) === EDGE_BODY_BLOCK);
    if (allBlock) delete out.edgeBodyMode;
  }

  if (Array.isArray(out.edgeDyeMode) && out.edgeDyeMode.length > 0) {
    const allDeflect = out.edgeDyeMode.every((m) => Array.isArray(m)
      && Number(m[0]) === EDGE_DYE_DEFLECT
      && Number(m[1]) === EDGE_DYE_DEFLECT
      && Number(m[2]) === EDGE_DYE_DEFLECT);
    if (allDeflect) delete out.edgeDyeMode;
  }

  if (Array.isArray(out.edgePermeabilityRGB) && out.edgePermeabilityRGB.length > 0) {
    const allBlocked = out.edgePermeabilityRGB.every((m) => Array.isArray(m)
      && Number(m[0]) <= 0
      && Number(m[1]) <= 0
      && Number(m[2]) <= 0);
    if (allBlocked) delete out.edgePermeabilityRGB;
  }

  return out;
}

/**
 * Parse and validate creature-spec.v2 JSON text.
 *
 * Validation is intentionally strict so downstream body construction can assume
 * required arrays/sections exist.
 *
 * @param {string} jsonText
 * @returns {object}
 * @throws {Error} When schemaVersion or required sections are invalid.
 */
export function parseCreatureSpec(jsonText) {
  const obj = JSON.parse(jsonText);
  if (!obj || typeof obj !== 'object') throw new Error('Invalid JSON object');
  if (obj.schemaVersion !== CREATURE_SPEC_VERSION) {
    throw new Error(`Unsupported schemaVersion: ${obj.schemaVersion}`);
  }
  if (!obj.space || !Number.isFinite(obj.space.width) || !Number.isFinite(obj.space.height)) {
    throw new Error('CreatureSpec missing valid space.width/space.height');
  }
  if (!Array.isArray(obj.rigidBodies) || !Array.isArray(obj.softBodies) || !Array.isArray(obj.hybridJoints)) {
    throw new Error('CreatureSpec missing rigidBodies/softBodies/hybridJoints arrays');
  }
  for (const rb of obj.rigidBodies) {
    if (!Array.isArray(rb.hull) || rb.hull.length < 3) throw new Error('Rigid body missing hull vertices');
  }
  for (const sb of obj.softBodies) {
    if (!Array.isArray(sb.nodes) || !Array.isArray(sb.springs)) throw new Error('Soft body missing nodes/springs');
  }
  return obj;
}

/**
 * Convert a validated creature spec into runtime rigid/soft/hybrid body arrays.
 *
 * Coordinates are scaled into an `n x n` simulation domain while preserving
 * relative morphology and solver-ready edge/node attributes.
 *
 * @param {object} spec Parsed creature spec object.
 * @param {number} n Target simulation domain size.
 * @param {object} controls Mass defaults and runtime control constants.
 * @returns {object}
 */
export function buildBodiesFromCreatureSpec(spec, n, controls) {
  const srcW = Math.max(1, Number(spec.space?.width) || n);
  const srcH = Math.max(1, Number(spec.space?.height) || n);
  const sx = n / srcW;
  const sy = n / srcH;
  const sRest = 0.5 * (sx + sy);

  const rigid = [];
  for (let rbi = 0; rbi < (spec.rigidBodies || []).length; rbi++) {
    const rb = spec.rigidBodies[rbi];
    const hull = (rb.hull || [])
      .map((p) => ({ x: Number(p?.x) * sx, y: Number(p?.y) * sy }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (hull.length < 3) continue;

    const c = centroid(hull);
    const verticesLocal = hull.map((p) => ({ x: p.x - c.x, y: p.y - c.y }));
    const r = Math.max(2.5, ...verticesLocal.map((v) => Math.hypot(v.x, v.y)));
    const sides = Math.max(3, verticesLocal.length);
    const mass = finiteOr(Number(rb.mass), controls.massHeavy);

    const subPolysLocal = Array.isArray(rb.subHulls)
      ? rb.subHulls
          .map((poly) => (Array.isArray(poly) ? poly : []))
          .map((poly) => poly
            .map((p) => ({ x: (Number(p?.x) || 0) * sx - c.x, y: (Number(p?.y) || 0) * sy - c.y }))
            .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)))
          .filter((poly) => poly.length >= 3)
      : null;

    rigid.push({
      x: c.x,
      y: c.y,
      vx: 0,
      vy: 0,
      r,
      sides,
      verticesLocal,
      subPolysLocal,
      edgeDyeMode: normalizeEdgeDyeModeList(rb.edgeDyeMode, sides),
      edgeBodyMode: normalizeEdgeBodyModeList(rb.edgeBodyMode, sides),
      edgePermeabilityRGB: normalizeEdgePermeabilityList(rb.edgePermeabilityRGB, sides),
      digestEnabled: !!rb.digestEnabled,
      digestRGB: normalizeRGB(rb.digestRGB),
      consumeDyeRGB: normalizeBinaryRGB(rb.consumeDyeRGB ?? (rb.digestEnabled ? [1, 1, 1] : [0, 0, 0])),
      insideCorrectionEnabled: rb?.insideCorrectionEnabled !== false,
      mass,
      theta: 0,
      omega: 0,
      inertia: finiteOr(Number(rb.inertia), 0.5 * mass * r * r),
    });
  }

  const soft = { nodes: [], springs: [] };
  const softMembraneClusters = [];
  const softNodeMap = new Map();
  let clusterId = 0;
  for (let sbi = 0; sbi < (spec.softBodies || []).length; sbi++) {
    const sb = spec.softBodies[sbi];
    const solverMode = normalizeSoftSolverMode(sb?.solverMode);
    const base = soft.nodes.length;
    for (let i = 0; i < sb.nodes.length; i++) {
      const p = sb.nodes[i] || {};
      soft.nodes.push({
        x: (Number(p.x) || 0) * sx,
        y: (Number(p.y) || 0) * sy,
        vx: 0,
        vy: 0,
        mass: finiteOr(Number(p.mass), controls.massSoft),
        r: finiteOr(Number(p.r), 1.6),
        clusterId,
        digestEnabled: !!p.digestEnabled,
        digestRGB: normalizeRGB(p.digestRGB),
        shapeMemoryWeight: clamp(Number.isFinite(Number(p.shapeMemoryWeight)) ? Number(p.shapeMemoryWeight) : 1, 0, 1),
      });
      softNodeMap.set(`${sbi}:${i}`, base + i);
    }

    const springSource = (solverMode === 'membrane')
      ? buildMembranePerimeterSpringsFromNodes(sb.nodes)
      : sb.springs;

    for (const sp of (springSource || [])) {
      if (!Array.isArray(sp)) continue;
      const [aRaw, bRaw, restRaw, edgeBodyRaw, edgeDyeRaw] = sp;
      const a = Number(aRaw);
      const b = Number(bRaw);
      if (!Number.isInteger(a) || !Number.isInteger(b)) continue;
      if (a < 0 || b < 0 || a >= sb.nodes.length || b >= sb.nodes.length) continue;
      soft.springs.push([
        base + a,
        base + b,
        Math.max(1e-4, finiteOr(Number(restRaw), 1) * sRest),
        normalizeEdgeBodyMode(edgeBodyRaw),
        normalizeEdgeDyeMode(edgeDyeRaw),
      ]);
    }

    if (solverMode === 'membrane') {
      softMembraneClusters.push({
        clusterId,
        restArea: Math.max(1e-4, Number(sb?.restArea) || 0),
        pressureGain: Math.max(0.001, Number(sb?.pressureGain) || MEMBRANE_DEFAULT_PRESSURE_GAIN),
        radialDamping: Math.max(0, Math.min(0.2, Number(sb?.radialDamping) || MEMBRANE_DEFAULT_RADIAL_DAMPING)),
        shapeMemoryGain: Math.max(0, Math.min(0.35, Number(sb?.shapeMemoryGain) || MEMBRANE_DEFAULT_SHAPE_MEMORY_GAIN)),
        insideCorrectionEnabled: Number.isFinite(Number(sb?.insideCorrectionEnabled))
          ? (Number(sb.insideCorrectionEnabled) > 0 ? 1 : 0)
          : 1,
        solverMode,
      });
    }

    clusterId += 1;
  }

  const hybrid = [];
  for (const j of (spec.hybridJoints || [])) {
    const rigidIndex = Number(j.rigidBodyIndex);
    const softBodyIndex = Number(j.softBodyIndex);
    const softNodeIndex = Number(j.softNodeIndex);
    if (!Number.isInteger(rigidIndex) || rigidIndex < 0 || rigidIndex >= rigid.length) continue;

    const globalSoft = softNodeMap.get(`${softBodyIndex}:${softNodeIndex}`);
    if (!Number.isInteger(globalSoft) || globalSoft < 0 || globalSoft >= soft.nodes.length) continue;

    const rb = rigid[rigidIndex];
    const edgeA = Math.max(0, Math.min(rb.sides - 1, Number(j.edgeA) | 0));
    let edgeB = Math.max(0, Math.min(rb.sides - 1, Number(j.edgeB) | 0));
    // Guardrail: avoid degenerate hybrid constraints that pin both rest links
    // to the same rigid vertex (can inject solver jitter under fluid load).
    if (rb.sides > 1 && edgeA === edgeB) edgeB = (edgeA + 1) % rb.sides;

    const aPos = rigidVertexWorld(rb, edgeA);
    const bPos = rigidVertexWorld(rb, edgeB);
    const p = soft.nodes[globalSoft];

    const defaultRestA = Math.hypot(p.x - aPos.x, p.y - aPos.y);
    const defaultRestB = Math.hypot(p.x - bPos.x, p.y - bPos.y);
    const maxRest = Math.max(6, rb.r * 1.5);
    const restA = Math.min(maxRest, Math.max(0.8, finiteOr(Number(j.restA), defaultRestA)));
    const restB = Math.min(maxRest, Math.max(0.8, finiteOr(Number(j.restB), defaultRestB)));

    hybrid.push({
      rigidIndex,
      nodeIndex: globalSoft,
      vertexA: edgeA,
      vertexB: edgeB,
      restA,
      restB,
    });
  }

  return { rigid, soft, hybrid, softMembraneClusters };
}

function buildRigidEdgePermeabilityFromField(hull, field, width, height, threshold = 0.5) {
  if (!(field instanceof Float32Array) || !Number.isFinite(width) || !Number.isFinite(height)) {
    return Array.from({ length: Math.max(0, hull?.length || 0) }, () => [0, 0, 0]);
  }

  const th = clamp(Number(threshold) || 0.5, 0, 1);
  const sides = Math.max(0, hull?.length || 0);
  const out = [];
  for (let i = 0; i < sides; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % sides];
    const mx = ((Number(a?.x) || 0) + (Number(b?.x) || 0)) * 0.5;
    const my = ((Number(a?.y) || 0) + (Number(b?.y) || 0)) * 0.5;
    const v = clamp(sampleBilinearField(field, width, height, mx, my, 0), 0, 1);
    const pass = v >= th ? 1 : 0;
    out.push([pass, pass, pass]);
  }
  return out;
}

function buildRigidExportFromCompilerPieces(rigidPieces, nodes, options, width, height) {
  const rigidPermeabilityField = (options?.fields?.rigidPermeabilityMap instanceof Float32Array)
    ? options.fields.rigidPermeabilityMap
    : (Array.isArray(options?.fields?.rigidPermeabilityMap)
      ? Float32Array.from(options.fields.rigidPermeabilityMap)
      : null);
  const rigidPermeabilityThreshold = Number(options?.rigidPermeabilityThreshold);

  const grouped = new Map();
  for (let i = 0; i < rigidPieces.length; i++) {
    const rp = rigidPieces[i] || {};
    const hull = (rp.hull || [])
      .map((p) => ({ x: Number(p?.x) || 0, y: Number(p?.y) || 0 }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (hull.length < 3) continue;
    const compoundId = rp.compoundId || `compound_piece_${i}`;
    if (!grouped.has(compoundId)) grouped.set(compoundId, []);
    grouped.get(compoundId).push({
      hull,
      sourceNodeIds: Array.isArray(rp.sourceNodeIds)
        ? rp.sourceNodeIds.map((id) => Number(id)).filter((id) => Number.isInteger(id) && nodes[id])
        : [],
    });
  }

  const rigidBodies = [];
  const components = [];

  for (const [compoundId, pieces] of grouped.entries()) {
    const allPoints = [];
    const sourceNodeSet = new Set();
    const subHulls = [];

    for (const p of pieces) {
      subHulls.push(p.hull.map((v) => ({ x: v.x, y: v.y })));
      for (const v of p.hull) allPoints.push(v);
      for (const sid of p.sourceNodeIds) sourceNodeSet.add(sid);
    }

    // Preserve compiler-authored concavity when a compound already comes as a single contour piece.
    // Only fall back to convex merge when multiple disjoint sub-hulls must be combined.
    const outerHull = pieces.length === 1
      ? pieces[0].hull.map((v) => ({ x: v.x, y: v.y }))
      : convexHull(allPoints);
    if (outerHull.length < 3) continue;

    const c = centroid(outerHull);
    const r = Math.max(2.5, ...outerHull.map((p) => Math.hypot(p.x - c.x, p.y - c.y)));
    const sides = outerHull.length;

    const rigidIndex = rigidBodies.length;
    rigidBodies.push({
      id: `rigid_${rigidIndex}`,
      compoundId,
      hull: outerHull,
      subHulls,
      mass: finiteOr(Number(options.massHeavy), 5),
      edgeBodyMode: Array.from({ length: sides }, () => EDGE_BODY_BLOCK),
      edgeDyeMode: Array.from({ length: sides }, () => [...EDGE_DYE_DEFLECT_RGB]),
      edgePermeabilityRGB: buildRigidEdgePermeabilityFromField(
        outerHull,
        rigidPermeabilityField,
        width,
        height,
        rigidPermeabilityThreshold,
      ),
      digestEnabled: false,
      digestRGB: [1, 1, 1],
      consumeDyeRGB: [0, 0, 0],
    });

    components.push({
      index: rigidIndex,
      nodeIds: sourceNodeSet,
      radius: r,
    });
  }

  // Compound fusion path: no runtime inter-piece spring/weld constraints.
  return { rigidBodies, components };
}

function buildRigidExport(tris, nodes, options, width, height) {
  const rigidPermeabilityField = (options?.fields?.rigidPermeabilityMap instanceof Float32Array)
    ? options.fields.rigidPermeabilityMap
    : (Array.isArray(options?.fields?.rigidPermeabilityMap)
      ? Float32Array.from(options.fields.rigidPermeabilityMap)
      : null);
  const rigidPermeabilityThreshold = Number(options?.rigidPermeabilityThreshold);

  const comps = triangleComponents(tris);
  const rigidBodies = [];
  const components = [];

  for (let compIdx = 0; compIdx < comps.length; compIdx++) {
    const comp = comps[compIdx];

    const triPieces = [];
    for (const t of comp) {
      const idsRaw = [Number(t.a), Number(t.b), Number(t.c)];
      const ids = [...new Set(idsRaw)].filter((id) => Number.isInteger(id) && nodes[id]);
      if (ids.length < 3) continue;
      triPieces.push({
        nodeIds: new Set(ids),
        area: triangleAreaAbs(nodes[ids[0]], nodes[ids[1]], nodes[ids[2]]),
      });
    }

    const mergedPieces = mergeRigidPiecesConvex(triPieces, nodes);

    for (const piece of mergedPieces) {
      const hullWithIds = convexHullWithIds([...piece.nodeIds].map((id) => ({ id, x: nodes[id].x, y: nodes[id].y })));
      if (hullWithIds.length < 3) continue;
      const hull = hullWithIds.map((p) => ({ x: p.x, y: p.y }));
      const hullSourceIds = hullWithIds.map((p) => p.id);

      const c = centroid(hull);
      const r = Math.max(2.5, ...hull.map((p) => Math.hypot(p.x - c.x, p.y - c.y)));
      const sides = hull.length;

      const rigidIndex = rigidBodies.length;
      rigidBodies.push({
        id: `rigid_${rigidIndex}`,
        compoundId: `compound_${compIdx}`,
        hull,
        mass: finiteOr(Number(options.massHeavy), 5),
        edgeBodyMode: Array.from({ length: sides }, () => EDGE_BODY_BLOCK),
        edgeDyeMode: Array.from({ length: sides }, () => [...EDGE_DYE_DEFLECT_RGB]),
        edgePermeabilityRGB: buildRigidEdgePermeabilityFromField(
          hull,
          rigidPermeabilityField,
          width,
          height,
          rigidPermeabilityThreshold,
        ),
        digestEnabled: false,
        digestRGB: [1, 1, 1],
        consumeDyeRGB: [0, 0, 0],
      });

      components.push({
        index: rigidIndex,
        nodeIds: new Set(piece.nodeIds),
        radius: r,
      });

    }

  }

  return { rigidBodies, components };
}

function mergeRigidPiecesConvex(triPieces, nodes) {
  const pieces = triPieces.map((p) => ({ nodeIds: new Set(p.nodeIds), area: p.area }));
  let changed = true;
  while (changed) {
    changed = false;
    outer: for (let i = 0; i < pieces.length; i++) {
      for (let j = i + 1; j < pieces.length; j++) {
        const a = pieces[i];
        const b = pieces[j];
        const shared = [...a.nodeIds].filter((id) => b.nodeIds.has(id));
        if (shared.length < 2) continue;

        const mergedIds = new Set([...a.nodeIds, ...b.nodeIds]);
        const mergedHull = convexHull([...mergedIds].map((id) => ({ x: nodes[id].x, y: nodes[id].y })));
        const hullArea = polygonAreaAbs(mergedHull);
        const sumArea = a.area + b.area;
        const eps = Math.max(1e-4, sumArea * 0.02);
        if (Math.abs(hullArea - sumArea) <= eps) {
          pieces[i] = { nodeIds: mergedIds, area: sumArea };
          pieces.splice(j, 1);
          changed = true;
          break outer;
        }
      }
    }
  }
  return pieces;
}

function triangleAreaAbs(a, b, c) {
  return Math.abs((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) * 0.5);
}

function polygonAreaAbs(poly) {
  if (!poly || poly.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    s += p.x * q.y - q.x * p.y;
  }
  return Math.abs(s * 0.5);
}

function convexHullWithIds(pointsWithIds) {
  if (!pointsWithIds || pointsWithIds.length < 3) return pointsWithIds || [];
  const pts = [...pointsWithIds]
    .map((p) => ({ id: p.id, x: p.x, y: p.y }))
    .sort((a, b) => (a.x - b.x) || (a.y - b.y));

  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function buildBoundaryLoopsFromEdgeList(edges, nodeCount) {
  const adjacency = new Map();
  const unique = new Set();
  const edgeKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

  for (const e of (edges || [])) {
    const a = Number(e?.[0]);
    const b = Number(e?.[1]);
    if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
    const k = edgeKey(a, b);
    if (unique.has(k)) continue;
    unique.add(k);
    if (!adjacency.has(a)) adjacency.set(a, []);
    if (!adjacency.has(b)) adjacency.set(b, []);
    adjacency.get(a).push(b);
    adjacency.get(b).push(a);
  }

  for (const vs of adjacency.values()) vs.sort((a, b) => a - b);

  const orderedEdges = [...unique]
    .map((k) => k.split('-').map((x) => Number(x)))
    .sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));

  const used = new Set();
  const loops = [];

  for (const [startA, startB] of orderedEdges) {
    const startKey = edgeKey(startA, startB);
    if (used.has(startKey)) continue;

    const loop = [startA, startB];
    used.add(startKey);

    let prev = startA;
    let curr = startB;
    let closed = false;
    const guard = Math.max(16, nodeCount * 4 + 8);

    for (let iter = 0; iter < guard; iter++) {
      const neighbors = adjacency.get(curr) || [];
      let next = null;
      for (const n of neighbors) {
        if (n === prev) continue;
        const ek = edgeKey(curr, n);
        if (used.has(ek)) continue;
        next = n;
        break;
      }

      if (next == null) {
        const closeKey = edgeKey(curr, startA);
        if (curr !== startA && neighbors.includes(startA) && !used.has(closeKey)) {
          used.add(closeKey);
          closed = true;
        }
        break;
      }

      used.add(edgeKey(curr, next));
      if (next === startA) {
        closed = true;
        break;
      }

      loop.push(next);
      prev = curr;
      curr = next;
    }

    if (closed && loop.length >= 3) loops.push(loop);
  }

  return loops;
}

function collectMaskComponents(mask, width, height) {
  const seen = new Uint8Array(mask.length);
  const idx = (x, y) => y * width + x;
  const comps = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = idx(x, y);
      if (!mask[i] || seen[i]) continue;

      const stack = [[x, y]];
      seen[i] = 1;
      const cells = new Set();

      while (stack.length) {
        const [cx, cy] = stack.pop();
        cells.add(`${cx},${cy}`);
        const n4 = [[cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]];
        for (const [nx, ny] of n4) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const ni = idx(nx, ny);
          if (!mask[ni] || seen[ni]) continue;
          seen[ni] = 1;
          stack.push([nx, ny]);
        }
      }

      if (cells.size) comps.push(cells);
    }
  }

  comps.sort((a, b) => b.size - a.size);
  return comps;
}

function traceMaskComponentLoops(cellSet, step = 1) {
  const edgeMap = new Map();
  const addEdge = (x0, y0, x1, y1) => {
    const p0 = `${x0},${y0}`;
    const p1 = `${x1},${y1}`;
    if (!edgeMap.has(p0)) edgeMap.set(p0, []);
    edgeMap.get(p0).push({ from: p0, to: p1 });
  };

  const hasCell = (x, y) => cellSet.has(`${x},${y}`);
  for (const key of cellSet) {
    const [cx, cy] = key.split(',').map((v) => Number(v));
    const x0 = cx * step;
    const y0 = cy * step;
    const x1 = x0 + step;
    const y1 = y0 + step;

    if (!hasCell(cx, cy - 1)) addEdge(x0, y0, x1, y0);
    if (!hasCell(cx + 1, cy)) addEdge(x1, y0, x1, y1);
    if (!hasCell(cx, cy + 1)) addEdge(x1, y1, x0, y1);
    if (!hasCell(cx - 1, cy)) addEdge(x0, y1, x0, y0);
  }

  const loops = [];
  while (true) {
    const startEntry = [...edgeMap.entries()].find(([, arr]) => arr.length > 0);
    if (!startEntry) break;

    const [start] = startEntry;
    let current = start;
    const loop = [];
    const guard = edgeMap.size * 8 + 64;
    let steps = 0;

    while (steps++ < guard) {
      const outgoing = edgeMap.get(current);
      if (!outgoing || outgoing.length === 0) break;
      const edge = outgoing.pop();
      const [x, y] = edge.from.split(',').map((v) => Number(v));
      loop.push({ x, y });
      current = edge.to;
      if (current === start) break;
    }

    if (loop.length >= 3) loops.push(loop);
  }

  return loops;
}

function signedArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    s += a.x * b.y - b.x * a.y;
  }
  return s * 0.5;
}

function simplifyCollinearLoop(poly, eps = 1e-6) {
  if (!Array.isArray(poly) || poly.length < 3) return poly || [];
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[(i - 1 + poly.length) % poly.length];
    const b = poly[i];
    const c = poly[(i + 1) % poly.length];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(cross) <= eps) continue;
    out.push(b);
  }
  return out.length >= 3 ? out : poly;
}

function pointSegmentDistance(p, a, b) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const apx = p.x - a.x;
  const apy = p.y - a.y;
  const den = Math.max(1e-9, abx * abx + aby * aby);
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / den));
  const cx = a.x + abx * t;
  const cy = a.y + aby * t;
  return Math.hypot(p.x - cx, p.y - cy);
}

function simplifyDouglasPeuckerClosed(poly, epsilon = 0.8) {
  if (!Array.isArray(poly) || poly.length < 4) return poly || [];
  const open = [...poly, poly[0]];
  const keep = new Uint8Array(open.length);
  keep[0] = 1;
  keep[open.length - 1] = 1;

  const stack = [[0, open.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    let maxDist = -1;
    let maxIdx = -1;
    for (let i = a + 1; i < b; i++) {
      const d = pointSegmentDistance(open[i], open[a], open[b]);
      if (d > maxDist) {
        maxDist = d;
        maxIdx = i;
      }
    }
    if (maxDist > epsilon && maxIdx > a && maxIdx < b) {
      keep[maxIdx] = 1;
      stack.push([a, maxIdx], [maxIdx, b]);
    }
  }

  const out = [];
  for (let i = 0; i < open.length - 1; i++) {
    if (keep[i]) out.push(open[i]);
  }
  return out.length >= 3 ? out : poly;
}

function sampleBilinearField(field, width, height, x, y, fallback = 0.5) {
  if (!(field instanceof Float32Array) || field.length < width * height) return fallback;
  const cx = Math.max(0, Math.min(width - 1.001, Number(x) || 0));
  const cy = Math.max(0, Math.min(height - 1.001, Number(y) || 0));
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
  const sx = cx - x0, sy = cy - y0;
  const i00 = y0 * width + x0, i10 = y0 * width + x1, i01 = y1 * width + x0, i11 = y1 * width + x1;
  const v00 = Number(field[i00]);
  const v10 = Number(field[i10]);
  const v01 = Number(field[i01]);
  const v11 = Number(field[i11]);
  const a = (Number.isFinite(v00) ? v00 : fallback) * (1 - sx) + (Number.isFinite(v10) ? v10 : fallback) * sx;
  const b = (Number.isFinite(v01) ? v01 : fallback) * (1 - sx) + (Number.isFinite(v11) ? v11 : fallback) * sx;
  const out = a * (1 - sy) + b * sy;
  return Number.isFinite(out) ? out : fallback;
}

function sampleClosedArcPoint(poly, segLen, cumLen, perimeter, arcLen) {
  let s = Number(arcLen) || 0;
  s = ((s % perimeter) + perimeter) % perimeter;

  let edge = 0;
  while (edge + 1 < cumLen.length && cumLen[edge + 1] <= s) edge += 1;
  const a = poly[edge % poly.length];
  const b = poly[(edge + 1) % poly.length];
  const len = Math.max(1e-9, segLen[edge % poly.length]);
  const t = Math.max(0, Math.min(1, (s - cumLen[edge]) / len));
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  };
}

function resampleClosedLoopByEdgeMap({
  poly,
  width,
  height,
  edgeLengthField = null,
  minEdgeLength = MEMBRANE_DEFAULT_MIN_EDGE_LENGTH,
  maxEdgeLength = MEMBRANE_DEFAULT_MIN_EDGE_LENGTH * 2,
  maxNodes = MEMBRANE_DEFAULT_MAX_NODES,
}) {
  if (!Array.isArray(poly) || poly.length < 3) return poly || [];

  const minEdge = Math.max(0.75, Number(minEdgeLength) || MEMBRANE_DEFAULT_MIN_EDGE_LENGTH);
  const maxEdge = Math.max(minEdge, Number(maxEdgeLength) || (minEdge * 2));
  const nodeCap = Math.max(3, Math.round(Number(maxNodes) || MEMBRANE_DEFAULT_MAX_NODES));

  let perimeter = 0;
  const segLen = new Float64Array(poly.length);
  const cumLen = new Float64Array(poly.length + 1);
  cumLen[0] = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const len = Math.hypot((b.x || 0) - (a.x || 0), (b.y || 0) - (a.y || 0));
    segLen[i] = Math.max(1e-9, len);
    perimeter += segLen[i];
    cumLen[i + 1] = perimeter;
  }
  if (!Number.isFinite(perimeter) || perimeter <= 1e-6) return poly;

  const positions = [0];
  let s = 0;
  let guard = nodeCap * 6;
  while (positions.length < nodeCap && guard-- > 0) {
    const p = sampleClosedArcPoint(poly, segLen, cumLen, perimeter, s);
    const mapV = clamp(sampleBilinearField(edgeLengthField, width, height, p.x, p.y, 0.5), 0, 1);
    const target = minEdge + mapV * (maxEdge - minEdge);
    const step = Math.max(minEdge, Math.min(maxEdge, target));
    if ((s + step) >= (perimeter - minEdge * 0.35)) break;
    s += step;
    positions.push(s);
  }

  while (positions.length < nodeCap) {
    const tail = perimeter - positions[positions.length - 1];
    if (tail <= maxEdge * 1.05) break;
    positions.push(positions[positions.length - 1] + maxEdge);
  }

  if (positions.length < 3) {
    const fallbackN = Math.max(3, Math.min(nodeCap, Math.round(perimeter / Math.max(minEdge, 1))));
    const out = [];
    for (let i = 0; i < fallbackN; i++) {
      out.push(sampleClosedArcPoint(poly, segLen, cumLen, perimeter, (i * perimeter) / fallbackN));
    }
    return out;
  }

  return positions.map((arc) => sampleClosedArcPoint(poly, segLen, cumLen, perimeter, arc));
}

/**
 * Extract perimeter rings from a thresholded soft occupancy field.
 *
 * Rings are simplified and re-sampled with optional edge-length modulation,
 * producing deterministic membrane node loops for export.
 *
 * @param {object} params
 * @param {number} params.width
 * @param {number} params.height
 * @param {ArrayLike<number>} params.softField
 * @param {ArrayLike<number>|null} [params.edgeLengthField]
 * @param {number} [params.threshold]
 * @param {number} [params.minEdgeLength]
 * @param {number} [params.maxEdgeLength]
 * @param {number} [params.maxNodes]
 * @param {number} [params.simplifyEpsilon]
 * @returns {Array<Array<{x:number,y:number}>>}
 */
export function buildMembraneRingsFromSoftField({
  width,
  height,
  softField,
  edgeLengthField = null,
  threshold = 0.35,
  minEdgeLength = MEMBRANE_DEFAULT_MIN_EDGE_LENGTH,
  maxEdgeLength = 8.0,
  maxNodes = MEMBRANE_DEFAULT_MAX_NODES,
  simplifyEpsilon = MEMBRANE_DEFAULT_SIMPLIFY_EPS,
}) {
  const total = Math.max(1, width * height);
  const mask = new Uint8Array(total);
  const th = Math.max(0, Math.min(1, Number(threshold) || 0.35));
  for (let i = 0; i < total; i++) {
    mask[i] = (Number(softField?.[i]) || 0) >= th ? 1 : 0;
  }

  const comps = collectMaskComponents(mask, width, height);
  const rings = [];

  for (const comp of comps) {
    const loops = traceMaskComponentLoops(comp, 1);
    if (!loops.length) continue;

    let hull = loops[0];
    let bestArea = Math.abs(signedArea(hull));
    for (let i = 1; i < loops.length; i++) {
      const a = Math.abs(signedArea(loops[i]));
      if (a > bestArea) {
        bestArea = a;
        hull = loops[i];
      }
    }

    hull = simplifyCollinearLoop(hull, 1e-6);
    hull = simplifyDouglasPeuckerClosed(hull, Math.max(0, Number(simplifyEpsilon) || MEMBRANE_DEFAULT_SIMPLIFY_EPS));
    hull = simplifyCollinearLoop(hull, 1e-6);
    hull = resampleClosedLoopByEdgeMap({
      poly: hull,
      width,
      height,
      edgeLengthField,
      minEdgeLength,
      maxEdgeLength,
      maxNodes,
    });
    hull = simplifyCollinearLoop(hull, 1e-6);

    if (!Array.isArray(hull) || hull.length < 3) continue;
    if (signedArea(hull) < 0) hull = [...hull].reverse();
    rings.push(hull);
  }

  return rings;
}

function buildSoftMembraneExportFromField({ width, height, softField, edgeLengthField, shapeMemoryField, threshold, options }) {
  const rings = buildMembraneRingsFromSoftField({
    width,
    height,
    softField,
    edgeLengthField,
    threshold,
    minEdgeLength: Number(options?.membraneMinEdgeLength) || MEMBRANE_DEFAULT_MIN_EDGE_LENGTH,
    maxEdgeLength: Math.max(
      Number(options?.membraneMinEdgeLength) || MEMBRANE_DEFAULT_MIN_EDGE_LENGTH,
      Number(options?.membraneMaxEdgeLength) || 8.0,
    ),
    maxNodes: Number(options?.membraneMaxNodes) || MEMBRANE_DEFAULT_MAX_NODES,
    simplifyEpsilon: Number(options?.membraneSimplifyEpsilon) || MEMBRANE_DEFAULT_SIMPLIFY_EPS,
  });

  const softBodies = [];
  const components = [];

  for (const hull of rings) {
    const softNodes = hull.map((p) => ({
      x: p.x,
      y: p.y,
      mass: finiteOr(Number(options.massSoft), 0.6),
      r: 1.6,
      digestEnabled: false,
      digestRGB: [1, 1, 1],
      shapeMemoryWeight: clamp(sampleBilinearField(shapeMemoryField, width, height, p.x, p.y, 1), 0, 1),
    }));

    const springs = [];
    for (let i = 0; i < softNodes.length; i++) {
      const a = softNodes[i];
      const b = softNodes[(i + 1) % softNodes.length];
      springs.push([
        i,
        (i + 1) % softNodes.length,
        Math.max(1e-3, Math.hypot((b.x || 0) - (a.x || 0), (b.y || 0) - (a.y || 0))),
        EDGE_BODY_BLOCK,
        [...EDGE_DYE_DEFLECT_RGB],
      ]);
    }

    const bodyIndex = softBodies.length;
    softBodies.push({
      id: `soft_${bodyIndex}`,
      solverMode: 'membrane',
      restArea: Math.max(1e-4, polygonAreaAbs(softNodes)),
      pressureGain: Math.max(0.001, Number(options?.membranePressureGain) || MEMBRANE_DEFAULT_PRESSURE_GAIN),
      radialDamping: Math.max(0, Math.min(0.2, Number(options?.membraneRadialDamping) || MEMBRANE_DEFAULT_RADIAL_DAMPING)),
      shapeMemoryGain: Math.max(0, Math.min(0.35, Number(options?.membraneShapeMemoryGain) || MEMBRANE_DEFAULT_SHAPE_MEMORY_GAIN)),
      insideCorrectionEnabled: 1,
      nodes: softNodes,
      springs,
    });

    components.push({
      index: bodyIndex,
      nodeIds: new Set(),
      sourceToLocal: new Map(),
    });
  }

  return { softBodies, components };
}

function buildSoftExport(tris, nodes, options, softCrossBeams = []) {
  const enableBoundaryRing = options?.softBoundaryRingSprings !== false;
  const enableSeamWeldSprings = options?.softSeamWeldSprings !== false;
  const seamAxisEps = Math.max(1e-9, Number(options?.softSeamAxisEpsilon) || 1e-6);
  const softSolverModeDefault = normalizeSoftSolverMode(options?.softSolverMode);
  const softSolverModesByComponent = options?.softSolverModesByComponent || null;
  const boundaryRingStrideRaw = Math.max(2, Math.round(Number(options?.softBoundaryRingStride) || 2));
  const boundaryRingMaxSpanFactor = Math.max(1.5, Number(options?.softBoundaryRingMaxSpanFactor) || 2.75);
  const crossBeamMaxSpanFactor = Math.max(1.25, Number(options?.softCrossBeamMaxSpanFactor) || 2.75);
  const comps = triangleComponents(tris);
  const softBodies = [];
  const components = [];

  for (const comp of comps) {
    const pointIds = new Set();
    for (const t of comp) {
      pointIds.add(t.a); pointIds.add(t.b); pointIds.add(t.c);
    }
    const ids = [...pointIds];
    const remap = new Map();

    const softNodes = ids.map((id, i) => {
      remap.set(id, i);
      const p = nodes[id];
      return {
        x: p.x,
        y: p.y,
        mass: finiteOr(Number(options.massSoft), 0.6),
        r: 1.6,
        digestEnabled: false,
        digestRGB: [1, 1, 1],
      };
    });

    const edgeCount = new Map();
    const edgeOpposites = new Map();
    for (const t of comp) {
      const edges = [
        [t.a, t.b, t.c],
        [t.b, t.c, t.a],
        [t.c, t.a, t.b],
      ];
      for (const [u, v, opp] of edges) {
        const k = u < v ? `${u}-${v}` : `${v}-${u}`;
        edgeCount.set(k, (edgeCount.get(k) || 0) + 1);
        if (!edgeOpposites.has(k)) edgeOpposites.set(k, []);
        edgeOpposites.get(k).push(opp);
      }
    }

    const springs = [];
    const springSet = new Set();
    const springKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);
    const boundaryEdges = [];

    for (const [k, c] of edgeCount.entries()) {
      const [ua, ub] = k.split('-').map((x) => Number(x));
      const a = remap.get(ua);
      const b = remap.get(ub);
      if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
      const pa = softNodes[a];
      const pb = softNodes[b];
      if (!pa || !pb) continue;
      const rest = Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y));
      const boundary = c === 1;
      springs.push([
        a,
        b,
        rest,
        boundary ? EDGE_BODY_BLOCK : 0,
        boundary ? [...EDGE_DYE_DEFLECT_RGB] : [0, 0, 0],
      ]);
      if (boundary) boundaryEdges.push([a, b]);
      springSet.add(springKey(a, b));
    }

    const boundaryLoops = buildBoundaryLoopsFromEdgeList(boundaryEdges, softNodes.length);

    if (enableSeamWeldSprings) {
      for (const [k, c] of edgeCount.entries()) {
        if (c !== 2) continue;
        const [ua, ub] = k.split('-').map((x) => Number(x));
        const pu = nodes?.[ua];
        const pv = nodes?.[ub];
        if (!pu || !pv) continue;

        const axisAligned = Math.abs((pu.x || 0) - (pv.x || 0)) <= seamAxisEps
          || Math.abs((pu.y || 0) - (pv.y || 0)) <= seamAxisEps;
        if (!axisAligned) continue;

        const oppRaw = edgeOpposites.get(k) || [];
        const oppUnique = [...new Set(oppRaw.filter((x) => Number.isInteger(Number(x))).map((x) => Number(x)))].sort((a, b) => a - b);
        if (oppUnique.length < 2) continue;

        const a = remap.get(oppUnique[0]);
        const b = remap.get(oppUnique[1]);
        if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;

        const springId = springKey(a, b);
        if (springSet.has(springId)) continue;

        const pa = softNodes[a];
        const pb = softNodes[b];
        if (!pa || !pb) continue;

        springs.push([
          a,
          b,
          Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y)),
          0,
          [0, 0, 0],
        ]);
        springSet.add(springId);
      }
    }

    const structuralRest = springs.map((sp) => Number(sp?.[2])).filter((v) => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
    const medianStructuralRest = structuralRest.length
      ? structuralRest[Math.floor(structuralRest.length * 0.5)]
      : 1;
    const maxCrossBeamRest = Math.max(1e-3, medianStructuralRest * crossBeamMaxSpanFactor);

    // Double cross-beam square reinforcement for soft lattice, span-bounded to avoid long-range shear overconstraint.
    for (const beam of (softCrossBeams || [])) {
      const ua = Number(beam?.[0]);
      const ub = Number(beam?.[1]);
      const a = remap.get(ua);
      const b = remap.get(ub);
      if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
      const key = springKey(a, b);
      if (springSet.has(key)) continue;
      const pa = softNodes[a];
      const pb = softNodes[b];
      if (!pa || !pb) continue;
      const rest = Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y));
      if (rest > maxCrossBeamRest + 1e-6) continue;
      springs.push([
        a,
        b,
        rest,
        0,
        [0, 0, 0],
      ]);
      springSet.add(key);
    }

    if (enableBoundaryRing) {
      const loops = boundaryLoops;
      const boundaryEdgeRestByKey = new Map();
      for (const [a, b] of boundaryEdges) {
        if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
        const pa = softNodes[a];
        const pb = softNodes[b];
        if (!pa || !pb) continue;
        boundaryEdgeRestByKey.set(springKey(a, b), Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y)));
      }

      for (const loop of loops) {
        if (!Array.isArray(loop) || loop.length < 4) continue;
        const stride = Math.max(2, Math.min(Math.floor(loop.length / 2), boundaryRingStrideRaw));
        if (!Number.isInteger(stride) || stride < 2) continue;

        const loopEdgeRest = [];
        for (let i = 0; i < loop.length; i++) {
          const eKey = springKey(loop[i], loop[(i + 1) % loop.length]);
          const rest = Number(boundaryEdgeRestByKey.get(eKey));
          if (Number.isFinite(rest) && rest > 0) loopEdgeRest.push(rest);
        }
        loopEdgeRest.sort((a, b) => a - b);
        const medianEdgeRest = loopEdgeRest.length
          ? loopEdgeRest[Math.floor(loopEdgeRest.length * 0.5)]
          : 1;
        const maxAllowedRingRest = Math.max(1e-3, medianEdgeRest * boundaryRingMaxSpanFactor);

        for (let i = 0; i < loop.length; i++) {
          const a = loop[i];
          const b = loop[(i + stride) % loop.length];
          if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;

          const key = springKey(a, b);
          if (springSet.has(key)) continue;

          const pa = softNodes[a];
          const pb = softNodes[b];
          if (!pa || !pb) continue;

          const rest = Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y));
          if (rest > maxAllowedRingRest + 1e-6) continue;

          springs.push([
            a,
            b,
            rest,
            0,
            [0, 0, 0],
          ]);
          springSet.add(key);
        }
      }
    }

    const bodyIndex = softBodies.length;
    const modeRaw = Array.isArray(softSolverModesByComponent)
      ? softSolverModesByComponent[bodyIndex]
      : (softSolverModesByComponent && typeof softSolverModesByComponent === 'object'
        ? softSolverModesByComponent[bodyIndex]
        : softSolverModeDefault);
    const solverMode = normalizeSoftSolverMode(modeRaw);

    let exportNodes = softNodes;
    let exportSprings = springs;
    let exportNodeIds = pointIds;
    let exportSourceToLocal = remap;
    let membraneRestArea = null;

    if (solverMode === 'membrane') {
      const majorLoop = [...boundaryLoops].sort((a, b) => (b?.length || 0) - (a?.length || 0))[0] || null;
      let ringBuilt = false;

      if (Array.isArray(majorLoop) && majorLoop.length >= 3) {
        const oldToNew = new Map();
        const membraneNodes = [];
        const membraneNodeIds = new Set();
        const membraneSourceToLocal = new Map();

        for (const oldIdxRaw of majorLoop) {
          const oldIdx = Number(oldIdxRaw);
          if (!Number.isInteger(oldIdx) || oldToNew.has(oldIdx)) continue;
          const srcId = ids[oldIdx];
          const srcNode = softNodes[oldIdx];
          if (!srcNode || !Number.isInteger(srcId)) continue;
          const newIdx = membraneNodes.length;
          oldToNew.set(oldIdx, newIdx);
          membraneNodes.push({ ...srcNode });
          membraneNodeIds.add(srcId);
          membraneSourceToLocal.set(srcId, newIdx);
        }

        const membraneSprings = [];
        if (membraneNodes.length >= 3) {
          for (let i = 0; i < majorLoop.length; i++) {
            const aOld = Number(majorLoop[i]);
            const bOld = Number(majorLoop[(i + 1) % majorLoop.length]);
            const a = oldToNew.get(aOld);
            const b = oldToNew.get(bOld);
            if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
            const pa = membraneNodes[a];
            const pb = membraneNodes[b];
            membraneSprings.push([
              a,
              b,
              Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y)),
              EDGE_BODY_BLOCK,
              [...EDGE_DYE_DEFLECT_RGB],
            ]);
          }
        }

        if (membraneNodes.length >= 3 && membraneSprings.length >= 3) {
          membraneRestArea = Math.max(1e-4, polygonAreaAbs(membraneNodes));
          exportNodes = membraneNodes;
          exportSprings = membraneSprings;
          exportNodeIds = membraneNodeIds;
          exportSourceToLocal = membraneSourceToLocal;
          ringBuilt = true;
        }
      }

      // Guardrail: membrane mode must remain perimeter-only even for malformed/non-manifold
      // triangle soups where boundary loop extraction fails.
      if (!ringBuilt && softNodes.length >= 3) {
        const hull = convexHullWithIds(softNodes.map((p, idx) => ({ id: idx, x: p.x, y: p.y })));
        if (Array.isArray(hull) && hull.length >= 3) {
          const membraneNodes = [];
          const membraneNodeIds = new Set();
          const membraneSourceToLocal = new Map();
          const oldToNew = new Map();

          for (const hp of hull) {
            const oldIdx = Number(hp?.id);
            if (!Number.isInteger(oldIdx) || oldToNew.has(oldIdx)) continue;
            const srcId = ids[oldIdx];
            const srcNode = softNodes[oldIdx];
            if (!srcNode || !Number.isInteger(srcId)) continue;
            const newIdx = membraneNodes.length;
            oldToNew.set(oldIdx, newIdx);
            membraneNodes.push({ ...srcNode });
            membraneNodeIds.add(srcId);
            membraneSourceToLocal.set(srcId, newIdx);
          }

          const membraneSprings = [];
          for (let i = 0; i < membraneNodes.length; i++) {
            const a = i;
            const b = (i + 1) % membraneNodes.length;
            if (a === b) continue;
            const pa = membraneNodes[a];
            const pb = membraneNodes[b];
            membraneSprings.push([
              a,
              b,
              Math.max(1e-3, Math.hypot(pb.x - pa.x, pb.y - pa.y)),
              EDGE_BODY_BLOCK,
              [...EDGE_DYE_DEFLECT_RGB],
            ]);
          }

          if (membraneNodes.length >= 3 && membraneSprings.length >= 3) {
            membraneRestArea = Math.max(1e-4, polygonAreaAbs(membraneNodes));
            exportNodes = membraneNodes;
            exportSprings = membraneSprings;
            exportNodeIds = membraneNodeIds;
            exportSourceToLocal = membraneSourceToLocal;
          }
        }
      }

      if (!Number.isFinite(membraneRestArea) || membraneRestArea <= 0) {
        membraneRestArea = Math.max(1e-4, polygonAreaAbs(exportNodes));
      }
    }

    const softBody = {
      id: `soft_${bodyIndex}`,
      solverMode,
      nodes: exportNodes,
      springs: exportSprings,
    };
    if (solverMode === 'membrane') {
      softBody.restArea = Math.max(1e-4, Number(options?.membraneRestArea) || membraneRestArea || 1);
      softBody.pressureGain = Math.max(0.001, Number(options?.membranePressureGain) || MEMBRANE_DEFAULT_PRESSURE_GAIN);
      softBody.radialDamping = Math.max(0, Math.min(0.2, Number(options?.membraneRadialDamping) || MEMBRANE_DEFAULT_RADIAL_DAMPING));
      softBody.shapeMemoryGain = Math.max(0, Math.min(0.35, Number(options?.membraneShapeMemoryGain) || MEMBRANE_DEFAULT_SHAPE_MEMORY_GAIN));
      const membraneInsideRaw = Number(options?.membraneInsideCorrectionEnabled);
      softBody.insideCorrectionEnabled = Number.isFinite(membraneInsideRaw)
        ? (membraneInsideRaw > 0 ? 1 : 0)
        : 1;
    }

    softBodies.push(softBody);

    components.push({
      index: bodyIndex,
      nodeIds: exportNodeIds,
      sourceToLocal: exportSourceToLocal,
    });
  }

  return { softBodies, components };
}

function buildHybridExport({ rigidComps, softComps, rigidBodies, sourceNodes }) {
  const links = [];
  const used = new Set();

  for (const rc of rigidComps) {
    for (const sc of softComps) {
      const shared = [];
      for (const nid of rc.nodeIds) if (sc.nodeIds.has(nid)) shared.push(nid);
      for (const nid of shared) {
        const softNodeLocal = sc.sourceToLocal.get(nid);
        if (softNodeLocal == null) continue;

        const key = `${rc.index}:${sc.index}:${softNodeLocal}`;
        if (used.has(key)) continue;
        used.add(key);

        const p = sourceNodes[nid];
        const rb = rigidBodies[rc.index];
        if (!p || !rb || !rb.hull?.length) continue;

        const { vA, vB } = nearestHullEdgeForPoint(rb.hull, p.x, p.y);
        const aPos = rb.hull[vA];
        const bPos = rb.hull[vB];
        const maxRest = Math.max(6, rc.radius * 0.55);

        links.push({
          rigidBodyIndex: rc.index,
          softBodyIndex: sc.index,
          softNodeIndex: softNodeLocal,
          edgeA: vA,
          edgeB: vB,
          restA: Math.min(maxRest, Math.max(0.8, Math.hypot(p.x - aPos.x, p.y - aPos.y))),
          restB: Math.min(maxRest, Math.max(0.8, Math.hypot(p.x - bPos.x, p.y - bPos.y))),
        });
      }
    }
  }

  return links;
}

function centroid(points) {
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / Math.max(1, points.length), y: sy / Math.max(1, points.length) };
}

function rigidVerticesWorld(rb) {
  if (Array.isArray(rb.verticesLocal) && rb.verticesLocal.length >= 3) {
    const th = rb.theta || 0;
    const c = Math.cos(th), s = Math.sin(th);
    return rb.verticesLocal.map((v) => ({
      x: rb.x + v.x * c - v.y * s,
      y: rb.y + v.x * s + v.y * c,
    }));
  }
  const sides = Math.max(3, rb.sides || 3);
  const rot = (rb.theta || 0) + (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
  const verts = [];
  for (let i = 0; i < sides; i++) {
    const a = rot + (i / sides) * Math.PI * 2;
    verts.push({ x: rb.x + Math.cos(a) * rb.r, y: rb.y + Math.sin(a) * rb.r });
  }
  return verts;
}

function rigidVertexWorld(rb, vi) {
  const verts = rigidVerticesWorld(rb);
  const n = verts.length || 1;
  return verts[((vi % n) + n) % n];
}

function nearestHullEdgeForPoint(hull, px, py) {
  if (!hull?.length) return { vA: 0, vB: 0 };
  if (hull.length === 1) return { vA: 0, vB: 0 };

  const EDGE_EPS2 = 1e-8;

  let nearestVertex = 0;
  let nearestVertexD2 = Number.POSITIVE_INFINITY;
  for (let i = 0; i < hull.length; i++) {
    const dx = px - hull[i].x;
    const dy = py - hull[i].y;
    const d2 = dx * dx + dy * dy;
    if (d2 < nearestVertexD2) {
      nearestVertexD2 = d2;
      nearestVertex = i;
    }
  }

  // Deterministic tie-break for shared nodes that land directly on a rigid vertex.
  // Prefer a local non-degenerate incident edge when possible.
  if (nearestVertexD2 <= 1e-6) {
    const n = hull.length;
    const prev = (nearestVertex - 1 + n) % n;
    const next = (nearestVertex + 1) % n;
    const prevLen2 = (hull[nearestVertex].x - hull[prev].x) ** 2 + (hull[nearestVertex].y - hull[prev].y) ** 2;
    const nextLen2 = (hull[nearestVertex].x - hull[next].x) ** 2 + (hull[nearestVertex].y - hull[next].y) ** 2;

    const prevValid = prevLen2 > EDGE_EPS2;
    const nextValid = nextLen2 > EDGE_EPS2;
    if (prevValid && nextValid) {
      return nextLen2 <= prevLen2
        ? { vA: nearestVertex, vB: next }
        : { vA: prev, vB: nearestVertex };
    }
    if (nextValid) return { vA: nearestVertex, vB: next };
    if (prevValid) return { vA: prev, vB: nearestVertex };
  }

  let bestIdx = 0;
  let bestD2 = Number.POSITIVE_INFINITY;
  let fallbackIdx = 0;
  let fallbackD2 = Number.POSITIVE_INFINITY;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const len2 = abx * abx + aby * aby;
    const apx = px - a.x;
    const apy = py - a.y;
    const denom = Math.max(1e-6, len2);
    const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / denom));
    const cx = a.x + abx * t;
    const cy = a.y + aby * t;
    const dx = px - cx;
    const dy = py - cy;
    const d2 = dx * dx + dy * dy;

    if (d2 < fallbackD2) {
      fallbackD2 = d2;
      fallbackIdx = i;
    }
    if (len2 <= EDGE_EPS2) continue;
    if (d2 < bestD2) {
      bestD2 = d2;
      bestIdx = i;
    }
  }

  const idx = Number.isFinite(bestD2) ? bestIdx : fallbackIdx;
  return { vA: idx, vB: (idx + 1) % hull.length };
}

function triangleComponents(tris) {
  if (!tris.length) return [];
  const nodeToTri = new Map();
  for (let i = 0; i < tris.length; i++) {
    const t = tris[i];
    for (const n of [t.a, t.b, t.c]) {
      if (!nodeToTri.has(n)) nodeToTri.set(n, []);
      nodeToTri.get(n).push(i);
    }
  }

  const seen = new Uint8Array(tris.length);
  const comps = [];
  for (let i = 0; i < tris.length; i++) {
    if (seen[i]) continue;
    const stack = [i];
    seen[i] = 1;
    const comp = [];
    while (stack.length) {
      const ti = stack.pop();
      comp.push(tris[ti]);
      for (const n of [tris[ti].a, tris[ti].b, tris[ti].c]) {
        for (const ni of nodeToTri.get(n) || []) {
          if (!seen[ni]) {
            seen[ni] = 1;
            stack.push(ni);
          }
        }
      }
    }
    comps.push(comp);
  }
  return comps;
}

function convexHull(points) {
  if (!points || points.length < 3) return points || [];
  const pts = [...points]
    .map((p) => ({ x: p.x, y: p.y }))
    .sort((a, b) => (a.x - b.x) || (a.y - b.y));

  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return [...lower, ...upper];
}

/**
 * Normalize rigid per-edge body modes to full edge count.
 *
 * Sparse/missing entries are backfilled as blocking (1), which preserves the
 * strict-collision default when evolving edge arrays incrementally.
 */
function normalizeEdgeBodyModeList(list, count) {
  if (!Array.isArray(list) || !list.length) return Array.from({ length: count }, () => EDGE_BODY_BLOCK);
  return Array.from({ length: count }, (_, i) => {
    const v = list[i];
    return v === undefined ? EDGE_BODY_BLOCK : normalizeEdgeBodyMode(v);
  });
}

/**
 * Normalize rigid per-edge dye modes to full edge count.
 *
 * Sparse/missing entries default to DEFLECT for all channels, matching solver
 * safety expectations for unspecified edges.
 */
function normalizeEdgeDyeModeList(list, count) {
  if (!Array.isArray(list) || !list.length) return Array.from({ length: count }, () => [...EDGE_DYE_DEFLECT_RGB]);
  return Array.from({ length: count }, (_, i) => {
    const v = list[i];
    return v === undefined ? [...EDGE_DYE_DEFLECT_RGB] : normalizeEdgeDyeMode(v);
  });
}

/**
 * Normalize rigid per-edge permeability masks to full edge count.
 *
 * Sparse/missing entries default to fully blocked RGB [0,0,0], so transport
 * mutators can safely add permeability edges without implicit leaks.
 */
function normalizeEdgePermeabilityList(list, count) {
  if (!Array.isArray(list) || !list.length) return Array.from({ length: count }, () => [0, 0, 0]);
  return Array.from({ length: count }, (_, i) => {
    const v = list[i];
    return v === undefined ? [0, 0, 0] : normalizeBinaryRGB(v);
  });
}

/**
 * Normalize one edge body mode into binary collision semantics.
 * Only explicit `1` remains blocking; all other values become pass-through.
 * @param {unknown} v
 * @returns {0|1}
 */
function normalizeEdgeBodyMode(v) {
  return Number(v) === EDGE_BODY_BLOCK ? EDGE_BODY_BLOCK : 0;
}

/**
 * Normalize one edge dye mode into strict RGB channel enums.
 * Missing/invalid values fall back to DEFLECT per channel for safety.
 * @param {unknown} v
 * @returns {[number, number, number]}
 */
function normalizeEdgeDyeMode(v) {
  if (Array.isArray(v) && v.length >= 3) {
    return [
      normalizeEdgeDyeModeChannel(v[0]),
      normalizeEdgeDyeModeChannel(v[1]),
      normalizeEdgeDyeModeChannel(v[2]),
    ];
  }
  return [...EDGE_DYE_DEFLECT_RGB];
}

/**
 * Normalize one dye channel into the strict enum used by edge dye traits.
 *
 * @param {unknown} v
 * @returns {0|1|2}
 */
function normalizeEdgeDyeModeChannel(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return EDGE_DYE_DEFLECT;
  if (n === EDGE_DYE_PASS || n === EDGE_DYE_DEFLECT || n === EDGE_DYE_ABSORB) return n;
  return EDGE_DYE_DEFLECT;
}

/**
 * Normalize RGB-like payloads into binary permeability semantics.
 *
 * Any channel strictly greater than zero becomes permeable (1); all others are
 * treated as blocked (0). Non-array/short payloads deterministically collapse
 * to fully blocked `[0,0,0]`.
 *
 * @param {unknown} v
 * @returns {[0|1,0|1,0|1]}
 */
function normalizeBinaryRGB(v) {
  if (!Array.isArray(v) || v.length < 3) return [0, 0, 0];
  return [
    Number(v[0]) > 0 ? 1 : 0,
    Number(v[1]) > 0 ? 1 : 0,
    Number(v[2]) > 0 ? 1 : 0,
  ];
}

function normalizeRGB(v) {
  if (!Array.isArray(v) || v.length < 3) return [1, 1, 1];
  return [finiteOr(Number(v[0]), 1), finiteOr(Number(v[1]), 1), finiteOr(Number(v[2]), 1)];
}

function finiteOr(v, fallback) {
  return Number.isFinite(v) ? v : fallback;
}
