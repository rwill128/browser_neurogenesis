import { compileFieldToMesh } from '/field-to-structure-core.js';
import { createCreatureSpecFromMesh, parseCreatureSpec, buildMembraneRingsFromSoftField } from '/creature-spec.js';

const paintCanvas = document.getElementById('paint');
const densityCanvas = document.getElementById('densityPaint');
const membraneEdgeCanvas = document.getElementById('membraneEdgePaint');
const membraneShapeCanvas = document.getElementById('membraneShapePaint');
const softPermeabilityCanvas = document.getElementById('softPermeabilityPaint');
const rigidPermeabilityCanvas = document.getElementById('rigidPermeabilityPaint');
const rigidConsumeCanvas = document.getElementById('rigidConsumePaint');
const meshCanvas = document.getElementById('mesh');
const windTunnelFrame = document.getElementById('windTunnelFrame');
const pctx = paintCanvas.getContext('2d');
const dctx = densityCanvas.getContext('2d');
const ectx = membraneEdgeCanvas.getContext('2d');
const sctx = membraneShapeCanvas.getContext('2d');
const spctx = softPermeabilityCanvas.getContext('2d');
const prctx = rigidPermeabilityCanvas.getContext('2d');
const cctx = rigidConsumeCanvas.getContext('2d');
const mctx = meshCanvas.getContext('2d');
const modeEl = document.getElementById('paintMode');
const fieldPaintTargetEl = document.getElementById('fieldPaintTarget');
const brushEl = document.getElementById('brush');
const thresholdEl = document.getElementById('threshold');
const softDensityPaintEl = document.getElementById('softDensityPaint');
const membraneEdgePaintEl = document.getElementById('membraneEdgePaintValue');
const membraneShapePaintEl = document.getElementById('membraneShapePaintValue');
const softPermeabilityPaintEl = document.getElementById('softPermeabilityPaintValue');
const rigidPermeabilityPaintEl = document.getElementById('rigidPermeabilityPaintValue');
const rigidConsumeChannelEl = document.getElementById('rigidConsumeChannel');
const rigidConsumePaintEl = document.getElementById('rigidConsumePaintValue');
const rigidCompileModeEl = document.getElementById('rigidCompileMode');
const rigidPrimitiveSideMinEl = document.getElementById('rigidPrimitiveSideMin');
const rigidPrimitiveSideMaxEl = document.getElementById('rigidPrimitiveSideMax');
const softInfillModeEl = document.getElementById('softInfillMode');
const softMinCellSizeEl = document.getElementById('softMinCellSize');
const softBoundaryRingEl = document.getElementById('softBoundaryRing');
const softSolverModeEl = document.getElementById('softSolverMode');
const membraneMinEdgeLengthEl = document.getElementById('membraneMinEdgeLength');
const membraneMaxEdgeLengthEl = document.getElementById('membraneMaxEdgeLength');
const clearBtn = document.getElementById('clearBtn');
const compileBtn = document.getElementById('compileBtn');
const randomizeBtn = document.getElementById('randomizeBtn');
const randomizeTraitBtn = document.getElementById('randomizeTraitBtn');
const randomizeDensityBtn = document.getElementById('randomizeDensityBtn');
const randomizeMembraneEdgeBtn = document.getElementById('randomizeMembraneEdgeBtn');
const randomizeMembraneShapeBtn = document.getElementById('randomizeMembraneShapeBtn');
const randomizeSoftPermeabilityBtn = document.getElementById('randomizeSoftPermeabilityBtn');
const randomizeRigidPermeabilityBtn = document.getElementById('randomizeRigidPermeabilityBtn');
const randomizeRigidConsumeBtn = document.getElementById('randomizeRigidConsumeBtn');
const randomFieldPresetEl = document.getElementById('randomFieldPreset');
const randomFieldSeedEl = document.getElementById('randomFieldSeed');
const exportBtn = document.getElementById('exportBtn');
const importBtn = document.getElementById('importBtn');
const importFile = document.getElementById('importFile');
const out = document.getElementById('out');
const windEmitterStrengthEl = document.getElementById('windEmitterStrength');
const windEmitterJetVyEl = document.getElementById('windEmitterJetVy');
const windEmitterRadiusEl = document.getElementById('windEmitterRadius');
const windEmitterYFractionEl = document.getElementById('windEmitterYFraction');
const windEmitterSpinEl = document.getElementById('windEmitterSpin');
const windEmitterCurlGainEl = document.getElementById('windEmitterCurlGain');
const windEmitterDriftGainEl = document.getElementById('windEmitterDriftGain');
const windEmitterChaosGainEl = document.getElementById('windEmitterChaosGain');
const windEmitterWobbleAmpEl = document.getElementById('windEmitterWobbleAmp');
const windEmitterWobbleFreqEl = document.getElementById('windEmitterWobbleFreq');
const windEmitterLockPositionEl = document.getElementById('windEmitterLockPosition');
const applyWindEmitterBtn = document.getElementById('applyWindEmitterBtn');

const W = 128, H = 128;
const rigid = new Float32Array(W * H);
const soft = new Float32Array(W * H);
const softDensity = new Float32Array(W * H).fill(0.5);
const membraneEdgeMap = new Float32Array(W * H).fill(0.5);
const membraneShapeMap = new Float32Array(W * H).fill(1.0);
const softPermeabilityMap = new Float32Array(W * H).fill(0.0);
const rigidPermeabilityMap = new Float32Array(W * H).fill(0.0);
const rigidEdgeConsumeMapR = new Float32Array(W * H).fill(0.0);
const rigidEdgeConsumeMapG = new Float32Array(W * H).fill(0.0);
const rigidEdgeConsumeMapB = new Float32Array(W * H).fill(0.0);
let lastMesh = null;
let lastCompiledSpec = null;
let lastCompiledFields = null;
let compileRevision = 0;
let windTunnelReady = false;

const fieldPanels = Array.from(document.querySelectorAll('.field-panel'));

function idx(x, y) { return y * W + x; }

function isolateLargestContiguousTraitBody(rigidField, softField, threshold = 0.35) {
  const total = W * H;
  const th = Math.max(0, Math.min(1, Number(threshold) || 0.35));
  const occupied = new Uint8Array(total);

  for (let i = 0; i < total; i++) {
    const rv = Number(rigidField?.[i]) || 0;
    const sv = Number(softField?.[i]) || 0;
    if (Math.max(rv, sv) >= th) occupied[i] = 1;
  }

  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  let bestStart = -1;
  let bestCells = 0;
  let bestWeightedArea = -1;
  let componentCount = 0;

  const tryComponent = (start) => {
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    visited[start] = 1;

    let cells = 0;
    let weightedArea = 0;
    while (head < tail) {
      const i = queue[head++];
      cells += 1;
      const rv = Number(rigidField?.[i]) || 0;
      const sv = Number(softField?.[i]) || 0;
      weightedArea += Math.max(rv, sv);

      const x = i % W;
      const y = (i / W) | 0;
      if (x > 0) {
        const ni = i - 1;
        if (occupied[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni; }
      }
      if (x + 1 < W) {
        const ni = i + 1;
        if (occupied[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni; }
      }
      if (y > 0) {
        const ni = i - W;
        if (occupied[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni; }
      }
      if (y + 1 < H) {
        const ni = i + W;
        if (occupied[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni; }
      }
    }

    return { cells, weightedArea };
  };

  for (let i = 0; i < total; i++) {
    if (!occupied[i] || visited[i]) continue;
    componentCount += 1;
    const { cells, weightedArea } = tryComponent(i);
    if (cells > bestCells || (cells === bestCells && weightedArea > bestWeightedArea)) {
      bestCells = cells;
      bestWeightedArea = weightedArea;
      bestStart = i;
    }
  }

  if (bestStart < 0) {
    return {
      rigidField: rigidField instanceof Float32Array ? rigidField : Float32Array.from(rigidField || []),
      softField: softField instanceof Float32Array ? softField : Float32Array.from(softField || []),
      keptCells: 0,
      removedCells: 0,
      componentCount,
    };
  }

  const keep = new Uint8Array(total);
  let head = 0;
  let tail = 0;
  queue[tail++] = bestStart;
  keep[bestStart] = 1;
  while (head < tail) {
    const i = queue[head++];
    const x = i % W;
    const y = (i / W) | 0;
    if (x > 0) {
      const ni = i - 1;
      if (occupied[ni] && !keep[ni]) { keep[ni] = 1; queue[tail++] = ni; }
    }
    if (x + 1 < W) {
      const ni = i + 1;
      if (occupied[ni] && !keep[ni]) { keep[ni] = 1; queue[tail++] = ni; }
    }
    if (y > 0) {
      const ni = i - W;
      if (occupied[ni] && !keep[ni]) { keep[ni] = 1; queue[tail++] = ni; }
    }
    if (y + 1 < H) {
      const ni = i + W;
      if (occupied[ni] && !keep[ni]) { keep[ni] = 1; queue[tail++] = ni; }
    }
  }

  const filteredRigid = new Float32Array(total);
  const filteredSoft = new Float32Array(total);
  let occupiedCount = 0;
  let keptCount = 0;
  for (let i = 0; i < total; i++) {
    const rv = Number(rigidField?.[i]) || 0;
    const sv = Number(softField?.[i]) || 0;
    if (Math.max(rv, sv) >= th) occupiedCount += 1;
    if (keep[i]) {
      filteredRigid[i] = rv;
      filteredSoft[i] = sv;
      keptCount += 1;
    }
  }

  return {
    rigidField: filteredRigid,
    softField: filteredSoft,
    keptCells: keptCount,
    removedCells: Math.max(0, occupiedCount - keptCount),
    componentCount,
  };
}

function sampleMapBilinear(field, x, y, fallback = 0.5) {
  if (!(field instanceof Float32Array) || field.length < W * H) return fallback;
  const cx = Math.max(0, Math.min(W - 1.001, Number(x) || 0));
  const cy = Math.max(0, Math.min(H - 1.001, Number(y) || 0));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(W - 1, x0 + 1);
  const y1 = Math.min(H - 1, y0 + 1);
  const sx = cx - x0;
  const sy = cy - y0;
  const i00 = y0 * W + x0;
  const i10 = y0 * W + x1;
  const i01 = y1 * W + x0;
  const i11 = y1 * W + x1;
  const v00 = Number.isFinite(Number(field[i00])) ? Number(field[i00]) : fallback;
  const v10 = Number.isFinite(Number(field[i10])) ? Number(field[i10]) : fallback;
  const v01 = Number.isFinite(Number(field[i01])) ? Number(field[i01]) : fallback;
  const v11 = Number.isFinite(Number(field[i11])) ? Number(field[i11]) : fallback;
  const a = v00 * (1 - sx) + v10 * sx;
  const b = v01 * (1 - sx) + v11 * sx;
  return Math.max(0, Math.min(1, a * (1 - sy) + b * sy));
}

function membraneShapeColor(weight, alpha = 0.95) {
  const w = Math.max(0, Math.min(1, Number(weight) || 0));
  const r = Math.round(220 * (1 - w));
  const g = Math.round(180 + 60 * w);
  const b = Math.round(120 + 120 * w);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function smoothstep(edge0, edge1, x) {
  const den = Math.max(1e-9, edge1 - edge0);
  const t = Math.max(0, Math.min(1, (x - edge0) / den));
  return t * t * (3 - 2 * t);
}

function mulberry32(seed) {
  let t = (seed >>> 0) || 1;
  return () => {
    t += 0x6D2B79F5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function hashNoise2D(x, y, seed = 0) {
  let h = (Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(seed | 0, 2246822519)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  return (h >>> 0) / 4294967295;
}

function valueNoise2D(x, y, seed = 0) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const sx = x - x0;
  const sy = y - y0;
  const u = sx * sx * (3 - 2 * sx);
  const v = sy * sy * (3 - 2 * sy);

  const n00 = hashNoise2D(x0, y0, seed);
  const n10 = hashNoise2D(x1, y0, seed);
  const n01 = hashNoise2D(x0, y1, seed);
  const n11 = hashNoise2D(x1, y1, seed);
  const a = n00 * (1 - u) + n10 * u;
  const b = n01 * (1 - u) + n11 * u;
  return a * (1 - v) + b * v;
}

function samplePatternValue(kind, nx, ny, params = {}) {
  const TAU = Math.PI * 2;
  const a1 = Number(params.a1) || 0;
  const a2 = Number(params.a2) || 0;
  const f1 = Number(params.f1) || 3;
  const f2 = Number(params.f2) || 5;
  const warp = Number(params.warp) || 0;
  const phase = Number(params.phase) || 0;
  const seed = Number(params.seed) || 1;
  const cx = (Number(params.cx) || 0.5);
  const cy = (Number(params.cy) || 0.5);

  if (kind === 'sine-lines') {
    const line = nx * Math.cos(a1) + ny * Math.sin(a1);
    const cross = nx * Math.cos(a2) + ny * Math.sin(a2);
    const warped = line + warp * Math.sin(TAU * (cross * f2 + phase));
    return 0.5 + 0.5 * Math.sin(TAU * (warped * f1 + phase));
  }

  if (kind === 'wave-interference') {
    const v1 = Math.sin(TAU * (f1 * (nx * Math.cos(a1) + ny * Math.sin(a1)) + phase));
    const v2 = Math.sin(TAU * (f2 * (nx * Math.cos(a2) + ny * Math.sin(a2)) - phase * 0.7));
    const v3 = Math.sin(TAU * ((f1 * 0.6 + f2 * 0.35) * (nx + ny * 0.7) + phase * 1.3));
    return Math.max(0, Math.min(1, 0.5 + 0.25 * v1 + 0.2 * v2 + 0.18 * v3));
  }

  if (kind === 'radial-blobs') {
    const dx = nx - cx;
    const dy = ny - cy;
    const r = Math.sqrt(dx * dx + dy * dy);
    const ang = Math.atan2(dy, dx);
    const ring = 0.5 + 0.5 * Math.sin(TAU * (r * (f1 + 1.2) + 0.25 * Math.sin(ang * 3 + phase)));
    const n = valueNoise2D(nx * (f2 * 1.8 + 2.5), ny * (f2 * 1.6 + 2.1), seed + 19);
    return Math.max(0, Math.min(1, 0.58 * ring + 0.42 * n));
  }

  if (kind === 'flow-ridges') {
    const n1 = valueNoise2D(nx * (f1 * 2.2 + 2), ny * (f1 * 1.9 + 2), seed + 7);
    const n2 = valueNoise2D((nx + warp * (n1 - 0.5)) * (f2 + 2), (ny - warp * (n1 - 0.5)) * (f2 + 2), seed + 31);
    const ridges = 1 - Math.abs(2 * n2 - 1);
    const line = Math.sin(TAU * (f1 * (nx * Math.cos(a1) + ny * Math.sin(a1)) + phase));
    return Math.max(0, Math.min(1, 0.65 * ridges + 0.35 * (0.5 + 0.5 * line)));
  }

  return valueNoise2D(nx * 7, ny * 7, seed);
}

function makeRandomPatternParams(rand, seedOffset = 0) {
  return {
    a1: rand() * Math.PI * 2,
    a2: rand() * Math.PI * 2,
    f1: 1.4 + rand() * 6.2,
    f2: 1.2 + rand() * 6.8,
    warp: 0.04 + rand() * 0.22,
    phase: rand() * Math.PI * 2,
    seed: Math.floor(rand() * 1e9) + seedOffset,
    cx: 0.2 + rand() * 0.6,
    cy: 0.2 + rand() * 0.6,
  };
}

function generateRandomFields({
  preset = 'sine-lines',
  seed = 42,
  targets = {
    trait: true,
    density: true,
    membraneEdge: true,
    membraneShape: true,
    softPermeability: true,
    rigidPermeability: true,
    rigidConsume: true,
  },
} = {}) {
  const baseSeed = Number.isFinite(Number(seed)) ? Math.floor(Number(seed)) : (Date.now() & 0x7fffffff);
  const rand = mulberry32(baseSeed);
  const presets = ['sine-lines', 'wave-interference', 'radial-blobs', 'flow-ridges'];
  const primary = preset === 'mixed' ? presets[Math.floor(rand() * presets.length)] : preset;
  const secondary = preset === 'mixed'
    ? presets[Math.floor(rand() * presets.length)]
    : presets[(presets.indexOf(primary) + 1 + Math.floor(rand() * 2)) % presets.length];

  const targetFlags = {
    trait: !!targets?.trait,
    density: !!targets?.density,
    membraneEdge: !!targets?.membraneEdge,
    membraneShape: !!targets?.membraneShape,
    softPermeability: !!targets?.softPermeability,
    rigidPermeability: !!targets?.rigidPermeability,
    rigidConsume: !!targets?.rigidConsume,
  };

  const rigidParams = makeRandomPatternParams(rand, 11);
  const softParams = makeRandomPatternParams(rand, 23);
  const densityParams = makeRandomPatternParams(rand, 37);
  const edgeParams = makeRandomPatternParams(rand, 53);
  const shapeParams = makeRandomPatternParams(rand, 71);
  const softPermParams = makeRandomPatternParams(rand, 79);
  const permParams = makeRandomPatternParams(rand, 89);
  const consumeRParams = makeRandomPatternParams(rand, 101);
  const consumeGParams = makeRandomPatternParams(rand, 131);
  const consumeBParams = makeRandomPatternParams(rand, 151);

  const margin = 2;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = idx(x, y);
      const nx = (x + 0.5) / W;
      const ny = (y + 0.5) / H;

      const edgeFadeX = smoothstep(0, margin / W, nx) * (1 - smoothstep(1 - margin / W, 1, nx));
      const edgeFadeY = smoothstep(0, margin / H, ny) * (1 - smoothstep(1 - margin / H, 1, ny));
      const edgeFade = Math.max(0, Math.min(1, edgeFadeX * edgeFadeY));

      const rigidSignal = samplePatternValue(primary, nx, ny, rigidParams);
      const softSignal = samplePatternValue(secondary, nx, ny, softParams);

      const rigidWin = smoothstep(0.46, 0.78, rigidSignal - softSignal * 0.72 + 0.08);
      const softWin = smoothstep(0.46, 0.78, softSignal - rigidSignal * 0.72 + 0.08);

      if (targetFlags.trait) {
        if (rigidWin >= softWin) {
          rigid[i] = rigidWin * edgeFade;
          soft[i] = 0;
        } else {
          soft[i] = softWin * edgeFade;
          rigid[i] = 0;
        }
      }

      if (targetFlags.density) {
        const dens = samplePatternValue('wave-interference', nx, ny, densityParams);
        softDensity[i] = Math.max(0, Math.min(1, 0.1 + 0.8 * dens));
      }

      if (targetFlags.membraneEdge) {
        const edgeV = samplePatternValue('flow-ridges', nx, ny, edgeParams);
        membraneEdgeMap[i] = Math.max(0, Math.min(1, edgeV));
      }

      if (targetFlags.membraneShape) {
        const shapeV = samplePatternValue('radial-blobs', nx, ny, shapeParams);
        membraneShapeMap[i] = Math.max(0, Math.min(1, 0.15 + 0.85 * shapeV));
      }

      if (targetFlags.softPermeability) {
        const softPermV = samplePatternValue('flow-ridges', nx, ny, softPermParams);
        softPermeabilityMap[i] = soft[i] > 0.12 ? smoothstep(0.54, 0.82, softPermV) : 0;
      }

      if (targetFlags.rigidPermeability) {
        const permV = samplePatternValue('sine-lines', nx, ny, permParams);
        rigidPermeabilityMap[i] = rigid[i] > 0.12 ? smoothstep(0.56, 0.84, permV) : 0;
      }

      if (targetFlags.rigidConsume) {
        const consumeMask = smoothstep(0.58, 0.9, rigidSignal);
        const rV = samplePatternValue('wave-interference', nx, ny, consumeRParams);
        const gV = samplePatternValue('wave-interference', nx, ny, consumeGParams);
        const bV = samplePatternValue('wave-interference', nx, ny, consumeBParams);
        rigidEdgeConsumeMapR[i] = rigid[i] > 0.1 ? consumeMask * smoothstep(0.72, 0.9, rV) : 0;
        rigidEdgeConsumeMapG[i] = rigid[i] > 0.1 ? consumeMask * smoothstep(0.72, 0.9, gV) : 0;
        rigidEdgeConsumeMapB[i] = rigid[i] > 0.1 ? consumeMask * smoothstep(0.72, 0.9, bV) : 0;
      }
    }
  }

  drawFields();
  compileNow();
  return {
    seed: baseSeed,
    preset,
    rigidPattern: primary,
    softPattern: secondary,
    targets: targetFlags,
  };
}

function buildSpecFromCurrentFields(mesh, traitFields = null) {
  if (!mesh) return null;
  const rigidFieldForExport = traitFields?.rigidField || rigid;
  const softFieldForExport = traitFields?.softField || soft;
  const membraneMinEdgeLength = Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4);
  const membraneMaxEdgeLength = Math.max(membraneMinEdgeLength, Number(membraneMaxEdgeLengthEl?.value) || 8);
  return createCreatureSpecFromMesh(mesh, {
    name: 'mesh-lab-creature',
    fields: {
      rigidField: rigidFieldForExport,
      softField: softFieldForExport,
      softDensityField: softDensity,
      membraneEdgeMap,
      membraneShapeMap,
      softPermeabilityMap,
      rigidPermeabilityMap,
      rigidEdgeConsumeMapR,
      rigidEdgeConsumeMapG,
      rigidEdgeConsumeMapB,
    },
    threshold: Math.max(0, Math.min(1, Number(thresholdEl?.value) || 0.35)),
    softBoundaryRingSprings: !!softBoundaryRingEl?.checked,
    softSolverMode: softSolverModeEl?.value || 'membrane',
    membraneMinEdgeLength,
    membraneMaxEdgeLength,
  });
}

function finiteOr(raw, fallback) {
  const v = Number(raw);
  return Number.isFinite(v) ? v : fallback;
}

function readWindTunnelEmitterOptions() {
  const defaultRadius = Math.max(7, W / 13);
  const defaultWobbleAmp = Math.max(1.4, W * 0.02);

  return {
    emitterStrength: Math.max(0.1, finiteOr(windEmitterStrengthEl?.value, 3.8)),
    emitterJetVy: Math.max(-4, Math.min(6, finiteOr(windEmitterJetVyEl?.value, 1.35))),
    emitterRadius: Math.max(3, finiteOr(windEmitterRadiusEl?.value, defaultRadius)),
    emitterYFraction: Math.max(0.06, Math.min(0.35, finiteOr(windEmitterYFractionEl?.value, 0.12))),
    emitterSpin: finiteOr(windEmitterSpinEl?.value, 1.55),
    emitterCurlGain: Math.max(0, finiteOr(windEmitterCurlGainEl?.value, 1.95)),
    emitterDriftGain: Math.max(0, finiteOr(windEmitterDriftGainEl?.value, 0.22)),
    emitterChaosGain: Math.max(0, finiteOr(windEmitterChaosGainEl?.value, 1.45)),
    emitterWobbleAmp: Math.max(0, finiteOr(windEmitterWobbleAmpEl?.value, defaultWobbleAmp)),
    emitterWobbleFreq: Math.max(0, finiteOr(windEmitterWobbleFreqEl?.value, 0.11)),
    emitterLockPosition: windEmitterLockPositionEl?.checked !== false,
  };
}

function pushSpecToWindTunnel(spec) {
  if (!spec || !windTunnelFrame?.contentWindow) return;
  const payload = {
    type: 'gpuLabEmbedReset',
    spec,
    options: {
      grid: W,
      targetSpanFraction: 0.42,
      importScale: 1,
      ...readWindTunnelEmitterOptions(),
    },
  };

  // Same-origin iframe; no wildcard posting.
  windTunnelFrame.contentWindow.postMessage(payload, window.location.origin);
}

function drawFields() {
  const traitImg = pctx.createImageData(W, H);
  const densityImg = dctx.createImageData(W, H);
  const edgeImg = ectx.createImageData(W, H);
  const shapeImg = sctx.createImageData(W, H);
  const softPermeabilityImg = spctx.createImageData(W, H);
  const permeabilityImg = prctx.createImageData(W, H);
  const consumeImg = cctx.createImageData(W, H);

  for (let i = 0; i < rigid.length; i++) {
    const r = Math.max(0, Math.min(1, rigid[i]));
    const s = Math.max(0, Math.min(1, soft[i]));
    const dens = Math.max(0, Math.min(1, softDensity[i]));
    const edge = Math.max(0, Math.min(1, membraneEdgeMap[i]));
    const shape = Math.max(0, Math.min(1, membraneShapeMap[i]));
    const softPermeability = Math.max(0, Math.min(1, softPermeabilityMap[i]));
    const permeability = Math.max(0, Math.min(1, rigidPermeabilityMap[i]));
    const consumeR = Math.max(0, Math.min(1, rigidEdgeConsumeMapR[i]));
    const consumeG = Math.max(0, Math.min(1, rigidEdgeConsumeMapG[i]));
    const consumeB = Math.max(0, Math.min(1, rigidEdgeConsumeMapB[i]));

    // Trait plane: rigid red, soft blue.
    traitImg.data[i * 4] = Math.min(255, r * 255);
    traitImg.data[i * 4 + 1] = 0;
    traitImg.data[i * 4 + 2] = Math.min(255, s * 255);
    traitImg.data[i * 4 + 3] = 255;

    // Density plane: green-only.
    densityImg.data[i * 4] = 0;
    densityImg.data[i * 4 + 1] = Math.round(40 + dens * 180);
    densityImg.data[i * 4 + 2] = 0;
    densityImg.data[i * 4 + 3] = 255;

    // Edge-length map: grayscale (black=min edge, white=max edge).
    const eg = Math.round(edge * 255);
    edgeImg.data[i * 4] = eg;
    edgeImg.data[i * 4 + 1] = eg;
    edgeImg.data[i * 4 + 2] = eg;
    edgeImg.data[i * 4 + 3] = 255;

    // Shape-memory map: amber->cyan gradient to show give/stiff bias.
    shapeImg.data[i * 4] = Math.round(220 * (1 - shape));
    shapeImg.data[i * 4 + 1] = Math.round(180 + 60 * shape);
    shapeImg.data[i * 4 + 2] = Math.round(120 + 120 * shape);
    shapeImg.data[i * 4 + 3] = 255;

    // Soft edge permeability map: grayscale (black=blocked, white=permeable).
    const spg = Math.round(softPermeability * 255);
    softPermeabilityImg.data[i * 4] = spg;
    softPermeabilityImg.data[i * 4 + 1] = spg;
    softPermeabilityImg.data[i * 4 + 2] = spg;
    softPermeabilityImg.data[i * 4 + 3] = 255;

    // Rigid edge permeability map: grayscale (black=blocked, white=permeable).
    const pg = Math.round(permeability * 255);
    permeabilityImg.data[i * 4] = pg;
    permeabilityImg.data[i * 4 + 1] = pg;
    permeabilityImg.data[i * 4 + 2] = pg;
    permeabilityImg.data[i * 4 + 3] = 255;

    // Edge consume-dye map: RGB channels represent absorb mask by color.
    consumeImg.data[i * 4] = Math.round(consumeR * 255);
    consumeImg.data[i * 4 + 1] = Math.round(consumeG * 255);
    consumeImg.data[i * 4 + 2] = Math.round(consumeB * 255);
    consumeImg.data[i * 4 + 3] = 255;
  }

  const blit = (ctx, canvas, img) => {
    const tmp = document.createElement('canvas');
    tmp.width = W;
    tmp.height = H;
    tmp.getContext('2d').putImageData(img, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tmp, 0, 0, W, H, 0, 0, canvas.width, canvas.height);
  };

  blit(pctx, paintCanvas, traitImg);
  blit(dctx, densityCanvas, densityImg);
  blit(ectx, membraneEdgeCanvas, edgeImg);
  blit(sctx, membraneShapeCanvas, shapeImg);
  blit(spctx, softPermeabilityCanvas, softPermeabilityImg);
  blit(prctx, rigidPermeabilityCanvas, permeabilityImg);
  blit(cctx, rigidConsumeCanvas, consumeImg);
}

function paintTrait(clientX, clientY) {
  const rect = paintCanvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * W;
  const y = ((clientY - rect.top) / rect.height) * H;
  const r = Math.max(1, Number(brushEl.value) || 50) * (W / paintCanvas.width);
  const mode = modeEl.value;
  const minX = Math.max(0, Math.floor(x - r));
  const maxX = Math.min(W - 1, Math.ceil(x + r));
  const minY = Math.max(0, Math.floor(y - r));
  const maxY = Math.min(H - 1, Math.ceil(y + r));

  for (let yy = minY; yy <= maxY; yy++) {
    for (let xx = minX; xx <= maxX; xx++) {
      const d = Math.hypot(xx - x, yy - y);
      if (d > r) continue;
      const t = 1 - d / r;
      const i = idx(xx, yy);
      if (mode === 'rigid') {
        rigid[i] = Math.max(rigid[i], t);
        soft[i] *= (1 - t);
      } else if (mode === 'soft') {
        soft[i] = Math.max(soft[i], t);
        rigid[i] *= (1 - t);
      } else {
        rigid[i] *= (1 - t);
        soft[i] *= (1 - t);
      }
    }
  }
  drawFields();
}

function paintScalarMap(targetArray, canvasEl, clientX, clientY, paintValue, eraseValue, erase = false) {
  const rect = canvasEl.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * W;
  const y = ((clientY - rect.top) / rect.height) * H;
  const r = Math.max(1, Number(brushEl.value) || 50) * (W / canvasEl.width);
  const target = erase
    ? Math.max(0, Math.min(1, Number(eraseValue)))
    : Math.max(0, Math.min(1, Number(paintValue)));

  const minX = Math.max(0, Math.floor(x - r));
  const maxX = Math.min(W - 1, Math.ceil(x + r));
  const minY = Math.max(0, Math.floor(y - r));
  const maxY = Math.min(H - 1, Math.ceil(y + r));

  for (let yy = minY; yy <= maxY; yy++) {
    for (let xx = minX; xx <= maxX; xx++) {
      const d = Math.hypot(xx - x, yy - y);
      if (d > r) continue;
      const t = 1 - d / r;
      const i = idx(xx, yy);
      targetArray[i] = targetArray[i] * (1 - t) + target * t;
    }
  }
}

function paintDensity(clientX, clientY, erase = false) {
  paintScalarMap(
    softDensity,
    densityCanvas,
    clientX,
    clientY,
    Number(softDensityPaintEl?.value) || 0.5,
    0.5,
    erase,
  );
  drawFields();
}

function paintMembraneEdgeMap(clientX, clientY, erase = false) {
  paintScalarMap(
    membraneEdgeMap,
    membraneEdgeCanvas,
    clientX,
    clientY,
    Number(membraneEdgePaintEl?.value) || 0.5,
    0.5,
    erase,
  );
  drawFields();
}

function paintMembraneShapeMap(clientX, clientY, erase = false) {
  paintScalarMap(
    membraneShapeMap,
    membraneShapeCanvas,
    clientX,
    clientY,
    Number(membraneShapePaintEl?.value) || 1.0,
    1.0,
    erase,
  );
  drawFields();
}

function paintSoftPermeabilityMap(clientX, clientY, erase = false) {
  paintScalarMap(
    softPermeabilityMap,
    softPermeabilityCanvas,
    clientX,
    clientY,
    Number(softPermeabilityPaintEl?.value) || 1.0,
    0.0,
    erase,
  );
  drawFields();
}

function paintRigidPermeabilityMap(clientX, clientY, erase = false) {
  paintScalarMap(
    rigidPermeabilityMap,
    rigidPermeabilityCanvas,
    clientX,
    clientY,
    Number(rigidPermeabilityPaintEl?.value) || 1.0,
    0.0,
    erase,
  );
  drawFields();
}

function paintRigidConsumeMap(clientX, clientY, erase = false) {
  const channel = String(rigidConsumeChannelEl?.value || 'r');
  const paintValue = Number(rigidConsumePaintEl?.value) || 1.0;

  if (channel === 'erase') {
    paintScalarMap(rigidEdgeConsumeMapR, rigidConsumeCanvas, clientX, clientY, 0.0, 0.0, true);
    paintScalarMap(rigidEdgeConsumeMapG, rigidConsumeCanvas, clientX, clientY, 0.0, 0.0, true);
    paintScalarMap(rigidEdgeConsumeMapB, rigidConsumeCanvas, clientX, clientY, 0.0, 0.0, true);
    drawFields();
    return;
  }

  const target = channel === 'g'
    ? rigidEdgeConsumeMapG
    : (channel === 'b' ? rigidEdgeConsumeMapB : rigidEdgeConsumeMapR);
  paintScalarMap(target, rigidConsumeCanvas, clientX, clientY, paintValue, 0.0, erase);
  drawFields();
}

function drawMesh(mesh) {
  mctx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
  const sx = meshCanvas.width / W;
  const sy = meshCanvas.height / H;
  const membranePreview = (softSolverModeEl?.value || 'membrane') === 'membrane';
  const compiledSoftField = mesh?.__compileFields?.softField || soft;

  if (!membranePreview) {
    // Spring mode preview: show compiled soft triangles + optional cross-beams.
    for (const tri of mesh.triangles) {
      if (tri.kind !== 'soft') continue;
      const a = mesh.nodes[tri.a], b = mesh.nodes[tri.b], c = mesh.nodes[tri.c];
      mctx.beginPath();
      mctx.moveTo(a.x * sx, a.y * sy);
      mctx.lineTo(b.x * sx, b.y * sy);
      mctx.lineTo(c.x * sx, c.y * sy);
      mctx.closePath();
      mctx.fillStyle = 'rgba(90,130,255,0.18)';
      mctx.strokeStyle = 'rgba(120,170,255,0.55)';
      mctx.fill();
      mctx.stroke();
    }

    if (Array.isArray(mesh.softCrossBeams)) {
      mctx.strokeStyle = 'rgba(140,220,255,0.92)';
      mctx.lineWidth = 1.5;
      for (const [ai, bi] of mesh.softCrossBeams) {
        const a = mesh.nodes[ai];
        const b = mesh.nodes[bi];
        if (!a || !b) continue;
        mctx.beginPath();
        mctx.moveTo(a.x * sx, a.y * sy);
        mctx.lineTo(b.x * sx, b.y * sy);
        mctx.stroke();
      }
      mctx.lineWidth = 1;
    }
  } else {
    // Membrane mode preview: draw the same resampled perimeter ring used for export.
    const thr = Math.max(0, Math.min(1, Number(thresholdEl?.value) || 0.35));
    const minEdge = Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4);
    const maxEdge = Math.max(minEdge, Number(membraneMaxEdgeLengthEl?.value) || 8);
    const rings = buildMembraneRingsFromSoftField({
      width: W,
      height: H,
      softField: compiledSoftField,
      edgeLengthField: membraneEdgeMap,
      threshold: thr,
      minEdgeLength: minEdge,
      maxEdgeLength: maxEdge,
    });

    mctx.lineWidth = 2;
    for (const ring of rings) {
      if (!Array.isArray(ring) || ring.length < 3) continue;

      // Color each membrane edge/node by sampled membrane-stiffness field value.
      for (let i = 0; i < ring.length; i++) {
        const a = ring[i];
        const b = ring[(i + 1) % ring.length];
        const wa = sampleMapBilinear(membraneShapeMap, a.x, a.y, 1);
        const wb = sampleMapBilinear(membraneShapeMap, b.x, b.y, 1);
        const wEdge = 0.5 * (wa + wb);
        mctx.strokeStyle = membraneShapeColor(wEdge, 0.95);
        mctx.beginPath();
        mctx.moveTo(a.x * sx, a.y * sy);
        mctx.lineTo(b.x * sx, b.y * sy);
        mctx.stroke();
      }

      for (const p of ring) {
        const wNode = sampleMapBilinear(membraneShapeMap, p.x, p.y, 1);
        mctx.fillStyle = membraneShapeColor(wNode, 0.98);
        mctx.beginPath();
        mctx.arc(p.x * sx, p.y * sy, 2.0, 0, Math.PI * 2);
        mctx.fill();
      }
    }

    const gx = 10;
    const gy = meshCanvas.height - 24;
    const gw = 140;
    const gh = 8;
    const grad = mctx.createLinearGradient(gx, gy, gx + gw, gy);
    grad.addColorStop(0, membraneShapeColor(0, 1));
    grad.addColorStop(1, membraneShapeColor(1, 1));
    mctx.fillStyle = grad;
    mctx.fillRect(gx, gy, gw, gh);
    mctx.strokeStyle = 'rgba(255,255,255,0.35)';
    mctx.strokeRect(gx, gy, gw, gh);
    mctx.fillStyle = 'rgba(230,230,230,0.9)';
    mctx.font = '11px system-ui';
    mctx.fillText('membrane stiffness: soft → stiff', gx, gy - 4);

    mctx.lineWidth = 1;
  }

  // Compiler-stage rigid decomposition preview (authoritative for export/import path).
  if (Array.isArray(mesh.rigidPieces)) {
    for (const p of mesh.rigidPieces) {
      if (!p.hull?.length) continue;
      mctx.beginPath();
      for (let i = 0; i < p.hull.length; i++) {
        const hp = p.hull[i];
        const x = hp.x * sx;
        const y = hp.y * sy;
        if (i === 0) mctx.moveTo(x, y);
        else mctx.lineTo(x, y);
      }
      mctx.closePath();
      mctx.strokeStyle = 'rgba(255,220,130,0.95)';
      mctx.lineWidth = 2;
      mctx.stroke();

      // Show explicit contour vertices so spacing differences are visible.
      mctx.fillStyle = 'rgba(255,245,190,0.98)';
      for (const hp of p.hull) {
        mctx.beginPath();
        mctx.arc(hp.x * sx, hp.y * sy, 1.8, 0, Math.PI * 2);
        mctx.fill();
      }

      mctx.lineWidth = 1;
    }
  }

  out.textContent = JSON.stringify({
    ...mesh.meta,
    compileRevision,
    softSolverMode: softSolverModeEl?.value || 'membrane',
    softPreview: ((softSolverModeEl?.value || 'membrane') === 'membrane')
      ? 'resampled membrane ring from painted mask'
      : 'triangulated soft mesh',
    membraneShapePreview: ((softSolverModeEl?.value || 'membrane') === 'membrane')
      ? 'ring edge/node color encodes membrane stiffness map (soft→stiff)'
      : undefined,
    rigidContourPreview: 'rigid contours are perimeter-resampled from border vertices; dots show exported vertices',
    contiguousBodyFilter: {
      keptCells: mesh?.__compileFields?.keptCells ?? null,
      removedCells: mesh?.__compileFields?.removedCells ?? null,
      componentCount: mesh?.__compileFields?.componentCount ?? null,
      policy: 'largest contiguous painted body only (export + wind tunnel)',
    },
    membraneMinEdgeLength: Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4),
    membraneMaxEdgeLength: Math.max(Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4), Number(membraneMaxEdgeLengthEl?.value) || 8),
  }, null, 2);
}

function setWidgetEnabled(el, enabled, disabledTitle = '') {
  if (!el) return;
  el.disabled = !enabled;
  el.title = enabled ? '' : disabledTitle;

  const ownerLabel = el.closest('label');
  if (ownerLabel) {
    ownerLabel.classList.toggle('control-disabled', !enabled);
    ownerLabel.title = enabled ? '' : disabledTitle;
  }
}

function syncFieldPanelVisibility() {
  const target = fieldPaintTargetEl?.value || 'trait';
  for (const panel of fieldPanels) {
    if (!panel) continue;
    panel.hidden = panel.dataset.fieldPanel !== target;
  }

  const traitSelected = target === 'trait';
  const densitySelected = target === 'density';
  const edgeSelected = target === 'membraneEdge';
  const shapeSelected = target === 'membraneShape';
  const softPermeabilitySelected = target === 'softPermeability';
  const permeabilitySelected = target === 'rigidPermeability';
  const consumeSelected = target === 'rigidConsume';

  setWidgetEnabled(modeEl, traitSelected, 'Trait paint mode only applies when Field to paint = Trait field');
  setWidgetEnabled(softDensityPaintEl, densitySelected, 'Resolution paint value only applies when Field to paint = Resolution field');
  setWidgetEnabled(membraneEdgePaintEl, edgeSelected, 'Perimeter edge paint value only applies when Field to paint = Membrane edge-length field');
  setWidgetEnabled(membraneShapePaintEl, shapeSelected, 'Membrane stiffness paint value only applies when Field to paint = Membrane stiffness field');
  setWidgetEnabled(softPermeabilityPaintEl, softPermeabilitySelected, 'Soft permeability paint value only applies when Field to paint = Soft permeability field');
  setWidgetEnabled(rigidPermeabilityPaintEl, permeabilitySelected, 'Rigid permeability paint value only applies when Field to paint = Rigid permeability field');
  setWidgetEnabled(rigidConsumeChannelEl, consumeSelected, 'Consume channel only applies when Field to paint = Rigid edge consume-dye field');
  setWidgetEnabled(rigidConsumePaintEl, consumeSelected, 'Consume paint value only applies when Field to paint = Rigid edge consume-dye field');

  return target;
}

function syncSoftModeUi() {
  const membraneMode = (softSolverModeEl?.value || 'membrane') === 'membrane';
  if (!softInfillModeEl) return membraneMode;
  if (membraneMode) {
    softInfillModeEl.value = 'none';
    softInfillModeEl.disabled = true;
    softInfillModeEl.title = 'Membrane mode uses perimeter-only representation (no interior infill)';
    if (membraneMinEdgeLengthEl) {
      membraneMinEdgeLengthEl.disabled = false;
      membraneMinEdgeLengthEl.title = 'Minimum perimeter edge length (membrane ring + rigid contour resampling)';
    }
    if (membraneMaxEdgeLengthEl) {
      membraneMaxEdgeLengthEl.disabled = false;
      membraneMaxEdgeLengthEl.title = 'Maximum perimeter edge length (membrane ring + rigid contour resampling)';
    }
  } else {
    if (softInfillModeEl.value === 'none') softInfillModeEl.value = 'triangles';
    softInfillModeEl.disabled = false;
    softInfillModeEl.title = '';
    if (membraneMinEdgeLengthEl) {
      membraneMinEdgeLengthEl.disabled = false;
      membraneMinEdgeLengthEl.title = 'Minimum perimeter edge length (applies to rigid contour resampling even in spring mode)';
    }
    if (membraneMaxEdgeLengthEl) {
      membraneMaxEdgeLengthEl.disabled = false;
      membraneMaxEdgeLengthEl.title = 'Maximum perimeter edge length (applies to rigid contour resampling even in spring mode)';
    }
  }
  return membraneMode;
}

function compileNow() {
  try {
    const membraneMode = syncSoftModeUi();
    const requestedSoftMin = Math.max(1, Math.min(Math.max(1, W - 1), Math.round(Number(softMinCellSizeEl?.value) || 3)));
    const perimeterMinEdge = Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4);
    const perimeterMaxEdge = Math.max(perimeterMinEdge, Number(membraneMaxEdgeLengthEl?.value) || 8);
    const threshold = Math.max(0, Math.min(1, Number(thresholdEl.value) || 0.35));
    const largestBody = isolateLargestContiguousTraitBody(rigid, soft, threshold);

    const mesh = compileFieldToMesh({
      width: W,
      height: H,
      rigidField: largestBody.rigidField,
      softField: largestBody.softField,
      density: 1,
      threshold,
      connectivityMode: 'largest',
      minComponentTriangles: 0,
      rigidCompileMode: rigidCompileModeEl?.value || 'contours',
      rigidPrimitiveSideMin: Math.max(2, Math.min(64, Math.round(Number(rigidPrimitiveSideMinEl?.value) || 4))),
      rigidPrimitiveSideMax: Math.max(2, Math.min(96, Math.round(Number(rigidPrimitiveSideMaxEl?.value) || 10))),
      // In contour mode, rigid hulls now follow the same perimeter-resampling model as membrane rings.
      rigidEdgeLengthField: membraneEdgeMap,
      rigidMinEdgeLength: perimeterMinEdge,
      rigidMaxEdgeLength: perimeterMaxEdge,
      softInfillMode: membraneMode ? 'none' : (softInfillModeEl?.value || 'triangles'),
      softDensityField: softDensity,
      // Membrane mode should preserve boundary fidelity; coarse soft cells make boxy/square contours.
      softMinCellSize: membraneMode ? 1 : requestedSoftMin,
      softBoundaryCellCap: membraneMode ? 1 : 2,
    });
    mesh.__compileFields = {
      rigidField: largestBody.rigidField,
      softField: largestBody.softField,
      keptCells: largestBody.keptCells,
      removedCells: largestBody.removedCells,
      componentCount: largestBody.componentCount,
    };
    lastMesh = mesh;
    lastCompiledFields = mesh.__compileFields;
    lastCompiledSpec = buildSpecFromCurrentFields(mesh, lastCompiledFields);
    compileRevision += 1;
    drawMesh(mesh);
    pushSpecToWindTunnel(lastCompiledSpec);
    return mesh;
  } catch (err) {
    const msg = `Compile failed: ${String(err?.message || err)}`;
    console.error('[mesh-lab] compileNow failed', err);
    out.textContent = msg;
    return null;
  }
}

let traitPainting = false;
let densityPainting = false;
let membraneEdgePainting = false;
let membraneShapePainting = false;
let softPermeabilityPainting = false;
let rigidPermeabilityPainting = false;
let rigidConsumePainting = false;

paintCanvas.addEventListener('mousedown', (e) => { traitPainting = true; paintTrait(e.clientX, e.clientY); });
paintCanvas.addEventListener('mousemove', (e) => { if (traitPainting) paintTrait(e.clientX, e.clientY); });

densityCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
densityCanvas.addEventListener('mousedown', (e) => {
  densityPainting = true;
  paintDensity(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
densityCanvas.addEventListener('mousemove', (e) => {
  if (!densityPainting) return;
  paintDensity(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

membraneEdgeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
membraneEdgeCanvas.addEventListener('mousedown', (e) => {
  membraneEdgePainting = true;
  paintMembraneEdgeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
membraneEdgeCanvas.addEventListener('mousemove', (e) => {
  if (!membraneEdgePainting) return;
  paintMembraneEdgeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

membraneShapeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
membraneShapeCanvas.addEventListener('mousedown', (e) => {
  membraneShapePainting = true;
  paintMembraneShapeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
membraneShapeCanvas.addEventListener('mousemove', (e) => {
  if (!membraneShapePainting) return;
  paintMembraneShapeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

softPermeabilityCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
softPermeabilityCanvas.addEventListener('mousedown', (e) => {
  softPermeabilityPainting = true;
  paintSoftPermeabilityMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
softPermeabilityCanvas.addEventListener('mousemove', (e) => {
  if (!softPermeabilityPainting) return;
  paintSoftPermeabilityMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

rigidPermeabilityCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
rigidPermeabilityCanvas.addEventListener('mousedown', (e) => {
  rigidPermeabilityPainting = true;
  paintRigidPermeabilityMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
rigidPermeabilityCanvas.addEventListener('mousemove', (e) => {
  if (!rigidPermeabilityPainting) return;
  paintRigidPermeabilityMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

rigidConsumeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
rigidConsumeCanvas.addEventListener('mousedown', (e) => {
  rigidConsumePainting = true;
  paintRigidConsumeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
rigidConsumeCanvas.addEventListener('mousemove', (e) => {
  if (!rigidConsumePainting) return;
  paintRigidConsumeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

window.addEventListener('mouseup', () => {
  traitPainting = false;
  densityPainting = false;
  membraneEdgePainting = false;
  membraneShapePainting = false;
  softPermeabilityPainting = false;
  rigidPermeabilityPainting = false;
  rigidConsumePainting = false;
});

clearBtn.addEventListener('click', () => {
  rigid.fill(0);
  soft.fill(0);
  softDensity.fill(0.5);
  membraneEdgeMap.fill(0.5);
  membraneShapeMap.fill(1.0);
  softPermeabilityMap.fill(0.0);
  rigidPermeabilityMap.fill(0.0);
  rigidEdgeConsumeMapR.fill(0.0);
  rigidEdgeConsumeMapG.fill(0.0);
  rigidEdgeConsumeMapB.fill(0.0);
  drawFields();
  compileNow();
});
compileBtn.addEventListener('click', compileNow);
function randomizeWithTargets(targets) {
  const preset = String(randomFieldPresetEl?.value || 'sine-lines');
  const rawSeed = Number(randomFieldSeedEl?.value);
  const seed = Number.isFinite(rawSeed) ? Math.floor(rawSeed) : (Date.now() & 0x7fffffff);
  const info = generateRandomFields({ preset, seed, targets });
  if (randomFieldSeedEl && Number.isFinite(Number(info?.seed))) {
    randomFieldSeedEl.value = String(info.seed);
  }
  return info;
}

if (randomizeBtn) {
  randomizeBtn.addEventListener('click', () => {
    randomizeWithTargets({
      trait: true,
      density: true,
      membraneEdge: true,
      membraneShape: true,
      softPermeability: true,
      rigidPermeability: true,
      rigidConsume: true,
    });
  });
}
if (randomizeTraitBtn) {
  randomizeTraitBtn.addEventListener('click', () => randomizeWithTargets({ trait: true }));
}
if (randomizeDensityBtn) {
  randomizeDensityBtn.addEventListener('click', () => randomizeWithTargets({ density: true }));
}
if (randomizeMembraneEdgeBtn) {
  randomizeMembraneEdgeBtn.addEventListener('click', () => randomizeWithTargets({ membraneEdge: true }));
}
if (randomizeMembraneShapeBtn) {
  randomizeMembraneShapeBtn.addEventListener('click', () => randomizeWithTargets({ membraneShape: true }));
}
if (randomizeSoftPermeabilityBtn) {
  randomizeSoftPermeabilityBtn.addEventListener('click', () => randomizeWithTargets({ softPermeability: true }));
}
if (randomizeRigidPermeabilityBtn) {
  randomizeRigidPermeabilityBtn.addEventListener('click', () => randomizeWithTargets({ rigidPermeability: true }));
}
if (randomizeRigidConsumeBtn) {
  randomizeRigidConsumeBtn.addEventListener('click', () => randomizeWithTargets({ rigidConsume: true }));
}
if (fieldPaintTargetEl) fieldPaintTargetEl.addEventListener('change', syncFieldPanelVisibility);
if (softInfillModeEl) softInfillModeEl.addEventListener('change', compileNow);
if (softMinCellSizeEl) softMinCellSizeEl.addEventListener('change', compileNow);
if (softSolverModeEl) softSolverModeEl.addEventListener('change', compileNow);
if (membraneMinEdgeLengthEl) membraneMinEdgeLengthEl.addEventListener('change', compileNow);
if (membraneMaxEdgeLengthEl) membraneMaxEdgeLengthEl.addEventListener('change', compileNow);
if (rigidCompileModeEl) rigidCompileModeEl.addEventListener('change', compileNow);
if (rigidPrimitiveSideMinEl) rigidPrimitiveSideMinEl.addEventListener('change', compileNow);
if (rigidPrimitiveSideMaxEl) rigidPrimitiveSideMaxEl.addEventListener('change', compileNow);

const windEmitterInputs = [
  windEmitterStrengthEl,
  windEmitterJetVyEl,
  windEmitterRadiusEl,
  windEmitterYFractionEl,
  windEmitterSpinEl,
  windEmitterCurlGainEl,
  windEmitterDriftGainEl,
  windEmitterChaosGainEl,
  windEmitterWobbleAmpEl,
  windEmitterWobbleFreqEl,
  windEmitterLockPositionEl,
].filter(Boolean);

for (const el of windEmitterInputs) {
  el.addEventListener('change', () => {
    if (lastCompiledSpec) pushSpecToWindTunnel(lastCompiledSpec);
  });
}

if (applyWindEmitterBtn) {
  applyWindEmitterBtn.addEventListener('click', () => {
    if (!lastCompiledSpec) {
      const mesh = compileNow();
      if (!mesh || !lastCompiledSpec) return;
    }
    pushSpecToWindTunnel(lastCompiledSpec);
  });
}

exportBtn.addEventListener('click', () => {
  const mesh = compileNow();
  if (!mesh) return;
  const spec = lastCompiledSpec || buildSpecFromCurrentFields(mesh, lastCompiledFields);
  if (!spec) return;
  const blob = new Blob([JSON.stringify(spec, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `creature-spec-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

importBtn.addEventListener('click', () => importFile.click());
importFile.addEventListener('change', async () => {
  const f = importFile.files?.[0];
  if (!f) return;
  const text = await f.text();
  const spec = parseCreatureSpec(text);
  if (softSolverModeEl) {
    const importedMode = String(spec.softBodies?.[0]?.solverMode || '').toLowerCase();
    const mode = importedMode === 'spring' ? 'spring' : 'membrane';
    softSolverModeEl.value = mode;
  }
  syncSoftModeUi();

  const authoring = spec.authoring?.fields;
  if (authoring && Number(authoring.width) === W && Number(authoring.height) === H && Array.isArray(authoring.rigid) && Array.isArray(authoring.soft)) {
    rigid.set(authoring.rigid);
    soft.set(authoring.soft);
    if (Array.isArray(authoring.softDensity) && authoring.softDensity.length === W * H) softDensity.set(authoring.softDensity);
    else softDensity.fill(0.5);
    if (Array.isArray(authoring.membraneEdgeMap) && authoring.membraneEdgeMap.length === W * H) membraneEdgeMap.set(authoring.membraneEdgeMap);
    else membraneEdgeMap.fill(0.5);
    if (Array.isArray(authoring.membraneShapeMap) && authoring.membraneShapeMap.length === W * H) membraneShapeMap.set(authoring.membraneShapeMap);
    else membraneShapeMap.fill(1.0);
    if (Array.isArray(authoring.softPermeabilityMap) && authoring.softPermeabilityMap.length === W * H) softPermeabilityMap.set(authoring.softPermeabilityMap);
    else softPermeabilityMap.fill(0.0);
    if (Array.isArray(authoring.rigidPermeabilityMap) && authoring.rigidPermeabilityMap.length === W * H) rigidPermeabilityMap.set(authoring.rigidPermeabilityMap);
    else rigidPermeabilityMap.fill(0.0);
    if (Array.isArray(authoring.rigidEdgeConsumeMapR) && authoring.rigidEdgeConsumeMapR.length === W * H) rigidEdgeConsumeMapR.set(authoring.rigidEdgeConsumeMapR);
    else rigidEdgeConsumeMapR.fill(0.0);
    if (Array.isArray(authoring.rigidEdgeConsumeMapG) && authoring.rigidEdgeConsumeMapG.length === W * H) rigidEdgeConsumeMapG.set(authoring.rigidEdgeConsumeMapG);
    else rigidEdgeConsumeMapG.fill(0.0);
    if (Array.isArray(authoring.rigidEdgeConsumeMapB) && authoring.rigidEdgeConsumeMapB.length === W * H) rigidEdgeConsumeMapB.set(authoring.rigidEdgeConsumeMapB);
    else rigidEdgeConsumeMapB.fill(0.0);
    drawFields();
    compileNow();
    return;
  }

  // Fallback preview for specs that don't carry authoring fields.
  mctx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
  const sx = meshCanvas.width / W;
  const sy = meshCanvas.height / H;

  for (const rb of spec.rigidBodies || []) {
    if (!rb.hull?.length) continue;
    mctx.beginPath();
    for (let i = 0; i < rb.hull.length; i++) {
      const p = rb.hull[i];
      const x = p.x * sx;
      const y = p.y * sy;
      if (i === 0) mctx.moveTo(x, y);
      else mctx.lineTo(x, y);
    }
    mctx.closePath();
    mctx.fillStyle = 'rgba(255,90,90,0.16)';
    mctx.strokeStyle = 'rgba(255,140,140,0.8)';
    mctx.fill();
    mctx.stroke();
  }

  for (const sb of spec.softBodies || []) {
    for (const [ai, bi] of sb.springs || []) {
      const a = sb.nodes?.[ai];
      const b = sb.nodes?.[bi];
      if (!a || !b) continue;
      mctx.strokeStyle = 'rgba(120,170,255,0.8)';
      mctx.beginPath();
      mctx.moveTo(a.x * sx, a.y * sy);
      mctx.lineTo(b.x * sx, b.y * sy);
      mctx.stroke();
    }
  }

  out.textContent = JSON.stringify({
    schemaVersion: spec.schemaVersion,
    rigidBodies: spec.rigidBodies?.length || 0,
    softBodies: spec.softBodies?.length || 0,
    hybridJoints: spec.hybridJoints?.length || 0,
    note: 'No authoring fields embedded; showing solver-structure preview.',
  }, null, 2);
});

if (windTunnelFrame) {
  windTunnelFrame.addEventListener('load', () => {
    windTunnelReady = false;
    // In case embed-ready message is missed, try a direct push shortly after load.
    setTimeout(() => pushSpecToWindTunnel(lastCompiledSpec), 250);
  });

  window.addEventListener('message', (event) => {
    if (event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type === 'gpuLabEmbedReady') {
      windTunnelReady = true;
      pushSpecToWindTunnel(lastCompiledSpec);
      return;
    }

    if (data.type === 'gpuLabEmbedResetAck') {
      if (data.ok === false) {
        out.textContent = JSON.stringify({
          windTunnel: 'reset failed',
          error: String(data.error || 'unknown error from embedded GPU lab'),
          hint: 'Check browser console for stack and confirm WebGPU is available.',
        }, null, 2);
      }
      return;
    }
  });
}

syncFieldPanelVisibility();
drawFields();
compileNow();
