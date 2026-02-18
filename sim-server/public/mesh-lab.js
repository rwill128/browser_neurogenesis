import { compileFieldToMesh } from '/field-to-structure-core.js';
import { createCreatureSpecFromMesh, parseCreatureSpec, buildMembraneRingsFromSoftField } from '/creature-spec.js';

const paintCanvas = document.getElementById('paint');
const densityCanvas = document.getElementById('densityPaint');
const membraneEdgeCanvas = document.getElementById('membraneEdgePaint');
const membraneShapeCanvas = document.getElementById('membraneShapePaint');
const softPermeabilityCanvas = document.getElementById('softPermeabilityPaint');
const softEdgeDyeModeCanvas = document.getElementById('softEdgeDyeModePaint');
const softEdgeVelocityModeCanvas = document.getElementById('softEdgeVelocityModePaint');
const softEdgeMomentumModeCanvas = document.getElementById('softEdgeMomentumModePaint');
const rigidPermeabilityCanvas = document.getElementById('rigidPermeabilityPaint');
const rigidEdgeVelocityModeCanvas = document.getElementById('rigidEdgeVelocityModePaint');
const rigidEdgeDyeModeCanvas = document.getElementById('rigidEdgeDyeModePaint');
const meshCanvas = document.getElementById('mesh');
const windTunnelFrame = document.getElementById('windTunnelFrame');
const runtimeSolverPathEls = Array.from(document.querySelectorAll('input[name="runtimeSolverPath"]'));
const pctx = paintCanvas.getContext('2d');
const dctx = densityCanvas.getContext('2d');
const ectx = membraneEdgeCanvas.getContext('2d');
const sctx = membraneShapeCanvas.getContext('2d');
const spctx = softPermeabilityCanvas.getContext('2d');
const sdctx = softEdgeDyeModeCanvas.getContext('2d');
const svctx = softEdgeVelocityModeCanvas.getContext('2d');
const smctx = softEdgeMomentumModeCanvas.getContext('2d');
const prctx = rigidPermeabilityCanvas.getContext('2d');
const rvctx = rigidEdgeVelocityModeCanvas.getContext('2d');
const cctx = rigidEdgeDyeModeCanvas.getContext('2d');
const mctx = meshCanvas.getContext('2d');
const modeEl = document.getElementById('paintMode');
const fieldPaintTargetEl = document.getElementById('fieldPaintTarget');
const brushEl = document.getElementById('brush');
const brushValueEl = document.getElementById('brushValue');
const thresholdEl = document.getElementById('threshold');
const thresholdValueEl = document.getElementById('thresholdValue');
const softDensityPaintEl = document.getElementById('softDensityPaint');
const softDensityPaintValueEl = document.getElementById('softDensityPaintValue');
const membraneEdgePaintEl = document.getElementById('membraneEdgePaintValue');
const membraneEdgePaintValueEl = document.getElementById('membraneEdgePaintValueLabel');
const membraneShapePaintEl = document.getElementById('membraneShapePaintValue');
const membraneShapePaintValueEl = document.getElementById('membraneShapePaintValueLabel');
const softPermeabilityPaintEl = document.getElementById('softPermeabilityPaintValue');
const softPermeabilityPaintValueEl = document.getElementById('softPermeabilityPaintValueLabel');
const softEdgeDyeChannelEl = document.getElementById('softEdgeDyeChannel');
const softEdgeDyeModeEl = document.getElementById('softEdgeDyeMode');
const softEdgeDyePaintEl = document.getElementById('softEdgeDyePaintValue');
const softEdgeDyePaintValueEl = document.getElementById('softEdgeDyePaintValueLabel');
const softEdgeVelocityModeEl = document.getElementById('softEdgeVelocityMode');
const softEdgeMomentumPaintEl = document.getElementById('softEdgeMomentumPaintValue');
const softEdgeMomentumPaintValueEl = document.getElementById('softEdgeMomentumPaintValueLabel');
const rigidPermeabilityPaintEl = document.getElementById('rigidPermeabilityPaintValue');
const rigidPermeabilityPaintValueEl = document.getElementById('rigidPermeabilityPaintValueLabel');
const rigidEdgeVelocityModeEl = document.getElementById('rigidEdgeVelocityMode');
const rigidEdgeDyeChannelEl = document.getElementById('rigidEdgeDyeChannel');
const rigidEdgeDyeModeEl = document.getElementById('rigidEdgeDyeMode');
const rigidEdgeDyePaintEl = document.getElementById('rigidEdgeDyePaintValue');
const rigidEdgeDyePaintValueEl = document.getElementById('rigidEdgeDyePaintValueLabel');
const rigidCompileModeEl = document.getElementById('rigidCompileMode');
const rigidPrimitiveSideMinEl = document.getElementById('rigidPrimitiveSideMin');
const rigidPrimitiveSideMinValueEl = document.getElementById('rigidPrimitiveSideMinValue');
const rigidPrimitiveSideMaxEl = document.getElementById('rigidPrimitiveSideMax');
const rigidPrimitiveSideMaxValueEl = document.getElementById('rigidPrimitiveSideMaxValue');
const softInfillModeEl = document.getElementById('softInfillMode');
const softMinCellSizeEl = document.getElementById('softMinCellSize');
const softMinCellSizeValueEl = document.getElementById('softMinCellSizeValue');
const softBoundaryRingEl = document.getElementById('softBoundaryRing');
const softSolverModeEl = document.getElementById('softSolverMode');
const membraneMinEdgeLengthEl = document.getElementById('membraneMinEdgeLength');
const membraneMinEdgeLengthValueEl = document.getElementById('membraneMinEdgeLengthValue');
const membraneMaxEdgeLengthEl = document.getElementById('membraneMaxEdgeLength');
const membraneMaxEdgeLengthValueEl = document.getElementById('membraneMaxEdgeLengthValue');
const clearBtn = document.getElementById('clearBtn');
const compileBtn = document.getElementById('compileBtn');
const randomizeBtn = document.getElementById('randomizeBtn');
const randomizeTraitBtn = document.getElementById('randomizeTraitBtn');
const randomizeDensityBtn = document.getElementById('randomizeDensityBtn');
const randomizeMembraneEdgeBtn = document.getElementById('randomizeMembraneEdgeBtn');
const randomizeMembraneShapeBtn = document.getElementById('randomizeMembraneShapeBtn');
const randomizeSoftPermeabilityBtn = document.getElementById('randomizeSoftPermeabilityBtn');
const randomizeSoftEdgeDyeModeBtn = document.getElementById('randomizeSoftEdgeDyeModeBtn');
const randomizeSoftEdgeVelocityModeBtn = document.getElementById('randomizeSoftEdgeVelocityModeBtn');
const randomizeSoftEdgeMomentumModeBtn = document.getElementById('randomizeSoftEdgeMomentumModeBtn');
const randomizeRigidPermeabilityBtn = document.getElementById('randomizeRigidPermeabilityBtn');
const randomizeRigidEdgeVelocityModeBtn = document.getElementById('randomizeRigidEdgeVelocityModeBtn');
const randomizeRigidEdgeDyeModeBtn = document.getElementById('randomizeRigidEdgeDyeModeBtn');
const randomizeSoftDyeBlockRBtn = document.getElementById('randomizeSoftDyeBlockRBtn');
const randomizeSoftDyePassRBtn = document.getElementById('randomizeSoftDyePassRBtn');
const randomizeSoftDyeEatRBtn = document.getElementById('randomizeSoftDyeEatRBtn');
const randomizeSoftDyeBlockGBtn = document.getElementById('randomizeSoftDyeBlockGBtn');
const randomizeSoftDyePassGBtn = document.getElementById('randomizeSoftDyePassGBtn');
const randomizeSoftDyeEatGBtn = document.getElementById('randomizeSoftDyeEatGBtn');
const randomizeSoftDyeBlockBBtn = document.getElementById('randomizeSoftDyeBlockBBtn');
const randomizeSoftDyePassBBtn = document.getElementById('randomizeSoftDyePassBBtn');
const randomizeSoftDyeEatBBtn = document.getElementById('randomizeSoftDyeEatBBtn');
const randomizeRigidDyeBlockRBtn = document.getElementById('randomizeRigidDyeBlockRBtn');
const randomizeRigidDyePassRBtn = document.getElementById('randomizeRigidDyePassRBtn');
const randomizeRigidDyeEatRBtn = document.getElementById('randomizeRigidDyeEatRBtn');
const randomizeRigidDyeBlockGBtn = document.getElementById('randomizeRigidDyeBlockGBtn');
const randomizeRigidDyePassGBtn = document.getElementById('randomizeRigidDyePassGBtn');
const randomizeRigidDyeEatGBtn = document.getElementById('randomizeRigidDyeEatGBtn');
const randomizeRigidDyeBlockBBtn = document.getElementById('randomizeRigidDyeBlockBBtn');
const randomizeRigidDyePassBBtn = document.getElementById('randomizeRigidDyePassBBtn');
const randomizeRigidDyeEatBBtn = document.getElementById('randomizeRigidDyeEatBBtn');
const randomFieldPresetEl = document.getElementById('randomFieldPreset');
const randomFieldSeedEl = document.getElementById('randomFieldSeed');
const randomFieldSeedValueEl = document.getElementById('randomFieldSeedValue');
const exportBtn = document.getElementById('exportBtn');
const importBtn = document.getElementById('importBtn');
const importFile = document.getElementById('importFile');
const out = document.getElementById('out');
const segmentStatsOut = document.getElementById('segmentStatsOut');
const segmentStatsList = document.getElementById('segmentStatsList');
const copySegmentStatsBtn = document.getElementById('copySegmentStatsBtn');
const copyOutBtn = document.getElementById('copyOutBtn');
const windEmitterStrengthEl = document.getElementById('windEmitterStrength');
const windEmitterStrengthValueEl = document.getElementById('windEmitterStrengthValue');
const windEmitterJetVyEl = document.getElementById('windEmitterJetVy');
const windEmitterJetVyValueEl = document.getElementById('windEmitterJetVyValue');
const windEmitterRadiusEl = document.getElementById('windEmitterRadius');
const windEmitterRadiusValueEl = document.getElementById('windEmitterRadiusValue');
const windEmitterYFractionEl = document.getElementById('windEmitterYFraction');
const windEmitterYFractionValueEl = document.getElementById('windEmitterYFractionValue');
const windEmitterSpinEl = document.getElementById('windEmitterSpin');
const windEmitterSpinValueEl = document.getElementById('windEmitterSpinValue');
const windEmitterCurlGainEl = document.getElementById('windEmitterCurlGain');
const windEmitterCurlGainValueEl = document.getElementById('windEmitterCurlGainValue');
const windEmitterDriftGainEl = document.getElementById('windEmitterDriftGain');
const windEmitterDriftGainValueEl = document.getElementById('windEmitterDriftGainValue');
const windEmitterChaosGainEl = document.getElementById('windEmitterChaosGain');
const windEmitterChaosGainValueEl = document.getElementById('windEmitterChaosGainValue');
const windEmitterWobbleAmpEl = document.getElementById('windEmitterWobbleAmp');
const windEmitterWobbleAmpValueEl = document.getElementById('windEmitterWobbleAmpValue');
const windEmitterWobbleFreqEl = document.getElementById('windEmitterWobbleFreq');
const windEmitterWobbleFreqValueEl = document.getElementById('windEmitterWobbleFreqValue');
const windEmitterLockPositionEl = document.getElementById('windEmitterLockPosition');
const applyWindEmitterBtn = document.getElementById('applyWindEmitterBtn');

const W = 128, H = 128;
const rigid = new Float32Array(W * H);
const soft = new Float32Array(W * H);
const softDensity = new Float32Array(W * H).fill(0.5);
const membraneEdgeMap = new Float32Array(W * H).fill(0.5);
const membraneShapeMap = new Float32Array(W * H).fill(1.0);
const softPermeabilityMap = new Float32Array(W * H).fill(0.0);

// Soft dye policy winner-take-all fields (3 behaviors × 3 channels).
const softEdgeDyeBlockMapR = new Float32Array(W * H).fill(0.0);
const softEdgeDyeBlockMapG = new Float32Array(W * H).fill(0.0);
const softEdgeDyeBlockMapB = new Float32Array(W * H).fill(0.0);
const softEdgeDyePassMapR = new Float32Array(W * H).fill(0.0);
const softEdgeDyePassMapG = new Float32Array(W * H).fill(0.0);
const softEdgeDyePassMapB = new Float32Array(W * H).fill(0.0);
const softEdgeDyeEatMapR = new Float32Array(W * H).fill(0.0);
const softEdgeDyeEatMapG = new Float32Array(W * H).fill(0.0);
const softEdgeDyeEatMapB = new Float32Array(W * H).fill(0.0);

// Legacy/preview packed soft dye mode maps (derived from winner fields).
const softEdgeDyeModeMapR = new Float32Array(W * H).fill(0.0);
const softEdgeDyeModeMapG = new Float32Array(W * H).fill(0.0);
const softEdgeDyeModeMapB = new Float32Array(W * H).fill(0.0);

const softEdgeVelocityModeMap = new Float32Array(W * H).fill(0.0);
const softEdgeMomentumModeMap = new Float32Array(W * H).fill(1.0);
const rigidPermeabilityMap = new Float32Array(W * H).fill(0.0);
const rigidEdgeVelocityMap = new Float32Array(W * H).fill(0.0);

// Rigid dye policy winner-take-all fields (3 behaviors × 3 channels).
const rigidEdgeDyeBlockMapR = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyeBlockMapG = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyeBlockMapB = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyePassMapR = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyePassMapG = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyePassMapB = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyeEatMapR = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyeEatMapG = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyeEatMapB = new Float32Array(W * H).fill(0.0);

// Legacy/preview packed rigid dye mode maps (derived from winner fields).
const rigidEdgeDyeModeMapR = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyeModeMapG = new Float32Array(W * H).fill(0.0);
const rigidEdgeDyeModeMapB = new Float32Array(W * H).fill(0.0);

// Backward-compatible legacy absorb-only fields (derived from rigid EAT maps).
const rigidEdgeConsumeMapR = new Float32Array(W * H).fill(0.0);
const rigidEdgeConsumeMapG = new Float32Array(W * H).fill(0.0);
const rigidEdgeConsumeMapB = new Float32Array(W * H).fill(0.0);
let lastMesh = null;
let lastCompiledSpec = null;
let lastCompiledFields = null;
let compileRevision = 0;
let windTunnelReady = false;
let highlightedSegmentId = null;
let lastCompileHint = '';

const fieldPanels = Array.from(document.querySelectorAll('.field-panel'));
const activeFieldHelpEl = document.getElementById('activeFieldHelp');

function idx(x, y) { return y * W + x; }

async function copyTextToClipboard(text) {
  const payload = String(text || '');
  if (!payload) return false;
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(payload);
      return true;
    }
  } catch {}

  try {
    const ta = document.createElement('textarea');
    ta.value = payload;
    ta.setAttribute('readonly', 'readonly');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return !!ok;
  } catch {
    return false;
  }
}

function flashCopyButtonState(btn, labelOk = 'Copied', labelErr = 'Copy failed') {
  if (!btn) return;
  const prev = btn.textContent;
  return (ok) => {
    btn.textContent = ok ? labelOk : labelErr;
    setTimeout(() => { btn.textContent = prev; }, 1100);
  };
}

const EDGE_DYE_BLOCK_SCALAR = 0.0;
const EDGE_DYE_PASS_SCALAR = 0.5;
const EDGE_DYE_EAT_SCALAR = 1.0;

function edgeDyeModeToScalar(mode) {
  const m = String(mode || 'block').toLowerCase();
  if (m === 'eat' || m === 'absorb') return EDGE_DYE_EAT_SCALAR;
  if (m === 'pass') return EDGE_DYE_PASS_SCALAR;
  return EDGE_DYE_BLOCK_SCALAR;
}

function edgeVelocityModeToScalar(mode) {
  return String(mode || 'block').toLowerCase() === 'pass' ? 1.0 : 0.0;
}

function edgeModeScalarPreview(v) {
  const n = Math.max(0, Math.min(1, Number(v) || 0));
  if (n >= 0.67) return 255; // EAT=true
  return 160; // NO-OP (EAT=false)
}

function syncSoftDyeModeMapsFromWinnerFields() {
  for (let i = 0; i < W * H; i++) {
    softEdgeDyeModeMapR[i] = (Number(softEdgeDyeEatMapR[i]) || 0) >= 0.5 ? EDGE_DYE_EAT_SCALAR : EDGE_DYE_PASS_SCALAR;
    softEdgeDyeModeMapG[i] = (Number(softEdgeDyeEatMapG[i]) || 0) >= 0.5 ? EDGE_DYE_EAT_SCALAR : EDGE_DYE_PASS_SCALAR;
    softEdgeDyeModeMapB[i] = (Number(softEdgeDyeEatMapB[i]) || 0) >= 0.5 ? EDGE_DYE_EAT_SCALAR : EDGE_DYE_PASS_SCALAR;
  }
}

function syncRigidDyeModeMapsFromWinnerFields() {
  for (let i = 0; i < W * H; i++) {
    rigidEdgeDyeModeMapR[i] = (Number(rigidEdgeDyeEatMapR[i]) || 0) >= 0.5 ? EDGE_DYE_EAT_SCALAR : EDGE_DYE_PASS_SCALAR;
    rigidEdgeDyeModeMapG[i] = (Number(rigidEdgeDyeEatMapG[i]) || 0) >= 0.5 ? EDGE_DYE_EAT_SCALAR : EDGE_DYE_PASS_SCALAR;
    rigidEdgeDyeModeMapB[i] = (Number(rigidEdgeDyeEatMapB[i]) || 0) >= 0.5 ? EDGE_DYE_EAT_SCALAR : EDGE_DYE_PASS_SCALAR;

    // Back-compat absorb-only maps (legacy payloads/tests use these names).
    rigidEdgeConsumeMapR[i] = Math.max(0, Math.min(1, Number(rigidEdgeDyeEatMapR[i]) || 0));
    rigidEdgeConsumeMapG[i] = Math.max(0, Math.min(1, Number(rigidEdgeDyeEatMapG[i]) || 0));
    rigidEdgeConsumeMapB[i] = Math.max(0, Math.min(1, Number(rigidEdgeDyeEatMapB[i]) || 0));
  }
}

function populateWinnerFieldsFromModeScalarMap(modeMap, blockMap, passMap, eatMap) {
  if (!modeMap || modeMap.length !== W * H) return;
  for (let i = 0; i < W * H; i++) {
    const s = Math.max(0, Math.min(1, Number(modeMap[i]) || 0));
    blockMap[i] = 0;
    passMap[i] = 0;
    eatMap[i] = s >= 0.67 ? s : 0;
  }
}

function sampleSegmentAverageMap(field, ax, ay, bx, by, fallback = 0) {
  if (!(field instanceof Float32Array) || field.length !== W * H) return fallback;
  const x0 = Number(ax) || 0;
  const y0 = Number(ay) || 0;
  const x1 = Number(bx) || 0;
  const y1 = Number(by) || 0;
  const len = Math.hypot(x1 - x0, y1 - y0);
  const samples = Math.max(2, Math.min(256, Math.ceil(len * 2) + 1));
  let sum = 0;
  for (let i = 0; i < samples; i++) {
    const t = samples <= 1 ? 0 : (i / (samples - 1));
    const x = x0 + (x1 - x0) * t;
    const y = y0 + (y1 - y0) * t;
    sum += sampleMapBilinear(field, x, y, fallback);
  }
  return sum / samples;
}

function dyeModeNameFromCode(mode) {
  const m = Number(mode);
  return m === 2 ? 'EAT' : 'NO-OP';
}

function velocityModeNameFromCode(mode) {
  return Number(mode) === 0 ? 'PASS' : 'BLOCK';
}

function normalizeVelocityModeCode(mode, fallback = 1) {
  const m = Number(mode);
  if (m === 0) return 0;
  if (m === 1) return 1;
  return Number(fallback) === 0 ? 0 : 1;
}

function normalizeDyeModeCode(mode, fallback = 1) {
  const m = Number(mode);
  if (m === 2) return 2; // EAT
  if (m === 1) return 1; // NO-OP
  return Number(fallback) === 2 ? 2 : 1;
}

function applySegmentPolicyOverride(seg, { velocityCode, dyeCodes } = {}) {
  if (!lastCompiledSpec || !seg) return false;

  if (seg.type === 'rigid') {
    const rb = lastCompiledSpec.rigidBodies?.[seg.bodyIndex];
    const edgeIndex = Number(seg.segmentIndex) | 0;
    if (!rb || edgeIndex < 0) return false;

    rb.edgeVelocityMode = Array.isArray(rb.edgeVelocityMode) ? rb.edgeVelocityMode : [];
    const prevVel = rb.edgeVelocityMode[edgeIndex];
    rb.edgeVelocityMode[edgeIndex] = normalizeVelocityModeCode(velocityCode, prevVel);

    rb.edgeDyeMode = Array.isArray(rb.edgeDyeMode) ? rb.edgeDyeMode : [];
    const prevDye = Array.isArray(rb.edgeDyeMode[edgeIndex]) ? rb.edgeDyeMode[edgeIndex] : [1, 1, 1];
    const nextDye = Array.isArray(dyeCodes) ? dyeCodes : prevDye;
    rb.edgeDyeMode[edgeIndex] = [
      normalizeDyeModeCode(nextDye[0], prevDye[0]),
      normalizeDyeModeCode(nextDye[1], prevDye[1]),
      normalizeDyeModeCode(nextDye[2], prevDye[2]),
    ];
  } else if (seg.type === 'soft') {
    const sb = lastCompiledSpec.softBodies?.[seg.softBodyIndex];
    const springIndex = Number(seg.springIndex) | 0;
    const spring = sb?.springs?.[springIndex];
    if (!sb || !Array.isArray(spring) || springIndex < 0) return false;

    spring[5] = normalizeVelocityModeCode(velocityCode, spring[5]);
    const prevDye = Array.isArray(spring[4]) ? spring[4] : [1, 1, 1];
    const nextDye = Array.isArray(dyeCodes) ? dyeCodes : prevDye;
    spring[4] = [
      normalizeDyeModeCode(nextDye[0], prevDye[0]),
      normalizeDyeModeCode(nextDye[1], prevDye[1]),
      normalizeDyeModeCode(nextDye[2], prevDye[2]),
    ];
  } else {
    return false;
  }

  updateSegmentStatsPreview(lastCompiledSpec);
  pushSpecToWindTunnel(lastCompiledSpec);
  return true;
}

function computeChannelEatStats(ax, ay, bx, by, eatMap, eatThreshold = 0.5) {
  const eatAvg = sampleSegmentAverageMap(eatMap, ax, ay, bx, by, 0);
  return {
    eatAvg: +eatAvg.toFixed(4),
    eats: eatAvg >= eatThreshold,
  };
}

function updateSegmentStatsPreview(spec) {
  if (!segmentStatsOut) return;
  if (!spec || !Array.isArray(spec.rigidBodies) || !Array.isArray(spec.softBodies)) {
    if (segmentStatsList) segmentStatsList.textContent = 'No segments yet. Compile a mesh first.';
    segmentStatsOut.textContent = 'No segment stats yet. Compile a mesh first.';
    return;
  }

  const segments = [];
  let softGlobalIndex = 0;

  for (let rbi = 0; rbi < (spec.rigidBodies || []).length; rbi++) {
    const rb = spec.rigidBodies[rbi];
    const hull = rb?.hull || [];
    for (let ei = 0; ei < hull.length; ei++) {
      const a = hull[ei];
      const b = hull[(ei + 1) % hull.length];
      if (!a || !b) continue;
      const ax = Number(a.x) || 0;
      const ay = Number(a.y) || 0;
      const bx = Number(b.x) || 0;
      const by = Number(b.y) || 0;
      const dye = Array.isArray(rb?.edgeDyeMode?.[ei]) ? rb.edgeDyeMode[ei] : [1, 1, 1];
      const vel = Array.isArray(rb?.edgeVelocityMode) ? Number(rb.edgeVelocityMode[ei]) : 1;
      const momentum = Array.isArray(rb?.edgeMomentumCoupling)
        ? Number(rb.edgeMomentumCoupling[ei])
        : (Array.isArray(rb?.edgeMomentumTransfer) ? Number(rb.edgeMomentumTransfer[ei]) : 1);

      segments.push({
        id: `R${rbi}:${ei}`,
        type: 'rigid',
        bodyIndex: rbi,
        segmentIndex: ei,
        midpoint: { x: +(((ax + bx) * 0.5).toFixed(3)), y: +(((ay + by) * 0.5).toFixed(3)) },
        length: +(Math.hypot(bx - ax, by - ay).toFixed(3)),
        velocityMode: velocityModeNameFromCode(vel),
        momentumCoupling: +(Math.max(0, Math.min(1, Number.isFinite(momentum) ? momentum : 1)).toFixed(3)),
        dyeMode: {
          r: dyeModeNameFromCode(dye[0]),
          g: dyeModeNameFromCode(dye[1]),
          b: dyeModeNameFromCode(dye[2]),
        },
        eatStats: {
          r: computeChannelEatStats(ax, ay, bx, by, rigidEdgeDyeEatMapR),
          g: computeChannelEatStats(ax, ay, bx, by, rigidEdgeDyeEatMapG),
          b: computeChannelEatStats(ax, ay, bx, by, rigidEdgeDyeEatMapB),
        },
      });
    }
  }

  for (let sbi = 0; sbi < (spec.softBodies || []).length; sbi++) {
    const sb = spec.softBodies[sbi];
    const nodes = sb?.nodes || [];
    const springs = sb?.springs || [];
    for (let spi = 0; spi < springs.length; spi++) {
      const sp = springs[spi];
      if (!Array.isArray(sp) || sp.length < 2) continue;
      const a = nodes[Number(sp[0])];
      const b = nodes[Number(sp[1])];
      if (!a || !b) continue;
      const ax = Number(a.x) || 0;
      const ay = Number(a.y) || 0;
      const bx = Number(b.x) || 0;
      const by = Number(b.y) || 0;
      const dye = Array.isArray(sp[4]) ? sp[4] : [1, 1, 1];
      const vel = Number(sp[5]);
      const momentum = Number(sp[6]);

      segments.push({
        id: `S${softGlobalIndex}`,
        type: 'soft',
        softBodyIndex: sbi,
        springIndex: spi,
        midpoint: { x: +(((ax + bx) * 0.5).toFixed(3)), y: +(((ay + by) * 0.5).toFixed(3)) },
        length: +(Math.hypot(bx - ax, by - ay).toFixed(3)),
        velocityMode: velocityModeNameFromCode(vel),
        momentumCoupling: +(Math.max(0, Math.min(1, Number.isFinite(momentum) ? momentum : 1)).toFixed(3)),
        dyeMode: {
          r: dyeModeNameFromCode(dye[0]),
          g: dyeModeNameFromCode(dye[1]),
          b: dyeModeNameFromCode(dye[2]),
        },
        eatStats: {
          r: computeChannelEatStats(ax, ay, bx, by, softEdgeDyeEatMapR),
          g: computeChannelEatStats(ax, ay, bx, by, softEdgeDyeEatMapG),
          b: computeChannelEatStats(ax, ay, bx, by, softEdgeDyeEatMapB),
        },
      });
      softGlobalIndex += 1;
    }
  }

  renderSegmentStatsList(segments);

  const payload = {
    segmentIdLegend: {
      rigid: 'R<rigidBodyIndex>:<edgeIndex>',
      soft: 'S<globalSoftSpringIndex>',
    },
    summary: {
      rigidBodies: spec.rigidBodies.length,
      softBodies: spec.softBodies.length,
      segments: segments.length,
      rigidSegments: segments.filter((s) => s.type === 'rigid').length,
      softSegments: segments.filter((s) => s.type === 'soft').length,
    },
    segments,
  };
  if (segments.length === 0 && lastCompileHint) payload.compileHint = lastCompileHint;
  segmentStatsOut.textContent = JSON.stringify(payload, null, 2);
}

function isolateLargestContiguousTraitBody(rigidField, softField, threshold = 0.35) {
  const total = W * H;
  const th = Math.max(0, Math.min(1, Number(threshold) || 0.35));
  const occupied = new Uint8Array(total);
  let maxOccupancy = 0;

  for (let i = 0; i < total; i++) {
    const rv = Number(rigidField?.[i]) || 0;
    const sv = Number(softField?.[i]) || 0;
    const occ = Math.max(rv, sv);
    if (occ > maxOccupancy) maxOccupancy = occ;
    if (occ >= th) occupied[i] = 1;
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
      maxOccupancy,
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
    maxOccupancy,
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
    softEdgeDyeMode: true,
    softDyeBlockR: true,
    softDyePassR: true,
    softDyeEatR: true,
    softDyeBlockG: true,
    softDyePassG: true,
    softDyeEatG: true,
    softDyeBlockB: true,
    softDyePassB: true,
    softDyeEatB: true,
    softEdgeVelocityMode: true,
    softEdgeMomentumMode: true,
    rigidPermeability: true,
    rigidEdgeDyeMode: true,
    rigidDyeBlockR: true,
    rigidDyePassR: true,
    rigidDyeEatR: true,
    rigidDyeBlockG: true,
    rigidDyePassG: true,
    rigidDyeEatG: true,
    rigidDyeBlockB: true,
    rigidDyePassB: true,
    rigidDyeEatB: true,
    rigidEdgeVelocityMode: true,
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
    softEdgeDyeMode: !!targets?.softEdgeDyeMode,
    softDyeBlockR: !!targets?.softDyeBlockR,
    softDyePassR: !!targets?.softDyePassR,
    softDyeEatR: !!targets?.softDyeEatR,
    softDyeBlockG: !!targets?.softDyeBlockG,
    softDyePassG: !!targets?.softDyePassG,
    softDyeEatG: !!targets?.softDyeEatG,
    softDyeBlockB: !!targets?.softDyeBlockB,
    softDyePassB: !!targets?.softDyePassB,
    softDyeEatB: !!targets?.softDyeEatB,
    softEdgeVelocityMode: !!targets?.softEdgeVelocityMode,
    softEdgeMomentumMode: !!targets?.softEdgeMomentumMode,
    rigidPermeability: !!targets?.rigidPermeability,
    rigidEdgeDyeMode: !!targets?.rigidEdgeDyeMode,
    rigidDyeBlockR: !!targets?.rigidDyeBlockR,
    rigidDyePassR: !!targets?.rigidDyePassR,
    rigidDyeEatR: !!targets?.rigidDyeEatR,
    rigidDyeBlockG: !!targets?.rigidDyeBlockG,
    rigidDyePassG: !!targets?.rigidDyePassG,
    rigidDyeEatG: !!targets?.rigidDyeEatG,
    rigidDyeBlockB: !!targets?.rigidDyeBlockB,
    rigidDyePassB: !!targets?.rigidDyePassB,
    rigidDyeEatB: !!targets?.rigidDyeEatB,
    rigidEdgeVelocityMode: !!targets?.rigidEdgeVelocityMode,
  };

  const rigidParams = makeRandomPatternParams(rand, 11);
  const softParams = makeRandomPatternParams(rand, 23);
  const densityParams = makeRandomPatternParams(rand, 37);
  const edgeParams = makeRandomPatternParams(rand, 53);
  const shapeParams = makeRandomPatternParams(rand, 71);
  const softPermParams = makeRandomPatternParams(rand, 79);
  const softDyeBlockRParams = makeRandomPatternParams(rand, 81);
  const softDyePassRParams = makeRandomPatternParams(rand, 82);
  const softDyeEatRParams = makeRandomPatternParams(rand, 83);
  const softDyeBlockGParams = makeRandomPatternParams(rand, 84);
  const softDyePassGParams = makeRandomPatternParams(rand, 85);
  const softDyeEatGParams = makeRandomPatternParams(rand, 86);
  const softDyeBlockBParams = makeRandomPatternParams(rand, 87);
  const softDyePassBParams = makeRandomPatternParams(rand, 88);
  const softDyeEatBParams = makeRandomPatternParams(rand, 89);
  const softVelocityParams = makeRandomPatternParams(rand, 90);
  const softMomentumParams = makeRandomPatternParams(rand, 91);
  const rigidVelocityParams = makeRandomPatternParams(rand, 93);
  const permParams = makeRandomPatternParams(rand, 97);
  const rigidDyeBlockRParams = makeRandomPatternParams(rand, 101);
  const rigidDyePassRParams = makeRandomPatternParams(rand, 102);
  const rigidDyeEatRParams = makeRandomPatternParams(rand, 103);
  const rigidDyeBlockGParams = makeRandomPatternParams(rand, 104);
  const rigidDyePassGParams = makeRandomPatternParams(rand, 105);
  const rigidDyeEatGParams = makeRandomPatternParams(rand, 106);
  const rigidDyeBlockBParams = makeRandomPatternParams(rand, 107);
  const rigidDyePassBParams = makeRandomPatternParams(rand, 108);
  const rigidDyeEatBParams = makeRandomPatternParams(rand, 109);

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

      const softMask = smoothstep(0.52, 0.85, softSignal);
      const rigidMask = smoothstep(0.58, 0.9, rigidSignal);

      const randomDyeScalar = (params, isSoft) => {
        const base = samplePatternValue('wave-interference', nx, ny, params);
        const gate = isSoft ? (soft[i] > 0.12 ? softMask : 0) : (rigid[i] > 0.1 ? rigidMask : 0);
        return Math.max(0, Math.min(1, base * gate));
      };

      if (targetFlags.softEdgeDyeMode || targetFlags.softDyeBlockR) {
        softEdgeDyeBlockMapR[i] = randomDyeScalar(softDyeBlockRParams, true);
      }
      if (targetFlags.softEdgeDyeMode || targetFlags.softDyePassR) {
        softEdgeDyePassMapR[i] = randomDyeScalar(softDyePassRParams, true);
      }
      if (targetFlags.softEdgeDyeMode || targetFlags.softDyeEatR) {
        softEdgeDyeEatMapR[i] = randomDyeScalar(softDyeEatRParams, true);
      }
      if (targetFlags.softEdgeDyeMode || targetFlags.softDyeBlockG) {
        softEdgeDyeBlockMapG[i] = randomDyeScalar(softDyeBlockGParams, true);
      }
      if (targetFlags.softEdgeDyeMode || targetFlags.softDyePassG) {
        softEdgeDyePassMapG[i] = randomDyeScalar(softDyePassGParams, true);
      }
      if (targetFlags.softEdgeDyeMode || targetFlags.softDyeEatG) {
        softEdgeDyeEatMapG[i] = randomDyeScalar(softDyeEatGParams, true);
      }
      if (targetFlags.softEdgeDyeMode || targetFlags.softDyeBlockB) {
        softEdgeDyeBlockMapB[i] = randomDyeScalar(softDyeBlockBParams, true);
      }
      if (targetFlags.softEdgeDyeMode || targetFlags.softDyePassB) {
        softEdgeDyePassMapB[i] = randomDyeScalar(softDyePassBParams, true);
      }
      if (targetFlags.softEdgeDyeMode || targetFlags.softDyeEatB) {
        softEdgeDyeEatMapB[i] = randomDyeScalar(softDyeEatBParams, true);
      }

      if (targetFlags.softEdgeVelocityMode) {
        const v = samplePatternValue('sine-lines', nx, ny, softVelocityParams);
        softEdgeVelocityModeMap[i] = (soft[i] > 0.12 && v >= 0.58) ? 1 : 0;
      }

      if (targetFlags.softEdgeMomentumMode) {
        const m = samplePatternValue('flow-ridges', nx, ny, softMomentumParams);
        softEdgeMomentumModeMap[i] = soft[i] > 0.12
          ? Math.max(0, Math.min(1, 0.2 + 0.8 * m))
          : 1;
      }

      if (targetFlags.rigidPermeability) {
        const permV = samplePatternValue('sine-lines', nx, ny, permParams);
        rigidPermeabilityMap[i] = rigid[i] > 0.12 ? smoothstep(0.56, 0.84, permV) : 0;
      }

      if (targetFlags.rigidEdgeVelocityMode) {
        const v = samplePatternValue('sine-lines', nx, ny, rigidVelocityParams);
        rigidEdgeVelocityMap[i] = (rigid[i] > 0.12 && v >= 0.58) ? 1 : 0;
      }

      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyeBlockR) {
        rigidEdgeDyeBlockMapR[i] = randomDyeScalar(rigidDyeBlockRParams, false);
      }
      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyePassR) {
        rigidEdgeDyePassMapR[i] = randomDyeScalar(rigidDyePassRParams, false);
      }
      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyeEatR) {
        rigidEdgeDyeEatMapR[i] = randomDyeScalar(rigidDyeEatRParams, false);
      }
      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyeBlockG) {
        rigidEdgeDyeBlockMapG[i] = randomDyeScalar(rigidDyeBlockGParams, false);
      }
      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyePassG) {
        rigidEdgeDyePassMapG[i] = randomDyeScalar(rigidDyePassGParams, false);
      }
      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyeEatG) {
        rigidEdgeDyeEatMapG[i] = randomDyeScalar(rigidDyeEatGParams, false);
      }
      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyeBlockB) {
        rigidEdgeDyeBlockMapB[i] = randomDyeScalar(rigidDyeBlockBParams, false);
      }
      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyePassB) {
        rigidEdgeDyePassMapB[i] = randomDyeScalar(rigidDyePassBParams, false);
      }
      if (targetFlags.rigidEdgeDyeMode || targetFlags.rigidDyeEatB) {
        rigidEdgeDyeEatMapB[i] = randomDyeScalar(rigidDyeEatBParams, false);
      }
    }
  }

  syncSoftDyeModeMapsFromWinnerFields();
  syncRigidDyeModeMapsFromWinnerFields();
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
      softEdgeDyeBlockMapR,
      softEdgeDyeBlockMapG,
      softEdgeDyeBlockMapB,
      softEdgeDyePassMapR,
      softEdgeDyePassMapG,
      softEdgeDyePassMapB,
      softEdgeDyeEatMapR,
      softEdgeDyeEatMapG,
      softEdgeDyeEatMapB,
      // Backward-compatible packed mode aliases.
      softEdgeDyeModeMapR,
      softEdgeDyeModeMapG,
      softEdgeDyeModeMapB,
      softEdgeVelocityModeMap,
      softEdgeMomentumModeMap,
      rigidPermeabilityMap,
      rigidEdgeDyeBlockMapR,
      rigidEdgeDyeBlockMapG,
      rigidEdgeDyeBlockMapB,
      rigidEdgeDyePassMapR,
      rigidEdgeDyePassMapG,
      rigidEdgeDyePassMapB,
      rigidEdgeDyeEatMapR,
      rigidEdgeDyeEatMapG,
      rigidEdgeDyeEatMapB,
      // Backward-compatible packed mode + consume aliases.
      rigidEdgeDyeModeMapR,
      rigidEdgeDyeModeMapG,
      rigidEdgeDyeModeMapB,
      rigidEdgeVelocityMap,
      rigidEdgeVelocityModeMap: rigidEdgeVelocityMap,
      rigidEdgeConsumeMapR,
      rigidEdgeConsumeMapG,
      rigidEdgeConsumeMapB,
    },
    threshold: Math.max(0, Math.min(1, Number(thresholdEl?.value) || 0.35)),
    softBoundaryRingSprings: !!softBoundaryRingEl?.checked,
    softSolverMode: softSolverModeEl?.value || 'membrane',
    membraneMinEdgeLength,
    membraneMaxEdgeLength,
    dyeEatBooleanMode: true,
  });
}

function normalizeRuntimeSolverPath(raw) {
  const mode = String(raw || '').trim().toLowerCase();
  return mode === 'gpu-only' ? 'gpu-only' : 'baseline';
}

function getRuntimeSolverPath() {
  const picked = runtimeSolverPathEls.find((el) => el?.checked);
  return normalizeRuntimeSolverPath(picked?.value);
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
      solverPath: getRuntimeSolverPath(),
      ...readWindTunnelEmitterOptions(),
    },
  };

  // Same-origin iframe; no wildcard posting.
  windTunnelFrame.contentWindow.postMessage(payload, window.location.origin);

  // Re-apply active highlight after reset.
  if (highlightedSegmentId) {
    windTunnelFrame.contentWindow.postMessage({
      type: 'gpuLabHighlightSegment',
      segmentId: highlightedSegmentId,
      pulseMs: 1800,
    }, window.location.origin);
  }
}

function sendSegmentHighlight(segmentId, pulseMs = 2200) {
  highlightedSegmentId = segmentId || null;
  if (!windTunnelFrame?.contentWindow || !highlightedSegmentId) return;
  windTunnelFrame.contentWindow.postMessage({
    type: 'gpuLabHighlightSegment',
    segmentId: highlightedSegmentId,
    pulseMs,
  }, window.location.origin);
}

function renderSegmentStatsList(segments = []) {
  if (!segmentStatsList) return;
  segmentStatsList.innerHTML = '';
  if (!Array.isArray(segments) || segments.length === 0) {
    segmentStatsList.textContent = lastCompileHint || 'No segments yet.';
    return;
  }

  const frag = document.createDocumentFragment();
  for (const seg of segments) {
    const row = document.createElement('div');
    row.className = 'segment-row';

    const btn = document.createElement('button');
    btn.className = 'segment-id-btn';
    if (highlightedSegmentId && highlightedSegmentId === seg.id) {
      btn.classList.add('active');
    }
    btn.textContent = seg.id;
    btn.title = 'Highlight this segment in wind tunnel';
    btn.addEventListener('click', () => {
      highlightedSegmentId = seg.id;
      renderSegmentStatsList(segments);
      sendSegmentHighlight(seg.id, 2600);
    });

    const controls = document.createElement('div');
    controls.className = 'segment-edit-controls';

    const makeModeLabel = (text) => {
      const el = document.createElement('span');
      el.className = 'segment-edit-label';
      el.textContent = text;
      return el;
    };

    const velSelect = document.createElement('select');
    velSelect.className = 'segment-edit-select';
    velSelect.title = 'Velocity mode for this segment';
    velSelect.innerHTML = '<option value="1">BLOCK</option><option value="0">PASS</option>';
    velSelect.value = String(seg.velocityMode === 'PASS' ? 0 : 1);

    const makeDyeSelect = (channelName) => {
      const sel = document.createElement('select');
      sel.className = 'segment-edit-select';
      sel.title = `${channelName} dye mode for this segment`;
      sel.innerHTML = '<option value="1">NO-OP</option><option value="2">EAT</option>';
      return sel;
    };

    const dyeR = makeDyeSelect('Red');
    const dyeG = makeDyeSelect('Green');
    const dyeB = makeDyeSelect('Blue');
    dyeR.value = String(seg.dyeMode?.r === 'EAT' ? 2 : 1);
    dyeG.value = String(seg.dyeMode?.g === 'EAT' ? 2 : 1);
    dyeB.value = String(seg.dyeMode?.b === 'EAT' ? 2 : 1);

    const commitEdit = () => {
      applySegmentPolicyOverride(seg, {
        velocityCode: Number(velSelect.value),
        dyeCodes: [Number(dyeR.value), Number(dyeG.value), Number(dyeB.value)],
      });
    };

    velSelect.addEventListener('change', commitEdit);
    dyeR.addEventListener('change', commitEdit);
    dyeG.addEventListener('change', commitEdit);
    dyeB.addEventListener('change', commitEdit);

    controls.appendChild(makeModeLabel('V'));
    controls.appendChild(velSelect);
    controls.appendChild(makeModeLabel('R'));
    controls.appendChild(dyeR);
    controls.appendChild(makeModeLabel('G'));
    controls.appendChild(dyeG);
    controls.appendChild(makeModeLabel('B'));
    controls.appendChild(dyeB);

    const meta = document.createElement('div');
    meta.className = 'segment-meta';
    meta.textContent = `${seg.type} | len=${seg.length?.toFixed ? seg.length.toFixed(2) : seg.length}`;

    row.appendChild(btn);
    row.appendChild(controls);
    row.appendChild(meta);
    frag.appendChild(row);
  }

  segmentStatsList.appendChild(frag);
}

function drawFields() {
  const traitImg = pctx.createImageData(W, H);
  const densityImg = dctx.createImageData(W, H);
  const edgeImg = ectx.createImageData(W, H);
  const shapeImg = sctx.createImageData(W, H);
  const softPermeabilityImg = spctx.createImageData(W, H);
  const softEdgeDyeModeImg = sdctx.createImageData(W, H);
  const softEdgeVelocityModeImg = svctx.createImageData(W, H);
  const softEdgeMomentumModeImg = smctx.createImageData(W, H);
  const permeabilityImg = prctx.createImageData(W, H);
  const rigidEdgeVelocityModeImg = rvctx.createImageData(W, H);
  const rigidEdgeDyeModeImg = cctx.createImageData(W, H);

  for (let i = 0; i < rigid.length; i++) {
    const r = Math.max(0, Math.min(1, rigid[i]));
    const s = Math.max(0, Math.min(1, soft[i]));
    const dens = Math.max(0, Math.min(1, softDensity[i]));
    const edge = Math.max(0, Math.min(1, membraneEdgeMap[i]));
    const shape = Math.max(0, Math.min(1, membraneShapeMap[i]));
    const softPermeability = Math.max(0, Math.min(1, softPermeabilityMap[i]));
    const softDyeR = Math.max(0, Math.min(1, softEdgeDyeModeMapR[i]));
    const softDyeG = Math.max(0, Math.min(1, softEdgeDyeModeMapG[i]));
    const softDyeB = Math.max(0, Math.min(1, softEdgeDyeModeMapB[i]));
    const softVelocityMode = Math.max(0, Math.min(1, softEdgeVelocityModeMap[i]));
    const softMomentumMode = Math.max(0, Math.min(1, softEdgeMomentumModeMap[i]));
    const permeability = Math.max(0, Math.min(1, rigidPermeabilityMap[i]));
    const rigidVelocityMode = Math.max(0, Math.min(1, rigidEdgeVelocityMap[i]));
    const rigidDyeR = Math.max(0, Math.min(1, rigidEdgeDyeModeMapR[i]));
    const rigidDyeG = Math.max(0, Math.min(1, rigidEdgeDyeModeMapG[i]));
    const rigidDyeB = Math.max(0, Math.min(1, rigidEdgeDyeModeMapB[i]));

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

    // Soft edge dye mode map (RGB channels show NO-OP/EAT quantized policy intensity).
    softEdgeDyeModeImg.data[i * 4] = edgeModeScalarPreview(softDyeR);
    softEdgeDyeModeImg.data[i * 4 + 1] = edgeModeScalarPreview(softDyeG);
    softEdgeDyeModeImg.data[i * 4 + 2] = edgeModeScalarPreview(softDyeB);
    softEdgeDyeModeImg.data[i * 4 + 3] = 255;

    // Soft edge velocity mode: grayscale (black=BLOCK, white=PASS).
    const svg = Math.round(softVelocityMode * 255);
    softEdgeVelocityModeImg.data[i * 4] = svg;
    softEdgeVelocityModeImg.data[i * 4 + 1] = svg;
    softEdgeVelocityModeImg.data[i * 4 + 2] = svg;
    softEdgeVelocityModeImg.data[i * 4 + 3] = 255;

    // Soft edge momentum coupling: grayscale (black=0, white=1).
    const smg = Math.round(softMomentumMode * 255);
    softEdgeMomentumModeImg.data[i * 4] = smg;
    softEdgeMomentumModeImg.data[i * 4 + 1] = smg;
    softEdgeMomentumModeImg.data[i * 4 + 2] = smg;
    softEdgeMomentumModeImg.data[i * 4 + 3] = 255;

    // Rigid edge permeability map: grayscale (black=blocked, white=permeable).
    const pg = Math.round(permeability * 255);
    permeabilityImg.data[i * 4] = pg;
    permeabilityImg.data[i * 4 + 1] = pg;
    permeabilityImg.data[i * 4 + 2] = pg;
    permeabilityImg.data[i * 4 + 3] = 255;

    // Rigid edge velocity mode: grayscale (black=BLOCK, white=PASS).
    const rvg = Math.round(rigidVelocityMode * 255);
    rigidEdgeVelocityModeImg.data[i * 4] = rvg;
    rigidEdgeVelocityModeImg.data[i * 4 + 1] = rvg;
    rigidEdgeVelocityModeImg.data[i * 4 + 2] = rvg;
    rigidEdgeVelocityModeImg.data[i * 4 + 3] = 255;

    // Rigid edge dye mode map (RGB channels show NO-OP/EAT quantized policy intensity).
    rigidEdgeDyeModeImg.data[i * 4] = edgeModeScalarPreview(rigidDyeR);
    rigidEdgeDyeModeImg.data[i * 4 + 1] = edgeModeScalarPreview(rigidDyeG);
    rigidEdgeDyeModeImg.data[i * 4 + 2] = edgeModeScalarPreview(rigidDyeB);
    rigidEdgeDyeModeImg.data[i * 4 + 3] = 255;
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
  blit(sdctx, softEdgeDyeModeCanvas, softEdgeDyeModeImg);
  blit(svctx, softEdgeVelocityModeCanvas, softEdgeVelocityModeImg);
  blit(smctx, softEdgeMomentumModeCanvas, softEdgeMomentumModeImg);
  blit(prctx, rigidPermeabilityCanvas, permeabilityImg);
  blit(rvctx, rigidEdgeVelocityModeCanvas, rigidEdgeVelocityModeImg);
  blit(cctx, rigidEdgeDyeModeCanvas, rigidEdgeDyeModeImg);
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

function resolveSoftDyeEatMap(channel) {
  const ch = String(channel || 'r').toLowerCase();
  if (ch === 'g') return softEdgeDyeEatMapG;
  if (ch === 'b') return softEdgeDyeEatMapB;
  return softEdgeDyeEatMapR;
}

function resolveRigidDyeEatMap(channel) {
  const ch = String(channel || 'r').toLowerCase();
  if (ch === 'g') return rigidEdgeDyeEatMapG;
  if (ch === 'b') return rigidEdgeDyeEatMapB;
  return rigidEdgeDyeEatMapR;
}

function clearSoftDyeChannelMaps(channel, clientX, clientY) {
  const ch = String(channel || 'r').toLowerCase();
  const maps = ch === 'g'
    ? [softEdgeDyeBlockMapG, softEdgeDyePassMapG, softEdgeDyeEatMapG]
    : (ch === 'b'
      ? [softEdgeDyeBlockMapB, softEdgeDyePassMapB, softEdgeDyeEatMapB]
      : [softEdgeDyeBlockMapR, softEdgeDyePassMapR, softEdgeDyeEatMapR]);
  for (const map of maps) {
    paintScalarMap(map, softEdgeDyeModeCanvas, clientX, clientY, 0.0, 0.0, true);
  }
}

function clearRigidDyeChannelMaps(channel, clientX, clientY) {
  const ch = String(channel || 'r').toLowerCase();
  const maps = ch === 'g'
    ? [rigidEdgeDyeBlockMapG, rigidEdgeDyePassMapG, rigidEdgeDyeEatMapG]
    : (ch === 'b'
      ? [rigidEdgeDyeBlockMapB, rigidEdgeDyePassMapB, rigidEdgeDyeEatMapB]
      : [rigidEdgeDyeBlockMapR, rigidEdgeDyePassMapR, rigidEdgeDyeEatMapR]);
  for (const map of maps) {
    paintScalarMap(map, rigidEdgeDyeModeCanvas, clientX, clientY, 0.0, 0.0, true);
  }
}

function paintSoftEdgeDyeModeMap(clientX, clientY, erase = false) {
  const channel = String(softEdgeDyeChannelEl?.value || 'r');
  const mode = String(softEdgeDyeModeEl?.value || 'noop');
  const paintValue = Number(softEdgeDyePaintEl?.value) || 1.0;

  if (channel === 'erase') {
    for (const ch of ['r', 'g', 'b']) clearSoftDyeChannelMaps(ch, clientX, clientY);
    syncSoftDyeModeMapsFromWinnerFields();
    drawFields();
    return;
  }

  clearSoftDyeChannelMaps(channel, clientX, clientY);
  if (!erase && mode === 'eat') {
    paintScalarMap(resolveSoftDyeEatMap(channel), softEdgeDyeModeCanvas, clientX, clientY, paintValue, 0.0, false);
  }
  syncSoftDyeModeMapsFromWinnerFields();
  drawFields();
}

function paintSoftEdgeVelocityModeMap(clientX, clientY, erase = false) {
  paintScalarMap(
    softEdgeVelocityModeMap,
    softEdgeVelocityModeCanvas,
    clientX,
    clientY,
    edgeVelocityModeToScalar(softEdgeVelocityModeEl?.value || 'block'),
    0.0,
    erase,
  );
  drawFields();
}

function paintSoftEdgeMomentumModeMap(clientX, clientY, erase = false) {
  paintScalarMap(
    softEdgeMomentumModeMap,
    softEdgeMomentumModeCanvas,
    clientX,
    clientY,
    Number(softEdgeMomentumPaintEl?.value) || 1.0,
    1.0,
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

function paintRigidEdgeVelocityModeMap(clientX, clientY, erase = false) {
  paintScalarMap(
    rigidEdgeVelocityMap,
    rigidEdgeVelocityModeCanvas,
    clientX,
    clientY,
    edgeVelocityModeToScalar(rigidEdgeVelocityModeEl?.value || 'block'),
    0.0,
    erase,
  );
  drawFields();
}

function paintRigidEdgeDyeModeMap(clientX, clientY, erase = false) {
  const channel = String(rigidEdgeDyeChannelEl?.value || 'r');
  const mode = String(rigidEdgeDyeModeEl?.value || 'noop');
  const paintValue = Number(rigidEdgeDyePaintEl?.value) || 1.0;

  if (channel === 'erase') {
    for (const ch of ['r', 'g', 'b']) clearRigidDyeChannelMaps(ch, clientX, clientY);
    syncRigidDyeModeMapsFromWinnerFields();
    drawFields();
    return;
  }

  clearRigidDyeChannelMaps(channel, clientX, clientY);
  if (!erase && mode === 'eat') {
    paintScalarMap(resolveRigidDyeEatMap(channel), rigidEdgeDyeModeCanvas, clientX, clientY, paintValue, 0.0, false);
  }
  syncRigidDyeModeMapsFromWinnerFields();
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

  // Velocity policy overlay for compiled preview: BLOCK=red solid, PASS=green dashed.
  const specForPreview = lastCompiledSpec;
  let velocityPassEdges = 0;
  let velocityBlockEdges = 0;
  if (specForPreview) {
    const drawVelocityEdge = (ax, ay, bx, by, pass) => {
      mctx.strokeStyle = pass ? 'rgba(120,255,140,0.95)' : 'rgba(255,90,90,0.98)';
      mctx.lineWidth = pass ? 1.6 : 2.2;
      mctx.setLineDash(pass ? [6, 4] : []);
      mctx.beginPath();
      mctx.moveTo(ax * sx, ay * sy);
      mctx.lineTo(bx * sx, by * sy);
      mctx.stroke();
      mctx.setLineDash([]);
      mctx.lineWidth = 1;
    };

    for (const rb of (specForPreview.rigidBodies || [])) {
      const hull = rb?.hull || [];
      for (let i = 0; i < hull.length; i++) {
        const a = hull[i];
        const b = hull[(i + 1) % hull.length];
        if (!a || !b) continue;
        const velRaw = Array.isArray(rb?.edgeVelocityMode) ? Number(rb.edgeVelocityMode[i]) : 1;
        const pass = velRaw === 0;
        if (pass) velocityPassEdges += 1;
        else velocityBlockEdges += 1;
        drawVelocityEdge(a.x, a.y, b.x, b.y, pass);
      }
    }

    for (const sb of (specForPreview.softBodies || [])) {
      const nodes = sb?.nodes || [];
      for (const sp of (sb?.springs || [])) {
        if (!Array.isArray(sp) || sp.length < 2) continue;
        const a = nodes[Number(sp[0])];
        const b = nodes[Number(sp[1])];
        if (!a || !b) continue;
        const bodyRaw = Number(sp[3]);
        const velRaw = Number(sp[5]);
        const pass = (velRaw === 0) || (Number.isNaN(velRaw) && bodyRaw === 0);
        if (pass) velocityPassEdges += 1;
        else velocityBlockEdges += 1;
        drawVelocityEdge(a.x, a.y, b.x, b.y, pass);
      }
    }

    mctx.fillStyle = 'rgba(245,245,245,0.92)';
    mctx.font = '11px system-ui';
    mctx.fillText('velocity overlay: BLOCK=red solid, PASS=green dashed', 10, 14);
  }

  const specForJsonPreview = lastCompiledSpec || buildSpecFromCurrentFields(mesh, mesh?.__compileFields);
  out.textContent = JSON.stringify(specForJsonPreview || {
    ok: false,
    error: 'No compiled CreatureSpec available',
  }, null, 2);
  updateSegmentStatsPreview(specForJsonPreview);
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

function syncFieldHelpText(target) {
  if (!activeFieldHelpEl) return;
  const textByTarget = {
    trait: 'Active field: Trait field. Paint rigid vs soft occupancy for compile/isolate passes.',
    density: 'Active field: Resolution map. Darker paint compiles finer/smaller local primitives; brighter paint compiles coarser/larger ones.',
    membraneEdge: 'Active field: Perimeter edge-length map. Dark paint drives shorter edges; bright paint drives longer edges.',
    membraneShape: 'Active field: Membrane stiffness map. Dark paint makes softer perimeter response; bright paint makes stiffer response.',
    softPermeability: 'Active field: Soft edge permeability. Darker zones block more flow; brighter zones allow more flow through soft edges.',
    softEdgeDyeMode: 'Active field: Soft edge dye policy. Pick channel + mode, then paint where soft edges should EAT or NO-OP for dye.',
    softEdgeVelocityMode: 'Active field: Soft edge velocity mode. Paint BLOCK vs PASS behavior for fluid velocity coupling at soft edges.',
    softEdgeMomentumMode: 'Active field: Soft edge momentum coupling. 0 keeps velocity exchange low; 1 allows full momentum coupling.',
    rigidPermeability: 'Active field: Rigid edge permeability. Darker zones block more flow; brighter zones allow more flow through rigid edges.',
    rigidEdgeVelocityMode: 'Active field: Rigid edge velocity mode. Paint BLOCK vs PASS behavior for fluid velocity coupling at rigid edges.',
    rigidEdgeDyeMode: 'Active field: Rigid edge dye policy. Pick channel + mode, then paint where rigid edges should EAT or NO-OP for dye.',
  };
  activeFieldHelpEl.textContent = textByTarget[target] || textByTarget.trait;
}

function syncFieldPanelVisibility() {
  const target = fieldPaintTargetEl?.value || 'trait';
  syncFieldHelpText(target);
  for (const panel of fieldPanels) {
    if (!panel) continue;
    panel.hidden = panel.dataset.fieldPanel !== target;
  }

  const traitSelected = target === 'trait';
  const densitySelected = target === 'density';
  const edgeSelected = target === 'membraneEdge';
  const shapeSelected = target === 'membraneShape';
  const softPermeabilitySelected = target === 'softPermeability';
  const softDyeModeSelected = target === 'softEdgeDyeMode';
  const softVelocityModeSelected = target === 'softEdgeVelocityMode';
  const softMomentumModeSelected = target === 'softEdgeMomentumMode';
  const permeabilitySelected = target === 'rigidPermeability';
  const rigidVelocityModeSelected = target === 'rigidEdgeVelocityMode';
  const rigidDyeModeSelected = target === 'rigidEdgeDyeMode';

  setWidgetEnabled(modeEl, traitSelected, 'Trait paint mode only applies when Field to paint = Trait field');
  setWidgetEnabled(softDensityPaintEl, densitySelected, 'Resolution paint value only applies when Field to paint = Resolution field');
  setWidgetEnabled(membraneEdgePaintEl, edgeSelected, 'Perimeter edge paint value only applies when Field to paint = Membrane edge-length field');
  setWidgetEnabled(membraneShapePaintEl, shapeSelected, 'Membrane stiffness paint value only applies when Field to paint = Membrane stiffness field');
  setWidgetEnabled(softPermeabilityPaintEl, softPermeabilitySelected, 'Soft permeability paint value only applies when Field to paint = Soft permeability field');
  setWidgetEnabled(softEdgeDyeChannelEl, softDyeModeSelected, 'Soft dye channel only applies when Field to paint = Soft edge dye policy fields');
  setWidgetEnabled(softEdgeDyeModeEl, softDyeModeSelected, 'Soft dye mode only applies when Field to paint = Soft edge dye policy fields');
  setWidgetEnabled(softEdgeDyePaintEl, softDyeModeSelected, 'Soft dye field paint only applies when Field to paint = Soft edge dye policy fields');
  setWidgetEnabled(softEdgeVelocityModeEl, softVelocityModeSelected, 'Soft velocity mode only applies when Field to paint = Soft edge velocity mode field');
  setWidgetEnabled(softEdgeMomentumPaintEl, softMomentumModeSelected, 'Soft momentum paint value only applies when Field to paint = Soft edge momentum coupling field');
  setWidgetEnabled(rigidPermeabilityPaintEl, permeabilitySelected, 'Rigid permeability paint value only applies when Field to paint = Rigid permeability field');
  setWidgetEnabled(rigidEdgeVelocityModeEl, rigidVelocityModeSelected, 'Rigid velocity mode only applies when Field to paint = Rigid edge velocity mode field');
  setWidgetEnabled(rigidEdgeDyeChannelEl, rigidDyeModeSelected, 'Rigid dye channel only applies when Field to paint = Rigid edge dye policy fields');
  setWidgetEnabled(rigidEdgeDyeModeEl, rigidDyeModeSelected, 'Rigid dye mode only applies when Field to paint = Rigid edge dye policy fields');
  setWidgetEnabled(rigidEdgeDyePaintEl, rigidDyeModeSelected, 'Rigid dye field paint only applies when Field to paint = Rigid edge dye policy fields');

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
    syncSoftDyeModeMapsFromWinnerFields();
    syncRigidDyeModeMapsFromWinnerFields();
    const membraneMode = syncSoftModeUi();
    const requestedSoftMin = Math.max(1, Math.min(Math.max(1, W - 1), Math.round(Number(softMinCellSizeEl?.value) || 3)));
    const perimeterMinEdge = Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4);
    const perimeterMaxEdge = Math.max(perimeterMinEdge, Number(membraneMaxEdgeLengthEl?.value) || 8);
    const threshold = Math.max(0, Math.min(1, Number(thresholdEl.value) || 0.35));
    const largestBody = isolateLargestContiguousTraitBody(rigid, soft, threshold);

    lastCompileHint = '';
    if ((largestBody?.keptCells || 0) <= 0) {
      const maxOcc = Math.max(0, Math.min(1, Number(largestBody?.maxOccupancy) || 0));
      if (maxOcc > 0.001) {
        lastCompileHint = `No trait cells passed threshold ${threshold.toFixed(2)} (max paint ${maxOcc.toFixed(2)}). Lower Threshold or paint stronger in Trait field.`;
      } else {
        lastCompileHint = 'No trait paint to compile yet. Set Field to paint = Trait field and paint rigid/soft regions first.';
      }
    }

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
    lastCompileHint = msg;
    out.textContent = msg;
    if (segmentStatsOut) segmentStatsOut.textContent = msg;
    if (segmentStatsList) segmentStatsList.textContent = msg;
    return null;
  }
}

let traitPainting = false;
let densityPainting = false;
let membraneEdgePainting = false;
let membraneShapePainting = false;
let softPermeabilityPainting = false;
let softEdgeDyeModePainting = false;
let softEdgeVelocityModePainting = false;
let softEdgeMomentumModePainting = false;
let rigidPermeabilityPainting = false;
let rigidEdgeVelocityModePainting = false;
let rigidEdgeDyeModePainting = false;

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

softEdgeDyeModeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
softEdgeDyeModeCanvas.addEventListener('mousedown', (e) => {
  softEdgeDyeModePainting = true;
  paintSoftEdgeDyeModeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
softEdgeDyeModeCanvas.addEventListener('mousemove', (e) => {
  if (!softEdgeDyeModePainting) return;
  paintSoftEdgeDyeModeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

softEdgeVelocityModeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
softEdgeVelocityModeCanvas.addEventListener('mousedown', (e) => {
  softEdgeVelocityModePainting = true;
  paintSoftEdgeVelocityModeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
softEdgeVelocityModeCanvas.addEventListener('mousemove', (e) => {
  if (!softEdgeVelocityModePainting) return;
  paintSoftEdgeVelocityModeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

softEdgeMomentumModeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
softEdgeMomentumModeCanvas.addEventListener('mousedown', (e) => {
  softEdgeMomentumModePainting = true;
  paintSoftEdgeMomentumModeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
softEdgeMomentumModeCanvas.addEventListener('mousemove', (e) => {
  if (!softEdgeMomentumModePainting) return;
  paintSoftEdgeMomentumModeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
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

rigidEdgeVelocityModeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
rigidEdgeVelocityModeCanvas.addEventListener('mousedown', (e) => {
  rigidEdgeVelocityModePainting = true;
  paintRigidEdgeVelocityModeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
rigidEdgeVelocityModeCanvas.addEventListener('mousemove', (e) => {
  if (!rigidEdgeVelocityModePainting) return;
  paintRigidEdgeVelocityModeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

rigidEdgeDyeModeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
rigidEdgeDyeModeCanvas.addEventListener('mousedown', (e) => {
  rigidEdgeDyeModePainting = true;
  paintRigidEdgeDyeModeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
rigidEdgeDyeModeCanvas.addEventListener('mousemove', (e) => {
  if (!rigidEdgeDyeModePainting) return;
  paintRigidEdgeDyeModeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

window.addEventListener('mouseup', () => {
  traitPainting = false;
  densityPainting = false;
  membraneEdgePainting = false;
  membraneShapePainting = false;
  softPermeabilityPainting = false;
  softEdgeDyeModePainting = false;
  softEdgeVelocityModePainting = false;
  softEdgeMomentumModePainting = false;
  rigidPermeabilityPainting = false;
  rigidEdgeVelocityModePainting = false;
  rigidEdgeDyeModePainting = false;
});

clearBtn.addEventListener('click', () => {
  rigid.fill(0);
  soft.fill(0);
  softDensity.fill(0.5);
  membraneEdgeMap.fill(0.5);
  membraneShapeMap.fill(1.0);
  softPermeabilityMap.fill(0.0);
  softEdgeDyeBlockMapR.fill(0.0);
  softEdgeDyeBlockMapG.fill(0.0);
  softEdgeDyeBlockMapB.fill(0.0);
  softEdgeDyePassMapR.fill(0.0);
  softEdgeDyePassMapG.fill(0.0);
  softEdgeDyePassMapB.fill(0.0);
  softEdgeDyeEatMapR.fill(0.0);
  softEdgeDyeEatMapG.fill(0.0);
  softEdgeDyeEatMapB.fill(0.0);
  softEdgeVelocityModeMap.fill(0.0);
  softEdgeMomentumModeMap.fill(1.0);
  rigidPermeabilityMap.fill(0.0);
  rigidEdgeVelocityMap.fill(0.0);
  rigidEdgeDyeBlockMapR.fill(0.0);
  rigidEdgeDyeBlockMapG.fill(0.0);
  rigidEdgeDyeBlockMapB.fill(0.0);
  rigidEdgeDyePassMapR.fill(0.0);
  rigidEdgeDyePassMapG.fill(0.0);
  rigidEdgeDyePassMapB.fill(0.0);
  rigidEdgeDyeEatMapR.fill(0.0);
  rigidEdgeDyeEatMapG.fill(0.0);
  rigidEdgeDyeEatMapB.fill(0.0);
  syncSoftDyeModeMapsFromWinnerFields();
  syncRigidDyeModeMapsFromWinnerFields();
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
    syncPaintControlValueLabels();
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
      softEdgeDyeMode: true,
      softDyeBlockR: true,
      softDyePassR: true,
      softDyeEatR: true,
      softDyeBlockG: true,
      softDyePassG: true,
      softDyeEatG: true,
      softDyeBlockB: true,
      softDyePassB: true,
      softDyeEatB: true,
      softEdgeVelocityMode: true,
      softEdgeMomentumMode: true,
      rigidPermeability: true,
      rigidEdgeDyeMode: true,
      rigidDyeBlockR: true,
      rigidDyePassR: true,
      rigidDyeEatR: true,
      rigidDyeBlockG: true,
      rigidDyePassG: true,
      rigidDyeEatG: true,
      rigidDyeBlockB: true,
      rigidDyePassB: true,
      rigidDyeEatB: true,
      rigidEdgeVelocityMode: true,
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
if (randomizeSoftEdgeDyeModeBtn) {
  randomizeSoftEdgeDyeModeBtn.addEventListener('click', () => randomizeWithTargets({
    softEdgeDyeMode: true,
    softDyeBlockR: true,
    softDyePassR: true,
    softDyeEatR: true,
    softDyeBlockG: true,
    softDyePassG: true,
    softDyeEatG: true,
    softDyeBlockB: true,
    softDyePassB: true,
    softDyeEatB: true,
  }));
}
if (randomizeSoftDyeBlockRBtn) randomizeSoftDyeBlockRBtn.addEventListener('click', () => randomizeWithTargets({ softDyeBlockR: true }));
if (randomizeSoftDyePassRBtn) randomizeSoftDyePassRBtn.addEventListener('click', () => randomizeWithTargets({ softDyePassR: true }));
if (randomizeSoftDyeEatRBtn) randomizeSoftDyeEatRBtn.addEventListener('click', () => randomizeWithTargets({ softDyeEatR: true }));
if (randomizeSoftDyeBlockGBtn) randomizeSoftDyeBlockGBtn.addEventListener('click', () => randomizeWithTargets({ softDyeBlockG: true }));
if (randomizeSoftDyePassGBtn) randomizeSoftDyePassGBtn.addEventListener('click', () => randomizeWithTargets({ softDyePassG: true }));
if (randomizeSoftDyeEatGBtn) randomizeSoftDyeEatGBtn.addEventListener('click', () => randomizeWithTargets({ softDyeEatG: true }));
if (randomizeSoftDyeBlockBBtn) randomizeSoftDyeBlockBBtn.addEventListener('click', () => randomizeWithTargets({ softDyeBlockB: true }));
if (randomizeSoftDyePassBBtn) randomizeSoftDyePassBBtn.addEventListener('click', () => randomizeWithTargets({ softDyePassB: true }));
if (randomizeSoftDyeEatBBtn) randomizeSoftDyeEatBBtn.addEventListener('click', () => randomizeWithTargets({ softDyeEatB: true }));
if (randomizeSoftEdgeVelocityModeBtn) {
  randomizeSoftEdgeVelocityModeBtn.addEventListener('click', () => randomizeWithTargets({ softEdgeVelocityMode: true }));
}
if (randomizeSoftEdgeMomentumModeBtn) {
  randomizeSoftEdgeMomentumModeBtn.addEventListener('click', () => randomizeWithTargets({ softEdgeMomentumMode: true }));
}
if (randomizeRigidPermeabilityBtn) {
  randomizeRigidPermeabilityBtn.addEventListener('click', () => randomizeWithTargets({ rigidPermeability: true }));
}
if (randomizeRigidEdgeDyeModeBtn) {
  randomizeRigidEdgeDyeModeBtn.addEventListener('click', () => randomizeWithTargets({
    rigidEdgeDyeMode: true,
    rigidDyeBlockR: true,
    rigidDyePassR: true,
    rigidDyeEatR: true,
    rigidDyeBlockG: true,
    rigidDyePassG: true,
    rigidDyeEatG: true,
    rigidDyeBlockB: true,
    rigidDyePassB: true,
    rigidDyeEatB: true,
  }));
}
if (randomizeRigidDyeBlockRBtn) randomizeRigidDyeBlockRBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyeBlockR: true }));
if (randomizeRigidDyePassRBtn) randomizeRigidDyePassRBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyePassR: true }));
if (randomizeRigidDyeEatRBtn) randomizeRigidDyeEatRBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyeEatR: true }));
if (randomizeRigidDyeBlockGBtn) randomizeRigidDyeBlockGBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyeBlockG: true }));
if (randomizeRigidDyePassGBtn) randomizeRigidDyePassGBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyePassG: true }));
if (randomizeRigidDyeEatGBtn) randomizeRigidDyeEatGBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyeEatG: true }));
if (randomizeRigidDyeBlockBBtn) randomizeRigidDyeBlockBBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyeBlockB: true }));
if (randomizeRigidDyePassBBtn) randomizeRigidDyePassBBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyePassB: true }));
if (randomizeRigidDyeEatBBtn) randomizeRigidDyeEatBBtn.addEventListener('click', () => randomizeWithTargets({ rigidDyeEatB: true }));
if (randomizeRigidEdgeVelocityModeBtn) {
  randomizeRigidEdgeVelocityModeBtn.addEventListener('click', () => randomizeWithTargets({ rigidEdgeVelocityMode: true }));
}
const syncPaintControlValueLabels = () => {
  if (brushValueEl && brushEl) {
    const brush = Math.max(1, Number(brushEl.value) || 50);
    brushValueEl.textContent = `${Math.round(brush)} px`;
  }
  if (thresholdValueEl && thresholdEl) {
    const threshold = Math.max(0, Math.min(1, Number(thresholdEl.value) || 0.35));
    thresholdValueEl.textContent = threshold.toFixed(2);
  }

  if (softDensityPaintValueEl && softDensityPaintEl) softDensityPaintValueEl.textContent = (Number(softDensityPaintEl.value) || 0).toFixed(2);
  if (membraneEdgePaintValueEl && membraneEdgePaintEl) membraneEdgePaintValueEl.textContent = (Number(membraneEdgePaintEl.value) || 0).toFixed(2);
  if (membraneShapePaintValueEl && membraneShapePaintEl) membraneShapePaintValueEl.textContent = (Number(membraneShapePaintEl.value) || 0).toFixed(2);
  if (softPermeabilityPaintValueEl && softPermeabilityPaintEl) softPermeabilityPaintValueEl.textContent = (Number(softPermeabilityPaintEl.value) || 0).toFixed(2);
  if (softEdgeDyePaintValueEl && softEdgeDyePaintEl) softEdgeDyePaintValueEl.textContent = (Number(softEdgeDyePaintEl.value) || 0).toFixed(2);
  if (softEdgeMomentumPaintValueEl && softEdgeMomentumPaintEl) softEdgeMomentumPaintValueEl.textContent = (Number(softEdgeMomentumPaintEl.value) || 0).toFixed(2);
  if (rigidPermeabilityPaintValueEl && rigidPermeabilityPaintEl) rigidPermeabilityPaintValueEl.textContent = (Number(rigidPermeabilityPaintEl.value) || 0).toFixed(2);
  if (rigidEdgeDyePaintValueEl && rigidEdgeDyePaintEl) rigidEdgeDyePaintValueEl.textContent = (Number(rigidEdgeDyePaintEl.value) || 0).toFixed(2);

  const rigidMin = Math.max(2, Math.round(Number(rigidPrimitiveSideMinEl?.value) || 4));
  const rigidMaxRaw = Math.max(2, Math.round(Number(rigidPrimitiveSideMaxEl?.value) || 10));
  const rigidMax = Math.max(rigidMin, rigidMaxRaw);
  if (rigidPrimitiveSideMinValueEl) rigidPrimitiveSideMinValueEl.textContent = String(rigidMin);
  if (rigidPrimitiveSideMaxValueEl) rigidPrimitiveSideMaxValueEl.textContent = String(rigidMax);

  if (softMinCellSizeValueEl && softMinCellSizeEl) softMinCellSizeValueEl.textContent = String(Math.max(1, Math.round(Number(softMinCellSizeEl.value) || 3)));

  const perimeterMin = Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4);
  const perimeterMaxRaw = Math.max(1, Number(membraneMaxEdgeLengthEl?.value) || 8);
  const perimeterMax = Math.max(perimeterMin, perimeterMaxRaw);
  if (membraneMinEdgeLengthValueEl) membraneMinEdgeLengthValueEl.textContent = perimeterMin.toFixed(1);
  if (membraneMaxEdgeLengthValueEl) membraneMaxEdgeLengthValueEl.textContent = perimeterMax.toFixed(1);

  if (randomFieldSeedValueEl && randomFieldSeedEl) randomFieldSeedValueEl.textContent = String(Math.max(0, Math.round(Number(randomFieldSeedEl.value) || 0)));

  if (windEmitterJetVyValueEl && windEmitterJetVyEl) windEmitterJetVyValueEl.textContent = (Number(windEmitterJetVyEl.value) || 0).toFixed(2);
  if (windEmitterStrengthValueEl && windEmitterStrengthEl) windEmitterStrengthValueEl.textContent = (Number(windEmitterStrengthEl.value) || 0).toFixed(1);
  if (windEmitterRadiusValueEl && windEmitterRadiusEl) windEmitterRadiusValueEl.textContent = (Number(windEmitterRadiusEl.value) || 0).toFixed(1);
  if (windEmitterYFractionValueEl && windEmitterYFractionEl) windEmitterYFractionValueEl.textContent = (Number(windEmitterYFractionEl.value) || 0).toFixed(2);
  if (windEmitterSpinValueEl && windEmitterSpinEl) windEmitterSpinValueEl.textContent = (Number(windEmitterSpinEl.value) || 0).toFixed(2);
  if (windEmitterCurlGainValueEl && windEmitterCurlGainEl) windEmitterCurlGainValueEl.textContent = (Number(windEmitterCurlGainEl.value) || 0).toFixed(2);
  if (windEmitterDriftGainValueEl && windEmitterDriftGainEl) windEmitterDriftGainValueEl.textContent = (Number(windEmitterDriftGainEl.value) || 0).toFixed(2);
  if (windEmitterChaosGainValueEl && windEmitterChaosGainEl) windEmitterChaosGainValueEl.textContent = (Number(windEmitterChaosGainEl.value) || 0).toFixed(2);
  if (windEmitterWobbleAmpValueEl && windEmitterWobbleAmpEl) windEmitterWobbleAmpValueEl.textContent = (Number(windEmitterWobbleAmpEl.value) || 0).toFixed(1);
  if (windEmitterWobbleFreqValueEl && windEmitterWobbleFreqEl) windEmitterWobbleFreqValueEl.textContent = (Number(windEmitterWobbleFreqEl.value) || 0).toFixed(3);
};

const sliderReadoutInputs = [
  brushEl,
  thresholdEl,
  softDensityPaintEl,
  membraneEdgePaintEl,
  membraneShapePaintEl,
  softPermeabilityPaintEl,
  softEdgeDyePaintEl,
  softEdgeMomentumPaintEl,
  rigidPermeabilityPaintEl,
  rigidEdgeDyePaintEl,
  rigidPrimitiveSideMinEl,
  rigidPrimitiveSideMaxEl,
  softMinCellSizeEl,
  membraneMinEdgeLengthEl,
  membraneMaxEdgeLengthEl,
  randomFieldSeedEl,
  windEmitterJetVyEl,
  windEmitterStrengthEl,
  windEmitterRadiusEl,
  windEmitterYFractionEl,
  windEmitterSpinEl,
  windEmitterCurlGainEl,
  windEmitterDriftGainEl,
  windEmitterChaosGainEl,
  windEmitterWobbleAmpEl,
  windEmitterWobbleFreqEl,
].filter(Boolean);
for (const el of sliderReadoutInputs) {
  el.addEventListener('input', syncPaintControlValueLabels);
  el.addEventListener('change', syncPaintControlValueLabels);
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

    softEdgeDyeBlockMapR.fill(0); softEdgeDyePassMapR.fill(0); softEdgeDyeEatMapR.fill(0);
    softEdgeDyeBlockMapG.fill(0); softEdgeDyePassMapG.fill(0); softEdgeDyeEatMapG.fill(0);
    softEdgeDyeBlockMapB.fill(0); softEdgeDyePassMapB.fill(0); softEdgeDyeEatMapB.fill(0);

    if (Array.isArray(authoring.softEdgeDyeBlockMapR) && authoring.softEdgeDyeBlockMapR.length === W * H) softEdgeDyeBlockMapR.set(authoring.softEdgeDyeBlockMapR);
    if (Array.isArray(authoring.softEdgeDyePassMapR) && authoring.softEdgeDyePassMapR.length === W * H) softEdgeDyePassMapR.set(authoring.softEdgeDyePassMapR);
    if (Array.isArray(authoring.softEdgeDyeEatMapR) && authoring.softEdgeDyeEatMapR.length === W * H) softEdgeDyeEatMapR.set(authoring.softEdgeDyeEatMapR);
    if (Array.isArray(authoring.softEdgeDyeBlockMapG) && authoring.softEdgeDyeBlockMapG.length === W * H) softEdgeDyeBlockMapG.set(authoring.softEdgeDyeBlockMapG);
    if (Array.isArray(authoring.softEdgeDyePassMapG) && authoring.softEdgeDyePassMapG.length === W * H) softEdgeDyePassMapG.set(authoring.softEdgeDyePassMapG);
    if (Array.isArray(authoring.softEdgeDyeEatMapG) && authoring.softEdgeDyeEatMapG.length === W * H) softEdgeDyeEatMapG.set(authoring.softEdgeDyeEatMapG);
    if (Array.isArray(authoring.softEdgeDyeBlockMapB) && authoring.softEdgeDyeBlockMapB.length === W * H) softEdgeDyeBlockMapB.set(authoring.softEdgeDyeBlockMapB);
    if (Array.isArray(authoring.softEdgeDyePassMapB) && authoring.softEdgeDyePassMapB.length === W * H) softEdgeDyePassMapB.set(authoring.softEdgeDyePassMapB);
    if (Array.isArray(authoring.softEdgeDyeEatMapB) && authoring.softEdgeDyeEatMapB.length === W * H) softEdgeDyeEatMapB.set(authoring.softEdgeDyeEatMapB);

    // Backward-compatible import path from packed mode maps.
    if (Array.isArray(authoring.softEdgeDyeModeMapR) && authoring.softEdgeDyeModeMapR.length === W * H) {
      populateWinnerFieldsFromModeScalarMap(authoring.softEdgeDyeModeMapR, softEdgeDyeBlockMapR, softEdgeDyePassMapR, softEdgeDyeEatMapR);
    }
    if (Array.isArray(authoring.softEdgeDyeModeMapG) && authoring.softEdgeDyeModeMapG.length === W * H) {
      populateWinnerFieldsFromModeScalarMap(authoring.softEdgeDyeModeMapG, softEdgeDyeBlockMapG, softEdgeDyePassMapG, softEdgeDyeEatMapG);
    }
    if (Array.isArray(authoring.softEdgeDyeModeMapB) && authoring.softEdgeDyeModeMapB.length === W * H) {
      populateWinnerFieldsFromModeScalarMap(authoring.softEdgeDyeModeMapB, softEdgeDyeBlockMapB, softEdgeDyePassMapB, softEdgeDyeEatMapB);
    }

    if (Array.isArray(authoring.softEdgeVelocityModeMap) && authoring.softEdgeVelocityModeMap.length === W * H) softEdgeVelocityModeMap.set(authoring.softEdgeVelocityModeMap);
    else softEdgeVelocityModeMap.fill(0.0);
    if (Array.isArray(authoring.softEdgeMomentumModeMap) && authoring.softEdgeMomentumModeMap.length === W * H) softEdgeMomentumModeMap.set(authoring.softEdgeMomentumModeMap);
    else softEdgeMomentumModeMap.fill(1.0);

    if (Array.isArray(authoring.rigidPermeabilityMap) && authoring.rigidPermeabilityMap.length === W * H) rigidPermeabilityMap.set(authoring.rigidPermeabilityMap);
    else rigidPermeabilityMap.fill(0.0);

    if (Array.isArray(authoring.rigidEdgeVelocityMap) && authoring.rigidEdgeVelocityMap.length === W * H) rigidEdgeVelocityMap.set(authoring.rigidEdgeVelocityMap);
    else if (Array.isArray(authoring.rigidEdgeVelocityModeMap) && authoring.rigidEdgeVelocityModeMap.length === W * H) rigidEdgeVelocityMap.set(authoring.rigidEdgeVelocityModeMap);
    else rigidEdgeVelocityMap.fill(0.0);

    rigidEdgeDyeBlockMapR.fill(0); rigidEdgeDyePassMapR.fill(0); rigidEdgeDyeEatMapR.fill(0);
    rigidEdgeDyeBlockMapG.fill(0); rigidEdgeDyePassMapG.fill(0); rigidEdgeDyeEatMapG.fill(0);
    rigidEdgeDyeBlockMapB.fill(0); rigidEdgeDyePassMapB.fill(0); rigidEdgeDyeEatMapB.fill(0);

    if (Array.isArray(authoring.rigidEdgeDyeBlockMapR) && authoring.rigidEdgeDyeBlockMapR.length === W * H) rigidEdgeDyeBlockMapR.set(authoring.rigidEdgeDyeBlockMapR);
    if (Array.isArray(authoring.rigidEdgeDyePassMapR) && authoring.rigidEdgeDyePassMapR.length === W * H) rigidEdgeDyePassMapR.set(authoring.rigidEdgeDyePassMapR);
    if (Array.isArray(authoring.rigidEdgeDyeEatMapR) && authoring.rigidEdgeDyeEatMapR.length === W * H) rigidEdgeDyeEatMapR.set(authoring.rigidEdgeDyeEatMapR);
    if (Array.isArray(authoring.rigidEdgeDyeBlockMapG) && authoring.rigidEdgeDyeBlockMapG.length === W * H) rigidEdgeDyeBlockMapG.set(authoring.rigidEdgeDyeBlockMapG);
    if (Array.isArray(authoring.rigidEdgeDyePassMapG) && authoring.rigidEdgeDyePassMapG.length === W * H) rigidEdgeDyePassMapG.set(authoring.rigidEdgeDyePassMapG);
    if (Array.isArray(authoring.rigidEdgeDyeEatMapG) && authoring.rigidEdgeDyeEatMapG.length === W * H) rigidEdgeDyeEatMapG.set(authoring.rigidEdgeDyeEatMapG);
    if (Array.isArray(authoring.rigidEdgeDyeBlockMapB) && authoring.rigidEdgeDyeBlockMapB.length === W * H) rigidEdgeDyeBlockMapB.set(authoring.rigidEdgeDyeBlockMapB);
    if (Array.isArray(authoring.rigidEdgeDyePassMapB) && authoring.rigidEdgeDyePassMapB.length === W * H) rigidEdgeDyePassMapB.set(authoring.rigidEdgeDyePassMapB);
    if (Array.isArray(authoring.rigidEdgeDyeEatMapB) && authoring.rigidEdgeDyeEatMapB.length === W * H) rigidEdgeDyeEatMapB.set(authoring.rigidEdgeDyeEatMapB);

    // Backward-compatible import path from packed mode/consume maps.
    if (Array.isArray(authoring.rigidEdgeDyeModeMapR) && authoring.rigidEdgeDyeModeMapR.length === W * H) {
      populateWinnerFieldsFromModeScalarMap(authoring.rigidEdgeDyeModeMapR, rigidEdgeDyeBlockMapR, rigidEdgeDyePassMapR, rigidEdgeDyeEatMapR);
    }
    if (Array.isArray(authoring.rigidEdgeDyeModeMapG) && authoring.rigidEdgeDyeModeMapG.length === W * H) {
      populateWinnerFieldsFromModeScalarMap(authoring.rigidEdgeDyeModeMapG, rigidEdgeDyeBlockMapG, rigidEdgeDyePassMapG, rigidEdgeDyeEatMapG);
    }
    if (Array.isArray(authoring.rigidEdgeDyeModeMapB) && authoring.rigidEdgeDyeModeMapB.length === W * H) {
      populateWinnerFieldsFromModeScalarMap(authoring.rigidEdgeDyeModeMapB, rigidEdgeDyeBlockMapB, rigidEdgeDyePassMapB, rigidEdgeDyeEatMapB);
    }
    if (Array.isArray(authoring.rigidEdgeConsumeMapR) && authoring.rigidEdgeConsumeMapR.length === W * H) {
      rigidEdgeDyeEatMapR.set(authoring.rigidEdgeConsumeMapR);
    }
    if (Array.isArray(authoring.rigidEdgeConsumeMapG) && authoring.rigidEdgeConsumeMapG.length === W * H) {
      rigidEdgeDyeEatMapG.set(authoring.rigidEdgeConsumeMapG);
    }
    if (Array.isArray(authoring.rigidEdgeConsumeMapB) && authoring.rigidEdgeConsumeMapB.length === W * H) {
      rigidEdgeDyeEatMapB.set(authoring.rigidEdgeConsumeMapB);
    }

    syncSoftDyeModeMapsFromWinnerFields();
    syncRigidDyeModeMapsFromWinnerFields();
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
  updateSegmentStatsPreview(spec);
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

if (copySegmentStatsBtn) {
  const report = flashCopyButtonState(copySegmentStatsBtn);
  copySegmentStatsBtn.addEventListener('click', async () => {
    const ok = await copyTextToClipboard(segmentStatsOut?.textContent || '');
    report(ok);
  });
}

if (copyOutBtn) {
  const report = flashCopyButtonState(copyOutBtn);
  copyOutBtn.addEventListener('click', async () => {
    const ok = await copyTextToClipboard(out?.textContent || '');
    report(ok);
  });
}

for (const solverEl of runtimeSolverPathEls) {
  solverEl?.addEventListener('change', () => {
    if (!solverEl.checked) return;
    if (lastCompiledSpec) pushSpecToWindTunnel(lastCompiledSpec);
  });
}

syncFieldPanelVisibility();
syncPaintControlValueLabels();
drawFields();
compileNow();
