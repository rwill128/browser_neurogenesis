const windFrame = document.getElementById('windFrame');
const scenarioPresetEl = document.getElementById('scenarioPreset');
const loadPresetBtn = document.getElementById('loadPresetBtn');
const randomScenarioBtn = document.getElementById('randomScenarioBtn');
const curriculumPrevBtn = document.getElementById('curriculumPrevBtn');
const curriculumNextBtn = document.getElementById('curriculumNextBtn');
const curriculumStatusEl = document.getElementById('curriculumStatus');
const scenarioPresetHintEl = document.getElementById('scenarioPresetHint');
const fixtureTypeEl = document.getElementById('fixtureType');
const motionModeEl = document.getElementById('motionMode');
const circleRadiusEl = document.getElementById('circleRadius');
const circleSpeedEl = document.getElementById('circleSpeed');
const circleRadiusLabelEl = document.getElementById('circleRadiusLabel');
const circleSpeedLabelEl = document.getElementById('circleSpeedLabel');
const circleRadiusValueEl = document.getElementById('circleRadiusValue');
const circleSpeedValueEl = document.getElementById('circleSpeedValue');
const motionHintEl = document.getElementById('motionHint');
const dyeChannelEl = document.getElementById('dyeChannel');
const dyeModeEl = document.getElementById('dyeMode');
const dyeModeLabelEl = document.getElementById('dyeModeLabel');
const velocityModeEl = document.getElementById('velocityMode');
const velocityContractHintEl = document.getElementById('velocityContractHint');
const effectiveDyeModeStatusEl = document.getElementById('effectiveDyeModeStatus');
const dyeModeHintEl = document.getElementById('dyeModeHint');
const velocityModeHintEl = document.getElementById('velocityModeHint');
const momentumHintEl = document.getElementById('momentumHint');
const momentumEl = document.getElementById('momentum');
const momentumValueEl = document.getElementById('momentumValue');
const showSegmentIdsEl = document.getElementById('showSegmentIds');
const showExtraVisualsEl = document.getElementById('showExtraVisuals');
const applyBtn = document.getElementById('applyBtn');
const autoApplyEl = document.getElementById('autoApply');
const applyModeHintEl = document.getElementById('applyModeHint');
const scenarioOut = document.getElementById('scenarioOut');
const copyScenarioBtn = document.getElementById('copyScenarioBtn');
const downloadScenarioBtn = document.getElementById('downloadScenarioBtn');
const loadScenarioBtn = document.getElementById('loadScenarioBtn');
const loadScenarioFile = document.getElementById('loadScenarioFile');
const downloadScreenshotBtn = document.getElementById('downloadScreenshotBtn');
const captureThumb = document.getElementById('captureThumb');

const EDGE_DYE_PASS = 0;
const EDGE_DYE_EAT = 2;
const EDGE_VEL_PASS = 0;
const EDGE_VEL_BLOCK = 1;

let embedReady = false;
let lastPayload = null;
let scenarioPresets = [];
let currentScenarioMeta = null;
let curriculumOrder = [];
let curriculumIndex = -1;
let embedStatusPollTimer = null;
let embedStallFrames = 0;
let lastEmbedFrame = -1;
let lastEmbedResyncAt = 0;

const DEFAULT_SCENARIO_PRESETS = [
  {
    id: 'rigid-red-eat-block-pinned',
    name: 'Rigid line · red EAT · velocity BLOCK · pinned',
    description: 'Baseline confrontation: rigid segment absorbs red dye while staying immobile.',
    fixtureType: 'rigid-line',
    motionMode: 'pinned',
    circleRadius: 12,
    circleSpeed: 0.8,
    dyeChannel: 'r',
    dyeMode: 'eat',
    velocityMode: 'block',
    momentum: 1,
  },
  {
    id: 'rigid-red-noop-block-pinned',
    name: 'Rigid line · red NO-OP · velocity BLOCK · pinned',
    description: 'Control pair for the baseline: no dye removal, same geometry and flow coupling.',
    fixtureType: 'rigid-line',
    motionMode: 'pinned',
    circleRadius: 12,
    circleSpeed: 0.8,
    dyeChannel: 'r',
    dyeMode: 'noop',
    velocityMode: 'block',
    momentum: 1,
  },
  // channel-block preset removed per rollback to EAT/NO-OP interaction contract.
  {
    id: 'soft-green-eat-block-pinned',
    name: 'Soft line · green EAT · velocity BLOCK · pinned',
    description: 'Soft-segment version to verify per-spring channel absorption behavior.',
    fixtureType: 'soft-line',
    motionMode: 'pinned',
    circleRadius: 12,
    circleSpeed: 0.8,
    dyeChannel: 'g',
    dyeMode: 'eat',
    velocityMode: 'block',
    momentum: 1,
  },
  {
    id: 'soft-green-noop-block-pinned',
    name: 'Soft line · green NO-OP · velocity BLOCK · pinned',
    description: 'Soft control pair: no-op dye channel with identical fixture setup.',
    fixtureType: 'soft-line',
    motionMode: 'pinned',
    circleRadius: 12,
    circleSpeed: 0.8,
    dyeChannel: 'g',
    dyeMode: 'noop',
    velocityMode: 'block',
    momentum: 1,
  },
  {
    id: 'rigid-blue-eat-pass-circle',
    name: 'Rigid line · blue EAT · velocity PASS · circle drag',
    description: 'Dynamic trajectory case to inspect trailing removal under moving fixtures.',
    fixtureType: 'rigid-line',
    motionMode: 'circle',
    circleRadius: 14,
    circleSpeed: 1.2,
    dyeChannel: 'b',
    dyeMode: 'eat',
    velocityMode: 'pass',
    momentum: 0.6,
  },
];

const CURRICULUM_DEFAULT_IDS = [
  'rigid-red-noop-block-pinned',
  'rigid-red-eat-block-pinned',
  'soft-green-noop-block-pinned',
  'soft-green-eat-block-pinned',
  'rigid-blue-eat-pass-circle',
];

function clamp(v, lo, hi) {
  const n = Number(v);
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

let autoApplyDebounce = null;
function queueAutoApply() {
  if (!autoApplyEl?.checked) return;
  if (autoApplyDebounce) clearTimeout(autoApplyDebounce);
  autoApplyDebounce = setTimeout(() => {
    autoApplyDebounce = null;
    pushScenario();
  }, 120);
}

function normalizeNumericInput(el, lo, hi, fallback, digits = null) {
  if (!el) return fallback;
  const raw = Number(el.value);
  const clamped = clamp(Number.isFinite(raw) ? raw : fallback, lo, hi);
  const next = typeof digits === 'number' ? clamped.toFixed(digits) : String(clamped);
  if (String(el.value) !== String(next)) el.value = next;
  return clamped;
}

function updateRangeValueLabels() {
  if (circleRadiusValueEl) {
    const v = normalizeNumericInput(circleRadiusEl, 0, 48, 12, 0);
    circleRadiusValueEl.textContent = v.toFixed(0);
  }
  if (circleSpeedValueEl) {
    const v = normalizeNumericInput(circleSpeedEl, 0, 6, 0.8, 2);
    circleSpeedValueEl.textContent = v.toFixed(2);
  }
  if (momentumValueEl) {
    const v = normalizeNumericInput(momentumEl, 0, 1, 1, 2);
    momentumValueEl.textContent = v.toFixed(2);
  }
  updateControlHints();
}

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

function timestampTag() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function downloadTextFile(filename, text, mimeType = 'application/json') {
  const blob = new Blob([String(text || '')], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadDataUrl(filename, dataUrl) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function flashButton(btn, ok, okText = 'Done', failText = 'Failed') {
  if (!btn) return;
  const prev = btn.textContent;
  btn.textContent = ok ? okText : failText;
  setTimeout(() => { btn.textContent = prev; }, 1100);
}

function setControlValue(el, value) {
  if (!el) return;
  const v = String(value ?? '');
  if (Array.from(el.options || []).some((o) => String(o.value) === v)) {
    el.value = v;
  }
}

function setInputValue(el, value) {
  if (!el) return;
  const n = Number(value);
  if (Number.isFinite(n)) el.value = String(n);
}

function randomChoice(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomInRange(min, max, step = 0.05) {
  const lo = Number(min);
  const hi = Number(max);
  const st = Math.max(1e-6, Number(step) || 0.05);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return lo;
  const n = lo + Math.random() * (hi - lo);
  return Math.round(n / st) * st;
}

function updateMotionControlState() {
  const mode = String(motionModeEl?.value || 'pinned');
  const circleEnabled = mode === 'circle';
  if (circleRadiusEl) circleRadiusEl.disabled = !circleEnabled;
  if (circleSpeedEl) circleSpeedEl.disabled = !circleEnabled;
  circleRadiusLabelEl?.classList.toggle('control-disabled', !circleEnabled);
  circleSpeedLabelEl?.classList.toggle('control-disabled', !circleEnabled);
  if (motionHintEl) {
    motionHintEl.textContent = circleEnabled
      ? 'Circle controls are active for dragged-in-circle motion.'
      : 'Circle controls are ignored while motion is pinned.';
  }
}

function updateDyeControlState() {
  const velocityMode = String(velocityModeEl?.value || 'block').toLowerCase();
  const allowEat = velocityMode === 'pass';
  if (dyeModeEl) {
    if (!allowEat && String(dyeModeEl.value || 'noop').toLowerCase() === 'eat') {
      dyeModeEl.value = 'noop';
    }
    dyeModeEl.disabled = !allowEat;
  }
  dyeModeLabelEl?.classList.toggle('control-disabled', !allowEat);
  if (velocityContractHintEl) {
    velocityContractHintEl.textContent = allowEat
      ? 'Current contract note: velocity=PASS allows dye mode selection (NO-OP or EAT).'
      : 'Current contract note: velocity=BLOCK is a hard boundary condition, so dye mode is locked to NO-OP.';
  }
  updateEffectiveDyeModeStatus();
  updateControlHints();
}

function updateApplyModeState() {
  const autoApplyOn = !!autoApplyEl?.checked;
  if (applyBtn) {
    applyBtn.disabled = autoApplyOn;
    applyBtn.title = autoApplyOn
      ? 'Auto-apply is ON. Changes are pushed automatically.'
      : 'Auto-apply is OFF. Click to push the current scenario.';
  }
  if (applyModeHintEl) {
    applyModeHintEl.textContent = autoApplyOn
      ? 'Auto-apply ON: slider drags apply automatically (debounced).'
      : 'Auto-apply OFF: click “Apply scenario” after changing controls.';
  }
}

function applyScenarioPreset(preset, meta = null) {
  if (!preset || typeof preset !== 'object') return;
  setControlValue(fixtureTypeEl, preset.fixtureType || 'rigid-line');
  setControlValue(motionModeEl, preset.motionMode || 'pinned');
  setControlValue(dyeChannelEl, preset.dyeChannel || 'r');
  setControlValue(dyeModeEl, preset.dyeMode || 'noop');
  setControlValue(velocityModeEl, preset.velocityMode || 'block');
  setInputValue(circleRadiusEl, preset.circleRadius ?? 12);
  setInputValue(circleSpeedEl, preset.circleSpeed ?? 0.8);
  setInputValue(momentumEl, preset.momentum ?? 1);
  updateRangeValueLabels();
  updateMotionControlState();
  updateDyeControlState();

  currentScenarioMeta = {
    id: preset.id || null,
    name: preset.name || preset.id || 'custom-scenario',
    source: meta?.source || 'preset',
  };

  if (scenarioPresetHintEl) {
    scenarioPresetHintEl.textContent = String(preset.description || '');
  }
}

function getSelectedPreset() {
  const id = String(scenarioPresetEl?.value || '');
  return scenarioPresets.find((p) => String(p.id) === id) || null;
}

function resolveCurriculumOrder() {
  const idSet = new Set(scenarioPresets.map((p) => String(p.id || '')));
  curriculumOrder = CURRICULUM_DEFAULT_IDS.filter((id) => idSet.has(id));
  if (curriculumOrder.length === 0) {
    curriculumOrder = scenarioPresets.map((p) => String(p.id || '')).filter(Boolean);
  }
}

function updateCurriculumStatus() {
  if (!curriculumStatusEl) return;
  if (curriculumOrder.length === 0 || curriculumIndex < 0) {
    curriculumStatusEl.textContent = '';
    return;
  }
  const id = curriculumOrder[curriculumIndex];
  const preset = scenarioPresets.find((p) => String(p.id) === id);
  const label = preset?.name || id;
  curriculumStatusEl.textContent = `Curriculum ${curriculumIndex + 1}/${curriculumOrder.length}: ${label}`;
}

function loadCurriculumStep(nextIndex) {
  if (curriculumOrder.length === 0) return false;
  curriculumIndex = Math.max(0, Math.min(curriculumOrder.length - 1, Number(nextIndex) || 0));
  const id = curriculumOrder[curriculumIndex];
  const preset = scenarioPresets.find((p) => String(p.id) === id);
  if (!preset) return false;
  if (scenarioPresetEl) scenarioPresetEl.value = String(id);
  applyScenarioPreset(preset, { source: 'curriculum' });
  updateCurriculumStatus();
  return true;
}

function generateRandomScenarioPreset() {
  const fixtureType = randomChoice(['rigid-line', 'soft-line']) || 'rigid-line';
  const motionMode = Math.random() < 0.7 ? 'pinned' : 'circle';
  const dyeChannel = randomChoice(['r', 'g', 'b']) || 'r';
  const dyeMode = randomChoice(['noop', 'eat']) || 'noop';
  const velocityMode = Math.random() < 0.7 ? 'block' : 'pass';
  const momentum = clamp(randomInRange(0.2, 1.0, 0.05), 0, 1);
  const circleRadius = motionMode === 'circle' ? clamp(randomInRange(8, 22, 1), 0, 48) : 12;
  const circleSpeed = motionMode === 'circle' ? clamp(randomInRange(0.35, 1.8, 0.05), 0, 6) : 0.8;

  return {
    id: `random-${timestampTag()}`,
    name: 'Random generated scenario',
    description: `Auto-generated: ${fixtureType}, ${motionMode}, ${dyeChannel.toUpperCase()} ${dyeMode.toUpperCase()}, velocity ${velocityMode.toUpperCase()}, momentum ${momentum.toFixed(2)}.`,
    fixtureType,
    motionMode,
    circleRadius,
    circleSpeed,
    dyeChannel,
    dyeMode,
    velocityMode,
    momentum,
  };
}

function populateScenarioPresetDropdown() {
  if (!scenarioPresetEl) return;
  scenarioPresetEl.innerHTML = '';
  for (const p of scenarioPresets) {
    const opt = document.createElement('option');
    opt.value = String(p.id);
    opt.textContent = String(p.name || p.id);
    scenarioPresetEl.appendChild(opt);
  }

  resolveCurriculumOrder();
  if (scenarioPresets.length > 0) {
    scenarioPresetEl.value = String(scenarioPresets[0].id);
    applyScenarioPreset(scenarioPresets[0], { source: 'preset' });
    curriculumIndex = curriculumOrder.findIndex((id) => id === String(scenarioPresets[0].id));
    updateCurriculumStatus();
  }
}

async function loadScenarioPresetCatalog() {
  try {
    const res = await fetch('/interaction-scenarios.json?v=20260217e', { cache: 'no-store' });
    if (res.ok) {
      const json = await res.json();
      if (Array.isArray(json) && json.length > 0) {
        scenarioPresets = json;
      } else {
        scenarioPresets = [...DEFAULT_SCENARIO_PRESETS];
      }
    } else {
      scenarioPresets = [...DEFAULT_SCENARIO_PRESETS];
    }
  } catch {
    scenarioPresets = [...DEFAULT_SCENARIO_PRESETS];
  }

  populateScenarioPresetDropdown();
}

function dyeModeCode(mode) {
  const m = String(mode || 'noop').toLowerCase();
  if (m === 'eat') return EDGE_DYE_EAT;
  // NO-OP and PASS aliases both map to pass-through.
  return EDGE_DYE_PASS;
}

function effectiveDyeMode(mode, velocityMode) {
  // Contract: velocity BLOCK is the hard boundary condition; disable EAT in this branch.
  const vel = String(velocityMode || 'block').toLowerCase();
  if (vel !== 'pass') return 'noop';
  return String(mode || 'noop').toLowerCase() === 'eat' ? 'eat' : 'noop';
}

function updateEffectiveDyeModeStatus() {
  if (!effectiveDyeModeStatusEl) return;
  const requested = String(dyeModeEl?.value || 'noop').toLowerCase();
  const velocityMode = String(velocityModeEl?.value || 'block').toLowerCase();
  const effective = effectiveDyeMode(requested, velocityMode);
  if (requested === 'eat' && effective !== 'eat') {
    effectiveDyeModeStatusEl.textContent = 'Effective selected-channel dye behavior: NO-OP (EAT disabled by velocity BLOCK)';
    return;
  }
  effectiveDyeModeStatusEl.textContent = `Effective selected-channel dye behavior: ${effective === 'eat' ? 'EAT' : 'NO-OP'}`;
}

function updateControlHints() {
  const velocityMode = String(velocityModeEl?.value || 'block').toLowerCase();
  const requestedDyeMode = String(dyeModeEl?.value || 'noop').toLowerCase();
  const effective = effectiveDyeMode(requestedDyeMode, velocityMode);
  const momentum = clamp(momentumEl?.value, 0, 1);

  if (dyeModeHintEl) {
    dyeModeHintEl.textContent = effective === 'eat'
      ? 'Dye mode: selected channel is removed when fluid crosses the segment.'
      : 'Dye mode: selected channel passes through unchanged.';
  }

  if (velocityModeHintEl) {
    velocityModeHintEl.textContent = velocityMode === 'block'
      ? 'Velocity mode: BLOCK makes this segment a hard flow boundary.'
      : 'Velocity mode: PASS allows fluid to cross this segment.';
  }

  if (momentumHintEl) {
    const pct = Math.round(momentum * 100);
    momentumHintEl.textContent = `Momentum coupling: ${pct}% transfer from fluid to fixture (0% decoupled, 100% fully coupled).`;
  }
}

function velocityModeCode(mode) {
  return String(mode || 'block').toLowerCase() === 'pass' ? EDGE_VEL_PASS : EDGE_VEL_BLOCK;
}

function makeRgbPolicy(channel, mode) {
  const rgb = [EDGE_DYE_PASS, EDGE_DYE_PASS, EDGE_DYE_PASS];
  const idx = String(channel || 'r').toLowerCase() === 'g'
    ? 1
    : (String(channel || 'r').toLowerCase() === 'b' ? 2 : 0);
  rgb[idx] = dyeModeCode(mode);
  return rgb;
}

function permeabilityFromDyeModeRgb(rgb) {
  // Rigid runtime treats dye PASS as DEFLECT unless permeability channel is enabled.
  // For interaction-lab channel semantics, map PASS channels to permeability=1.
  return [0, 1, 2].map((ci) => (Number(rgb?.[ci]) === EDGE_DYE_PASS ? 1 : 0));
}

function emitterTintForScenario() {
  return [135, 190, 255];
}

function makeRigidLineSpec({ channel, dyeMode, velocityMode, momentum }) {
  const cx = 64;
  const cy = 64;
  const halfLen = 24;
  const halfThick = 2;

  const hull = [
    { x: cx - halfLen, y: cy - halfThick },
    { x: cx + halfLen, y: cy - halfThick },
    { x: cx + halfLen, y: cy + halfThick },
    { x: cx - halfLen, y: cy + halfThick },
  ];

  const vel = velocityModeCode(velocityMode);
  const dye = makeRgbPolicy(channel, dyeMode);
  const permeabilityRgb = permeabilityFromDyeModeRgb(dye);
  const sides = hull.length;

  return {
    schemaVersion: 'creature-spec.v2',
    createdAt: new Date().toISOString(),
    name: 'interaction-lab-rigid-line',
    space: { width: 128, height: 128 },
    rigidBodies: [{
      id: 'rigid_0',
      hull,
      mass: 8,
      edgeBodyMode: Array.from({ length: sides }, () => vel),
      edgeDyeMode: Array.from({ length: sides }, () => [...dye]),
      edgePermeabilityRGB: Array.from({ length: sides }, () => [...permeabilityRgb]),
      edgeVelocityMode: Array.from({ length: sides }, () => vel),
      edgeMomentumCoupling: Array.from({ length: sides }, () => clamp(momentum, 0, 1)),
      insideCorrectionEnabled: 1,
      digestEnabled: false,
      digestRGB: [1, 1, 1],
      consumeDyeRGB: [0, 0, 0],
    }],
    softBodies: [],
    hybridJoints: [],
  };
}

function makeSoftLineSpec({ channel, dyeMode, velocityMode, momentum }) {
  const a = { x: 40, y: 64 };
  const b = { x: 88, y: 64 };
  const rest = Math.hypot(b.x - a.x, b.y - a.y);
  const vel = velocityModeCode(velocityMode);
  const dye = makeRgbPolicy(channel, dyeMode);

  return {
    schemaVersion: 'creature-spec.v2',
    createdAt: new Date().toISOString(),
    name: 'interaction-lab-soft-line',
    space: { width: 128, height: 128 },
    rigidBodies: [],
    softBodies: [{
      id: 'soft_0',
      solverMode: 'spring',
      nodes: [a, b],
      springs: [[0, 1, rest, vel, [...dye], vel, clamp(momentum, 0, 1)]],
      insideCorrectionEnabled: 1,
    }],
    hybridJoints: [],
  };
}

function buildScenarioPayload() {
  const fixtureType = String(fixtureTypeEl?.value || 'rigid-line');
  const motionMode = String(motionModeEl?.value || 'pinned');
  const channel = String(dyeChannelEl?.value || 'r');
  const requestedDyeMode = String(dyeModeEl?.value || 'noop');
  const velocityMode = String(velocityModeEl?.value || 'block');
  const dyeMode = effectiveDyeMode(requestedDyeMode, velocityMode);
  const momentum = clamp(momentumEl?.value, 0, 1);
  const overlayShowSegmentIds = !!showSegmentIdsEl?.checked;
  const overlayShowExtraVisuals = !!showExtraVisualsEl?.checked;

  const spec = fixtureType === 'soft-line'
    ? makeSoftLineSpec({ channel, dyeMode, velocityMode, momentum })
    : makeRigidLineSpec({ channel, dyeMode, velocityMode, momentum });

  const circleRadius = Math.max(0, Number(circleRadiusEl?.value) || 12);
  const circleSpeed = Math.max(0, Number(circleSpeedEl?.value) || 0.8);

  const interactionLab = fixtureType === 'soft-line'
    ? {
        enabled: true,
        targetType: 'soft',
        mode: motionMode,
        anchorX: 40,
        anchorY: 64,
        // Soft fixture motion is anchored on the first node; center this node at x=40
        // so the segment midpoint still circles around x=64 (matching rigid fixtures).
        circleCenterX: 40,
        circleCenterY: 64,
        circleRadius,
        circleAngularSpeed: circleSpeed,
        thetaSpin: 0,
        softNodeIndices: [0, 1],
        softBasePositions: [{ x: 40, y: 64 }, { x: 88, y: 64 }],
      }
    : {
        enabled: true,
        targetType: 'rigid',
        rigidIndex: 0,
        mode: motionMode,
        anchorX: 64,
        anchorY: 64,
        anchorTheta: 0,
        circleCenterX: 64,
        circleCenterY: 64,
        circleRadius,
        circleAngularSpeed: circleSpeed,
        thetaSpin: 0,
      };

  const payload = {
    type: 'gpuLabEmbedReset',
    spec,
    options: {
      grid: 128,
      importScale: 1,
      targetSpanFraction: 0.46,
      emitterStrength: 5.2,
      emitterJetVy: 2.2,
      emitterRadius: 7,
      emitterYFraction: 0.1,
      emitterSpin: 0,
      emitterCurlGain: 0,
      emitterDriftGain: 0,
      emitterChaosGain: 0,
      emitterWobbleAmp: 0,
      emitterWobbleFreq: 0,
      emitterLockPosition: true,
      overlayShowSegmentIds,
      overlayShowExtraVisuals,
      interactionLab,
    },
  };

  const truthRow = {
    scenarioPresetId: currentScenarioMeta?.id || null,
    scenarioPresetName: currentScenarioMeta?.name || null,
    scenarioSource: currentScenarioMeta?.source || 'manual',
    fixtureType,
    motionMode,
    channel,
    requestedDyeMode,
    dyeMode,
    effectiveDyeMode: dyeMode,
    velocityMode,
    momentum,
    overlayShowSegmentIds,
    overlayShowExtraVisuals,
  };

  return { payload, truthRow };
}

function renderScenarioDebug(row, payload, extra = {}) {
  scenarioOut.textContent = JSON.stringify({
    truthTableRow: row,
    payload,
    ...extra,
    notes: {
      segmentLabelsInWindTunnel: 'R<body>:<edge> for rigid, S<softSpring> for soft',
      expectedSelectionRule: 'For selected channel at each segment: NO-OP means no dye removal, EAT removes dye',
      velocityDyeContract: 'velocity=BLOCK is treated as hard boundary condition, so requested EAT is auto-disabled (effective mode becomes NO-OP) unless velocity=PASS',
      screenshotWorkflow: 'Load/create scenario -> Apply -> Download screenshot -> send screenshot for analysis',
      generationWorkflow: 'Use presets, curriculum buttons, or Random scenario to auto-generate cases',
    },
  }, null, 2);
}

function postPayloadToEmbed(payload) {
  if (!embedReady || !windFrame?.contentWindow || !payload) return;
  windFrame.contentWindow.postMessage(payload, window.location.origin);
}

function getEmbedStatus() {
  try {
    const api = windFrame?.contentWindow?.__gpuLabApi;
    if (!api || typeof api.getStatus !== 'function') return null;
    return api.getStatus() || null;
  } catch {
    return null;
  }
}

function maybeResyncEmbedMotion(status) {
  if (!lastPayload || !embedReady) return;
  const desiredMode = String(motionModeEl?.value || 'pinned').toLowerCase();
  const actualMode = String(status?.interactionLab?.motion || '').toLowerCase();
  const now = performance.now();

  const frameNow = Number(status?.frame) || 0;
  if (frameNow <= lastEmbedFrame) embedStallFrames += 1;
  else embedStallFrames = 0;
  lastEmbedFrame = frameNow;

  const fixtureVx = Number(status?.interactionLab?.fixturePose?.vx) || 0;
  const fixtureVy = Number(status?.interactionLab?.fixturePose?.vy) || 0;
  const fixtureSpeed = Math.hypot(fixtureVx, fixtureVy);
  const requestedCircleSpeed = Math.max(0, Number(circleSpeedEl?.value) || 0);
  const requestedCircleRadius = Math.max(0, Number(circleRadiusEl?.value) || 0);

  const modeMismatch = desiredMode && actualMode && desiredMode !== actualMode;
  const stalledCircle = desiredMode === 'circle'
    && actualMode === 'circle'
    && requestedCircleSpeed > 0.05
    && requestedCircleRadius > 0.5
    && fixtureSpeed < 0.02;

  if ((modeMismatch || stalledCircle || embedStallFrames >= 3) && (now - lastEmbedResyncAt) > 800) {
    lastEmbedResyncAt = now;
    postPayloadToEmbed(lastPayload);
  }
}

function ensureEmbedStatusPoll() {
  if (embedStatusPollTimer) return;
  embedStatusPollTimer = setInterval(() => {
    const status = getEmbedStatus();
    if (!status) return;
    maybeResyncEmbedMotion(status);
  }, 450);
}

function pushScenario() {
  const { payload, truthRow } = buildScenarioPayload();
  lastPayload = payload;
  renderScenarioDebug(truthRow, payload);
  postPayloadToEmbed(payload);
}

function extractPayloadFromScenarioJson(raw) {
  if (raw && typeof raw === 'object') {
    if (raw.type === 'gpuLabEmbedReset') {
      return { payload: raw, truthRow: null, source: 'payload' };
    }
    if (raw.payload?.type === 'gpuLabEmbedReset') {
      return {
        payload: raw.payload,
        truthRow: raw.truthTableRow || raw.row || null,
        source: 'wrapper',
      };
    }
  }
  throw new Error('Expected JSON to be either gpuLabEmbedReset payload or { payload, truthTableRow }.');
}

function captureWindFramePngDataUrl() {
  const frameDoc = windFrame?.contentDocument;
  const canvas = frameDoc?.getElementById('view');
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error('Wind frame canvas not ready yet. Start/apply scenario and try again.');
  }
  return canvas.toDataURL('image/png');
}

applyBtn?.addEventListener('click', pushScenario);
autoApplyEl?.addEventListener('change', () => {
  updateApplyModeState();
  if (autoApplyEl.checked) pushScenario();
});

if (loadPresetBtn) {
  loadPresetBtn.addEventListener('click', () => {
    const preset = getSelectedPreset();
    if (!preset) return;
    applyScenarioPreset(preset, { source: 'preset' });
    curriculumIndex = curriculumOrder.findIndex((id) => id === String(preset.id));
    updateCurriculumStatus();
    pushScenario();
    flashButton(loadPresetBtn, true, 'Loaded', 'Load failed');
  });
}

if (randomScenarioBtn) {
  randomScenarioBtn.addEventListener('click', () => {
    const randomPreset = generateRandomScenarioPreset();
    applyScenarioPreset(randomPreset, { source: 'random' });
    curriculumIndex = -1;
    updateCurriculumStatus();
    pushScenario();
    flashButton(randomScenarioBtn, true, 'Generated', 'Generate failed');
  });
}

if (curriculumPrevBtn) {
  curriculumPrevBtn.addEventListener('click', () => {
    if (!loadCurriculumStep(curriculumIndex < 0 ? 0 : curriculumIndex - 1)) return;
    pushScenario();
    flashButton(curriculumPrevBtn, true, 'Loaded', 'Load failed');
  });
}

if (curriculumNextBtn) {
  curriculumNextBtn.addEventListener('click', () => {
    if (!loadCurriculumStep(curriculumIndex < 0 ? 0 : curriculumIndex + 1)) return;
    pushScenario();
    flashButton(curriculumNextBtn, true, 'Loaded', 'Load failed');
  });
}

scenarioPresetEl?.addEventListener('change', () => {
  const preset = getSelectedPreset();
  if (!preset) return;
  applyScenarioPreset(preset, { source: 'preset' });
  curriculumIndex = curriculumOrder.findIndex((id) => id === String(preset.id));
  updateCurriculumStatus();
  if (autoApplyEl?.checked) pushScenario();
});

for (const el of [
  fixtureTypeEl,
  motionModeEl,
  circleRadiusEl,
  circleSpeedEl,
  dyeChannelEl,
  dyeModeEl,
  velocityModeEl,
  momentumEl,
  showSegmentIdsEl,
  showExtraVisualsEl,
]) {
  el?.addEventListener('change', () => {
    if (el === motionModeEl) updateMotionControlState();
    if (el === velocityModeEl) updateDyeControlState();
    else if (el === dyeModeEl) {
      updateEffectiveDyeModeStatus();
      updateControlHints();
    }
    if (el === circleRadiusEl || el === circleSpeedEl || el === momentumEl) updateRangeValueLabels();
    if (!currentScenarioMeta || currentScenarioMeta.source !== 'manual') {
      currentScenarioMeta = {
        id: currentScenarioMeta?.id || null,
        name: currentScenarioMeta?.name || 'manual-edit',
        source: 'manual',
      };
    }

    // Motion toggles are coarse behavioral mode switches; apply immediately
    // so selecting "Dragged in circle" always takes effect even if auto-apply is off.
    if (el === motionModeEl) {
      pushScenario();
      return;
    }

    if (autoApplyEl?.checked) pushScenario();
  });
}

for (const el of [circleRadiusEl, circleSpeedEl, momentumEl]) {
  el?.addEventListener('input', () => {
    updateRangeValueLabels();
    if (!currentScenarioMeta || currentScenarioMeta.source !== 'manual') {
      currentScenarioMeta = {
        id: currentScenarioMeta?.id || null,
        name: currentScenarioMeta?.name || 'manual-edit',
        source: 'manual',
      };
    }
    queueAutoApply();
  });
}

circleRadiusEl?.addEventListener('blur', updateRangeValueLabels);
circleSpeedEl?.addEventListener('blur', updateRangeValueLabels);
momentumEl?.addEventListener('blur', updateRangeValueLabels);

windFrame?.addEventListener('load', () => {
  embedReady = false;
  embedStallFrames = 0;
  lastEmbedFrame = -1;
  setTimeout(() => {
    if (lastPayload) postPayloadToEmbed(lastPayload);
  }, 250);
});

window.addEventListener('message', (event) => {
  if (event.origin !== window.location.origin) return;
  const data = event.data || {};
  if (data.type === 'gpuLabEmbedReady') {
    embedReady = true;
    embedStallFrames = 0;
    lastEmbedFrame = -1;
    ensureEmbedStatusPoll();
    if (lastPayload) postPayloadToEmbed(lastPayload);
    else pushScenario();
    return;
  }
  if (data.type === 'gpuLabEmbedResetAck' && data.ok === false) {
    scenarioOut.textContent = JSON.stringify({
      ok: false,
      error: String(data.error || 'unknown embed reset failure'),
      lastPayload,
    }, null, 2);
  }
});

if (copyScenarioBtn) {
  copyScenarioBtn.addEventListener('click', async () => {
    const ok = await copyTextToClipboard(scenarioOut?.textContent || '');
    flashButton(copyScenarioBtn, ok, 'Copied', 'Copy failed');
  });
}

if (downloadScenarioBtn) {
  downloadScenarioBtn.addEventListener('click', () => {
    try {
      const snapshot = scenarioOut?.textContent
        ? JSON.parse(scenarioOut.textContent)
        : { payload: lastPayload };
      downloadTextFile(`interaction-scenario-${timestampTag()}.json`, JSON.stringify(snapshot, null, 2));
      flashButton(downloadScenarioBtn, true, 'Downloaded', 'Download failed');
    } catch {
      flashButton(downloadScenarioBtn, false, 'Downloaded', 'Download failed');
    }
  });
}

if (loadScenarioBtn && loadScenarioFile) {
  loadScenarioBtn.addEventListener('click', () => loadScenarioFile.click());
  loadScenarioFile.addEventListener('change', async () => {
    const f = loadScenarioFile.files?.[0];
    if (!f) return;
    try {
      const text = await f.text();
      const raw = JSON.parse(text);
      const { payload, truthRow, source } = extractPayloadFromScenarioJson(raw);
      lastPayload = payload;
      currentScenarioMeta = {
        id: truthRow?.scenarioPresetId || null,
        name: truthRow?.scenarioPresetName || f.name,
        source: 'file',
      };
      curriculumIndex = -1;
      updateCurriculumStatus();
      renderScenarioDebug(truthRow || { loadedFromFile: f.name }, payload, {
        loaded: { source, file: f.name },
      });
      postPayloadToEmbed(payload);
      flashButton(loadScenarioBtn, true, 'Loaded', 'Load failed');
    } catch (err) {
      renderScenarioDebug({ error: 'load-failed' }, lastPayload, {
        loadError: String(err?.message || err),
      });
      flashButton(loadScenarioBtn, false, 'Loaded', 'Load failed');
    } finally {
      loadScenarioFile.value = '';
    }
  });
}

if (downloadScreenshotBtn) {
  downloadScreenshotBtn.addEventListener('click', () => {
    try {
      const dataUrl = captureWindFramePngDataUrl();
      downloadDataUrl(`interaction-shot-${timestampTag()}.png`, dataUrl);
      if (captureThumb) {
        captureThumb.src = dataUrl;
        captureThumb.style.display = 'block';
      }
      flashButton(downloadScreenshotBtn, true, 'Downloaded', 'Shot failed');
    } catch {
      flashButton(downloadScreenshotBtn, false, 'Downloaded', 'Shot failed');
    }
  });
}

updateRangeValueLabels();
updateMotionControlState();
updateDyeControlState();
updateApplyModeState();

window.addEventListener('beforeunload', () => {
  if (embedStatusPollTimer) {
    clearInterval(embedStatusPollTimer);
    embedStatusPollTimer = null;
  }
});

loadScenarioPresetCatalog().finally(() => {
  pushScenario();
});
