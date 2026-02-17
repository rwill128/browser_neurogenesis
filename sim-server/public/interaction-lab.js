const windFrame = document.getElementById('windFrame');
const fixtureTypeEl = document.getElementById('fixtureType');
const motionModeEl = document.getElementById('motionMode');
const circleRadiusEl = document.getElementById('circleRadius');
const circleSpeedEl = document.getElementById('circleSpeed');
const dyeChannelEl = document.getElementById('dyeChannel');
const dyeModeEl = document.getElementById('dyeMode');
const velocityModeEl = document.getElementById('velocityMode');
const momentumEl = document.getElementById('momentum');
const applyBtn = document.getElementById('applyBtn');
const autoApplyEl = document.getElementById('autoApply');
const scenarioOut = document.getElementById('scenarioOut');

const EDGE_DYE_PASS = 0;
const EDGE_DYE_EAT = 2;
const EDGE_VEL_PASS = 0;
const EDGE_VEL_BLOCK = 1;

let embedReady = false;
let lastPayload = null;

function clamp(v, lo, hi) {
  const n = Number(v);
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

function dyeModeCode(mode) {
  const m = String(mode || 'noop').toLowerCase();
  return m === 'eat' ? EDGE_DYE_EAT : EDGE_DYE_PASS;
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
  const dyeMode = String(dyeModeEl?.value || 'noop');
  const velocityMode = String(velocityModeEl?.value || 'block');
  const momentum = clamp(momentumEl?.value, 0, 1);

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
        circleCenterX: 64,
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
      interactionLab,
    },
  };

  const truthRow = {
    fixtureType,
    motionMode,
    channel,
    dyeMode,
    velocityMode,
    momentum,
  };

  return { payload, truthRow };
}

function renderScenarioDebug(row, payload) {
  scenarioOut.textContent = JSON.stringify({
    truthTableRow: row,
    payload,
    notes: {
      segmentLabelsInWindTunnel: 'R<body>:<edge> for rigid, S<softSpring> for soft',
      expectedSelectionRule: 'For each segment/channel: avg(BLOCK,PASS,EAT) along segment -> highest wins',
    },
  }, null, 2);
}

function pushScenario() {
  const { payload, truthRow } = buildScenarioPayload();
  lastPayload = payload;
  renderScenarioDebug(truthRow, payload);
  if (!embedReady || !windFrame?.contentWindow) return;
  windFrame.contentWindow.postMessage(payload, window.location.origin);
}

applyBtn?.addEventListener('click', pushScenario);

for (const el of [
  fixtureTypeEl,
  motionModeEl,
  circleRadiusEl,
  circleSpeedEl,
  dyeChannelEl,
  dyeModeEl,
  velocityModeEl,
  momentumEl,
]) {
  el?.addEventListener('change', () => {
    if (autoApplyEl?.checked) pushScenario();
  });
}

windFrame?.addEventListener('load', () => {
  embedReady = false;
  setTimeout(() => {
    if (lastPayload && windFrame?.contentWindow) {
      windFrame.contentWindow.postMessage(lastPayload, window.location.origin);
    }
  }, 250);
});

window.addEventListener('message', (event) => {
  if (event.origin !== window.location.origin) return;
  const data = event.data || {};
  if (data.type === 'gpuLabEmbedReady') {
    embedReady = true;
    pushScenario();
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

pushScenario();
