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

function clamp(v, lo, hi) {
  const n = Number(v);
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
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

function renderScenarioDebug(row, payload, extra = {}) {
  scenarioOut.textContent = JSON.stringify({
    truthTableRow: row,
    payload,
    ...extra,
    notes: {
      segmentLabelsInWindTunnel: 'R<body>:<edge> for rigid, S<softSpring> for soft',
      expectedSelectionRule: 'For each segment/channel: EAT=true removes dye, EAT=false is no-op',
      screenshotWorkflow: 'Load scenario JSON -> Apply -> Download screenshot -> send screenshot for analysis',
    },
  }, null, 2);
}

function postPayloadToEmbed(payload) {
  if (!embedReady || !windFrame?.contentWindow || !payload) return;
  windFrame.contentWindow.postMessage(payload, window.location.origin);
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
    if (lastPayload) postPayloadToEmbed(lastPayload);
  }, 250);
});

window.addEventListener('message', (event) => {
  if (event.origin !== window.location.origin) return;
  const data = event.data || {};
  if (data.type === 'gpuLabEmbedReady') {
    embedReady = true;
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

pushScenario();
