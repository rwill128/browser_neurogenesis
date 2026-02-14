const statusEl = document.getElementById('status');
const canvas = document.getElementById('canvas');

const worldSelect = document.getElementById('worldSelect');
const newWorldBtn = document.getElementById('newWorldBtn');

const scenarioSelect = document.getElementById('scenarioSelect');
const seedInput = document.getElementById('seedInput');
const setScenarioBtn = document.getElementById('setScenarioBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resumeBtn = document.getElementById('resumeBtn');
const runModeSelect = document.getElementById('runModeSelect');
const setRunModeBtn = document.getElementById('setRunModeBtn');
const fitBtn = document.getElementById('fitBtn');
const showFluidVelocityToggle = document.getElementById('showFluidVelocityToggle');
const showFluidActiveTilesToggle = document.getElementById('showFluidActiveTilesToggle');

const prevCreatureBtn = document.getElementById('prevCreatureBtn');
const nextCreatureBtn = document.getElementById('nextCreatureBtn');
const toggleFollowBtn = document.getElementById('toggleFollowBtn');
const creatureStatsModeBtn = document.getElementById('creatureStatsModeBtn');

const refreshConfigBtn = document.getElementById('refreshConfigBtn');
const applyConfigBtn = document.getElementById('applyConfigBtn');

const worldStatsEl = document.getElementById('worldStats');
const creatureStatsEl = document.getElementById('creatureStats');
const creaturePointDetailsEl = document.getElementById('creaturePointDetails');
const creatureSpringDetailsEl = document.getElementById('creatureSpringDetails');
const configEditorEl = document.getElementById('configEditor');

const historySliderEl = document.getElementById('historySlider');
const historyLiveBtn = document.getElementById('historyLiveBtn');
const historyInfoEl = document.getElementById('historyInfo');

const ctx = canvas.getContext('2d');
const renderPerfByWorld = new Map();

let renderRafPending = false;
let renderPanelsOnNextFrame = false;

function scheduleRender({ includePanels = false } = {}) {
  if (includePanels) renderPanelsOnNextFrame = true;
  if (renderRafPending) return;
  renderRafPending = true;

  requestAnimationFrame(() => {
    renderRafPending = false;
    if (!currentWorldId) return;
    const snap = lastSnapshotByWorld.get(currentWorldId);
    if (!snap) return;
    const shouldRenderPanels = renderPanelsOnNextFrame;
    renderPanelsOnNextFrame = false;
    drawSnapshot(snap, { includePanels: shouldRenderPanels });
  });
}

let creatureStatsMode = 'compact';
let showFluidVelocityArrows = false;
let showFluidActiveTilesOverlay = false;

function syncCreatureStatsModeUI() {
  if (creatureStatsModeBtn) {
    creatureStatsModeBtn.textContent = creatureStatsMode === 'compact' ? 'Mode: Compact' : 'Mode: Expanded';
  }
  const showDetails = creatureStatsMode === 'expanded';
  if (creaturePointDetailsEl) {
    creaturePointDetailsEl.style.display = showDetails ? '' : 'none';
    if (creaturePointDetailsEl.previousElementSibling) {
      creaturePointDetailsEl.previousElementSibling.style.display = showDetails ? '' : 'none';
    }
  }
  if (creatureSpringDetailsEl) {
    creatureSpringDetailsEl.style.display = showDetails ? '' : 'none';
    if (creatureSpringDetailsEl.previousElementSibling) {
      creatureSpringDetailsEl.previousElementSibling.style.display = showDetails ? '' : 'none';
    }
  }
}

function compactCreatureEntries(entries) {
  const keep = new Set([
    '@@Identity', 'id', 'following', 'points', 'springs', 'nodeTypes',
    '@@Lifecycle', 'birthOrigin', 'generation', 'ticksSinceBirth', 'canReproduce',
    '@@Energy', 'energy', 'currentMaxEnergy', 'reproThreshold', 'gain.photo', 'gain.eat', 'gain.pred',
    '@@Growth', 'growth.events', 'growth.nodesAdded', 'growth.energySpent', 'growth.supp.energy', 'growth.supp.population',
    '@@Topology', 'topology.version', 'topology.rlResets',
    '@@Reproduction Suppression', 'reproSupp.density', 'reproSupp.resources'
  ]);
  return entries.filter(([k]) => (typeof k === 'string' && k.startsWith('@@')) || keep.has(k));
}

const NODE_TYPE_COLORS = {
  PREDATOR: '#ff4d4d',
  EATER: '#ff9f1c',
  PHOTOSYNTHETIC: '#7CFF6B',
  NEURON: '#b48bff',
  EMITTER: '#4dd6ff',
  SWIMMER: '#4d7cff',
  EYE: '#ffffff',
  JET: '#00e5ff',
  ATTRACTOR: '#ffd24d',
  REPULSOR: '#ff4de1'
};

const CONFIG_FIELDS = [
  'PHOTOSYNTHESIS_EFFICIENCY',
  'globalNutrientMultiplier',
  'globalLightMultiplier',
  'CREATURE_POPULATION_FLOOR',
  'CREATURE_POPULATION_CEILING',
  'PARTICLE_POPULATION_FLOOR',
  'PARTICLE_POPULATION_CEILING',
  'ENERGY_PER_PARTICLE',
  'EATER_NODE_ENERGY_COST',
  'PREDATOR_NODE_ENERGY_COST',
  'SWIMMER_NODE_ENERGY_COST',
  'JET_NODE_ENERGY_COST',
  'ATTRACTOR_NODE_ENERGY_COST',
  'REPULSOR_NODE_ENERGY_COST',
  'FLUID_CURRENT_STRENGTH_ON_BODY',
  'FLUID_SOLVER_ITERATIONS_VELOCITY',
  'FLUID_SOLVER_ITERATIONS_PRESSURE',
  'FLUID_SOLVER_ITERATIONS_DENSITY',
  'FLUID_STEP_EVERY_N_TICKS',
  'FLUID_MOMENTUM_ONLY_STEP_EVERY_N_TICKS',
  'FLUID_MOMENTUM_ACTIVITY_SPEED_THRESHOLD',
  'FLUID_FADE_RATE',
  'MIN_VISCOSITY_MULTIPLIER',
  'MAX_VISCOSITY_MULTIPLIER',
  'VISCOSITY_LANDSCAPE_NOISE_SCALE',
  'VISCOSITY_LANDSCAPE_OCTAVES',
  'VISCOSITY_LANDSCAPE_LACUNARITY',
  'VISCOSITY_LANDSCAPE_GAIN',
  'VISCOSITY_LANDSCAPE_CONTRAST',
  'VISCOSITY_LANDSCAPE_BANDS',
  'LANDSCAPE_DYE_EMITTERS_ENABLED',
  'LANDSCAPE_DYE_EMITTER_COUNT',
  'LANDSCAPE_DYE_EMITTER_STRENGTH_MIN',
  'LANDSCAPE_DYE_EMITTER_STRENGTH_MAX',
  'LANDSCAPE_DYE_EMITTER_RADIUS_CELLS',
  'LANDSCAPE_VELOCITY_EMITTERS_ENABLED',
  'LANDSCAPE_VELOCITY_EMITTER_COUNT',
  'LANDSCAPE_VELOCITY_EMITTER_STRENGTH_MIN',
  'LANDSCAPE_VELOCITY_EMITTER_STRENGTH_MAX',
  'LANDSCAPE_VELOCITY_EMITTER_RADIUS_CELLS',
  'LANDSCAPE_VELOCITY_EMITTER_LOCAL_SPEED_CAP',
  'LANDSCAPE_VELOCITY_EMITTER_BUDGET_MAX',
  'LANDSCAPE_VELOCITY_EMITTER_BUDGET_REFILL_PER_SEC',
  'SOFT_BODY_PUSH_STRENGTH',
  'BODY_REPULSION_STRENGTH',
  'SPRING_OVERSTRETCH_KILL_ENABLED',
  'FORCE_ALL_SPRINGS_RIGID',
  'RIGID_CONSTRAINT_PROJECTION_ENABLED',
  'RIGID_CONSTRAINT_PROJECTION_ITERATIONS',
  'RIGID_CONSTRAINT_MAX_RELATIVE_ERROR',
  'TRIANGLE_EXTRUSION_MUTATION_CHANCE_MULTIPLIER',
  'PHYSICS_MOTION_GUARD_ENABLED',
  'PHYSICS_NONFINITE_FORCE_ZERO',
  'PHYSICS_MAX_ACCELERATION_MAGNITUDE',
  'PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP',
  'REPRO_RESOURCE_MIN_NUTRIENT',
  'REPRO_RESOURCE_MIN_LIGHT',
  'FAILED_REPRODUCTION_COOLDOWN_TICKS'
];

let ws = null;
let currentWorldId = null;
let wsConnected = false;
let lastWsMessageAt = 0;

const cameraByWorld = new Map();
const selectionByWorld = new Map();
const followByWorld = new Map();
const lastSnapshotByWorld = new Map();
const lastStatusByWorld = new Map();
const configDraftByWorld = new Map();
const frameBufferByWorld = new Map();
const scrubTickByWorld = new Map();
const scrubFetchInFlightByWorld = new Map();
const scrubFetchLastAtByWorld = new Map();

function fmt(n, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return '—';
  return x.toFixed(digits);
}

function getScrubTick(worldId) {
  const v = scrubTickByWorld.get(worldId);
  if (v === null || v === undefined) return null;
  const n = Math.floor(Number(v));
  return Number.isFinite(n) ? Math.max(0, n) : null;
}

function getFrameBufferMeta(worldId) {
  return frameBufferByWorld.get(worldId) || {
    available: 0,
    max: 0,
    stepStride: 1,
    latestSeq: null,
    latestTick: null,
    archiveFrames: 0,
    archiveOldestTick: null,
    archiveLatestTick: null
  };
}

function updateHistoryUi(worldId) {
  if (!historySliderEl || !historyInfoEl) return;

  const meta = getFrameBufferMeta(worldId);
  const oldestTick = Number.isFinite(Number(meta.archiveOldestTick)) ? Math.max(0, Math.floor(Number(meta.archiveOldestTick))) : 0;
  const latestTick = Number.isFinite(Number(meta.archiveLatestTick)) ? Math.max(oldestTick, Math.floor(Number(meta.archiveLatestTick))) : oldestTick;
  const archiveFrames = Math.max(1, Math.floor(Number(meta.archiveFrames) || 1));

  const scrubTick = getScrubTick(worldId);
  const isLive = (scrubTick === null) || scrubTick >= latestTick;
  const safeTick = isLive ? latestTick : Math.max(oldestTick, Math.min(latestTick, scrubTick));
  if (!isLive) scrubTickByWorld.set(worldId, safeTick);

  const stepStride = Math.max(1, Math.floor(Number(meta.stepStride) || 1));
  const toFrameIndex = (tick) => {
    const t = Math.max(oldestTick, Math.min(latestTick, Math.floor(Number(tick) || oldestTick)));
    return Math.max(0, Math.min(archiveFrames - 1, Math.round((t - oldestTick) / stepStride)));
  };

  historySliderEl.min = '0';
  historySliderEl.max = String(Math.max(0, archiveFrames - 1));
  historySliderEl.value = String(toFrameIndex(safeTick));

  if (isLive) {
    historyInfoEl.textContent = `live @ tick ${latestTick} · archive ${archiveFrames} frame(s)`;
  } else {
    const frameIdx = toFrameIndex(safeTick);
    historyInfoEl.textContent = `frame ${frameIdx}/${Math.max(0, archiveFrames - 1)} · tick ${safeTick} (latest ${latestTick})`;
  }
}

async function loadScrubFrameByTick(worldId, tick) {
  const safeTick = Math.max(0, Math.floor(Number(tick) || 0));
  if (!worldId) return;
  if (scrubFetchInFlightByWorld.get(worldId)) return;

  scrubFetchInFlightByWorld.set(worldId, true);
  scrubFetchLastAtByWorld.set(worldId, Date.now());

  try {
    const snap = await apiGet(`/api/worlds/${encodeURIComponent(worldId)}/frame/${encodeURIComponent(safeTick)}`);
    if (!snap || currentWorldId !== worldId) return;

    lastSnapshotByWorld.set(worldId, snap);
    scheduleRender({ includePanels: true });
  } catch {
    // ignore transient history misses while archive grows
  } finally {
    scrubFetchInFlightByWorld.set(worldId, false);
  }
}

function setScrubTick(worldId, nextTickOrNull) {
  if (!worldId) return;

  const meta = getFrameBufferMeta(worldId);
  const oldestTick = Number.isFinite(Number(meta.archiveOldestTick)) ? Math.max(0, Math.floor(Number(meta.archiveOldestTick))) : 0;
  const latestTick = Number.isFinite(Number(meta.archiveLatestTick)) ? Math.max(oldestTick, Math.floor(Number(meta.archiveLatestTick))) : oldestTick;

  if (nextTickOrNull === null || nextTickOrNull === undefined) {
    scrubTickByWorld.set(worldId, null);
    updateHistoryUi(worldId);
    scheduleRender({ includePanels: true });
    return;
  }

  const rawTick = Number(nextTickOrNull);
  const parsedTick = Number.isFinite(rawTick) ? Math.floor(rawTick) : latestTick;
  const safeTick = Math.max(oldestTick, Math.min(latestTick, parsedTick));
  scrubTickByWorld.set(worldId, safeTick);
  updateHistoryUi(worldId);
  loadScrubFrameByTick(worldId, safeTick);
}

function colorForVertex(v) {
  const name = v?.nodeTypeName || null;
  if (name && NODE_TYPE_COLORS[name]) return NODE_TYPE_COLORS[name];
  return 'rgba(255,255,255,0.75)';
}

function asDisplayValue(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return '—';
    if (Math.abs(v) >= 1000) return String(Math.round(v));
    if (Math.abs(v) >= 1) return String(Number(v.toFixed(3)));
    return String(Number(v.toFixed(6)));
  }
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'string') return v;
  return JSON.stringify(v);
}

function setKV(el, entries) {
  el.innerHTML = '';
  for (const [k, v] of entries) {
    if (typeof k === 'string' && k.startsWith('@@')) {
      const ds = document.createElement('div');
      ds.className = 'section';
      ds.textContent = k.slice(2);
      el.appendChild(ds);
      continue;
    }
    const dk = document.createElement('div');
    dk.className = 'k';
    dk.textContent = k;
    const dv = document.createElement('div');
    dv.className = 'v';
    dv.textContent = asDisplayValue(v);
    el.appendChild(dk);
    el.appendChild(dv);
  }
}

function setDetailText(el, text) {
  if (!el) return;
  el.textContent = text;
}

function buildPointDetailsText(selected) {
  const verts = Array.isArray(selected?.vertices) ? selected.vertices : [];
  if (verts.length === 0) return 'no points';

  const header = '# idx type move x y r m eye target grabber grabbing exert sees tMag tDir';
  const lines = [header];

  for (const v of verts) {
    lines.push([
      'P',
      asDisplayValue(v.index),
      v.nodeTypeName || v.nodeType || '—',
      v.movementTypeName || v.movementType || '—',
      asDisplayValue(v.x),
      asDisplayValue(v.y),
      asDisplayValue(v.radius),
      asDisplayValue(v.mass),
      v.isDesignatedEye ? '1' : '0',
      v.eyeTargetTypeName || v.eyeTargetType || '—',
      v.canBeGrabber ? '1' : '0',
      v.isGrabbing ? '1' : '0',
      asDisplayValue(v.currentExertionLevel),
      v.seesTarget ? '1' : '0',
      asDisplayValue(v.nearestTargetMagnitude),
      asDisplayValue(v.nearestTargetDirection)
    ].join(' '));
  }

  return lines.join('\n');
}

function buildSpringDetailsText(selected) {
  const springs = Array.isArray(selected?.springs) ? selected.springs : [];
  if (springs.length === 0) return 'no springs';

  const header = '# idx a b rigid rest curr strain stiff damp';
  const lines = [header];

  springs.forEach((s, idx) => {
    lines.push([
      'S',
      idx,
      asDisplayValue(s.a),
      asDisplayValue(s.b),
      s.isRigid ? '1' : '0',
      asDisplayValue(s.restLength),
      asDisplayValue(s.currentLength),
      asDisplayValue(s.strain),
      asDisplayValue(s.stiffness),
      asDisplayValue(s.dampingFactor)
    ].join(' '));
  });

  return lines.join('\n');
}

function getCamera(worldId) {
  if (!cameraByWorld.has(worldId)) {
    cameraByWorld.set(worldId, {
      zoom: 1,
      offsetX: 0,
      offsetY: 0,
      initialized: false,
      worldW: 1,
      worldH: 1
    });
  }
  return cameraByWorld.get(worldId);
}

function fitCameraToWorld(worldId, worldW, worldH) {
  const cam = getCamera(worldId);
  const rect = canvas.getBoundingClientRect();
  const pad = 16;

  const w = Math.max(1, Number(worldW) || 1);
  const h = Math.max(1, Number(worldH) || 1);
  const availW = Math.max(1, rect.width - pad * 2);
  const availH = Math.max(1, rect.height - pad * 2);

  const zoom = Math.min(availW / w, availH / h);

  cam.zoom = Math.max(0.0005, Math.min(zoom, 50));
  cam.offsetX = pad + (availW - w * cam.zoom) / 2;
  cam.offsetY = pad + (availH - h * cam.zoom) / 2;
  cam.initialized = true;
  cam.worldW = w;
  cam.worldH = h;
}

function screenToWorld(cam, sx, sy) {
  return {
    x: (sx - cam.offsetX) / cam.zoom,
    y: (sy - cam.offsetY) / cam.zoom
  };
}

function zoomAround(cam, sx, sy, zoomFactor) {
  const before = screenToWorld(cam, sx, sy);
  const nextZoom = Math.max(0.0005, Math.min(cam.zoom * zoomFactor, 80));
  cam.zoom = nextZoom;
  cam.offsetX = sx - before.x * cam.zoom;
  cam.offsetY = sy - before.y * cam.zoom;
}

function eventToCanvasXY(ev) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ev.clientX - rect.left,
    y: ev.clientY - rect.top
  };
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', () => {
  resizeCanvas();
  if (currentWorldId) {
    const cam = getCamera(currentWorldId);
    cam.initialized = false;
  }
});
resizeCanvas();

async function apiGet(path) {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

async function apiPost(path, payload = null) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: payload ? JSON.stringify(payload) : '{}'
  });
  const json = await res.json().catch(() => null);
  if (!res.ok) throw new Error(json?.error || `${res.status} ${res.statusText}`);
  return json;
}

function sortedCreatures(snap) {
  const arr = Array.isArray(snap?.creatures) ? [...snap.creatures] : [];
  arr.sort((a, b) => Number(a?.id || 0) - Number(b?.id || 0));
  return arr;
}

function updateFollowButton() {
  if (!currentWorldId) {
    toggleFollowBtn.textContent = 'Follow (F)';
    return;
  }
  const sel = selectionByWorld.get(currentWorldId);
  const follow = followByWorld.get(currentWorldId);
  const on = (sel !== null && sel !== undefined && String(sel) === String(follow));
  toggleFollowBtn.textContent = on ? 'Following (F)' : 'Follow (F)';
}

function renderConfigEditor(worldId) {
  const draft = configDraftByWorld.get(worldId) || {};
  configEditorEl.innerHTML = '';

  for (const key of CONFIG_FIELDS) {
    const row = document.createElement('div');
    row.className = 'configField';

    const label = document.createElement('label');
    label.textContent = key;

    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.dataset.configKey = key;
    const v = draft[key];
    if (typeof v === 'number' && Number.isFinite(v)) input.value = String(v);
    else input.value = '';

    row.appendChild(label);
    row.appendChild(input);
    configEditorEl.appendChild(row);
  }
}

async function refreshConfig(worldId) {
  if (!worldId) return;
  const out = await apiGet(`/api/worlds/${encodeURIComponent(worldId)}/config`);
  configDraftByWorld.set(worldId, out?.values || {});
  renderConfigEditor(worldId);
}

async function applyConfig(worldId) {
  if (!worldId) return;

  const overrides = {};
  for (const input of configEditorEl.querySelectorAll('input[data-config-key]')) {
    const key = input.dataset.configKey;
    const raw = String(input.value || '').trim();
    if (!raw) continue;
    const n = Number(raw);
    if (!Number.isFinite(n)) continue;
    overrides[key] = n;
  }

  if (Object.keys(overrides).length === 0) return;

  await apiPost(`/api/worlds/${encodeURIComponent(worldId)}/control/configOverrides`, { overrides });
  await refreshConfig(worldId);
}

function renderPanels(worldId) {
  const snap = lastSnapshotByWorld.get(worldId) || null;
  const status = lastStatusByWorld.get(worldId) || null;
  const cam = getCamera(worldId);

  if (!snap) {
    setKV(worldStatsEl, [['world', worldId], ['snapshot', 'waiting…']]);
    setKV(creatureStatsEl, [['selected', 'none']]);
    setDetailText(creaturePointDetailsEl, 'waiting for snapshot…');
    setDetailText(creatureSpringDetailsEl, 'waiting for snapshot…');
    updateFollowButton();
    return;
  }

  const removedTotal = Number(snap?.instabilityTelemetry?.totalRemoved) || 0;
  const reasons = snap?.instabilityTelemetry?.removedByReason || {};
  const topReason = Object.keys(reasons).sort((a, b) => (reasons[b] || 0) - (reasons[a] || 0))[0] || '—';

  const energyGains = snap?.worldStats?.globalEnergyGains || {};
  const energyCosts = snap?.worldStats?.globalEnergyCosts || {};
  const mutationStats = snap?.mutationStats || snap?.worldStats?.mutationStats || {};
  const mutationEntries = Object.entries(mutationStats)
    .map(([k, v]) => [k, Number(v) || 0])
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  const mutationTopSummary = mutationEntries.slice(0, 8).map(([k, v]) => `${k}:${v}`).join(', ');

  const scrubTick = getScrubTick(worldId);
  const frameMeta = getFrameBufferMeta(worldId);

  setKV(worldStatsEl, [
    ['@@World'],
    ['world', worldId],
    ['scenario', snap.scenario],
    ['runMode', status?.runMode || 'realtime'],
    ['tick', snap.tick],
    ['time', snap.time],
    ['sps', status?.stepsPerSecond],
    ['dt', status?.dt],
    ['zoom', cam.zoom],

    ['@@Population & Frames'],
    ['creatures', snap.populations?.creatures],
    ['particles', snap.populations?.particles],
    ['fluidActiveCells', snap.fluid?.activeCells],
    ['viewTick', scrubTick === null ? `live@${frameMeta.archiveLatestTick ?? snap.tick}` : scrubTick],
    ['frameBuffer', `${frameMeta.available || 0}/${frameMeta.max || 0}`],
    ['archiveFrames', frameMeta.archiveFrames || 0],
    ['archiveRange', `${frameMeta.archiveOldestTick ?? 0}..${frameMeta.archiveLatestTick ?? snap.tick}`],
    ['frameStride', frameMeta.stepStride || 1],

    ['@@Instability'],
    ['removedTotal', removedTotal],
    ['topRemovalReason', `${topReason} (${reasons[topReason] || 0})`],
    ['removedByPhysicsKind', JSON.stringify(snap?.instabilityTelemetry?.removedByPhysicsKind || {})],
    ['removedByBirthOrigin', JSON.stringify(snap?.instabilityTelemetry?.removedByBirthOrigin || {})],
    ['removedByLifecycleStage', JSON.stringify(snap?.instabilityTelemetry?.removedByLifecycleStage || {})],

    ['@@Energy'],
    ['globalGain.photo', energyGains?.photosynthesis],
    ['globalGain.eat', energyGains?.eating],
    ['globalGain.pred', energyGains?.predation],
    ['globalCost.base', energyCosts?.baseNodes],
    ['globalCost.neuron', energyCosts?.neuronNodes],
    ['globalCost.eater', energyCosts?.eaterNodes],
    ['globalCost.predator', energyCosts?.predatorNodes],

    ['@@Mutation Stats'],
    ['mutationStatKeys', mutationEntries.length],
    ['mutationTop', mutationTopSummary || '—'],
    ['mutationStats', JSON.stringify(mutationStats)]
  ]);

  const selectedId = selectionByWorld.get(worldId) || null;
  const selected = selectedId !== null
    ? (snap.creatures || []).find((c) => String(c.id) === String(selectedId))
    : null;

  if (!selected) {
    setKV(creatureStatsEl, [
      ['selected', 'none'],
      ['hint', 'click creature or use prev/next']
    ]);
    setDetailText(creaturePointDetailsEl, 'select a creature to inspect per-point details');
    setDetailText(creatureSpringDetailsEl, 'select a creature to inspect per-spring details');
  } else {
    const fs = selected.fullStats || {};
    const growth = fs.growth || {};
    const growthProgram = growth.program || {};
    const repro = fs.reproductionSuppression || {};
    const costs = fs.energyCostsByType || {};
    const gains = fs.energyGains || {};
    const topo = fs.topology || {};
    const following = String(followByWorld.get(worldId) || '') === String(selected.id) ? 'yes' : 'no';

    const creatureEntries = [
      ['@@Identity'],
      ['id', selected.id],
      ['following', following],
      ['centerX', selected.center?.x],
      ['centerY', selected.center?.y],
      ['points', selected.vertices?.length],
      ['springs', selected.springs?.length],
      ['nodeTypes', JSON.stringify(selected.nodeTypeCounts || {})],

      ['@@Lifecycle'],
      ['birthOrigin', fs.birthOrigin],
      ['generation', fs.generation],
      ['parentBodyId', fs.parentBodyId],
      ['lineageRootId', fs.lineageRootId],
      ['ticksSinceBirth', fs.ticksSinceBirth],
      ['absoluteAgeTicks', fs.absoluteAgeTicks],
      ['reproEventsCompleted', fs.reproductionEventsCompleted],
      ['ticksSinceLastRepro', fs.ticksSinceLastReproduction],
      ['canReproduce', fs.canReproduce],

      ['@@Energy'],
      ['energy', selected.energy],
      ['currentMaxEnergy', fs.currentMaxEnergy],
      ['reproThreshold', fs.reproductionEnergyThreshold],
      ['gain.photo', gains.photosynthesis],
      ['gain.eat', gains.eating],
      ['gain.pred', gains.predation],

      ['@@Morphology & Actuation'],
      ['stiffness(avg)', fs.stiffness],
      ['damping(avg)', fs.damping],
      ['motorInterval', fs.motorImpulseInterval],
      ['motorCap', fs.motorImpulseMagnitudeCap],
      ['emitterStrength', fs.emitterStrength],
      ['emitterDir', `${asDisplayValue(fs?.emitterDirection?.x)}, ${asDisplayValue(fs?.emitterDirection?.y)}`],
      ['actEvals', selected.actuationTelemetry?.evaluations],
      ['actSkips', selected.actuationTelemetry?.skips],
      ['actAvgInterval', selected.actuationTelemetry?.avgEffectiveInterval],

      ['@@Reproduction Genes'],
      ['numOffspring', fs.numOffspring],
      ['offspringRadius', fs.offspringSpawnRadius],
      ['pointAddChance', fs.pointAddChance],
      ['springConnectionRadius', fs.springConnectionRadius],
      ['reproCooldownGene', fs.reproductionCooldownGene],
      ['effectiveReproCooldown', fs.effectiveReproductionCooldown],
      ['rewardStrategy', fs.rewardStrategy],

      ['@@Dye Ecology'],
      ['dye.hue', fs.dyePreferredHue],
      ['dye.tolerance', fs.dyeHueTolerance],
      ['dye.gain', fs.dyeResponseGain],
      ['dye.sign', fs.dyeResponseSign],
      ['dye.affinity', JSON.stringify(fs.dyeNodeTypeAffinity || {})],

      ['@@Growth'],
      ['growth.events', growth.eventsCompleted],
      ['growth.nodesAdded', growth.nodesAdded],
      ['growth.energySpent', growth.totalEnergySpent],
      ['growth.supp.energy', growth.suppressedByEnergy],
      ['growth.supp.cooldown', growth.suppressedByCooldown],
      ['growth.supp.population', growth.suppressedByPopulation],
      ['growth.supp.dye', growth.suppressedByDye],
      ['growth.supp.maxPoints', growth.suppressedByMaxPoints],
      ['growth.supp.noCapacity', growth.suppressedByNoCapacity],
      ['growth.supp.chanceRoll', growth.suppressedByChanceRoll],
      ['growth.supp.placement', growth.suppressedByPlacement],
      ['growth.program.novelty', growthProgram.noveltyScore],
      ['growth.program.ip', growthProgram.ip],
      ['growth.program.halted', growthProgram.halted],
      ['growth.program.executed', growthProgram.executed],
      ['growth.program.wait', growthProgram.waitRemaining],
      ['growth.program.backJumps', growthProgram.backwardsJumpsInWindow],
      ['growth.program.regs', JSON.stringify(growthProgram.regs || [])],
      ['growth.program.ops', JSON.stringify(growthProgram.opCounts || {})],

      ['@@Topology'],
      ['topology.version', topo.nnTopologyVersion],
      ['topology.rlResets', topo.rlTopologyResets],

      ['@@Reproduction Suppression'],
      ['reproSupp.density', repro.density],
      ['reproSupp.resources', repro.resources],
      ['reproSupp.fertilityRoll', repro.fertilityRoll],
      ['reproSupp.dye', repro.dye],
      ['reproSupp.resourceDebits', repro.resourceDebits],

      ['@@Energy Cost Breakdown'],
      ['cost.base', costs.base],
      ['cost.emitter', costs.emitter],
      ['cost.eater', costs.eater],
      ['cost.predator', costs.predator],
      ['cost.neuron', costs.neuron],
      ['cost.swimmer', costs.swimmer],
      ['cost.photosynthetic', costs.photosynthetic],
      ['cost.grabbing', costs.grabbing],
      ['cost.eye', costs.eye],
      ['cost.jet', costs.jet],
      ['cost.attractor', costs.attractor],
      ['cost.repulsor', costs.repulsor]
    ];

    setKV(creatureStatsEl, creatureStatsMode === 'compact' ? compactCreatureEntries(creatureEntries) : creatureEntries);

    setDetailText(creaturePointDetailsEl, buildPointDetailsText(selected));
    setDetailText(creatureSpringDetailsEl, buildSpringDetailsText(selected));
  }

  updateFollowButton();
}

function drawSnapshot(snap, { includePanels = true } = {}) {
  const renderStartMs = performance.now();
  const rect = canvas.getBoundingClientRect();

  const worldW = Number(snap?.world?.width) || 1;
  const worldH = Number(snap?.world?.height) || 1;

  const worldId = currentWorldId || snap?.id || 'w0';
  const cam = getCamera(worldId);

  if (!cam.initialized || cam.worldW !== worldW || cam.worldH !== worldH) {
    fitCameraToWorld(worldId, worldW, worldH);
  }

  const followId = followByWorld.get(worldId) || null;
  if (followId !== null) {
    const target = (snap.creatures || []).find((c) => c && String(c.id) === String(followId));
    if (target?.center) {
      // Following should be inspectable, not a tiny dot at fit-world zoom.
      const MIN_FOLLOW_ZOOM = 0.25;
      if (cam.zoom < MIN_FOLLOW_ZOOM) {
        cam.zoom = MIN_FOLLOW_ZOOM;
      }

      const cx = Number(target.center.x) || 0;
      const cy = Number(target.center.y) || 0;
      cam.offsetX = (rect.width / 2) - cx * cam.zoom;
      cam.offsetY = (rect.height / 2) - cy * cam.zoom;
    }
  }

  ctx.clearRect(0, 0, rect.width, rect.height);

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.strokeRect(cam.offsetX, cam.offsetY, worldW * cam.zoom, worldH * cam.zoom);

  const worldToScreenX = (x) => cam.offsetX + x * cam.zoom;
  const worldToScreenY = (y) => cam.offsetY + y * cam.zoom;

  const viewMinX = (-cam.offsetX) / cam.zoom;
  const viewMaxX = (rect.width - cam.offsetX) / cam.zoom;
  const viewMinY = (-cam.offsetY) / cam.zoom;
  const viewMaxY = (rect.height - cam.offsetY) / cam.zoom;
  const cullMargin = 250 / Math.max(cam.zoom, 0.05);

  // Dense fluid path
  if (snap.fluidDense && snap.fluidDense.rgbaBase64 && Number(snap.fluidDense.gridSize) > 0) {
    const N = Number(snap.fluidDense.gridSize);
    try {
      const bin = atob(snap.fluidDense.rgbaBase64);
      const bytes = new Uint8ClampedArray(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

      const img = new ImageData(bytes, N, N);
      const off = document.createElement('canvas');
      off.width = N;
      off.height = N;
      const offCtx = off.getContext('2d');
      offCtx.putImageData(img, 0, 0);

      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.globalAlpha = 0.95;
      ctx.drawImage(off, cam.offsetX, cam.offsetY, worldW * cam.zoom, worldH * cam.zoom);
      ctx.restore();
    } catch {
      // ignore; sparse fallback below
    }
  } else {
    const fluid = snap.fluid;
    const cellW = Number(fluid?.worldCell?.width) || 0;
    const cellH = Number(fluid?.worldCell?.height) || 0;
    if (fluid && Array.isArray(fluid.cells) && cellW > 0 && cellH > 0) {
      for (const cell of fluid.cells) {
        const r = Number(cell.r) || 0;
        const g = Number(cell.g) || 0;
        const b = Number(cell.b) || 0;
        const dye = Number(cell.dye) || (r + g + b);
        const speed = Number(cell.speed) || 0;

        const baseAlpha = (dye / 255) * 0.35 + Math.min(0.15, speed * 0.04);
        const zoomOutBoost = Math.max(1, 0.25 / Math.max(0.0001, cam.zoom));
        const alpha = Math.max(0.06, Math.min(0.85, baseAlpha * zoomOutBoost));
        ctx.fillStyle = `rgba(${Math.min(255, r)},${Math.min(255, g)},${Math.min(255, b)},${alpha})`;

        const x = Number(cell.x);
        const y = Number(cell.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

        ctx.fillRect(
          worldToScreenX(x - cellW * 0.5),
          worldToScreenY(y - cellH * 0.5),
          cellW * cam.zoom,
          cellH * cam.zoom
        );
      }
    }
  }

  if (showFluidVelocityArrows) {
    const fluid = snap.fluid;
    const cellW = Number(fluid?.worldCell?.width) || 0;
    const cellH = Number(fluid?.worldCell?.height) || 0;
    if (fluid && Array.isArray(fluid.cells) && cellW > 0 && cellH > 0) {
      const minSpeed = 0.01;
      const minArrowPx = 6;
      const maxArrowPx = 22;
      const speedToPx = 4.5;

      ctx.save();
      ctx.strokeStyle = 'rgba(170,210,255,0.85)';
      ctx.lineWidth = 1.25;

      for (const cell of fluid.cells) {
        const x = Number(cell.x);
        const y = Number(cell.y);
        const vx = Number(cell.vx) || 0;
        const vy = Number(cell.vy) || 0;
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

        const speed = Math.hypot(vx, vy);
        if (!Number.isFinite(speed) || speed < minSpeed) continue;

        const ux = vx / speed;
        const uy = vy / speed;
        const lenPx = Math.max(minArrowPx, Math.min(maxArrowPx, minArrowPx + speed * speedToPx));

        const sx = worldToScreenX(x);
        const sy = worldToScreenY(y);
        const ex = sx + ux * lenPx;
        const ey = sy + uy * lenPx;

        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        const head = Math.max(3, Math.min(6, lenPx * 0.28));
        const a = Math.atan2(ey - sy, ex - sx);
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - head * Math.cos(a - Math.PI / 7), ey - head * Math.sin(a - Math.PI / 7));
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - head * Math.cos(a + Math.PI / 7), ey - head * Math.sin(a + Math.PI / 7));
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  if (showFluidActiveTilesOverlay) {
    const fluid = snap.fluid;
    const debug = fluid?.activeTileDebug;
    const cellW = Number(fluid?.worldCell?.width) || 0;
    const cellH = Number(fluid?.worldCell?.height) || 0;
    const tileSizeCells = Math.max(1, Math.floor(Number(debug?.tileSizeCells) || 1));
    if (debug && Array.isArray(debug.cells) && cellW > 0 && cellH > 0) {
      const tileW = cellW * tileSizeCells;
      const tileH = cellH * tileSizeCells;
      for (const t of debug.cells) {
        const tx = Number(t.tx);
        const ty = Number(t.ty);
        if (!Number.isFinite(tx) || !Number.isFinite(ty)) continue;
        const x = tx * tileW;
        const y = ty * tileH;
        ctx.fillStyle = t.kind === 'momentumOnly' ? 'rgba(80,200,255,0.14)' : 'rgba(255,165,0,0.10)';
        ctx.fillRect(worldToScreenX(x), worldToScreenY(y), tileW * cam.zoom, tileH * cam.zoom);
      }
    }
  }

  if (Array.isArray(snap.particles) && snap.particles.length) {
    for (const p of snap.particles) {
      const x = Number(p?.x);
      const y = Number(p?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < viewMinX - cullMargin || x > viewMaxX + cullMargin || y < viewMinY - cullMargin || y > viewMaxY + cullMargin) continue;
      const life = Math.max(0, Math.min(1, Number(p?.life) || 0));
      const size = Math.max(0.5, Math.min(4.0, Number(p?.size) || 1));
      const r = Math.max(0.6, size * cam.zoom);
      ctx.fillStyle = `rgba(220,220,250,${0.08 + life * 0.35})`;
      ctx.beginPath();
      ctx.arc(worldToScreenX(x), worldToScreenY(y), Math.min(r, 4.5), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const selectedId = selectionByWorld.get(worldId) || null;
  const totalCreatures = Array.isArray(snap.creatures) ? snap.creatures.length : 0;
  let drawnCreatures = 0;

  for (const c of snap.creatures || []) {
    const verts = c.vertices || [];
    const springs = c.springs || [];
    const isSelected = selectedId !== null && String(c.id) === String(selectedId);

    if (!isSelected && c?.center) {
      const cx = Number(c.center.x);
      const cy = Number(c.center.y);
      if (Number.isFinite(cx) && Number.isFinite(cy)) {
        if (cx < viewMinX - cullMargin || cx > viewMaxX + cullMargin || cy < viewMinY - cullMargin || cy > viewMaxY + cullMargin) {
          continue;
        }
      }
    }
    drawnCreatures += 1;

    ctx.lineWidth = isSelected ? 1.5 : 1;
    ctx.strokeStyle = isSelected ? 'rgba(220,220,240,0.35)' : 'rgba(220,220,240,0.18)';
    ctx.beginPath();
    for (const s of springs) {
      if (s.isRigid) continue;
      const a = verts[s.a];
      const b = verts[s.b];
      if (!a || !b) continue;
      ctx.moveTo(worldToScreenX(a.x), worldToScreenY(a.y));
      ctx.lineTo(worldToScreenX(b.x), worldToScreenY(b.y));
    }
    ctx.stroke();

    ctx.lineWidth = isSelected ? 2.0 : 1.5;
    ctx.strokeStyle = isSelected ? 'rgba(255, 230, 90, 0.65)' : 'rgba(255, 230, 90, 0.40)';
    ctx.beginPath();
    for (const s of springs) {
      if (!s.isRigid) continue;
      const a = verts[s.a];
      const b = verts[s.b];
      if (!a || !b) continue;
      ctx.moveTo(worldToScreenX(a.x), worldToScreenY(a.y));
      ctx.lineTo(worldToScreenX(b.x), worldToScreenY(b.y));
    }
    ctx.stroke();

    for (const v of verts) {
      const r = Math.max(1.35, (v.radius || 2) * cam.zoom);
      ctx.fillStyle = colorForVertex(v);
      ctx.beginPath();
      ctx.arc(worldToScreenX(v.x), worldToScreenY(v.y), Math.min(r, 8), 0, Math.PI * 2);
      ctx.fill();

      if (v.isDesignatedEye) {
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(worldToScreenX(v.x), worldToScreenY(v.y), Math.min(r + 2, 10), 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    if (isSelected && c.center) {
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.beginPath();
      ctx.arc(worldToScreenX(c.center.x), worldToScreenY(c.center.y), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const renderMs = performance.now() - renderStartMs;
  const prevPerf = renderPerfByWorld.get(worldId) || { avgRenderMs: renderMs };
  const avgRenderMs = prevPerf.avgRenderMs * 0.85 + renderMs * 0.15;
  renderPerfByWorld.set(worldId, {
    lastRenderMs: renderMs,
    avgRenderMs,
    drawnCreatures,
    totalCreatures,
    atTick: Number(snap.tick) || 0
  });

  if (includePanels) {
    renderPanels(worldId);
  }
}

async function refreshScenarios() {
  const data = await apiGet('/api/scenarios');
  scenarioSelect.innerHTML = '';
  for (const s of data.scenarios || []) {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = s.name;
    scenarioSelect.appendChild(opt);
  }
}

async function refreshWorlds() {
  const data = await apiGet('/api/worlds');
  worldSelect.innerHTML = '';
  for (const w of data.worlds || []) {
    const opt = document.createElement('option');
    opt.value = w.id;
    opt.textContent = `${w.id} (${w.scenario})${w.paused ? ' [paused]' : ''}${w.crashed ? ' [crashed]' : ''}`;
    worldSelect.appendChild(opt);
  }

  if (!currentWorldId) {
    currentWorldId = data.defaultWorldId || (data.worlds?.[0]?.id ?? null);
  }

  if (currentWorldId) {
    worldSelect.value = currentWorldId;
  }

  for (const w of data.worlds || []) {
    if (!scrubTickByWorld.has(w.id)) scrubTickByWorld.set(w.id, null);
    if (!frameBufferByWorld.has(w.id)) {
      frameBufferByWorld.set(w.id, {
        available: 0,
        max: 0,
        stepStride: 1,
        latestSeq: null,
        latestTick: null,
        archiveFrames: 0,
        archiveOldestTick: null,
        archiveLatestTick: null
      });
    }
  }

  return data;
}

function connectStream({ worldId, mode = 'render', hz = 2 } = {}) {
  if (ws) {
    try { ws.close(); } catch {}
    ws = null;
  }

  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${window.location.host}/ws?world=${encodeURIComponent(worldId)}&mode=${encodeURIComponent(mode)}&hz=${encodeURIComponent(hz)}`;
  ws = new WebSocket(url);

  ws.addEventListener('open', () => {
    wsConnected = true;
    lastWsMessageAt = Date.now();
    statusEl.textContent = `stream connected (world=${worldId})…`;
  });

  ws.addEventListener('message', (ev) => {
    lastWsMessageAt = Date.now();
    try {
      const msg = JSON.parse(ev.data);
      if (msg.kind === 'error') {
        statusEl.textContent = `error: ${msg.error}`;
        return;
      }

      if (msg.kind === 'status') {
        const s = msg.data;
        lastStatusByWorld.set(worldId, s);

        const fb = s?.frameBuffer || {};
        frameBufferByWorld.set(worldId, {
          available: Number(fb.available) || 0,
          max: Number(fb.max) || 0,
          stepStride: Math.max(1, Number(fb.stepStride) || 1),
          latestSeq: Number.isFinite(Number(fb.latestSeq)) ? Number(fb.latestSeq) : null,
          latestTick: Number.isFinite(Number(fb.latestTick)) ? Number(fb.latestTick) : null,
          archiveFrames: Number(fb.archiveFrames) || 0,
          archiveOldestTick: Number.isFinite(Number(fb.archiveOldestTick)) ? Number(fb.archiveOldestTick) : null,
          archiveLatestTick: Number.isFinite(Number(fb.archiveLatestTick)) ? Number(fb.archiveLatestTick) : null
        });
        if (currentWorldId === worldId) {
          updateHistoryUi(worldId);
        }

        const cam = currentWorldId ? getCamera(currentWorldId) : null;
        const zoomLabel = cam ? ` zoom=${fmt(cam.zoom, 3)}` : '';
        const scrubTick = getScrubTick(worldId);
        const scrubLabel = scrubTick === null ? ' view=live' : ` view=tick:${scrubTick}`;
        if (runModeSelect && s?.runMode) {
          runModeSelect.value = String(s.runMode);
        }
        const renderPerf = renderPerfByWorld.get(worldId) || null;
        const renderLabel = renderPerf
          ? ` renderMs=${fmt(renderPerf.lastRenderMs, 2)} avg=${fmt(renderPerf.avgRenderMs, 2)} draw=${renderPerf.drawnCreatures}/${renderPerf.totalCreatures}`
          : '';
        statusEl.textContent = `world=${s.id} scenario=${s.scenario} seed=${s.seed} mode=${s.runMode || 'realtime'} tick=${s.tick} t=${fmt(s.time, 2)} dt=${fmt(s.dt, 5)} paused=${s.paused} sps=${s.stepsPerSecond} stepWallMs=${s.lastStepWallMs}${renderLabel}${zoomLabel}${scrubLabel}`;
      } else if (msg.kind === 'snapshot') {
        const snap = msg.data;

        // keep selection sane
        const sel = selectionByWorld.get(worldId);
        if (sel !== undefined && sel !== null) {
          const exists = (snap.creatures || []).some((c) => String(c.id) === String(sel));
          if (!exists) {
            selectionByWorld.delete(worldId);
            if (String(followByWorld.get(worldId) || '') === String(sel)) {
              followByWorld.delete(worldId);
            }
          }
        }

        const scrubTick = getScrubTick(worldId);
        if (scrubTick === null) {
          lastSnapshotByWorld.set(worldId, snap);
          drawSnapshot(snap);
        }
      }
    } catch {
      // ignore malformed
    }
  });

  ws.addEventListener('close', () => {
    wsConnected = false;
    statusEl.textContent = 'stream disconnected (retrying in 1s)…';
    setTimeout(() => {
      if (currentWorldId) connectStream({ worldId: currentWorldId, mode, hz });
    }, 1000);
  });

  ws.addEventListener('error', () => {
    wsConnected = false;
    try { ws.close(); } catch {}
  });

  return ws;
}

function cycleCreature(delta) {
  if (!currentWorldId) return;
  const snap = lastSnapshotByWorld.get(currentWorldId);
  if (!snap) return;

  const list = sortedCreatures(snap);
  if (!list.length) return;

  const cur = selectionByWorld.get(currentWorldId);
  const ids = list.map((c) => c.id);
  let idx = ids.findIndex((id) => String(id) === String(cur));
  if (idx < 0) idx = 0;
  else idx = (idx + delta + ids.length) % ids.length;

  const nextId = ids[idx];
  selectionByWorld.set(currentWorldId, nextId);

  if (followByWorld.has(currentWorldId)) {
    followByWorld.set(currentWorldId, nextId);
  }

  scheduleRender({ includePanels: true });
}

function toggleFollowSelected() {
  if (!currentWorldId) return;
  const sel = selectionByWorld.get(currentWorldId);
  if (sel === undefined || sel === null) return;

  const cur = followByWorld.get(currentWorldId);
  const on = String(cur || '') === String(sel);
  if (on) followByWorld.delete(currentWorldId);
  else followByWorld.set(currentWorldId, sel);

  updateFollowButton();
}

historySliderEl?.addEventListener('input', () => {
  if (!currentWorldId) return;
  const meta = getFrameBufferMeta(currentWorldId);
  const oldestTick = Number.isFinite(Number(meta.archiveOldestTick)) ? Math.max(0, Math.floor(Number(meta.archiveOldestTick))) : 0;
  const latestTick = Number.isFinite(Number(meta.archiveLatestTick)) ? Math.max(oldestTick, Math.floor(Number(meta.archiveLatestTick))) : oldestTick;
  const archiveFrames = Math.max(1, Math.floor(Number(meta.archiveFrames) || 1));
  const frameIdx = Math.max(0, Math.min(archiveFrames - 1, Math.floor(Number(historySliderEl.value) || 0)));
  const stepStride = Math.max(1, Math.floor(Number(meta.stepStride) || 1));

  const nextTick = archiveFrames <= 1
    ? latestTick
    : Math.max(oldestTick, Math.min(latestTick, oldestTick + (frameIdx * stepStride)));

  setScrubTick(currentWorldId, nextTick);
});

historyLiveBtn?.addEventListener('click', () => {
  if (!currentWorldId) return;
  setScrubTick(currentWorldId, null);
});

worldSelect.addEventListener('change', async () => {
  currentWorldId = worldSelect.value;
  updateHistoryUi(currentWorldId);
  await refreshConfig(currentWorldId);
  connectStream({ worldId: currentWorldId, mode: 'render', hz: 10 });
});

newWorldBtn.addEventListener('click', async () => {
  const scenario = scenarioSelect.value;
  const seed = Number(seedInput.value || 0) >>> 0;
  const out = await apiPost('/api/worlds', { scenario, seed });
  currentWorldId = out.id;
  scrubTickByWorld.set(currentWorldId, null);
  await refreshWorlds();
  updateHistoryUi(currentWorldId);
  await refreshConfig(currentWorldId);
  connectStream({ worldId: currentWorldId, mode: 'render', hz: 10 });
});

setScenarioBtn.addEventListener('click', async () => {
  const name = scenarioSelect.value;
  const seed = Number(seedInput.value || 0) >>> 0;
  if (!currentWorldId) return;
  await apiPost(`/api/worlds/${encodeURIComponent(currentWorldId)}/control/setScenario`, { name, seed });
  selectionByWorld.delete(currentWorldId);
  followByWorld.delete(currentWorldId);
  scrubTickByWorld.set(currentWorldId, null);
  frameBufferByWorld.set(currentWorldId, {
    available: 0,
    max: 0,
    stepStride: 1,
    latestSeq: null,
    latestTick: null,
    archiveFrames: 0,
    archiveOldestTick: null,
    archiveLatestTick: null
  });
  updateHistoryUi(currentWorldId);
  await refreshConfig(currentWorldId);
});

pauseBtn.addEventListener('click', async () => {
  if (!currentWorldId) return;
  await apiPost(`/api/worlds/${encodeURIComponent(currentWorldId)}/control/pause`);
  await refreshWorlds();
});

resumeBtn.addEventListener('click', async () => {
  if (!currentWorldId) return;
  await apiPost(`/api/worlds/${encodeURIComponent(currentWorldId)}/control/resume`);
  await refreshWorlds();
});

setRunModeBtn?.addEventListener('click', async () => {
  if (!currentWorldId) return;
  const mode = String(runModeSelect?.value || 'realtime');
  await apiPost(`/api/worlds/${encodeURIComponent(currentWorldId)}/control/runMode`, { mode });
  await refreshWorlds();
});

fitBtn.addEventListener('click', () => {
  if (!currentWorldId) return;
  const cam = getCamera(currentWorldId);
  fitCameraToWorld(currentWorldId, cam.worldW || 1, cam.worldH || 1);
  scheduleRender({ includePanels: false });
});

if (showFluidVelocityToggle) {
  showFluidVelocityToggle.checked = showFluidVelocityArrows;
  showFluidVelocityToggle.addEventListener('change', () => {
    showFluidVelocityArrows = !!showFluidVelocityToggle.checked;
    scheduleRender({ includePanels: false });
  });
}

if (showFluidActiveTilesToggle) {
  showFluidActiveTilesToggle.checked = showFluidActiveTilesOverlay;
  showFluidActiveTilesToggle.addEventListener('change', () => {
    showFluidActiveTilesOverlay = !!showFluidActiveTilesToggle.checked;
    scheduleRender({ includePanels: false });
  });
}

prevCreatureBtn.addEventListener('click', () => cycleCreature(-1));
nextCreatureBtn.addEventListener('click', () => cycleCreature(1));
toggleFollowBtn.addEventListener('click', () => toggleFollowSelected());
creatureStatsModeBtn?.addEventListener('click', () => {
  creatureStatsMode = creatureStatsMode === 'compact' ? 'expanded' : 'compact';
  syncCreatureStatsModeUI();
  scheduleRender({ includePanels: true });
});

refreshConfigBtn.addEventListener('click', async () => {
  if (!currentWorldId) return;
  await refreshConfig(currentWorldId);
});

applyConfigBtn.addEventListener('click', async () => {
  if (!currentWorldId) return;
  await applyConfig(currentWorldId);
});

window.addEventListener('keydown', (ev) => {
  if (!currentWorldId) return;
  if (ev.key.toLowerCase() === 'f') {
    toggleFollowSelected();
    ev.preventDefault();
  }
  if (ev.key === 'ArrowLeft') {
    cycleCreature(-1);
    ev.preventDefault();
  }
  if (ev.key === 'ArrowRight') {
    cycleCreature(1);
    ev.preventDefault();
  }
  if (ev.key === 'Escape') {
    selectionByWorld.delete(currentWorldId);
    followByWorld.delete(currentWorldId);
    scheduleRender({ includePanels: true });
  }
});

let isDragging = false;
let lastDragX = 0;
let lastDragY = 0;
let dragMovedPx = 0;

canvas.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0) return;
  if (!currentWorldId) return;

  const p = eventToCanvasXY(ev);
  isDragging = true;
  dragMovedPx = 0;
  lastDragX = p.x;
  lastDragY = p.y;
  canvas.setPointerCapture(ev.pointerId);
});

canvas.addEventListener('pointermove', (ev) => {
  if (!isDragging) return;
  if (!currentWorldId) return;

  const p = eventToCanvasXY(ev);
  const cam = getCamera(currentWorldId);
  const dx = p.x - lastDragX;
  const dy = p.y - lastDragY;
  dragMovedPx += Math.sqrt(dx * dx + dy * dy);

  cam.offsetX += dx;
  cam.offsetY += dy;

  lastDragX = p.x;
  lastDragY = p.y;

  scheduleRender({ includePanels: false });
});

function selectCreatureAtCanvasPoint(canvasPoint) {
  if (!currentWorldId) return;
  const snap = lastSnapshotByWorld.get(currentWorldId);
  if (!snap) return;

  const cam = getCamera(currentWorldId);
  const w = screenToWorld(cam, canvasPoint.x, canvasPoint.y);

  let best = null;
  let bestD2 = Infinity;

  for (const c of snap.creatures || []) {
    if (!c?.center) continue;
    const dx = (Number(c.center.x) || 0) - w.x;
    const dy = (Number(c.center.y) || 0) - w.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = c;
    }
  }

  const pickRadiusWorld = Math.max(25, 14 / Math.max(0.0005, cam.zoom));
  if (best && bestD2 <= pickRadiusWorld * pickRadiusWorld) {
    selectionByWorld.set(currentWorldId, best.id);
  } else {
    selectionByWorld.delete(currentWorldId);
    followByWorld.delete(currentWorldId);
  }

  scheduleRender({ includePanels: true });
}

function endDrag(ev) {
  if (!isDragging) return;
  isDragging = false;
  try { canvas.releasePointerCapture(ev.pointerId); } catch {}

  if (dragMovedPx <= 4) {
    selectCreatureAtCanvasPoint(eventToCanvasXY(ev));
  }
}

canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

canvas.addEventListener('dblclick', () => {
  if (!currentWorldId) return;
  const cam = getCamera(currentWorldId);
  fitCameraToWorld(currentWorldId, cam.worldW || 1, cam.worldH || 1);
  scheduleRender({ includePanels: false });
});

canvas.addEventListener('wheel', (ev) => {
  if (!currentWorldId) return;
  ev.preventDefault();

  const cam = getCamera(currentWorldId);
  const p = eventToCanvasXY(ev);

  const zoomFactor = Math.exp(-ev.deltaY * 0.0015);
  zoomAround(cam, p.x, p.y, zoomFactor);
  scheduleRender({ includePanels: false });
}, { passive: false });

syncCreatureStatsModeUI();

await refreshScenarios();
await refreshWorlds();
if (currentWorldId) {
  if (!scrubTickByWorld.has(currentWorldId)) scrubTickByWorld.set(currentWorldId, null);
  updateHistoryUi(currentWorldId);
  await refreshConfig(currentWorldId);
  connectStream({ worldId: currentWorldId, mode: 'render', hz: 10 });
}

setInterval(async () => {
  if (!currentWorldId) return;

  // Fallback when WS is blocked/stale: keep UI alive via HTTP polling.
  const wsStale = !wsConnected || (Date.now() - lastWsMessageAt) > 2500;
  if (!wsStale) return;

  try {
    const status = await apiGet(`/api/worlds/${encodeURIComponent(currentWorldId)}/status`);
    const scrubTick = getScrubTick(currentWorldId);
    const snap = scrubTick === null
      ? await apiGet(`/api/worlds/${encodeURIComponent(currentWorldId)}/snapshot?mode=render`)
      : await apiGet(`/api/worlds/${encodeURIComponent(currentWorldId)}/frame/${encodeURIComponent(scrubTick)}`);

    lastStatusByWorld.set(currentWorldId, status);
    lastSnapshotByWorld.set(currentWorldId, snap);

    if (runModeSelect && status?.runMode) {
      runModeSelect.value = String(status.runMode);
    }

    const fb = status?.frameBuffer || {};
    frameBufferByWorld.set(currentWorldId, {
      available: Number(fb.available) || 0,
      max: Number(fb.max) || 0,
      stepStride: Math.max(1, Number(fb.stepStride) || 1),
      latestSeq: Number.isFinite(Number(fb.latestSeq)) ? Number(fb.latestSeq) : null,
      latestTick: Number.isFinite(Number(fb.latestTick)) ? Number(fb.latestTick) : null,
      archiveFrames: Number(fb.archiveFrames) || 0,
      archiveOldestTick: Number.isFinite(Number(fb.archiveOldestTick)) ? Number(fb.archiveOldestTick) : null,
      archiveLatestTick: Number.isFinite(Number(fb.archiveLatestTick)) ? Number(fb.archiveLatestTick) : null
    });

    updateHistoryUi(currentWorldId);
    drawSnapshot(snap);

    const cam = getCamera(currentWorldId);
    const zoomLabel = cam ? ` zoom=${fmt(cam.zoom, 3)}` : '';
    const activeScrubTick = getScrubTick(currentWorldId);
    const scrubLabel = activeScrubTick === null ? ' view=live' : ` view=tick:${activeScrubTick}`;
    const renderPerf = renderPerfByWorld.get(currentWorldId) || null;
    const renderLabel = renderPerf
      ? ` renderMs=${fmt(renderPerf.lastRenderMs, 2)} avg=${fmt(renderPerf.avgRenderMs, 2)} draw=${renderPerf.drawnCreatures}/${renderPerf.totalCreatures}`
      : '';
    statusEl.textContent = `world=${status.id} scenario=${status.scenario} seed=${status.seed} mode=${status.runMode || 'realtime'} tick=${status.tick} t=${fmt(status.time, 2)} dt=${fmt(status.dt, 5)} paused=${status.paused} sps=${status.stepsPerSecond} stepWallMs=${status.lastStepWallMs}${renderLabel}${zoomLabel}${scrubLabel} (http fallback)`;
  } catch {
    // keep quiet; WS reconnect loop + periodic world refresh continue.
  }
}, 1500);

setInterval(() => {
  refreshWorlds()
    .then(() => {
      if (currentWorldId) updateHistoryUi(currentWorldId);
      const tick = currentWorldId ? getScrubTick(currentWorldId) : null;
      if (currentWorldId && tick !== null) {
        loadScrubFrameByTick(currentWorldId, tick);
      }
    })
    .catch(() => {});
}, 4000);
