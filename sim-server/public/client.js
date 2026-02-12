const statusEl = document.getElementById('status');
const canvas = document.getElementById('canvas');

const worldSelect = document.getElementById('worldSelect');
const newWorldBtn = document.getElementById('newWorldBtn');

const scenarioSelect = document.getElementById('scenarioSelect');
const seedInput = document.getElementById('seedInput');
const setScenarioBtn = document.getElementById('setScenarioBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resumeBtn = document.getElementById('resumeBtn');
const fitBtn = document.getElementById('fitBtn');

const prevCreatureBtn = document.getElementById('prevCreatureBtn');
const nextCreatureBtn = document.getElementById('nextCreatureBtn');
const toggleFollowBtn = document.getElementById('toggleFollowBtn');

const refreshConfigBtn = document.getElementById('refreshConfigBtn');
const applyConfigBtn = document.getElementById('applyConfigBtn');

const worldStatsEl = document.getElementById('worldStats');
const creatureStatsEl = document.getElementById('creatureStats');
const creaturePointDetailsEl = document.getElementById('creaturePointDetails');
const creatureSpringDetailsEl = document.getElementById('creatureSpringDetails');
const configEditorEl = document.getElementById('configEditor');

const ctx = canvas.getContext('2d');

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
  'ENERGY_PER_PARTICLE',
  'EATER_NODE_ENERGY_COST',
  'PREDATOR_NODE_ENERGY_COST',
  'SWIMMER_NODE_ENERGY_COST',
  'JET_NODE_ENERGY_COST',
  'ATTRACTOR_NODE_ENERGY_COST',
  'REPULSOR_NODE_ENERGY_COST',
  'FLUID_CURRENT_STRENGTH_ON_BODY',
  'SOFT_BODY_PUSH_STRENGTH',
  'BODY_REPULSION_STRENGTH',
  'REPRO_RESOURCE_MIN_NUTRIENT',
  'REPRO_RESOURCE_MIN_LIGHT',
  'FAILED_REPRODUCTION_COOLDOWN_TICKS'
];

let ws = null;
let currentWorldId = null;

const cameraByWorld = new Map();
const selectionByWorld = new Map();
const followByWorld = new Map();
const lastSnapshotByWorld = new Map();
const lastStatusByWorld = new Map();
const configDraftByWorld = new Map();

function fmt(n, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return '—';
  return x.toFixed(digits);
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

  setKV(worldStatsEl, [
    ['world', worldId],
    ['scenario', snap.scenario],
    ['tick', snap.tick],
    ['time', snap.time],
    ['sps', status?.stepsPerSecond],
    ['dt', status?.dt],
    ['creatures', snap.populations?.creatures],
    ['particles', snap.populations?.particles],
    ['fluidActiveCells', snap.fluid?.activeCells],
    ['removedTotal', removedTotal],
    ['topRemovalReason', `${topReason} (${reasons[topReason] || 0})`],
    ['removedByPhysicsKind', JSON.stringify(snap?.instabilityTelemetry?.removedByPhysicsKind || {})],
    ['globalGain.photo', energyGains?.photosynthesis],
    ['globalGain.eat', energyGains?.eating],
    ['globalGain.pred', energyGains?.predation],
    ['globalCost.base', energyCosts?.baseNodes],
    ['globalCost.neuron', energyCosts?.neuronNodes],
    ['globalCost.eater', energyCosts?.eaterNodes],
    ['globalCost.predator', energyCosts?.predatorNodes],
    ['mutationStats', JSON.stringify(snap?.mutationStats || {})],
    ['zoom', cam.zoom]
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
    const repro = fs.reproductionSuppression || {};
    const costs = fs.energyCostsByType || {};
    const gains = fs.energyGains || {};
    const topo = fs.topology || {};
    const following = String(followByWorld.get(worldId) || '') === String(selected.id) ? 'yes' : 'no';

    setKV(creatureStatsEl, [
      ['id', selected.id],
      ['following', following],
      ['energy', selected.energy],
      ['currentMaxEnergy', fs.currentMaxEnergy],
      ['reproThreshold', fs.reproductionEnergyThreshold],
      ['ticksSinceBirth', fs.ticksSinceBirth],
      ['canReproduce', fs.canReproduce],
      ['rewardStrategy', fs.rewardStrategy],
      ['dye.hue', fs.dyePreferredHue],
      ['dye.tolerance', fs.dyeHueTolerance],
      ['dye.gain', fs.dyeResponseGain],
      ['dye.sign', fs.dyeResponseSign],
      ['dye.affinity', JSON.stringify(fs.dyeNodeTypeAffinity || {})],
      ['centerX', selected.center?.x],
      ['centerY', selected.center?.y],
      ['points', selected.vertices?.length],
      ['springs', selected.springs?.length],
      ['stiffness(avg)', fs.stiffness],
      ['damping(avg)', fs.damping],
      ['motorInterval', fs.motorImpulseInterval],
      ['motorCap', fs.motorImpulseMagnitudeCap],
      ['emitterStrength', fs.emitterStrength],
      ['emitterDir', `${asDisplayValue(fs?.emitterDirection?.x)}, ${asDisplayValue(fs?.emitterDirection?.y)}`],
      ['numOffspring', fs.numOffspring],
      ['offspringRadius', fs.offspringSpawnRadius],
      ['pointAddChance', fs.pointAddChance],
      ['springConnectionRadius', fs.springConnectionRadius],
      ['reproCooldownGene', fs.reproductionCooldownGene],
      ['effectiveReproCooldown', fs.effectiveReproductionCooldown],
      ['nodeTypes', JSON.stringify(selected.nodeTypeCounts || {})],
      ['actEvals', selected.actuationTelemetry?.evaluations],
      ['actSkips', selected.actuationTelemetry?.skips],
      ['actAvgInterval', selected.actuationTelemetry?.avgEffectiveInterval],
      ['gain.photo', gains.photosynthesis],
      ['gain.eat', gains.eating],
      ['gain.pred', gains.predation],
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
      ['topology.version', topo.nnTopologyVersion],
      ['topology.rlResets', topo.rlTopologyResets],
      ['reproSupp.density', repro.density],
      ['reproSupp.resources', repro.resources],
      ['reproSupp.fertilityRoll', repro.fertilityRoll],
      ['reproSupp.dye', repro.dye],
      ['reproSupp.resourceDebits', repro.resourceDebits],
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
    ]);

    setDetailText(creaturePointDetailsEl, buildPointDetailsText(selected));
    setDetailText(creatureSpringDetailsEl, buildSpringDetailsText(selected));
  }

  updateFollowButton();
}

function drawSnapshot(snap) {
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

        const alpha = Math.max(0.03, Math.min(0.35, (dye / 255) * 0.35 + Math.min(0.15, speed * 0.04)));
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

  if (Array.isArray(snap.particles) && snap.particles.length) {
    for (const p of snap.particles) {
      const x = Number(p?.x);
      const y = Number(p?.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
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

  for (const c of snap.creatures || []) {
    const verts = c.vertices || [];
    const springs = c.springs || [];
    const isSelected = selectedId !== null && String(c.id) === String(selectedId);

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
      const r = Math.max(0.75, (v.radius || 2) * cam.zoom);
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

  renderPanels(worldId);
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

  return data;
}

function connectStream({ worldId, mode = 'renderFull', hz = 10 } = {}) {
  if (ws) {
    try { ws.close(); } catch {}
    ws = null;
  }

  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${window.location.host}/ws?world=${encodeURIComponent(worldId)}&mode=${encodeURIComponent(mode)}&hz=${encodeURIComponent(hz)}`;
  ws = new WebSocket(url);

  ws.addEventListener('open', () => {
    statusEl.textContent = `stream connected (world=${worldId})…`;
  });

  ws.addEventListener('message', (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.kind === 'error') {
        statusEl.textContent = `error: ${msg.error}`;
        return;
      }

      if (msg.kind === 'status') {
        const s = msg.data;
        lastStatusByWorld.set(worldId, s);
        const cam = currentWorldId ? getCamera(currentWorldId) : null;
        const zoomLabel = cam ? ` zoom=${fmt(cam.zoom, 3)}` : '';
        statusEl.textContent = `world=${s.id} scenario=${s.scenario} seed=${s.seed} tick=${s.tick} t=${fmt(s.time, 2)} dt=${fmt(s.dt, 5)} paused=${s.paused} sps=${s.stepsPerSecond} stepWallMs=${s.lastStepWallMs}${zoomLabel}`;
      } else if (msg.kind === 'snapshot') {
        const snap = msg.data;
        lastSnapshotByWorld.set(worldId, snap);

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

        drawSnapshot(snap);
      }
    } catch {
      // ignore malformed
    }
  });

  ws.addEventListener('close', () => {
    statusEl.textContent = 'stream disconnected (retrying in 1s)…';
    setTimeout(() => {
      if (currentWorldId) connectStream({ worldId: currentWorldId, mode, hz });
    }, 1000);
  });

  ws.addEventListener('error', () => {
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

  renderPanels(currentWorldId);
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

worldSelect.addEventListener('change', async () => {
  currentWorldId = worldSelect.value;
  await refreshConfig(currentWorldId);
  connectStream({ worldId: currentWorldId, mode: 'renderFull', hz: 10 });
});

newWorldBtn.addEventListener('click', async () => {
  const scenario = scenarioSelect.value;
  const seed = Number(seedInput.value || 0) >>> 0;
  const out = await apiPost('/api/worlds', { scenario, seed });
  currentWorldId = out.id;
  await refreshWorlds();
  await refreshConfig(currentWorldId);
  connectStream({ worldId: currentWorldId, mode: 'renderFull', hz: 10 });
});

setScenarioBtn.addEventListener('click', async () => {
  const name = scenarioSelect.value;
  const seed = Number(seedInput.value || 0) >>> 0;
  if (!currentWorldId) return;
  await apiPost(`/api/worlds/${encodeURIComponent(currentWorldId)}/control/setScenario`, { name, seed });
  selectionByWorld.delete(currentWorldId);
  followByWorld.delete(currentWorldId);
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

fitBtn.addEventListener('click', () => {
  if (!currentWorldId) return;
  const cam = getCamera(currentWorldId);
  fitCameraToWorld(currentWorldId, cam.worldW || 1, cam.worldH || 1);
});

prevCreatureBtn.addEventListener('click', () => cycleCreature(-1));
nextCreatureBtn.addEventListener('click', () => cycleCreature(1));
toggleFollowBtn.addEventListener('click', () => toggleFollowSelected());

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
    renderPanels(currentWorldId);
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

  renderPanels(currentWorldId);
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
});

canvas.addEventListener('wheel', (ev) => {
  if (!currentWorldId) return;
  ev.preventDefault();

  const cam = getCamera(currentWorldId);
  const p = eventToCanvasXY(ev);

  const zoomFactor = Math.exp(-ev.deltaY * 0.0015);
  zoomAround(cam, p.x, p.y, zoomFactor);
}, { passive: false });

await refreshScenarios();
await refreshWorlds();
if (currentWorldId) {
  await refreshConfig(currentWorldId);
  connectStream({ worldId: currentWorldId, mode: 'renderFull', hz: 10 });
}

setInterval(() => {
  refreshWorlds().catch(() => {});
}, 4000);
