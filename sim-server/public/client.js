const statusEl = document.getElementById('status');
const telemetryEl = document.getElementById('telemetry');
const canvas = document.getElementById('canvas');

const worldSelect = document.getElementById('worldSelect');
const newWorldBtn = document.getElementById('newWorldBtn');

const scenarioSelect = document.getElementById('scenarioSelect');
const seedInput = document.getElementById('seedInput');
const setScenarioBtn = document.getElementById('setScenarioBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resumeBtn = document.getElementById('resumeBtn');
const fitBtn = document.getElementById('fitBtn');

const ctx = canvas.getContext('2d');

let ws = null;
let currentWorldId = null;

const cameraByWorld = new Map();

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

function fmt(n, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return '—';
  return x.toFixed(digits);
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

function colorForVertex(v) {
  const name = v?.nodeTypeName || null;
  if (name && NODE_TYPE_COLORS[name]) return NODE_TYPE_COLORS[name];
  return 'rgba(255,255,255,0.75)';
}

const selectionByWorld = new Map();
const followByWorld = new Map();

function drawSnapshot(snap) {
  const rect = canvas.getBoundingClientRect();

  const worldW = Number(snap?.world?.width) || 1;
  const worldH = Number(snap?.world?.height) || 1;

  const worldId = currentWorldId || snap?.id || 'w0';
  const cam = getCamera(worldId);

  // auto-fit on first snapshot or world resize
  if (!cam.initialized || cam.worldW !== worldW || cam.worldH !== worldH) {
    fitCameraToWorld(worldId, worldW, worldH);
  }

  // follow camera (center selected creature)
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

  // world bounds
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.strokeRect(cam.offsetX, cam.offsetY, worldW * cam.zoom, worldH * cam.zoom);

  const worldToScreenX = (x) => cam.offsetX + x * cam.zoom;
  const worldToScreenY = (y) => cam.offsetY + y * cam.zoom;

  // Fluid
  const fluid = snap.fluid;

  // Prefer dense grid when available (highest fidelity)
  if (snap.fluidDense && snap.fluidDense.rgbaBase64 && Number(snap.fluidDense.gridSize) > 0) {
    const N = Number(snap.fluidDense.gridSize) || 0;
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
      // fall back to sparse
    }
  } else {
    // Sparse cell list — NOTE: server snapshot uses worldCell.width/height (not cellSize)
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

  const selectedId = selectionByWorld.get(worldId) || null;

  // Particles (draw after fluid, before creatures)
  if (Array.isArray(snap.particles) && snap.particles.length) {
    ctx.fillStyle = 'rgba(220,220,250,0.45)';
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

  for (const c of snap.creatures || []) {
    const verts = c.vertices || [];
    const springs = c.springs || [];

    const isSelected = selectedId !== null && String(c.id) === String(selectedId);

    // non-rigid springs
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

    // rigid springs
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

    // points
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
      ctx.fillStyle = 'rgba(255,255,255,0.65)';
      ctx.beginPath();
      ctx.arc(worldToScreenX(c.center.x), worldToScreenY(c.center.y), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // telemetry panel: world + fluid + selected creature
  const selected = selectedId !== null
    ? (snap.creatures || []).find((c) => c && String(c.id) === String(selectedId))
    : null;

  telemetryEl.textContent = JSON.stringify({
    world: worldId,
    scenario: snap.scenario,
    tick: snap.tick,
    time: snap.time,
    populations: snap.populations,
    camera: { zoom: cam.zoom, offsetX: cam.offsetX, offsetY: cam.offsetY },
    fluid: fluid ? {
      gridSize: fluid.gridSize,
      activeCells: fluid.activeCells,
      scannedCells: fluid.scannedCells,
      maxDye: fluid.maxDye,
      maxSpeed: fluid.maxSpeed
    } : null,
    fluidDense: snap.fluidDense ? { gridSize: snap.fluidDense.gridSize } : null,
    particles: Array.isArray(snap.particles) ? { count: snap.particles.length } : null,
    selectedCreature: selected ? {
      id: selected.id,
      energy: selected.energy,
      center: selected.center,
      nodeTypeCounts: selected.nodeTypeCounts,
      actuationTelemetry: selected.actuationTelemetry
    } : null,
    following: followId
  }, null, 2);
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
        const cam = currentWorldId ? getCamera(currentWorldId) : null;
        const zoomLabel = cam ? ` zoom=${fmt(cam.zoom, 3)}` : '';
        statusEl.textContent = `world=${s.id} scenario=${s.scenario} seed=${s.seed} tick=${s.tick} t=${fmt(s.time, 2)} dt=${fmt(s.dt, 5)} paused=${s.paused} sps=${s.stepsPerSecond} stepWallMs=${s.lastStepWallMs}${zoomLabel}`;
      } else if (msg.kind === 'snapshot') {
        const snap = msg.data;
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

worldSelect.addEventListener('change', async () => {
  currentWorldId = worldSelect.value;
  connectStream({ worldId: currentWorldId, mode: 'renderFull', hz: 10 });
});

newWorldBtn.addEventListener('click', async () => {
  const scenario = scenarioSelect.value;
  const seed = Number(seedInput.value || 0) >>> 0;
  const out = await apiPost('/api/worlds', { scenario, seed });
  currentWorldId = out.id;
  await refreshWorlds();
  connectStream({ worldId: currentWorldId, mode: 'renderFull', hz: 10 });
});

setScenarioBtn.addEventListener('click', async () => {
  const name = scenarioSelect.value;
  const seed = Number(seedInput.value || 0) >>> 0;
  if (!currentWorldId) return;
  await apiPost(`/api/worlds/${encodeURIComponent(currentWorldId)}/control/setScenario`, { name, seed });
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

fitBtn.addEventListener('click', async () => {
  if (!currentWorldId) return;
  const cam = getCamera(currentWorldId);
  fitCameraToWorld(currentWorldId, cam.worldW || 1, cam.worldH || 1);
});

// Follow selected creature
window.addEventListener('keydown', (ev) => {
  if (!currentWorldId) return;
  if (ev.key.toLowerCase() === 'f') {
    const sel = selectionByWorld.get(currentWorldId);
    if (sel !== undefined && sel !== null) {
      const isFollowing = String(followByWorld.get(currentWorldId) || '') === String(sel);
      if (isFollowing) followByWorld.delete(currentWorldId);
      else followByWorld.set(currentWorldId, sel);
    }
  }
  if (ev.key === 'Escape') {
    selectionByWorld.delete(currentWorldId);
    followByWorld.delete(currentWorldId);
  }
});

// Pan + zoom controls
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

async function maybeSelectCreatureAt(ev) {
  if (!currentWorldId) return;
  if (dragMovedPx > 4) return; // treat as pan, not click

  // Query a lite snapshot for selection without stopping the stream.
  // (Could also cache the last render snapshot; keep it simple for now.)
  const snap = await apiGet(`/api/worlds/${encodeURIComponent(currentWorldId)}/snapshot?mode=render`);
  const cam = getCamera(currentWorldId);

  const p = eventToCanvasXY(ev);
  const w = screenToWorld(cam, p.x, p.y);

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

  // 120 world units default pick radius (scaled a bit by zoom)
  const pickRadius = 120;
  if (best && bestD2 <= pickRadius * pickRadius) {
    selectionByWorld.set(currentWorldId, best.id);
  } else {
    selectionByWorld.delete(currentWorldId);
    followByWorld.delete(currentWorldId);
  }
}

async function endDrag(ev) {
  if (!isDragging) return;
  isDragging = false;

  try { canvas.releasePointerCapture(ev.pointerId); } catch {}

  // click selects
  try {
    await maybeSelectCreatureAt(ev);
  } catch {
    // ignore
  }
}
canvas.addEventListener('pointerup', (ev) => { endDrag(ev); });
canvas.addEventListener('pointercancel', (ev) => { endDrag(ev); });

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

  // wheel delta is in screen pixels; use exponential scaling for smoothness.
  const zoomFactor = Math.exp(-ev.deltaY * 0.0015);
  zoomAround(cam, p.x, p.y, zoomFactor);
}, { passive: false });

await refreshScenarios();
await refreshWorlds();
if (currentWorldId) connectStream({ worldId: currentWorldId, mode: 'renderFull', hz: 10 });

setInterval(() => {
  // keep world labels fresh (paused/crashed)
  refreshWorlds().catch(() => {});
}, 2000);
