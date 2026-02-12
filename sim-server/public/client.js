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

  ctx.clearRect(0, 0, rect.width, rect.height);

  // world bounds
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.strokeRect(cam.offsetX, cam.offsetY, worldW * cam.zoom, worldH * cam.zoom);

  const worldToScreenX = (x) => cam.offsetX + x * cam.zoom;
  const worldToScreenY = (y) => cam.offsetY + y * cam.zoom;

  // Fluid (sparse cell list)
  const fluid = snap.fluid;
  if (fluid && Array.isArray(fluid.cells) && Number.isFinite(fluid.cellSize) && fluid.cellSize > 0) {
    const cs = fluid.cellSize;
    for (const cell of fluid.cells) {
      const gx = cell.gx;
      const gy = cell.gy;
      const rgb = cell.rgb || [80, 80, 110];
      const alpha = Math.max(0.06, Math.min(0.45, (cell.density || 0) / 255));
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
      ctx.fillRect(
        worldToScreenX(gx * cs),
        worldToScreenY(gy * cs),
        cs * cam.zoom,
        cs * cam.zoom
      );
    }
  }

  for (const c of snap.creatures || []) {
    const verts = c.vertices || [];
    const springs = c.springs || [];

    // non-rigid springs
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(220,220,240,0.18)';
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
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 230, 90, 0.40)';
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

  return data;
}

function connectStream({ worldId, mode = 'render', hz = 10 } = {}) {
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
        telemetryEl.textContent = JSON.stringify({
          world: snap.id,
          scenario: snap.scenario,
          tick: snap.tick,
          time: snap.time,
          populations: snap.populations,
          creatures: (snap.creatures || []).length,
          fluidCells: snap.fluid?.cells?.length ?? 0
        }, null, 2);
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
  connectStream({ worldId: currentWorldId, mode: 'render', hz: 10 });
});

newWorldBtn.addEventListener('click', async () => {
  const scenario = scenarioSelect.value;
  const seed = Number(seedInput.value || 0) >>> 0;
  const out = await apiPost('/api/worlds', { scenario, seed });
  currentWorldId = out.id;
  await refreshWorlds();
  connectStream({ worldId: currentWorldId, mode: 'render', hz: 10 });
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

// Pan + zoom controls
let isDragging = false;
let lastDragX = 0;
let lastDragY = 0;

canvas.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0) return;
  if (!currentWorldId) return;
  isDragging = true;
  lastDragX = ev.clientX;
  lastDragY = ev.clientY;
  canvas.setPointerCapture(ev.pointerId);
});

canvas.addEventListener('pointermove', (ev) => {
  if (!isDragging) return;
  if (!currentWorldId) return;
  const cam = getCamera(currentWorldId);
  const dx = ev.clientX - lastDragX;
  const dy = ev.clientY - lastDragY;
  cam.offsetX += dx;
  cam.offsetY += dy;
  lastDragX = ev.clientX;
  lastDragY = ev.clientY;
});

function endDrag(ev) {
  if (!isDragging) return;
  isDragging = false;
  try { canvas.releasePointerCapture(ev.pointerId); } catch {}
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

  // wheel delta is in screen pixels; use exponential scaling for smoothness.
  const zoomFactor = Math.exp(-ev.deltaY * 0.0015);
  zoomAround(cam, ev.clientX, ev.clientY, zoomFactor);
}, { passive: false });

await refreshScenarios();
await refreshWorlds();
if (currentWorldId) connectStream({ worldId: currentWorldId, mode: 'render', hz: 10 });

setInterval(() => {
  // keep world labels fresh (paused/crashed)
  refreshWorlds().catch(() => {});
}, 2000);
