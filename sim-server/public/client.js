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

const ctx = canvas.getContext('2d');

let ws = null;
let currentWorldId = null;

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeCanvas);
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

function drawSnapshot(snap) {
  const rect = canvas.getBoundingClientRect();
  const pad = 12;

  const worldW = Number(snap?.world?.width) || 1;
  const worldH = Number(snap?.world?.height) || 1;
  const scale = Math.min((rect.width - pad * 2) / worldW, (rect.height - pad * 2) / worldH);

  ctx.clearRect(0, 0, rect.width, rect.height);

  const fluid = snap.fluid;
  if (fluid && Array.isArray(fluid.cells) && Number.isFinite(fluid.cellSize) && fluid.cellSize > 0) {
    const cs = fluid.cellSize;
    for (const cell of fluid.cells) {
      const gx = cell.gx;
      const gy = cell.gy;
      const rgb = cell.rgb || [80, 80, 110];
      const alpha = Math.max(0.08, Math.min(0.5, (cell.density || 0) / 255));
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
      ctx.fillRect(pad + gx * cs * scale, pad + gy * cs * scale, cs * scale, cs * scale);
    }
  }

  ctx.lineWidth = 1;
  for (const c of snap.creatures || []) {
    const verts = c.vertices || [];
    const springs = c.springs || [];

    ctx.strokeStyle = 'rgba(220,220,240,0.22)';
    ctx.beginPath();
    for (const s of springs) {
      const a = verts[s.a];
      const b = verts[s.b];
      if (!a || !b) continue;
      ctx.moveTo(pad + a.x * scale, pad + a.y * scale);
      ctx.lineTo(pad + b.x * scale, pad + b.y * scale);
    }
    ctx.stroke();

    for (const v of verts) {
      const r = Math.max(1.0, (v.radius || 2) * scale);
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.beginPath();
      ctx.arc(pad + v.x * scale, pad + v.y * scale, Math.min(r, 6), 0, Math.PI * 2);
      ctx.fill();
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
        statusEl.textContent = `world=${s.id} scenario=${s.scenario} seed=${s.seed} tick=${s.tick} t=${fmt(s.time, 2)} dt=${fmt(s.dt, 5)} paused=${s.paused} sps=${s.stepsPerSecond} stepWallMs=${s.lastStepWallMs}`;
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

await refreshScenarios();
await refreshWorlds();
if (currentWorldId) connectStream({ worldId: currentWorldId, mode: 'render', hz: 10 });

setInterval(() => {
  // keep world labels fresh (paused/crashed)
  refreshWorlds().catch(() => {});
}, 2000);
