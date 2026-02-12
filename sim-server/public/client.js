const statusEl = document.getElementById('status');
const telemetryEl = document.getElementById('telemetry');
const canvas = document.getElementById('canvas');
const scenarioSelect = document.getElementById('scenarioSelect');
const seedInput = document.getElementById('seedInput');
const setScenarioBtn = document.getElementById('setScenarioBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resumeBtn = document.getElementById('resumeBtn');

const ctx = canvas.getContext('2d');

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
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

function fmt(n, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return '—';
  return x.toFixed(digits);
}

function drawSnapshot(snap) {
  const w = snap?.world?.width ?? null;
  const h = snap?.world?.height ?? null;

  // snapshot currently doesn’t include world dims directly; infer from creature bounds if needed.
  // For now, assume node micro worlds and draw in a normalized fit based on max coordinates.
  let worldW = 0;
  let worldH = 0;
  for (const c of snap.creatures || []) {
    for (const v of c.vertices || []) {
      worldW = Math.max(worldW, v.x || 0);
      worldH = Math.max(worldH, v.y || 0);
    }
  }
  worldW = Math.max(worldW, 1);
  worldH = Math.max(worldH, 1);

  const rect = canvas.getBoundingClientRect();
  const pad = 12;
  const scale = Math.min((rect.width - pad * 2) / worldW, (rect.height - pad * 2) / worldH);

  ctx.clearRect(0, 0, rect.width, rect.height);

  // Fluid (sparse cell list)
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

  // Springs and points
  ctx.lineWidth = 1;
  for (const c of snap.creatures || []) {
    const verts = c.vertices || [];
    const springs = c.springs || [];

    // springs
    ctx.strokeStyle = 'rgba(220,220,240,0.25)';
    ctx.beginPath();
    for (const s of springs) {
      const a = verts[s.a];
      const b = verts[s.b];
      if (!a || !b) continue;
      ctx.moveTo(pad + a.x * scale, pad + a.y * scale);
      ctx.lineTo(pad + b.x * scale, pad + b.y * scale);
    }
    ctx.stroke();

    // points
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

async function loop() {
  try {
    const status = await apiGet('/api/status');
    statusEl.textContent = `scenario=${status.scenario} seed=${status.seed} tick=${status.tick} t=${fmt(status.time, 2)} dt=${fmt(status.dt, 5)} paused=${status.paused} sps=${status.stepsPerSecond} stepWallMs=${status.lastStepWallMs}`;

    const snap = await apiGet('/api/snapshot?mode=render');
    telemetryEl.textContent = JSON.stringify({
      tick: snap.tick,
      time: snap.time,
      populations: snap.populations,
      creatures: (snap.creatures || []).length,
      fluidCells: snap.fluid?.cells?.length ?? 0
    }, null, 2);

    drawSnapshot(snap);
  } catch (err) {
    statusEl.textContent = `error: ${String(err?.message || err)}`;
  } finally {
    setTimeout(loop, 200);
  }
}

setScenarioBtn.addEventListener('click', async () => {
  const name = scenarioSelect.value;
  const seed = Number(seedInput.value || 0) >>> 0;
  await apiPost('/api/control/setScenario', { name, seed });
});

pauseBtn.addEventListener('click', async () => {
  await apiPost('/api/control/pause');
});

resumeBtn.addEventListener('click', async () => {
  await apiPost('/api/control/resume');
});

await refreshScenarios();
loop();
