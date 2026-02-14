import { compileFieldToMesh } from '/field-to-structure-core.js';
import { createCreatureSpecFromMesh, parseCreatureSpec } from '/creature-spec.js';

const paintCanvas = document.getElementById('paint');
const meshCanvas = document.getElementById('mesh');
const pctx = paintCanvas.getContext('2d');
const mctx = meshCanvas.getContext('2d');
const modeEl = document.getElementById('paintMode');
const brushEl = document.getElementById('brush');
const densityEl = document.getElementById('density');
const thresholdEl = document.getElementById('threshold');
const clearBtn = document.getElementById('clearBtn');
const compileBtn = document.getElementById('compileBtn');
const exportBtn = document.getElementById('exportBtn');
const importBtn = document.getElementById('importBtn');
const importFile = document.getElementById('importFile');
const out = document.getElementById('out');

const W = 128, H = 128;
const rigid = new Float32Array(W * H);
const soft = new Float32Array(W * H);
let painting = false;
let lastMesh = null;

function idx(x, y) { return y * W + x; }

function drawFields() {
  const img = pctx.createImageData(W, H);
  for (let i = 0; i < rigid.length; i++) {
    img.data[i * 4] = Math.min(255, rigid[i] * 255);
    img.data[i * 4 + 1] = 0;
    img.data[i * 4 + 2] = Math.min(255, soft[i] * 255);
    img.data[i * 4 + 3] = 255;
  }
  const tmp = document.createElement('canvas');
  tmp.width = W; tmp.height = H;
  tmp.getContext('2d').putImageData(img, 0, 0);
  pctx.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
  pctx.drawImage(tmp, 0, 0, W, H, 0, 0, paintCanvas.width, paintCanvas.height);
}

function paint(clientX, clientY) {
  const rect = paintCanvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * W;
  const y = ((clientY - rect.top) / rect.height) * H;
  const r = Math.max(1, Number(brushEl.value) || 14) * (W / paintCanvas.width);
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
      if (mode === 'rigid') rigid[i] = Math.max(rigid[i], t);
      else if (mode === 'soft') soft[i] = Math.max(soft[i], t);
      else { rigid[i] *= (1 - t); soft[i] *= (1 - t); }
    }
  }
  drawFields();
}

function drawMesh(mesh) {
  mctx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
  const sx = meshCanvas.width / W;
  const sy = meshCanvas.height / H;

  // Soft-body debug mesh stays visible; rigid now renders via contour hull only.
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
      mctx.lineWidth = 1;
    }
  }

  if (Array.isArray(mesh.rigidWelds) && Array.isArray(mesh.rigidPieces)) {
    for (const w of mesh.rigidWelds) {
      const a = mesh.rigidPieces[w.a];
      const b = mesh.rigidPieces[w.b];
      if (!a?.hull?.length || !b?.hull?.length) continue;
      const a0 = a.hull[w.a0 % a.hull.length];
      const a1 = a.hull[w.a1 % a.hull.length];
      const b0 = b.hull[w.b0 % b.hull.length];
      const b1 = b.hull[w.b1 % b.hull.length];
      if (!a0 || !a1 || !b0 || !b1) continue;
      const ma = { x: (a0.x + a1.x) * 0.5, y: (a0.y + a1.y) * 0.5 };
      const mb = { x: (b0.x + b1.x) * 0.5, y: (b0.y + b1.y) * 0.5 };
      mctx.strokeStyle = 'rgba(255,180,60,0.95)';
      mctx.beginPath();
      mctx.moveTo(ma.x * sx, ma.y * sy);
      mctx.lineTo(mb.x * sx, mb.y * sy);
      mctx.stroke();
    }
  }

  out.textContent = JSON.stringify(mesh.meta, null, 2);
}

function compileNow() {
  const mesh = compileFieldToMesh({
    width: W,
    height: H,
    rigidField: rigid,
    softField: soft,
    density: Math.max(1, Number(densityEl.value) || 4),
    threshold: Math.max(0, Math.min(1, Number(thresholdEl.value) || 0.35)),
    connectivityMode: 'largest',
    minComponentTriangles: 0,
  });
  lastMesh = mesh;
  drawMesh(mesh);
}

paintCanvas.addEventListener('mousedown', (e) => { painting = true; paint(e.clientX, e.clientY); });
window.addEventListener('mouseup', () => { painting = false; });
paintCanvas.addEventListener('mousemove', (e) => { if (painting) paint(e.clientX, e.clientY); });
clearBtn.addEventListener('click', () => { rigid.fill(0); soft.fill(0); drawFields(); compileNow(); });
compileBtn.addEventListener('click', compileNow);

exportBtn.addEventListener('click', () => {
  if (!lastMesh) compileNow();
  const spec = createCreatureSpecFromMesh(lastMesh, {
    name: 'mesh-lab-creature',
    fields: { rigidField: rigid, softField: soft },
  });
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

  const authoring = spec.authoring?.fields;
  if (authoring && Number(authoring.width) === W && Number(authoring.height) === H && Array.isArray(authoring.rigid) && Array.isArray(authoring.soft)) {
    rigid.set(authoring.rigid);
    soft.set(authoring.soft);
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
});

drawFields();
compileNow();
