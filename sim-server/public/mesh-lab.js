import { compileFieldToMesh } from '/field-to-structure-core.js';
import { createCreatureSpecFromMesh, parseCreatureSpec } from '/creature-spec.js';

const paintCanvas = document.getElementById('paint');
const densityCanvas = document.getElementById('densityPaint');
const meshCanvas = document.getElementById('mesh');
const pctx = paintCanvas.getContext('2d');
const dctx = densityCanvas.getContext('2d');
const mctx = meshCanvas.getContext('2d');
const modeEl = document.getElementById('paintMode');
const brushEl = document.getElementById('brush');
const thresholdEl = document.getElementById('threshold');
const softDensityPaintEl = document.getElementById('softDensityPaint');
const softInfillModeEl = document.getElementById('softInfillMode');
const softMinCellSizeEl = document.getElementById('softMinCellSize');
const clearBtn = document.getElementById('clearBtn');
const compileBtn = document.getElementById('compileBtn');
const exportBtn = document.getElementById('exportBtn');
const importBtn = document.getElementById('importBtn');
const importFile = document.getElementById('importFile');
const out = document.getElementById('out');

const W = 128, H = 128;
const rigid = new Float32Array(W * H);
const soft = new Float32Array(W * H);
const softDensity = new Float32Array(W * H).fill(0.5);
let lastMesh = null;

function idx(x, y) { return y * W + x; }

function drawFields() {
  const traitImg = pctx.createImageData(W, H);
  const densityImg = dctx.createImageData(W, H);

  for (let i = 0; i < rigid.length; i++) {
    const r = Math.max(0, Math.min(1, rigid[i]));
    const s = Math.max(0, Math.min(1, soft[i]));
    const dens = Math.max(0, Math.min(1, softDensity[i]));

    // Trait plane: rigid red, soft blue, no density overlay.
    traitImg.data[i * 4] = Math.min(255, r * 255);
    traitImg.data[i * 4 + 1] = 0;
    traitImg.data[i * 4 + 2] = Math.min(255, s * 255);
    traitImg.data[i * 4 + 3] = 255;

    // Density plane: green-only, default mid-green around 0.5.
    densityImg.data[i * 4] = 0;
    densityImg.data[i * 4 + 1] = Math.round(40 + dens * 180);
    densityImg.data[i * 4 + 2] = 0;
    densityImg.data[i * 4 + 3] = 255;
  }

  const traitTmp = document.createElement('canvas');
  traitTmp.width = W; traitTmp.height = H;
  traitTmp.getContext('2d').putImageData(traitImg, 0, 0);
  pctx.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
  pctx.drawImage(traitTmp, 0, 0, W, H, 0, 0, paintCanvas.width, paintCanvas.height);

  const densTmp = document.createElement('canvas');
  densTmp.width = W; densTmp.height = H;
  densTmp.getContext('2d').putImageData(densityImg, 0, 0);
  dctx.clearRect(0, 0, densityCanvas.width, densityCanvas.height);
  dctx.drawImage(densTmp, 0, 0, W, H, 0, 0, densityCanvas.width, densityCanvas.height);
}

function paintTrait(clientX, clientY) {
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
      if (mode === 'rigid') {
        rigid[i] = Math.max(rigid[i], t);
        soft[i] *= (1 - t);
      } else if (mode === 'soft') {
        soft[i] = Math.max(soft[i], t);
        rigid[i] *= (1 - t);
      } else {
        rigid[i] *= (1 - t);
        soft[i] *= (1 - t);
      }
    }
  }
  drawFields();
}

function paintDensity(clientX, clientY, erase = false) {
  const rect = densityCanvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * W;
  const y = ((clientY - rect.top) / rect.height) * H;
  const r = Math.max(1, Number(brushEl.value) || 14) * (W / densityCanvas.width);
  const target = erase ? 0.5 : Math.max(0, Math.min(1, Number(softDensityPaintEl?.value) || 0.5));

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
      softDensity[i] = softDensity[i] * (1 - t) + target * t;
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

  if (Array.isArray(mesh.softCrossBeams)) {
    mctx.strokeStyle = 'rgba(140,220,255,0.92)';
    mctx.lineWidth = 1.5;
    for (const [ai, bi] of mesh.softCrossBeams) {
      const a = mesh.nodes[ai];
      const b = mesh.nodes[bi];
      if (!a || !b) continue;
      mctx.beginPath();
      mctx.moveTo(a.x * sx, a.y * sy);
      mctx.lineTo(b.x * sx, b.y * sy);
      mctx.stroke();
    }
    mctx.lineWidth = 1;
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

  out.textContent = JSON.stringify(mesh.meta, null, 2);
}

function compileNow() {
  const mesh = compileFieldToMesh({
    width: W,
    height: H,
    rigidField: rigid,
    softField: soft,
    density: 1,
    threshold: Math.max(0, Math.min(1, Number(thresholdEl.value) || 0.35)),
    connectivityMode: 'largest',
    minComponentTriangles: 0,
    softInfillMode: softInfillModeEl?.value || 'triangles',
    softDensityField: softDensity,
    softMinCellSize: Math.max(1, Math.min(Math.max(1, W - 1), Math.round(Number(softMinCellSizeEl?.value) || 3))),
  });
  lastMesh = mesh;
  drawMesh(mesh);
}

let traitPainting = false;
let densityPainting = false;

paintCanvas.addEventListener('mousedown', (e) => { traitPainting = true; paintTrait(e.clientX, e.clientY); });
paintCanvas.addEventListener('mousemove', (e) => { if (traitPainting) paintTrait(e.clientX, e.clientY); });

densityCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
densityCanvas.addEventListener('mousedown', (e) => {
  densityPainting = true;
  paintDensity(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
densityCanvas.addEventListener('mousemove', (e) => {
  if (!densityPainting) return;
  paintDensity(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

window.addEventListener('mouseup', () => {
  traitPainting = false;
  densityPainting = false;
});

clearBtn.addEventListener('click', () => { rigid.fill(0); soft.fill(0); softDensity.fill(0.5); drawFields(); compileNow(); });
compileBtn.addEventListener('click', compileNow);
if (softInfillModeEl) softInfillModeEl.addEventListener('change', compileNow);
if (softMinCellSizeEl) softMinCellSizeEl.addEventListener('change', compileNow);

exportBtn.addEventListener('click', () => {
  if (!lastMesh) compileNow();
  const spec = createCreatureSpecFromMesh(lastMesh, {
    name: 'mesh-lab-creature',
    fields: { rigidField: rigid, softField: soft, softDensityField: softDensity },
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
    if (Array.isArray(authoring.softDensity) && authoring.softDensity.length === W * H) softDensity.set(authoring.softDensity);
    else softDensity.fill(0.5);
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
