import { compileFieldToMesh } from '/field-to-structure-core.js';
import { createCreatureSpecFromMesh, parseCreatureSpec, buildMembraneRingsFromSoftField } from '/creature-spec.js';

const paintCanvas = document.getElementById('paint');
const densityCanvas = document.getElementById('densityPaint');
const membraneEdgeCanvas = document.getElementById('membraneEdgePaint');
const membraneShapeCanvas = document.getElementById('membraneShapePaint');
const rigidPermeabilityCanvas = document.getElementById('rigidPermeabilityPaint');
const meshCanvas = document.getElementById('mesh');
const pctx = paintCanvas.getContext('2d');
const dctx = densityCanvas.getContext('2d');
const ectx = membraneEdgeCanvas.getContext('2d');
const sctx = membraneShapeCanvas.getContext('2d');
const prctx = rigidPermeabilityCanvas.getContext('2d');
const mctx = meshCanvas.getContext('2d');
const modeEl = document.getElementById('paintMode');
const brushEl = document.getElementById('brush');
const thresholdEl = document.getElementById('threshold');
const softDensityPaintEl = document.getElementById('softDensityPaint');
const membraneEdgePaintEl = document.getElementById('membraneEdgePaintValue');
const membraneShapePaintEl = document.getElementById('membraneShapePaintValue');
const rigidPermeabilityPaintEl = document.getElementById('rigidPermeabilityPaintValue');
const rigidCompileModeEl = document.getElementById('rigidCompileMode');
const rigidPrimitiveSideMinEl = document.getElementById('rigidPrimitiveSideMin');
const rigidPrimitiveSideMaxEl = document.getElementById('rigidPrimitiveSideMax');
const softInfillModeEl = document.getElementById('softInfillMode');
const softMinCellSizeEl = document.getElementById('softMinCellSize');
const softBoundaryRingEl = document.getElementById('softBoundaryRing');
const softSolverModeEl = document.getElementById('softSolverMode');
const membraneMinEdgeLengthEl = document.getElementById('membraneMinEdgeLength');
const membraneMaxEdgeLengthEl = document.getElementById('membraneMaxEdgeLength');
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
const membraneEdgeMap = new Float32Array(W * H).fill(0.5);
const membraneShapeMap = new Float32Array(W * H).fill(1.0);
const rigidPermeabilityMap = new Float32Array(W * H).fill(0.0);
let lastMesh = null;
let compileRevision = 0;

function idx(x, y) { return y * W + x; }

function drawFields() {
  const traitImg = pctx.createImageData(W, H);
  const densityImg = dctx.createImageData(W, H);
  const edgeImg = ectx.createImageData(W, H);
  const shapeImg = sctx.createImageData(W, H);
  const permeabilityImg = prctx.createImageData(W, H);

  for (let i = 0; i < rigid.length; i++) {
    const r = Math.max(0, Math.min(1, rigid[i]));
    const s = Math.max(0, Math.min(1, soft[i]));
    const dens = Math.max(0, Math.min(1, softDensity[i]));
    const edge = Math.max(0, Math.min(1, membraneEdgeMap[i]));
    const shape = Math.max(0, Math.min(1, membraneShapeMap[i]));
    const permeability = Math.max(0, Math.min(1, rigidPermeabilityMap[i]));

    // Trait plane: rigid red, soft blue.
    traitImg.data[i * 4] = Math.min(255, r * 255);
    traitImg.data[i * 4 + 1] = 0;
    traitImg.data[i * 4 + 2] = Math.min(255, s * 255);
    traitImg.data[i * 4 + 3] = 255;

    // Density plane: green-only.
    densityImg.data[i * 4] = 0;
    densityImg.data[i * 4 + 1] = Math.round(40 + dens * 180);
    densityImg.data[i * 4 + 2] = 0;
    densityImg.data[i * 4 + 3] = 255;

    // Edge-length map: grayscale (black=min edge, white=max edge).
    const eg = Math.round(edge * 255);
    edgeImg.data[i * 4] = eg;
    edgeImg.data[i * 4 + 1] = eg;
    edgeImg.data[i * 4 + 2] = eg;
    edgeImg.data[i * 4 + 3] = 255;

    // Shape-memory map: amber->cyan gradient to show give/stiff bias.
    shapeImg.data[i * 4] = Math.round(220 * (1 - shape));
    shapeImg.data[i * 4 + 1] = Math.round(180 + 60 * shape);
    shapeImg.data[i * 4 + 2] = Math.round(120 + 120 * shape);
    shapeImg.data[i * 4 + 3] = 255;

    // Rigid edge permeability map: grayscale (black=blocked, white=permeable).
    const pg = Math.round(permeability * 255);
    permeabilityImg.data[i * 4] = pg;
    permeabilityImg.data[i * 4 + 1] = pg;
    permeabilityImg.data[i * 4 + 2] = pg;
    permeabilityImg.data[i * 4 + 3] = 255;
  }

  const blit = (ctx, canvas, img) => {
    const tmp = document.createElement('canvas');
    tmp.width = W;
    tmp.height = H;
    tmp.getContext('2d').putImageData(img, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tmp, 0, 0, W, H, 0, 0, canvas.width, canvas.height);
  };

  blit(pctx, paintCanvas, traitImg);
  blit(dctx, densityCanvas, densityImg);
  blit(ectx, membraneEdgeCanvas, edgeImg);
  blit(sctx, membraneShapeCanvas, shapeImg);
  blit(prctx, rigidPermeabilityCanvas, permeabilityImg);
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

function paintScalarMap(targetArray, canvasEl, clientX, clientY, paintValue, eraseValue, erase = false) {
  const rect = canvasEl.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * W;
  const y = ((clientY - rect.top) / rect.height) * H;
  const r = Math.max(1, Number(brushEl.value) || 14) * (W / canvasEl.width);
  const target = erase
    ? Math.max(0, Math.min(1, Number(eraseValue)))
    : Math.max(0, Math.min(1, Number(paintValue)));

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
      targetArray[i] = targetArray[i] * (1 - t) + target * t;
    }
  }
}

function paintDensity(clientX, clientY, erase = false) {
  paintScalarMap(
    softDensity,
    densityCanvas,
    clientX,
    clientY,
    Number(softDensityPaintEl?.value) || 0.5,
    0.5,
    erase,
  );
  drawFields();
}

function paintMembraneEdgeMap(clientX, clientY, erase = false) {
  paintScalarMap(
    membraneEdgeMap,
    membraneEdgeCanvas,
    clientX,
    clientY,
    Number(membraneEdgePaintEl?.value) || 0.5,
    0.5,
    erase,
  );
  drawFields();
}

function paintMembraneShapeMap(clientX, clientY, erase = false) {
  paintScalarMap(
    membraneShapeMap,
    membraneShapeCanvas,
    clientX,
    clientY,
    Number(membraneShapePaintEl?.value) || 1.0,
    1.0,
    erase,
  );
  drawFields();
}

function paintRigidPermeabilityMap(clientX, clientY, erase = false) {
  paintScalarMap(
    rigidPermeabilityMap,
    rigidPermeabilityCanvas,
    clientX,
    clientY,
    Number(rigidPermeabilityPaintEl?.value) || 1.0,
    0.0,
    erase,
  );
  drawFields();
}

function drawMesh(mesh) {
  mctx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
  const sx = meshCanvas.width / W;
  const sy = meshCanvas.height / H;
  const membranePreview = (softSolverModeEl?.value || 'spring') === 'membrane';

  if (!membranePreview) {
    // Spring mode preview: show compiled soft triangles + optional cross-beams.
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
  } else {
    // Membrane mode preview: draw the same resampled perimeter ring used for export.
    const thr = Math.max(0, Math.min(1, Number(thresholdEl?.value) || 0.35));
    const minEdge = Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4);
    const maxEdge = Math.max(minEdge, Number(membraneMaxEdgeLengthEl?.value) || 8);
    const rings = buildMembraneRingsFromSoftField({
      width: W,
      height: H,
      softField: soft,
      edgeLengthField: membraneEdgeMap,
      threshold: thr,
      minEdgeLength: minEdge,
      maxEdgeLength: maxEdge,
    });

    mctx.strokeStyle = 'rgba(120,220,255,0.95)';
    mctx.fillStyle = 'rgba(120,240,255,0.95)';
    mctx.lineWidth = 2;
    for (const ring of rings) {
      if (!Array.isArray(ring) || ring.length < 3) continue;
      mctx.beginPath();
      for (let i = 0; i < ring.length; i++) {
        const p = ring[i];
        const x = p.x * sx;
        const y = p.y * sy;
        if (i === 0) mctx.moveTo(x, y);
        else mctx.lineTo(x, y);
      }
      mctx.closePath();
      mctx.stroke();

      for (const p of ring) {
        mctx.beginPath();
        mctx.arc(p.x * sx, p.y * sy, 1.8, 0, Math.PI * 2);
        mctx.fill();
      }
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

  out.textContent = JSON.stringify({
    ...mesh.meta,
    compileRevision,
    softSolverMode: softSolverModeEl?.value || 'spring',
    softPreview: ((softSolverModeEl?.value || 'spring') === 'membrane')
      ? 'resampled membrane ring from painted mask'
      : 'triangulated soft mesh',
    membraneMinEdgeLength: Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4),
    membraneMaxEdgeLength: Math.max(Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4), Number(membraneMaxEdgeLengthEl?.value) || 8),
  }, null, 2);
}

function syncSoftModeUi() {
  const membraneMode = (softSolverModeEl?.value || 'spring') === 'membrane';
  if (!softInfillModeEl) return membraneMode;
  if (membraneMode) {
    softInfillModeEl.value = 'none';
    softInfillModeEl.disabled = true;
    softInfillModeEl.title = 'Membrane mode uses perimeter-only representation (no interior infill)';
    if (membraneMinEdgeLengthEl) {
      membraneMinEdgeLengthEl.disabled = false;
      membraneMinEdgeLengthEl.title = 'Minimum edge length for exported membrane ring';
    }
    if (membraneMaxEdgeLengthEl) {
      membraneMaxEdgeLengthEl.disabled = false;
      membraneMaxEdgeLengthEl.title = 'Maximum edge length for exported membrane ring';
    }
  } else {
    if (softInfillModeEl.value === 'none') softInfillModeEl.value = 'triangles';
    softInfillModeEl.disabled = false;
    softInfillModeEl.title = '';
    if (membraneMinEdgeLengthEl) {
      membraneMinEdgeLengthEl.disabled = true;
      membraneMinEdgeLengthEl.title = 'Enable membrane mode to edit membrane edge spacing';
    }
    if (membraneMaxEdgeLengthEl) {
      membraneMaxEdgeLengthEl.disabled = true;
      membraneMaxEdgeLengthEl.title = 'Enable membrane mode to edit membrane edge spacing';
    }
  }
  return membraneMode;
}

function compileNow() {
  try {
    const membraneMode = syncSoftModeUi();
    const requestedSoftMin = Math.max(1, Math.min(Math.max(1, W - 1), Math.round(Number(softMinCellSizeEl?.value) || 3)));

    const mesh = compileFieldToMesh({
      width: W,
      height: H,
      rigidField: rigid,
      softField: soft,
      density: 1,
      threshold: Math.max(0, Math.min(1, Number(thresholdEl.value) || 0.35)),
      connectivityMode: 'largest',
      minComponentTriangles: 0,
      rigidCompileMode: rigidCompileModeEl?.value || 'contours',
      rigidPrimitiveSideMin: Math.max(2, Math.min(64, Math.round(Number(rigidPrimitiveSideMinEl?.value) || 4))),
      rigidPrimitiveSideMax: Math.max(2, Math.min(96, Math.round(Number(rigidPrimitiveSideMaxEl?.value) || 10))),
      softInfillMode: membraneMode ? 'none' : (softInfillModeEl?.value || 'triangles'),
      softDensityField: softDensity,
      // Membrane mode should preserve boundary fidelity; coarse soft cells make boxy/square contours.
      softMinCellSize: membraneMode ? 1 : requestedSoftMin,
      softBoundaryCellCap: membraneMode ? 1 : 2,
    });
    lastMesh = mesh;
    compileRevision += 1;
    drawMesh(mesh);
    return mesh;
  } catch (err) {
    const msg = `Compile failed: ${String(err?.message || err)}`;
    console.error('[mesh-lab] compileNow failed', err);
    out.textContent = msg;
    return null;
  }
}

let traitPainting = false;
let densityPainting = false;
let membraneEdgePainting = false;
let membraneShapePainting = false;
let rigidPermeabilityPainting = false;

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

membraneEdgeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
membraneEdgeCanvas.addEventListener('mousedown', (e) => {
  membraneEdgePainting = true;
  paintMembraneEdgeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
membraneEdgeCanvas.addEventListener('mousemove', (e) => {
  if (!membraneEdgePainting) return;
  paintMembraneEdgeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

membraneShapeCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
membraneShapeCanvas.addEventListener('mousedown', (e) => {
  membraneShapePainting = true;
  paintMembraneShapeMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
membraneShapeCanvas.addEventListener('mousemove', (e) => {
  if (!membraneShapePainting) return;
  paintMembraneShapeMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

rigidPermeabilityCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
rigidPermeabilityCanvas.addEventListener('mousedown', (e) => {
  rigidPermeabilityPainting = true;
  paintRigidPermeabilityMap(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
rigidPermeabilityCanvas.addEventListener('mousemove', (e) => {
  if (!rigidPermeabilityPainting) return;
  paintRigidPermeabilityMap(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

window.addEventListener('mouseup', () => {
  traitPainting = false;
  densityPainting = false;
  membraneEdgePainting = false;
  membraneShapePainting = false;
  rigidPermeabilityPainting = false;
});

clearBtn.addEventListener('click', () => {
  rigid.fill(0);
  soft.fill(0);
  softDensity.fill(0.5);
  membraneEdgeMap.fill(0.5);
  membraneShapeMap.fill(1.0);
  rigidPermeabilityMap.fill(0.0);
  drawFields();
  compileNow();
});
compileBtn.addEventListener('click', compileNow);
if (softInfillModeEl) softInfillModeEl.addEventListener('change', compileNow);
if (softMinCellSizeEl) softMinCellSizeEl.addEventListener('change', compileNow);
if (softSolverModeEl) softSolverModeEl.addEventListener('change', compileNow);
if (membraneMinEdgeLengthEl) membraneMinEdgeLengthEl.addEventListener('change', compileNow);
if (membraneMaxEdgeLengthEl) membraneMaxEdgeLengthEl.addEventListener('change', compileNow);
if (rigidCompileModeEl) rigidCompileModeEl.addEventListener('change', compileNow);
if (rigidPrimitiveSideMinEl) rigidPrimitiveSideMinEl.addEventListener('change', compileNow);
if (rigidPrimitiveSideMaxEl) rigidPrimitiveSideMaxEl.addEventListener('change', compileNow);

exportBtn.addEventListener('click', () => {
  const mesh = compileNow();
  if (!mesh) return;
  const membraneMinEdgeLength = Math.max(1, Number(membraneMinEdgeLengthEl?.value) || 4);
  const membraneMaxEdgeLength = Math.max(membraneMinEdgeLength, Number(membraneMaxEdgeLengthEl?.value) || 8);
  const spec = createCreatureSpecFromMesh(mesh, {
    name: 'mesh-lab-creature',
    fields: {
      rigidField: rigid,
      softField: soft,
      softDensityField: softDensity,
      membraneEdgeMap,
      membraneShapeMap,
      rigidPermeabilityMap,
    },
    threshold: Math.max(0, Math.min(1, Number(thresholdEl?.value) || 0.35)),
    softBoundaryRingSprings: !!softBoundaryRingEl?.checked,
    softSolverMode: softSolverModeEl?.value || 'spring',
    membraneMinEdgeLength,
    membraneMaxEdgeLength,
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
  if (softSolverModeEl) {
    const mode = (spec.softBodies?.[0]?.solverMode === 'membrane') ? 'membrane' : 'spring';
    softSolverModeEl.value = mode;
  }
  syncSoftModeUi();

  const authoring = spec.authoring?.fields;
  if (authoring && Number(authoring.width) === W && Number(authoring.height) === H && Array.isArray(authoring.rigid) && Array.isArray(authoring.soft)) {
    rigid.set(authoring.rigid);
    soft.set(authoring.soft);
    if (Array.isArray(authoring.softDensity) && authoring.softDensity.length === W * H) softDensity.set(authoring.softDensity);
    else softDensity.fill(0.5);
    if (Array.isArray(authoring.membraneEdgeMap) && authoring.membraneEdgeMap.length === W * H) membraneEdgeMap.set(authoring.membraneEdgeMap);
    else membraneEdgeMap.fill(0.5);
    if (Array.isArray(authoring.membraneShapeMap) && authoring.membraneShapeMap.length === W * H) membraneShapeMap.set(authoring.membraneShapeMap);
    else membraneShapeMap.fill(1.0);
    if (Array.isArray(authoring.rigidPermeabilityMap) && authoring.rigidPermeabilityMap.length === W * H) rigidPermeabilityMap.set(authoring.rigidPermeabilityMap);
    else rigidPermeabilityMap.fill(0.0);
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
