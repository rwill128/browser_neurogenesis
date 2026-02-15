import { parseCreatureSpec, buildBodiesFromCreatureSpec } from '/creature-spec.js';
import { EDGE_DYE_MODE, normalizeEdgeDyeModeRGB, applyBodyEdgeFieldBarriers } from '/dye-barrier.js';
import { resolveRigidVsSoftNodeCollision, resolveRigidVsRigidPolygonCollision, getRigidCollisionPolysWorld } from '/rigid-collision.js';
import { sanitizeSoftSprings, ensureLambdaCacheSize, buildSoftClusterBoundaryLoops } from '/soft-xpbd.js';

const out = document.getElementById('out');
const runBtn = document.getElementById('runBtn');
const stopBtn = document.getElementById('stopBtn');
const clearViscBtn = document.getElementById('clearViscBtn');
const importSpecBtn = document.getElementById('importSpecBtn');
const importSpecFile = document.getElementById('importSpecFile');
const fpsHud = document.getElementById('fpsHud');
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');

const gridEl = document.getElementById('gridSize');
const dtEl = document.getElementById('dt');
const fadeEl = document.getElementById('fade');
const viscosityEl = document.getElementById('viscosity');
const impulseEl = document.getElementById('impulse');
const radiusEl = document.getElementById('radius');
const brushSizeEl = document.getElementById('brushSize');
const paintValueEl = document.getElementById('paintValue');
const showViscEl = document.getElementById('showVisc');
const showCollisionHullEl = document.getElementById('showCollisionHull');
const scenarioPresetEl = document.getElementById('scenarioPreset');
const massLightEl = document.getElementById('massLight');
const massHeavyEl = document.getElementById('massHeavy');
const massSoftEl = document.getElementById('massSoft');
const bodyDragEl = document.getElementById('bodyDrag');
const bodyFeedbackEl = document.getElementById('bodyFeedback');

const GENERATED_MINI_SCENARIOS_URL = '/generated-mini-scenarios.json';
let generatedMiniScenarios = new Map();

function log(v) { out.textContent = typeof v === 'string' ? v : JSON.stringify(v, null, 2); }

const WORKGROUP = 8;
const JACOBI_ITERS = 20;
const SOFT_SPRING_STIFFNESS_DEFAULT = 3.0; // requested stronger default baseline
const SOFT_XPBD_ITERS = 10;
const SOFT_XPBD_BASE_COMPLIANCE = 0.0012;
const SOFT_AREA_XPBD_ITERS = 6;
const SOFT_AREA_BASE_COMPLIANCE = 0.0009;
const SOFT_INTEGRATION_SCALE = 24;
const FLUID_COUPLING_COMPONENT_LIMIT = 12;

const EDGE_BODY_MODE = {
  PASS: 0,
  BLOCK: 1,
};

function readControls() {
  return {
    n: Math.max(32, Number(gridEl.value) || 256),
    dt: Number(dtEl.value) || 0.03,
    fade: Number(fadeEl.value) || 0.9999,
    viscosity: Number(viscosityEl.value) || 0.00001,
    impulse: Number(impulseEl.value) || 2.5,
    radius: Number(radiusEl.value) || 6,
    massLight: Math.max(0.05, Number(massLightEl.value) || 1.2),
    massHeavy: Math.max(0.05, Number(massHeavyEl.value) || 5.0),
    massSoft: Math.max(0.02, Number(massSoftEl.value) || 0.6),
    bodyDrag: Math.max(0, Number(bodyDragEl.value) || 0.55),
    bodyFeedback: Math.max(0, Number(bodyFeedbackEl.value) || 0.012),
  };
}

function createBuffer(device, bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
  return device.createBuffer({ size: bytes, usage });
}

function createUniformBuffer(device) {
  return device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}

function uploadUniforms(device, uniformBuffer, s) {
  const a = new ArrayBuffer(32);
  const u32 = new Uint32Array(a);
  const f32 = new Float32Array(a);
  u32[0] = s.n;
  f32[1] = s.dt;
  f32[2] = s.fade;
  f32[3] = s.impulse;
  f32[4] = s.radius;
  f32[5] = s.viscosity;
  device.queue.writeBuffer(uniformBuffer, 0, a);
}

function makeDefaultViscMap(n) {
  const m = new Float32Array(n * n);
  m.fill(0.5);
  return m;
}

const commonWgsl = `
struct Params {
  n: u32,
  dt: f32,
  fade: f32,
  impulse: f32,
  radius: f32,
  viscosity_scale: f32,
  _pad0: f32,
  _pad1: f32,
};
@group(0) @binding(0) var<uniform> p: Params;

fn idx(x:u32,y:u32)->u32 { return y * p.n + x; }
fn clampi(v:i32, lo:i32, hi:i32)->i32 { return min(max(v, lo), hi); }
fn sampleBilinear(field: ptr<storage, array<f32>, read>, fx:f32, fy:f32)->f32 {
  let n1 = f32(p.n - 1u);
  let x = clamp(fx, 0.0, n1);
  let y = clamp(fy, 0.0, n1);
  let x0 = u32(floor(x));
  let y0 = u32(floor(y));
  let x1 = min(x0 + 1u, p.n - 1u);
  let y1 = min(y0 + 1u, p.n - 1u);
  let sx = x - f32(x0);
  let sy = y - f32(y0);
  let v00 = (*field)[idx(x0,y0)];
  let v10 = (*field)[idx(x1,y0)];
  let v01 = (*field)[idx(x0,y1)];
  let v11 = (*field)[idx(x1,y1)];
  let a = mix(v00,v10,sx);
  let b = mix(v01,v11,sx);
  return mix(a,b,sy);
}
`;

const advectVelWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read> vx0: array<f32>;
@group(0) @binding(2) var<storage, read> vy0: array<f32>;
@group(0) @binding(3) var<storage, read_write> vx1: array<f32>;
@group(0) @binding(4) var<storage, read_write> vy1: array<f32>;
@group(0) @binding(5) var<storage, read> viscMap: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let i = idx(gid.x, gid.y);
  let x = f32(gid.x);
  let y = f32(gid.y);
  let px = x - p.dt * vx0[i];
  let py = y - p.dt * vy0[i];
  let localVisc = max(0.0, viscMap[i]) * p.viscosity_scale;
  let decay = 1.0 / (1.0 + 4.0 * localVisc * p.dt);
  var nx = sampleBilinear(&vx0, px, py) * decay;
  var ny = sampleBilinear(&vy0, px, py) * decay;
  let vmax = 6.0;
  let mag = sqrt(nx * nx + ny * ny);
  if (mag > vmax) {
    let s = vmax / mag;
    nx = nx * s;
    ny = ny * s;
  }
  vx1[i] = nx;
  vy1[i] = ny;
}
`;

const divergenceWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read> vx: array<f32>;
@group(0) @binding(2) var<storage, read> vy: array<f32>;
@group(0) @binding(3) var<storage, read_write> div: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let n = i32(p.n) - 1;
  let xl = u32(clampi(x - 1, 0, n));
  let xr = u32(clampi(x + 1, 0, n));
  let yt = u32(clampi(y - 1, 0, n));
  let yb = u32(clampi(y + 1, 0, n));
  let dx = vx[idx(xr, gid.y)] - vx[idx(xl, gid.y)];
  let dy = vy[idx(gid.x, yb)] - vy[idx(gid.x, yt)];
  div[idx(gid.x, gid.y)] = 0.5 * (dx + dy);
}
`;

const jacobiPressureWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read> pressure0: array<f32>;
@group(0) @binding(2) var<storage, read> div: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure1: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let n = i32(p.n) - 1;
  let xl = u32(clampi(x - 1, 0, n));
  let xr = u32(clampi(x + 1, 0, n));
  let yt = u32(clampi(y - 1, 0, n));
  let yb = u32(clampi(y + 1, 0, n));
  let sumN = pressure0[idx(xl, gid.y)] + pressure0[idx(xr, gid.y)] + pressure0[idx(gid.x, yt)] + pressure0[idx(gid.x, yb)];
  pressure1[idx(gid.x, gid.y)] = (sumN - div[idx(gid.x, gid.y)]) * 0.25;
}
`;

const projectWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read_write> vx: array<f32>;
@group(0) @binding(2) var<storage, read_write> vy: array<f32>;
@group(0) @binding(3) var<storage, read> pressure: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let n = i32(p.n) - 1;
  let xl = u32(clampi(x - 1, 0, n));
  let xr = u32(clampi(x + 1, 0, n));
  let yt = u32(clampi(y - 1, 0, n));
  let yb = u32(clampi(y + 1, 0, n));
  let i = idx(gid.x, gid.y);
  vx[i] = vx[i] - 0.5 * (pressure[idx(xr, gid.y)] - pressure[idx(xl, gid.y)]);
  vy[i] = vy[i] - 0.5 * (pressure[idx(gid.x, yb)] - pressure[idx(gid.x, yt)]);

  let edge = (gid.x == 0u) || (gid.x == p.n - 1u) || (gid.y == 0u) || (gid.y == p.n - 1u);
  if (edge) {
    if (gid.x == 0u || gid.x == p.n - 1u) {
      vx[i] = 0.0;
      vy[i] = vy[i] * 0.75;
    }
    if (gid.y == 0u || gid.y == p.n - 1u) {
      vy[i] = 0.0;
      vx[i] = vx[i] * 0.75;
    }
  }
}
`;

const advectDyeWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read> vx: array<f32>;
@group(0) @binding(2) var<storage, read> vy: array<f32>;
@group(0) @binding(3) var<storage, read> r0: array<f32>;
@group(0) @binding(4) var<storage, read> g0: array<f32>;
@group(0) @binding(5) var<storage, read> b0: array<f32>;
@group(0) @binding(6) var<storage, read_write> r1: array<f32>;
@group(0) @binding(7) var<storage, read_write> g1: array<f32>;
@group(0) @binding(8) var<storage, read_write> b1: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let i = idx(gid.x, gid.y);
  let x = f32(gid.x);
  let y = f32(gid.y);
  let px = x - p.dt * vx[i];
  let py = y - p.dt * vy[i];
  r1[i] = sampleBilinear(&r0, px, py) * p.fade;
  g1[i] = sampleBilinear(&g0, px, py) * p.fade;
  b1[i] = sampleBilinear(&b0, px, py) * p.fade;
}
`;

const injectWgsl = commonWgsl + `
@group(0) @binding(1) var<storage, read_write> vx: array<f32>;
@group(0) @binding(2) var<storage, read_write> vy: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> g: array<f32>;
@group(0) @binding(5) var<storage, read_write> b: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let cx = f32(p.n) * 0.5;
  let cy = f32(p.n) * 0.5;
  let dx = f32(gid.x) - cx;
  let dy = f32(gid.y) - cy;
  let d2 = dx*dx + dy*dy;
  let i = idx(gid.x, gid.y);
  let r2 = p.radius * p.radius;
  if (d2 < r2) {
    vx[i] = vx[i] + p.impulse;
    vy[i] = vy[i] + 1.2 * sin(f32(i) * 0.0007);
    r[i] = 255.0;
    g[i] = 140.0;
    b[i] = 60.0;
  }
}
`;

async function createPipeline(device, code) {
  const module = device.createShaderModule({ code });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  return {
    pipeline,
    bg(resources) {
      return device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: resources.map((buffer, i) => ({ binding: i, resource: { buffer } }))
      });
    }
  };
}

let running = false;
let sim = null;
let pendingImportedSpecs = [];
let painting = false;
let panning = false;
let panLastX = 0;
let panLastY = 0;

function uploadViscMap() {
  if (!sim) return;
  sim.device.queue.writeBuffer(sim.viscMapGpu, 0, sim.viscMapCpu);
}

function drawViscosityOverlay() {
  if (!sim) return;
  const n = sim.controls.n;
  const img = ctx.createImageData(n, n);
  for (let i = 0; i < sim.viscMapCpu.length; i++) {
    const v = Math.max(0, Math.min(1, sim.viscMapCpu[i]));
    const d = v - 0.5; // neutral is transparent
    const o = i * 4;
    if (Math.abs(d) < 0.03) {
      img.data[o] = 0;
      img.data[o + 1] = 0;
      img.data[o + 2] = 0;
      img.data[o + 3] = 0;
      continue;
    }
    const t = Math.min(1, Math.abs(d) / 0.5);
    if (d > 0) {
      img.data[o] = 255;
      img.data[o + 1] = 40;
      img.data[o + 2] = 30;
    } else {
      img.data[o] = 45;
      img.data[o + 1] = 140;
      img.data[o + 2] = 255;
    }
    img.data[o + 3] = Math.floor(145 * t);
  }
  const tmp = document.createElement('canvas');
  tmp.width = n; tmp.height = n;
  tmp.getContext('2d').putImageData(img, 0, 0);
  const view = getCameraView(sim);
  ctx.drawImage(tmp, view.x, view.y, view.w, view.h, 0, 0, canvas.width, canvas.height);
}

function resetViscMap() {
  if (!sim) return;
  sim.viscMapCpu.fill(0.5);
  uploadViscMap();
  log({ ok: true, msg: 'viscosity map reset' });
}

function clampCamera(s) {
  const n = s.controls.n;
  const cam = s.camera;
  const minZoom = 1;
  const maxZoom = Math.max(1, n / 64);
  cam.zoom = Math.max(minZoom, Math.min(maxZoom, cam.zoom));
  const halfW = n / (2 * cam.zoom);
  const halfH = n / (2 * cam.zoom);
  cam.x = Math.max(halfW, Math.min(n - halfW, cam.x));
  cam.y = Math.max(halfH, Math.min(n - halfH, cam.y));
}

function getCameraView(s) {
  clampCamera(s);
  const n = s.controls.n;
  const cam = s.camera;
  const vw = n / cam.zoom;
  const vh = n / cam.zoom;
  return { x: cam.x - vw * 0.5, y: cam.y - vh * 0.5, w: vw, h: vh };
}

function screenToWorld(s, clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const nx = (clientX - rect.left) / Math.max(1, rect.width);
  const ny = (clientY - rect.top) / Math.max(1, rect.height);
  const v = getCameraView(s);
  return {
    x: v.x + nx * v.w,
    y: v.y + ny * v.h,
  };
}

function worldToScreen(s, wx, wy) {
  const v = getCameraView(s);
  const sx = ((wx - v.x) / v.w) * canvas.width;
  const sy = ((wy - v.y) / v.h) * canvas.height;
  return { x: sx, y: sy };
}

function paintAt(clientX, clientY, erase = false) {
  if (!sim) return;
  const p = screenToWorld(sim, clientX, clientY);
  const x = p.x;
  const y = p.y;
  const view = getCameraView(sim);
  const r = Math.max(1, Number(brushSizeEl.value) || 12) * (view.w / canvas.width);
  const value = erase ? 0.05 : Math.max(0, Math.min(1, Number(paintValueEl.value) || 0.85));

  const minX = Math.max(0, Math.floor(x - r));
  const maxX = Math.min(sim.controls.n - 1, Math.ceil(x + r));
  const minY = Math.max(0, Math.floor(y - r));
  const maxY = Math.min(sim.controls.n - 1, Math.ceil(y + r));

  for (let yy = minY; yy <= maxY; yy++) {
    for (let xx = minX; xx <= maxX; xx++) {
      const dx = xx - x;
      const dy = yy - y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > r) continue;
      const t = 1 - d / r;
      const idx = yy * sim.controls.n + xx;
      const current = sim.viscMapCpu[idx];
      sim.viscMapCpu[idx] = current * (1 - t) + value * t;
    }
  }
  uploadViscMap();
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('mousedown', (e) => {
  if (!sim) return;
  const panIntent = e.button === 1 || e.altKey || (e.button === 2 && !e.shiftKey);
  if (panIntent) {
    panning = true;
    panLastX = e.clientX;
    panLastY = e.clientY;
    return;
  }
  painting = true;
  paintAt(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
window.addEventListener('mouseup', () => { painting = false; panning = false; });
canvas.addEventListener('mousemove', (e) => {
  if (!sim) return;
  if (panning) {
    const dx = e.clientX - panLastX;
    const dy = e.clientY - panLastY;
    panLastX = e.clientX;
    panLastY = e.clientY;
    const view = getCameraView(sim);
    sim.camera.x -= (dx / canvas.width) * view.w;
    sim.camera.y -= (dy / canvas.height) * view.h;
    clampCamera(sim);
    return;
  }
  if (!painting) return;
  paintAt(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});
canvas.addEventListener('wheel', (e) => {
  if (!sim) return;
  e.preventDefault();
  const before = screenToWorld(sim, e.clientX, e.clientY);
  const zoomMul = e.deltaY < 0 ? 1.12 : (1 / 1.12);
  sim.camera.zoom *= zoomMul;
  clampCamera(sim);
  const after = screenToWorld(sim, e.clientX, e.clientY);
  sim.camera.x += before.x - after.x;
  sim.camera.y += before.y - after.y;
  clampCamera(sim);
}, { passive: false });

function sampleFieldBilinear(field, n, x, y) {
  const cx = Math.max(0, Math.min(n - 1.001, x));
  const cy = Math.max(0, Math.min(n - 1.001, y));
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(n - 1, x0 + 1), y1 = Math.min(n - 1, y0 + 1);
  const sx = cx - x0, sy = cy - y0;
  const i00 = y0 * n + x0, i10 = y0 * n + x1, i01 = y1 * n + x0, i11 = y1 * n + x1;

  const v00 = Number(field[i00]);
  const v10 = Number(field[i10]);
  const v01 = Number(field[i01]);
  const v11 = Number(field[i11]);

  const a = (Number.isFinite(v00) ? v00 : 0) * (1 - sx) + (Number.isFinite(v10) ? v10 : 0) * sx;
  const b = (Number.isFinite(v01) ? v01 : 0) * (1 - sx) + (Number.isFinite(v11) ? v11 : 0) * sx;
  const out = a * (1 - sy) + b * sy;
  if (!Number.isFinite(out)) return 0;
  return Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, out));
}

function rigidVerticesWorld(b) {
  if (Array.isArray(b.verticesLocal) && b.verticesLocal.length >= 3) {
    const th = b.theta || 0;
    const c = Math.cos(th), s = Math.sin(th);
    return b.verticesLocal.map((v) => ({
      x: b.x + v.x * c - v.y * s,
      y: b.y + v.x * s + v.y * c,
    }));
  }
  const sides = Math.max(3, b.sides || 3);
  const rot = (b.theta || 0) + (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
  const verts = [];
  for (let i = 0; i < sides; i++) {
    const a = rot + (i / sides) * Math.PI * 2;
    verts.push({ x: b.x + Math.cos(a) * b.r, y: b.y + Math.sin(a) * b.r });
  }
  return verts;
}

function rigidVertexWorld(b, vertexIndex) {
  const verts = rigidVerticesWorld(b);
  const n = verts.length || 1;
  return verts[((vertexIndex % n) + n) % n];
}

function initBodies(n, controls) {
  const scale = n / 256;
  const bigMode = n >= 1024;
  const bodyScale = bigMode ? 0.5 : 1.0;
  const rigidCount = bigMode ? 10 : 2;
  const softClusterCount = bigMode ? 10 : 1;

  const rigidShapeCycle = [3, 4, 5, 6];
  const rigid = [];
  for (let i = 0; i < rigidCount; i++) {
    const t = rigidCount <= 1 ? 0.5 : i / (rigidCount - 1);
    const mass = (i % 2 === 0) ? controls.massLight : controls.massHeavy;
    const sides = rigidShapeCycle[i % rigidShapeCycle.length];
    const r = ((i % 2 === 0) ? 5 : 6) * scale * bodyScale * (sides >= 5 ? 0.95 : 1.0);
    const edgeDyeMode = Array.from({ length: sides }, (_, ei) => {
      const phase = (ei + i) % 3;
      if (phase === 0) return [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.ABSORB];
      if (phase === 1) return [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT];
      return [EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS];
    });
    const edgeBodyMode = Array.from({ length: sides }, (_, ei) => ((ei + i) % 2 === 0) ? EDGE_BODY_MODE.BLOCK : EDGE_BODY_MODE.PASS);
    const edgePermeabilityRGB = Array.from({ length: sides }, () => [0, 0, 0]);
    const digestRGB = (i % 3 === 0) ? [1, 0.2, 0.2] : ((i % 3 === 1) ? [0.2, 1, 0.2] : [0.2, 0.2, 1]);
    const consumeDyeRGB = (i % 2) === 0 ? [1, 1, 1] : [0, 0, 0];
    rigid.push({
      x: n * (0.15 + 0.7 * ((t + Math.random() * 0.1) % 1)),
      y: n * (0.2 + 0.6 * Math.random()),
      vx: 0,
      vy: 0,
      r,
      sides,
      edgeDyeMode,
      edgeBodyMode,
      edgePermeabilityRGB,
      digestEnabled: (i % 2) === 0,
      digestRGB,
      consumeDyeRGB,
      mass,
      theta: Math.random() * Math.PI * 2,
      omega: 0,
      inertia: 0.5 * mass * r * r,
    });
  }

  const softNodes = [];
  const springs = [];
  const softShapeCycle = [3, 4, 6];

  for (let c = 0; c < softClusterCount; c++) {
    const cx = n * (0.18 + 0.64 * Math.random());
    const cy = n * (0.2 + 0.6 * Math.random());
    const nodeCount = softShapeCycle[c % softShapeCycle.length];
    const radius = (8 + (nodeCount === 6 ? 2 : 0)) * scale * bodyScale;
    const base = softNodes.length;

    const local = [];
    for (let i = 0; i < nodeCount; i++) {
      const a = (i / nodeCount) * Math.PI * 2;
      local.push({ x: cx + Math.cos(a) * radius, y: cy + Math.sin(a) * radius });
    }

    const clusterDigestRGB = (c % 3 === 0) ? [1, 0.15, 0.15] : ((c % 3 === 1) ? [0.15, 1, 0.15] : [0.15, 0.15, 1]);
    for (const p of local) {
      softNodes.push({
        x: p.x,
        y: p.y,
        vx: 0,
        vy: 0,
        mass: controls.massSoft,
        r: 1.4 * scale * bodyScale,
        clusterId: c,
        digestEnabled: (c % 2) === 0,
        digestRGB: clusterDigestRGB,
      });
    }

    // Ring springs (solid edges: rigid bodies should collide with them).
    for (let i = 0; i < nodeCount; i++) {
      const j = (i + 1) % nodeCount;
      const a = local[i], b = local[j];
      const ringDyeMode = ((i + c) % 3 === 0)
        ? [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.ABSORB]
        : (((i + c) % 3 === 1)
            ? [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT]
            : [EDGE_DYE_MODE.ABSORB, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.PASS]);
      springs.push([
        base + i,
        base + j,
        Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
        EDGE_BODY_MODE.BLOCK,
        ringDyeMode,
      ]);
    }
    // Cross/diagonal springs for shape retention (internal, non-solid by default).
    for (let i = 0; i < nodeCount; i++) {
      const j = (i + 2) % nodeCount;
      if (i < j || nodeCount <= 4) {
        const a = local[i], b = local[j];
        springs.push([
          base + i,
          base + j,
          Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
          EDGE_BODY_MODE.PASS,
          [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS],
        ]);
      }
    }
    // Opposite braces for even polygons (especially hex) to prevent skew collapse.
    if (nodeCount % 2 === 0) {
      for (let i = 0; i < nodeCount / 2; i++) {
        const j = (i + nodeCount / 2) % nodeCount;
        const a = local[i], b = local[j];
        springs.push([
          base + i,
          base + j,
          Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
          EDGE_BODY_MODE.PASS,
          [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS],
        ]);
      }
    }
    // Hex-specific mirror chords to remove the last skew mode and keep visual symmetry.
    if (nodeCount === 6) {
      const extraPairs = [[0, 4], [1, 5]];
      for (const [i, j] of extraPairs) {
        const a = local[i], b = local[j];
        springs.push([
          base + i,
          base + j,
          Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)),
          EDGE_BODY_MODE.PASS,
          [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS],
        ]);
      }
    }
  }

  const hybrid = [];
  // Minimal hybrid archetype: rigid triangle edge-attached to a soft triangle with one free soft apex.
  const triangleCandidates = rigid
    .map((rb, idx) => ({ rb, idx }))
    .filter(({ rb }) => (rb.sides || 0) === 3)
    .sort((a, b) => {
      const ma = Math.min(a.rb.x, n - a.rb.x, a.rb.y, n - a.rb.y) - a.rb.r;
      const mb = Math.min(b.rb.x, n - b.rb.x, b.rb.y, n - b.rb.y) - b.rb.r;
      return mb - ma;
    });

  if (triangleCandidates.length > 0) {
    const triRigidIndex = triangleCandidates[0].idx;
    const rb = rigid[triRigidIndex];
    const va = rigidVertexWorld(rb, 1);
    const vb = rigidVertexWorld(rb, 2);
    const mx = (va.x + vb.x) * 0.5;
    const my = (va.y + vb.y) * 0.5;
    const ex = vb.x - va.x, ey = vb.y - va.y;
    const el = Math.max(1e-6, Math.hypot(ex, ey));

    // Choose the edge normal that points further into the domain to avoid corner clipping.
    const n1 = { x: -ey / el, y: ex / el };
    const n2 = { x: -n1.x, y: -n1.y };
    const marginScore = (px, py) => Math.min(px, n - px, py, n - py);
    const apexDist = rb.r * 0.95;
    const a1 = { x: mx + n1.x * apexDist, y: my + n1.y * apexDist };
    const a2 = { x: mx + n2.x * apexDist, y: my + n2.y * apexDist };
    const apexRaw = marginScore(a1.x, a1.y) >= marginScore(a2.x, a2.y) ? a1 : a2;
    const margin = rb.r * 0.75;
    const apex = {
      x: Math.max(margin, Math.min(n - margin, apexRaw.x)),
      y: Math.max(margin, Math.min(n - margin, apexRaw.y)),
    };

    const nodeIndex = softNodes.length;
    softNodes.push({
      x: apex.x,
      y: apex.y,
      vx: 0,
      vy: 0,
      mass: controls.massSoft,
      r: 1.4 * scale * bodyScale,
      clusterId: softClusterCount + 1000,
      digestEnabled: false,
      digestRGB: [1, 1, 1],
    });
    hybrid.push({
      rigidIndex: triRigidIndex,
      nodeIndex,
      vertexA: 1,
      vertexB: 2,
      restA: Math.hypot(apex.x - va.x, apex.y - va.y),
      restB: Math.hypot(apex.x - vb.x, apex.y - vb.y),
    });
  }

  return { rigid, soft: { nodes: softNodes, springs }, hybrid };
}

function initEmitters(n) {
  if (n < 1024) return [];
  const count = n >= 2048 ? 16 : 10;
  const palette = [
    [255, 70, 50],
    [50, 170, 255],
    [255, 220, 70],
    [170, 90, 255],
    [80, 255, 170],
  ];
  const emitters = [];
  for (let i = 0; i < count; i++) {
    const c = palette[i % palette.length];
    emitters.push({
      x: n * (0.1 + 0.8 * Math.random()),
      y: n * (0.1 + 0.8 * Math.random()),
      vx: (Math.random() * 2 - 1) * 0.2,
      vy: (Math.random() * 2 - 1) * 0.2,
      r: (n >= 2048 ? 14 : 10) + Math.random() * 6,
      cr: c[0],
      cg: c[1],
      cb: c[2],
      strength: n >= 2048 ? 1.6 : 1.2,
      spin: (Math.random() < 0.5 ? -1 : 1) * (0.45 + Math.random() * 0.55),
      swirlJitter: Math.random() * Math.PI * 2,
    });
  }
  return emitters;
}

function buildPresetEmitters(n, preset) {
  const emitters = [];
  const mk = (xf, yf, vxf, vyf, color, radius, strength, spin = 0.8) => ({
    x: n * xf,
    y: n * yf,
    vx: vxf,
    vy: vyf,
    r: radius,
    cr: color[0],
    cg: color[1],
    cb: color[2],
    strength,
    spin,
    swirlJitter: Math.random() * Math.PI * 2,
  });

  if (preset === 'vortexGarden') {
    const colors = [[255,90,70],[80,170,255],[255,220,90],[190,90,255],[90,255,190]];
    const count = Math.max(8, Math.floor(n / 120));
    for (let i = 0; i < count; i++) {
      const a = (i / count) * Math.PI * 2;
      const rf = 0.23 + (i % 3) * 0.06;
      emitters.push(mk(
        0.5 + Math.cos(a) * rf,
        0.5 + Math.sin(a) * rf,
        -Math.sin(a) * 0.22,
        Math.cos(a) * 0.22,
        colors[i % colors.length],
        Math.max(8, n / 42),
        1.35,
        (i % 2 ? 1 : -1) * 1.0
      ));
    }
  } else if (preset === 'shearCanals') {
    const lanes = 5;
    for (let i = 0; i < lanes; i++) {
      const y = 0.16 + (i / (lanes - 1)) * 0.68;
      const dir = i % 2 === 0 ? 1 : -1;
      emitters.push(mk(0.08, y, 0.36 * dir, 0, [255,150,80], Math.max(7, n / 55), 1.05, 0.55 * dir));
      emitters.push(mk(0.92, y, -0.36 * dir, 0, [90,180,255], Math.max(7, n / 55), 1.05, -0.55 * dir));
    }
  } else if (preset === 'islands') {
    const centers = [[0.22,0.24],[0.78,0.28],[0.30,0.75],[0.75,0.72],[0.52,0.50]];
    const colors = [[255,110,80],[70,160,255],[255,220,90],[180,90,255],[90,255,180]];
    for (let i = 0; i < centers.length; i++) {
      const c = centers[i];
      emitters.push(mk(c[0], c[1], (Math.random()*2-1)*0.12, (Math.random()*2-1)*0.12, colors[i], Math.max(10, n / 35), 1.45, (i%2?1:-1)*0.9));
    }
  } else if (preset === 'checkerPlumes') {
    const cols = 4, rows = 4;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const xf = 0.16 + x * 0.22;
        const yf = 0.16 + y * 0.22;
        const odd = (x + y) % 2 === 1;
        const color = odd ? [255,120,70] : [90,170,255];
        emitters.push(mk(xf, yf, odd ? 0.15 : -0.15, odd ? -0.08 : 0.08, color, Math.max(6, n / 65), 0.95, odd ? 0.7 : -0.7));
      }
    }
  } else {
    return initEmitters(n);
  }
  return emitters;
}

function applyPresetViscosityTerrain(sim, preset) {
  const n = sim.controls.n;
  const m = sim.viscMapCpu;
  m.fill(0.5);
  const addBlob = (xf, yf, rf, target) => {
    const cx = n * xf, cy = n * yf, rr = n * rf;
    const minX = Math.max(0, Math.floor(cx - rr));
    const maxX = Math.min(n - 1, Math.ceil(cx + rr));
    const minY = Math.max(0, Math.floor(cy - rr));
    const maxY = Math.min(n - 1, Math.ceil(cy + rr));
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const d = Math.hypot(x - cx, y - cy);
        if (d > rr) continue;
        const w = 1 - d / rr;
        const i = y * n + x;
        m[i] = m[i] * (1 - w) + target * w;
      }
    }
  };

  if (preset === 'vortexGarden') {
    addBlob(0.5, 0.5, 0.26, 0.15);
    addBlob(0.5, 0.5, 0.42, 0.78);
  } else if (preset === 'shearCanals') {
    for (let i = 0; i < 6; i++) {
      addBlob(0.5, 0.12 + i * 0.15, 0.06, i % 2 === 0 ? 0.2 : 0.82);
    }
  } else if (preset === 'islands') {
    addBlob(0.24, 0.24, 0.16, 0.88);
    addBlob(0.76, 0.30, 0.14, 0.84);
    addBlob(0.30, 0.76, 0.15, 0.86);
    addBlob(0.74, 0.72, 0.13, 0.82);
    addBlob(0.52, 0.50, 0.19, 0.18);
  } else if (preset === 'checkerPlumes') {
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        addBlob(0.1 + x * 0.2, 0.1 + y * 0.2, 0.07, ((x + y) % 2 === 0) ? 0.2 : 0.8);
      }
    }
  }

  uploadViscMap();
}

function cloneMiniBodies(miniBodies, controls) {
  const rigid = Array.isArray(miniBodies?.rigid)
    ? miniBodies.rigid.map((rb) => {
        const sides = Math.max(3, Number(rb?.sides) || 3);
        const edgeDyeMode = Array.isArray(rb?.edgeDyeMode)
          ? rb.edgeDyeMode.map((m) => normalizeEdgeDyeModeRGB(m))
          : Array.from({ length: sides }, () => [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT]);
        const edgeBodyMode = Array.isArray(rb?.edgeBodyMode)
          ? rb.edgeBodyMode.map((m) => Number(m) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK)
          : Array.from({ length: sides }, () => EDGE_BODY_MODE.BLOCK);
        const edgePermeabilityRGB = Array.isArray(rb?.edgePermeabilityRGB)
          ? rb.edgePermeabilityRGB.map((m) => [Number(m?.[0]) > 0 ? 1 : 0, Number(m?.[1]) > 0 ? 1 : 0, Number(m?.[2]) > 0 ? 1 : 0])
          : Array.from({ length: sides }, () => [0, 0, 0]);
        const mass = Math.max(0.05, Number(rb?.mass) || controls.massHeavy || 3.5);
        const r = Math.max(1.0, Number(rb?.r) || 6);
        return {
          x: Number(rb?.x) || 0,
          y: Number(rb?.y) || 0,
          vx: Number(rb?.vx) || 0,
          vy: Number(rb?.vy) || 0,
          r,
          sides,
          theta: Number(rb?.theta) || 0,
          omega: Number(rb?.omega) || 0,
          edgeDyeMode,
          edgeBodyMode,
          edgePermeabilityRGB,
          digestEnabled: Boolean(rb?.digestEnabled),
          digestRGB: Array.isArray(rb?.digestRGB) ? [Number(rb.digestRGB[0]) || 0, Number(rb.digestRGB[1]) || 0, Number(rb.digestRGB[2]) || 0] : [0, 0, 0],
          consumeDyeRGB: Array.isArray(rb?.consumeDyeRGB) ? [Number(rb.consumeDyeRGB[0]) > 0 ? 1 : 0, Number(rb.consumeDyeRGB[1]) > 0 ? 1 : 0, Number(rb.consumeDyeRGB[2]) > 0 ? 1 : 0] : [0, 0, 0],
          mass,
          inertia: Math.max(0.05, Number(rb?.inertia) || (0.5 * mass * r * r)),
          verticesLocal: Array.isArray(rb?.verticesLocal)
            ? rb.verticesLocal.map((v) => ({ x: Number(v?.x) || 0, y: Number(v?.y) || 0 }))
            : undefined,
        };
      })
    : [];

  const softNodes = Array.isArray(miniBodies?.soft?.nodes)
    ? miniBodies.soft.nodes.map((n) => ({
        x: Number(n?.x) || 0,
        y: Number(n?.y) || 0,
        vx: Number(n?.vx) || 0,
        vy: Number(n?.vy) || 0,
        mass: Math.max(0.02, Number(n?.mass) || controls.massSoft || 0.6),
        r: Math.max(0.2, Number(n?.r) || 1.2),
        clusterId: Number.isFinite(Number(n?.clusterId)) ? Number(n.clusterId) : 0,
        digestEnabled: Boolean(n?.digestEnabled),
        digestRGB: Array.isArray(n?.digestRGB) ? [Number(n.digestRGB[0]) || 0, Number(n.digestRGB[1]) || 0, Number(n.digestRGB[2]) || 0] : [0, 0, 0],
      }))
    : [];

  const sanitized = sanitizeSoftSprings(miniBodies?.soft?.springs, softNodes.length, {
    edgeBodyPass: EDGE_BODY_MODE.PASS,
    edgeBodyBlock: EDGE_BODY_MODE.BLOCK,
    restFloor: 1e-3,
  });
  const softSprings = sanitized.springs.map((sp) => [
    Number(sp[0]) | 0,
    Number(sp[1]) | 0,
    Math.max(1e-3, Number(sp[2]) || 1),
    Number(sp[3]) === EDGE_BODY_MODE.PASS ? EDGE_BODY_MODE.PASS : EDGE_BODY_MODE.BLOCK,
    normalizeEdgeDyeModeRGB(sp[4]),
  ]);

  const hybrid = Array.isArray(miniBodies?.hybrid)
    ? miniBodies.hybrid.map((h) => ({
        rigidIndex: Number(h?.rigidIndex) | 0,
        nodeIndex: Number(h?.nodeIndex) | 0,
        vertexA: Number(h?.vertexA) | 0,
        vertexB: Number(h?.vertexB) | 0,
        restA: Math.max(0.8, Number(h?.restA) || 0.8),
        restB: Math.max(0.8, Number(h?.restB) || 0.8),
      }))
    : [];

  return {
    rigid,
    soft: { nodes: softNodes, springs: softSprings },
    hybrid,
    topologyGuardrails: {
      droppedSoftSprings: sanitized.dropped,
    },
  };
}

function getMiniScenarioFromPreset(preset) {
  const p = String(preset || '');
  if (!p.startsWith('mini:')) return null;
  const id = p.slice(5);
  return generatedMiniScenarios.get(id) || null;
}

function ensureGridOption(value) {
  const s = String(value);
  if (![...gridEl.options].some((o) => o.value === s)) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    gridEl.appendChild(opt);
  }
}

function ensureMiniScenarioGridSelection(preset) {
  const mini = getMiniScenarioFromPreset(preset);
  if (!mini) return null;
  const requiredGrid = Math.max(32, Number(mini.grid) || 100);
  ensureGridOption(requiredGrid);
  if (String(gridEl.value) !== String(requiredGrid)) {
    gridEl.value = String(requiredGrid);
  }
  return mini;
}

function applyMiniScenarioPreset(sim, mini) {
  sim.bodies = cloneMiniBodies(mini?.bodies, sim.controls);
  sim.emitters = Array.isArray(mini?.emitters)
    ? mini.emitters.map((e) => ({ ...e }))
    : [];

  sim.frame = 0;
  sim.couplingTelemetry = [];
  sim.lastRigidContacts = [];
  sim.softXPBDLambda = null;
  sim.softAreaRest = null;
  sim.softAreaLambda = null;
  sim.viscMapCpu.fill(0.5);
  uploadViscMap();
  focusCameraOnBodies(sim, sim.bodies);

  log({
    ok: true,
    msg: 'mini scenario loaded',
    id: mini.id,
    combo: mini.combo,
    seed: mini.seed,
    steps: mini.steps,
    grid: mini.grid,
    droppedSoftSprings: sim.bodies?.topologyGuardrails?.droppedSoftSprings || 0,
  });
}

async function loadGeneratedMiniScenarios() {
  if (!scenarioPresetEl) return;
  const previous = scenarioPresetEl.value;
  try {
    const res = await fetch(`${GENERATED_MINI_SCENARIOS_URL}?t=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) return;
    const payload = await res.json();
    const scenarios = Array.isArray(payload?.scenarios) ? payload.scenarios : [];

    for (const opt of [...scenarioPresetEl.querySelectorAll('option[data-generated-mini="1"]')]) {
      opt.remove();
    }

    generatedMiniScenarios = new Map();
    const sorted = [...scenarios]
      .filter((s) => s && s.id && s.bodies)
      .sort((a, b) => String(b.createdAt || '').localeCompare(String(a.createdAt || '')))
      .slice(0, 24);

    for (const s of sorted) {
      const id = String(s.id);
      generatedMiniScenarios.set(id, s);
      const opt = document.createElement('option');
      opt.value = `mini:${id}`;
      opt.dataset.generatedMini = '1';
      const combo = s.combo || 'mixed';
      const steps = Number(s.steps) || 0;
      opt.textContent = `${s.name || `Mini ${id}`} (${combo}, ${steps} steps)`;
      scenarioPresetEl.appendChild(opt);
    }

    if ([...scenarioPresetEl.options].some((o) => o.value === previous)) {
      scenarioPresetEl.value = previous;
    }
  } catch {
    // Optional file; ignore when absent.
  }
}

function applyScenarioPreset(sim, preset) {
  if (!sim) return;
  const p = preset || 'baseline';
  const mini = getMiniScenarioFromPreset(p);
  if (mini) {
    if (Number(mini.grid) !== sim.controls.n) {
      log({
        ok: false,
        msg: 'mini scenario requires matching grid; select preset again to reinit',
        id: mini.id,
        requiredGrid: mini.grid,
        currentGrid: sim.controls.n,
      });
      return;
    }
    applyMiniScenarioPreset(sim, mini);
    return;
  }

  sim.emitters = buildPresetEmitters(sim.controls.n, p);
  applyPresetViscosityTerrain(sim, p);
  sim.camera.x = sim.controls.n * 0.5;
  sim.camera.y = sim.controls.n * 0.5;
  sim.camera.zoom = sim.controls.n >= 1024 ? 1.8 : 1.0;
}

function applyEmitters(sim, r, g, b, vx, vy) {
  const n = sim.controls.n;
  for (const e of sim.emitters || []) {
    e.x += e.vx;
    e.y += e.vy;
    if (e.x < e.r || e.x > n - e.r) e.vx *= -1;
    if (e.y < e.r || e.y > n - e.r) e.vy *= -1;
    e.x = Math.max(e.r, Math.min(n - e.r, e.x));
    e.y = Math.max(e.r, Math.min(n - e.r, e.y));

    const minX = Math.max(0, Math.floor(e.x - e.r));
    const maxX = Math.min(n - 1, Math.ceil(e.x + e.r));
    const minY = Math.max(0, Math.floor(e.y - e.r));
    const maxY = Math.min(n - 1, Math.ceil(e.y + e.r));
    const pulse = 0.75 + 0.25 * Math.sin(sim.frame * 0.03 + e.swirlJitter);
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - e.x;
        const dy = y - e.y;
        const d = Math.hypot(dx, dy);
        if (d > e.r) continue;
        const nd = d / Math.max(1e-6, e.r);
        const w = (1 - nd) * e.strength * pulse;
        const i = y * n + x;
        r[i] = Math.min(255, r[i] + e.cr * 0.03 * w);
        g[i] = Math.min(255, g[i] + e.cg * 0.03 * w);
        b[i] = Math.min(255, b[i] + e.cb * 0.03 * w);

        const tx = d > 1e-6 ? (-dy / d) : 0;
        const ty = d > 1e-6 ? (dx / d) : 0;
        const swirlProfile = (1 - nd) * (0.35 + 0.65 * nd);
        const swirl = e.spin * 0.11 * swirlProfile;
        vx[i] += (e.vx * 0.028 + tx * swirl) * w;
        vy[i] += (e.vy * 0.028 + ty * swirl) * w;
      }
    }
  }
}

function computeSoftCentroid(nodes) {
  let sx = 0;
  let sy = 0;
  for (const node of nodes) {
    sx += node.x;
    sy += node.y;
  }
  const inv = nodes.length > 0 ? (1 / nodes.length) : 0;
  return { x: sx * inv, y: sy * inv };
}

function applyBounceBoundary(body, n, damping = 0.82) {
  const minX = body.r;
  const maxX = n - body.r;
  const minY = body.r;
  const maxY = n - body.r;

  if (body.x < minX) {
    body.x = minX;
    if (body.vx < 0) body.vx = -body.vx * damping;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  } else if (body.x > maxX) {
    body.x = maxX;
    if (body.vx > 0) body.vx = -body.vx * damping;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  }

  if (body.y < minY) {
    body.y = minY;
    if (body.vy < 0) body.vy = -body.vy * damping;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  } else if (body.y > maxY) {
    body.y = maxY;
    if (body.vy > 0) body.vy = -body.vy * damping;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  }
}

function resolveCircleCollision(a, b, restitution = 0.35) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const d2 = dx * dx + dy * dy;
  const minDist = (a.r || 1) + (b.r || 1);
  if (d2 <= 1e-10) {
    const jitter = 0.01;
    a.x -= jitter; b.x += jitter;
    return;
  }
  if (d2 >= minDist * minDist) return;

  const d = Math.sqrt(d2);
  const nx = dx / d;
  const ny = dy / d;
  const penetration = minDist - d;

  const ma = Math.max(0.02, a.mass || 1);
  const mb = Math.max(0.02, b.mass || 1);
  const invA = 1 / ma;
  const invB = 1 / mb;
  const invSum = invA + invB;

  // Positional correction
  const corr = (penetration / Math.max(1e-6, invSum)) * 0.85;
  a.x -= nx * corr * invA;
  a.y -= ny * corr * invA;
  b.x += nx * corr * invB;
  b.y += ny * corr * invB;

  // Impulse resolution
  const rvx = b.vx - a.vx;
  const rvy = b.vy - a.vy;
  const vn = rvx * nx + rvy * ny;
  if (vn > 0) return;
  const j = (-(1 + restitution) * vn) / Math.max(1e-6, invSum);
  const ix = j * nx;
  const iy = j * ny;
  a.vx -= ix * invA;
  a.vy -= iy * invA;
  b.vx += ix * invB;
  b.vy += iy * invB;
}

function closestPointOnSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const ab2 = abx * abx + aby * aby;
  if (ab2 < 1e-8) return { t: 0, x: ax, y: ay, abx, aby, ab2 };
  const apx = px - ax;
  const apy = py - ay;
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / ab2));
  return { t, x: ax + abx * t, y: ay + aby * t, abx, aby, ab2 };
}

function resolveRigidVsSoftEdgeCollision(rigid, a, b, restitution = 0.28) {
  const cp = closestPointOnSegment(rigid.x, rigid.y, a.x, a.y, b.x, b.y);
  let nx = rigid.x - cp.x;
  let ny = rigid.y - cp.y;
  let dist = Math.hypot(nx, ny);
  const minDist = Math.max(0.8, rigid.r || 1);
  if (dist >= minDist) return;

  if (dist < 1e-6) {
    const invLen = 1 / Math.max(1e-6, Math.hypot(-cp.aby, cp.abx));
    nx = -cp.aby * invLen;
    ny = cp.abx * invLen;
    dist = 1e-6;
  } else {
    nx /= dist;
    ny /= dist;
  }

  const penetration = minDist - dist;
  rigid.x += nx * penetration * 0.92;
  rigid.y += ny * penetration * 0.92;

  const edgeVx = (a.vx + b.vx) * 0.5;
  const edgeVy = (a.vy + b.vy) * 0.5;
  const rvx = rigid.vx - edgeVx;
  const rvy = rigid.vy - edgeVy;
  const vn = rvx * nx + rvy * ny;
  if (vn < 0) {
    const j = -(1 + restitution) * vn;
    rigid.vx += nx * j;
    rigid.vy += ny * j;
  }
}

function resolveSoftNodeVsSoftEdgeCollision(node, a, b, restitution = 0.12) {
  const cp = closestPointOnSegment(node.x, node.y, a.x, a.y, b.x, b.y);
  let nx = node.x - cp.x;
  let ny = node.y - cp.y;
  let dist = Math.hypot(nx, ny);
  const minDist = Math.max(0.4, node.r || 1.0);
  if (dist >= minDist) return;

  if (dist < 1e-6) {
    const invLen = 1 / Math.max(1e-6, Math.hypot(-cp.aby, cp.abx));
    nx = -cp.aby * invLen;
    ny = cp.abx * invLen;
    dist = 1e-6;
  } else {
    nx /= dist;
    ny /= dist;
  }

  const penetration = minDist - dist;
  node.x += nx * penetration * 0.92;
  node.y += ny * penetration * 0.92;
  a.x -= nx * penetration * 0.04;
  a.y -= ny * penetration * 0.04;
  b.x -= nx * penetration * 0.04;
  b.y -= ny * penetration * 0.04;

  const edgeVx = (a.vx + b.vx) * 0.5;
  const edgeVy = (a.vy + b.vy) * 0.5;
  const rvx = node.vx - edgeVx;
  const rvy = node.vy - edgeVy;
  const vn = rvx * nx + rvy * ny;
  if (vn < 0) {
    const j = -(1 + restitution) * vn;
    node.vx += nx * j;
    node.vy += ny * j;
  }
}

function enforceFluidEdgeBoundariesCpu(vxField, vyField, n) {
  const last = n - 1;
  for (let x = 0; x < n; x++) {
    const top = x;
    const bottom = last * n + x;
    vyField[top] = 0;
    vyField[bottom] = 0;
    vxField[top] *= 0.7;
    vxField[bottom] *= 0.7;
  }
  for (let y = 0; y < n; y++) {
    const left = y * n;
    const right = y * n + last;
    vxField[left] = 0;
    vxField[right] = 0;
    vyField[left] *= 0.7;
    vyField[right] *= 0.7;
  }
}

function applySoftSpringsXPBDVelocity(s, dtPos, stiffnessScale, lambdaCache) {
  if (!s?.nodes?.length || !s?.springs?.length) return;
  const alpha = (SOFT_XPBD_BASE_COMPLIANCE / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < SOFT_XPBD_ITERS; iter++) {
    for (let si = 0; si < s.springs.length; si++) {
      const [i, j, rest] = s.springs[si];
      const a = s.nodes[i];
      const b = s.nodes[j];
      if (!a || !b) continue;

      const ax = a.x + a.vx * dtPos;
      const ay = a.y + a.vy * dtPos;
      const bx = b.x + b.vx * dtPos;
      const by = b.y + b.vy * dtPos;

      const dx = bx - ax;
      const dy = by - ay;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = dx / d;
      const ny = dy / d;
      const C = d - rest;

      const wA = 1 / Math.max(0.02, a.mass || 1);
      const wB = 1 / Math.max(0.02, b.mass || 1);
      const wSum = wA + wB;
      if (wSum <= 1e-9) continue;

      const lambdaPrev = Number(lambdaCache[si]) || 0;
      let dl = (-C - alpha * lambdaPrev) / (wSum + alpha);
      if (!Number.isFinite(dl)) continue;
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      lambdaCache[si] = lambdaNext;

      const corrAx = -wA * dl * nx;
      const corrAy = -wA * dl * ny;
      const corrBx = wB * dl * nx;
      const corrBy = wB * dl * ny;

      a.vx += corrAx / dtPos;
      a.vy += corrAy / dtPos;
      b.vx += corrBx / dtPos;
      b.vy += corrBy / dtPos;
    }
  }
}

function signedAreaPredicted(nodes, indices, dtPos) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    const ax = a.x + a.vx * dtPos;
    const ay = a.y + a.vy * dtPos;
    const bx = b.x + b.vx * dtPos;
    const by = b.y + b.vy * dtPos;
    s += ax * by - bx * ay;
  }
  return 0.5 * s;
}

function ensureSoftAreaRestState(sim, s, loops, dtPos) {
  sim.softAreaRest = sim.softAreaRest || new Map();
  sim.softAreaLambda = sim.softAreaLambda || new Map();

  const live = new Set(loops.map((l) => l.clusterId));
  for (const key of sim.softAreaRest.keys()) if (!live.has(key)) sim.softAreaRest.delete(key);
  for (const key of sim.softAreaLambda.keys()) if (!live.has(key)) sim.softAreaLambda.delete(key);

  for (const loop of loops) {
    if (!sim.softAreaRest.has(loop.clusterId)) {
      const a0 = signedAreaPredicted(s.nodes, loop.indices, dtPos);
      sim.softAreaRest.set(loop.clusterId, Math.abs(a0) > 1e-4 ? a0 : 1e-4);
    }
    const prev = Number(sim.softAreaLambda.get(loop.clusterId));
    sim.softAreaLambda.set(loop.clusterId, Number.isFinite(prev) ? Math.max(-20, Math.min(20, prev)) : 0);
  }
}

function applySoftAreaXPBDVelocity(sim, s, loops, dtPos, stiffnessScale) {
  if (!loops.length) return;
  const alpha = (SOFT_AREA_BASE_COMPLIANCE / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < SOFT_AREA_XPBD_ITERS; iter++) {
    for (const loop of loops) {
      const ids = loop.indices;
      const m = ids.length;
      if (m < 3) continue;
      const restArea = sim.softAreaRest.get(loop.clusterId);
      if (!Number.isFinite(restArea)) continue;

      const area = signedAreaPredicted(s.nodes, ids, dtPos);
      const C = area - restArea;

      const gradX = new Array(m);
      const gradY = new Array(m);
      let sumWGrad2 = 0;

      for (let k = 0; k < m; k++) {
        const prev = s.nodes[ids[(k - 1 + m) % m]];
        const next = s.nodes[ids[(k + 1) % m]];
        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nx = next.x + next.vx * dtPos;
        const ny = next.y + next.vy * dtPos;
        const gx = 0.5 * (ny - py);
        const gy = 0.5 * (px - nx);
        gradX[k] = gx;
        gradY[k] = gy;

        const node = s.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        sumWGrad2 += w * (gx * gx + gy * gy);
      }

      if (sumWGrad2 <= 1e-10) continue;

      const lambdaPrev = Number(sim.softAreaLambda.get(loop.clusterId)) || 0;
      let dl = (-C - alpha * lambdaPrev) / (sumWGrad2 + alpha);
      if (!Number.isFinite(dl)) continue;
      dl = Math.max(-2.0, Math.min(2.0, dl));
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      sim.softAreaLambda.set(loop.clusterId, lambdaNext);

      for (let k = 0; k < m; k++) {
        const node = s.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        node.vx += (w * gradX[k] * dl) / dtPos;
        node.vy += (w * gradY[k] * dl) / dtPos;
      }
    }
  }
}

function stepBodiesAndInject(sim, vxField, vyField) {
  const n = sim.controls.n;
  const dt = sim.controls.dt;
  const dtNorm = Math.max(0.2, Math.min(1.5, (dt * 60) || 1));
  const bodies = sim.bodies;
  const dragK = sim.controls.bodyDrag;
  const feedbackK = sim.controls.bodyFeedback;
  const viscMap = sim.viscMapCpu;

  const localHoneyDrag = (x, y) => {
    const v = sampleFieldBilinear(viscMap, n, x, y);
    // Unified viscosity response curve for both rigid and soft bodies.
    return 1.0 + Math.pow(Math.max(0, Math.min(1, v)), 2.2) * 14.0;
  };
  const viscosityMotionResponse = (honey, vmaxBase) => ({
    damp: Math.max(0.72, 1.0 - 0.018 * honey),
    vmax: Math.max(0.4, vmaxBase / (1 + 0.28 * honey)),
  });

  if (bodies.rigid[0]) bodies.rigid[0].mass = sim.controls.massLight;
  if (bodies.rigid[1]) bodies.rigid[1].mass = sim.controls.massHeavy;
  for (const rb of bodies.rigid) {
    rb.inertia = 0.5 * rb.mass * rb.r * rb.r;
  }
  for (const node of bodies.soft.nodes) node.mass = sim.controls.massSoft;

  const softCentroidBefore = computeSoftCentroid(bodies.soft.nodes);
  const computeRigidCenter = (arr) => {
    if (!arr.length) return { x: 0, y: 0 };
    let sx = 0, sy = 0;
    for (const r of arr) { sx += r.x; sy += r.y; }
    return { x: sx / arr.length, y: sy / arr.length };
  };
  const rigidCenterBefore = computeRigidCenter(bodies.rigid);

  let rigidCarryTransfer = 0;
  let softCarryTransfer = 0;

  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const invMass = 1 / Math.max(0.05, b.mass);
    const invInertia = 1 / Math.max(0.05, b.inertia || 1);
    const sampleVerts = rigidVerticesWorld(b);
    const sampleCount = Math.max(1, sampleVerts.length);
    let forceX = 0;
    let forceY = 0;
    let torque = 0;

    for (let si = 0; si < sampleCount; si++) {
      const sx = sampleVerts[si].x;
      const sy = sampleVerts[si].y;
      const rx = sx - b.x;
      const ry = sy - b.y;
      const fx = sampleFieldBilinear(vxField, n, sx, sy);
      const fy = sampleFieldBilinear(vyField, n, sx, sy);
      const localVx = b.vx + (-(b.omega || 0) * ry);
      const localVy = b.vy + ((b.omega || 0) * rx);
      const relX = fx - localVx;
      const relY = fy - localVy;
      const honey = localHoneyDrag(sx, sy);
      const fpx = relX * dragK * honey;
      const fpy = relY * dragK * honey;
      forceX += fpx;
      forceY += fpy;
      torque += rx * fpy - ry * fpx;
    }

    forceX /= sampleCount;
    forceY /= sampleCount;
    torque /= sampleCount;

    const ax = forceX * invMass;
    const ay = forceY * invMass;
    const alpha = torque * invInertia;

    const swimPhase = sim.frame * 0.08 + bi * 2.1;
    const swimX = Math.cos(swimPhase) * 0.012 * invMass;
    const swimY = Math.sin(swimPhase * 1.6) * 0.009 * invMass;
    const swimTorque = Math.sin(swimPhase * 1.1) * 0.0025;

    b.vx += ax * dt * 60 + swimX * dtNorm;
    b.vy += ay * dt * 60 + swimY * dtNorm;
    b.omega = (b.omega || 0) + alpha * dt * 60 + swimTorque * dtNorm;

    const centerHoney = localHoneyDrag(b.x, b.y);
    const rigidVisc = viscosityMotionResponse(centerHoney, 3.2);
    b.vx *= rigidVisc.damp;
    b.vy *= rigidVisc.damp;
    b.omega *= Math.max(0.72, 0.99 - 0.01 * centerHoney);

    const bMax = rigidVisc.vmax;
    const bMag = Math.hypot(b.vx, b.vy);
    if (bMag > bMax) {
      b.vx = (b.vx / bMag) * bMax;
      b.vy = (b.vy / bMag) * bMax;
    }
    b.omega = Math.max(-0.25, Math.min(0.25, b.omega));

    rigidCarryTransfer += Math.hypot(ax, ay);
    b.x = b.x + b.vx * dt * 22;
    b.y = b.y + b.vy * dt * 28;
    b.theta = (b.theta || 0) + b.omega * dt * 60;
    applyBounceBoundary(b, n, 0.84);
  }

  const s = bodies.soft;
  const softCentroid = computeSoftCentroid(s.nodes);
  for (let i = 0; i < s.nodes.length; i++) {
    const node = s.nodes[i];
    const fx = sampleFieldBilinear(vxField, n, node.x, node.y);
    const fy = sampleFieldBilinear(vyField, n, node.x, node.y);
    const invMass = 1 / Math.max(0.02, node.mass);
    const cx = node.x - softCentroid.x;
    const cy = node.y - softCentroid.y;
    const activeSwimPhase = sim.frame * 0.12 + i * 1.57;
    const activeSwimAmp = 0.008 * (1 + 0.2 * Math.sin(sim.frame * 0.05 + i));
    const swimX = (-cy * activeSwimAmp + Math.cos(activeSwimPhase) * 0.004) * invMass;
    const swimY = (cx * activeSwimAmp + Math.sin(activeSwimPhase) * 0.004) * invMass;
    const honey = localHoneyDrag(node.x, node.y);
    const carryX = (fx - node.vx) * dragK * honey * invMass;
    const carryY = (fy - node.vy) * dragK * honey * invMass;
    node.vx += carryX * dt * 60 + swimX * dtNorm;
    node.vy += carryY * dt * 60 + swimY * dtNorm;
    const softVisc = viscosityMotionResponse(honey, 2.8);
    node.vx *= softVisc.damp;
    node.vy *= softVisc.damp;
    const nMax = softVisc.vmax;
    const nMag = Math.hypot(node.vx, node.vy);
    if (nMag > nMax) {
      node.vx = (node.vx / nMag) * nMax;
      node.vy = (node.vy / nMag) * nMax;
    }
    softCarryTransfer += Math.hypot(carryX, carryY);
  }

  const dtPos = Math.max(1e-4, dt * SOFT_INTEGRATION_SCALE);
  if (!sim.softXPBDLambda || sim.softXPBDLambda.length !== s.springs.length) {
    sim.softXPBDLambda = ensureLambdaCacheSize(sim.softXPBDLambda, s.springs.length, 20);
  }
  applySoftSpringsXPBDVelocity(s, dtPos, SOFT_SPRING_STIFFNESS_DEFAULT, sim.softXPBDLambda);

  const softClusterLoops = buildSoftClusterBoundaryLoops(s.nodes, s.springs, {
    blockMode: EDGE_BODY_MODE.BLOCK,
  });
  ensureSoftAreaRestState(sim, s, softClusterLoops, dtPos);
  applySoftAreaXPBDVelocity(sim, s, softClusterLoops, dtPos, SOFT_SPRING_STIFFNESS_DEFAULT);

  for (let iter = 0; iter < 5; iter++) {
    // Hybrid rigid-soft attachment constraints (weld-like springs to rigid edge vertices).
    for (const h of (bodies.hybrid || [])) {
      const rb = bodies.rigid[h.rigidIndex];
      const node = s.nodes[h.nodeIndex];
      if (!rb || !node) continue;
      const va = rigidVertexWorld(rb, h.vertexA);
      const vb = rigidVertexWorld(rb, h.vertexB);
      const pairs = [[va, h.restA], [vb, h.restB]];
      for (const [anchor, rest] of pairs) {
        const dx = node.x - anchor.x;
        const dy = node.y - anchor.y;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const err = (d - rest) * 0.74;
        const nx = dx / d, ny = dy / d;
        node.vx -= nx * err * 0.052 * dtNorm;
        node.vy -= ny * err * 0.052 * dtNorm;
        // Matched, softer reaction into rigid body to avoid hybrid jitter.
        rb.vx += nx * err * 0.0075 * dtNorm;
        rb.vy += ny * err * 0.0075 * dtNorm;
        rb.omega = (rb.omega || 0) + (nx * ny) * err * 0.00075 * dtNorm;
      }
    }
  }

  const hybridNodeVCap = 3.2;
  for (const node of s.nodes) {
    const vmag = Math.hypot(node.vx, node.vy);
    if (vmag > hybridNodeVCap) {
      node.vx = (node.vx / vmag) * hybridNodeVCap;
      node.vy = (node.vy / vmag) * hybridNodeVCap;
    }
    node.x = node.x + node.vx * dt * SOFT_INTEGRATION_SCALE;
    node.y = node.y + node.vy * dt * SOFT_INTEGRATION_SCALE;
    applyBounceBoundary(node, n, 0.78);
  }

  for (const rb of bodies.rigid) {
    const vmag = Math.hypot(rb.vx, rb.vy);
    const vcap = 4.0;
    if (vmag > vcap) {
      rb.vx = (rb.vx / vmag) * vcap;
      rb.vy = (rb.vy / vmag) * vcap;
    }
    rb.omega = Math.max(-0.22, Math.min(0.22, rb.omega || 0));
  }

  const rigidContactDebug = [];
  const hybridAttachedByRigid = new Map();
  for (const h of (bodies.hybrid || [])) {
    const ri = Number(h?.rigidIndex) | 0;
    const ni = Number(h?.nodeIndex) | 0;
    if (ri < 0 || ri >= bodies.rigid.length) continue;
    if (ni < 0 || ni >= s.nodes.length) continue;
    if (!hybridAttachedByRigid.has(ri)) hybridAttachedByRigid.set(ri, new Set());
    hybridAttachedByRigid.get(ri).add(ni);
  }

  // Body-body collisions: rigid↔rigid, rigid↔soft, soft↔soft
  for (let iter = 0; iter < 2; iter++) {
    for (let i = 0; i < bodies.rigid.length; i++) {
      for (let j = i + 1; j < bodies.rigid.length; j++) {
        resolveRigidVsRigidPolygonCollision(bodies.rigid[i], bodies.rigid[j], 0.32, {
          contacts: rigidContactDebug,
          aIndex: i,
          bIndex: j,
          iter,
          phase: 'pre-soft',
        });
      }
    }
    for (let rbi = 0; rbi < bodies.rigid.length; rbi++) {
      const rb = bodies.rigid[rbi];
      const attachedNodeSet = hybridAttachedByRigid.get(rbi) || null;
      for (let ni = 0; ni < s.nodes.length; ni++) {
        if (attachedNodeSet && attachedNodeSet.has(ni)) continue; // avoid parent rigid fighting its own hybrid-attached node
        const sn = s.nodes[ni];
        // Recompute rigid polygon from latest body state per-contact;
        // stale hull snapshots caused missed/odd contacts after position updates.
        resolveRigidVsSoftNodeCollision(rb, sn, null, 0.18);
      }
      for (const [i, j, _rest, edgeBodyMode] of s.springs) {
        if (edgeBodyMode !== EDGE_BODY_MODE.BLOCK) continue;
        if (attachedNodeSet && (attachedNodeSet.has(i) || attachedNodeSet.has(j))) continue;
        resolveRigidVsSoftEdgeCollision(rb, s.nodes[i], s.nodes[j], 0.16);
      }
    }
    for (let i = 0; i < s.nodes.length; i++) {
      for (let j = i + 1; j < s.nodes.length; j++) {
        resolveCircleCollision(s.nodes[i], s.nodes[j], 0.22);
      }
    }
    // Soft-node vs foreign soft-edge blocking for solid edges.
    for (let ni = 0; ni < s.nodes.length; ni++) {
      const node = s.nodes[ni];
      for (const [i, j, _rest, edgeBodyMode] of s.springs) {
        if (edgeBodyMode !== EDGE_BODY_MODE.BLOCK) continue;
        if (i === ni || j === ni) continue;
        const a = s.nodes[i], b = s.nodes[j];
        if (a.clusterId === node.clusterId && b.clusterId === node.clusterId) continue;
        resolveSoftNodeVsSoftEdgeCollision(node, a, b, 0.12);
      }
    }

    // Re-run rigid-rigid contacts after rigid-soft pushes to avoid late interpenetration.
    for (let i = 0; i < bodies.rigid.length; i++) {
      for (let j = i + 1; j < bodies.rigid.length; j++) {
        resolveRigidVsRigidPolygonCollision(bodies.rigid[i], bodies.rigid[j], 0.32, {
          contacts: rigidContactDebug,
          aIndex: i,
          bIndex: j,
          iter,
          phase: 'post-soft',
        });
      }
    }

    for (const rb of bodies.rigid) applyBounceBoundary(rb, n, 0.84);
    for (const sn of s.nodes) applyBounceBoundary(sn, n, 0.78);
  }

  sim.lastRigidContacts = rigidContactDebug.length > 64 ? rigidContactDebug.slice(0, 64) : rigidContactDebug;

  let injectedMomentum = 0;
  const injectPoint = (px, py, pvx, pvy, localFluidX, localFluidY, mass, rad=3.0, swimInjectX = 0, swimInjectY = 0) => {
    if (!Number.isFinite(px) || !Number.isFinite(py)) return;
    const radius = Math.max(0.4, Number.isFinite(rad) ? rad : 3.0);
    const minX = Math.max(0, Math.floor(px - radius));
    const maxX = Math.min(n - 1, Math.ceil(px + radius));
    const minY = Math.max(0, Math.floor(py - radius));
    const maxY = Math.min(n - 1, Math.ceil(py + radius));
    const relXRaw = pvx - localFluidX + swimInjectX;
    const relYRaw = pvy - localFluidY + swimInjectY;
    const relX = Number.isFinite(relXRaw) ? Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, relXRaw)) : 0;
    const relY = Number.isFinite(relYRaw) ? Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, relYRaw)) : 0;
    const scaleRaw = feedbackK * Math.max(0.1, Number.isFinite(mass) ? mass : 0.1);
    const scale = Number.isFinite(scaleRaw) ? Math.max(0, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, scaleRaw)) : 0;
    if (scale <= 0) return;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - px, dy = y - py;
        const d = Math.hypot(dx, dy);
        if (d > radius) continue;
        const w = 1 - d / radius;
        const idx = y * n + x;
        const jxRaw = relX * scale * w;
        const jyRaw = relY * scale * w;
        const jx = Number.isFinite(jxRaw) ? Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, jxRaw)) : 0;
        const jy = Number.isFinite(jyRaw) ? Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, jyRaw)) : 0;
        const nextVx = Number(vxField[idx]) + jx;
        const nextVy = Number(vyField[idx]) + jy;
        vxField[idx] = Number.isFinite(nextVx) ? Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, nextVx)) : 0;
        vyField[idx] = Number.isFinite(nextVy) ? Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, nextVy)) : 0;
        injectedMomentum += Math.hypot(jx, jy);
      }
    }
  };

  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const fx = sampleFieldBilinear(vxField, n, b.x, b.y);
    const fy = sampleFieldBilinear(vyField, n, b.x, b.y);
    const swimPhase = sim.frame * 0.08 + bi * 2.1;
    injectPoint(b.x, b.y, b.vx, b.vy, fx, fy, b.mass, b.r * 0.8, Math.cos(swimPhase) * 0.015, Math.sin(swimPhase) * 0.012);
  }
  for (let i = 0; i < s.nodes.length; i++) {
    const node = s.nodes[i];
    const fx = sampleFieldBilinear(vxField, n, node.x, node.y);
    const fy = sampleFieldBilinear(vyField, n, node.x, node.y);
    const swimPhase = sim.frame * 0.12 + i * 1.57;
    injectPoint(node.x, node.y, node.vx, node.vy, fx, fy, node.mass, 2.2, Math.cos(swimPhase) * 0.01, Math.sin(swimPhase) * 0.01);
  }

  const softCentroidAfter = computeSoftCentroid(s.nodes);
  const rigidCenterAfter = computeRigidCenter(bodies.rigid);

  const metrics = {
    rigidCenterDelta: Math.hypot(rigidCenterAfter.x - rigidCenterBefore.x, rigidCenterAfter.y - rigidCenterBefore.y),
    softCentroidDelta: Math.hypot(softCentroidAfter.x - softCentroidBefore.x, softCentroidAfter.y - softCentroidBefore.y),
    rigidCarryTransfer,
    softCarryTransfer,
    injectedMomentum,
  };
  sim.couplingTelemetry = sim.couplingTelemetry || [];
  sim.couplingTelemetry.push(metrics);
  if (sim.couplingTelemetry.length > 120) sim.couplingTelemetry.shift();
  return metrics;
}


function summarizeCouplingTelemetry(telemetry) {
  if (!Array.isArray(telemetry) || telemetry.length === 0) {
    return {
      rigidCenterDeltaAvg: 0,
      softCentroidDeltaAvg: 0,
      rigidCarryTransferAvg: 0,
      softCarryTransferAvg: 0,
      injectedMomentumAvg: 0,
    };
  }
  const acc = { rigidCenterDelta: 0, softCentroidDelta: 0, rigidCarryTransfer: 0, softCarryTransfer: 0, injectedMomentum: 0 };
  for (const t of telemetry) {
    acc.rigidCenterDelta += t.rigidCenterDelta || 0;
    acc.softCentroidDelta += t.softCentroidDelta || 0;
    acc.rigidCarryTransfer += t.rigidCarryTransfer || 0;
    acc.softCarryTransfer += t.softCarryTransfer || 0;
    acc.injectedMomentum += t.injectedMomentum || 0;
  }
  const k = 1 / telemetry.length;
  return {
    rigidCenterDeltaAvg: +(acc.rigidCenterDelta * k).toFixed(4),
    softCentroidDeltaAvg: +(acc.softCentroidDelta * k).toFixed(4),
    rigidCarryTransferAvg: +(acc.rigidCarryTransfer * k).toFixed(4),
    softCarryTransferAvg: +(acc.softCarryTransfer * k).toFixed(4),
    injectedMomentumAvg: +(acc.injectedMomentum * k).toFixed(4),
  };
}

function pointInPolygon(x, y, verts) {
  let inside = false;
  for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
    const xi = verts[i].x, yi = verts[i].y;
    const xj = verts[j].x, yj = verts[j].y;
    const intersect = ((yi > y) !== (yj > y)) &&
      (x < ((xj - xi) * (y - yi)) / Math.max(1e-9, (yj - yi)) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function normalizeBinaryRGB(v) {
  if (!Array.isArray(v) || v.length < 3) return [0, 0, 0];
  return [Number(v[0]) > 0 ? 1 : 0, Number(v[1]) > 0 ? 1 : 0, Number(v[2]) > 0 ? 1 : 0];
}

function applyDigestiveCapture(sim, r, g, b) {
  const n = sim.controls.n;
  const rigidCapture = 0.055;
  const softCapture = 0.045;
  let captured = 0;

  for (const rb of sim.bodies.rigid) {
    const consumeMask = normalizeBinaryRGB(rb.consumeDyeRGB || (rb.digestEnabled ? [1, 1, 1] : [0, 0, 0]));
    if (consumeMask[0] === 0 && consumeMask[1] === 0 && consumeMask[2] === 0) continue;
    const dig = rb.digestRGB || [1, 1, 1];
    const verts = rigidVerticesWorld(rb);
    let minX = n - 1, minY = n - 1, maxX = 0, maxY = 0;
    for (const v of verts) {
      minX = Math.min(minX, v.x); minY = Math.min(minY, v.y);
      maxX = Math.max(maxX, v.x); maxY = Math.max(maxY, v.y);
    }
    minX = Math.max(0, Math.floor(minX)); minY = Math.max(0, Math.floor(minY));
    maxX = Math.min(n - 1, Math.ceil(maxX)); maxY = Math.min(n - 1, Math.ceil(maxY));

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (!pointInPolygon(x + 0.5, y + 0.5, verts)) continue;
        const i = y * n + x;
        const takeR = r[i] * rigidCapture * dig[0] * consumeMask[0];
        const takeG = g[i] * rigidCapture * dig[1] * consumeMask[1];
        const takeB = b[i] * rigidCapture * dig[2] * consumeMask[2];
        r[i] -= takeR; g[i] -= takeG; b[i] -= takeB;
        captured += takeR + takeG + takeB;
      }
    }
  }

  const clusters = new Map();
  for (const node of sim.bodies.soft.nodes) {
    const id = node.clusterId ?? 0;
    if (!clusters.has(id)) clusters.set(id, []);
    clusters.get(id).push(node);
  }
  for (const nodes of clusters.values()) {
    if (nodes.length < 3) continue;
    if (!nodes[0].digestEnabled) continue;
    const dig = nodes[0].digestRGB || [1, 1, 1];
    let cx = 0, cy = 0;
    for (const n0 of nodes) { cx += n0.x; cy += n0.y; }
    cx /= nodes.length; cy /= nodes.length;
    const verts = [...nodes]
      .map((p) => ({ x: p.x, y: p.y, a: Math.atan2(p.y - cy, p.x - cx) }))
      .sort((a, b2) => a.a - b2.a)
      .map(({ x, y }) => ({ x, y }));

    let minX = n - 1, minY = n - 1, maxX = 0, maxY = 0;
    for (const v of verts) {
      minX = Math.min(minX, v.x); minY = Math.min(minY, v.y);
      maxX = Math.max(maxX, v.x); maxY = Math.max(maxY, v.y);
    }
    minX = Math.max(0, Math.floor(minX)); minY = Math.max(0, Math.floor(minY));
    maxX = Math.min(n - 1, Math.ceil(maxX)); maxY = Math.min(n - 1, Math.ceil(maxY));

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (!pointInPolygon(x + 0.5, y + 0.5, verts)) continue;
        const i = y * n + x;
        const takeR = r[i] * softCapture * dig[0];
        const takeG = g[i] * softCapture * dig[1];
        const takeB = b[i] * softCapture * dig[2];
        r[i] -= takeR; g[i] -= takeG; b[i] -= takeB;
        captured += takeR + takeG + takeB;
      }
    }
  }

  sim.digestiveCapture = (sim.digestiveCapture || 0) * 0.97 + captured * 0.03;
}

function edgeModeColor(modeRGB, blocked) {
  const m = normalizeEdgeDyeModeRGB(modeRGB);
  const key = `${m[0]}-${m[1]}-${m[2]}`;
  if (!blocked) return 'rgba(90,180,190,0.35)';
  if (key === `${EDGE_DYE_MODE.PASS}-${EDGE_DYE_MODE.PASS}-${EDGE_DYE_MODE.PASS}`) return 'rgba(120,150,255,0.95)';
  if (key === `${EDGE_DYE_MODE.DEFLECT}-${EDGE_DYE_MODE.DEFLECT}-${EDGE_DYE_MODE.DEFLECT}`) return '#ffffff';
  if (key === `${EDGE_DYE_MODE.ABSORB}-${EDGE_DYE_MODE.ABSORB}-${EDGE_DYE_MODE.ABSORB}`) return 'rgba(255,180,70,0.95)';
  return 'rgba(210,120,255,0.95)';
}

function drawRegularPolygon(cx, cy, radius, sides, rotation = 0) {
  const n = Math.max(3, sides | 0);
  for (let i = 0; i < n; i++) {
    const a = rotation + (i / n) * Math.PI * 2;
    const x = cx + Math.cos(a) * radius;
    const y = cy + Math.sin(a) * radius;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

function getBodiesBounds(bodies) {
  if (!bodies) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  const includePoint = (x, y) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  };

  for (const rb of bodies.rigid || []) {
    const verts = rigidVerticesWorld(rb);
    for (const p of verts) includePoint(p.x, p.y);
  }
  for (const n of bodies.soft?.nodes || []) includePoint(n.x, n.y);

  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) return null;
  return { minX, minY, maxX, maxY, cx: (minX + maxX) * 0.5, cy: (minY + maxY) * 0.5 };
}

function focusCameraOnBodies(sim, bodies) {
  if (!sim || !bodies) return;
  const b = getBodiesBounds(bodies);
  if (!b) return;
  sim.camera.x = b.cx;
  sim.camera.y = b.cy;
  // Keep current zoom but clamp camera into world bounds.
  clampCamera(sim);
}

function mergeBodiesIntoSim(target, incoming) {
  if (!target || !incoming) return;
  target.rigid = target.rigid || [];
  target.soft = target.soft || { nodes: [], springs: [] };
  target.hybrid = target.hybrid || [];

  const rigidOffset = target.rigid.length;
  const nodeOffset = target.soft.nodes.length;
  const maxCluster = target.soft.nodes.reduce((m, n) => Math.max(m, n.clusterId || 0), -1);
  const clusterOffset = maxCluster + 1;

  for (const rb of incoming.rigid || []) target.rigid.push({ ...rb });

  for (const n of incoming.soft?.nodes || []) {
    target.soft.nodes.push({ ...n, clusterId: (n.clusterId || 0) + clusterOffset });
  }
  for (const s of incoming.soft?.springs || []) {
    const [a, b, rest, edgeBodyMode, edgeDyeMode] = s;
    target.soft.springs.push([a + nodeOffset, b + nodeOffset, rest, edgeBodyMode, edgeDyeMode]);
  }

  for (const h of incoming.hybrid || []) {
    target.hybrid.push({
      ...h,
      rigidIndex: (h.rigidIndex || 0) + rigidOffset,
      nodeIndex: (h.nodeIndex || 0) + nodeOffset,
    });
  }

}

function isConcavePolygon(verts) {
  if (!Array.isArray(verts) || verts.length < 4) return false;
  let hasPos = false;
  let hasNeg = false;
  for (let i = 0; i < verts.length; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % verts.length];
    const c = verts[(i + 2) % verts.length];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (cross > 1e-6) hasPos = true;
    if (cross < -1e-6) hasNeg = true;
    if (hasPos && hasNeg) return true;
  }
  return false;
}

function drawBodiesOverlay(sim) {
  const smooth = 0.35;
  const v = getCameraView(sim);
  const collisionDebug = !!showCollisionHullEl?.checked;
  let concaveCount = 0;

  ctx.save();
  ctx.lineWidth = 1.5;
  for (let i = 0; i < sim.bodies.rigid.length; i++) {
    const b = sim.bodies.rigid[i];
    if (b._rx == null) {
      b._rx = b.x; b._ry = b.y; b._rtheta = b.theta || 0;
    } else {
      b._rx += (b.x - b._rx) * smooth;
      b._ry += (b.y - b._ry) * smooth;
      b._rtheta += ((b.theta || 0) - b._rtheta) * smooth;
    }
    ctx.fillStyle = b.digestEnabled ? 'rgba(255, 90, 120, 0.22)' : 'rgba(255,255,255,0.06)';
    const vertsW = rigidVerticesWorld({ ...b, x: b._rx, y: b._ry, theta: b._rtheta });
    const verts = vertsW.map((v) => worldToScreen(sim, v.x, v.y));
    const solverVertsW = rigidVerticesWorld(b);
    if (isConcavePolygon(solverVertsW)) concaveCount += 1;
    const sides = verts.length;
    ctx.beginPath();
    for (let vi = 0; vi < sides; vi++) {
      const p = verts[vi];
      if (vi === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fill();
    for (let ei = 0; ei < sides; ei++) {
      const a = verts[ei];
      const b2 = verts[(ei + 1) % sides];
      const dyeMode = !b.edgeDyeMode ? EDGE_DYE_MODE.DEFLECT : b.edgeDyeMode[ei];
      const bodyMode = !b.edgeBodyMode ? EDGE_BODY_MODE.BLOCK : b.edgeBodyMode[ei];
      ctx.strokeStyle = edgeModeColor(dyeMode, bodyMode === EDGE_BODY_MODE.BLOCK);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b2.x, b2.y);
      ctx.stroke();
    }

    if (collisionDebug) {
      const solverVerts = solverVertsW.map((p) => worldToScreen(sim, p.x, p.y));

      // Solver collision hull (actual polygon used by rigid-vs-soft concave contact).
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = isConcavePolygon(solverVertsW) ? 'rgba(80,255,120,0.98)' : 'rgba(120,220,255,0.95)';
      ctx.beginPath();
      for (let vi = 0; vi < solverVerts.length; vi++) {
        const p = solverVerts[vi];
        if (vi === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.stroke();

      // Solver convex proxies used by rigid-rigid polygon collisions.
      const proxyPolys = getRigidCollisionPolysWorld(b);
      ctx.strokeStyle = 'rgba(255,170,70,0.9)';
      for (const poly of proxyPolys) {
        const ps = poly.map((pp) => worldToScreen(sim, pp.x, pp.y));
        if (ps.length < 3) continue;
        ctx.beginPath();
        for (let k = 0; k < ps.length; k++) {
          const p = ps[k];
          if (k === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.stroke();
      }
      ctx.setLineDash([]);

      for (const p of solverVerts) {
        ctx.fillStyle = 'rgba(255,255,0,0.95)';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  const s = sim.bodies.soft;
  for (const node of s.nodes) {
    if (node._rx == null) {
      node._rx = node.x; node._ry = node.y;
    } else {
      node._rx += (node.x - node._rx) * smooth;
      node._ry += (node.y - node._ry) * smooth;
    }
  }

  const softClusters = new Map();
  for (const node of s.nodes) {
    const cid = node.clusterId ?? 0;
    if (!softClusters.has(cid)) softClusters.set(cid, []);
    softClusters.get(cid).push(node);
  }
  for (const nodes of softClusters.values()) {
    if (!nodes.length || !nodes[0].digestEnabled) continue;
    let cx = 0, cy = 0;
    for (const n0 of nodes) { cx += n0._rx; cy += n0._ry; }
    cx /= nodes.length; cy /= nodes.length;
    const ordered = [...nodes]
      .map((p) => ({ x: p._rx, y: p._ry, a: Math.atan2(p._ry - cy, p._rx - cx) }))
      .sort((a, b2) => a.a - b2.a);
    ctx.fillStyle = 'rgba(255, 80, 140, 0.2)';
    ctx.beginPath();
    for (let i = 0; i < ordered.length; i++) {
      const p = worldToScreen(sim, ordered[i].x, ordered[i].y);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fill();
  }
  for (const [i, j, _rest, edgeBodyMode, edgeDyeMode] of s.springs) {
    const a = s.nodes[i], b = s.nodes[j];
    const pa = worldToScreen(sim, a._rx, a._ry);
    const pb = worldToScreen(sim, b._rx, b._ry);
    const baseColor = edgeModeColor(edgeDyeMode, edgeBodyMode === EDGE_BODY_MODE.BLOCK);
    // Keep blocked+deflect soft perimeter cyan-ish for readability.
    const m = normalizeEdgeDyeModeRGB(edgeDyeMode);
    if (edgeBodyMode === EDGE_BODY_MODE.BLOCK && m[0] === EDGE_DYE_MODE.DEFLECT && m[1] === EDGE_DYE_MODE.DEFLECT && m[2] === EDGE_DYE_MODE.DEFLECT) {
      ctx.strokeStyle = '#00ffd0';
    } else {
      ctx.strokeStyle = baseColor;
    }
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  for (const node of s.nodes) {
    const p = worldToScreen(sim, node._rx, node._ry);
    ctx.fillStyle = '#00ffd0';
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(1.6, 2.2 * Math.max(1, sim.camera.zoom * 0.6)), 0, Math.PI * 2);
    ctx.fill();
  }

  // Visualize hybrid rigid-soft attachments.
  for (const h of (sim.bodies.hybrid || [])) {
    const rb = sim.bodies.rigid[h.rigidIndex];
    const node = s.nodes[h.nodeIndex];
    if (!rb || !node) continue;
    const va = rigidVertexWorld(rb, h.vertexA);
    const vb = rigidVertexWorld(rb, h.vertexB);
    const pA = worldToScreen(sim, va.x, va.y);
    const pB = worldToScreen(sim, vb.x, vb.y);
    const pN = worldToScreen(sim, node._rx, node._ry);
    ctx.strokeStyle = 'rgba(255,120,220,0.95)';
    ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(pN.x, pN.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pB.x, pB.y); ctx.lineTo(pN.x, pN.y); ctx.stroke();
  }

  if (collisionDebug && Array.isArray(sim.lastRigidContacts)) {
    ctx.fillStyle = 'rgba(255,90,90,0.95)';
    for (const c of sim.lastRigidContacts.slice(0, 24)) {
      if (!Array.isArray(c.contact) || c.contact.length < 2) continue;
      const p = worldToScreen(sim, c.contact[0], c.contact[1]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  const collisionDebugSuffix = collisionDebug
    ? ` | solver hull debug ON (concave ${concaveCount}/${sim.bodies.rigid.length}, contacts ${(sim.lastRigidContacts || []).length})`
    : '';
  ctx.fillText(`Dye edges: PASS=blue, DEFLECT=white/cyan, ABSORB=amber, MIXED=violet | zoom ${sim.camera.zoom.toFixed(2)}x${collisionDebugSuffix}`, 10, canvas.height - 28);
  ctx.fillStyle = 'rgba(0,255,208,0.95)';
  const line2 = collisionDebug
    ? 'Body edges: BLOCK (bright) vs PASS (dim) | dashed green/cyan=solver hull, dashed amber=rigid-rigid convex proxies | Alt+drag/right-drag pan, wheel zoom'
    : 'Body edges: BLOCK (bright) vs PASS (dim) | hybrid links=magenta | Alt+drag/right-drag pan, wheel zoom';
  ctx.fillText(line2, 10, canvas.height - 12);
  ctx.restore();
}

async function initSim() {
  const controls = readControls();
  if (!navigator.gpu) throw new Error('WebGPU unavailable in browser');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice();

  const cells = controls.n * controls.n;
  const bytes = cells * 4;

  const uniform = createUniformBuffer(device);
  uploadUniforms(device, uniform, controls);

  const vxA = createBuffer(device, bytes), vxB = createBuffer(device, bytes);
  const vyA = createBuffer(device, bytes), vyB = createBuffer(device, bytes);
  const div = createBuffer(device, bytes);
  const pA = createBuffer(device, bytes), pB = createBuffer(device, bytes);
  const rA = createBuffer(device, bytes), rB = createBuffer(device, bytes);
  const gA = createBuffer(device, bytes), gB = createBuffer(device, bytes);
  const bA = createBuffer(device, bytes), bB = createBuffer(device, bytes);

  const viscMapCpu = makeDefaultViscMap(controls.n);
  const viscMapGpu = createBuffer(device, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(viscMapGpu, 0, viscMapCpu);

  const readR = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readG = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readB = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readVx = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readVy = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const inject = await createPipeline(device, injectWgsl);
  const advVel = await createPipeline(device, advectVelWgsl);
  const divPipe = await createPipeline(device, divergenceWgsl);
  const jacobiP = await createPipeline(device, jacobiPressureWgsl);
  const project = await createPipeline(device, projectWgsl);
  const advDye = await createPipeline(device, advectDyeWgsl);

  return {
    controls, cells, bytes,
    device, uniform,
    inject, advVel, divPipe, jacobiP, project, advDye,
    vx0: vxA, vx1: vxB, vy0: vyA, vy1: vyB,
    pr0: pA, pr1: pB,
    rr0: rA, rr1: rB, gg0: gA, gg1: gB, bb0: bA, bb1: bB,
    viscMapCpu, viscMapGpu,
    div, readR, readG, readB, readVx, readVy,
    bodies: initBodies(controls.n, controls),
    emitters: initEmitters(controls.n),
    camera: { x: controls.n * 0.5, y: controls.n * 0.5, zoom: controls.n >= 1024 ? 1.8 : 1.0 },
    couplingTelemetry: [],
    frame: 0, t0: performance.now(),
  };
}

function workgroups(n) { return Math.ceil(n / WORKGROUP); }

async function stepAndRender() {
  if (!running || !sim) return;
  const s = sim;
  const uiControls = readControls();

  // Grid-size changes require full GPU buffer reallocation; hot-swapping n causes dimension mismatches.
  if (uiControls.n !== s.controls.n) {
    running = false;
    sim = null;
    log({ ok: true, msg: `reinitializing for grid ${uiControls.n}` });
    await start();
    return;
  }

  s.controls = { ...s.controls, ...uiControls, n: s.controls.n };
  uploadUniforms(s.device, s.uniform, s.controls);

  const enc = s.device.createCommandEncoder();

  let pass = enc.beginComputePass();
  pass.setPipeline(s.inject.pipeline);
  pass.setBindGroup(0, s.inject.bg([s.uniform, s.vx0, s.vy0, s.rr0, s.gg0, s.bb0]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(s.advVel.pipeline);
  pass.setBindGroup(0, s.advVel.bg([s.uniform, s.vx0, s.vy0, s.vx1, s.vy1, s.viscMapGpu]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();
  [s.vx0, s.vx1] = [s.vx1, s.vx0];
  [s.vy0, s.vy1] = [s.vy1, s.vy0];

  pass = enc.beginComputePass();
  pass.setPipeline(s.divPipe.pipeline);
  pass.setBindGroup(0, s.divPipe.bg([s.uniform, s.vx0, s.vy0, s.div]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();

  for (let i = 0; i < JACOBI_ITERS; i++) {
    pass = enc.beginComputePass();
    pass.setPipeline(s.jacobiP.pipeline);
    pass.setBindGroup(0, s.jacobiP.bg([s.uniform, s.pr0, s.div, s.pr1]));
    pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
    pass.end();
    [s.pr0, s.pr1] = [s.pr1, s.pr0];
  }

  pass = enc.beginComputePass();
  pass.setPipeline(s.project.pipeline);
  pass.setBindGroup(0, s.project.bg([s.uniform, s.vx0, s.vy0, s.pr0]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(s.advDye.pipeline);
  pass.setBindGroup(0, s.advDye.bg([s.uniform, s.vx0, s.vy0, s.rr0, s.gg0, s.bb0, s.rr1, s.gg1, s.bb1]));
  pass.dispatchWorkgroups(workgroups(s.controls.n), workgroups(s.controls.n));
  pass.end();
  [s.rr0, s.rr1] = [s.rr1, s.rr0];
  [s.gg0, s.gg1] = [s.gg1, s.gg0];
  [s.bb0, s.bb1] = [s.bb1, s.bb0];

  // Read back every frame so body integration/render cadence stays coherent (avoids apparent doubling/jitter).
  const doReadback = true;
  if (doReadback) {
    enc.copyBufferToBuffer(s.rr0, 0, s.readR, 0, s.bytes);
    enc.copyBufferToBuffer(s.gg0, 0, s.readG, 0, s.bytes);
    enc.copyBufferToBuffer(s.bb0, 0, s.readB, 0, s.bytes);
    enc.copyBufferToBuffer(s.vx0, 0, s.readVx, 0, s.bytes);
    enc.copyBufferToBuffer(s.vy0, 0, s.readVy, 0, s.bytes);
  }

  s.device.queue.submit([enc.finish()]);

  if (doReadback) {
    await Promise.all([
      s.readR.mapAsync(GPUMapMode.READ),
      s.readG.mapAsync(GPUMapMode.READ),
      s.readB.mapAsync(GPUMapMode.READ),
      s.readVx.mapAsync(GPUMapMode.READ),
      s.readVy.mapAsync(GPUMapMode.READ),
    ]);

    const r = new Float32Array(s.readR.getMappedRange().slice(0));
    const g = new Float32Array(s.readG.getMappedRange().slice(0));
    const b = new Float32Array(s.readB.getMappedRange().slice(0));
    const vx = new Float32Array(s.readVx.getMappedRange().slice(0));
    const vy = new Float32Array(s.readVy.getMappedRange().slice(0));
    s.readR.unmap(); s.readG.unmap(); s.readB.unmap(); s.readVx.unmap(); s.readVy.unmap();

    // Rigid + soft coupling: carry/drag from flow + two-way pushback/swim impulses.
    const couplingInstant = stepBodiesAndInject(s, vx, vy);
    applyEmitters(s, r, g, b, vx, vy);
    applyBodyEdgeFieldBarriers({ sim: s, r, g, b, vx, vy, rigidVerticesWorld });
    applyDigestiveCapture(s, r, g, b);
    enforceFluidEdgeBoundariesCpu(vx, vy, s.controls.n);

    // Last-resort guardrail: prevent non-finite/unsafe velocity components from
    // being re-uploaded into the next GPU fluid step.
    for (let i = 0; i < s.cells; i++) {
      const vxi = Number(vx[i]);
      const vyi = Number(vy[i]);
      vx[i] = Number.isFinite(vxi) ? Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, vxi)) : 0;
      vy[i] = Number.isFinite(vyi) ? Math.max(-FLUID_COUPLING_COMPONENT_LIMIT, Math.min(FLUID_COUPLING_COMPONENT_LIMIT, vyi)) : 0;
    }

    s.device.queue.writeBuffer(s.vx0, 0, vx);
    s.device.queue.writeBuffer(s.vy0, 0, vy);
    s.device.queue.writeBuffer(s.rr0, 0, r);
    s.device.queue.writeBuffer(s.gg0, 0, g);
    s.device.queue.writeBuffer(s.bb0, 0, b);

    const n = s.controls.n;
    const img = ctx.createImageData(n, n);
    const px = img.data;
    let sum = 0;
    for (let i = 0; i < s.cells; i++) {
      const ri = Math.max(0, Math.min(255, r[i]));
      const gi = Math.max(0, Math.min(255, g[i]));
      const bi = Math.max(0, Math.min(255, b[i]));
      const o = i * 4;
      px[o] = ri;
      px[o + 1] = gi;
      px[o + 2] = bi;
      px[o + 3] = 255;
      sum += ri + gi + bi;
    }

    const tmp = document.createElement('canvas');
    tmp.width = n; tmp.height = n;
    tmp.getContext('2d').putImageData(img, 0, 0);
    const view = getCameraView(s);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tmp, view.x, view.y, view.w, view.h, 0, 0, canvas.width, canvas.height);
    if (!showViscEl || showViscEl.checked) drawViscosityOverlay();
    drawBodiesOverlay(s);

    const elapsed = (performance.now() - s.t0) / 1000;
    const fpsNow = +(s.frame / Math.max(1e-6, elapsed)).toFixed(1);
    if (fpsHud) fpsHud.textContent = `FPS: ${fpsNow}`;
    const couplingAverages = summarizeCouplingTelemetry(s.couplingTelemetry);
    const couplingSnapshot = { ...couplingAverages, ...Object.fromEntries(Object.entries(couplingInstant || {}).map(([k,v]) => [k+'Now', +((v || 0).toFixed(4))])) };
    const rigidContacts = Array.isArray(s.lastRigidContacts) ? s.lastRigidContacts : [];
    window.__gpuLabCoupling = couplingSnapshot;
    window.__gpuLabRigidContacts = rigidContacts;

    log({
      ok: true,
      mode: 'live-fluid',
      grid: n,
      frames: s.frame,
      fps: fpsNow,
      dyeEnergy: +sum.toFixed(1),
      viscosityScale: s.controls.viscosity,
      massLight: s.controls.massLight,
      massHeavy: s.controls.massHeavy,
      massSoft: s.controls.massSoft,
      bodyDrag: s.controls.bodyDrag,
      bodyFeedback: s.controls.bodyFeedback,
      paintValue: Number(paintValueEl.value) || 0.85,
      brushSize: Number(brushSizeEl.value) || 12,
      digestiveCapture: +((s.digestiveCapture || 0).toFixed(2)),
      coupling: couplingSnapshot,
      rigidContactCount: rigidContacts.length,
      rigidContactPairs: (showCollisionHullEl?.checked ? rigidContacts.slice(0, 8) : undefined),
      droppedSoftSprings: s.bodies?.topologyGuardrails?.droppedSoftSprings || 0,
    });
  }

  s.frame += 1;
  if (running) requestAnimationFrame(() => stepAndRender());
}

async function start() {
  if (running) return;
  running = true;
  if (fpsHud) fpsHud.textContent = 'FPS: --';

  const selectedPreset = scenarioPresetEl?.value || 'baseline';
  if (selectedPreset.startsWith('mini:') && generatedMiniScenarios.size === 0) {
    await loadGeneratedMiniScenarios();
  }
  ensureMiniScenarioGridSelection(selectedPreset);

  sim = await initSim();
  applyScenarioPreset(sim, selectedPreset);
  if (pendingImportedSpecs.length) {
    let lastImported = null;
    for (const spec of pendingImportedSpecs) {
      const imported = buildBodiesFromCreatureSpec(spec, sim.controls.n, sim.controls);
      mergeBodiesIntoSim(sim.bodies, imported);
      lastImported = imported;
    }
    if (lastImported) focusCameraOnBodies(sim, lastImported);
    pendingImportedSpecs = [];
  }
  log('starting live GPU fluid sim...');
  stepAndRender().catch((e) => {
    running = false;
    log({ ok: false, error: String(e) });
  });
}

function stop() {
  running = false;
  if (fpsHud) fpsHud.textContent = 'FPS: --';
  log('stopped');
}

runBtn.addEventListener('click', () => start().catch((e) => log({ ok: false, error: String(e) })));
stopBtn.addEventListener('click', stop);
clearViscBtn.addEventListener('click', resetViscMap);
if (scenarioPresetEl) {
  scenarioPresetEl.addEventListener('change', async () => {
    const preset = scenarioPresetEl.value || 'baseline';
    const mini = ensureMiniScenarioGridSelection(preset);
    if (!sim) return;
    if (mini && sim.controls.n !== Number(mini.grid || sim.controls.n)) {
      running = false;
      sim = null;
      await start();
      return;
    }
    applyScenarioPreset(sim, preset);
  });
}

if (importSpecBtn && importSpecFile) {
  importSpecBtn.addEventListener('click', () => importSpecFile.click());
  importSpecFile.addEventListener('change', async () => {
    const f = importSpecFile.files?.[0];
    if (!f) return;
    try {
      const text = await f.text();
      const spec = parseCreatureSpec(text);
      const targetN = readControls().n;

      // Important: if a stale sim instance exists at a different grid,
      // importing directly into it gets wiped by the next reinit.
      const simMatchesTargetGrid = !!sim && sim.controls.n === targetN;

      if (simMatchesTargetGrid) {
        const imported = buildBodiesFromCreatureSpec(spec, sim.controls.n, sim.controls);
        mergeBodiesIntoSim(sim.bodies, imported);
        focusCameraOnBodies(sim, imported);
        log({
          ok: true,
          msg: 'CreatureSpec imported (appended)',
          name: spec.name || 'unnamed',
          grid: sim.controls.n,
          rigidAdded: imported.rigid?.length || 0,
          softNodesAdded: imported.soft?.nodes?.length || 0,
        });
      } else {
        pendingImportedSpecs.push(spec);
        log({
          ok: true,
          msg: 'CreatureSpec queued for import on next start/reinit at selected grid',
          name: spec.name || 'unnamed',
          selectedGrid: targetN,
          currentGrid: sim?.controls?.n ?? null,
        });
      }
    } catch (e) {
      log({ ok: false, error: `Import failed: ${String(e)}` });
    } finally {
      importSpecFile.value = '';
    }
  });
}

loadGeneratedMiniScenarios().finally(() => {
  log('ready: choose scenario, paint, then Start');
});
