const out = document.getElementById('out');
const runBtn = document.getElementById('runBtn');
const stopBtn = document.getElementById('stopBtn');
const clearViscBtn = document.getElementById('clearViscBtn');
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
const massLightEl = document.getElementById('massLight');
const massHeavyEl = document.getElementById('massHeavy');
const massSoftEl = document.getElementById('massSoft');
const bodyDragEl = document.getElementById('bodyDrag');
const bodyFeedbackEl = document.getElementById('bodyFeedback');

function log(v) { out.textContent = typeof v === 'string' ? v : JSON.stringify(v, null, 2); }

const WORKGROUP = 8;
const JACOBI_ITERS = 20;

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
  const a = field[i00] * (1 - sx) + field[i10] * sx;
  const b = field[i01] * (1 - sx) + field[i11] * sx;
  return a * (1 - sy) + b * sy;
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
    const edgeDyeBlock = Array.from({ length: sides }, (_, ei) => ((ei + i) % 2 === 0) ? 1 : 0);
    rigid.push({
      x: n * (0.15 + 0.7 * ((t + Math.random() * 0.1) % 1)),
      y: n * (0.2 + 0.6 * Math.random()),
      vx: 0,
      vy: 0,
      r,
      sides,
      edgeDyeBlock,
      digestEnabled: (i % 2) === 0,
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
      });
    }

    // Ring springs (solid edges: rigid bodies should collide with them).
    for (let i = 0; i < nodeCount; i++) {
      const j = (i + 1) % nodeCount;
      const a = local[i], b = local[j];
      const ringDyeBlock = ((i + c) % 2 === 0) ? 1 : 0; // per-edge dye permeability toggle
      springs.push([base + i, base + j, Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)), 1, ringDyeBlock]);
    }
    // Cross/diagonal springs for shape retention (internal, non-solid by default).
    for (let i = 0; i < nodeCount; i++) {
      const j = (i + 2) % nodeCount;
      if (i < j || nodeCount <= 4) {
        const a = local[i], b = local[j];
        springs.push([base + i, base + j, Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)), 0, 0]);
      }
    }
    // Opposite braces for even polygons (especially hex) to prevent skew collapse.
    if (nodeCount % 2 === 0) {
      for (let i = 0; i < nodeCount / 2; i++) {
        const j = (i + nodeCount / 2) % nodeCount;
        const a = local[i], b = local[j];
        springs.push([base + i, base + j, Math.max(1e-3, Math.hypot(b.x - a.x, b.y - a.y)), 0, 0]);
      }
    }
  }

  return { rigid, soft: { nodes: softNodes, springs } };
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

function applyImpermeableSegmentFieldBarrier(n, ax, ay, bx, by, r, g, b, vx, vy, thickness = 1.4) {
  const minX = Math.max(0, Math.floor(Math.min(ax, bx) - thickness - 1));
  const maxX = Math.min(n - 1, Math.ceil(Math.max(ax, bx) + thickness + 1));
  const minY = Math.max(0, Math.floor(Math.min(ay, by) - thickness - 1));
  const maxY = Math.min(n - 1, Math.ceil(Math.max(ay, by) + thickness + 1));
  const ex = bx - ax;
  const ey = by - ay;
  const el = Math.max(1e-6, Math.hypot(ex, ey));
  const nx = -ey / el;
  const ny = ex / el;
  const tx = ex / el;
  const ty = ey / el;

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const cp = closestPointOnSegment(x + 0.5, y + 0.5, ax, ay, bx, by);
      const dx = (x + 0.5) - cp.x;
      const dy = (y + 0.5) - cp.y;
      const d = Math.hypot(dx, dy);
      if (d > thickness) continue;
      const w = 1 - (d / Math.max(1e-6, thickness));
      const i = y * n + x;

      // Wall behavior: remove normal velocity, keep tangential flow (deflection, not absorption).
      const vn = vx[i] * nx + vy[i] * ny;
      vx[i] -= vn * nx * w;
      vy[i] -= vn * ny * w;

      // Deflect dye along tangent rather than stripping it.
      const vt = vx[i] * tx + vy[i] * ty;
      const step = vt >= 0 ? 1 : -1;
      const txi = Math.max(0, Math.min(n - 1, Math.round(x + tx * step)));
      const tyi = Math.max(0, Math.min(n - 1, Math.round(y + ty * step)));
      const ti = tyi * n + txi;
      if (ti !== i) {
        const move = 0.22 * w;
        const dr = r[i] * move, dg = g[i] * move, db = b[i] * move;
        r[i] -= dr; g[i] -= dg; b[i] -= db;
        r[ti] = Math.min(255, r[ti] + dr);
        g[ti] = Math.min(255, g[ti] + dg);
        b[ti] = Math.min(255, b[ti] + db);
      }
    }
  }
}

function applyBodyEdgeFieldBarriers(sim, r, g, b, vx, vy) {
  const n = sim.controls.n;
  const softThickness = Math.max(1.2, 1.1 * (n / 256));
  const rigidThickness = Math.max(1.4, 1.2 * (n / 256));

  for (const rb of sim.bodies.rigid) {
    const sides = Math.max(3, rb.sides || 4);
    const verts = [];
    for (let i = 0; i < sides; i++) {
      const a = (rb.theta || 0) + (i / sides) * Math.PI * 2;
      verts.push({ x: rb.x + Math.cos(a) * rb.r, y: rb.y + Math.sin(a) * rb.r });
    }
    for (let i = 0; i < sides; i++) {
      const edgeBlock = !rb.edgeDyeBlock ? 1 : rb.edgeDyeBlock[i];
      if (!edgeBlock) continue;
      const a = verts[i];
      const b2 = verts[(i + 1) % sides];
      applyImpermeableSegmentFieldBarrier(n, a.x, a.y, b2.x, b2.y, r, g, b, vx, vy, rigidThickness);
    }
  }

  const s = sim.bodies.soft;
  for (const [i, j, _rest, edgeSolid, edgeDyeBlock] of s.springs) {
    if (!edgeSolid || !edgeDyeBlock) continue;
    const a = s.nodes[i], b2 = s.nodes[j];
    applyImpermeableSegmentFieldBarrier(n, a.x, a.y, b2.x, b2.y, r, g, b, vx, vy, softThickness);
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

function stepBodiesAndInject(sim, vxField, vyField) {
  const n = sim.controls.n;
  const dt = sim.controls.dt;
  const bodies = sim.bodies;
  const dragK = sim.controls.bodyDrag;
  const feedbackK = sim.controls.bodyFeedback;
  const viscMap = sim.viscMapCpu;

  const localHoneyDrag = (x, y) => {
    const v = sampleFieldBilinear(viscMap, n, x, y);
    // Make high-viscosity paint feel like honey for bodies.
    return 1.0 + Math.pow(Math.max(0, Math.min(1, v)), 2.2) * 14.0;
  };

  if (bodies.rigid[0]) {
    bodies.rigid[0].mass = sim.controls.massLight;
    bodies.rigid[0].inertia = 0.5 * bodies.rigid[0].mass * bodies.rigid[0].r * bodies.rigid[0].r;
  }
  if (bodies.rigid[1]) {
    bodies.rigid[1].mass = sim.controls.massHeavy;
    bodies.rigid[1].inertia = 0.5 * bodies.rigid[1].mass * bodies.rigid[1].r * bodies.rigid[1].r;
  }
  for (const node of bodies.soft.nodes) node.mass = sim.controls.massSoft;

  const softCentroidBefore = computeSoftCentroid(bodies.soft.nodes);
  const rigidCenterBefore = {
    x: (bodies.rigid[0].x + bodies.rigid[1].x) * 0.5,
    y: (bodies.rigid[0].y + bodies.rigid[1].y) * 0.5,
  };

  let rigidCarryTransfer = 0;
  let softCarryTransfer = 0;

  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const invMass = 1 / Math.max(0.05, b.mass);
    const invInertia = 1 / Math.max(0.05, b.inertia || 1);
    const sampleCount = Math.max(3, b.sides || (bi === 0 ? 3 : 4));
    let forceX = 0;
    let forceY = 0;
    let torque = 0;

    for (let si = 0; si < sampleCount; si++) {
      const a = (b.theta || 0) + (si / sampleCount) * Math.PI * 2;
      const rx = Math.cos(a) * b.r;
      const ry = Math.sin(a) * b.r;
      const sx = b.x + rx;
      const sy = b.y + ry;
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

    b.vx += ax * dt * 60 + swimX;
    b.vy += ay * dt * 60 + swimY;
    b.omega = (b.omega || 0) + alpha * dt * 60 + swimTorque;

    const centerHoney = localHoneyDrag(b.x, b.y);
    const linDamp = Math.max(0.72, 1.0 - 0.018 * centerHoney);
    const angDamp = Math.max(0.70, 0.992 - 0.012 * centerHoney);
    b.vx *= linDamp;
    b.vy *= linDamp;
    b.omega *= angDamp;

    const bMaxBase = 3.2;
    const bMax = Math.max(0.45, bMaxBase / (1 + 0.28 * centerHoney));
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
    const carryX = (fx - node.vx) * dragK * 0.8 * honey * invMass;
    const carryY = (fy - node.vy) * dragK * 0.8 * honey * invMass;
    node.vx += carryX * dt * 60 + swimX;
    node.vy += carryY * dt * 60 + swimY;
    const nodeDamp = Math.max(0.70, 1.0 - 0.02 * honey);
    node.vx *= nodeDamp;
    node.vy *= nodeDamp;
    const nMaxBase = 2.8;
    const nMax = Math.max(0.35, nMaxBase / (1 + 0.32 * honey));
    const nMag = Math.hypot(node.vx, node.vy);
    if (nMag > nMax) {
      node.vx = (node.vx / nMag) * nMax;
      node.vy = (node.vy / nMag) * nMax;
    }
    softCarryTransfer += Math.hypot(carryX, carryY);
  }

  for (let iter = 0; iter < 7; iter++) {
    for (const [i, j, rest] of s.springs) {
      const a = s.nodes[i], b = s.nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const err = (d - rest) * 0.68;
      const nx = dx / d, ny = dy / d;
      a.vx += nx * err * 0.034; a.vy += ny * err * 0.034;
      b.vx -= nx * err * 0.034; b.vy -= ny * err * 0.034;
    }
  }

  for (const node of s.nodes) {
    node.x = node.x + node.vx * dt * 24;
    node.y = node.y + node.vy * dt * 24;
    applyBounceBoundary(node, n, 0.78);
  }

  // Body-body collisions: rigid↔rigid, rigid↔soft, soft↔soft
  for (let iter = 0; iter < 2; iter++) {
    for (let i = 0; i < bodies.rigid.length; i++) {
      for (let j = i + 1; j < bodies.rigid.length; j++) {
        resolveCircleCollision(bodies.rigid[i], bodies.rigid[j], 0.45);
      }
    }
    for (const rb of bodies.rigid) {
      for (const sn of s.nodes) {
        resolveCircleCollision(rb, sn, 0.35);
      }
      for (const [i, j, _rest, edgeSolid] of s.springs) {
        if (!edgeSolid) continue;
        resolveRigidVsSoftEdgeCollision(rb, s.nodes[i], s.nodes[j], 0.3);
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
      for (const [i, j, _rest, edgeSolid] of s.springs) {
        if (!edgeSolid) continue;
        if (i === ni || j === ni) continue;
        const a = s.nodes[i], b = s.nodes[j];
        if (a.clusterId === node.clusterId && b.clusterId === node.clusterId) continue;
        resolveSoftNodeVsSoftEdgeCollision(node, a, b, 0.12);
      }
    }
    for (const rb of bodies.rigid) applyBounceBoundary(rb, n, 0.84);
    for (const sn of s.nodes) applyBounceBoundary(sn, n, 0.78);
  }

  let injectedMomentum = 0;
  const injectPoint = (px, py, pvx, pvy, localFluidX, localFluidY, mass, rad=3.0, swimInjectX = 0, swimInjectY = 0) => {
    const minX = Math.max(0, Math.floor(px - rad));
    const maxX = Math.min(n - 1, Math.ceil(px + rad));
    const minY = Math.max(0, Math.floor(py - rad));
    const maxY = Math.min(n - 1, Math.ceil(py + rad));
    const relX = pvx - localFluidX + swimInjectX;
    const relY = pvy - localFluidY + swimInjectY;
    const scale = feedbackK * Math.max(0.1, mass);
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - px, dy = y - py;
        const d = Math.hypot(dx, dy);
        if (d > rad) continue;
        const w = 1 - d / rad;
        const idx = y * n + x;
        const jx = relX * scale * w;
        const jy = relY * scale * w;
        vxField[idx] += jx;
        vyField[idx] += jy;
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
  const rigidCenterAfter = {
    x: (bodies.rigid[0].x + bodies.rigid[1].x) * 0.5,
    y: (bodies.rigid[0].y + bodies.rigid[1].y) * 0.5,
  };

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

function applyDigestiveCapture(sim, r, g, b) {
  const n = sim.controls.n;
  const rigidCapture = 0.055;
  const softCapture = 0.045;
  let captured = 0;

  for (const rb of sim.bodies.rigid) {
    if (!rb.digestEnabled) continue;
    const sides = Math.max(3, rb.sides || 4);
    const verts = [];
    for (let i = 0; i < sides; i++) {
      const a = (rb.theta || 0) + (i / sides) * Math.PI * 2;
      verts.push({ x: rb.x + Math.cos(a) * rb.r, y: rb.y + Math.sin(a) * rb.r });
    }
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
        const takeR = r[i] * rigidCapture;
        const takeG = g[i] * rigidCapture;
        const takeB = b[i] * rigidCapture;
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
        const takeR = r[i] * softCapture;
        const takeG = g[i] * softCapture;
        const takeB = b[i] * softCapture;
        r[i] -= takeR; g[i] -= takeG; b[i] -= takeB;
        captured += takeR + takeG + takeB;
      }
    }
  }

  sim.digestiveCapture = (sim.digestiveCapture || 0) * 0.97 + captured * 0.03;
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

function drawBodiesOverlay(sim) {
  const smooth = 0.35;
  const v = getCameraView(sim);
  const pxPerWorld = canvas.width / v.w;
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
    const sp = worldToScreen(sim, b._rx, b._ry);
    ctx.fillStyle = b.digestEnabled ? 'rgba(255, 90, 120, 0.22)' : 'rgba(255,255,255,0.06)';
    ctx.beginPath();
    const sides = Math.max(3, b.sides || (i === 0 ? 3 : 4));
    const rot = (b._rtheta || 0) + (sides === 3 ? -Math.PI * 0.5 : Math.PI * 0.25);
    drawRegularPolygon(sp.x, sp.y, b.r * pxPerWorld, sides, rot);
    ctx.fill();

    const verts = [];
    for (let vi = 0; vi < sides; vi++) {
      const a = rot + (vi / sides) * Math.PI * 2;
      verts.push({ x: sp.x + Math.cos(a) * b.r * pxPerWorld, y: sp.y + Math.sin(a) * b.r * pxPerWorld });
    }
    for (let ei = 0; ei < sides; ei++) {
      const a = verts[ei];
      const b2 = verts[(ei + 1) % sides];
      const blocked = !b.edgeDyeBlock ? 1 : b.edgeDyeBlock[ei];
      ctx.strokeStyle = blocked ? '#ffffff' : 'rgba(140,180,255,0.95)';
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b2.x, b2.y);
      ctx.stroke();
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
  for (const [i, j, _rest, edgeSolid, edgeDyeBlock] of s.springs) {
    const a = s.nodes[i], b = s.nodes[j];
    const pa = worldToScreen(sim, a._rx, a._ry);
    const pb = worldToScreen(sim, b._rx, b._ry);
    if (!edgeSolid) ctx.strokeStyle = 'rgba(90,180,190,0.45)';
    else ctx.strokeStyle = edgeDyeBlock ? '#00ffd0' : 'rgba(120,150,255,0.95)';
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

  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.fillText(`Blocked edges=white/cyan, permeable=blue, digest ON=magenta fill | zoom ${sim.camera.zoom.toFixed(2)}x`, 10, canvas.height - 28);
  ctx.fillStyle = 'rgba(0,255,208,0.95)';
  ctx.fillText('Rigid+soft mixed edge permeability | Alt+drag/right-drag pan, wheel zoom', 10, canvas.height - 12);
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
    applyBodyEdgeFieldBarriers(s, r, g, b, vx, vy);
    applyDigestiveCapture(s, r, g, b);
    enforceFluidEdgeBoundariesCpu(vx, vy, s.controls.n);
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
    window.__gpuLabCoupling = couplingSnapshot;

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
    });
  }

  s.frame += 1;
  if (running) requestAnimationFrame(() => stepAndRender());
}

async function start() {
  if (running) return;
  running = true;
  if (fpsHud) fpsHud.textContent = 'FPS: --';
  sim = await initSim();
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
log('ready: paint on right viscosity pane, then Start');
