const out = document.getElementById('out');
const runBtn = document.getElementById('runBtn');
const stopBtn = document.getElementById('stopBtn');
const clearViscBtn = document.getElementById('clearViscBtn');
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const viscCanvas = document.getElementById('viscMap');
const viscCtx = viscCanvas.getContext('2d');

const gridEl = document.getElementById('gridSize');
const dtEl = document.getElementById('dt');
const fadeEl = document.getElementById('fade');
const viscosityEl = document.getElementById('viscosity');
const impulseEl = document.getElementById('impulse');
const radiusEl = document.getElementById('radius');
const brushSizeEl = document.getElementById('brushSize');
const paintValueEl = document.getElementById('paintValue');

function log(v) { out.textContent = typeof v === 'string' ? v : JSON.stringify(v, null, 2); }

const WORKGROUP = 8;
const JACOBI_ITERS = 20;

function readControls() {
  return {
    n: Math.max(32, Number(gridEl.value) || 256),
    dt: Number(dtEl.value) || 0.1,
    fade: Number(fadeEl.value) || 0.996,
    viscosity: Number(viscosityEl.value) || 0.001,
    impulse: Number(impulseEl.value) || 7,
    radius: Number(radiusEl.value) || 8,
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
  vx1[i] = sampleBilinear(&vx0, px, py) * decay;
  vy1[i] = sampleBilinear(&vy0, px, py) * decay;
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

function uploadViscMap() {
  if (!sim) return;
  sim.device.queue.writeBuffer(sim.viscMapGpu, 0, sim.viscMapCpu);
}

function drawViscPane() {
  if (!sim) return;
  const n = sim.controls.n;
  const img = viscCtx.createImageData(n, n);
  for (let i = 0; i < sim.viscMapCpu.length; i++) {
    const v = Math.max(0, Math.min(1, sim.viscMapCpu[i]));
    const o = i * 4;
    img.data[o] = v * 255;
    img.data[o + 1] = (1 - v) * 180;
    img.data[o + 2] = (1 - v) * 255;
    img.data[o + 3] = 255;
  }
  const tmp = document.createElement('canvas');
  tmp.width = n; tmp.height = n;
  tmp.getContext('2d').putImageData(img, 0, 0);
  viscCtx.clearRect(0, 0, viscCanvas.width, viscCanvas.height);
  viscCtx.drawImage(tmp, 0, 0, n, n, 0, 0, viscCanvas.width, viscCanvas.height);
}

function resetViscMap() {
  if (!sim) return;
  sim.viscMapCpu.fill(0.5);
  uploadViscMap();
  drawViscPane();
  log({ ok: true, msg: 'viscosity map reset' });
}

function paintAt(clientX, clientY, erase = false) {
  if (!sim) return;
  const rect = viscCanvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * sim.controls.n;
  const y = ((clientY - rect.top) / rect.height) * sim.controls.n;
  const r = Math.max(1, Number(brushSizeEl.value) || 12) * (sim.controls.n / viscCanvas.width);
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
  drawViscPane();
}

viscCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
viscCanvas.addEventListener('mousedown', (e) => {
  painting = true;
  paintAt(e.clientX, e.clientY, e.button === 2 || e.shiftKey);
});
window.addEventListener('mouseup', () => { painting = false; });
viscCanvas.addEventListener('mousemove', (e) => {
  if (!painting) return;
  paintAt(e.clientX, e.clientY, (e.buttons & 2) !== 0 || e.shiftKey);
});

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

function initBodies(n) {
  const rigid = [
    { x: n * 0.22, y: n * 0.28, vx: 0, vy: 0, r: 5 },
    { x: n * 0.75, y: n * 0.62, vx: 0, vy: 0, r: 6 },
  ];
  const softNodes = [
    { x: n * 0.50, y: n * 0.35, vx: 0, vy: 0 },
    { x: n * 0.55, y: n * 0.40, vx: 0, vy: 0 },
    { x: n * 0.46, y: n * 0.42, vx: 0, vy: 0 },
    { x: n * 0.51, y: n * 0.47, vx: 0, vy: 0 },
  ];
  const springs = [[0,1,6],[1,2,7],[2,3,6],[3,0,7],[0,2,8],[1,3,8]];
  return { rigid, soft: { nodes: softNodes, springs } };
}

function stepBodiesAndInject(sim, vxField, vyField) {
  const n = sim.controls.n;
  const dt = sim.controls.dt;
  const bodies = sim.bodies;

  for (const b of bodies.rigid) {
    const fx = sampleFieldBilinear(vxField, n, b.x, b.y);
    const fy = sampleFieldBilinear(vyField, n, b.x, b.y);
    b.vx += (fx - b.vx) * 0.22 + Math.sin(sim.frame * 0.03 + b.x * 0.01) * 0.01;
    b.vy += (fy - b.vy) * 0.22;
    b.x = (b.x + b.vx * dt * 30 + n) % n;
    b.y = (b.y + b.vy * dt * 30 + n) % n;
  }

  const s = bodies.soft;
  for (const node of s.nodes) {
    const fx = sampleFieldBilinear(vxField, n, node.x, node.y);
    const fy = sampleFieldBilinear(vyField, n, node.x, node.y);
    node.vx += (fx - node.vx) * 0.28;
    node.vy += (fy - node.vy) * 0.28;
  }
  for (let iter = 0; iter < 2; iter++) {
    for (const [i, j, rest] of s.springs) {
      const a = s.nodes[i], b = s.nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const err = (d - rest) * 0.18;
      const nx = dx / d, ny = dy / d;
      a.vx += nx * err * 0.02; a.vy += ny * err * 0.02;
      b.vx -= nx * err * 0.02; b.vy -= ny * err * 0.02;
    }
  }
  for (const node of s.nodes) {
    node.x = (node.x + node.vx * dt * 26 + n) % n;
    node.y = (node.y + node.vy * dt * 26 + n) % n;
  }

  const injectPoint = (px, py, pvx, pvy, rad=3.0) => {
    const minX = Math.max(0, Math.floor(px - rad));
    const maxX = Math.min(n - 1, Math.ceil(px + rad));
    const minY = Math.max(0, Math.floor(py - rad));
    const maxY = Math.min(n - 1, Math.ceil(py + rad));
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - px, dy = y - py;
        const d = Math.hypot(dx, dy);
        if (d > rad) continue;
        const w = 1 - d / rad;
        const idx = y * n + x;
        vxField[idx] += pvx * 0.06 * w;
        vyField[idx] += pvy * 0.06 * w;
      }
    }
  };

  for (const b of bodies.rigid) injectPoint(b.x, b.y, b.vx, b.vy, b.r * 0.8);
  for (const node of s.nodes) injectPoint(node.x, node.y, node.vx, node.vy, 2.2);
}

function drawBodiesOverlay(sim) {
  const n = sim.controls.n;
  const sx = canvas.width / n;
  ctx.save();
  ctx.lineWidth = 1.5;
  for (const b of sim.bodies.rigid) {
    ctx.strokeStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(b.x * sx, b.y * sx, b.r * sx, 0, Math.PI * 2);
    ctx.stroke();
  }
  const s = sim.bodies.soft;
  ctx.strokeStyle = '#7ee0ff';
  for (const [i, j] of s.springs) {
    const a = s.nodes[i], b = s.nodes[j];
    ctx.beginPath();
    ctx.moveTo(a.x * sx, a.y * sx);
    ctx.lineTo(b.x * sx, b.y * sx);
    ctx.stroke();
  }
  for (const node of s.nodes) {
    ctx.fillStyle = '#00ffd0';
    ctx.beginPath();
    ctx.arc(node.x * sx, node.y * sx, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
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
    bodies: initBodies(controls.n),
    frame: 0, t0: performance.now(),
  };
}

function workgroups(n) { return Math.ceil(n / WORKGROUP); }

async function stepAndRender() {
  if (!running || !sim) return;
  const s = sim;
  s.controls = { ...s.controls, ...readControls() };
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

  const doReadback = (s.frame % 2) === 0;
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

    // Foundation for rigid+soft body coupling: fluid carries bodies, bodies inject momentum back.
    stepBodiesAndInject(s, vx, vy);
    s.device.queue.writeBuffer(s.vx0, 0, vx);
    s.device.queue.writeBuffer(s.vy0, 0, vy);

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
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tmp, 0, 0, n, n, 0, 0, canvas.width, canvas.height);
    drawBodiesOverlay(s);

    drawViscPane();

    const elapsed = (performance.now() - s.t0) / 1000;
    log({
      ok: true,
      mode: 'live-fluid',
      grid: n,
      frames: s.frame,
      fps: +(s.frame / Math.max(1e-6, elapsed)).toFixed(1),
      dyeEnergy: +sum.toFixed(1),
      viscosityScale: s.controls.viscosity,
      paintValue: Number(paintValueEl.value) || 0.85,
      brushSize: Number(brushSizeEl.value) || 12,
    });
  }

  s.frame += 1;
  if (running) requestAnimationFrame(() => stepAndRender());
}

async function start() {
  if (running) return;
  running = true;
  sim = await initSim();
  drawViscPane();
  log('starting live GPU fluid sim...');
  stepAndRender().catch((e) => {
    running = false;
    log({ ok: false, error: String(e) });
  });
}

function stop() {
  running = false;
  log('stopped');
}

runBtn.addEventListener('click', () => start().catch((e) => log({ ok: false, error: String(e) })));
stopBtn.addEventListener('click', stop);
clearViscBtn.addEventListener('click', resetViscMap);
log('ready: paint on right viscosity pane, then Start');
