const out = document.getElementById('out');
const runBtn = document.getElementById('runBtn');

function log(v) { out.textContent = typeof v === 'string' ? v : JSON.stringify(v, null, 2); }

const N = 256;
const CELLS = N * N;
const BYTES = CELLS * 4;
const WORKGROUP = 8;

function wg() { return Math.ceil(N / WORKGROUP); }

function createBuffer(device, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
  return device.createBuffer({ size: BYTES, usage });
}

function createUniformBuffer(device) {
  return device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}

function uploadUniforms(device, uniformBuffer, dt, fade) {
  const a = new ArrayBuffer(16);
  const u32 = new Uint32Array(a);
  const f32 = new Float32Array(a);
  u32[0] = N;
  f32[1] = dt;
  f32[2] = fade;
  device.queue.writeBuffer(uniformBuffer, 0, a);
}

const commonWgsl = `
struct Params {
  n: u32,
  dt: f32,
  fade: f32,
  _pad: f32,
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

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n || gid.y >= p.n) { return; }
  let i = idx(gid.x, gid.y);
  let x = f32(gid.x);
  let y = f32(gid.y);
  let vx = vx0[i];
  let vy = vy0[i];
  let px = x - p.dt * vx;
  let py = y - p.dt * vy;
  vx1[i] = sampleBilinear(&vx0, px, py);
  vy1[i] = sampleBilinear(&vy0, px, py);
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
  if (d2 < 36.0) {
    vx[i] = vx[i] + 10.0;
    vy[i] = vy[i] + 1.5 * sin(f32(i) * 0.0005);
    r[i] = 255.0;
    g[i] = 120.0;
    b[i] = 40.0;
  }
}
`;

async function createPipeline(device, code, entries) {
  const module = device.createShaderModule({ code });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  return {
    pipeline,
    makeBindGroup(resources) {
      return device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: resources.map((buffer, i) => ({ binding: i, resource: { buffer } }))
      });
    }
  };
}

async function runFluid() {
  if (!navigator.gpu) return log({ ok: false, reason: 'WebGPU unavailable in browser' });
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return log({ ok: false, reason: 'No adapter' });
  const device = await adapter.requestDevice();

  const uniform = createUniformBuffer(device);
  uploadUniforms(device, uniform, 0.12, 0.996);

  const vxA = createBuffer(device), vxB = createBuffer(device);
  const vyA = createBuffer(device), vyB = createBuffer(device);
  const div = createBuffer(device);
  const pA = createBuffer(device), pB = createBuffer(device);
  const rA = createBuffer(device), rB = createBuffer(device);
  const gA = createBuffer(device), gB = createBuffer(device);
  const bA = createBuffer(device), bB = createBuffer(device);

  const inject = await createPipeline(device, injectWgsl);
  const advVel = await createPipeline(device, advectVelWgsl);
  const divPipe = await createPipeline(device, divergenceWgsl);
  const jacobiP = await createPipeline(device, jacobiPressureWgsl);
  const project = await createPipeline(device, projectWgsl);
  const advDye = await createPipeline(device, advectDyeWgsl);

  const readback = device.createBuffer({ size: BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  let vx0 = vxA, vx1 = vxB, vy0 = vyA, vy1 = vyB;
  let pr0 = pA, pr1 = pB;
  let rr0 = rA, rr1 = rB, gg0 = gA, gg1 = gB, bb0 = bA, bb1 = bB;

  const t0 = performance.now();
  const STEPS = 180;

  for (let s = 0; s < STEPS; s++) {
    const enc = device.createCommandEncoder();

    {
      const pass = enc.beginComputePass();
      pass.setPipeline(inject.pipeline);
      pass.setBindGroup(0, inject.makeBindGroup([uniform, vx0, vy0, rr0, gg0, bb0]));
      pass.dispatchWorkgroups(wg(), wg());
      pass.end();
    }

    {
      const pass = enc.beginComputePass();
      pass.setPipeline(advVel.pipeline);
      pass.setBindGroup(0, advVel.makeBindGroup([uniform, vx0, vy0, vx1, vy1]));
      pass.dispatchWorkgroups(wg(), wg());
      pass.end();
    }
    [vx0, vx1] = [vx1, vx0];
    [vy0, vy1] = [vy1, vy0];

    {
      const pass = enc.beginComputePass();
      pass.setPipeline(divPipe.pipeline);
      pass.setBindGroup(0, divPipe.makeBindGroup([uniform, vx0, vy0, div]));
      pass.dispatchWorkgroups(wg(), wg());
      pass.end();
    }

    for (let i = 0; i < 20; i++) {
      const pass = enc.beginComputePass();
      pass.setPipeline(jacobiP.pipeline);
      pass.setBindGroup(0, jacobiP.makeBindGroup([uniform, pr0, div, pr1]));
      pass.dispatchWorkgroups(wg(), wg());
      pass.end();
      [pr0, pr1] = [pr1, pr0];
    }

    {
      const pass = enc.beginComputePass();
      pass.setPipeline(project.pipeline);
      pass.setBindGroup(0, project.makeBindGroup([uniform, vx0, vy0, pr0]));
      pass.dispatchWorkgroups(wg(), wg());
      pass.end();
    }

    {
      const pass = enc.beginComputePass();
      pass.setPipeline(advDye.pipeline);
      pass.setBindGroup(0, advDye.makeBindGroup([uniform, vx0, vy0, rr0, gg0, bb0, rr1, gg1, bb1]));
      pass.dispatchWorkgroups(wg(), wg());
      pass.end();
    }
    [rr0, rr1] = [rr1, rr0];
    [gg0, gg1] = [gg1, gg0];
    [bb0, bb1] = [bb1, bb0];

    if (s === STEPS - 1) {
      enc.copyBufferToBuffer(rr0, 0, readback, 0, BYTES);
    }

    device.queue.submit([enc.finish()]);
  }

  await readback.mapAsync(GPUMapMode.READ);
  const arr = new Float32Array(readback.getMappedRange());
  let sum = 0;
  let max = 0;
  let nz = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    sum += v;
    if (v > max) max = v;
    if (v > 0.5) nz++;
  }
  readback.unmap();
  const elapsed = performance.now() - t0;

  log({
    ok: true,
    backend: 'webgpu',
    grid: N,
    steps: STEPS,
    sps: +((STEPS * 1000) / elapsed).toFixed(2),
    elapsedMs: +elapsed.toFixed(2),
    dye: { sum: +sum.toFixed(2), max: +max.toFixed(2), footprintPct: +(100 * nz / arr.length).toFixed(3) }
  });
}

runBtn.addEventListener('click', () => runFluid().catch((e) => log({ ok: false, error: String(e) })));
log('ready: click Run WebGPU smoke test (now running fluid compute passes)');
