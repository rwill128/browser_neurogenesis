const out = document.getElementById('out');
const runBtn = document.getElementById('runBtn');

function log(obj) { out.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2); }

async function runSmoke() {
  if (!navigator.gpu) {
    log({ ok: false, reason: 'navigator.gpu unavailable in browser context' });
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { log({ ok: false, reason: 'no adapter' }); return; }
  const device = await adapter.requestDevice();

  const n = 1024;
  const bytes = n * 4;
  const src = new Float32Array(n);
  for (let i = 0; i < n; i++) src[i] = i;

  const storage = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(storage, 0, src);

  const module = device.createShaderModule({ code: `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i < ${n}u) {
        data[i] = data[i] + 1.0;
      }
    }
  `});

  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: storage } }] });

  const t0 = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  enc.copyBufferToBuffer(storage, 0, readback, 0, bytes);
  device.queue.submit([enc.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const arr = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  const t1 = performance.now();

  const ok = arr[0] === 1 && arr[10] === 11 && arr[n - 1] === n;
  log({ ok, backend: 'webgpu', elapsedMs: +(t1 - t0).toFixed(3), sample: [arr[0], arr[1], arr[10], arr[n - 1]] });
}

runBtn.addEventListener('click', () => runSmoke().catch(e => log({ ok:false, error:String(e) })));
log('ready');
