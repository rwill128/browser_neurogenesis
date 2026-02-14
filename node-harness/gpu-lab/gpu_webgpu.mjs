export async function createGpuBackend({ size }) {
  const out = {
    name: 'gpu-webgpu',
    available: false,
    reason: null,
    step() {},
    dispose() {}
  };

  const gpu = globalThis?.navigator?.gpu;
  if (!gpu) {
    out.reason = 'navigator.gpu unavailable in this runtime';
    return out;
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    out.reason = 'no WebGPU adapter';
    return out;
  }

  const device = await adapter.requestDevice();
  const cells = Math.max(4, Math.floor(Number(size) || 128)) ** 2;
  const byteLen = cells * 4;

  const vx = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const vy = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const densityR = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const densityG = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const densityB = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });

  // Milestone bootstrap: device and buffers wired. Next commit adds compute shader passes.
  out.available = true;
  out.reason = null;
  out.step = function step() {
    // no-op compute pass for now; intentionally verifies backend plumbing first.
    const enc = device.createCommandEncoder();
    device.queue.submit([enc.finish()]);
  };
  out.dispose = function dispose() {
    vx.destroy();
    vy.destroy();
    densityR.destroy();
    densityG.destroy();
    densityB.destroy();
  };

  return out;
}
