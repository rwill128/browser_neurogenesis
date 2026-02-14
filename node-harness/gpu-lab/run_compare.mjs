import { createCpuBackend } from './cpu_ref.mjs';
import { createGpuBackend } from './gpu_stub.mjs';
import { snapshotMetrics } from './schema.mjs';

const cfg = {
  size: Number(process.env.GRID || 256),
  steps: Number(process.env.STEPS || 120),
  dt: 0.001,
  diffusion: 0.0005,
  viscosity: 0.0009
};

function nowMs() { return performance.now(); }

function seedFluid(fluid) {
  const cx = Math.floor(fluid.size * 0.5);
  const cy = Math.floor(fluid.size * 0.5);
  fluid.addDensity(cx, cy, 255, 120, 60, 200);
  fluid.addVelocity(cx, cy, 2.5, 0.0);
}

function runCpu() {
  const cpu = createCpuBackend(cfg);
  seedFluid(cpu.fluid);
  const t0 = nowMs();
  for (let i = 0; i < cfg.steps; i++) cpu.step(i);
  const elapsed = nowMs() - t0;
  const metrics = snapshotMetrics({
    vx: cpu.fluid.Vx,
    vy: cpu.fluid.Vy,
    densityR: cpu.fluid.densityR,
    densityG: cpu.fluid.densityG,
    densityB: cpu.fluid.densityB
  });
  return { backend: cpu.name, sps: (cfg.steps / elapsed) * 1000, elapsedMs: elapsed, metrics };
}

function runGpuStub() {
  const gpu = createGpuBackend();
  const t0 = nowMs();
  for (let i = 0; i < cfg.steps; i++) gpu.step(null, cfg);
  const elapsed = nowMs() - t0;
  return { backend: gpu.name, sps: (cfg.steps / Math.max(1e-9, elapsed)) * 1000, elapsedMs: elapsed };
}

console.log(JSON.stringify({ cfg, cpu: runCpu(), gpu: runGpuStub() }, null, 2));
