import test from 'node:test';
import assert from 'node:assert/strict';
import { integrateSoftBodiesGpuOnly } from '../../sim-server/public/runtime-solvers/stepSoftIntegrateGpuOnly.js';

function applyBounceBoundary(body, n, damping = 0.8) {
  const r = Number(body?.r) || 0;
  const min = r;
  const max = n - r;
  if (body.x < min) {
    body.x = min;
    body.vx = Math.abs(body.vx || 0) * damping;
  } else if (body.x > max) {
    body.x = max;
    body.vx = -Math.abs(body.vx || 0) * damping;
  }
  if (body.y < min) {
    body.y = min;
    body.vy = Math.abs(body.vy || 0) * damping;
  } else if (body.y > max) {
    body.y = max;
    body.vy = -Math.abs(body.vy || 0) * damping;
  }
}

function runBaselineSoftIntegration({ soft, n, dt, scale, cap }) {
  for (const node of soft.nodes) {
    const vmag = Math.hypot(node.vx, node.vy);
    if (vmag > cap) {
      node.vx = (node.vx / vmag) * cap;
      node.vy = (node.vy / vmag) * cap;
    }
    node.x = node.x + node.vx * dt * scale;
    node.y = node.y + node.vy * dt * scale;
    applyBounceBoundary(node, n, 0.78);
  }
}

function installMockGpuGlobals() {
  const prev = {
    GPUBufferUsage: globalThis.GPUBufferUsage,
    GPUMapMode: globalThis.GPUMapMode,
  };
  globalThis.GPUBufferUsage = {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    COPY_SRC: 1 << 2,
    MAP_READ: 1 << 3,
    UNIFORM: 1 << 4,
  };
  globalThis.GPUMapMode = { READ: 1 };
  return () => {
    globalThis.GPUBufferUsage = prev.GPUBufferUsage;
    globalThis.GPUMapMode = prev.GPUMapMode;
  };
}

function createMockWgslDevice() {
  class MockBuffer {
    constructor(size) {
      this.bytes = new Uint8Array(size);
      this.mappedRange = null;
    }
    write(offset, src) {
      const data = src instanceof Uint8Array ? src : new Uint8Array(src.buffer || src, src.byteOffset || 0, src.byteLength || src.length);
      this.bytes.set(data, offset);
    }
    copyFrom(src, srcOffset, dstOffset, size) {
      this.bytes.set(src.bytes.slice(srcOffset, srcOffset + size), dstOffset);
    }
    async mapAsync() {}
    getMappedRange(offset = 0, size = this.bytes.length - offset) {
      this.mappedRange = this.bytes.slice(offset, offset + size).buffer;
      return this.mappedRange;
    }
    unmap() { this.mappedRange = null; }
    destroy() {}
  }

  const device = {
    queue: {
      writeBuffer(buffer, offset, data) {
        const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        buffer.write(offset, src);
      },
      submit(cmds) {
        for (const cmd of cmds) cmd.execute?.();
      },
    },
    createShaderModule() { return {}; },
    async createComputePipelineAsync() {
      return { getBindGroupLayout() { return {}; } };
    },
    createBuffer({ size }) { return new MockBuffer(size); },
    createBindGroup({ entries }) {
      const byBinding = new Map(entries.map((e) => [e.binding, e.resource.buffer]));
      return {
        params: byBinding.get(0),
        x: byBinding.get(1),
        y: byBinding.get(2),
        vx: byBinding.get(3),
        vy: byBinding.get(4),
      };
    },
    createCommandEncoder() {
      const copies = [];
      let activeBindGroup = null;
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup(_idx, bindGroup) { activeBindGroup = bindGroup; },
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          copies.push(() => dst.copyFrom(src, srcOffset, dstOffset, size));
        },
        finish() {
          return {
            execute() {
              const paramsView = new DataView(activeBindGroup.params.bytes.buffer, activeBindGroup.params.bytes.byteOffset, activeBindGroup.params.bytes.byteLength);
              const count = paramsView.getUint32(0, true);
              const dt = paramsView.getFloat32(16, true);
              const scale = paramsView.getFloat32(20, true);
              const cap = paramsView.getFloat32(24, true);

              const x = new Float32Array(activeBindGroup.x.bytes.buffer, activeBindGroup.x.bytes.byteOffset, count);
              const y = new Float32Array(activeBindGroup.y.bytes.buffer, activeBindGroup.y.bytes.byteOffset, count);
              const vx = new Float32Array(activeBindGroup.vx.bytes.buffer, activeBindGroup.vx.bytes.byteOffset, count);
              const vy = new Float32Array(activeBindGroup.vy.bytes.buffer, activeBindGroup.vy.bytes.byteOffset, count);

              for (let i = 0; i < count; i++) {
                let vxv = vx[i];
                let vyv = vy[i];
                const speed = Math.hypot(vxv, vyv);
                if (speed > cap) {
                  const inv = cap / Math.max(speed, 1e-9);
                  vxv *= inv;
                  vyv *= inv;
                }
                x[i] += vxv * dt * scale;
                y[i] += vyv * dt * scale;
                vx[i] = vxv;
                vy[i] = vyv;
              }

              for (const copy of copies) copy();
            },
          };
        },
      };
    },
  };

  return device;
}

test('soft integration parity: baseline loop and gpu-only module produce matching node states', async () => {
  const n = 128;
  const dt = 0.013;
  const softIntegrationScale = 24;
  const hybridNodeVCap = 3.2;

  const seedNodes = [
    { x: 10, y: 12, vx: 2.2, vy: -1.8, r: 1.4 },
    { x: 126, y: 60, vx: 5.9, vy: 1.1, r: 2.0 },
    { x: 50, y: 3, vx: -0.8, vy: -6.2, r: 1.2 },
    { x: 90, y: 127, vx: 0.4, vy: 4.4, r: 1.5 },
  ];

  const baseline = { nodes: structuredClone(seedNodes) };
  const gpuOnly = { nodes: structuredClone(seedNodes) };

  runBaselineSoftIntegration({ soft: baseline, n, dt, scale: softIntegrationScale, cap: hybridNodeVCap });
  await integrateSoftBodiesGpuOnly({
    soft: gpuOnly,
    n,
    dt,
    softIntegrationScale,
    hybridNodeVCap,
    applyBounceBoundary,
  });

  assert.equal(gpuOnly.nodes.length, baseline.nodes.length);
  for (let i = 0; i < baseline.nodes.length; i++) {
    const b = baseline.nodes[i];
    const g = gpuOnly.nodes[i];
    assert.ok(Math.abs(g.x - b.x) < 1e-9, `node ${i} x mismatch: ${g.x} vs ${b.x}`);
    assert.ok(Math.abs(g.y - b.y) < 1e-9, `node ${i} y mismatch: ${g.y} vs ${b.y}`);
    assert.ok(Math.abs(g.vx - b.vx) < 1e-9, `node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
    assert.ok(Math.abs(g.vy - b.vy) < 1e-9, `node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
  }
});

test('soft integration WGSL offload path matches baseline and survives capacity growth', async () => {
  const restoreGpu = installMockGpuGlobals();
  try {
    const n = 128;
    const dt = 0.013;
    const softIntegrationScale = 24;
    const hybridNodeVCap = 3.2;
    const wgslOffload = {
      enabled: true,
      device: createMockWgslDevice(),
      state: {},
    };

    const smallSeed = [
      { x: 12, y: 18, vx: 1.4, vy: -0.7, r: 1.2 },
      { x: 120, y: 15, vx: 3.6, vy: 0.2, r: 1.8 },
      { x: 64, y: 125, vx: 0.1, vy: 4.8, r: 1.5 },
    ];

    const smallBaseline = { nodes: structuredClone(smallSeed) };
    const smallGpuOnly = { nodes: structuredClone(smallSeed) };
    runBaselineSoftIntegration({ soft: smallBaseline, n, dt, scale: softIntegrationScale, cap: hybridNodeVCap });
    await integrateSoftBodiesGpuOnly({
      soft: smallGpuOnly,
      n,
      dt,
      softIntegrationScale,
      hybridNodeVCap,
      applyBounceBoundary,
      wgslOffload,
    });

    assert.equal(smallGpuOnly.nodes.length, smallBaseline.nodes.length);
    for (let i = 0; i < smallBaseline.nodes.length; i++) {
      const b = smallBaseline.nodes[i];
      const g = smallGpuOnly.nodes[i];
      assert.ok(Math.abs(g.x - b.x) < 5e-6, `small node ${i} x mismatch: ${g.x} vs ${b.x}`);
      assert.ok(Math.abs(g.y - b.y) < 5e-6, `small node ${i} y mismatch: ${g.y} vs ${b.y}`);
      assert.ok(Math.abs(g.vx - b.vx) < 5e-6, `small node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
      assert.ok(Math.abs(g.vy - b.vy) < 5e-6, `small node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    }

    const largeSeed = Array.from({ length: 300 }, (_, i) => ({
      x: (i % 30) * 3 + 10,
      y: Math.floor(i / 30) * 3 + 10,
      vx: ((i % 7) - 3) * 0.9,
      vy: ((i % 11) - 5) * 0.7,
      r: 1.1,
    }));

    const largeBaseline = { nodes: structuredClone(largeSeed) };
    const largeGpuOnly = { nodes: structuredClone(largeSeed) };
    runBaselineSoftIntegration({ soft: largeBaseline, n, dt, scale: softIntegrationScale, cap: hybridNodeVCap });
    await integrateSoftBodiesGpuOnly({
      soft: largeGpuOnly,
      n,
      dt,
      softIntegrationScale,
      hybridNodeVCap,
      applyBounceBoundary,
      wgslOffload,
    });

    assert.equal(largeGpuOnly.nodes.length, largeBaseline.nodes.length);
    for (let i = 0; i < largeBaseline.nodes.length; i++) {
      const b = largeBaseline.nodes[i];
      const g = largeGpuOnly.nodes[i];
      assert.ok(Math.abs(g.x - b.x) < 5e-6, `large node ${i} x mismatch: ${g.x} vs ${b.x}`);
      assert.ok(Math.abs(g.y - b.y) < 5e-6, `large node ${i} y mismatch: ${g.y} vs ${b.y}`);
      assert.ok(Math.abs(g.vx - b.vx) < 5e-6, `large node ${i} vx mismatch: ${g.vx} vs ${b.vx}`);
      assert.ok(Math.abs(g.vy - b.vy) < 5e-6, `large node ${i} vy mismatch: ${g.vy} vs ${b.vy}`);
    }

    assert.equal(wgslOffload.state.lastMode, 'wgsl-validated');
    assert.equal(wgslOffload.state.lastSourceRoute, 'wgsl-integrate-authoritative');
    assert.ok(Number.isInteger(wgslOffload.state.lastProposalSignature), 'expected deterministic WGSL proposal signature');
    assert.equal(wgslOffload.state.lastParity?.source, 'wgsl-integrate-proposal-vs-cpu');
    assert.equal(wgslOffload.state.lastParity?.mismatchCount, 0, 'expected zero CPU-vs-WGSL parity mismatches in mock device run');
  } finally {
    restoreGpu();
  }
});
