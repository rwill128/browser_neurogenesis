import test from 'node:test';
import assert from 'node:assert/strict';

import { stabilizeRigidPostIntegrateGpuOnly } from '../../sim-server/public/runtime-solvers/stepRigidPostIntegrateGpuOnly.js';

function makeBodies() {
  return [
    { x: 2, y: 3, vx: 7.2, vy: -1.1, omega: 0.41 },
    { x: 4, y: 9, vx: -2.4, vy: 1.8, omega: -0.77 },
    { x: 8, y: 1, vx: 0.2, vy: -0.15, omega: 0.06 },
    { x: 5, y: 7, vx: Number.NaN, vy: Infinity, omega: -Infinity },
  ];
}

function baselineRigidPostIntegrateClamp(rigidBodies, velocityCap = 4.0, omegaCap = 0.22) {
  for (const rb of rigidBodies) {
    const vmag = Math.hypot(rb.vx, rb.vy);
    if (vmag > velocityCap) {
      rb.vx = (rb.vx / vmag) * velocityCap;
      rb.vy = (rb.vy / vmag) * velocityCap;
    }
    rb.omega = Math.max(-omegaCap, Math.min(omegaCap, rb.omega || 0));
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

  return {
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
        vx: byBinding.get(1),
        vy: byBinding.get(2),
        omega: byBinding.get(3),
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
              const vCap = paramsView.getFloat32(16, true);
              const wCap = paramsView.getFloat32(20, true);

              const vx = new Float32Array(activeBindGroup.vx.bytes.buffer, activeBindGroup.vx.bytes.byteOffset, count);
              const vy = new Float32Array(activeBindGroup.vy.bytes.buffer, activeBindGroup.vy.bytes.byteOffset, count);
              const omega = new Float32Array(activeBindGroup.omega.bytes.buffer, activeBindGroup.omega.bytes.byteOffset, count);

              for (let i = 0; i < count; i++) {
                let vxv = vx[i];
                let vyv = vy[i];
                const speed = Math.hypot(vxv, vyv);
                if (speed > vCap) {
                  const inv = vCap / Math.max(speed, 1e-9);
                  vxv *= inv;
                  vyv *= inv;
                }
                vx[i] = vxv;
                vy[i] = vyv;
                omega[i] = Math.max(-wCap, Math.min(wCap, omega[i]));
              }

              for (const copy of copies) copy();
            },
          };
        },
      };
    },
  };
}

test('gpu-only rigid post-integrate stabilization matches baseline clamp semantics', async () => {
  const baselineBodies = makeBodies();
  const gpuBodies = makeBodies();

  baselineRigidPostIntegrateClamp(baselineBodies, 4.0, 0.22);
  const runtime = await stabilizeRigidPostIntegrateGpuOnly({ rigidBodies: gpuBodies, velocityCap: 4.0, omegaCap: 0.22 });

  assert.deepEqual(gpuBodies, baselineBodies);
  assert.deepEqual(runtime, { mode: 'cpu', reason: 'wgsl-unavailable' });
});

function assertBodiesApproxEqual(actual, expected, eps = 1e-6) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    assert.equal(a.x, e.x);
    assert.equal(a.y, e.y);
    for (const key of ['vx', 'vy', 'omega']) {
      const av = a[key];
      const ev = e[key];
      if (Number.isNaN(ev)) {
        assert.ok(Number.isNaN(av), `body ${i} ${key} expected NaN`);
      } else {
        assert.ok(Math.abs(av - ev) <= eps, `body ${i} ${key} mismatch: ${av} vs ${ev}`);
      }
    }
  }
}

test('gpu-only rigid post-integrate WGSL offload path matches baseline clamp semantics', async () => {
  const restoreGpu = installMockGpuGlobals();
  try {
    const baselineBodies = makeBodies();
    const gpuBodies = makeBodies();
    const wgslOffload = {
      enabled: true,
      device: createMockWgslDevice(),
      state: {},
    };

    baselineRigidPostIntegrateClamp(baselineBodies, 4.0, 0.22);
    const runtime = await stabilizeRigidPostIntegrateGpuOnly({
      rigidBodies: gpuBodies,
      velocityCap: 4.0,
      omegaCap: 0.22,
      wgslOffload,
    });

    assertBodiesApproxEqual(gpuBodies, baselineBodies);
    assert.deepEqual(runtime, { mode: 'wgsl', reason: 'ok' });
    assert.equal(wgslOffload.state.lastMode, 'wgsl');
    assert.equal(wgslOffload.state.lastError, null);
  } finally {
    restoreGpu();
  }
});
