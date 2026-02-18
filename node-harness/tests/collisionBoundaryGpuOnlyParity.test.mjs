import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCollisionBoundaryPassGpuOnly } from '../../sim-server/public/runtime-solvers/stepCollisionBoundaryGpuOnly.js';

function makeState() {
  return {
    bodies: {
      rigid: [
        { x: -0.5, y: 3.1, vx: -0.8, vy: 0.2, omega: 0.1, r: 0.5 },
        { x: 31.2, y: 30.6, vx: 0.7, vy: -1.1, omega: -0.08, r: 1.5 },
      ],
    },
    soft: {
      nodes: [
        { x: 0.2, y: -0.4, vx: 0.5, vy: -0.6, clusterId: 1, r: 0.5 },
        { x: 31.8, y: 12.5, vx: -0.3, vy: 0.4, clusterId: 2, r: 1.5 },
        { x: 15.5, y: 31.1, vx: 0.1, vy: 0.9, clusterId: 2, r: 1.5 },
      ],
    },
    n: 32,
  };
}

function bounceStub(body, n, bounce) {
  const min = body.r;
  const max = n - body.r;
  if (body.x < min) {
    body.x = min;
    if (body.vx < 0) body.vx = -body.vx * bounce;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  } else if (body.x > max) {
    body.x = max;
    if (body.vx > 0) body.vx = -body.vx * bounce;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  }
  if (body.y < min) {
    body.y = min;
    if (body.vy < 0) body.vy = -body.vy * bounce;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  } else if (body.y > max) {
    body.y = max;
    if (body.vy > 0) body.vy = -body.vy * bounce;
    if (typeof body.omega === 'number') body.omega *= 0.9;
  }
}

function runBaselineInline(state) {
  for (const rb of state.bodies.rigid) bounceStub(rb, state.n, 0.84);
  for (const sn of state.soft.nodes) bounceStub(sn, state.n, 0.78);
}

function assertBoundaryStateApprox(actual, expected, eps = 1e-5) {
  const compareList = (alist, elist) => {
    assert.equal(alist.length, elist.length);
    for (let i = 0; i < alist.length; i++) {
      const a = alist[i];
      const e = elist[i];
      for (const key of ['x', 'y', 'vx', 'vy']) {
        assert.ok(Math.abs(Number(a[key]) - Number(e[key])) <= eps, `${key} mismatch at index ${i}`);
      }
      if (typeof e.omega === 'number') {
        assert.ok(Math.abs(Number(a.omega) - Number(e.omega)) <= eps, `omega mismatch at index ${i}`);
      }
    }
  };

  compareList(actual.bodies.rigid, expected.bodies.rigid);
  compareList(actual.soft.nodes, expected.soft.nodes);
}

function createMockWgslDevice() {
  const storage = new WeakMap();
  const usage = globalThis.GPUBufferUsage;

  function ensure(buffer, size) {
    const current = storage.get(buffer);
    if (!current || current.byteLength < size) {
      storage.set(buffer, new ArrayBuffer(size));
    }
    return storage.get(buffer);
  }

  function readFloat32(buffer, count) {
    return new Float32Array(ensure(buffer, count * 4).slice(0, count * 4));
  }

  function writeFloat32(buffer, values) {
    const data = ensure(buffer, values.byteLength);
    new Uint8Array(data).set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  }

  return {
    createShaderModule() { return {}; },
    async createComputePipelineAsync() {
      return {
        getBindGroupLayout() { return {}; },
      };
    },
    createBuffer({ size }) {
      const buf = {
        size,
        destroy() {},
        mapAsync: async () => {},
        getMappedRange: (offset = 0, len = size) => ensure(buf, size).slice(offset, offset + len),
        unmap() {},
      };
      ensure(buf, size);
      return buf;
    },
    createBindGroup({ entries }) {
      const map = new Map(entries.map((entry) => [entry.binding, entry.resource.buffer]));
      return { __buffers: map };
    },
    createCommandEncoder() {
      const ops = [];
      return {
        beginComputePass() {
          const pass = { pipeline: null, bindGroup: null, count: 0 };
          return {
            setPipeline(pipeline) { pass.pipeline = pipeline; },
            setBindGroup(_index, bindGroup) { pass.bindGroup = bindGroup; },
            dispatchWorkgroups(count) { pass.count = count; },
            end() { ops.push({ type: 'compute', pass }); },
          };
        },
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push({ type: 'copy', src, srcOffset, dst, dstOffset, size });
        },
        finish() { return ops; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset || 0, data.byteLength || data.length);
        const target = ensure(buffer, offset + bytes.byteLength);
        new Uint8Array(target).set(bytes, offset);
      },
      submit(commandsList) {
        for (const commands of commandsList) {
          for (const cmd of commands) {
            if (cmd.type === 'compute') {
              const buffers = cmd.pass.bindGroup.__buffers;
              const params = ensure(buffers.get(0), 32);
              const paramsView = new DataView(params);
              const count = paramsView.getUint32(0, true);
              const n = paramsView.getFloat32(16, true);
              const damping = paramsView.getFloat32(20, true);
              const hasOmega = paramsView.getFloat32(24, true) > 0.5;

              const x = readFloat32(buffers.get(1), count);
              const y = readFloat32(buffers.get(2), count);
              const vx = readFloat32(buffers.get(3), count);
              const vy = readFloat32(buffers.get(4), count);
              const omega = readFloat32(buffers.get(5), count);
              const r = readFloat32(buffers.get(6), count);

              for (let i = 0; i < count; i++) {
                const minX = r[i];
                const maxX = n - r[i];
                const minY = r[i];
                const maxY = n - r[i];

                if (x[i] < minX) {
                  x[i] = minX;
                  if (vx[i] < 0) vx[i] = -vx[i] * damping;
                  if (hasOmega) omega[i] *= 0.9;
                } else if (x[i] > maxX) {
                  x[i] = maxX;
                  if (vx[i] > 0) vx[i] = -vx[i] * damping;
                  if (hasOmega) omega[i] *= 0.9;
                }

                if (y[i] < minY) {
                  y[i] = minY;
                  if (vy[i] < 0) vy[i] = -vy[i] * damping;
                  if (hasOmega) omega[i] *= 0.9;
                } else if (y[i] > maxY) {
                  y[i] = maxY;
                  if (vy[i] > 0) vy[i] = -vy[i] * damping;
                  if (hasOmega) omega[i] *= 0.9;
                }
              }

              writeFloat32(buffers.get(1), x);
              writeFloat32(buffers.get(2), y);
              writeFloat32(buffers.get(3), vx);
              writeFloat32(buffers.get(4), vy);
              writeFloat32(buffers.get(5), omega);
            } else if (cmd.type === 'copy') {
              const src = ensure(cmd.src, cmd.srcOffset + cmd.size);
              const dst = ensure(cmd.dst, cmd.dstOffset + cmd.size);
              new Uint8Array(dst).set(
                new Uint8Array(src, cmd.srcOffset, cmd.size),
                cmd.dstOffset,
              );
            }
          }
        }
      },
    },
  };
}

test('collision boundary parity: gpu-only module matches baseline rigid+soft boundary sweep', async () => {
  const baseline = makeState();
  const gpuOnly = makeState();

  runBaselineInline(baseline);
  const runtime = await applyCollisionBoundaryPassGpuOnly({
    rigidBodies: gpuOnly.bodies.rigid,
    soft: gpuOnly.soft,
    n: gpuOnly.n,
    rigidBounce: 0.84,
    softBounce: 0.78,
    applyBounceBoundary: bounceStub,
  });

  assert.deepEqual(gpuOnly, baseline);
  assert.deepEqual(runtime, { mode: 'cpu', reason: 'wgsl-unavailable' });
});

test('collision boundary WGSL offload matches baseline boundary semantics', async () => {
  globalThis.GPUBufferUsage = {
    STORAGE: 1 << 0,
    COPY_DST: 1 << 1,
    COPY_SRC: 1 << 2,
    MAP_READ: 1 << 3,
    UNIFORM: 1 << 4,
  };
  globalThis.GPUMapMode = { READ: 1 };

  const baseline = makeState();
  const gpuOnly = makeState();
  const wgslOffload = {
    enabled: true,
    device: createMockWgslDevice(),
    state: {},
  };

  runBaselineInline(baseline);
  const runtime = await applyCollisionBoundaryPassGpuOnly({
    rigidBodies: gpuOnly.bodies.rigid,
    soft: gpuOnly.soft,
    n: gpuOnly.n,
    rigidBounce: 0.84,
    softBounce: 0.78,
    applyBounceBoundary: bounceStub,
    wgslOffload,
  });

  assert.deepEqual(runtime, { mode: 'wgsl', reason: 'ok' });
  assert.equal(wgslOffload.state.lastMode, 'wgsl');
  assert.equal(wgslOffload.state.lastError, null);
  assertBoundaryStateApprox(gpuOnly, baseline);
});
