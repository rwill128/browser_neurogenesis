const WGSL_WORKGROUP_SIZE = 64;

const softMembranePressureAreaProbeWgsl = /* wgsl */`
struct Params {
  membrane_count: u32,
  _pad0: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> membrane_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> loop_indices: array<u32>;
@group(0) @binding(3) var<storage, read> node_x: array<f32>;
@group(0) @binding(4) var<storage, read> node_y: array<f32>;
@group(0) @binding(5) var<storage, read> area_base: array<f32>;
@group(0) @binding(6) var<storage, read_write> area_now_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> area_err_out: array<f32>;

fn abs_area(start: u32, stop: u32) -> f32 {
  var accum = 0.0;
  let count = stop - start;
  if (count < 3u) {
    return 0.0;
  }

  for (var i = start; i < stop; i = i + 1u) {
    let next = select(i + 1u, start, (i + 1u) >= stop);
    let ia = loop_indices[i];
    let ib = loop_indices[next];
    accum = accum + (node_x[ia] * node_y[ib] - node_y[ia] * node_x[ib]);
  }

  return abs(0.5 * accum);
}

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mi = gid.x;
  if (mi >= params.membrane_count) {
    return;
  }

  let start = membrane_offsets[mi];
  let stop = membrane_offsets[mi + 1u];
  let now = abs_area(start, stop);
  let base = max(1e-4, area_base[mi]);
  let err = clamp((base - now) / base, -0.65, 0.65);

  area_now_out[mi] = now;
  area_err_out[mi] = err;
}
`;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function canUseWgslOffload(offload) {
  return Boolean(
    offload
      && offload.enabled === true
      && offload.device
      && typeof offload.device.createComputePipelineAsync === 'function'
      && typeof offload.device.createBuffer === 'function'
      && typeof offload.device.createCommandEncoder === 'function'
      && offload.state,
  );
}

function buildSoftMembranePressureWgslPrep({ membranes, loopByCluster, softNodes, softMembraneAreaBaseline }) {
  const valid = [];
  for (const membrane of membranes || []) {
    const cid = Number(membrane?.clusterId);
    if (!Number.isInteger(cid)) continue;
    const loop = loopByCluster.get(cid);
    if (!loop || !Array.isArray(loop.indices) || loop.indices.length < 3) continue;

    const seededBase = Math.max(1e-4, Number(membrane?.restArea) || 0);
    const areaBase = Math.max(1e-4, Number(softMembraneAreaBaseline.get(cid)) || seededBase || 1e-4);
    valid.push({ cid, loopIndices: loop.indices, areaBase });
  }

  const membraneCount = valid.length;
  const membraneOffsets = new Uint32Array(membraneCount + 1);
  let indexCount = 0;
  for (let i = 0; i < membraneCount; i++) {
    membraneOffsets[i] = indexCount;
    indexCount += valid[i].loopIndices.length;
  }
  membraneOffsets[membraneCount] = indexCount;

  const loopIndices = new Uint32Array(indexCount);
  const areaBase = new Float32Array(membraneCount);
  const clusterId = new Int32Array(membraneCount);

  let write = 0;
  for (let i = 0; i < membraneCount; i++) {
    const entry = valid[i];
    areaBase[i] = entry.areaBase;
    clusterId[i] = entry.cid;
    for (let k = 0; k < entry.loopIndices.length; k++) {
      loopIndices[write++] = entry.loopIndices[k] >>> 0;
    }
  }

  const nodeCount = Array.isArray(softNodes) ? softNodes.length : 0;
  const nodeX = new Float32Array(nodeCount);
  const nodeY = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    nodeX[i] = Number(softNodes[i]?.x) || 0;
    nodeY[i] = Number(softNodes[i]?.y) || 0;
  }

  const layout = {
    membraneOffsets,
    loopIndices,
    nodeX,
    nodeY,
    areaBase,
    clusterId,
  };
  layout.byteLength = membraneOffsets.byteLength + loopIndices.byteLength + nodeX.byteLength + nodeY.byteLength + areaBase.byteLength + clusterId.byteLength;

  return {
    plan: { membraneCount, nodeCount, indexCount },
    layout,
  };
}

async function ensureSoftMembranePressureProbeState(offload, prep) {
  const device = offload.device;
  const state = offload.state;
  const membraneCount = Math.max(1, prep.plan.membraneCount);
  const nodeCount = Math.max(1, prep.plan.nodeCount);
  const indexCount = Math.max(1, prep.plan.indexCount);

  if (!state.areaProbePipeline) {
    const module = device.createShaderModule({ code: softMembranePressureAreaProbeWgsl });
    state.areaProbePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.areaProbeBindGroup = null;
  }

  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;

  if (!state.areaProbeParams) {
    state.areaProbeParams = device.createBuffer({
      size: 16,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.areaProbeBindGroup = null;
  }

  if ((state.areaProbeMembraneCapacity || 0) < membraneCount) {
    const bytes = membraneCount * 4;
    state.areaProbeOffsets?.destroy?.();
    state.areaProbeAreaBase?.destroy?.();
    state.areaProbeAreaNow?.destroy?.();
    state.areaProbeAreaErr?.destroy?.();
    state.areaProbeAreaNowReadback?.destroy?.();
    state.areaProbeAreaErrReadback?.destroy?.();

    state.areaProbeOffsets = device.createBuffer({ size: (membraneCount + 1) * 4, usage: storageUsage });
    state.areaProbeAreaBase = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeAreaNow = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaProbeAreaErr = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.areaProbeAreaNowReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaProbeAreaErrReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.areaProbeMembraneCapacity = membraneCount;
    state.areaProbeBindGroup = null;
  }

  if ((state.areaProbeNodeCapacity || 0) < nodeCount) {
    const bytes = nodeCount * 4;
    state.areaProbeNodeX?.destroy?.();
    state.areaProbeNodeY?.destroy?.();
    state.areaProbeNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeCapacity = nodeCount;
    state.areaProbeBindGroup = null;
  }

  if ((state.areaProbeIndexCapacity || 0) < indexCount) {
    const bytes = indexCount * 4;
    state.areaProbeLoopIndices?.destroy?.();
    state.areaProbeLoopIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeIndexCapacity = indexCount;
    state.areaProbeBindGroup = null;
  }

  if (!state.areaProbeBindGroup) {
    state.areaProbeBindGroup = device.createBindGroup({
      layout: state.areaProbePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.areaProbeParams } },
        { binding: 1, resource: { buffer: state.areaProbeOffsets } },
        { binding: 2, resource: { buffer: state.areaProbeLoopIndices } },
        { binding: 3, resource: { buffer: state.areaProbeNodeX } },
        { binding: 4, resource: { buffer: state.areaProbeNodeY } },
        { binding: 5, resource: { buffer: state.areaProbeAreaBase } },
        { binding: 6, resource: { buffer: state.areaProbeAreaNow } },
        { binding: 7, resource: { buffer: state.areaProbeAreaErr } },
      ],
    });
  }

  return state;
}

async function dispatchSoftMembranePressureAreaProbe(offload, prep) {
  const device = offload.device;
  const state = await ensureSoftMembranePressureProbeState(offload, prep);
  const membraneCount = prep.plan.membraneCount;
  if (membraneCount <= 0) return false;

  const paramsBytes = new ArrayBuffer(16);
  new Uint32Array(paramsBytes)[0] = membraneCount >>> 0;

  device.queue.writeBuffer(state.areaProbeParams, 0, paramsBytes);
  device.queue.writeBuffer(state.areaProbeOffsets, 0, prep.layout.membraneOffsets);
  if (prep.layout.loopIndices.length > 0) {
    device.queue.writeBuffer(state.areaProbeLoopIndices, 0, prep.layout.loopIndices);
  }
  if (prep.layout.nodeX.length > 0) {
    device.queue.writeBuffer(state.areaProbeNodeX, 0, prep.layout.nodeX);
    device.queue.writeBuffer(state.areaProbeNodeY, 0, prep.layout.nodeY);
  }
  device.queue.writeBuffer(state.areaProbeAreaBase, 0, prep.layout.areaBase);

  const bytes = membraneCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.areaProbePipeline);
  pass.setBindGroup(0, state.areaProbeBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(membraneCount / WGSL_WORKGROUP_SIZE)));
  pass.end();

  encoder.copyBufferToBuffer(state.areaProbeAreaNow, 0, state.areaProbeAreaNowReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.areaProbeAreaErr, 0, state.areaProbeAreaErrReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.areaProbeAreaNowReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const nowMapped = state.areaProbeAreaNowReadback.getMappedRange(0, bytes);
  const areaNow = new Float32Array(nowMapped.slice(0));
  state.areaProbeAreaNowReadback.unmap();

  await state.areaProbeAreaErrReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const errMapped = state.areaProbeAreaErrReadback.getMappedRange(0, bytes);
  const areaErr = new Float32Array(errMapped.slice(0));
  state.areaProbeAreaErrReadback.unmap();

  state.lastAreaProbeDispatchCount = Math.max(1, Math.ceil(membraneCount / WGSL_WORKGROUP_SIZE));
  state.lastAreaProbeAreaNow = areaNow;
  state.lastAreaProbeAreaErr = areaErr;
  state.lastAreaProbeMembraneCount = membraneCount;
  return true;
}

export function applySoftMembraneCellPressureGpuOnly({
  sim,
  soft,
  loops,
  dtPos,
  signedAreaCurrent,
  clamp,
  membraneCellBasePressureGain = 0.08,
  membraneCellBaseRadialDamping = 0.06,
  wgslOffload,
}) {
  const membranes = sim?.bodies?.softMembraneClusters;
  if (!Array.isArray(membranes) || membranes.length === 0) return 0;
  if (!(sim.softMembraneAreaBaseline instanceof Map)) sim.softMembraneAreaBaseline = new Map();

  const loopByCluster = new Map();
  for (const loop of loops || []) {
    loopByCluster.set(loop.clusterId ?? 0, loop);
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const prep = buildSoftMembranePressureWgslPrep({
      membranes,
      loopByCluster,
      softNodes: soft?.nodes,
      softMembraneAreaBaseline: sim.softMembraneAreaBaseline,
    });
    wgslOffload.state.preparedPlan = prep.plan;
    wgslOffload.state.preparedLayout = prep.layout;
    wgslOffload.state.lastPreparedMembraneCount = prep.plan.membraneCount;
    wgslOffload.state.lastPreparedLoopIndexCount = prep.plan.indexCount;
    wgslOffload.state.lastPreparedLayoutBytes = prep.layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared';

    if (canUseWgslOffload(wgslOffload) && prep.plan.membraneCount > 0) {
      const serializedDispatch = (wgslOffload.state.pendingWgslAreaProbePromise || Promise.resolve())
        .catch(() => {})
        .then(() => dispatchSoftMembranePressureAreaProbe(wgslOffload, prep))
        .then((ran) => {
          wgslOffload.state.lastError = null;
          wgslOffload.state.lastMode = ran ? 'wgsl-area-probe' : 'cpu-membrane-authoritative';
        })
        .catch((err) => {
          wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
          wgslOffload.state.lastMode = 'cpu-membrane-authoritative';
        });
      wgslOffload.state.pendingWgslAreaProbePromise = serializedDispatch;
    }
  }

  let touched = 0;
  for (const membrane of membranes) {
    const cid = Number(membrane?.clusterId);
    if (!Number.isInteger(cid)) continue;
    const loop = loopByCluster.get(cid);
    if (!loop || !Array.isArray(loop.indices) || loop.indices.length < 3) continue;

    const areaNow = Math.abs(signedAreaCurrent(soft.nodes, loop.indices));
    if (!Number.isFinite(areaNow) || areaNow < 1e-6) continue;

    const seededBase = Math.max(1e-4, Number(membrane?.restArea) || areaNow);
    if (!sim.softMembraneAreaBaseline.has(cid)) {
      sim.softMembraneAreaBaseline.set(cid, seededBase);
    }
    const areaBase = Math.max(1e-4, Number(sim.softMembraneAreaBaseline.get(cid)) || seededBase);
    const err = clamp((areaBase - areaNow) / areaBase, -0.65, 0.65);
    if (Math.abs(err) < 1e-4) continue;

    const pressureGain = Math.max(0.005, Number(membrane?.pressureGain) || membraneCellBasePressureGain);
    const radialDamping = clamp(Number(membrane?.radialDamping) || membraneCellBaseRadialDamping, 0, 0.2);
    const gain = pressureGain * (1 + Math.min(1.4, Math.abs(err) * 2.2));

    let cx = 0;
    let cy = 0;
    for (const ni of loop.indices) {
      const node = soft.nodes[ni];
      if (!node) continue;
      cx += node.x;
      cy += node.y;
    }
    cx /= loop.indices.length;
    cy /= loop.indices.length;

    for (let k = 0; k < loop.indices.length; k++) {
      const iPrev = loop.indices[(k - 1 + loop.indices.length) % loop.indices.length];
      const iCurr = loop.indices[k];
      const iNext = loop.indices[(k + 1) % loop.indices.length];
      const prev = soft.nodes[iPrev];
      const curr = soft.nodes[iCurr];
      const next = soft.nodes[iNext];
      if (!prev || !curr || !next) continue;

      // Outward normal for CCW loop via right-normal accumulation.
      let nx = (curr.y - prev.y) + (next.y - curr.y);
      let ny = -((curr.x - prev.x) + (next.x - curr.x));
      let nLen = Math.hypot(nx, ny);
      if (!Number.isFinite(nLen) || nLen < 1e-6) {
        nx = curr.x - cx;
        ny = curr.y - cy;
        nLen = Math.hypot(nx, ny);
      }
      if (!Number.isFinite(nLen) || nLen < 1e-6) continue;
      nx /= nLen;
      ny /= nLen;

      const invMass = 1 / Math.max(0.02, curr.mass || 1);
      const impulse = err * gain * dtPos * invMass;
      curr.vx += nx * impulse;
      curr.vy += ny * impulse;

      if (radialDamping > 0) {
        const rv = curr.vx * nx + curr.vy * ny;
        curr.vx -= nx * rv * radialDamping;
        curr.vy -= ny * rv * radialDamping;
      }
    }

    touched += 1;
  }

  return touched;
}
