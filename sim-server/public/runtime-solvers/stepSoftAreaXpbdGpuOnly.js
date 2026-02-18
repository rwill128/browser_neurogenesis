/**
 * GPU-only runtime soft area XPBD pass.
 * Isolates area-preservation constraint stepping for the gpu-only runtime path
 * while preserving baseline/default behavior and call contracts.
 */
const WGSL_WORKGROUP_SIZE = 64;

const softAreaClusterProbeWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  clusterCount: u32,
  dtPos: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> nodeVX: array<f32>;
@group(0) @binding(4) var<storage, read> nodeVY: array<f32>;
@group(0) @binding(5) var<storage, read> clusterOffsets: array<u32>;
@group(0) @binding(6) var<storage, read> clusterNodeIndices: array<u32>;
@group(0) @binding(7) var<storage, read_write> areaOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= params.clusterCount) { return; }

  let start = clusterOffsets[ci];
  let end = clusterOffsets[ci + 1u];
  if (end <= start + 1u) {
    areaOut[ci] = 0.0;
    return;
  }

  var twiceSignedArea = 0.0;
  for (var ei = start; ei < end; ei = ei + 1u) {
    let ni = clusterNodeIndices[ei];
    let nj = clusterNodeIndices[select(start, ei + 1u, (ei + 1u) < end)];
    if (ni >= params.nodeCount || nj >= params.nodeCount) { continue; }

    let ax = nodeX[ni] + nodeVX[ni] * params.dtPos;
    let ay = nodeY[ni] + nodeVY[ni] * params.dtPos;
    let bx = nodeX[nj] + nodeVX[nj] * params.dtPos;
    let by = nodeY[nj] + nodeVY[nj] * params.dtPos;
    twiceSignedArea += ax * by - bx * ay;
  }

  areaOut[ci] = 0.5 * twiceSignedArea;
}
`;

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined') return false;
  if (typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function ensureSoftAreaProbeBuffers(offload, nodeCount, endpointCount, clusterCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.areaProbeNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.areaProbeNodeCapacity ? state.areaProbeNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaProbeNodeX?.destroy?.();
    state.areaProbeNodeY?.destroy?.();
    state.areaProbeNodeVX?.destroy?.();
    state.areaProbeNodeVY?.destroy?.();
    state.areaProbeNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeVX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeVY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeNodeCapacity = capacity;
    state.areaProbeBindGroup = null;
  }

  const requiredEndpointCapacity = Math.max(1, endpointCount);
  if ((state.areaProbeEndpointCapacity || 0) < requiredEndpointCapacity) {
    const capacity = Math.max(requiredEndpointCapacity, state.areaProbeEndpointCapacity ? state.areaProbeEndpointCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.areaProbeClusterNodeIndices?.destroy?.();
    state.areaProbeClusterNodeIndices = device.createBuffer({ size: bytes, usage: storageUsage });
    state.areaProbeEndpointCapacity = capacity;
    state.areaProbeBindGroup = null;
  }

  const requiredClusterCapacity = Math.max(1, clusterCount);
  if ((state.areaProbeClusterCapacity || 0) < requiredClusterCapacity) {
    const capacity = Math.max(requiredClusterCapacity, state.areaProbeClusterCapacity ? state.areaProbeClusterCapacity * 2 : 64);
    const bytes = capacity * 4;
    state.areaProbeClusterOffsets?.destroy?.();
    state.areaProbeAreaOut?.destroy?.();
    state.areaProbeAreaReadback?.destroy?.();
    state.areaProbeClusterOffsets = device.createBuffer({ size: (capacity + 1) * 4, usage: storageUsage });
    state.areaProbeAreaOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.areaProbeAreaReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.areaProbeClusterCapacity = capacity;
    state.areaProbeBindGroup = null;
  }

  if (!state.areaProbeParams) {
    state.areaProbeParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.areaProbeBindGroup = null;
  }

  return state;
}

async function dispatchSoftAreaWgslProbe({ sim, soft, offload, plan, dtPos }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const clusterCount = Number(plan?.clusterCount) || 0;
  const endpointCount = Number(plan?.endpointCount) || 0;
  if (nodes.length <= 0 || clusterCount <= 0 || endpointCount <= 0) return false;

  const state = ensureSoftAreaProbeBuffers(offload, nodes.length, endpointCount, clusterCount);
  const device = offload.device;

  if (!state.areaProbePipeline) {
    const module = device.createShaderModule({ code: softAreaClusterProbeWgsl });
    state.areaProbePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.areaProbeBindGroup = null;
  }

  if (!state.areaProbeBindGroup) {
    state.areaProbeBindGroup = device.createBindGroup({
      layout: state.areaProbePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.areaProbeParams } },
        { binding: 1, resource: { buffer: state.areaProbeNodeX } },
        { binding: 2, resource: { buffer: state.areaProbeNodeY } },
        { binding: 3, resource: { buffer: state.areaProbeNodeVX } },
        { binding: 4, resource: { buffer: state.areaProbeNodeVY } },
        { binding: 5, resource: { buffer: state.areaProbeClusterOffsets } },
        { binding: 6, resource: { buffer: state.areaProbeClusterNodeIndices } },
        { binding: 7, resource: { buffer: state.areaProbeAreaOut } },
      ],
    });
  }

  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  const nodeVX = new Float32Array(nodes.length);
  const nodeVY = new Float32Array(nodes.length);
  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni] || {};
    nodeX[ni] = Number(node.x) || 0;
    nodeY[ni] = Number(node.y) || 0;
    nodeVX[ni] = Number(node.vx) || 0;
    nodeVY[ni] = Number(node.vy) || 0;
  }

  const paramsBytes = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  paramsU32[0] = nodes.length >>> 0;
  paramsU32[1] = clusterCount >>> 0;
  paramsF32[2] = Number.isFinite(dtPos) ? dtPos : 0;

  device.queue.writeBuffer(state.areaProbeParams, 0, paramsBytes);
  device.queue.writeBuffer(state.areaProbeNodeX, 0, nodeX);
  device.queue.writeBuffer(state.areaProbeNodeY, 0, nodeY);
  device.queue.writeBuffer(state.areaProbeNodeVX, 0, nodeVX);
  device.queue.writeBuffer(state.areaProbeNodeVY, 0, nodeVY);
  device.queue.writeBuffer(state.areaProbeClusterOffsets, 0, plan.clusterOffsets);
  device.queue.writeBuffer(state.areaProbeClusterNodeIndices, 0, plan.clusterNodeIndices);

  const dispatchCount = Math.ceil(clusterCount / WGSL_WORKGROUP_SIZE);
  const bytes = clusterCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.areaProbePipeline);
  pass.setBindGroup(0, state.areaProbeBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  encoder.copyBufferToBuffer(state.areaProbeAreaOut, 0, state.areaProbeAreaReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.areaProbeAreaReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = state.areaProbeAreaReadback.getMappedRange(0, bytes);
  const areaByCluster = new Float32Array(mapped.slice(0));
  state.areaProbeAreaReadback.unmap();

  const restByCluster = plan.clusterRestArea instanceof Float32Array ? plan.clusterRestArea : new Float32Array(clusterCount);
  const clusterErrors = new Float32Array(clusterCount);
  let maxAbsError = 0;
  let sumAbsError = 0;
  for (let ci = 0; ci < clusterCount; ci++) {
    const rest = restByCluster[ci] || 0;
    const err = areaByCluster[ci] - rest;
    clusterErrors[ci] = err;
    const abs = Math.abs(err);
    if (abs > maxAbsError) maxAbsError = abs;
    sumAbsError += abs;
  }

  state.lastAreaProbeClusterCount = clusterCount;
  state.lastAreaProbeEndpointCount = endpointCount;
  state.lastAreaProbeDispatch = dispatchCount;
  state.lastAreaProbeAreaByCluster = areaByCluster;
  state.lastAreaProbeConstraintErrorByCluster = clusterErrors;
  state.lastAreaProbeAbsErrorMean = clusterCount > 0 ? sumAbsError / clusterCount : 0;
  state.lastAreaProbeAbsErrorMax = maxAbsError;
  state.lastAreaProbeLambdaSeedByCluster = plan.clusterLambda;
  state.lastAreaProbeRestAreaByCluster = restByCluster;
  state.lastAreaProbeSoftAreaLambdaSize = Number(sim?.softAreaLambda?.size) || 0;
  return true;
}

function signedAreaPredicted(nodes, indices, dtPos) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const a = nodes[indices[i]];
    const b = nodes[indices[(i + 1) % indices.length]];
    const ax = a.x + a.vx * dtPos;
    const ay = a.y + a.vy * dtPos;
    const bx = b.x + b.vx * dtPos;
    const by = b.y + b.vy * dtPos;
    s += ax * by - bx * ay;
  }
  return 0.5 * s;
}

function buildSoftAreaXpbdWgslPlan({ sim, soft, loops }) {
  const loopList = Array.isArray(loops) ? loops : [];
  const nodes = soft?.nodes || [];
  const clusterOffsets = new Uint32Array(loopList.length + 1);
  let endpointCount = 0;
  for (let li = 0; li < loopList.length; li++) {
    const len = Math.max(0, Number(loopList[li]?.indices?.length) || 0);
    endpointCount += len;
    clusterOffsets[li + 1] = endpointCount;
  }

  const clusterNodeIndices = new Uint32Array(endpointCount);
  const clusterNodeInvMass = new Float32Array(endpointCount);
  const clusterRestArea = new Float32Array(loopList.length);
  const clusterLambda = new Float32Array(loopList.length);

  let write = 0;
  for (let li = 0; li < loopList.length; li++) {
    const loop = loopList[li];
    const ids = Array.isArray(loop?.indices) ? loop.indices : [];
    const cid = loop?.clusterId;
    clusterRestArea[li] = Number(sim?.softAreaRest?.get?.(cid)) || 0;
    clusterLambda[li] = Number(sim?.softAreaLambda?.get?.(cid)) || 0;

    for (let k = 0; k < ids.length; k++) {
      const ni = Number(ids[k]) || 0;
      clusterNodeIndices[write] = ni;
      const node = nodes[ni];
      clusterNodeInvMass[write] = 1 / Math.max(0.02, Number(node?.mass) || 1);
      write += 1;
    }
  }

  return {
    clusterCount: loopList.length,
    endpointCount,
    clusterOffsets,
    clusterNodeIndices,
    clusterNodeInvMass,
    clusterRestArea,
    clusterLambda,
  };
}

function buildSoftAreaXpbdWgslLayout(plan) {
  return {
    clusterOffsets: plan.clusterOffsets,
    clusterNodeIndices: plan.clusterNodeIndices,
    clusterNodeInvMass: plan.clusterNodeInvMass,
    clusterRestArea: plan.clusterRestArea,
    clusterLambda: plan.clusterLambda,
    byteLength:
      plan.clusterOffsets.byteLength
      + plan.clusterNodeIndices.byteLength
      + plan.clusterNodeInvMass.byteLength
      + plan.clusterRestArea.byteLength
      + plan.clusterLambda.byteLength,
  };
}

export function applySoftAreaXPBDVelocityGpuOnly({
  sim,
  soft,
  loops,
  dtPos,
  stiffnessScale,
  softAreaXpbdIters,
  softAreaBaseCompliance,
  wgslOffload,
}) {
  if (!loops?.length) return;
  const alpha = (softAreaBaseCompliance / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    // WGSL prep ownership for area-XPBD offload: deterministic CSR loop layout
    // plus per-cluster rest/lambda seeds shared by probe + future solve kernels.
    const plan = buildSoftAreaXpbdWgslPlan({ sim, soft, loops });
    const layout = buildSoftAreaXpbdWgslLayout(plan);
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.lastPreparedClusterCount = plan.clusterCount;
    wgslOffload.state.lastPreparedEndpointCount = plan.endpointCount;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared';

    // Concrete WGSL area stage: per-cluster predicted signed-area probe.
    // CPU remains authoritative for lambda + velocity updates until reduction
    // kernels are landed; this probe validates deterministic geometry telemetry.
    if (canUseWgslOffload(wgslOffload)) {
      void dispatchSoftAreaWgslProbe({ sim, soft, offload: wgslOffload, plan, dtPos })
        .then((ran) => {
          if (ran) {
            wgslOffload.state.lastError = null;
            wgslOffload.state.lastMode = 'wgsl-probe';
          }
        })
        .catch((err) => {
          wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
          wgslOffload.state.lastMode = 'cpu-fallback';
        });
    }
  }

  for (let iter = 0; iter < softAreaXpbdIters; iter++) {
    for (const loop of loops) {
      const ids = loop.indices;
      const m = ids.length;
      if (m < 3) continue;
      const restArea = sim.softAreaRest.get(loop.clusterId);
      if (!Number.isFinite(restArea)) continue;

      const area = signedAreaPredicted(soft.nodes, ids, dtPos);
      const C = area - restArea;

      const gradX = new Array(m);
      const gradY = new Array(m);
      let sumWGrad2 = 0;

      for (let k = 0; k < m; k++) {
        const prev = soft.nodes[ids[(k - 1 + m) % m]];
        const next = soft.nodes[ids[(k + 1) % m]];
        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nx = next.x + next.vx * dtPos;
        const ny = next.y + next.vy * dtPos;
        const gx = 0.5 * (ny - py);
        const gy = 0.5 * (px - nx);
        gradX[k] = gx;
        gradY[k] = gy;

        const node = soft.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        sumWGrad2 += w * (gx * gx + gy * gy);
      }

      if (sumWGrad2 <= 1e-10) continue;

      const lambdaPrev = Number(sim.softAreaLambda.get(loop.clusterId)) || 0;
      let dl = (-C - alpha * lambdaPrev) / (sumWGrad2 + alpha);
      if (!Number.isFinite(dl)) continue;
      dl = Math.max(-2.0, Math.min(2.0, dl));
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      sim.softAreaLambda.set(loop.clusterId, lambdaNext);

      for (let k = 0; k < m; k++) {
        const node = soft.nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        node.vx += (w * gradX[k] * dl) / dtPos;
        node.vy += (w * gradY[k] * dl) / dtPos;
      }
    }
  }
}
