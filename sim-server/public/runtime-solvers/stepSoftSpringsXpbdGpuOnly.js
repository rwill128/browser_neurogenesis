/**
 * GPU-only runtime soft spring XPBD pass.
 * Keeps spring-constraint stepping isolated for the gpu-only runtime path while
 * preserving baseline/default behavior and call contracts.
 */
const WGSL_WORKGROUP_SIZE = 64;

const softSpringStretchProbeWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  springCount: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodeX: array<f32>;
@group(0) @binding(2) var<storage, read> nodeY: array<f32>;
@group(0) @binding(3) var<storage, read> springNodeA: array<u32>;
@group(0) @binding(4) var<storage, read> springNodeB: array<u32>;
@group(0) @binding(5) var<storage, read> springRest: array<f32>;
@group(0) @binding(6) var<storage, read_write> stretchOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let si = gid.x;
  if (si >= params.springCount) { return; }

  let ia = springNodeA[si];
  let ib = springNodeB[si];
  if (ia >= params.nodeCount || ib >= params.nodeCount) {
    stretchOut[si] = 0.0;
    return;
  }

  let dx = nodeX[ib] - nodeX[ia];
  let dy = nodeY[ib] - nodeY[ia];
  let d = max(length(vec2<f32>(dx, dy)), 1e-6);
  let rest = max(abs(springRest[si]), 1e-6);
  stretchOut[si] = (d - rest) / rest;
}
`;

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined') return false;
  if (typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function ensureProbeBuffers(offload, nodeCount, springCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const requiredNodeCapacity = Math.max(1, nodeCount);
  if ((state.probeNodeCapacity || 0) < requiredNodeCapacity) {
    const capacity = Math.max(requiredNodeCapacity, state.probeNodeCapacity ? state.probeNodeCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.probeNodeX?.destroy?.();
    state.probeNodeY?.destroy?.();
    state.probeNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeNodeCapacity = capacity;
    state.probeBindGroup = null;
  }

  const requiredSpringCapacity = Math.max(1, springCount);
  if ((state.probeSpringCapacity || 0) < requiredSpringCapacity) {
    const capacity = Math.max(requiredSpringCapacity, state.probeSpringCapacity ? state.probeSpringCapacity * 2 : 256);
    const bytes = capacity * 4;
    state.probeSpringNodeA?.destroy?.();
    state.probeSpringNodeB?.destroy?.();
    state.probeSpringRest?.destroy?.();
    state.probeStretchOut?.destroy?.();
    state.probeStretchReadback?.destroy?.();
    state.probeSpringNodeA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeSpringNodeB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeSpringRest = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeStretchOut = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC,
    });
    state.probeStretchReadback = device.createBuffer({
      size: bytes,
      usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ,
    });
    state.probeSpringCapacity = capacity;
    state.probeBindGroup = null;
  }

  if (!state.probeParams) {
    state.probeParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.probeBindGroup = null;
  }

  return state;
}

async function dispatchSoftSpringWgslProbe({ soft, offload, layout }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springCount = Number(layout?.springRestByColor?.length) || 0;
  if (nodes.length <= 0 || springCount <= 0) return false;

  const state = ensureProbeBuffers(offload, nodes.length, springCount);
  const device = offload.device;

  if (!state.probePipeline) {
    const module = device.createShaderModule({ code: softSpringStretchProbeWgsl });
    state.probePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.probeBindGroup = null;
  }

  if (!state.probeBindGroup) {
    state.probeBindGroup = device.createBindGroup({
      layout: state.probePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.probeParams } },
        { binding: 1, resource: { buffer: state.probeNodeX } },
        { binding: 2, resource: { buffer: state.probeNodeY } },
        { binding: 3, resource: { buffer: state.probeSpringNodeA } },
        { binding: 4, resource: { buffer: state.probeSpringNodeB } },
        { binding: 5, resource: { buffer: state.probeSpringRest } },
        { binding: 6, resource: { buffer: state.probeStretchOut } },
      ],
    });
  }

  const nodeX = new Float32Array(nodes.length);
  const nodeY = new Float32Array(nodes.length);
  for (let ni = 0; ni < nodes.length; ni++) {
    nodeX[ni] = Number(nodes[ni]?.x) || 0;
    nodeY[ni] = Number(nodes[ni]?.y) || 0;
  }

  const params = new Uint32Array(4);
  params[0] = nodes.length >>> 0;
  params[1] = springCount >>> 0;

  device.queue.writeBuffer(state.probeParams, 0, params);
  device.queue.writeBuffer(state.probeNodeX, 0, nodeX);
  device.queue.writeBuffer(state.probeNodeY, 0, nodeY);
  device.queue.writeBuffer(state.probeSpringNodeA, 0, layout.springNodeAByColor);
  device.queue.writeBuffer(state.probeSpringNodeB, 0, layout.springNodeBByColor);
  device.queue.writeBuffer(state.probeSpringRest, 0, layout.springRestByColor);

  const dispatchCount = Math.ceil(springCount / WGSL_WORKGROUP_SIZE);
  const bytes = springCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.probePipeline);
  pass.setBindGroup(0, state.probeBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  encoder.copyBufferToBuffer(state.probeStretchOut, 0, state.probeStretchReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.probeStretchReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = state.probeStretchReadback.getMappedRange(0, bytes);
  const stretchByColor = new Float32Array(mapped.slice(0));
  state.probeStretchReadback.unmap();

  let absMax = 0;
  let absSum = 0;
  for (let i = 0; i < stretchByColor.length; i++) {
    const abs = Math.abs(stretchByColor[i]);
    if (abs > absMax) absMax = abs;
    absSum += abs;
  }

  state.lastProbeNodeCount = nodes.length;
  state.lastProbeSpringCount = springCount;
  state.lastProbeDispatch = dispatchCount;
  state.lastProbeAbsMean = stretchByColor.length > 0 ? absSum / stretchByColor.length : 0;
  state.lastProbeAbsMax = absMax;
  state.lastProbeStretchByColor = stretchByColor;
  state.lastProbeColorBucketCount = Math.max(0, (layout?.springColorOffsets?.length || 1) - 1);
  return true;
}

export function buildSoftSpringXpbdWgslPlan({
  soft,
  skipClusterSet = null,
} = {}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const springs = Array.isArray(soft?.springs) ? soft.springs : [];

  const activeSpringIndices = [];
  const activeNodeAIndices = [];
  const activeNodeBIndices = [];
  const endpointNodeIndicesRaw = [];
  const endpointSpringIndicesRaw = [];
  const endpointSignsRaw = [];

  for (let si = 0; si < springs.length; si++) {
    const spring = springs[si];
    const i = Number(spring?.[0]);
    const j = Number(spring?.[1]);
    const rest = Number(spring?.[2]);
    if (!Number.isInteger(i) || !Number.isInteger(j)) continue;
    if (i < 0 || i >= nodes.length || j < 0 || j >= nodes.length) continue;
    if (!Number.isFinite(rest)) continue;

    const a = nodes[i];
    const b = nodes[j];
    if (!a || !b) continue;
    if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

    const activeIndex = activeSpringIndices.length;
    activeSpringIndices.push(si);
    activeNodeAIndices.push(i);
    activeNodeBIndices.push(j);

    endpointNodeIndicesRaw.push(i, j);
    endpointSpringIndicesRaw.push(activeIndex, activeIndex);
    endpointSignsRaw.push(-1, 1);
  }

  const endpointCount = endpointNodeIndicesRaw.length;
  const nodeEndpointCounts = new Uint32Array(nodes.length);
  for (let ei = 0; ei < endpointCount; ei++) {
    nodeEndpointCounts[endpointNodeIndicesRaw[ei]] += 1;
  }

  const nodeEndpointOffsets = new Uint32Array(nodes.length + 1);
  for (let ni = 0; ni < nodes.length; ni++) {
    nodeEndpointOffsets[ni + 1] = nodeEndpointOffsets[ni] + nodeEndpointCounts[ni];
  }

  const endpointNodeIndices = new Uint32Array(endpointCount);
  const endpointSpringIndices = new Uint32Array(endpointCount);
  const endpointSigns = new Int8Array(endpointCount);
  const cursor = nodeEndpointOffsets.slice(0, nodes.length);

  for (let ei = 0; ei < endpointCount; ei++) {
    const ni = endpointNodeIndicesRaw[ei];
    const dst = cursor[ni]++;
    endpointNodeIndices[dst] = ni;
    endpointSpringIndices[dst] = endpointSpringIndicesRaw[ei];
    endpointSigns[dst] = endpointSignsRaw[ei];
  }

  // Deterministic conflict-free spring coloring (no shared nodes per color).
  // This is a direct unblocker for WGSL spring solves: each color can dispatch
  // in parallel without atomics while preserving Gauss-Seidel ownership order.
  const nodeUsedColors = Array.from({ length: nodes.length }, () => new Set());
  const springColors = new Uint32Array(activeSpringIndices.length);
  let colorCount = 0;
  for (let ai = 0; ai < activeSpringIndices.length; ai++) {
    const i = activeNodeAIndices[ai];
    const j = activeNodeBIndices[ai];
    let color = 0;
    while (nodeUsedColors[i].has(color) || nodeUsedColors[j].has(color)) color++;
    nodeUsedColors[i].add(color);
    nodeUsedColors[j].add(color);
    springColors[ai] = color;
    if (color + 1 > colorCount) colorCount = color + 1;
  }

  const springColorCounts = new Uint32Array(colorCount);
  for (let ai = 0; ai < springColors.length; ai++) springColorCounts[springColors[ai]] += 1;
  const springColorOffsets = new Uint32Array(colorCount + 1);
  for (let ci = 0; ci < colorCount; ci++) springColorOffsets[ci + 1] = springColorOffsets[ci] + springColorCounts[ci];
  const springColorOrderedIndices = new Uint32Array(activeSpringIndices.length);
  const springColorCursor = springColorOffsets.slice(0, colorCount);
  for (let ai = 0; ai < springColors.length; ai++) {
    const ci = springColors[ai];
    const dst = springColorCursor[ci]++;
    springColorOrderedIndices[dst] = ai;
  }

  return {
    nodeCount: nodes.length,
    springCount: springs.length,
    activeSpringCount: activeSpringIndices.length,
    endpointCount,
    activeSpringIndices: Uint32Array.from(activeSpringIndices),
    endpointNodeIndices,
    endpointSpringIndices,
    endpointSigns,
    nodeEndpointOffsets,
    springColors,
    springColorOffsets,
    springColorOrderedIndices,
  };
}

export function buildSoftSpringXpbdWgslLayout({
  soft,
  plan,
} = {}) {
  const nodes = Array.isArray(soft?.nodes) ? soft.nodes : [];
  const activeSpringCount = Number(plan?.activeSpringCount) || 0;
  const activeSpringIndices = plan?.activeSpringIndices instanceof Uint32Array
    ? plan.activeSpringIndices
    : new Uint32Array(0);
  const springColorOrderedIndices = plan?.springColorOrderedIndices instanceof Uint32Array
    ? plan.springColorOrderedIndices
    : new Uint32Array(0);

  const springNodeA = new Uint32Array(activeSpringCount);
  const springNodeB = new Uint32Array(activeSpringCount);
  const springRest = new Float32Array(activeSpringCount);
  const springInvMassA = new Float32Array(activeSpringCount);
  const springInvMassB = new Float32Array(activeSpringCount);

  for (let ai = 0; ai < activeSpringCount; ai++) {
    const si = activeSpringIndices[ai];
    const spring = soft?.springs?.[si];
    const i = Number(spring?.[0]);
    const j = Number(spring?.[1]);
    const rest = Number(spring?.[2]);
    const a = nodes[i];
    const b = nodes[j];

    springNodeA[ai] = Number.isInteger(i) && i >= 0 ? i : 0;
    springNodeB[ai] = Number.isInteger(j) && j >= 0 ? j : 0;
    springRest[ai] = Number.isFinite(rest) ? rest : 0;
    springInvMassA[ai] = 1 / Math.max(0.02, Number(a?.mass) || 1);
    springInvMassB[ai] = 1 / Math.max(0.02, Number(b?.mass) || 1);
  }

  const springNodeAByColor = new Uint32Array(springColorOrderedIndices.length);
  const springNodeBByColor = new Uint32Array(springColorOrderedIndices.length);
  const springRestByColor = new Float32Array(springColorOrderedIndices.length);
  const springInvMassAByColor = new Float32Array(springColorOrderedIndices.length);
  const springInvMassBByColor = new Float32Array(springColorOrderedIndices.length);
  for (let oi = 0; oi < springColorOrderedIndices.length; oi++) {
    const ai = springColorOrderedIndices[oi];
    springNodeAByColor[oi] = springNodeA[ai];
    springNodeBByColor[oi] = springNodeB[ai];
    springRestByColor[oi] = springRest[ai];
    springInvMassAByColor[oi] = springInvMassA[ai];
    springInvMassBByColor[oi] = springInvMassB[ai];
  }

  // WebGPU storage buffers are naturally 4-byte addressed; widen signs now so
  // upcoming WGSL reduction kernels can consume endpoint direction directly.
  const endpointSignsI32 = new Int32Array(plan?.endpointCount || 0);
  for (let ei = 0; ei < endpointSignsI32.length; ei++) {
    endpointSignsI32[ei] = Number(plan?.endpointSigns?.[ei]) < 0 ? -1 : 1;
  }

  return {
    nodeEndpointOffsets: plan?.nodeEndpointOffsets || new Uint32Array(0),
    endpointSpringIndices: plan?.endpointSpringIndices || new Uint32Array(0),
    endpointSignsI32,
    springNodeA,
    springNodeB,
    springRest,
    springInvMassA,
    springInvMassB,
    springColorOffsets: plan?.springColorOffsets || new Uint32Array(0),
    springColorOrderedIndices,
    springNodeAByColor,
    springNodeBByColor,
    springRestByColor,
    springInvMassAByColor,
    springInvMassBByColor,
    byteLength:
      (plan?.nodeEndpointOffsets?.byteLength || 0)
      + (plan?.endpointSpringIndices?.byteLength || 0)
      + endpointSignsI32.byteLength
      + springNodeA.byteLength
      + springNodeB.byteLength
      + springRest.byteLength
      + springInvMassA.byteLength
      + springInvMassB.byteLength
      + (plan?.springColorOffsets?.byteLength || 0)
      + springColorOrderedIndices.byteLength
      + springNodeAByColor.byteLength
      + springNodeBByColor.byteLength
      + springRestByColor.byteLength
      + springInvMassAByColor.byteLength
      + springInvMassBByColor.byteLength,
  };
}

export function applySoftSpringsXPBDVelocityGpuOnly({
  soft,
  dtPos,
  stiffnessScale,
  lambdaCache,
  softXpbdIters,
  softXpbdBaseCompliance,
  clamp,
  skipClusterSet = null,
  wgslOffload,
}) {
  if (!soft?.nodes?.length || !soft?.springs?.length) return;
  if (!Array.isArray(lambdaCache) && !(lambdaCache instanceof Float32Array)) {
    throw new Error('gpu-only soft spring XPBD pass requires lambdaCache array-like');
  }
  if (typeof clamp !== 'function') {
    throw new Error('gpu-only soft spring XPBD pass requires clamp callback');
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    // Unblocker for upcoming WGSL XPBD stage: prepare deterministic CSR endpoint
    // ownership and spring SoA buffers now so the compute stage can run spring
    // solve + node reduction without atomics changing ownership semantics.
    const plan = buildSoftSpringXpbdWgslPlan({ soft, skipClusterSet });
    const layout = buildSoftSpringXpbdWgslLayout({ soft, plan });
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.lastPreparedSpringCount = plan.activeSpringCount;
    wgslOffload.state.lastPreparedEndpointCount = plan.endpointCount;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared';

    // First concrete WGSL stage for soft-spring XPBD path: dispatch a compute
    // probe over color-ordered springs so gpu-only runtime exercises real
    // spring buffer ownership on-device while CPU remains authoritative.
    if (canUseWgslOffload(wgslOffload)) {
      void dispatchSoftSpringWgslProbe({ soft, offload: wgslOffload, layout })
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

  const alpha = (softXpbdBaseCompliance / Math.max(0.2, stiffnessScale)) / Math.max(1e-8, dtPos * dtPos);

  for (let iter = 0; iter < softXpbdIters; iter++) {
    for (let si = 0; si < soft.springs.length; si++) {
      const [i, j, rest] = soft.springs[si];
      const a = soft.nodes[i];
      const b = soft.nodes[j];
      if (!a || !b) continue;
      if (skipClusterSet && (skipClusterSet.has(a.clusterId ?? 0) || skipClusterSet.has(b.clusterId ?? 0))) continue;

      const ax = a.x + a.vx * dtPos;
      const ay = a.y + a.vy * dtPos;
      const bx = b.x + b.vx * dtPos;
      const by = b.y + b.vy * dtPos;

      const dx = bx - ax;
      const dy = by - ay;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = dx / d;
      const ny = dy / d;
      const strainCap = Math.max(0.05, Math.abs(rest) * 0.45);
      const C = clamp(d - rest, -strainCap, strainCap);

      const wA = 1 / Math.max(0.02, a.mass || 1);
      const wB = 1 / Math.max(0.02, b.mass || 1);
      const wSum = wA + wB;
      if (wSum <= 1e-9) continue;

      const lambdaPrev = Number(lambdaCache[si]) || 0;
      let dl = (-C - alpha * lambdaPrev) / (wSum + alpha);
      if (!Number.isFinite(dl)) continue;
      const lambdaNext = Math.max(-20, Math.min(20, lambdaPrev + dl));
      dl = lambdaNext - lambdaPrev;
      lambdaCache[si] = lambdaNext;

      const corrAx = -wA * dl * nx;
      const corrAy = -wA * dl * ny;
      const corrBx = wB * dl * nx;
      const corrBy = wB * dl * ny;

      a.vx += corrAx / dtPos;
      a.vy += corrAy / dtPos;
      b.vx += corrBx / dtPos;
      b.vy += corrBy / dtPos;
    }
  }
}
