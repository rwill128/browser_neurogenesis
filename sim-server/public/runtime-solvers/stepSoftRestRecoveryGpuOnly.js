/**
 * GPU-only runtime soft spring-rest recovery pass.
 *
 * Owns soft spring-rest stabilization parameterization for the isolated gpu-only
 * runtime solver path so baseline/default stepping remains untouched.
 */

const WGSL_WORKGROUP_SIZE = 64;

const softRestRecoveryProbeWgsl = /* wgsl */`
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
@group(0) @binding(5) var<storage, read> springRestBaseline: array<f32>;
@group(0) @binding(6) var<storage, read_write> strainOut: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let si = gid.x;
  if (si >= params.springCount) { return; }

  let ia = springNodeA[si];
  let ib = springNodeB[si];
  if (ia >= params.nodeCount || ib >= params.nodeCount) {
    strainOut[si] = 0.0;
    return;
  }

  let rest = max(abs(springRestBaseline[si]), 1e-6);
  let dx = nodeX[ib] - nodeX[ia];
  let dy = nodeY[ib] - nodeY[ia];
  let d = max(length(vec2<f32>(dx, dy)), 1e-6);
  strainOut[si] = (d - rest) / rest;
}
`;

function pickProfile(profile, baseline, warning, severe) {
  return profile === 'baseline' ? baseline : (profile === 'warning' ? warning : severe);
}

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined' || typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function buildSoftRestRecoveryWgslPlan({ springs, softNodes } = {}) {
  const nodes = Array.isArray(softNodes) ? softNodes : [];
  const springList = Array.isArray(springs) ? springs : [];
  const springNodeA = [];
  const springNodeB = [];
  const activeSpringIndices = [];

  for (let si = 0; si < springList.length; si++) {
    const spring = springList[si];
    const i = Number(spring?.[0]);
    const j = Number(spring?.[1]);
    if (!Number.isInteger(i) || !Number.isInteger(j)) continue;
    if (i < 0 || j < 0 || i >= nodes.length || j >= nodes.length) continue;
    springNodeA.push(i);
    springNodeB.push(j);
    activeSpringIndices.push(si);
  }

  return {
    nodeCount: nodes.length,
    springCount: springList.length,
    activeSpringCount: activeSpringIndices.length,
    activeSpringIndices: Uint32Array.from(activeSpringIndices),
    springNodeA: Uint32Array.from(springNodeA),
    springNodeB: Uint32Array.from(springNodeB),
  };
}

function buildSoftRestRecoveryWgslLayout({ plan, restBaseline } = {}) {
  const activeIndices = plan?.activeSpringIndices instanceof Uint32Array ? plan.activeSpringIndices : new Uint32Array(0);
  const restByActiveSpring = new Float32Array(activeIndices.length);
  for (let i = 0; i < activeIndices.length; i++) {
    const si = activeIndices[i];
    restByActiveSpring[i] = Number(restBaseline?.[si]) || 0;
  }

  return {
    springNodeA: plan?.springNodeA instanceof Uint32Array ? plan.springNodeA : new Uint32Array(0),
    springNodeB: plan?.springNodeB instanceof Uint32Array ? plan.springNodeB : new Uint32Array(0),
    restByActiveSpring,
    byteLength:
      (plan?.springNodeA?.byteLength || 0)
      + (plan?.springNodeB?.byteLength || 0)
      + restByActiveSpring.byteLength,
  };
}

function ensureProbeBuffers(offload, nodeCount, springCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;
  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  const uniformUsage = globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST;

  const nodeCapacity = Math.max(1, nodeCount);
  if ((state.probeNodeCapacity || 0) < nodeCapacity) {
    const cap = Math.max(nodeCapacity, state.probeNodeCapacity ? state.probeNodeCapacity * 2 : 256);
    const bytes = cap * 4;
    state.probeNodeX?.destroy?.();
    state.probeNodeY?.destroy?.();
    state.probeNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeNodeCapacity = cap;
    state.probeBindGroup = null;
  }

  const springCapacity = Math.max(1, springCount);
  if ((state.probeSpringCapacity || 0) < springCapacity) {
    const cap = Math.max(springCapacity, state.probeSpringCapacity ? state.probeSpringCapacity * 2 : 256);
    const bytes = cap * 4;
    state.probeSpringNodeA?.destroy?.();
    state.probeSpringNodeB?.destroy?.();
    state.probeSpringRest?.destroy?.();
    state.probeStrainOut?.destroy?.();
    state.probeStrainReadback?.destroy?.();
    state.probeSpringNodeA = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeSpringNodeB = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeSpringRest = device.createBuffer({ size: bytes, usage: storageUsage });
    state.probeStrainOut = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.probeStrainReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.probeSpringCapacity = cap;
    state.probeBindGroup = null;
  }

  if (!state.probeParams) {
    state.probeParams = device.createBuffer({ size: 16, usage: uniformUsage });
    state.probeBindGroup = null;
  }

  return state;
}

async function dispatchSoftRestRecoveryWgslProbe({ softNodes, layout, offload }) {
  if (!canUseWgslOffload(offload)) return false;
  const nodes = Array.isArray(softNodes) ? softNodes : [];
  const springCount = Number(layout?.restByActiveSpring?.length) || 0;
  if (nodes.length <= 0 || springCount <= 0) return false;

  const state = ensureProbeBuffers(offload, nodes.length, springCount);
  const device = offload.device;

  if (!state.probePipeline) {
    const module = device.createShaderModule({ code: softRestRecoveryProbeWgsl });
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
        { binding: 6, resource: { buffer: state.probeStrainOut } },
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
  device.queue.writeBuffer(state.probeSpringNodeA, 0, layout.springNodeA);
  device.queue.writeBuffer(state.probeSpringNodeB, 0, layout.springNodeB);
  device.queue.writeBuffer(state.probeSpringRest, 0, layout.restByActiveSpring);

  const dispatchCount = Math.ceil(springCount / WGSL_WORKGROUP_SIZE);
  const bytes = springCount * 4;

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.probePipeline);
  pass.setBindGroup(0, state.probeBindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();
  encoder.copyBufferToBuffer(state.probeStrainOut, 0, state.probeStrainReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await state.probeStrainReadback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = state.probeStrainReadback.getMappedRange(0, bytes);
  const strainByActiveSpring = new Float32Array(mapped.slice(0));
  state.probeStrainReadback.unmap();

  let absMax = 0;
  let absSum = 0;
  for (let i = 0; i < strainByActiveSpring.length; i++) {
    const abs = Math.abs(strainByActiveSpring[i]);
    absSum += abs;
    if (abs > absMax) absMax = abs;
  }

  state.lastProbeDispatch = dispatchCount;
  state.lastProbeSpringCount = springCount;
  state.lastProbeAbsMean = springCount > 0 ? absSum / springCount : 0;
  state.lastProbeAbsMax = absMax;
  state.lastProbeStrainByActiveSpring = strainByActiveSpring;
  state.lastProbeSource = 'wgsl-rest-recovery-probe';
  return true;
}

export function buildSoftRestRecoveryOptionsGpuOnly({
  severeInterventionsOn,
  warningInterventionsOn,
  deform = null,
} = {}) {
  const severeProfile = severeInterventionsOn && (deform?.severeCollapseCount > 0);
  const warningProfile = warningInterventionsOn && !severeProfile && (deform?.warningCount > 0);
  const profile = severeProfile ? 'severe' : (warningProfile ? 'warning' : 'baseline');

  return {
    recoverRate: pickProfile(profile, 0.056, 0.036, 0.015),
    hardMinFactor: 0.7,
    hardMaxFactor: 1.45,
    jitterDeadband: 1e-5,
    adaptiveGainMax: pickProfile(profile, 2.4, 1.85, 1.35),
    adaptiveExponent: pickProfile(profile, 0.78, 0.9, 1.0),
    elongationBiasMax: pickProfile(profile, 1.22, 1.14, 1.08),
    compressionBiasMax: pickProfile(profile, 1.12, 1.08, 1.04),
    errorPivot: 0.16,
    outlierRecoveryCouplingMax: pickProfile(profile, 1.22, 1.12, 1.05),
    outlierErrorPivot: pickProfile(profile, 0.75, 0.85, 0.95),
    localEndpointCouplingMax: pickProfile(profile, 1.16, 1.08, 1.03),
    localDirectionalCouplingMax: pickProfile(profile, 1.1, 1.06, 1.02),
    localImbalanceCouplingMax: pickProfile(profile, 1.12, 1.07, 1.02),
    localConsensusCouplingMax: pickProfile(profile, 1.08, 1.04, 1.01),
    localOutlierCouplingMax: pickProfile(profile, 1.12, 1.07, 1.02),
    localOutlierErrorPivot: pickProfile(profile, 0.85, 0.95, 1.1),
    localErrorPivot: pickProfile(profile, 0.2, 0.25, 0.3),
    counterPolarityCouplingMax: pickProfile(profile, 1.09, 1.05, 1.02),
    smallRestRecoveryCouplingMax: pickProfile(profile, 1.14, 1.08, 1.0),
    smallRestPivot: pickProfile(profile, 0.9, 1.05, 1.2),
    lowErrorRecoveryCouplingMax: pickProfile(profile, 1.12, 1.06, 1.0),
    lowErrorRecoveryGate: pickProfile(profile, 0.08, 0.065, 0.05),
  };
}

export function applySoftRestRecoveryGpuOnly({
  springs,
  softNodes,
  restBaseline,
  severeInterventionsOn,
  warningInterventionsOn,
  deform,
  recoverSoftSpringRests,
  wgslOffload,
}) {
  if (!Array.isArray(springs) || !restBaseline || springs.length !== restBaseline.length) return;
  if (typeof recoverSoftSpringRests !== 'function') {
    throw new Error('gpu-only soft rest-recovery pass requires recoverSoftSpringRests callback');
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const plan = buildSoftRestRecoveryWgslPlan({ springs, softNodes });
    const layout = buildSoftRestRecoveryWgslLayout({ plan, restBaseline });
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastPreparedActiveSpringCount = plan.activeSpringCount;

    if (canUseWgslOffload(wgslOffload) && plan.activeSpringCount > 0) {
      const serializedProbe = (wgslOffload.state.pendingWgslRestRecoveryProbePromise || Promise.resolve())
        .then(async () => {
          const probeRan = await dispatchSoftRestRecoveryWgslProbe({ softNodes, layout, offload: wgslOffload });
          wgslOffload.state.lastError = null;
          wgslOffload.state.lastMode = probeRan ? 'wgsl-rest-recovery-probe' : 'cpu-rest-recovery-authoritative';
          if (!probeRan) wgslOffload.state.lastProbeSource = 'cpu-rest-recovery-authoritative';
        })
        .catch((err) => {
          wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
          wgslOffload.state.lastMode = 'cpu-rest-recovery-authoritative';
          wgslOffload.state.lastProbeSource = 'cpu-rest-recovery-authoritative';
        });
      wgslOffload.state.pendingWgslRestRecoveryProbePromise = serializedProbe;
    }
  }

  const options = buildSoftRestRecoveryOptionsGpuOnly({
    severeInterventionsOn,
    warningInterventionsOn,
    deform,
  });
  recoverSoftSpringRests(springs, restBaseline, options);

  if (wgslOffload?.state) {
    wgslOffload.state.lastAuthoritativeSource = 'cpu-rest-recovery-authoritative';
    if (!wgslOffload.state.lastMode) wgslOffload.state.lastMode = 'cpu-rest-recovery-authoritative';
    if (!wgslOffload.state.lastProbeSource) wgslOffload.state.lastProbeSource = 'cpu-rest-recovery-authoritative';
  }
}
