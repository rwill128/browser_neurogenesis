import { applyRigidInsideCorrectionPassGpuOnly } from './stepRigidInsideCorrectionGpuOnly.js';
import {
  buildSoftClusterKinematicsWgslPrep,
  computeSoftClusterKinematicsGpuOnly,
  projectNodesTowardClusterRigidMotionGpuOnly,
} from './stepSoftClusterKinematicsGpuOnly.js';
import { applySoftMembraneInsideCorrectionPassGpuOnly } from './stepSoftMembraneInsideCorrectionGpuOnly.js';

const WGSL_WORKGROUP_SIZE = 64;

const SOFT_CLUSTER_KINEMATICS_PROBE_WGSL = /* wgsl */`
struct Params {
  cluster_count: u32,
  _pad0: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cluster_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> node_x: array<f32>;
@group(0) @binding(3) var<storage, read> node_y: array<f32>;
@group(0) @binding(4) var<storage, read> node_vx: array<f32>;
@group(0) @binding(5) var<storage, read> node_vy: array<f32>;
@group(0) @binding(6) var<storage, read> node_mass: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_mass: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_x_mass: array<f32>;
@group(0) @binding(9) var<storage, read_write> out_y_mass: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_vx_mass: array<f32>;
@group(0) @binding(11) var<storage, read_write> out_vy_mass: array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= params.cluster_count) {
    return;
  }

  let start = cluster_offsets[ci];
  let stop = cluster_offsets[ci + 1u];

  var sum_mass = 0.0;
  var sum_x_mass = 0.0;
  var sum_y_mass = 0.0;
  var sum_vx_mass = 0.0;
  var sum_vy_mass = 0.0;

  for (var i = start; i < stop; i = i + 1u) {
    let m = max(0.02, node_mass[i]);
    sum_mass = sum_mass + m;
    sum_x_mass = sum_x_mass + node_x[i] * m;
    sum_y_mass = sum_y_mass + node_y[i] * m;
    sum_vx_mass = sum_vx_mass + node_vx[i] * m;
    sum_vy_mass = sum_vy_mass + node_vy[i] * m;
  }

  out_mass[ci] = sum_mass;
  out_x_mass[ci] = sum_x_mass;
  out_y_mass[ci] = sum_y_mass;
  out_vx_mass[ci] = sum_vx_mass;
  out_vy_mass[ci] = sum_vy_mass;
}
`;

function canUseWgslOffload(offload) {
  return Boolean(
    offload
      && offload.enabled === true
      && offload.device
      && typeof offload.device.createComputePipelineAsync === 'function'
      && typeof offload.device.createBuffer === 'function'
      && typeof offload.device.createCommandEncoder === 'function'
      && offload.state
      && globalThis.GPUBufferUsage
      && globalThis.GPUMapMode,
  );
}

function computeSoftClusterMassParity(layout, probe) {
  const clusterCount = Math.max(0, (layout?.clusterOffsets?.length || 1) - 1);
  let massAbsMax = 0;
  let xMassAbsMax = 0;
  let yMassAbsMax = 0;
  let vxMassAbsMax = 0;
  let vyMassAbsMax = 0;

  for (let ci = 0; ci < clusterCount; ci++) {
    const start = layout.clusterOffsets[ci] >>> 0;
    const stop = layout.clusterOffsets[ci + 1] >>> 0;
    let mass = 0;
    let xMass = 0;
    let yMass = 0;
    let vxMass = 0;
    let vyMass = 0;
    for (let i = start; i < stop; i++) {
      const m = Math.max(0.02, Number(layout.nodeMass?.[i]) || 0.02);
      mass += m;
      xMass += (Number(layout.nodeX?.[i]) || 0) * m;
      yMass += (Number(layout.nodeY?.[i]) || 0) * m;
      vxMass += (Number(layout.nodeVx?.[i]) || 0) * m;
      vyMass += (Number(layout.nodeVy?.[i]) || 0) * m;
    }
    massAbsMax = Math.max(massAbsMax, Math.abs((probe.mass?.[ci] || 0) - mass));
    xMassAbsMax = Math.max(xMassAbsMax, Math.abs((probe.xMass?.[ci] || 0) - xMass));
    yMassAbsMax = Math.max(yMassAbsMax, Math.abs((probe.yMass?.[ci] || 0) - yMass));
    vxMassAbsMax = Math.max(vxMassAbsMax, Math.abs((probe.vxMass?.[ci] || 0) - vxMass));
    vyMassAbsMax = Math.max(vyMassAbsMax, Math.abs((probe.vyMass?.[ci] || 0) - vyMass));
  }

  return {
    clusterCount,
    massAbsMax,
    xMassAbsMax,
    yMassAbsMax,
    vxMassAbsMax,
    vyMassAbsMax,
  };
}

async function ensureSoftClusterProbeState(offload, prep) {
  const device = offload.device;
  const state = offload.state;
  const clusterCount = Math.max(1, prep.plan.clusterCount);
  const nodeCount = Math.max(1, prep.plan.nodeCount);

  if (!state.softClusterProbePipeline) {
    const module = device.createShaderModule({ code: SOFT_CLUSTER_KINEMATICS_PROBE_WGSL });
    state.softClusterProbePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    state.softClusterProbeBindGroup = null;
  }

  const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST;
  if (!state.softClusterProbeParams) {
    state.softClusterProbeParams = device.createBuffer({
      size: 16,
      usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    state.softClusterProbeBindGroup = null;
  }

  if ((state.softClusterProbeClusterCapacity || 0) < clusterCount) {
    const bytes = clusterCount * 4;
    state.softClusterProbeOffsets?.destroy?.();
    state.softClusterProbeMass?.destroy?.();
    state.softClusterProbeXMass?.destroy?.();
    state.softClusterProbeYMass?.destroy?.();
    state.softClusterProbeVxMass?.destroy?.();
    state.softClusterProbeVyMass?.destroy?.();
    state.softClusterProbeMassReadback?.destroy?.();
    state.softClusterProbeXMassReadback?.destroy?.();
    state.softClusterProbeYMassReadback?.destroy?.();
    state.softClusterProbeVxMassReadback?.destroy?.();
    state.softClusterProbeVyMassReadback?.destroy?.();

    state.softClusterProbeOffsets = device.createBuffer({ size: (clusterCount + 1) * 4, usage: storageUsage });
    state.softClusterProbeMass = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.softClusterProbeXMass = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.softClusterProbeYMass = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.softClusterProbeVxMass = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });
    state.softClusterProbeVyMass = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_SRC });

    state.softClusterProbeMassReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.softClusterProbeXMassReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.softClusterProbeYMassReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.softClusterProbeVxMassReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });
    state.softClusterProbeVyMassReadback = device.createBuffer({ size: bytes, usage: globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ });

    state.softClusterProbeClusterCapacity = clusterCount;
    state.softClusterProbeBindGroup = null;
  }

  if ((state.softClusterProbeNodeCapacity || 0) < nodeCount) {
    const bytes = nodeCount * 4;
    state.softClusterProbeNodeX?.destroy?.();
    state.softClusterProbeNodeY?.destroy?.();
    state.softClusterProbeNodeVx?.destroy?.();
    state.softClusterProbeNodeVy?.destroy?.();
    state.softClusterProbeNodeMass?.destroy?.();

    state.softClusterProbeNodeX = device.createBuffer({ size: bytes, usage: storageUsage });
    state.softClusterProbeNodeY = device.createBuffer({ size: bytes, usage: storageUsage });
    state.softClusterProbeNodeVx = device.createBuffer({ size: bytes, usage: storageUsage });
    state.softClusterProbeNodeVy = device.createBuffer({ size: bytes, usage: storageUsage });
    state.softClusterProbeNodeMass = device.createBuffer({ size: bytes, usage: storageUsage });

    state.softClusterProbeNodeCapacity = nodeCount;
    state.softClusterProbeBindGroup = null;
  }

  if (!state.softClusterProbeBindGroup) {
    state.softClusterProbeBindGroup = device.createBindGroup({
      layout: state.softClusterProbePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.softClusterProbeParams } },
        { binding: 1, resource: { buffer: state.softClusterProbeOffsets } },
        { binding: 2, resource: { buffer: state.softClusterProbeNodeX } },
        { binding: 3, resource: { buffer: state.softClusterProbeNodeY } },
        { binding: 4, resource: { buffer: state.softClusterProbeNodeVx } },
        { binding: 5, resource: { buffer: state.softClusterProbeNodeVy } },
        { binding: 6, resource: { buffer: state.softClusterProbeNodeMass } },
        { binding: 7, resource: { buffer: state.softClusterProbeMass } },
        { binding: 8, resource: { buffer: state.softClusterProbeXMass } },
        { binding: 9, resource: { buffer: state.softClusterProbeYMass } },
        { binding: 10, resource: { buffer: state.softClusterProbeVxMass } },
        { binding: 11, resource: { buffer: state.softClusterProbeVyMass } },
      ],
    });
  }

  return state;
}

async function readF32(readback, bytes) {
  await readback.mapAsync(globalThis.GPUMapMode.READ, 0, bytes);
  const mapped = readback.getMappedRange(0, bytes);
  const values = new Float32Array(mapped.slice(0));
  readback.unmap();
  return values;
}

async function dispatchSoftClusterKinematicsProbe(offload, prep) {
  const device = offload.device;
  const state = await ensureSoftClusterProbeState(offload, prep);
  const clusterCount = prep.plan.clusterCount;
  if (clusterCount <= 0) return false;

  const params = new ArrayBuffer(16);
  new Uint32Array(params)[0] = clusterCount >>> 0;
  device.queue.writeBuffer(state.softClusterProbeParams, 0, params);
  device.queue.writeBuffer(state.softClusterProbeOffsets, 0, prep.layout.clusterOffsets);
  if (prep.layout.nodeX.length > 0) {
    device.queue.writeBuffer(state.softClusterProbeNodeX, 0, prep.layout.nodeX);
    device.queue.writeBuffer(state.softClusterProbeNodeY, 0, prep.layout.nodeY);
    device.queue.writeBuffer(state.softClusterProbeNodeVx, 0, prep.layout.nodeVx);
    device.queue.writeBuffer(state.softClusterProbeNodeVy, 0, prep.layout.nodeVy);
    device.queue.writeBuffer(state.softClusterProbeNodeMass, 0, prep.layout.nodeMass);
  }

  const bytes = clusterCount * 4;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.softClusterProbePipeline);
  pass.setBindGroup(0, state.softClusterProbeBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(clusterCount / WGSL_WORKGROUP_SIZE)));
  pass.end();

  encoder.copyBufferToBuffer(state.softClusterProbeMass, 0, state.softClusterProbeMassReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.softClusterProbeXMass, 0, state.softClusterProbeXMassReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.softClusterProbeYMass, 0, state.softClusterProbeYMassReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.softClusterProbeVxMass, 0, state.softClusterProbeVxMassReadback, 0, bytes);
  encoder.copyBufferToBuffer(state.softClusterProbeVyMass, 0, state.softClusterProbeVyMassReadback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  const probe = {
    mass: await readF32(state.softClusterProbeMassReadback, bytes),
    xMass: await readF32(state.softClusterProbeXMassReadback, bytes),
    yMass: await readF32(state.softClusterProbeYMassReadback, bytes),
    vxMass: await readF32(state.softClusterProbeVxMassReadback, bytes),
    vyMass: await readF32(state.softClusterProbeVyMassReadback, bytes),
  };

  state.lastSoftClusterProbe = probe;
  state.lastSoftClusterProbeParity = computeSoftClusterMassParity(prep.layout, probe);
  state.lastSoftClusterProbeDispatchCount = Math.max(1, Math.ceil(clusterCount / WGSL_WORKGROUP_SIZE));
  state.lastSoftClusterProbeClusterCount = clusterCount;
  state.lastSoftClusterProbeSignature = prep.signature;
  state.lastSoftClusterProbeSource = 'wgsl-soft-cluster-kinematics-probe';
  return true;
}

export async function applyPostCollisionRecoveryGpuOnly(args = {}) {
  const {
    bodies,
    soft,
    dtNorm,
    softMembraneClusterSet,
    softClusterCollisionLinearProjection,
    softClusterCollisionAngularProjection,
    membraneGainScale = 0.72,
    computeSoftClusterKinematics,
    projectNodesTowardClusterRigidMotion,
    applyRigidInsideCorrectionPass,
    applyMembraneInsideCorrectionPass,
    applyBounceBoundary,
    applyCollisionBoundaryPassGpuOnly,
    wgslOffload,
    n,
    softClusterLoops,
  } = args;

  if (!bodies || !soft || !Array.isArray(soft.nodes)) {
    return { rigidInsideCorrections: 0, membraneInsideCorrections: 0, postCollisionClusterKinematics: new Map() };
  }

  const dtNormSafe = Math.max(0, Number(dtNorm) || 0);
  const linearGain = (Number(softClusterCollisionLinearProjection) || 0) * dtNormSafe;
  const angularGain = (Number(softClusterCollisionAngularProjection) || 0) * dtNormSafe;

  const computeKinematics =
    typeof computeSoftClusterKinematics === 'function'
      ? computeSoftClusterKinematics
      : computeSoftClusterKinematicsGpuOnly;
  const projectTowardRigidMotion =
    typeof projectNodesTowardClusterRigidMotion === 'function'
      ? projectNodesTowardClusterRigidMotion
      : projectNodesTowardClusterRigidMotionGpuOnly;

  if (wgslOffload?.enabled === true && wgslOffload?.state) {
    const prep = buildSoftClusterKinematicsWgslPrep(soft.nodes);
    wgslOffload.state.preparedSoftClusterPlan = prep.plan;
    wgslOffload.state.preparedSoftClusterLayout = prep.layout;
    wgslOffload.state.preparedSoftClusterSignature = prep.signature;
    wgslOffload.state.lastPreparedSoftClusterNodeCount = prep.plan.nodeCount;
    wgslOffload.state.lastPreparedSoftClusterCount = prep.plan.clusterCount;
    wgslOffload.state.lastPreparedSoftClusterLayoutBytes = prep.layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared-soft-cluster-kinematics';
    wgslOffload.state.lastSoftClusterProbeSource = 'cpu-prepared-soft-cluster-kinematics';

    if (canUseWgslOffload(wgslOffload) && prep.plan.clusterCount > 0) {
      const serializedDispatch = (wgslOffload.state.pendingSoftClusterProbePromise || Promise.resolve())
        .catch(() => {})
        .then(async () => {
          const probeRan = await dispatchSoftClusterKinematicsProbe(wgslOffload, prep);
          return { probeRan };
        })
        .then(({ probeRan }) => {
          wgslOffload.state.lastError = null;
          if (probeRan) {
            wgslOffload.state.lastMode = 'wgsl-soft-cluster-kinematics-probe';
            wgslOffload.state.lastSoftClusterProbeSource = 'wgsl-soft-cluster-kinematics-probe';
          }
        })
        .catch((err) => {
          wgslOffload.state.lastError = String(err?.message || err || 'unknown-error');
          wgslOffload.state.lastMode = 'cpu-prepared-soft-cluster-kinematics';
          wgslOffload.state.lastSoftClusterProbeSource = 'cpu-prepared-soft-cluster-kinematics';
        });
      wgslOffload.state.pendingSoftClusterProbePromise = serializedDispatch;
    }
  }

  const postCollisionClusterKinematics = await Promise.resolve(computeKinematics(soft.nodes));
  await Promise.resolve(projectTowardRigidMotion(soft.nodes, postCollisionClusterKinematics, {
    linearGain,
    angularGain,
    membraneClusterSet: softMembraneClusterSet,
    membraneGainScale: Number(membraneGainScale) || 0.72,
  }));

  const rigidInsideCorrections =
    typeof applyRigidInsideCorrectionPass === 'function'
      ? applyRigidInsideCorrectionPass(bodies, soft, args.hybridAttachedByRigid)
      : applyRigidInsideCorrectionPassGpuOnly({
        rigidBodies: bodies.rigid,
        soft,
        hybridAttachedByRigid: args.hybridAttachedByRigid,
      });

  const membraneInsideCorrections =
    typeof applyMembraneInsideCorrectionPass === 'function'
      ? applyMembraneInsideCorrectionPass(args.sim, soft, softClusterLoops)
      : applySoftMembraneInsideCorrectionPassGpuOnly({
        sim: args.sim,
        soft,
        loops: softClusterLoops,
      });

  let boundaryRuntime = { mode: 'cpu-inline', reason: 'bounce-callback' };
  if (typeof applyCollisionBoundaryPassGpuOnly === 'function') {
    boundaryRuntime =
      (await Promise.resolve(applyCollisionBoundaryPassGpuOnly({
        rigidBodies: bodies.rigid,
        soft,
        n,
        rigidBounce: 0.84,
        softBounce: 0.78,
        applyBounceBoundary,
        wgslOffload,
      }))) || { mode: 'cpu-fallback', reason: 'unknown' };
  } else if (typeof applyBounceBoundary === 'function') {
    for (const rb of (bodies.rigid || [])) applyBounceBoundary(rb, n, 0.84);
    for (const sn of (soft.nodes || [])) applyBounceBoundary(sn, n, 0.78);
  } else {
    boundaryRuntime = { mode: 'cpu-inline', reason: 'no-boundary-handler' };
  }

  return {
    rigidInsideCorrections,
    membraneInsideCorrections,
    postCollisionClusterKinematics,
    boundaryRuntime,
  };
}
