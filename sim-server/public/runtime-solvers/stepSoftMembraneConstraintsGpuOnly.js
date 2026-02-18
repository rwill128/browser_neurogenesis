const WGSL_WORKGROUP_SIZE = 64;
const SHAPE_LAYOUT_STRIDE_FLOATS = 11;
const MEMBRANE_BOUNDARY_EDGE_LAYOUT_STRIDE_FLOATS = 11;
const MEMBRANE_BEND_LAYOUT_STRIDE_FLOATS = 11;

const softMembraneBoundaryEdgeProposalWgsl = /* wgsl */ `
struct Params {
  count : f32,
  dtPos : f32,
  _pad0 : f32,
  _pad1 : f32,
};

@group(0) @binding(0) var<storage, read> layout : array<f32>;
@group(0) @binding(1) var<storage, read_write> deltaVxAOut : array<f32>;
@group(0) @binding(2) var<storage, read_write> deltaVyAOut : array<f32>;
@group(0) @binding(3) var<storage, read_write> deltaVxBOut : array<f32>;
@group(0) @binding(4) var<storage, read_write> deltaVyBOut : array<f32>;
@group(0) @binding(5) var<storage, read_write> lambdaNextOut : array<f32>;
@group(0) @binding(6) var<uniform> params : Params;

fn finiteOrZero(v : f32) -> f32 {
  if (v == v && abs(v) < 1e20) {
    return v;
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (f32(i) >= params.count) {
    return;
  }
  let base = i * ${MEMBRANE_BOUNDARY_EDGE_LAYOUT_STRIDE_FLOATS}u;
  let ax = finiteOrZero(layout[base + 0u]);
  let ay = finiteOrZero(layout[base + 1u]);
  let bx = finiteOrZero(layout[base + 2u]);
  let by = finiteOrZero(layout[base + 3u]);
  let rest = max(1e-4, finiteOrZero(layout[base + 4u]));
  let wA = max(0.0, finiteOrZero(layout[base + 5u]));
  let wB = max(0.0, finiteOrZero(layout[base + 6u]));
  let lambdaPrev = finiteOrZero(layout[base + 7u]);
  let edgeAlpha = max(0.0, finiteOrZero(layout[base + 8u]));
  let cLimit = max(0.0, finiteOrZero(layout[base + 9u]));
  let lambdaLimit = max(1e-6, finiteOrZero(layout[base + 10u]));

  let dx = bx - ax;
  let dy = by - ay;
  let d = max(1e-6, sqrt(dx * dx + dy * dy));
  let nx = dx / d;
  let ny = dy / d;
  let C = clamp(d - rest, -cLimit, cLimit);
  let wSum = wA + wB;
  if (wSum <= 1e-9) {
    deltaVxAOut[i] = 0.0;
    deltaVyAOut[i] = 0.0;
    deltaVxBOut[i] = 0.0;
    deltaVyBOut[i] = 0.0;
    lambdaNextOut[i] = lambdaPrev;
    return;
  }

  var dl = (-C - edgeAlpha * lambdaPrev) / (wSum + edgeAlpha);
  if (!(dl == dl)) {
    dl = 0.0;
  }
  let lambdaNext = clamp(lambdaPrev + dl, -lambdaLimit, lambdaLimit);
  dl = lambdaNext - lambdaPrev;

  let invDt = 1.0 / max(params.dtPos, 1e-8);
  deltaVxAOut[i] = (-wA * dl * nx) * invDt;
  deltaVyAOut[i] = (-wA * dl * ny) * invDt;
  deltaVxBOut[i] = (wB * dl * nx) * invDt;
  deltaVyBOut[i] = (wB * dl * ny) * invDt;
  lambdaNextOut[i] = lambdaNext;
}
`;


const softMembraneBendProposalWgsl = /* wgsl */ `
struct Params {
  count : f32,
  dtPos : f32,
  _pad0 : f32,
  _pad1 : f32,
};

@group(0) @binding(0) var<storage, read> layout : array<f32>;
@group(0) @binding(1) var<storage, read_write> deltaVxPrevOut : array<f32>;
@group(0) @binding(2) var<storage, read_write> deltaVyPrevOut : array<f32>;
@group(0) @binding(3) var<storage, read_write> deltaVxNextOut : array<f32>;
@group(0) @binding(4) var<storage, read_write> deltaVyNextOut : array<f32>;
@group(0) @binding(5) var<storage, read_write> lambdaNextOut : array<f32>;
@group(0) @binding(6) var<uniform> params : Params;

fn finiteOrZero(v : f32) -> f32 {
  if (v == v && abs(v) < 1e20) {
    return v;
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (f32(i) >= params.count) {
    return;
  }
  let base = i * ${MEMBRANE_BEND_LAYOUT_STRIDE_FLOATS}u;
  let px = finiteOrZero(layout[base + 0u]);
  let py = finiteOrZero(layout[base + 1u]);
  let nxp = finiteOrZero(layout[base + 2u]);
  let nyp = finiteOrZero(layout[base + 3u]);
  let rest = max(1e-4, finiteOrZero(layout[base + 4u]));
  let wP = max(0.0, finiteOrZero(layout[base + 5u]));
  let wN = max(0.0, finiteOrZero(layout[base + 6u]));
  let lambdaPrev = finiteOrZero(layout[base + 7u]);
  let bendAlpha = max(0.0, finiteOrZero(layout[base + 8u]));
  let cLimit = max(0.0, finiteOrZero(layout[base + 9u]));
  let lambdaLimit = max(1e-6, finiteOrZero(layout[base + 10u]));

  let dx = nxp - px;
  let dy = nyp - py;
  let d = max(1e-6, sqrt(dx * dx + dy * dy));
  let ux = dx / d;
  let uy = dy / d;
  let C = clamp(d - rest, -cLimit, cLimit);
  let wSum = wP + wN;
  if (wSum <= 1e-9) {
    deltaVxPrevOut[i] = 0.0;
    deltaVyPrevOut[i] = 0.0;
    deltaVxNextOut[i] = 0.0;
    deltaVyNextOut[i] = 0.0;
    lambdaNextOut[i] = lambdaPrev;
    return;
  }

  var dl = (-C - bendAlpha * lambdaPrev) / (wSum + bendAlpha);
  if (!(dl == dl)) {
    dl = 0.0;
  }
  let lambdaNext = clamp(lambdaPrev + dl, -lambdaLimit, lambdaLimit);
  dl = lambdaNext - lambdaPrev;

  let invDt = 1.0 / max(params.dtPos, 1e-8);
  deltaVxPrevOut[i] = (-wP * dl * ux) * invDt;
  deltaVyPrevOut[i] = (-wP * dl * uy) * invDt;
  deltaVxNextOut[i] = (wN * dl * ux) * invDt;
  deltaVyNextOut[i] = (wN * dl * uy) * invDt;
  lambdaNextOut[i] = lambdaNext;
}
`;

const softMembraneShapeMemoryProposalWgsl = /* wgsl */ `
struct Params {
  count : f32,
  dtPos : f32,
  _pad0 : f32,
  _pad1 : f32,
};

@group(0) @binding(0) var<storage, read> layout : array<f32>;
@group(0) @binding(1) var<storage, read_write> deltaVxOut : array<f32>;
@group(0) @binding(2) var<storage, read_write> deltaVyOut : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

fn finiteOrZero(v : f32) -> f32 {
  if (v == v && abs(v) < 1e20) {
    return v;
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (f32(i) >= params.count) {
    return;
  }
  let base = i * ${SHAPE_LAYOUT_STRIDE_FLOATS}u;
  let px = finiteOrZero(layout[base + 1u]);
  let py = finiteOrZero(layout[base + 2u]);
  let tx = finiteOrZero(layout[base + 3u]);
  let ty = finiteOrZero(layout[base + 4u]);
  let corrPos = max(0.0, finiteOrZero(layout[base + 5u]));
  let maxShift = max(0.0, finiteOrZero(layout[base + 6u]));

  var ex = tx - px;
  var ey = ty - py;
  let eLen = sqrt(ex * ex + ey * ey);
  if (!(eLen == eLen) || eLen <= 1e-7) {
    deltaVxOut[i] = 0.0;
    deltaVyOut[i] = 0.0;
    return;
  }
  if (maxShift > 0.0 && eLen > maxShift) {
    let k = maxShift / max(eLen, 1e-9);
    ex = ex * k;
    ey = ey * k;
  }

  let invDt = 1.0 / max(params.dtPos, 1e-8);
  deltaVxOut[i] = ex * corrPos * invDt;
  deltaVyOut[i] = ey * corrPos * invDt;
}
`;

function checkFiniteFloat32Array(values) {
  if (!(values instanceof Float32Array)) {
    return { allFinite: false, firstBadIndex: -1, length: 0 };
  }
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      return { allFinite: false, firstBadIndex: i, length: values.length };
    }
  }
  return { allFinite: true, firstBadIndex: -1, length: values.length };
}

function computeShapeMemoryProposalSignature(layout, dtPos) {
  if (!(layout instanceof Float32Array)) return '';
  let sum = 0;
  for (let i = 0; i < layout.length; i += SHAPE_LAYOUT_STRIDE_FLOATS) {
    sum += Math.fround(layout[i + 3] || 0) * 0.37;
    sum += Math.fround(layout[i + 4] || 0) * 0.17;
    sum += Math.fround(layout[i + 5] || 0) * 0.11;
  }
  return `${layout.length}|${Math.fround(dtPos)}|${Math.fround(sum)}`;
}


function ensureSoftMembraneBendPipeline(offload) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device) return null;
  if (!state.membraneBendProposalPipelinePromise) {
    state.membraneBendProposalPipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: softMembraneBendProposalWgsl }),
        entryPoint: 'main',
      },
    }).catch((err) => {
      state.membraneBendProposalPipelinePromise = null;
      throw err;
    });
  }
  return state.membraneBendProposalPipelinePromise;
}

function ensureSoftMembraneShapeMemoryPipeline(offload) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device) return null;
  if (!state.shapeMemoryProposalPipelinePromise) {
    state.shapeMemoryProposalPipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: softMembraneShapeMemoryProposalWgsl }),
        entryPoint: 'main',
      },
    }).catch((err) => {
      state.shapeMemoryProposalPipelinePromise = null;
      throw err;
    });
  }
  return state.shapeMemoryProposalPipelinePromise;
}

function getGpuOnlyPipelineModeProfile(offload) {
  const modeProfile = String(offload?.modeProfile || '').trim().toLowerCase();
  if (modeProfile === 'gpu-only-fast') return 'gpu-only-fast';
  if (modeProfile === 'gpu-only-validated') return 'gpu-only-validated';
  return 'standard';
}

function ensureSoftMembraneBoundaryEdgePipeline(offload) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device) return null;
  if (!state.membraneBoundaryEdgeProposalPipelinePromise) {
    state.membraneBoundaryEdgeProposalPipelinePromise = device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: softMembraneBoundaryEdgeProposalWgsl }),
        entryPoint: 'main',
      },
    }).catch((err) => {
      state.membraneBoundaryEdgeProposalPipelinePromise = null;
      throw err;
    });
  }
  return state.membraneBoundaryEdgeProposalPipelinePromise;
}

async function dispatchSoftMembraneBoundaryEdgeProposal(offload, layout, dtPos, signature) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device || !(layout instanceof Float32Array) || layout.length === 0) return false;
  const count = Math.floor(layout.length / MEMBRANE_BOUNDARY_EDGE_LAYOUT_STRIDE_FLOATS);
  if (count <= 0) return false;

  const pipeline = await ensureSoftMembraneBoundaryEdgePipeline(offload);
  if (!pipeline) return false;

  const layoutBuffer = device.createBuffer({ size: layout.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBytes = count * Float32Array.BYTES_PER_ELEMENT;
  const deltaVxABuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const deltaVyABuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const deltaVxBBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const deltaVyBBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const lambdaNextBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readDeltaVxABuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readDeltaVyABuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readDeltaVxBBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readDeltaVyBBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readLambdaNextBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const params = new Float32Array([count, dtPos, 0, 0]);
  const paramBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(layoutBuffer, 0, layout);
  device.queue.writeBuffer(paramBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: layoutBuffer } },
      { binding: 1, resource: { buffer: deltaVxABuffer } },
      { binding: 2, resource: { buffer: deltaVyABuffer } },
      { binding: 3, resource: { buffer: deltaVxBBuffer } },
      { binding: 4, resource: { buffer: deltaVyBBuffer } },
      { binding: 5, resource: { buffer: lambdaNextBuffer } },
      { binding: 6, resource: { buffer: paramBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / WGSL_WORKGROUP_SIZE)));
  pass.end();
  encoder.copyBufferToBuffer(deltaVxABuffer, 0, readDeltaVxABuffer, 0, outBytes);
  encoder.copyBufferToBuffer(deltaVyABuffer, 0, readDeltaVyABuffer, 0, outBytes);
  encoder.copyBufferToBuffer(deltaVxBBuffer, 0, readDeltaVxBBuffer, 0, outBytes);
  encoder.copyBufferToBuffer(deltaVyBBuffer, 0, readDeltaVyBBuffer, 0, outBytes);
  encoder.copyBufferToBuffer(lambdaNextBuffer, 0, readLambdaNextBuffer, 0, outBytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    readDeltaVxABuffer.mapAsync(GPUMapMode.READ),
    readDeltaVyABuffer.mapAsync(GPUMapMode.READ),
    readDeltaVxBBuffer.mapAsync(GPUMapMode.READ),
    readDeltaVyBBuffer.mapAsync(GPUMapMode.READ),
    readLambdaNextBuffer.mapAsync(GPUMapMode.READ),
  ]);

  const deltaVxA = new Float32Array(readDeltaVxABuffer.getMappedRange().slice(0));
  const deltaVyA = new Float32Array(readDeltaVyABuffer.getMappedRange().slice(0));
  const deltaVxB = new Float32Array(readDeltaVxBBuffer.getMappedRange().slice(0));
  const deltaVyB = new Float32Array(readDeltaVyBBuffer.getMappedRange().slice(0));
  const lambdaNext = new Float32Array(readLambdaNextBuffer.getMappedRange().slice(0));
  readDeltaVxABuffer.unmap();
  readDeltaVyABuffer.unmap();
  readDeltaVxBBuffer.unmap();
  readDeltaVyBBuffer.unmap();
  readLambdaNextBuffer.unmap();

  const finite = {
    allFinite: checkFiniteFloat32Array(deltaVxA).allFinite
      && checkFiniteFloat32Array(deltaVyA).allFinite
      && checkFiniteFloat32Array(deltaVxB).allFinite
      && checkFiniteFloat32Array(deltaVyB).allFinite
      && checkFiniteFloat32Array(lambdaNext).allFinite,
  };

  state.lastMembraneBoundaryEdgeProposalSignature = String(signature || '');
  state.lastMembraneBoundaryEdgeProposalFinite = finite;
  state.lastMembraneBoundaryEdgeProposalSource = finite.allFinite
    ? 'wgsl-membrane-boundary-edge-proposal'
    : 'cpu-membrane-boundary-edge-authoritative-nonfinite';
  state.lastMode = finite.allFinite ? 'wgsl-membrane-boundary-edge-proposal' : 'cpu-membrane-boundary-edge-authoritative';

  if (finite.allFinite) {
    state.lastMembraneBoundaryEdgeProposalDeltaVxA = deltaVxA;
    state.lastMembraneBoundaryEdgeProposalDeltaVyA = deltaVyA;
    state.lastMembraneBoundaryEdgeProposalDeltaVxB = deltaVxB;
    state.lastMembraneBoundaryEdgeProposalDeltaVyB = deltaVyB;
    state.lastMembraneBoundaryEdgeProposalLambdaNext = lambdaNext;
  } else {
    state.lastMembraneBoundaryEdgeProposalDeltaVxA = null;
    state.lastMembraneBoundaryEdgeProposalDeltaVyA = null;
    state.lastMembraneBoundaryEdgeProposalDeltaVxB = null;
    state.lastMembraneBoundaryEdgeProposalDeltaVyB = null;
    state.lastMembraneBoundaryEdgeProposalLambdaNext = null;
  }

  layoutBuffer.destroy();
  deltaVxABuffer.destroy();
  deltaVyABuffer.destroy();
  deltaVxBBuffer.destroy();
  deltaVyBBuffer.destroy();
  lambdaNextBuffer.destroy();
  readDeltaVxABuffer.destroy();
  readDeltaVyABuffer.destroy();
  readDeltaVxBBuffer.destroy();
  readDeltaVyBBuffer.destroy();
  readLambdaNextBuffer.destroy();
  paramBuffer.destroy();
  return finite.allFinite;
}


async function dispatchSoftMembraneBendProposal(offload, layout, dtPos, signature) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device || !(layout instanceof Float32Array) || layout.length === 0) return false;
  const count = Math.floor(layout.length / MEMBRANE_BEND_LAYOUT_STRIDE_FLOATS);
  if (count <= 0) return false;

  const pipeline = await ensureSoftMembraneBendPipeline(offload);
  if (!pipeline) return false;

  const layoutBuffer = device.createBuffer({ size: layout.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBytes = count * Float32Array.BYTES_PER_ELEMENT;
  const deltaVxPrevBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const deltaVyPrevBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const deltaVxNextBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const deltaVyNextBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const lambdaNextBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readDeltaVxPrevBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readDeltaVyPrevBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readDeltaVxNextBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readDeltaVyNextBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readLambdaNextBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const params = new Float32Array([count, dtPos, 0, 0]);
  const paramBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(layoutBuffer, 0, layout);
  device.queue.writeBuffer(paramBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: layoutBuffer } },
      { binding: 1, resource: { buffer: deltaVxPrevBuffer } },
      { binding: 2, resource: { buffer: deltaVyPrevBuffer } },
      { binding: 3, resource: { buffer: deltaVxNextBuffer } },
      { binding: 4, resource: { buffer: deltaVyNextBuffer } },
      { binding: 5, resource: { buffer: lambdaNextBuffer } },
      { binding: 6, resource: { buffer: paramBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / WGSL_WORKGROUP_SIZE)));
  pass.end();
  encoder.copyBufferToBuffer(deltaVxPrevBuffer, 0, readDeltaVxPrevBuffer, 0, outBytes);
  encoder.copyBufferToBuffer(deltaVyPrevBuffer, 0, readDeltaVyPrevBuffer, 0, outBytes);
  encoder.copyBufferToBuffer(deltaVxNextBuffer, 0, readDeltaVxNextBuffer, 0, outBytes);
  encoder.copyBufferToBuffer(deltaVyNextBuffer, 0, readDeltaVyNextBuffer, 0, outBytes);
  encoder.copyBufferToBuffer(lambdaNextBuffer, 0, readLambdaNextBuffer, 0, outBytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    readDeltaVxPrevBuffer.mapAsync(GPUMapMode.READ),
    readDeltaVyPrevBuffer.mapAsync(GPUMapMode.READ),
    readDeltaVxNextBuffer.mapAsync(GPUMapMode.READ),
    readDeltaVyNextBuffer.mapAsync(GPUMapMode.READ),
    readLambdaNextBuffer.mapAsync(GPUMapMode.READ),
  ]);

  const deltaVxPrev = new Float32Array(readDeltaVxPrevBuffer.getMappedRange().slice(0));
  const deltaVyPrev = new Float32Array(readDeltaVyPrevBuffer.getMappedRange().slice(0));
  const deltaVxNext = new Float32Array(readDeltaVxNextBuffer.getMappedRange().slice(0));
  const deltaVyNext = new Float32Array(readDeltaVyNextBuffer.getMappedRange().slice(0));
  const lambdaNext = new Float32Array(readLambdaNextBuffer.getMappedRange().slice(0));
  readDeltaVxPrevBuffer.unmap();
  readDeltaVyPrevBuffer.unmap();
  readDeltaVxNextBuffer.unmap();
  readDeltaVyNextBuffer.unmap();
  readLambdaNextBuffer.unmap();

  const finite = {
    allFinite: checkFiniteFloat32Array(deltaVxPrev).allFinite
      && checkFiniteFloat32Array(deltaVyPrev).allFinite
      && checkFiniteFloat32Array(deltaVxNext).allFinite
      && checkFiniteFloat32Array(deltaVyNext).allFinite
      && checkFiniteFloat32Array(lambdaNext).allFinite,
  };

  state.lastMembraneBendProposalSignature = String(signature || '');
  state.lastMembraneBendProposalFinite = finite;
  state.lastMembraneBendProposalSource = finite.allFinite
    ? 'wgsl-membrane-bend-proposal'
    : 'cpu-membrane-bend-authoritative-nonfinite';
  state.lastMode = finite.allFinite ? 'wgsl-membrane-bend-proposal' : 'cpu-membrane-bend-authoritative';

  if (finite.allFinite) {
    state.lastMembraneBendProposalDeltaVxPrev = deltaVxPrev;
    state.lastMembraneBendProposalDeltaVyPrev = deltaVyPrev;
    state.lastMembraneBendProposalDeltaVxNext = deltaVxNext;
    state.lastMembraneBendProposalDeltaVyNext = deltaVyNext;
    state.lastMembraneBendProposalLambdaNext = lambdaNext;
  } else {
    state.lastMembraneBendProposalDeltaVxPrev = null;
    state.lastMembraneBendProposalDeltaVyPrev = null;
    state.lastMembraneBendProposalDeltaVxNext = null;
    state.lastMembraneBendProposalDeltaVyNext = null;
    state.lastMembraneBendProposalLambdaNext = null;
  }

  layoutBuffer.destroy();
  deltaVxPrevBuffer.destroy();
  deltaVyPrevBuffer.destroy();
  deltaVxNextBuffer.destroy();
  deltaVyNextBuffer.destroy();
  lambdaNextBuffer.destroy();
  readDeltaVxPrevBuffer.destroy();
  readDeltaVyPrevBuffer.destroy();
  readDeltaVxNextBuffer.destroy();
  readDeltaVyNextBuffer.destroy();
  readLambdaNextBuffer.destroy();
  paramBuffer.destroy();
  return finite.allFinite;
}

async function dispatchSoftMembraneShapeMemoryProposal(offload, layout, dtPos, signature) {
  const state = offload?.state;
  const device = offload?.device;
  if (!state || !device || !(layout instanceof Float32Array) || layout.length === 0) return false;
  const count = Math.floor(layout.length / SHAPE_LAYOUT_STRIDE_FLOATS);
  if (count <= 0) return false;

  const pipeline = await ensureSoftMembraneShapeMemoryPipeline(offload);
  if (!pipeline) return false;

  const layoutBuffer = device.createBuffer({ size: layout.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBytes = count * Float32Array.BYTES_PER_ELEMENT;
  const deltaVxBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const deltaVyBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readVxBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readVyBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const params = new Float32Array([count, dtPos, 0, 0]);
  const paramBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(layoutBuffer, 0, layout);
  device.queue.writeBuffer(paramBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: layoutBuffer } },
      { binding: 1, resource: { buffer: deltaVxBuffer } },
      { binding: 2, resource: { buffer: deltaVyBuffer } },
      { binding: 3, resource: { buffer: paramBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / WGSL_WORKGROUP_SIZE)));
  pass.end();
  encoder.copyBufferToBuffer(deltaVxBuffer, 0, readVxBuffer, 0, outBytes);
  encoder.copyBufferToBuffer(deltaVyBuffer, 0, readVyBuffer, 0, outBytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([readVxBuffer.mapAsync(GPUMapMode.READ), readVyBuffer.mapAsync(GPUMapMode.READ)]);
  const deltaVx = new Float32Array(readVxBuffer.getMappedRange().slice(0));
  const deltaVy = new Float32Array(readVyBuffer.getMappedRange().slice(0));
  readVxBuffer.unmap();
  readVyBuffer.unmap();

  const finiteVx = checkFiniteFloat32Array(deltaVx);
  const finiteVy = checkFiniteFloat32Array(deltaVy);
  const finite = { allFinite: finiteVx.allFinite && finiteVy.allFinite, vx: finiteVx, vy: finiteVy };
  state.lastShapeMemoryProposalFinite = finite;
  state.lastShapeMemoryProposalSignature = String(signature || '');
  state.lastShapeMemoryProposalDeltaVx = finite.allFinite ? deltaVx : null;
  state.lastShapeMemoryProposalDeltaVy = finite.allFinite ? deltaVy : null;
  state.lastShapeMemoryProposalSource = finite.allFinite ? 'wgsl-shape-memory-proposal' : 'cpu-shape-memory-authoritative-nonfinite';
  state.lastMode = finite.allFinite ? 'wgsl-shape-memory-proposal' : 'cpu-shape-memory-authoritative';
  state.lastShapeMemoryProposalDispatchCount = Math.max(1, Math.ceil(count / WGSL_WORKGROUP_SIZE));
  state.lastShapeMemoryProposalCount = count;

  layoutBuffer.destroy();
  deltaVxBuffer.destroy();
  deltaVyBuffer.destroy();
  readVxBuffer.destroy();
  readVyBuffer.destroy();
  paramBuffer.destroy();
  return finite.allFinite;
}

function ensureSoftMembraneLoopStateGpuOnly(sim, soft, loops, membraneSet) {
  sim.softMembraneLoopState = sim.softMembraneLoopState || new Map();
  const live = new Set();

  for (const loop of loops || []) {
    const cid = loop.clusterId ?? 0;
    if (!membraneSet.has(cid)) continue;
    const ids = loop.indices || [];
    if (ids.length < 3) continue;
    live.add(cid);

    const key = ids.join(',');
    let st = sim.softMembraneLoopState.get(cid);
    const needsReset = !st || st.key !== key || st.edgeRest.length !== ids.length;
    if (!needsReset) continue;

    st = {
      key,
      edgeRest: new Float32Array(ids.length),
      edgeLambda: new Float32Array(ids.length),
      bendRest: new Float32Array(ids.length),
      bendLambda: new Float32Array(ids.length),
      shapeRef: new Float32Array(ids.length * 2),
      shapeRefRms: 1,
    };

    let cx = 0;
    let cy = 0;
    let count = 0;
    for (let i = 0; i < ids.length; i++) {
      const node = soft.nodes[ids[i]];
      if (!node) continue;
      cx += node.x || 0;
      cy += node.y || 0;
      count += 1;
    }
    if (count > 0) {
      cx /= count;
      cy /= count;
    }

    let refR2 = 0;
    for (let i = 0; i < ids.length; i++) {
      const a = soft.nodes[ids[i]];
      const b = soft.nodes[ids[(i + 1) % ids.length]];
      if (a && b) {
        st.edgeRest[i] = Math.max(1e-4, Math.hypot((b.x || 0) - (a.x || 0), (b.y || 0) - (a.y || 0)));
      }

      const p = soft.nodes[ids[(i - 1 + ids.length) % ids.length]];
      const n = soft.nodes[ids[(i + 1) % ids.length]];
      if (p && n) {
        st.bendRest[i] = Math.max(1e-4, Math.hypot((n.x || 0) - (p.x || 0), (n.y || 0) - (p.y || 0)));
      }

      const node = soft.nodes[ids[i]];
      const rx = (node?.x || 0) - cx;
      const ry = (node?.y || 0) - cy;
      st.shapeRef[i * 2] = rx;
      st.shapeRef[i * 2 + 1] = ry;
      refR2 += rx * rx + ry * ry;
    }
    st.shapeRefRms = Math.max(1e-3, Math.sqrt(refR2 / Math.max(1, ids.length)));
    sim.softMembraneLoopState.set(cid, st);
  }

  for (const cid of [...sim.softMembraneLoopState.keys()]) {
    if (!live.has(cid)) sim.softMembraneLoopState.delete(cid);
  }
}

function canApplyAuthoritativeMembraneBoundaryEdgeProposal({ wgslOffload, signature, expectedCount }) {
  const state = wgslOffload?.state;
  if (!state || state.enableAuthoritativeMembraneBoundaryEdge !== true) return false;
  if (state.lastMembraneBoundaryEdgeProposalSource !== 'wgsl-membrane-boundary-edge-proposal') return false;
  if (String(state.lastMembraneBoundaryEdgeProposalSignature || '') !== String(signature || '')) return false;
  if (!(state.lastMembraneBoundaryEdgeProposalDeltaVxA instanceof Float32Array)) return false;
  if (!(state.lastMembraneBoundaryEdgeProposalDeltaVyA instanceof Float32Array)) return false;
  if (!(state.lastMembraneBoundaryEdgeProposalDeltaVxB instanceof Float32Array)) return false;
  if (!(state.lastMembraneBoundaryEdgeProposalDeltaVyB instanceof Float32Array)) return false;
  if (!(state.lastMembraneBoundaryEdgeProposalLambdaNext instanceof Float32Array)) return false;
  if (state.lastMembraneBoundaryEdgeProposalDeltaVxA.length !== expectedCount) return false;
  if (state.lastMembraneBoundaryEdgeProposalDeltaVyA.length !== expectedCount) return false;
  if (state.lastMembraneBoundaryEdgeProposalDeltaVxB.length !== expectedCount) return false;
  if (state.lastMembraneBoundaryEdgeProposalDeltaVyB.length !== expectedCount) return false;
  if (state.lastMembraneBoundaryEdgeProposalLambdaNext.length !== expectedCount) return false;
  if (state.lastMembraneBoundaryEdgeProposalFinite?.allFinite !== true) return false;
  return true;
}

function applyMembraneBoundaryVelocityDeltasAuthoritative({
  sim,
  soft,
  nodeAIndices,
  nodeBIndices,
  loopClusterIds,
  loopEdgeIndices,
  cpuDeltaVxA,
  cpuDeltaVyA,
  cpuDeltaVxB,
  cpuDeltaVyB,
  wgslDeltaVxA,
  wgslDeltaVyA,
  wgslDeltaVxB,
  wgslDeltaVyB,
  wgslLambdaNext,
}) {
  const count = Math.min(
    nodeAIndices.length,
    nodeBIndices.length,
    loopClusterIds.length,
    loopEdgeIndices.length,
    cpuDeltaVxA.length,
    cpuDeltaVyA.length,
    cpuDeltaVxB.length,
    cpuDeltaVyB.length,
    wgslDeltaVxA.length,
    wgslDeltaVyA.length,
    wgslDeltaVxB.length,
    wgslDeltaVyB.length,
    wgslLambdaNext.length,
  );

  for (let i = 0; i < count; i++) {
    const a = soft.nodes[nodeAIndices[i] | 0];
    const b = soft.nodes[nodeBIndices[i] | 0];
    if (!a || !b) continue;

    const cpuAx = Number(cpuDeltaVxA[i]);
    const cpuAy = Number(cpuDeltaVyA[i]);
    const cpuBx = Number(cpuDeltaVxB[i]);
    const cpuBy = Number(cpuDeltaVyB[i]);
    const wgslAx = Number(wgslDeltaVxA[i]);
    const wgslAy = Number(wgslDeltaVyA[i]);
    const wgslBx = Number(wgslDeltaVxB[i]);
    const wgslBy = Number(wgslDeltaVyB[i]);

    if (Number.isFinite(cpuAx) && Number.isFinite(wgslAx)) a.vx += -cpuAx + wgslAx;
    if (Number.isFinite(cpuAy) && Number.isFinite(wgslAy)) a.vy += -cpuAy + wgslAy;
    if (Number.isFinite(cpuBx) && Number.isFinite(wgslBx)) b.vx += -cpuBx + wgslBx;
    if (Number.isFinite(cpuBy) && Number.isFinite(wgslBy)) b.vy += -cpuBy + wgslBy;

    const clusterId = loopClusterIds[i] | 0;
    const edgeIndex = loopEdgeIndices[i] | 0;
    const st = sim.softMembraneLoopState?.get(clusterId);
    if (!st || !(st.edgeLambda instanceof Float32Array) || edgeIndex < 0 || edgeIndex >= st.edgeLambda.length) continue;

    const lambdaNext = Number(wgslLambdaNext[i]);
    if (Number.isFinite(lambdaNext)) {
      st.edgeLambda[edgeIndex] = lambdaNext;
    }
  }
}


function canApplyAuthoritativeMembraneBendProposal({ wgslOffload, signature, expectedCount }) {
  const state = wgslOffload?.state;
  if (!state || state.enableAuthoritativeMembraneBend !== true) return false;
  if (state.lastMembraneBendProposalSource !== 'wgsl-membrane-bend-proposal') return false;
  if (String(state.lastMembraneBendProposalSignature || '') !== String(signature || '')) return false;
  if (!(state.lastMembraneBendProposalDeltaVxPrev instanceof Float32Array)) return false;
  if (!(state.lastMembraneBendProposalDeltaVyPrev instanceof Float32Array)) return false;
  if (!(state.lastMembraneBendProposalDeltaVxNext instanceof Float32Array)) return false;
  if (!(state.lastMembraneBendProposalDeltaVyNext instanceof Float32Array)) return false;
  if (!(state.lastMembraneBendProposalLambdaNext instanceof Float32Array)) return false;
  if (state.lastMembraneBendProposalDeltaVxPrev.length !== expectedCount) return false;
  if (state.lastMembraneBendProposalDeltaVyPrev.length !== expectedCount) return false;
  if (state.lastMembraneBendProposalDeltaVxNext.length !== expectedCount) return false;
  if (state.lastMembraneBendProposalDeltaVyNext.length !== expectedCount) return false;
  if (state.lastMembraneBendProposalLambdaNext.length !== expectedCount) return false;
  if (state.lastMembraneBendProposalFinite?.allFinite !== true) return false;
  return true;
}

export function applySoftMembraneBoundaryXPBDVelocityGpuOnly({
  sim,
  soft,
  loops,
  dtPos,
  membraneClusterSet,
  clamp,
  membraneEdgeXpbdIters = 8,
  membraneEdgeBaseCompliance = 0.0007,
  membraneBendXpbdIters = 4,
  membraneBendBaseCompliance = 0.0022,
  wgslOffload,
}) {
  if (!(membraneClusterSet instanceof Set) || membraneClusterSet.size === 0) return 0;

  ensureSoftMembraneLoopStateGpuOnly(sim, soft, loops, membraneClusterSet);

  const edgeAlpha = membraneEdgeBaseCompliance / Math.max(1e-8, dtPos * dtPos);
  const bendAlpha = membraneBendBaseCompliance / Math.max(1e-8, dtPos * dtPos);
  const runWgslBoundaryProbe = wgslOffload?.enabled === true
    && wgslOffload?.state
    && wgslOffload?.device
    && getGpuOnlyPipelineModeProfile(wgslOffload) !== 'standard';
  const edgeProposalLayout = [];
  const edgeProposalNodeAIndices = [];
  const edgeProposalNodeBIndices = [];
  const edgeProposalLoopClusterIds = [];
  const edgeProposalLoopEdgeIndices = [];
  const edgeProposalCpuDeltaVxA = [];
  const edgeProposalCpuDeltaVyA = [];
  const edgeProposalCpuDeltaVxB = [];
  const edgeProposalCpuDeltaVyB = [];
  let edgeProposalSampleCount = 0;
  let edgeProposalSignatureAccumulator = 0;
  const bendProposalLayout = [];
  const bendProposalNodePrevIndices = [];
  const bendProposalNodeNextIndices = [];
  const bendProposalLoopClusterIds = [];
  const bendProposalLoopEdgeIndices = [];
  const bendProposalCpuDeltaVxPrev = [];
  const bendProposalCpuDeltaVyPrev = [];
  const bendProposalCpuDeltaVxNext = [];
  const bendProposalCpuDeltaVyNext = [];
  let bendProposalSampleCount = 0;
  let bendProposalSignatureAccumulator = 0;
  let touched = 0;

  for (let iter = 0; iter < membraneEdgeXpbdIters; iter++) {
    for (const loop of loops || []) {
      const cid = loop.clusterId ?? 0;
      if (!membraneClusterSet.has(cid)) continue;
      const ids = loop.indices || [];
      if (ids.length < 3) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st) continue;

      for (let i = 0; i < ids.length; i++) {
        const a = soft.nodes[ids[i]];
        const b = soft.nodes[ids[(i + 1) % ids.length]];
        if (!a || !b) continue;

        const ax = a.x + a.vx * dtPos;
        const ay = a.y + a.vy * dtPos;
        const bx = b.x + b.vx * dtPos;
        const by = b.y + b.vy * dtPos;
        const dx = bx - ax;
        const dy = by - ay;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const nx = dx / d;
        const ny = dy / d;

        const rest = Math.max(1e-4, Number(st.edgeRest[i]) || d);
        const C = clamp(d - rest, -Math.max(0.08, rest * 0.28), Math.max(0.08, rest * 0.28));
        const wA = 1 / Math.max(0.02, a.mass || 1);
        const wB = 1 / Math.max(0.02, b.mass || 1);
        const wSum = wA + wB;
        if (wSum <= 1e-9) continue;

        const lambdaPrev = Number(st.edgeLambda[i]) || 0;
        let dl = (-C - edgeAlpha * lambdaPrev) / (wSum + edgeAlpha);
        if (!Number.isFinite(dl)) continue;
        const lambdaNext = clamp(lambdaPrev + dl, -20, 20);
        dl = lambdaNext - lambdaPrev;
        st.edgeLambda[i] = lambdaNext;

        const cpuDeltaVxA = (-wA * dl * nx) / dtPos;
        const cpuDeltaVyA = (-wA * dl * ny) / dtPos;
        const cpuDeltaVxB = (wB * dl * nx) / dtPos;
        const cpuDeltaVyB = (wB * dl * ny) / dtPos;

        if (runWgslBoundaryProbe && iter === 0) {
          const cLimit = Math.max(0.08, rest * 0.28);
          edgeProposalLayout.push(
            ax,
            ay,
            bx,
            by,
            rest,
            wA,
            wB,
            lambdaPrev,
            edgeAlpha,
            cLimit,
            20,
          );
          edgeProposalNodeAIndices.push(ids[i] | 0);
          edgeProposalNodeBIndices.push(ids[(i + 1) % ids.length] | 0);
          edgeProposalLoopClusterIds.push(cid | 0);
          edgeProposalLoopEdgeIndices.push(i | 0);
          edgeProposalCpuDeltaVxA.push(cpuDeltaVxA);
          edgeProposalCpuDeltaVyA.push(cpuDeltaVyA);
          edgeProposalCpuDeltaVxB.push(cpuDeltaVxB);
          edgeProposalCpuDeltaVyB.push(cpuDeltaVyB);
          edgeProposalSampleCount += 1;
          edgeProposalSignatureAccumulator += Math.fround(rest) * 0.41 + Math.fround(lambdaPrev) * 0.19 + Math.fround(wA + wB) * 0.07;
        }

        a.vx += cpuDeltaVxA;
        a.vy += cpuDeltaVyA;
        b.vx += cpuDeltaVxB;
        b.vy += cpuDeltaVyB;
        touched += 1;
      }
    }
  }

  const edgeLayout = edgeProposalLayout.length > 0 ? Float32Array.from(edgeProposalLayout) : null;
  const edgeSignature = edgeLayout
    ? `${edgeProposalSampleCount}|${Math.fround(dtPos)}|${Math.fround(edgeProposalSignatureAccumulator)}`
    : '';

  const authoritativeBoundaryFromWgsl = canApplyAuthoritativeMembraneBoundaryEdgeProposal({
    wgslOffload,
    signature: edgeSignature,
    expectedCount: edgeProposalNodeAIndices.length,
  });

  if (authoritativeBoundaryFromWgsl) {
    applyMembraneBoundaryVelocityDeltasAuthoritative({
      sim,
      soft,
      nodeAIndices: edgeProposalNodeAIndices,
      nodeBIndices: edgeProposalNodeBIndices,
      loopClusterIds: edgeProposalLoopClusterIds,
      loopEdgeIndices: edgeProposalLoopEdgeIndices,
      cpuDeltaVxA: edgeProposalCpuDeltaVxA,
      cpuDeltaVyA: edgeProposalCpuDeltaVyA,
      cpuDeltaVxB: edgeProposalCpuDeltaVxB,
      cpuDeltaVyB: edgeProposalCpuDeltaVyB,
      wgslDeltaVxA: wgslOffload.state.lastMembraneBoundaryEdgeProposalDeltaVxA,
      wgslDeltaVyA: wgslOffload.state.lastMembraneBoundaryEdgeProposalDeltaVyA,
      wgslDeltaVxB: wgslOffload.state.lastMembraneBoundaryEdgeProposalDeltaVxB,
      wgslDeltaVyB: wgslOffload.state.lastMembraneBoundaryEdgeProposalDeltaVyB,
      wgslLambdaNext: wgslOffload.state.lastMembraneBoundaryEdgeProposalLambdaNext,
    });
  }

  if (wgslOffload?.state) {
    wgslOffload.state.lastMembraneBoundaryEdgeAuthoritativeSource = authoritativeBoundaryFromWgsl
      ? 'wgsl-membrane-boundary-edge-authoritative'
      : 'cpu-membrane-boundary-edge-authoritative';
    wgslOffload.state.lastMembraneBoundaryEdgeAuthoritativeSignature = edgeSignature;
  }

  for (let iter = 0; iter < membraneBendXpbdIters; iter++) {
    for (const loop of loops || []) {
      const cid = loop.clusterId ?? 0;
      if (!membraneClusterSet.has(cid)) continue;
      const ids = loop.indices || [];
      if (ids.length < 4) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st) continue;

      for (let i = 0; i < ids.length; i++) {
        const prev = soft.nodes[ids[(i - 1 + ids.length) % ids.length]];
        const next = soft.nodes[ids[(i + 1) % ids.length]];
        if (!prev || !next) continue;

        const px = prev.x + prev.vx * dtPos;
        const py = prev.y + prev.vy * dtPos;
        const nxp = next.x + next.vx * dtPos;
        const nyp = next.y + next.vy * dtPos;
        const dx = nxp - px;
        const dy = nyp - py;
        const d = Math.max(1e-6, Math.hypot(dx, dy));
        const ux = dx / d;
        const uy = dy / d;

        const rest = Math.max(1e-4, Number(st.bendRest[i]) || d);
        const C = clamp(d - rest, -Math.max(0.1, rest * 0.35), Math.max(0.1, rest * 0.35));
        const wP = 1 / Math.max(0.02, prev.mass || 1);
        const wN = 1 / Math.max(0.02, next.mass || 1);
        const wSum = wP + wN;
        if (wSum <= 1e-9) continue;

        const lambdaPrev = Number(st.bendLambda[i]) || 0;
        let dl = (-C - bendAlpha * lambdaPrev) / (wSum + bendAlpha);
        if (!Number.isFinite(dl)) continue;
        const lambdaNext = clamp(lambdaPrev + dl, -20, 20);
        dl = lambdaNext - lambdaPrev;
        st.bendLambda[i] = lambdaNext;

        const cpuDeltaVxPrev = (-wP * dl * ux) / dtPos;
        const cpuDeltaVyPrev = (-wP * dl * uy) / dtPos;
        const cpuDeltaVxNext = (wN * dl * ux) / dtPos;
        const cpuDeltaVyNext = (wN * dl * uy) / dtPos;

        if (runWgslBoundaryProbe && iter === 0) {
          const cLimit = Math.max(0.1, rest * 0.35);
          bendProposalLayout.push(
            px,
            py,
            nxp,
            nyp,
            rest,
            wP,
            wN,
            lambdaPrev,
            bendAlpha,
            cLimit,
            20,
          );
          bendProposalNodePrevIndices.push(ids[(i - 1 + ids.length) % ids.length] | 0);
          bendProposalNodeNextIndices.push(ids[(i + 1) % ids.length] | 0);
          bendProposalLoopClusterIds.push(cid | 0);
          bendProposalLoopEdgeIndices.push(i | 0);
          bendProposalCpuDeltaVxPrev.push(cpuDeltaVxPrev);
          bendProposalCpuDeltaVyPrev.push(cpuDeltaVyPrev);
          bendProposalCpuDeltaVxNext.push(cpuDeltaVxNext);
          bendProposalCpuDeltaVyNext.push(cpuDeltaVyNext);
          bendProposalSampleCount += 1;
          bendProposalSignatureAccumulator += Math.fround(rest) * 0.43 + Math.fround(lambdaPrev) * 0.23 + Math.fround(wP + wN) * 0.09;
        }

        prev.vx += cpuDeltaVxPrev;
        prev.vy += cpuDeltaVyPrev;
        next.vx += cpuDeltaVxNext;
        next.vy += cpuDeltaVyNext;
      }
    }
  }


  const bendLayout = bendProposalLayout.length > 0 ? Float32Array.from(bendProposalLayout) : null;
  const bendSignature = bendLayout
    ? `${bendProposalSampleCount}|${Math.fround(dtPos)}|${Math.fround(bendProposalSignatureAccumulator)}`
    : '';

  const authoritativeBendFromWgsl = canApplyAuthoritativeMembraneBendProposal({
    wgslOffload,
    signature: bendSignature,
    expectedCount: bendProposalNodePrevIndices.length,
  });

  if (authoritativeBendFromWgsl) {
    applyMembraneBoundaryVelocityDeltasAuthoritative({
      sim,
      soft,
      nodeAIndices: bendProposalNodePrevIndices,
      nodeBIndices: bendProposalNodeNextIndices,
      loopClusterIds: bendProposalLoopClusterIds,
      loopEdgeIndices: bendProposalLoopEdgeIndices,
      cpuDeltaVxA: bendProposalCpuDeltaVxPrev,
      cpuDeltaVyA: bendProposalCpuDeltaVyPrev,
      cpuDeltaVxB: bendProposalCpuDeltaVxNext,
      cpuDeltaVyB: bendProposalCpuDeltaVyNext,
      wgslDeltaVxA: wgslOffload.state.lastMembraneBendProposalDeltaVxPrev,
      wgslDeltaVyA: wgslOffload.state.lastMembraneBendProposalDeltaVyPrev,
      wgslDeltaVxB: wgslOffload.state.lastMembraneBendProposalDeltaVxNext,
      wgslDeltaVyB: wgslOffload.state.lastMembraneBendProposalDeltaVyNext,
      wgslLambdaNext: wgslOffload.state.lastMembraneBendProposalLambdaNext,
    });
  }

  if (wgslOffload?.state) {
    wgslOffload.state.lastMembraneBendAuthoritativeSource = authoritativeBendFromWgsl
      ? 'wgsl-membrane-bend-authoritative'
      : 'cpu-membrane-bend-authoritative';
    wgslOffload.state.lastMembraneBendAuthoritativeSignature = bendSignature;
  }

  if (runWgslBoundaryProbe) {
    wgslOffload.state.lastMembraneBoundaryEdgeProposalSignaturePrepared = edgeSignature;
    wgslOffload.state.lastMembraneBoundaryEdgeProposalLayoutBytes = edgeLayout?.byteLength || 0;
    if (!wgslOffload.state.lastMembraneBoundaryEdgeProposalSource) {
      wgslOffload.state.lastMembraneBoundaryEdgeProposalSource = 'cpu-membrane-boundary-edge-authoritative';
    }
    if (!wgslOffload.state.lastMode) {
      wgslOffload.state.lastMode = 'cpu-membrane-boundary-edge-authoritative';
    }

    if (edgeLayout && edgeLayout.length > 0) {
      const serializedDispatch = (wgslOffload.state.pendingWgslMembraneBoundaryEdgeProposalPromise || Promise.resolve())
        .catch(() => {})
        .then(() => dispatchSoftMembraneBoundaryEdgeProposal(wgslOffload, edgeLayout, dtPos, edgeSignature))
        .catch((err) => {
          wgslOffload.state.lastMembraneBoundaryEdgeProposalError = String(err?.message || err || 'unknown-error');
          wgslOffload.state.lastMembraneBoundaryEdgeProposalSource = 'cpu-membrane-boundary-edge-authoritative';
          wgslOffload.state.lastMode = 'cpu-membrane-boundary-edge-authoritative';
        });
      wgslOffload.state.pendingWgslMembraneBoundaryEdgeProposalPromise = serializedDispatch;
    }

    wgslOffload.state.lastMembraneBendProposalSignaturePrepared = bendSignature;
    wgslOffload.state.lastMembraneBendProposalLayoutBytes = bendLayout?.byteLength || 0;
    if (!wgslOffload.state.lastMembraneBendProposalSource) {
      wgslOffload.state.lastMembraneBendProposalSource = 'cpu-membrane-bend-authoritative';
    }

    if (bendLayout && bendLayout.length > 0) {
      const serializedBendDispatch = (wgslOffload.state.pendingWgslMembraneBendProposalPromise || Promise.resolve())
        .catch(() => {})
        .then(() => dispatchSoftMembraneBendProposal(wgslOffload, bendLayout, dtPos, bendSignature))
        .catch((err) => {
          wgslOffload.state.lastMembraneBendProposalError = String(err?.message || err || 'unknown-error');
          wgslOffload.state.lastMembraneBendProposalSource = 'cpu-membrane-bend-authoritative';
          wgslOffload.state.lastMode = 'cpu-membrane-bend-authoritative';
        });
      wgslOffload.state.pendingWgslMembraneBendProposalPromise = serializedBendDispatch;
    }
  }

  return touched;
}

function canApplyAuthoritativeShapeMemoryProposal({ wgslOffload, signature, expectedCount }) {
  const state = wgslOffload?.state;
  if (!state || state.enableAuthoritativeShapeMemory !== true) return false;
  if (state.lastShapeMemoryProposalSource !== 'wgsl-shape-memory-proposal') return false;
  if (String(state.lastShapeMemoryProposalSignature || '') !== String(signature || '')) return false;
  if (!(state.lastShapeMemoryProposalDeltaVx instanceof Float32Array)) return false;
  if (!(state.lastShapeMemoryProposalDeltaVy instanceof Float32Array)) return false;
  if (state.lastShapeMemoryProposalDeltaVx.length !== expectedCount) return false;
  if (state.lastShapeMemoryProposalDeltaVy.length !== expectedCount) return false;
  if (state.lastShapeMemoryProposalFinite?.allFinite !== true) return false;
  return true;
}

function applyShapeMemoryVelocityDeltasAuthoritative({ soft, nodeIndices, deltaVx, deltaVy, scale = 1 }) {
  const count = Math.min(nodeIndices.length, deltaVx.length, deltaVy.length);
  for (let i = 0; i < count; i++) {
    const nodeIndex = nodeIndices[i] | 0;
    const node = soft.nodes[nodeIndex];
    if (!node) continue;
    const dvx = Number(deltaVx[i]);
    const dvy = Number(deltaVy[i]);
    if (!Number.isFinite(dvx) || !Number.isFinite(dvy)) continue;
    node.vx += dvx * scale;
    node.vy += dvy * scale;
  }
}

export function applySoftMembraneShapeMemoryVelocityGpuOnly({
  sim,
  soft,
  loops,
  dtPos,
  membraneClusterMap,
  clamp,
  membraneShapeMemoryIters = 2,
  membraneShapeMemoryGain = 0.045,
  membraneShapeMemoryMaxShiftFrac = 0.08,
  wgslOffload,
}) {
  if (!(membraneClusterMap instanceof Map) || membraneClusterMap.size === 0) return 0;

  const shapeMemoryLayout = [];
  const nodeIndices = [];
  const cpuDeltaVx = [];
  const cpuDeltaVy = [];
  let touched = 0;
  for (let iter = 0; iter < membraneShapeMemoryIters; iter++) {
    for (const loop of loops || []) {
      const cid = Number(loop?.clusterId);
      if (!Number.isInteger(cid)) continue;
      const membrane = membraneClusterMap.get(cid);
      if (!membrane) continue;

      const shapeMemoryGainRaw = Number(membrane?.shapeMemoryGain);
      const shapeMemoryGainApplied = Number.isFinite(shapeMemoryGainRaw)
        ? clamp(shapeMemoryGainRaw, 0, 0.35)
        : membraneShapeMemoryGain;
      if (shapeMemoryGainApplied <= 1e-6) continue;

      const ids = loop.indices || [];
      if (ids.length < 3) continue;
      const st = sim.softMembraneLoopState?.get(cid);
      if (!st || !(st.shapeRef instanceof Float32Array) || st.shapeRef.length !== ids.length * 2) continue;

      let cx = 0;
      let cy = 0;
      let count = 0;
      for (const idx of ids) {
        const node = soft.nodes[idx];
        if (!node) continue;
        cx += (node.x || 0) + (node.vx || 0) * dtPos;
        cy += (node.y || 0) + (node.vy || 0) * dtPos;
        count += 1;
      }
      if (count < 3) continue;
      cx /= count;
      cy /= count;

      let dot = 0;
      let cross = 0;
      for (let i = 0; i < ids.length; i++) {
        const node = soft.nodes[ids[i]];
        if (!node) continue;
        const px = (node.x || 0) + (node.vx || 0) * dtPos - cx;
        const py = (node.y || 0) + (node.vy || 0) * dtPos - cy;
        const rx = st.shapeRef[i * 2] || 0;
        const ry = st.shapeRef[i * 2 + 1] || 0;
        dot += rx * px + ry * py;
        cross += rx * py - ry * px;
      }

      const theta = (Math.abs(dot) + Math.abs(cross)) > 1e-9 ? Math.atan2(cross, dot) : 0;
      const c = Math.cos(theta);
      const sn = Math.sin(theta);
      const maxShift = Math.max(0.02, (st.shapeRefRms || 1) * membraneShapeMemoryMaxShiftFrac);

      for (let i = 0; i < ids.length; i++) {
        const node = soft.nodes[ids[i]];
        if (!node) continue;

        const px = (node.x || 0) + (node.vx || 0) * dtPos;
        const py = (node.y || 0) + (node.vy || 0) * dtPos;
        const rx = st.shapeRef[i * 2] || 0;
        const ry = st.shapeRef[i * 2 + 1] || 0;

        const tx = cx + rx * c - ry * sn;
        const ty = cy + rx * sn + ry * c;

        const localWeight = clamp(Number.isFinite(Number(node.shapeMemoryWeight)) ? Number(node.shapeMemoryWeight) : 1, 0, 1);
        if (localWeight <= 1e-6) continue;
        const invMass = 1 / Math.max(0.02, Number(node.mass) || 1);
        const corrPos = shapeMemoryGainApplied * localWeight * invMass;

        const nodeIndex = Number(ids[i]);
        shapeMemoryLayout.push(
          Number.isFinite(nodeIndex) ? nodeIndex : -1,
          px,
          py,
          tx,
          ty,
          corrPos,
          maxShift,
          rx,
          ry,
          c,
          sn,
        );
        nodeIndices.push(Number.isFinite(nodeIndex) ? (nodeIndex | 0) : -1);

        let ex = tx - px;
        let ey = ty - py;
        const eLen = Math.hypot(ex, ey);
        let dvx = 0;
        let dvy = 0;
        if (Number.isFinite(eLen) && eLen > 1e-7) {
          if (eLen > maxShift) {
            const k = maxShift / eLen;
            ex *= k;
            ey *= k;
          }
          dvx = (ex * corrPos) / dtPos;
          dvy = (ey * corrPos) / dtPos;
        }
        const safeDvx = Number.isFinite(dvx) ? dvx : 0;
        const safeDvy = Number.isFinite(dvy) ? dvy : 0;
        cpuDeltaVx.push(safeDvx);
        cpuDeltaVy.push(safeDvy);
        node.vx += safeDvx;
        node.vy += safeDvy;
      }

      touched += 1;
    }
  }

  const layout = shapeMemoryLayout.length > 0 ? Float32Array.from(shapeMemoryLayout) : null;
  const signature = layout ? computeShapeMemoryProposalSignature(layout, dtPos) : '';
  const authoritativeFromWgsl = canApplyAuthoritativeShapeMemoryProposal({
    wgslOffload,
    signature,
    expectedCount: nodeIndices.length,
  });

  if (authoritativeFromWgsl) {
    applyShapeMemoryVelocityDeltasAuthoritative({
      soft,
      nodeIndices,
      deltaVx: cpuDeltaVx,
      deltaVy: cpuDeltaVy,
      scale: -1,
    });
    applyShapeMemoryVelocityDeltasAuthoritative({
      soft,
      nodeIndices,
      deltaVx: wgslOffload.state.lastShapeMemoryProposalDeltaVx,
      deltaVy: wgslOffload.state.lastShapeMemoryProposalDeltaVy,
      scale: 1,
    });
  }

  if (wgslOffload?.state) {
    wgslOffload.state.lastShapeMemoryAuthoritativeSource = authoritativeFromWgsl
      ? 'wgsl-shape-memory-authoritative'
      : 'cpu-shape-memory-authoritative';
    wgslOffload.state.lastShapeMemoryAuthoritativeSignature = signature;
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state && wgslOffload?.device && layout && layout.length > 0) {
    wgslOffload.state.lastShapeMemoryProposalLayoutBytes = layout.byteLength;
    wgslOffload.state.lastShapeMemoryProposalSignaturePrepared = signature;
    const serializedDispatch = (wgslOffload.state.pendingWgslShapeMemoryProposalPromise || Promise.resolve())
      .catch(() => {})
      .then(() => dispatchSoftMembraneShapeMemoryProposal(wgslOffload, layout, dtPos, signature))
      .catch((err) => {
        wgslOffload.state.lastShapeMemoryProposalError = String(err?.message || err || 'unknown-error');
        wgslOffload.state.lastShapeMemoryProposalSource = 'cpu-shape-memory-authoritative';
        wgslOffload.state.lastMode = 'cpu-shape-memory-authoritative';
      });
    wgslOffload.state.pendingWgslShapeMemoryProposalPromise = serializedDispatch;
    if (!wgslOffload.state.lastShapeMemoryProposalSource) {
      wgslOffload.state.lastShapeMemoryProposalSource = 'cpu-shape-memory-authoritative';
    }
    if (!wgslOffload.state.lastMode) {
      wgslOffload.state.lastMode = 'cpu-shape-memory-authoritative';
    }
  }

  return touched;
}
