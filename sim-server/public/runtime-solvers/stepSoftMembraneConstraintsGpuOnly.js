const WGSL_WORKGROUP_SIZE = 64;
const SHAPE_LAYOUT_STRIDE_FLOATS = 11;

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
}) {
  if (!(membraneClusterSet instanceof Set) || membraneClusterSet.size === 0) return 0;

  ensureSoftMembraneLoopStateGpuOnly(sim, soft, loops, membraneClusterSet);

  const edgeAlpha = membraneEdgeBaseCompliance / Math.max(1e-8, dtPos * dtPos);
  const bendAlpha = membraneBendBaseCompliance / Math.max(1e-8, dtPos * dtPos);
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

        a.vx += (-wA * dl * nx) / dtPos;
        a.vy += (-wA * dl * ny) / dtPos;
        b.vx += (wB * dl * nx) / dtPos;
        b.vy += (wB * dl * ny) / dtPos;
        touched += 1;
      }
    }
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

        prev.vx += (-wP * dl * ux) / dtPos;
        prev.vy += (-wP * dl * uy) / dtPos;
        next.vx += (wN * dl * ux) / dtPos;
        next.vy += (wN * dl * uy) / dtPos;
      }
    }
  }

  return touched;
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

        let ex = tx - px;
        let ey = ty - py;
        const eLen = Math.hypot(ex, ey);
        if (!Number.isFinite(eLen) || eLen <= 1e-7) continue;
        if (eLen > maxShift) {
          const k = maxShift / eLen;
          ex *= k;
          ey *= k;
        }

        node.vx += (ex * corrPos) / dtPos;
        node.vy += (ey * corrPos) / dtPos;
      }

      touched += 1;
    }
  }

  if (wgslOffload?.enabled === true && wgslOffload?.state && wgslOffload?.device && shapeMemoryLayout.length > 0) {
    const layout = Float32Array.from(shapeMemoryLayout);
    const signature = computeShapeMemoryProposalSignature(layout, dtPos);
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
