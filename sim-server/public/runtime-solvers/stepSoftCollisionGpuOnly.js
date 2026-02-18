/**
 * GPU-only runtime soft↔soft collision pass.
 * Isolates soft node-vs-node and soft node-vs-foreign-soft-edge collision stepping
 * for the gpu-only runtime path while preserving baseline/default behavior contracts.
 */

const WGSL_WORKGROUP_SIZE = 1;

const softNodeNodeCollisionWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  restitution: f32,
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> posX: array<f32>;
@group(0) @binding(2) var<storage, read_write> posY: array<f32>;
@group(0) @binding(3) var<storage, read_write> velX: array<f32>;
@group(0) @binding(4) var<storage, read_write> velY: array<f32>;
@group(0) @binding(5) var<storage, read> mass: array<f32>;
@group(0) @binding(6) var<storage, read> radius: array<f32>;

fn safeMass(v: f32) -> f32 {
  return max(0.02, v);
}

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x > 0u) { return; }

  let count = params.nodeCount;
  var i: u32 = 0u;
  loop {
    if (i >= count) { break; }
    var j: u32 = i + 1u;
    loop {
      if (j >= count) { break; }

      var ax = posX[i];
      var ay = posY[i];
      var avx = velX[i];
      var avy = velY[i];
      var bx = posX[j];
      var by = posY[j];
      var bvx = velX[j];
      var bvy = velY[j];

      let dx = bx - ax;
      let dy = by - ay;
      let d2 = dx * dx + dy * dy;
      let minDist = radius[i] + radius[j];
      if (d2 <= 1e-10) {
        ax = ax - 0.01;
        bx = bx + 0.01;
      } else {
        let minD2 = minDist * minDist;
        if (d2 < minD2) {
          let d = sqrt(d2);
          let nx = dx / d;
          let ny = dy / d;
          let penetration = minDist - d;

          let invA = 1.0 / safeMass(mass[i]);
          let invB = 1.0 / safeMass(mass[j]);
          let invSum = invA + invB;

          let corr = (penetration / max(1e-6, invSum)) * 0.85;
          ax = ax - nx * corr * invA;
          ay = ay - ny * corr * invA;
          bx = bx + nx * corr * invB;
          by = by + ny * corr * invB;

          let rvx = bvx - avx;
          let rvy = bvy - avy;
          let vn = rvx * nx + rvy * ny;
          if (vn <= 0.0) {
            let impulse = (-(1.0 + params.restitution) * vn) / max(1e-6, invSum);
            let ix = impulse * nx;
            let iy = impulse * ny;
            avx = avx - ix * invA;
            avy = avy - iy * invA;
            bvx = bvx + ix * invB;
            bvy = bvy + iy * invB;
          }
        }
      }

      posX[i] = ax; posY[i] = ay;
      velX[i] = avx; velY[i] = avy;
      posX[j] = bx; posY[j] = by;
      velX[j] = bvx; velY[j] = bvy;

      j = j + 1u;
    }
    i = i + 1u;
  }
}
`;

const softNodeEdgeCollisionWgsl = /* wgsl */`
struct Params {
  nodeCount: u32,
  pairCount: u32,
  restitution: f32,
  nodeEdgeSlop: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> posX: array<f32>;
@group(0) @binding(2) var<storage, read_write> posY: array<f32>;
@group(0) @binding(3) var<storage, read_write> velX: array<f32>;
@group(0) @binding(4) var<storage, read_write> velY: array<f32>;
@group(0) @binding(5) var<storage, read> radius: array<f32>;
@group(0) @binding(6) var<storage, read> pairs: array<u32>;

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
  return min(max(v, lo), hi);
}

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x > 0u) { return; }

  var k: u32 = 0u;
  loop {
    if (k >= params.pairCount) { break; }
    let base = k * 3u;
    let ni = pairs[base + 0u];
    let ia = pairs[base + 1u];
    let ib = pairs[base + 2u];
    if (ni >= params.nodeCount || ia >= params.nodeCount || ib >= params.nodeCount) {
      k = k + 1u;
      continue;
    }

    var nodeX = posX[ni];
    var nodeY = posY[ni];
    var nodeVx = velX[ni];
    var nodeVy = velY[ni];

    var ax = posX[ia];
    var ay = posY[ia];
    var avx = velX[ia];
    var avy = velY[ia];

    var bx = posX[ib];
    var by = posY[ib];
    var bvx = velX[ib];
    var bvy = velY[ib];

    let abx = bx - ax;
    let aby = by - ay;
    let ab2 = abx * abx + aby * aby;
    let t = select(clampf(((nodeX - ax) * abx + (nodeY - ay) * aby) / max(ab2, 1e-12), 0.0, 1.0), 0.0, ab2 <= 1e-12);
    let cpx = ax + abx * t;
    let cpy = ay + aby * t;

    var nx = nodeX - cpx;
    var ny = nodeY - cpy;
    var dist = sqrt(nx * nx + ny * ny);
    let minDist = max(0.4, radius[ni]);

    if (dist < minDist) {
      if (dist < 1e-6) {
        let invLen = 1.0 / max(1e-6, sqrt((-aby) * (-aby) + abx * abx));
        nx = -aby * invLen;
        ny = abx * invLen;
        dist = 1e-6;
      } else {
        nx = nx / dist;
        ny = ny / dist;
      }

      let penetration = minDist - dist;
      nodeX = nodeX + nx * penetration * 0.92;
      nodeY = nodeY + ny * penetration * 0.92;
      ax = ax - nx * penetration * 0.04;
      ay = ay - ny * penetration * 0.04;
      bx = bx - nx * penetration * 0.04;
      by = by - ny * penetration * 0.04;

      let edgeVx = (avx + bvx) * 0.5;
      let edgeVy = (avy + bvy) * 0.5;
      let rvx = nodeVx - edgeVx;
      let rvy = nodeVy - edgeVy;
      let vn = rvx * nx + rvy * ny;
      if (vn < 0.0) {
        let impulse = -(1.0 + params.restitution) * vn;
        nodeVx = nodeVx + nx * impulse;
        nodeVy = nodeVy + ny * impulse;
      }
    }

    posX[ni] = nodeX;
    posY[ni] = nodeY;
    velX[ni] = nodeVx;
    velY[ni] = nodeVy;

    posX[ia] = ax;
    posY[ia] = ay;
    velX[ia] = avx;
    velY[ia] = avy;

    posX[ib] = bx;
    posY[ib] = by;
    velX[ib] = bvx;
    velY[ib] = bvy;

    k = k + 1u;
  }
}
`;

function canUseWgslOffload(offload) {
  if (!offload || offload.enabled !== true) return false;
  if (!offload.device || typeof offload.device.createComputePipelineAsync !== 'function') return false;
  if (typeof globalThis.GPUBufferUsage === 'undefined' || typeof globalThis.GPUMapMode === 'undefined') return false;
  return true;
}

function createBuffer(device, size, usage) {
  return device.createBuffer({ size, usage });
}

async function ensureNodeNodeWgslState(offload, count) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;

  if (!state.softNodeNodeCollisionPipeline) {
    const module = device.createShaderModule({ code: softNodeNodeCollisionWgsl });
    state.softNodeNodeCollisionPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  const requiredCapacity = Math.max(1, count);
  if ((state.softNodeNodeCollisionCapacity || 0) < requiredCapacity) {
    const capacity = Math.max(requiredCapacity, state.softNodeNodeCollisionCapacity ? state.softNodeNodeCollisionCapacity * 2 : 128);
    const bytes = capacity * 4;
    const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.COPY_SRC;
    const readUsage = globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ;

    state.softNodeNodePosX?.destroy?.();
    state.softNodeNodePosY?.destroy?.();
    state.softNodeNodeVelX?.destroy?.();
    state.softNodeNodeVelY?.destroy?.();
    state.softNodeNodeMass?.destroy?.();
    state.softNodeNodeRadius?.destroy?.();
    state.softNodeNodeReadPosX?.destroy?.();
    state.softNodeNodeReadPosY?.destroy?.();
    state.softNodeNodeReadVelX?.destroy?.();
    state.softNodeNodeReadVelY?.destroy?.();

    state.softNodeNodePosX = createBuffer(device, bytes, storageUsage);
    state.softNodeNodePosY = createBuffer(device, bytes, storageUsage);
    state.softNodeNodeVelX = createBuffer(device, bytes, storageUsage);
    state.softNodeNodeVelY = createBuffer(device, bytes, storageUsage);
    state.softNodeNodeMass = createBuffer(device, bytes, globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST);
    state.softNodeNodeRadius = createBuffer(device, bytes, globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST);

    state.softNodeNodeReadPosX = createBuffer(device, bytes, readUsage);
    state.softNodeNodeReadPosY = createBuffer(device, bytes, readUsage);
    state.softNodeNodeReadVelX = createBuffer(device, bytes, readUsage);
    state.softNodeNodeReadVelY = createBuffer(device, bytes, readUsage);
    state.softNodeNodeCollisionCapacity = capacity;
    state.softNodeNodeCollisionBindGroup = null;
  }

  if (!state.softNodeNodeCollisionParams) {
    state.softNodeNodeCollisionParams = createBuffer(
      device,
      16,
      globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    );
  }

  if (!state.softNodeNodeCollisionBindGroup) {
    state.softNodeNodeCollisionBindGroup = device.createBindGroup({
      layout: state.softNodeNodeCollisionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.softNodeNodeCollisionParams } },
        { binding: 1, resource: { buffer: state.softNodeNodePosX } },
        { binding: 2, resource: { buffer: state.softNodeNodePosY } },
        { binding: 3, resource: { buffer: state.softNodeNodeVelX } },
        { binding: 4, resource: { buffer: state.softNodeNodeVelY } },
        { binding: 5, resource: { buffer: state.softNodeNodeMass } },
        { binding: 6, resource: { buffer: state.softNodeNodeRadius } },
      ],
    });
  }

  return state;
}

function allFinite(arr) {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) return false;
  }
  return true;
}

function hashSoftNodeEdgeCandidateTriples(triples) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < triples.length; i++) {
    hash ^= (triples[i] >>> 0);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function buildSoftNodeEdgeCandidateLayout({ nodes, springs, edgeBodyModeBlock }) {
  const nodeCount = Number(nodes?.length) || 0;
  const perNodePairs = Array.from({ length: nodeCount }, () => []);
  for (let ni = 0; ni < nodeCount; ni++) {
    const node = nodes[ni];
    for (const [i, j, _rest, edgeBodyMode] of springs) {
      if (edgeBodyMode !== edgeBodyModeBlock) continue;
      if (i === ni || j === ni) continue;
      const a = nodes[i];
      const b = nodes[j];
      if (!a || !b) continue;
      if (a.clusterId === node.clusterId && b.clusterId === node.clusterId) continue;
      perNodePairs[ni].push([i >>> 0, j >>> 0]);
    }
  }

  const nodeOffsets = new Uint32Array(nodeCount + 1);
  let pairCount = 0;
  for (let ni = 0; ni < nodeCount; ni++) {
    nodeOffsets[ni] = pairCount;
    pairCount += perNodePairs[ni].length;
  }
  nodeOffsets[nodeCount] = pairCount;

  const edgeNodeA = new Uint32Array(pairCount);
  const edgeNodeB = new Uint32Array(pairCount);
  const triples = new Uint32Array(pairCount * 3);
  let write = 0;
  for (let ni = 0; ni < nodeCount; ni++) {
    const bucket = perNodePairs[ni];
    for (let k = 0; k < bucket.length; k++) {
      const [i, j] = bucket[k];
      edgeNodeA[write] = i;
      edgeNodeB[write] = j;
      const base = write * 3;
      triples[base + 0] = ni >>> 0;
      triples[base + 1] = i;
      triples[base + 2] = j;
      write += 1;
    }
  }

  let signature = hashSoftNodeEdgeCandidateTriples(triples);
  signature ^= hashSoftNodeEdgeCandidateTriples(nodeOffsets);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashSoftNodeEdgeCandidateTriples(edgeNodeA);
  signature = Math.imul(signature, 0x01000193) >>> 0;
  signature ^= hashSoftNodeEdgeCandidateTriples(edgeNodeB);

  return {
    packed: triples,
    pairCount,
    nodeOffsets,
    edgeNodeA,
    edgeNodeB,
    signature: signature >>> 0,
  };
}

async function runNodeNodeCollisionWgsl(nodes, restitution, wgslOffload) {
  const count = nodes.length;
  if (count < 2) return { ok: true, reason: 'insufficient-node-count' };

  const state = await ensureNodeNodeWgslState(wgslOffload, count);
  const device = wgslOffload.device;
  const bytes = count * 4;

  const posX = new Float32Array(count);
  const posY = new Float32Array(count);
  const velX = new Float32Array(count);
  const velY = new Float32Array(count);
  const mass = new Float32Array(count);
  const radius = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const node = nodes[i] || {};
    posX[i] = Number(node.x) || 0;
    posY[i] = Number(node.y) || 0;
    velX[i] = Number(node.vx) || 0;
    velY[i] = Number(node.vy) || 0;
    mass[i] = Number(node.mass) || 1;
    radius[i] = Number(node.r) || 1;
  }

  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, count, true);
  view.setFloat32(4, Number(restitution) || 0, true);

  device.queue.writeBuffer(state.softNodeNodeCollisionParams, 0, params);
  device.queue.writeBuffer(state.softNodeNodePosX, 0, posX);
  device.queue.writeBuffer(state.softNodeNodePosY, 0, posY);
  device.queue.writeBuffer(state.softNodeNodeVelX, 0, velX);
  device.queue.writeBuffer(state.softNodeNodeVelY, 0, velY);
  device.queue.writeBuffer(state.softNodeNodeMass, 0, mass);
  device.queue.writeBuffer(state.softNodeNodeRadius, 0, radius);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.softNodeNodeCollisionPipeline);
  pass.setBindGroup(0, state.softNodeNodeCollisionBindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();

  encoder.copyBufferToBuffer(state.softNodeNodePosX, 0, state.softNodeNodeReadPosX, 0, bytes);
  encoder.copyBufferToBuffer(state.softNodeNodePosY, 0, state.softNodeNodeReadPosY, 0, bytes);
  encoder.copyBufferToBuffer(state.softNodeNodeVelX, 0, state.softNodeNodeReadVelX, 0, bytes);
  encoder.copyBufferToBuffer(state.softNodeNodeVelY, 0, state.softNodeNodeReadVelY, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    state.softNodeNodeReadPosX.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.softNodeNodeReadPosY.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.softNodeNodeReadVelX.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
    state.softNodeNodeReadVelY.mapAsync(globalThis.GPUMapMode.READ, 0, bytes),
  ]);

  const outPosX = new Float32Array(state.softNodeNodeReadPosX.getMappedRange(0, bytes).slice(0));
  const outPosY = new Float32Array(state.softNodeNodeReadPosY.getMappedRange(0, bytes).slice(0));
  const outVelX = new Float32Array(state.softNodeNodeReadVelX.getMappedRange(0, bytes).slice(0));
  const outVelY = new Float32Array(state.softNodeNodeReadVelY.getMappedRange(0, bytes).slice(0));

  state.softNodeNodeReadPosX.unmap();
  state.softNodeNodeReadPosY.unmap();
  state.softNodeNodeReadVelX.unmap();
  state.softNodeNodeReadVelY.unmap();

  const finite = allFinite(outPosX) && allFinite(outPosY) && allFinite(outVelX) && allFinite(outVelY);
  if (!finite) return { ok: false, reason: 'non-finite-node-node-wgsl' };

  for (let i = 0; i < count; i++) {
    const node = nodes[i];
    node.x = outPosX[i];
    node.y = outPosY[i];
    node.vx = outVelX[i];
    node.vy = outVelY[i];
  }

  return { ok: true, reason: 'ok' };
}

async function ensureNodeEdgeWgslState(offload, count, pairCount) {
  const state = offload.state || (offload.state = {});
  const device = offload.device;

  if (!state.softNodeEdgeCollisionPipeline) {
    const module = device.createShaderModule({ code: softNodeEdgeCollisionWgsl });
    state.softNodeEdgeCollisionPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  const requiredNodeCapacity = Math.max(1, count);
  if ((state.softNodeEdgeCollisionNodeCapacity || 0) < requiredNodeCapacity) {
    const nodeCapacity = Math.max(requiredNodeCapacity, state.softNodeEdgeCollisionNodeCapacity ? state.softNodeEdgeCollisionNodeCapacity * 2 : 128);
    const bytes = nodeCapacity * 4;
    const storageUsage = globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.COPY_SRC;
    const readUsage = globalThis.GPUBufferUsage.COPY_DST | globalThis.GPUBufferUsage.MAP_READ;

    state.softNodeEdgePosX?.destroy?.();
    state.softNodeEdgePosY?.destroy?.();
    state.softNodeEdgeVelX?.destroy?.();
    state.softNodeEdgeVelY?.destroy?.();
    state.softNodeEdgeRadius?.destroy?.();
    state.softNodeEdgeReadPosX?.destroy?.();
    state.softNodeEdgeReadPosY?.destroy?.();
    state.softNodeEdgeReadVelX?.destroy?.();
    state.softNodeEdgeReadVelY?.destroy?.();

    state.softNodeEdgePosX = createBuffer(device, bytes, storageUsage);
    state.softNodeEdgePosY = createBuffer(device, bytes, storageUsage);
    state.softNodeEdgeVelX = createBuffer(device, bytes, storageUsage);
    state.softNodeEdgeVelY = createBuffer(device, bytes, storageUsage);
    state.softNodeEdgeRadius = createBuffer(device, bytes, globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST);
    state.softNodeEdgeReadPosX = createBuffer(device, bytes, readUsage);
    state.softNodeEdgeReadPosY = createBuffer(device, bytes, readUsage);
    state.softNodeEdgeReadVelX = createBuffer(device, bytes, readUsage);
    state.softNodeEdgeReadVelY = createBuffer(device, bytes, readUsage);
    state.softNodeEdgeCollisionNodeCapacity = nodeCapacity;
    state.softNodeEdgeCollisionBindGroup = null;
  }

  const requiredPairCapacity = Math.max(1, pairCount * 3);
  if ((state.softNodeEdgeCollisionPairCapacity || 0) < requiredPairCapacity) {
    const pairCapacity = Math.max(requiredPairCapacity, state.softNodeEdgeCollisionPairCapacity ? state.softNodeEdgeCollisionPairCapacity * 2 : 256);
    state.softNodeEdgePairs?.destroy?.();
    state.softNodeEdgePairs = createBuffer(
      device,
      pairCapacity * Uint32Array.BYTES_PER_ELEMENT,
      globalThis.GPUBufferUsage.STORAGE | globalThis.GPUBufferUsage.COPY_DST,
    );
    state.softNodeEdgeCollisionPairCapacity = pairCapacity;
    state.softNodeEdgeCollisionBindGroup = null;
  }

  if (!state.softNodeEdgeCollisionParams) {
    state.softNodeEdgeCollisionParams = createBuffer(
      device,
      16,
      globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    );
  }

  if (!state.softNodeEdgeCollisionBindGroup) {
    state.softNodeEdgeCollisionBindGroup = device.createBindGroup({
      layout: state.softNodeEdgeCollisionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.softNodeEdgeCollisionParams } },
        { binding: 1, resource: { buffer: state.softNodeEdgePosX } },
        { binding: 2, resource: { buffer: state.softNodeEdgePosY } },
        { binding: 3, resource: { buffer: state.softNodeEdgeVelX } },
        { binding: 4, resource: { buffer: state.softNodeEdgeVelY } },
        { binding: 5, resource: { buffer: state.softNodeEdgeRadius } },
        { binding: 6, resource: { buffer: state.softNodeEdgePairs } },
      ],
    });
  }

  return state;
}

async function runNodeEdgeCollisionWgsl(nodes, nodeEdgePacked, restitution, nodeEdgeSlop, wgslOffload) {
  const count = nodes.length;
  const pairCount = Math.floor((nodeEdgePacked?.length || 0) / 3);
  if (count < 3 || pairCount <= 0) return { ok: true, reason: 'insufficient-node-edge-count' };

  const state = await ensureNodeEdgeWgslState(wgslOffload, count, pairCount);
  const device = wgslOffload.device;
  const nodeBytes = count * 4;

  const posX = new Float32Array(count);
  const posY = new Float32Array(count);
  const velX = new Float32Array(count);
  const velY = new Float32Array(count);
  const radius = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const node = nodes[i] || {};
    posX[i] = Number(node.x) || 0;
    posY[i] = Number(node.y) || 0;
    velX[i] = Number(node.vx) || 0;
    velY[i] = Number(node.vy) || 0;
    radius[i] = Number(node.r) || 1;
  }

  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, count, true);
  view.setUint32(4, pairCount, true);
  view.setFloat32(8, Number(restitution) || 0, true);
  view.setFloat32(12, Number(nodeEdgeSlop) || 0, true);

  device.queue.writeBuffer(state.softNodeEdgeCollisionParams, 0, params);
  device.queue.writeBuffer(state.softNodeEdgePosX, 0, posX);
  device.queue.writeBuffer(state.softNodeEdgePosY, 0, posY);
  device.queue.writeBuffer(state.softNodeEdgeVelX, 0, velX);
  device.queue.writeBuffer(state.softNodeEdgeVelY, 0, velY);
  device.queue.writeBuffer(state.softNodeEdgeRadius, 0, radius);
  device.queue.writeBuffer(state.softNodeEdgePairs, 0, nodeEdgePacked);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.softNodeEdgeCollisionPipeline);
  pass.setBindGroup(0, state.softNodeEdgeCollisionBindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();

  encoder.copyBufferToBuffer(state.softNodeEdgePosX, 0, state.softNodeEdgeReadPosX, 0, nodeBytes);
  encoder.copyBufferToBuffer(state.softNodeEdgePosY, 0, state.softNodeEdgeReadPosY, 0, nodeBytes);
  encoder.copyBufferToBuffer(state.softNodeEdgeVelX, 0, state.softNodeEdgeReadVelX, 0, nodeBytes);
  encoder.copyBufferToBuffer(state.softNodeEdgeVelY, 0, state.softNodeEdgeReadVelY, 0, nodeBytes);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    state.softNodeEdgeReadPosX.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes),
    state.softNodeEdgeReadPosY.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes),
    state.softNodeEdgeReadVelX.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes),
    state.softNodeEdgeReadVelY.mapAsync(globalThis.GPUMapMode.READ, 0, nodeBytes),
  ]);

  const outPosX = new Float32Array(state.softNodeEdgeReadPosX.getMappedRange(0, nodeBytes).slice(0));
  const outPosY = new Float32Array(state.softNodeEdgeReadPosY.getMappedRange(0, nodeBytes).slice(0));
  const outVelX = new Float32Array(state.softNodeEdgeReadVelX.getMappedRange(0, nodeBytes).slice(0));
  const outVelY = new Float32Array(state.softNodeEdgeReadVelY.getMappedRange(0, nodeBytes).slice(0));

  state.softNodeEdgeReadPosX.unmap();
  state.softNodeEdgeReadPosY.unmap();
  state.softNodeEdgeReadVelX.unmap();
  state.softNodeEdgeReadVelY.unmap();

  const finite = allFinite(outPosX) && allFinite(outPosY) && allFinite(outVelX) && allFinite(outVelY);
  if (!finite) return { ok: false, reason: 'non-finite-node-edge-wgsl' };

  for (let i = 0; i < count; i++) {
    const node = nodes[i];
    node.x = outPosX[i];
    node.y = outPosY[i];
    node.vx = outVelX[i];
    node.vy = outVelY[i];
  }

  return { ok: true, reason: 'ok' };
}

export async function resolveSoftSoftCollisionPassGpuOnly({
  soft,
  resolveCircleCollision,
  resolveSoftNodeVsSoftEdgeCollision,
  edgeBodyModeBlock,
  nodeNodeSlop = 0.22,
  nodeEdgeSlop = 0.12,
  wgslOffload,
}) {
  if (!soft || !Array.isArray(soft.nodes) || !Array.isArray(soft.springs)) return;
  if (typeof resolveCircleCollision !== 'function' || typeof resolveSoftNodeVsSoftEdgeCollision !== 'function') {
    throw new Error('gpu-only soft collision pass requires soft collision callbacks');
  }

  const nodes = soft.nodes;
  const springs = soft.springs;

  let nodeNodeSource = 'cpu-soft-node-node-authoritative';
  const canRunNodeNodeWgsl = canUseWgslOffload(wgslOffload) && nodes.length >= 2;
  if (canRunNodeNodeWgsl) {
    try {
      const result = await runNodeNodeCollisionWgsl(nodes, nodeNodeSlop, wgslOffload);
      if (result.ok) {
        nodeNodeSource = 'wgsl-soft-node-node-authoritative';
      } else {
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            resolveCircleCollision(nodes[i], nodes[j], nodeNodeSlop);
          }
        }
        nodeNodeSource = 'cpu-soft-node-node-fallback-nonfinite';
      }
    } catch (_err) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          resolveCircleCollision(nodes[i], nodes[j], nodeNodeSlop);
        }
      }
      nodeNodeSource = 'cpu-soft-node-node-fallback-error';
    }
  } else {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        resolveCircleCollision(nodes[i], nodes[j], nodeNodeSlop);
      }
    }
  }

  const nodeEdgeLayout = buildSoftNodeEdgeCandidateLayout({
    nodes,
    springs,
    edgeBodyModeBlock,
  });
  const nodeEdgePacked = nodeEdgeLayout.packed;
  let nodeEdgeSource = 'cpu-soft-node-edge-authoritative';
  const canRunNodeEdgeWgsl = canUseWgslOffload(wgslOffload) && nodeEdgeLayout.pairCount > 0;
  if (canRunNodeEdgeWgsl) {
    try {
      const result = await runNodeEdgeCollisionWgsl(nodes, nodeEdgePacked, nodeEdgeSlop, nodeEdgeSlop, wgslOffload);
      if (result.ok) {
        nodeEdgeSource = 'wgsl-soft-node-edge-authoritative';
      } else {
        for (let k = 0; k < nodeEdgePacked.length; k += 3) {
          const ni = nodeEdgePacked[k] | 0;
          const i = nodeEdgePacked[k + 1] | 0;
          const j = nodeEdgePacked[k + 2] | 0;
          const node = nodes[ni];
          const a = nodes[i];
          const b = nodes[j];
          if (!node || !a || !b) continue;
          resolveSoftNodeVsSoftEdgeCollision(node, a, b, nodeEdgeSlop);
        }
        nodeEdgeSource = 'cpu-soft-node-edge-fallback-nonfinite';
      }
    } catch (_err) {
      for (let k = 0; k < nodeEdgePacked.length; k += 3) {
        const ni = nodeEdgePacked[k] | 0;
        const i = nodeEdgePacked[k + 1] | 0;
        const j = nodeEdgePacked[k + 2] | 0;
        const node = nodes[ni];
        const a = nodes[i];
        const b = nodes[j];
        if (!node || !a || !b) continue;
        resolveSoftNodeVsSoftEdgeCollision(node, a, b, nodeEdgeSlop);
      }
      nodeEdgeSource = 'cpu-soft-node-edge-fallback-error';
    }
  } else {
    for (let k = 0; k < nodeEdgePacked.length; k += 3) {
      const ni = nodeEdgePacked[k] | 0;
      const i = nodeEdgePacked[k + 1] | 0;
      const j = nodeEdgePacked[k + 2] | 0;
      const node = nodes[ni];
      const a = nodes[i];
      const b = nodes[j];
      if (!node || !a || !b) continue;
      resolveSoftNodeVsSoftEdgeCollision(node, a, b, nodeEdgeSlop);
    }
  }

  if (wgslOffload?.state) {
    const state = wgslOffload.state;
    const modeProfile = String(wgslOffload?.modeProfile || '').toLowerCase();
    const nonStandardMode = modeProfile === 'gpu-only-fast' || modeProfile === 'gpu-only-validated';
    state.lastSoftNodeNodeCollisionSource = nodeNodeSource;
    state.lastSoftNodeEdgeCandidateLayoutSource = 'cpu-soft-node-edge-candidate-layout';
    state.lastSoftNodeEdgeCandidatePairCount = nodeEdgeLayout.pairCount;
    state.lastSoftNodeEdgeCandidateLayoutSignature = nodeEdgeLayout.signature >>> 0;
    state.lastSoftNodeEdgeCandidatePacked = nodeEdgePacked;
    state.lastSoftNodeEdgeCandidateNodeOffsets = nodeEdgeLayout.nodeOffsets;
    state.lastSoftNodeEdgeCandidateEdgeNodeA = nodeEdgeLayout.edgeNodeA;
    state.lastSoftNodeEdgeCandidateEdgeNodeB = nodeEdgeLayout.edgeNodeB;
    state.lastSoftNodeEdgeCollisionSource = nodeEdgeSource;
    state.lastSoftNodeEdgeNextStage = nodeEdgeSource.startsWith('wgsl-')
      ? nodeEdgeSource
      : (nonStandardMode ? 'wgsl-soft-node-edge-collision-pending' : 'cpu-soft-node-edge-authoritative');
    state.lastSourceRoute = nodeEdgeSource.startsWith('wgsl-')
      ? nodeEdgeSource
      : nodeNodeSource;
    state.lastMode = nodeEdgeSource.startsWith('wgsl-')
      ? 'wgsl-soft-node-edge-authoritative'
      : (nodeNodeSource.startsWith('wgsl-')
        ? 'wgsl-soft-node-node-authoritative'
        : 'cpu-soft-node-node-authoritative');
  }
}
