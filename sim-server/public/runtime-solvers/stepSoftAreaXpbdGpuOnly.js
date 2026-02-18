/**
 * GPU-only runtime soft area XPBD pass.
 * Isolates area-preservation constraint stepping for the gpu-only runtime path
 * while preserving baseline/default behavior and call contracts.
 */
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
    // Unblocker for the upcoming WGSL area-XPBD stage: pre-pack deterministic
    // cluster ownership + SoA buffers so compute kernels can execute per-cluster
    // lambda proposal/reduction without ambiguous ragged-loop indexing.
    const plan = buildSoftAreaXpbdWgslPlan({ sim, soft, loops });
    const layout = buildSoftAreaXpbdWgslLayout(plan);
    wgslOffload.state.preparedPlan = plan;
    wgslOffload.state.preparedLayout = layout;
    wgslOffload.state.lastPreparedClusterCount = plan.clusterCount;
    wgslOffload.state.lastPreparedEndpointCount = plan.endpointCount;
    wgslOffload.state.lastPreparedLayoutBytes = layout.byteLength;
    wgslOffload.state.lastMode = 'cpu-prepared';
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
