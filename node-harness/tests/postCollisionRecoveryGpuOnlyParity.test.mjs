import test from 'node:test';
import assert from 'node:assert/strict';
import { applyPostCollisionRecoveryGpuOnly } from '../../sim-server/public/runtime-solvers/stepPostCollisionRecoveryGpuOnly.js';

function makeState() {
  return {
    sim: { frame: 42 },
    bodies: {
      rigid: [
        { x: 6.5, y: 6.2, vx: 0.7, vy: -0.3, omega: 0.04 },
        { x: 14.1, y: 8.7, vx: -0.4, vy: 0.9, omega: -0.06 },
      ],
    },
    soft: {
      nodes: [
        { x: 5.8, y: 6.9, vx: 0.2, vy: -0.1, clusterId: 1 },
        { x: 6.9, y: 6.5, vx: -0.15, vy: 0.25, clusterId: 1 },
        { x: 13.6, y: 8.9, vx: 0.4, vy: -0.2, clusterId: 3 },
      ],
    },
    dtNorm: 1.35,
    n: 32,
    softClusterLoops: [{ clusterId: 1 }, { clusterId: 3 }],
    hybridAttachedByRigid: new Map([[0, new Set([0])]]),
    softMembraneClusterSet: new Set([3]),
    softClusterCollisionLinearProjection: 0.22,
    softClusterCollisionAngularProjection: 0.14,
  };
}

function makeCallbacks(log) {
  return {
    computeSoftClusterKinematics(nodes) {
      const by = new Map();
      for (const node of nodes) {
        const cid = node.clusterId ?? 0;
        const st = by.get(cid) || { x: 0, y: 0, vx: 0, vy: 0, count: 0, omega: 0.07 * (cid + 1) };
        st.x += node.x;
        st.y += node.y;
        st.vx += node.vx;
        st.vy += node.vy;
        st.count += 1;
        by.set(cid, st);
      }
      for (const st of by.values()) {
        const inv = 1 / Math.max(1, st.count);
        st.x *= inv;
        st.y *= inv;
        st.vx *= inv;
        st.vy *= inv;
      }
      log.push('kinematics');
      return by;
    },
    projectNodesTowardClusterRigidMotion(nodes, clusterMap, opts) {
      log.push(`project:${opts.linearGain.toFixed(6)}:${opts.angularGain.toFixed(6)}`);
      for (const node of nodes) {
        const cid = node.clusterId ?? 0;
        const c = clusterMap.get(cid);
        if (!c) continue;
        const rx = node.x - c.x;
        const ry = node.y - c.y;
        const targetVx = c.vx - c.omega * ry;
        const targetVy = c.vy + c.omega * rx;
        const scale = opts.membraneClusterSet?.has(cid) ? opts.membraneGainScale : 1;
        node.vx += (targetVx - node.vx) * opts.linearGain * scale;
        node.vy += (targetVy - node.vy) * opts.linearGain * scale;
        node.vx += (-c.omega * ry) * opts.angularGain * scale;
        node.vy += (c.omega * rx) * opts.angularGain * scale;
      }
    },
    applyRigidInsideCorrectionPass(bodies) {
      log.push('rigidInside');
      for (const rb of bodies.rigid) {
        rb.vx += 0.01;
        rb.vy -= 0.005;
      }
      return 2;
    },
    applyMembraneInsideCorrectionPass(_sim, soft) {
      log.push('membraneInside');
      for (const node of soft.nodes) {
        if ((node.clusterId ?? 0) === 3) node.vy += 0.03;
      }
      return 1;
    },
    applyBounceBoundary(body, _n, bounce) {
      log.push(`bounce:${bounce}`);
      body.vx *= bounce;
      body.vy *= bounce;
    },
  };
}

function runBaselineInline(state, callbacks) {
  const postCollisionClusterKinematics = callbacks.computeSoftClusterKinematics(state.soft.nodes);
  callbacks.projectNodesTowardClusterRigidMotion(state.soft.nodes, postCollisionClusterKinematics, {
    linearGain: state.softClusterCollisionLinearProjection * state.dtNorm,
    angularGain: state.softClusterCollisionAngularProjection * state.dtNorm,
    membraneClusterSet: state.softMembraneClusterSet,
    membraneGainScale: 0.72,
  });

  const rigidInsideCorrections = callbacks.applyRigidInsideCorrectionPass(state.bodies, state.soft, state.hybridAttachedByRigid);
  const membraneInsideCorrections = callbacks.applyMembraneInsideCorrectionPass(state.sim, state.soft, state.softClusterLoops);
  for (const rb of state.bodies.rigid) callbacks.applyBounceBoundary(rb, state.n, 0.84);
  for (const sn of state.soft.nodes) callbacks.applyBounceBoundary(sn, state.n, 0.78);
  return { rigidInsideCorrections, membraneInsideCorrections };
}

test('post-collision recovery parity: gpu-only module matches baseline projection + inside-correction sequencing', () => {
  const baselineState = makeState();
  const gpuState = makeState();

  const baselineLog = [];
  const gpuLog = [];
  const baselineCallbacks = makeCallbacks(baselineLog);
  const gpuCallbacks = makeCallbacks(gpuLog);

  const baseline = runBaselineInline(baselineState, baselineCallbacks);
  const gpu = applyPostCollisionRecoveryGpuOnly({
    sim: gpuState.sim,
    bodies: gpuState.bodies,
    soft: gpuState.soft,
    dtNorm: gpuState.dtNorm,
    softMembraneClusterSet: gpuState.softMembraneClusterSet,
    softClusterCollisionLinearProjection: gpuState.softClusterCollisionLinearProjection,
    softClusterCollisionAngularProjection: gpuState.softClusterCollisionAngularProjection,
    membraneGainScale: 0.72,
    computeSoftClusterKinematics: gpuCallbacks.computeSoftClusterKinematics,
    projectNodesTowardClusterRigidMotion: gpuCallbacks.projectNodesTowardClusterRigidMotion,
    applyRigidInsideCorrectionPass: gpuCallbacks.applyRigidInsideCorrectionPass,
    applyMembraneInsideCorrectionPass: gpuCallbacks.applyMembraneInsideCorrectionPass,
    applyBounceBoundary: gpuCallbacks.applyBounceBoundary,
    n: gpuState.n,
    softClusterLoops: gpuState.softClusterLoops,
    hybridAttachedByRigid: gpuState.hybridAttachedByRigid,
  });

  assert.equal(gpu.rigidInsideCorrections, baseline.rigidInsideCorrections);
  assert.equal(gpu.membraneInsideCorrections, baseline.membraneInsideCorrections);
  assert.deepEqual(gpuState, baselineState);
  assert.deepEqual(gpuLog, baselineLog, 'gpu-only orchestration sequence should match baseline sequencing');
});
