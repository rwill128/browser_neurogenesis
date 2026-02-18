import { stepRigidBodiesGpuOnly } from '../sim-server/public/runtime-solvers/stepRigidGpuOnly.js';
import { resolveRigidRigidCollisionPassGpuOnly } from '../sim-server/public/runtime-solvers/stepRigidCollisionGpuOnly.js';
import { resolveRigidSoftCollisionPassGpuOnly } from '../sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js';

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function parseNumberList(value, fallback) {
  const raw = String(value || '').trim();
  if (!raw) return [...fallback];
  return raw
    .split(',')
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v) && v > 0);
}

function parseFloatList(value, fallback) {
  const raw = String(value || '').trim();
  if (!raw) return [...fallback];
  return raw
    .split(',')
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v) && v >= 0);
}

function parseSweepList(value) {
  const allowed = new Set(['scale', 'rigid', 'soft', 'contact']);
  const raw = String(value || 'scale,rigid,soft,contact').trim();
  const sweeps = raw
    .split(',')
    .map((v) => v.trim().toLowerCase())
    .filter((v) => allowed.has(v));
  return sweeps.length > 0 ? sweeps : ['scale'];
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function scenarioSeed({ rigidCount, softNodeCount, contactDensity, scenarioKey }) {
  let hash = 2166136261;
  const mix = (n) => {
    hash ^= n >>> 0;
    hash = Math.imul(hash, 16777619);
  };
  mix(rigidCount * 97);
  mix(softNodeCount * 193);
  mix(Math.round(contactDensity * 1000) * 389);
  for (let i = 0; i < scenarioKey.length; i++) mix(scenarioKey.charCodeAt(i) * (i + 17));
  return hash >>> 0;
}

function makeFixture({
  rigidCount,
  softNodeCount,
  contactDensity,
  scenarioKey,
  gridSize = 256,
}) {
  const seed = scenarioSeed({ rigidCount, softNodeCount, contactDensity, scenarioKey });
  const rand = mulberry32(seed);
  const n = gridSize;
  const cells = n * n;
  const center = n * 0.5;

  const contact = clamp(Number(contactDensity) || 0, 0.05, 1.0);
  const spreadNorm = clamp(1.0 - contact * 0.8, 0.15, 1.0);
  const rigidSpread = (n - 28) * spreadNorm;
  const softSpread = (n - 28) * clamp(spreadNorm + 0.1, 0.18, 1.0);

  const rigid = [];
  for (let i = 0; i < rigidCount; i++) {
    const mass = 1 + rand() * 4;
    const r = 1.4 + rand() * 2.4;
    const x = clamp(center + (rand() - 0.5) * rigidSpread, 12, n - 12);
    const y = clamp(center + (rand() - 0.5) * rigidSpread, 12, n - 12);
    rigid.push({
      x,
      y,
      vx: -0.8 + rand() * 1.6,
      vy: -0.8 + rand() * 1.6,
      omega: -0.08 + rand() * 0.16,
      theta: rand() * Math.PI * 2,
      mass,
      inertia: 0.5 * mass * r * r,
      r,
      edgeMomentumCoupling: [rand(), rand(), rand(), rand()],
      verticesLocal: [
        { x: -r, y: -r * 0.8 },
        { x: r, y: -r * 0.8 },
        { x: r * 0.9, y: r },
        { x: -r * 0.9, y: r },
      ],
    });
  }

  const soft = {
    nodes: [],
    springs: [],
  };

  if (softNodeCount > 0) {
    const side = Math.max(2, Math.ceil(Math.sqrt(softNodeCount)));
    const spacing = clamp(softSpread / Math.max(2, side - 1), 2.4, 10.0);
    const startX = center - ((side - 1) * spacing) * 0.5;
    const startY = center - ((side - 1) * spacing) * 0.5;

    for (let i = 0; i < softNodeCount; i++) {
      const gx = i % side;
      const gy = Math.floor(i / side);
      const jitter = (1.0 - contact) * 2.1;
      const x = clamp(startX + gx * spacing + (rand() - 0.5) * jitter, 8, n - 8);
      const y = clamp(startY + gy * spacing + (rand() - 0.5) * jitter, 8, n - 8);
      soft.nodes.push({
        id: i,
        x,
        y,
        vx: 0,
        vy: 0,
        r: 1.2,
        clusterId: i % 8,
      });
    }

    const blockProb = clamp(0.18 + contact * 0.72, 0.1, 0.95);
    for (let i = 0; i < soft.nodes.length - 1; i++) {
      soft.springs.push([i, i + 1, 1, rand() < blockProb ? 1 : 0]);
      if (i + side < soft.nodes.length) {
        soft.springs.push([i, i + side, 1, rand() < blockProb ? 1 : 0]);
      }
      if (i + side + 1 < soft.nodes.length && (i % side) !== (side - 1) && rand() < 0.5 * contact) {
        soft.springs.push([i, i + side + 1, 1, rand() < blockProb ? 1 : 0]);
      }
    }
  }

  const vxField = new Float32Array(cells);
  const vyField = new Float32Array(cells);
  for (let i = 0; i < cells; i++) {
    const x = i % n;
    const y = (i / n) | 0;
    vxField[i] = Math.sin(x * 0.03) * 0.5 + Math.cos(y * 0.02) * 0.25;
    vyField[i] = Math.cos(x * 0.025) * 0.4 - Math.sin(y * 0.035) * 0.22;
  }

  return {
    n,
    sim: { frame: 17 },
    bodies: { rigid, soft },
    vxField,
    vyField,
    obstacleMask: new Float32Array(cells),
    bodyFeedbackPrevVx: new Float32Array(cells),
    bodyFeedbackPrevVy: new Float32Array(cells),
    fixtureMeta: {
      rigidCount,
      softNodeCount,
      softSpringCount: soft.springs.length,
      contactDensity: contact,
      seed,
      scenarioKey,
    },
  };
}

function sampleFluidForBodyCoupling(field, nGrid, sx, sy, rx, ry, _obs, _feedback, suppression) {
  const xi = Math.max(0, Math.min(nGrid - 1, Math.round(sx)));
  const yi = Math.max(0, Math.min(nGrid - 1, Math.round(sy)));
  const base = field[yi * nGrid + xi] || 0;
  return base - suppression * 0.03 * (rx + ry);
}

function rigidVerticesWorld(b) {
  const c = Math.cos(b.theta || 0);
  const s = Math.sin(b.theta || 0);
  return (b.verticesLocal || []).map((v) => ({
    x: b.x + v.x * c - v.y * s,
    y: b.y + v.x * s + v.y * c,
  }));
}

function applyBounceBoundary(b, n, damping = 0.84) {
  if (b.x < 0) { b.x = 0; b.vx = Math.abs(b.vx || 0) * damping; }
  if (b.y < 0) { b.y = 0; b.vy = Math.abs(b.vy || 0) * damping; }
  if (b.x > n - 1) { b.x = n - 1; b.vx = -Math.abs(b.vx || 0) * damping; }
  if (b.y > n - 1) { b.y = n - 1; b.vy = -Math.abs(b.vy || 0) * damping; }
}

function resolveRigidVsRigidPolygonCollision(a, b, slop) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const rr = (a.r || 1) + (b.r || 1) - slop;
  const d2 = dx * dx + dy * dy;
  if (d2 >= rr * rr) return;
  const d = Math.max(1e-6, Math.sqrt(d2));
  const nx = dx / d;
  const ny = dy / d;
  const depth = rr - d;
  const push = depth * 0.5;
  a.x -= nx * push; a.y -= ny * push;
  b.x += nx * push; b.y += ny * push;
  a.vx -= nx * 0.02; a.vy -= ny * 0.02;
  b.vx += nx * 0.02; b.vy += ny * 0.02;
}

function resolveRigidVsSoftNodeCollision(rb, sn, _poly, slop) {
  const dx = sn.x - rb.x;
  const dy = sn.y - rb.y;
  const rr = (rb.r || 1) + (sn.r || 1) - slop;
  const d2 = dx * dx + dy * dy;
  if (d2 >= rr * rr) return;
  const d = Math.max(1e-6, Math.sqrt(d2));
  const nx = dx / d;
  const ny = dy / d;
  const depth = rr - d;
  sn.x += nx * depth;
  sn.y += ny * depth;
}

function resolveRigidVsSoftEdgeCollision(rb, a, b, slop) {
  const mx = (a.x + b.x) * 0.5;
  const my = (a.y + b.y) * 0.5;
  resolveRigidVsSoftNodeCollision(rb, { x: mx, y: my, r: 0.8 }, null, slop);
}

function baselineStepRigid(args) {
  const {
    sim, bodies, vxField, vyField, n, dt, dtNorm, dragK, swimGain,
    localHoneyDrag, viscosityMotionResponse, obstacleMask,
    bodyFeedbackPrevVx, bodyFeedbackPrevVy, selfFeedbackSuppression,
  } = args;
  for (let bi = 0; bi < bodies.rigid.length; bi++) {
    const b = bodies.rigid[bi];
    const arr = Array.isArray(b?.edgeMomentumCoupling) ? b.edgeMomentumCoupling : null;
    let edgeMomentumScale = 1;
    if (arr && arr.length > 0) {
      let sum = 0;
      for (const v of arr) sum += clamp(Number(v) || 0, 0, 1);
      edgeMomentumScale = sum / arr.length;
    }
    const invMass = 1 / Math.max(0.05, b.mass);
    const invInertia = 1 / Math.max(0.05, b.inertia || 1);
    const sampleVerts = rigidVerticesWorld(b);
    let forceX = 0;
    let forceY = 0;
    let torque = 0;
    for (const v of sampleVerts) {
      const rx = v.x - b.x;
      const ry = v.y - b.y;
      const fx = sampleFluidForBodyCoupling(vxField, n, v.x, v.y, rx, ry, obstacleMask, bodyFeedbackPrevVx, selfFeedbackSuppression);
      const fy = sampleFluidForBodyCoupling(vyField, n, v.x, v.y, rx, ry, obstacleMask, bodyFeedbackPrevVy, selfFeedbackSuppression);
      const localVx = b.vx + (-(b.omega || 0) * ry);
      const localVy = b.vy + ((b.omega || 0) * rx);
      const relX = fx - localVx;
      const relY = fy - localVy;
      const honey = localHoneyDrag(v.x, v.y);
      const fpx = relX * dragK * honey * edgeMomentumScale;
      const fpy = relY * dragK * honey * edgeMomentumScale;
      forceX += fpx; forceY += fpy;
      torque += rx * fpy - ry * fpx;
    }
    const sampleCount = Math.max(1, sampleVerts.length);
    forceX /= sampleCount;
    forceY /= sampleCount;
    torque /= sampleCount;

    const ax = forceX * invMass;
    const ay = forceY * invMass;
    const alpha = torque * invInertia;
    const swimPhase = sim.frame * 0.08 + bi * 2.1;
    const swimX = swimGain * Math.cos(swimPhase) * 0.012 * invMass;
    const swimY = swimGain * Math.sin(swimPhase * 1.6) * 0.009 * invMass;
    const swimTorque = swimGain * Math.sin(swimPhase * 1.1) * 0.0025;

    b.vx += ax * dt * 60 + swimX * dtNorm;
    b.vy += ay * dt * 60 + swimY * dtNorm;
    b.omega = (b.omega || 0) + alpha * dt * 60 + swimTorque * dtNorm;

    const centerHoney = localHoneyDrag(b.x, b.y);
    const rigidVisc = viscosityMotionResponse(centerHoney, 3.2);
    b.vx *= rigidVisc.damp;
    b.vy *= rigidVisc.damp;
    b.omega *= Math.max(0.72, 0.99 - 0.01 * centerHoney);

    const bMax = rigidVisc.vmax;
    const bMag = Math.hypot(b.vx, b.vy);
    if (bMag > bMax) { b.vx = (b.vx / bMag) * bMax; b.vy = (b.vy / bMag) * bMax; }
    b.omega = Math.max(-0.25, Math.min(0.25, b.omega));

    b.x += b.vx * dt * 22;
    b.y += b.vy * dt * 28;
    b.theta = (b.theta || 0) + b.omega * dt * 60;
    applyBounceBoundary(b, n, 0.84);
  }
}

function baselineRigidRigidPass(rigidBodies) {
  for (let i = 0; i < rigidBodies.length; i++) {
    for (let j = i + 1; j < rigidBodies.length; j++) {
      resolveRigidVsRigidPolygonCollision(rigidBodies[i], rigidBodies[j], 0.32);
    }
  }
}

function baselineRigidSoftPass(rigidBodies, soft, {
  nodeObserver = resolveRigidVsSoftNodeCollision,
  edgeObserver = resolveRigidVsSoftEdgeCollision,
} = {}) {
  for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
    const rb = rigidBodies[rbi];
    for (let ni = 0; ni < soft.nodes.length; ni++) {
      nodeObserver(rb, soft.nodes[ni], null, 0.18);
    }
    for (const [i, j, _rest, edgeBodyMode] of soft.springs) {
      if (edgeBodyMode !== 1) continue;
      edgeObserver(rb, soft.nodes[i], soft.nodes[j], 0.16);
    }
  }
}

const STAGE_KEYS = ['rigidIntegrate', 'rigidRigidCollision', 'rigidSoftCollision'];

function parseBooleanEnv(name, fallback = false) {
  const raw = String(process.env[name] || '').trim().toLowerCase();
  if (!raw) return fallback;
  return raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on';
}

const detailedRigidSoft = parseBooleanEnv('RIGID_SOFT_DETAILED', false);
const useRigidSoftOffloadStub = parseBooleanEnv('RIGID_SOFT_USE_OFFLOAD_STUB', false);
const rigidSoftCpuSolverMode = String(process.env.RIGID_SOFT_CPU_SOLVER || 'benchmark').trim().toLowerCase();

function createStageStats() {
  const stats = {};
  for (const key of STAGE_KEYS) {
    stats[key] = {
      calls: 0,
      wallMs: 0,
      cpuMs: 0,
      gpuWaitMsEstimate: 0,
    };
  }
  if (detailedRigidSoft) {
    stats.rigidSoftDetail = {
      nodeCalls: 0,
      edgeCalls: 0,
      cpuFallbackNodeWallMs: 0,
      cpuFallbackEdgeWallMs: 0,
      cpuFallbackNodePairCount: 0,
      cpuFallbackEdgePairCount: 0,
      cpuFallbackUsedCompactNodePairsCount: 0,
      cpuFallbackUsedCompactEdgePairsCount: 0,
      routeCounts: {},
      fallbackReasonCounts: {},
      mode: rigidSoftCpuSolverMode,
      useOffloadStub: useRigidSoftOffloadStub,
    };
  }
  return stats;
}

function runTimedStage(stageStats, stageKey, fn) {
  if (!stageStats) return fn();
  const entry = stageStats[stageKey];
  const usageStart = process.cpuUsage();
  const t0 = performance.now();
  const result = fn();
  const wallMs = performance.now() - t0;
  const usage = process.cpuUsage(usageStart);
  const cpuMs = (usage.user + usage.system) / 1000;
  entry.calls += 1;
  entry.wallMs += wallMs;
  entry.cpuMs += cpuMs;
  entry.gpuWaitMsEstimate += Math.max(0, wallMs - cpuMs);
  return result;
}

function withRigidSoftObserver(detail, kind, fn) {
  if (!detail || typeof fn !== 'function') return fn;
  if (kind !== 'node' && kind !== 'edge') return fn;
  return (...args) => {
    if (kind === 'node') detail.nodeCalls += 1;
    else detail.edgeCalls += 1;
    return fn(...args);
  };
}

function finalizeStageStats(stageStats, steps) {
  const out = {};
  for (const key of STAGE_KEYS) {
    const s = stageStats[key];
    out[key] = {
      calls: s.calls,
      wallMs: s.wallMs,
      wallMsPerStep: s.wallMs / Math.max(1, steps),
      cpuMs: s.cpuMs,
      cpuMsPerStep: s.cpuMs / Math.max(1, steps),
      gpuWaitMsEstimate: s.gpuWaitMsEstimate,
      gpuWaitMsEstimatePerStep: s.gpuWaitMsEstimate / Math.max(1, steps),
    };
  }

  if (stageStats.rigidSoftDetail) {
    const detail = stageStats.rigidSoftDetail;
    const totalFallbackWallMs = detail.cpuFallbackNodeWallMs + detail.cpuFallbackEdgeWallMs;
    const stageWallMs = out.rigidSoftCollision.wallMs;
    const nonFallbackWallMs = Math.max(0, stageWallMs - totalFallbackWallMs);
    out.rigidSoftCollision.breakdown = {
      mode: detail.mode,
      useOffloadStub: detail.useOffloadStub,
      nodeCalls: detail.nodeCalls,
      edgeCalls: detail.edgeCalls,
      cpuFallbackNodeWallMs: detail.cpuFallbackNodeWallMs,
      cpuFallbackEdgeWallMs: detail.cpuFallbackEdgeWallMs,
      cpuFallbackNodeWallMsPerStep: detail.cpuFallbackNodeWallMs / Math.max(1, steps),
      cpuFallbackEdgeWallMsPerStep: detail.cpuFallbackEdgeWallMs / Math.max(1, steps),
      cpuFallbackNodePairCount: detail.cpuFallbackNodePairCount,
      cpuFallbackEdgePairCount: detail.cpuFallbackEdgePairCount,
      cpuFallbackUsedCompactNodePairsCount: detail.cpuFallbackUsedCompactNodePairsCount,
      cpuFallbackUsedCompactEdgePairsCount: detail.cpuFallbackUsedCompactEdgePairsCount,
      nonFallbackWallMs,
      nonFallbackWallMsPerStep: nonFallbackWallMs / Math.max(1, steps),
      routeCounts: detail.routeCounts,
      fallbackReasonCounts: detail.fallbackReasonCounts,
    };
  }

  return out;
}

function buildStepArgs(state) {
  return {
    sim: state.sim,
    bodies: state.bodies,
    vxField: state.vxField,
    vyField: state.vyField,
    n: state.n,
    dt: 0.012,
    dtNorm: 0.82,
    dragK: 1.1,
    swimGain: 0.55,
    localHoneyDrag: (x, y) => 1 + ((Math.sin(x * 0.014) + Math.cos(y * 0.017)) * 0.5 + 0.5) * 2.2,
    viscosityMotionResponse: (honey, vmaxBase) => ({
      damp: Math.max(0.72, 1 - 0.018 * honey),
      vmax: Math.max(0.4, vmaxBase / (1 + 0.28 * honey)),
    }),
    obstacleMask: state.obstacleMask,
    bodyFeedbackPrevVx: state.bodyFeedbackPrevVx,
    bodyFeedbackPrevVy: state.bodyFeedbackPrevVy,
    selfFeedbackSuppression: 0.82,
    rigidVerticesWorld,
    sampleFluidForBodyCoupling,
    applyBounceBoundary,
  };
}

function runStep(path, state, args, stageStats = null) {
  const rigidSoftDetail = stageStats?.rigidSoftDetail || null;
  const nodeObserver = withRigidSoftObserver(rigidSoftDetail, 'node', resolveRigidVsSoftNodeCollision);
  const edgeObserver = withRigidSoftObserver(rigidSoftDetail, 'edge', resolveRigidVsSoftEdgeCollision);

  if (path === 'gpu-only') {
    runTimedStage(stageStats, 'rigidIntegrate', () => stepRigidBodiesGpuOnly(args));
    runTimedStage(stageStats, 'rigidRigidCollision', () => resolveRigidRigidCollisionPassGpuOnly({
      rigidBodies: state.bodies.rigid,
      slop: 0.32,
      contacts: null,
      iter: 0,
      phase: 'bench',
      resolveRigidVsRigidPolygonCollision,
    }));

    const passArgs = {
      rigidBodies: state.bodies.rigid,
      soft: state.bodies.soft,
      edgeBodyModeBlock: 1,
      nodeSlop: 0.18,
      edgeSlop: 0.16,
    };

    if (rigidSoftCpuSolverMode !== 'internal') {
      passArgs.resolveRigidVsSoftNodeCollision = nodeObserver;
      passArgs.resolveRigidVsSoftEdgeCollision = edgeObserver;
    }

    if (useRigidSoftOffloadStub) {
      const offloadState = stageStats ? (stageStats.rigidSoftOffloadState ||= {}) : {};
      passArgs.wgslOffload = { enabled: true, state: offloadState };
    }

    const rigidSoftProfile = rigidSoftDetail ? {} : null;
    if (rigidSoftProfile) passArgs.profile = rigidSoftProfile;

    runTimedStage(stageStats, 'rigidSoftCollision', () => resolveRigidSoftCollisionPassGpuOnly(passArgs));

    if (rigidSoftProfile) {
      rigidSoftDetail.cpuFallbackNodeWallMs += Number(rigidSoftProfile.cpuFallbackNodeWallMs) || 0;
      rigidSoftDetail.cpuFallbackEdgeWallMs += Number(rigidSoftProfile.cpuFallbackEdgeWallMs) || 0;
      rigidSoftDetail.cpuFallbackNodePairCount += Number(rigidSoftProfile.cpuFallbackNodePairCount) || 0;
      rigidSoftDetail.cpuFallbackEdgePairCount += Number(rigidSoftProfile.cpuFallbackEdgePairCount) || 0;
      if (rigidSoftProfile.cpuFallbackUsedCompactNodePairs) rigidSoftDetail.cpuFallbackUsedCompactNodePairsCount += 1;
      if (rigidSoftProfile.cpuFallbackUsedCompactEdgePairs) rigidSoftDetail.cpuFallbackUsedCompactEdgePairsCount += 1;
      const routeKey = String(rigidSoftProfile.route || 'unknown');
      rigidSoftDetail.routeCounts[routeKey] = (rigidSoftDetail.routeCounts[routeKey] || 0) + 1;
      const reasonKey = String(rigidSoftProfile.fallbackReason || 'none');
      rigidSoftDetail.fallbackReasonCounts[reasonKey] = (rigidSoftDetail.fallbackReasonCounts[reasonKey] || 0) + 1;
    }
  } else {
    runTimedStage(stageStats, 'rigidIntegrate', () => baselineStepRigid(args));
    runTimedStage(stageStats, 'rigidRigidCollision', () => baselineRigidRigidPass(state.bodies.rigid));
    runTimedStage(stageStats, 'rigidSoftCollision', () => baselineRigidSoftPass(state.bodies.rigid, state.bodies.soft, {
      nodeObserver,
      edgeObserver,
    }));
  }
  state.sim.frame += 1;
}

function runScenario(path, fixture, { steps, warmupSteps }) {
  const warmupState = structuredClone(fixture);
  const warmupArgs = buildStepArgs(warmupState);
  for (let i = 0; i < warmupSteps; i++) runStep(path, warmupState, warmupArgs, null);

  const state = structuredClone(fixture);
  const args = buildStepArgs(state);
  const stageStats = createStageStats();

  const t0 = performance.now();
  for (let i = 0; i < steps; i++) runStep(path, state, args, stageStats);
  const elapsedMs = performance.now() - t0;

  const stageTimings = finalizeStageStats(stageStats, steps);
  const stageWallSum = STAGE_KEYS.reduce((sum, key) => sum + stageTimings[key].wallMs, 0);
  const stageCpuSum = STAGE_KEYS.reduce((sum, key) => sum + stageTimings[key].cpuMs, 0);
  const stageGpuWaitSum = STAGE_KEYS.reduce((sum, key) => sum + stageTimings[key].gpuWaitMsEstimate, 0);

  const result = {
    elapsedMs,
    msPerStep: elapsedMs / steps,
    stepsPerSec: (steps / elapsedMs) * 1000,
    fpsEquivalent: (steps / elapsedMs) * 1000,
    steps,
    warmupSteps,
    stageTimings,
    stageTotals: {
      wallMs: stageWallSum,
      cpuMs: stageCpuSum,
      gpuWaitMsEstimate: stageGpuWaitSum,
      nonStageWallMs: Math.max(0, elapsedMs - stageWallSum),
    },
  };

  if (stageStats.rigidSoftOffloadState) {
    const st = stageStats.rigidSoftOffloadState;
    result.rigidSoftRuntime = {
      lastSourceRoute: st.lastSourceRoute || null,
      lastMode: st.lastMode || null,
      lastRigidSoftResponseOwnership: st.lastRigidSoftResponseOwnership || null,
      lastRigidSoftResponseRoute: st.lastRigidSoftResponseRoute || null,
      lastRigidSoftResponseFallbackReason: st.lastRigidSoftResponseFallbackReason || null,
      lastNodeCollisionResponseSource: st.lastNodeCollisionResponseSource || null,
      lastEdgeCollisionResponseSource: st.lastEdgeCollisionResponseSource || null,
      lastPreparedNodePairCount: Number(st.lastPreparedNodePairCount) || 0,
      lastPreparedEdgePairCount: Number(st.lastPreparedEdgePairCount) || 0,
      lastPreparedNodeNarrowphasePairCount: Number(st.lastPreparedNodeNarrowphasePairCount) || 0,
      lastPreparedEdgeNarrowphasePairCount: Number(st.lastPreparedEdgeNarrowphasePairCount) || 0,
    };
  }

  return result;
}

function buildExperiments({
  scales,
  rigidSweep,
  softSweep,
  contactSweep,
  baseRigidCount,
  baseSoftCount,
  sweeps,
}) {
  const experiments = [];

  if (sweeps.includes('scale')) {
    for (const scale of scales) {
      experiments.push({
        sweep: 'scale',
        id: `scale:${scale}`,
        rigidCount: scale,
        softNodeCount: Math.max(24, Math.floor(scale * 0.35)),
        contactDensity: 0.55,
      });
    }
  }

  if (sweeps.includes('rigid')) {
    for (const rigidCount of rigidSweep) {
      experiments.push({
        sweep: 'rigid',
        id: `rigid:${rigidCount}`,
        rigidCount,
        softNodeCount: baseSoftCount,
        contactDensity: 0.55,
      });
    }
  }

  if (sweeps.includes('soft')) {
    for (const softNodeCount of softSweep) {
      experiments.push({
        sweep: 'soft',
        id: `soft:${softNodeCount}`,
        rigidCount: baseRigidCount,
        softNodeCount,
        contactDensity: 0.55,
      });
    }
  }

  if (sweeps.includes('contact')) {
    for (const density of contactSweep) {
      const d = clamp(density, 0.05, 1.0);
      experiments.push({
        sweep: 'contact',
        id: `contact:${d.toFixed(2)}`,
        rigidCount: baseRigidCount,
        softNodeCount: baseSoftCount,
        contactDensity: d,
      });
    }
  }

  return experiments;
}

function stageDeltaSummary(baseline, gpuOnly) {
  const delta = {};
  for (const key of STAGE_KEYS) {
    const b = baseline.stageTimings[key];
    const g = gpuOnly.stageTimings[key];
    delta[key] = {
      wallMsPerStepDelta: g.wallMsPerStep - b.wallMsPerStep,
      wallMsPerStepDeltaPct: b.wallMsPerStep > 0 ? ((g.wallMsPerStep - b.wallMsPerStep) / b.wallMsPerStep) * 100 : null,
      cpuMsPerStepDelta: g.cpuMsPerStep - b.cpuMsPerStep,
      gpuWaitMsEstimatePerStepDelta: g.gpuWaitMsEstimatePerStep - b.gpuWaitMsEstimatePerStep,
    };
  }
  return delta;
}

function parseEnvNumber(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === null || raw === '') return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

const scales = parseNumberList(process.env.SCALES, [25, 50, 100, 200, 400]);
const rigidSweep = parseNumberList(process.env.RIGID_SWEEP, scales);
const softSweep = parseNumberList(process.env.SOFT_SWEEP, [24, 48, 96, 192, 384]);
const contactSweep = parseFloatList(process.env.CONTACT_SWEEP, [0.15, 0.35, 0.55, 0.75, 0.92]);
const sweeps = parseSweepList(process.env.SWEEPS);
const fixedSteps = Math.max(4, parseEnvNumber('FIXED_STEPS', 64));
const warmupSteps = Math.max(0, parseEnvNumber('WARMUP_STEPS', 8));
const baseRigidCount = Math.max(1, parseEnvNumber('BASE_RIGID_COUNT', 100));
const baseSoftCount = Math.max(0, parseEnvNumber('BASE_SOFT_COUNT', 64));

const experiments = buildExperiments({
  scales,
  rigidSweep,
  softSweep,
  contactSweep,
  baseRigidCount,
  baseSoftCount,
  sweeps,
});

const rows = [];
for (const exp of experiments) {
  const fixture = makeFixture({
    rigidCount: exp.rigidCount,
    softNodeCount: exp.softNodeCount,
    contactDensity: exp.contactDensity,
    scenarioKey: exp.id,
  });

  const baseline = runScenario('baseline', fixture, { steps: fixedSteps, warmupSteps });
  const gpuOnly = runScenario('gpu-only', fixture, { steps: fixedSteps, warmupSteps });

  rows.push({
    ...exp,
    steps: fixedSteps,
    warmupSteps,
    fixture: fixture.fixtureMeta,
    baseline,
    gpuOnly,
    deltaMsPerStepPct: ((gpuOnly.msPerStep - baseline.msPerStep) / baseline.msPerStep) * 100,
    deltaStepsPerSecPct: ((gpuOnly.stepsPerSec - baseline.stepsPerSec) / baseline.stepsPerSec) * 100,
    stageDelta: stageDeltaSummary(baseline, gpuOnly),
  });
}

const scaleRows = rows
  .filter((r) => r.sweep === 'scale')
  .map((r) => ({
    scale: r.rigidCount,
    steps: r.steps,
    baseline: r.baseline,
    gpuOnly: r.gpuOnly,
    deltaMsPerStepPct: r.deltaMsPerStepPct,
    deltaStepsPerSecPct: r.deltaStepsPerSecPct,
    stageDelta: r.stageDelta,
  }));

const output = {
  meta: {
    mode: 'scenario-sweep-with-fixed-steps',
    timing: {
      perStageCpu: true,
      perStageGpuWaitEstimate: true,
      method: 'process.cpuUsage + wall-clock split',
      note: 'gpuWaitMsEstimate approximates non-CPU stage wall time; this harness does not use WebGPU timestamp queries yet.',
    },
    steps: {
      fixedSteps,
      warmupSteps,
    },
    sweeps,
  },
  scales: scaleRows,
  experiments: rows,
};

console.log(JSON.stringify(output, null, 2));
