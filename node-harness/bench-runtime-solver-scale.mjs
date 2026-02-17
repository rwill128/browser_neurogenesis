import { stepRigidBodiesGpuOnly } from '../sim-server/public/runtime-solvers/stepRigidGpuOnly.js';
import { resolveRigidRigidCollisionPassGpuOnly } from '../sim-server/public/runtime-solvers/stepRigidCollisionGpuOnly.js';
import { resolveRigidSoftCollisionPassGpuOnly } from '../sim-server/public/runtime-solvers/stepRigidSoftCollisionGpuOnly.js';

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function makeFixture(scale) {
  const rand = mulberry32(0xC0FFEE ^ scale);
  const n = 256;
  const cells = n * n;
  const rigid = [];
  for (let i = 0; i < scale; i++) {
    const mass = 1 + rand() * 4;
    const r = 1.5 + rand() * 2;
    rigid.push({
      x: 12 + rand() * (n - 24),
      y: 12 + rand() * (n - 24),
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
    nodes: Array.from({ length: Math.max(24, Math.floor(scale * 0.35)) }, (_, i) => ({
      id: i,
      x: 10 + (i % 32) * 6.2,
      y: 16 + Math.floor(i / 32) * 6.1,
      vx: 0,
      vy: 0,
      r: 1.2,
      clusterId: i % 8,
    })),
    springs: [],
  };
  for (let i = 0; i < soft.nodes.length - 1; i++) {
    soft.springs.push([i, i + 1, 1, (i % 3 === 0 ? 1 : 0)]);
  }

  const hybridAttachedByRigid = new Map();
  for (let i = 0; i < rigid.length; i += 7) {
    const ni = i % soft.nodes.length;
    hybridAttachedByRigid.set(i, new Set([ni]));
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
    hybridAttachedByRigid,
    vxField,
    vyField,
    obstacleMask: new Float32Array(cells),
    bodyFeedbackPrevVx: new Float32Array(cells),
    bodyFeedbackPrevVy: new Float32Array(cells),
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

function baselineRigidSoftPass(rigidBodies, soft, hybridAttachedByRigid) {
  for (let rbi = 0; rbi < rigidBodies.length; rbi++) {
    const rb = rigidBodies[rbi];
    const attachedNodeSet = hybridAttachedByRigid.get(rbi) || null;
    for (let ni = 0; ni < soft.nodes.length; ni++) {
      if (attachedNodeSet && attachedNodeSet.has(ni)) continue;
      resolveRigidVsSoftNodeCollision(rb, soft.nodes[ni], null, 0.18);
    }
    for (const [i, j, _rest, edgeBodyMode] of soft.springs) {
      if (edgeBodyMode !== 1) continue;
      if (attachedNodeSet && (attachedNodeSet.has(i) || attachedNodeSet.has(j))) continue;
      resolveRigidVsSoftEdgeCollision(rb, soft.nodes[i], soft.nodes[j], 0.16);
    }
  }
}

function runScenario(path, fixture, steps) {
  const state = structuredClone(fixture);
  const t0 = performance.now();
  for (let step = 0; step < steps; step++) {
    const args = {
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
      viscosityMotionResponse: (honey, vmaxBase) => ({ damp: Math.max(0.72, 1 - 0.018 * honey), vmax: Math.max(0.4, vmaxBase / (1 + 0.28 * honey)) }),
      obstacleMask: state.obstacleMask,
      bodyFeedbackPrevVx: state.bodyFeedbackPrevVx,
      bodyFeedbackPrevVy: state.bodyFeedbackPrevVy,
      selfFeedbackSuppression: 0.82,
      rigidVerticesWorld,
      sampleFluidForBodyCoupling,
      applyBounceBoundary,
    };

    if (path === 'gpu-only') {
      stepRigidBodiesGpuOnly(args);
      resolveRigidRigidCollisionPassGpuOnly({
        rigidBodies: state.bodies.rigid,
        slop: 0.32,
        contacts: null,
        iter: 0,
        phase: 'bench',
        resolveRigidVsRigidPolygonCollision,
      });
      resolveRigidSoftCollisionPassGpuOnly({
        rigidBodies: state.bodies.rigid,
        soft: state.bodies.soft,
        hybridAttachedByRigid: state.hybridAttachedByRigid,
        resolveRigidVsSoftNodeCollision,
        resolveRigidVsSoftEdgeCollision,
        edgeBodyModeBlock: 1,
        nodeSlop: 0.18,
        edgeSlop: 0.16,
      });
    } else {
      baselineStepRigid(args);
      baselineRigidRigidPass(state.bodies.rigid);
      baselineRigidSoftPass(state.bodies.rigid, state.bodies.soft, state.hybridAttachedByRigid);
    }

    state.sim.frame += 1;
  }
  const elapsedMs = performance.now() - t0;
  return {
    elapsedMs,
    msPerStep: elapsedMs / steps,
    stepsPerSec: (steps / elapsedMs) * 1000,
    fpsEquivalent: (steps / elapsedMs) * 1000,
  };
}

const scales = (process.env.SCALES || '25,50,100,200,400').split(',').map((v) => Number(v.trim())).filter((v) => Number.isFinite(v) && v > 0);
const rows = [];

for (const scale of scales) {
  const fixture = makeFixture(scale);
  const steps = Math.max(8, Math.round(2200 / scale));

  runScenario('baseline', fixture, Math.min(5, steps));
  runScenario('gpu-only', fixture, Math.min(5, steps));

  const baseline = runScenario('baseline', fixture, steps);
  const gpuOnly = runScenario('gpu-only', fixture, steps);
  rows.push({
    scale,
    steps,
    baseline,
    gpuOnly,
    deltaMsPerStepPct: ((gpuOnly.msPerStep - baseline.msPerStep) / baseline.msPerStep) * 100,
    deltaStepsPerSecPct: ((gpuOnly.stepsPerSec - baseline.stepsPerSec) / baseline.stepsPerSec) * 100,
  });
}

console.log(JSON.stringify({ scales: rows }, null, 2));
