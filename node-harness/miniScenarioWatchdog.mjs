import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  rigidVerticesWorld,
  resolveRigidVsRigidPolygonCollision,
  resolveRigidVsSoftNodeCollision,
  pointInPolygonInclusive,
} from '../sim-server/public/rigid-collision.js';
import { EDGE_DYE_MODE, EDGE_BODY_MODE } from '../sim-server/public/dye-barrier.js';
import { GPUFluidField } from '../js/gpuFluidField.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const CATALOG_PATH = path.join(REPO_ROOT, 'sim-server/public/generated-mini-scenarios.json');
const REPORT_DIR = path.join(REPO_ROOT, 'node-harness/reports');
const REPORT_LATEST = path.join(REPORT_DIR, 'mini-watchdog-latest.json');
const REPORT_HISTORY = path.join(REPORT_DIR, 'mini-watchdog-history.ndjson');

const GRID = 100;
const DT = 0.03;
const INTEGRATION_SCALE = 24;
const SPRING_ITERS = 10;
const SPRING_BASE_COMPLIANCE = 0.0012;
const AREA_ITERS = 6;
const AREA_BASE_COMPLIANCE = 0.0009;
const MAX_SCENARIOS = 24;

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function randRange(rng, lo, hi) {
  return lo + (hi - lo) * rng();
}

function randInt(rng, lo, hiInclusive) {
  return lo + Math.floor(rng() * (hiInclusive - lo + 1));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function clone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function makeShadowOnlyFluid({ size = GRID, dt = DT, scaleX = 1, scaleY = 1 } = {}) {
  const f = Object.create(GPUFluidField.prototype);
  f.size = size;
  f.dt = dt;
  f.scaleX = scaleX;
  f.scaleY = scaleY;
  f.maxVelComponent = 8;
  const cells = size * size;
  f.shadowVx = new Float32Array(cells).fill(0);
  f.shadowVy = new Float32Array(cells).fill(0);
  f.shadowDensityR = new Float32Array(cells).fill(0);
  f.shadowDensityG = new Float32Array(cells).fill(0);
  f.shadowDensityB = new Float32Array(cells).fill(0);
  f.shadowVxNext = new Float32Array(cells).fill(0);
  f.shadowVyNext = new Float32Array(cells).fill(0);
  f.shadowDensityRNext = new Float32Array(cells).fill(0);
  f.shadowDensityGNext = new Float32Array(cells).fill(0);
  f.shadowDensityBNext = new Float32Array(cells).fill(0);
  f._initShadowBackCompatViews();
  f.gpuEnabled = false;
  return f;
}

function edgeModesForSides(sides) {
  return {
    edgeDyeMode: Array.from({ length: sides }, () => [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT]),
    edgeBodyMode: Array.from({ length: sides }, () => EDGE_BODY_MODE.BLOCK),
    edgePermeabilityRGB: Array.from({ length: sides }, () => [0, 0, 0]),
  };
}

function makeRigid(rng, slot = 0) {
  const sidesChoices = [3, 4, 5, 6];
  const sides = sidesChoices[randInt(rng, 0, sidesChoices.length - 1)];
  const r = randRange(rng, 6.2, 9.4);
  const x = slot === 0 ? randRange(rng, 24, 36) : randRange(rng, 64, 76);
  const y = randRange(rng, 34, 66);
  const mass = randRange(rng, 1.2, 3.8);
  const modes = edgeModesForSides(sides);
  return {
    x,
    y,
    vx: randRange(rng, -0.12, 0.12),
    vy: randRange(rng, -0.12, 0.12),
    r,
    sides,
    theta: randRange(rng, 0, Math.PI * 2),
    omega: randRange(rng, -0.015, 0.015),
    mass,
    inertia: 0.5 * mass * r * r,
    digestEnabled: false,
    digestRGB: [0, 0, 0],
    consumeDyeRGB: [0, 0, 0],
    ...modes,
  };
}

function makeSoftCluster(rng, clusterId = 0, slot = 0) {
  const count = 6;
  const radius = randRange(rng, 7.5, 10.5);
  const cx = slot === 0 ? randRange(rng, 25, 37) : randRange(rng, 63, 75);
  const cy = randRange(rng, 34, 66);
  const nodes = [];
  const springs = [];
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    nodes.push({
      x: cx + Math.cos(a) * radius,
      y: cy + Math.sin(a) * radius,
      vx: randRange(rng, -0.08, 0.08),
      vy: randRange(rng, -0.08, 0.08),
      mass: randRange(rng, 0.45, 0.95),
      r: 1.2,
      clusterId,
      digestEnabled: false,
      digestRGB: [0, 0, 0],
    });
  }

  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count;
    const a = nodes[i];
    const b = nodes[j];
    springs.push([i, j, Math.hypot(b.x - a.x, b.y - a.y), EDGE_BODY_MODE.BLOCK, [EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT, EDGE_DYE_MODE.DEFLECT]]);
  }
  for (let i = 0; i < count; i++) {
    const j = (i + 2) % count;
    if (i < j) {
      const a = nodes[i];
      const b = nodes[j];
      springs.push([i, j, Math.hypot(b.x - a.x, b.y - a.y), EDGE_BODY_MODE.PASS, [EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS, EDGE_DYE_MODE.PASS]]);
    }
  }

  return { nodes, springs, center: { x: cx, y: cy }, radius };
}

function makeScenario(seedBase) {
  const rng = mulberry32(seedBase >>> 0);
  const combos = ['rigid-rigid', 'soft-soft', 'soft-rigid'];
  const combo = combos[seedBase % combos.length];
  const steps = 50 + (seedBase % 51);
  const createdAt = new Date().toISOString();
  const id = `mini-${Math.floor(seedBase % 1_000_000)}-${combo}`;

  const bodies = { rigid: [], soft: { nodes: [], springs: [] }, hybrid: [], rigidWelds: [] };

  if (combo === 'rigid-rigid') {
    bodies.rigid.push(makeRigid(rng, 0));
    bodies.rigid.push(makeRigid(rng, 1));
  } else if (combo === 'soft-soft') {
    const a = makeSoftCluster(rng, 0, 0);
    const b = makeSoftCluster(rng, 1, 1);
    bodies.soft.nodes.push(...a.nodes, ...b.nodes);
    bodies.soft.springs.push(...a.springs);
    const offset = a.nodes.length;
    bodies.soft.springs.push(...b.springs.map((s) => [s[0] + offset, s[1] + offset, s[2], s[3], s[4]]));
  } else {
    bodies.rigid.push(makeRigid(rng, 0));
    const s = makeSoftCluster(rng, 0, 1);
    bodies.soft.nodes.push(...s.nodes);
    bodies.soft.springs.push(...s.springs);
  }

  return {
    id,
    name: `Mini ${combo}`,
    createdAt,
    seed: seedBase >>> 0,
    grid: GRID,
    dt: DT,
    steps,
    combo,
    fluid: {
      size: GRID,
      dt: DT,
    },
    emitters: [],
    bodies,
  };
}

function applyBounceBoundary(body, n, damping = 0.84) {
  const r = Math.max(0.4, body.r || 0.8);
  const minX = r;
  const maxX = n - r;
  const minY = r;
  const maxY = n - r;

  if (body.x < minX) {
    body.x = minX;
    if (body.vx < 0) body.vx = -body.vx * damping;
  } else if (body.x > maxX) {
    body.x = maxX;
    if (body.vx > 0) body.vx = -body.vx * damping;
  }

  if (body.y < minY) {
    body.y = minY;
    if (body.vy < 0) body.vy = -body.vy * damping;
  } else if (body.y > maxY) {
    body.y = maxY;
    if (body.vy > 0) body.vy = -body.vy * damping;
  }
}

function resolveSoftNodeVsSoftNode(a, b, restitution = 0.1) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const d2 = dx * dx + dy * dy;
  const minDist = (a.r || 1) + (b.r || 1);
  if (d2 <= 1e-10 || d2 >= minDist * minDist) return false;

  const d = Math.sqrt(d2);
  const nx = dx / d;
  const ny = dy / d;
  const penetration = minDist - d;

  const ma = Math.max(0.02, a.mass || 1);
  const mb = Math.max(0.02, b.mass || 1);
  const invA = 1 / ma;
  const invB = 1 / mb;
  const invSum = invA + invB;

  const corr = (penetration / Math.max(1e-6, invSum)) * 0.85;
  a.x -= nx * corr * invA;
  a.y -= ny * corr * invA;
  b.x += nx * corr * invB;
  b.y += ny * corr * invB;

  const rvx = b.vx - a.vx;
  const rvy = b.vy - a.vy;
  const vn = rvx * nx + rvy * ny;
  if (vn > 0) return true;
  const j = (-(1 + restitution) * vn) / Math.max(1e-6, invSum);
  const ix = j * nx;
  const iy = j * ny;
  a.vx -= ix * invA;
  a.vy -= iy * invA;
  b.vx += ix * invB;
  b.vy += iy * invB;
  return true;
}

function clusterLoops(nodes) {
  const by = new Map();
  for (let i = 0; i < nodes.length; i++) {
    const cid = nodes[i].clusterId ?? 0;
    if (!by.has(cid)) by.set(cid, []);
    by.get(cid).push(i);
  }
  const loops = [];
  for (const [cid, ids] of by.entries()) {
    if (ids.length < 3) continue;
    let cx = 0;
    let cy = 0;
    for (const i of ids) {
      cx += nodes[i].x;
      cy += nodes[i].y;
    }
    cx /= ids.length;
    cy /= ids.length;
    const ordered = [...ids].sort((a, b) => Math.atan2(nodes[a].y - cy, nodes[a].x - cx) - Math.atan2(nodes[b].y - cy, nodes[b].x - cx));
    loops.push({ clusterId: cid, ids: ordered });
  }
  return loops;
}

function signedAreaPredicted(nodes, ids, dtPos) {
  let s = 0;
  for (let i = 0; i < ids.length; i++) {
    const a = nodes[ids[i]];
    const b = nodes[ids[(i + 1) % ids.length]];
    const ax = a.x + a.vx * dtPos;
    const ay = a.y + a.vy * dtPos;
    const bx = b.x + b.vx * dtPos;
    const by = b.y + b.vy * dtPos;
    s += ax * by - bx * ay;
  }
  return 0.5 * s;
}

function applyXPBDSprings(nodes, springs, dtPos, lambda, stiffness = 3.0) {
  if (!springs.length) return;
  const alpha = (SPRING_BASE_COMPLIANCE / Math.max(0.2, stiffness)) / Math.max(1e-8, dtPos * dtPos);
  for (let iter = 0; iter < SPRING_ITERS; iter++) {
    for (let si = 0; si < springs.length; si++) {
      const [ia, ib, rest] = springs[si];
      const a = nodes[ia];
      const b = nodes[ib];
      const ax = a.x + a.vx * dtPos;
      const ay = a.y + a.vy * dtPos;
      const bx = b.x + b.vx * dtPos;
      const by = b.y + b.vy * dtPos;
      const dx = bx - ax;
      const dy = by - ay;
      const d = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = dx / d;
      const ny = dy / d;
      const C = d - rest;
      const wA = 1 / Math.max(0.02, a.mass || 1);
      const wB = 1 / Math.max(0.02, b.mass || 1);
      const wSum = wA + wB;
      if (wSum <= 1e-9) continue;
      const lp = lambda[si] || 0;
      const dl = (-C - alpha * lp) / (wSum + alpha);
      lambda[si] = lp + dl;
      a.vx += (-wA * dl * nx) / dtPos;
      a.vy += (-wA * dl * ny) / dtPos;
      b.vx += (wB * dl * nx) / dtPos;
      b.vy += (wB * dl * ny) / dtPos;
    }
  }
}

function applyXPBDArea(nodes, loops, dtPos, restMap, lambdaMap, stiffness = 3.0) {
  if (!loops.length) return;
  const alpha = (AREA_BASE_COMPLIANCE / Math.max(0.2, stiffness)) / Math.max(1e-8, dtPos * dtPos);
  for (const loop of loops) {
    if (!restMap.has(loop.clusterId)) {
      const area0 = signedAreaPredicted(nodes, loop.ids, dtPos);
      restMap.set(loop.clusterId, Math.abs(area0) > 1e-4 ? area0 : 1e-4);
    }
    lambdaMap.set(loop.clusterId, 0);
  }

  for (let iter = 0; iter < AREA_ITERS; iter++) {
    for (const loop of loops) {
      const ids = loop.ids;
      const m = ids.length;
      const restArea = restMap.get(loop.clusterId);
      const area = signedAreaPredicted(nodes, ids, dtPos);
      const C = area - restArea;
      const gx = new Array(m);
      const gy = new Array(m);
      let sum = 0;
      for (let k = 0; k < m; k++) {
        const p = nodes[ids[(k - 1 + m) % m]];
        const n = nodes[ids[(k + 1) % m]];
        const px = p.x + p.vx * dtPos;
        const py = p.y + p.vy * dtPos;
        const nx = n.x + n.vx * dtPos;
        const ny = n.y + n.vy * dtPos;
        gx[k] = 0.5 * (ny - py);
        gy[k] = 0.5 * (px - nx);
        const node = nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        sum += w * (gx[k] * gx[k] + gy[k] * gy[k]);
      }
      if (sum <= 1e-10) continue;
      const lp = lambdaMap.get(loop.clusterId) || 0;
      const dl = clamp((-C - alpha * lp) / (sum + alpha), -2.0, 2.0);
      lambdaMap.set(loop.clusterId, lp + dl);
      for (let k = 0; k < m; k++) {
        const node = nodes[ids[k]];
        const w = 1 / Math.max(0.02, node.mass || 1);
        node.vx += (w * gx[k] * dl) / dtPos;
        node.vy += (w * gy[k] * dl) / dtPos;
      }
    }
  }
}

function convexSATDepth(polyA, polyB) {
  const axes = [];
  const addAxes = (poly) => {
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const ex = b.x - a.x;
      const ey = b.y - a.y;
      const len = Math.hypot(ex, ey);
      if (len < 1e-8) continue;
      axes.push({ x: -ey / len, y: ex / len });
    }
  };
  addAxes(polyA);
  addAxes(polyB);

  let minOverlap = Infinity;
  for (const axis of axes) {
    let aMin = Infinity; let aMax = -Infinity;
    let bMin = Infinity; let bMax = -Infinity;
    for (const p of polyA) {
      const d = p.x * axis.x + p.y * axis.y;
      aMin = Math.min(aMin, d); aMax = Math.max(aMax, d);
    }
    for (const p of polyB) {
      const d = p.x * axis.x + p.y * axis.y;
      bMin = Math.min(bMin, d); bMax = Math.max(bMax, d);
    }
    const overlap = Math.min(aMax, bMax) - Math.max(aMin, bMin);
    if (overlap <= 0) return 0;
    minOverlap = Math.min(minOverlap, overlap);
  }
  return Number.isFinite(minOverlap) ? minOverlap : 0;
}

function softClusterAreaRatios(nodes, restMap) {
  const loops = clusterLoops(nodes);
  const out = [];
  for (const loop of loops) {
    const area = signedAreaPredicted(nodes, loop.ids, 0);
    const rest = restMap.get(loop.clusterId);
    if (!Number.isFinite(rest) || Math.abs(rest) < 1e-6) continue;
    out.push({ clusterId: loop.clusterId, ratio: Math.abs(area / rest) });
  }
  return out;
}

function runScenario(scenario) {
  const sim = {
    n: scenario.grid,
    dt: scenario.dt,
    bodies: clone(scenario.bodies),
    springLambda: new Float32Array(scenario.bodies.soft.springs.length),
    areaRest: new Map(),
    areaLambda: new Map(),
    fluid: makeShadowOnlyFluid({
      size: Number(scenario?.fluid?.size) || scenario.grid,
      dt: Number(scenario?.fluid?.dt) || scenario.dt,
      scaleX: 1,
      scaleY: 1,
    }),
  };

  const loops0 = clusterLoops(sim.bodies.soft.nodes);
  for (const loop of loops0) {
    sim.areaRest.set(loop.clusterId, signedAreaPredicted(sim.bodies.soft.nodes, loop.ids, 0));
  }

  const actorPrevCenter = new Map();
  const violations = [];
  const telemetry = [];

  for (let step = 0; step < scenario.steps; step++) {
    // Deterministic fluid driver so each watchdog run exercises body↔fluid coupling.
    const swirlX = sim.n * (0.5 + Math.cos(step * 0.041) * 0.18);
    const swirlY = sim.n * (0.5 + Math.sin(step * 0.053) * 0.16);
    sim.fluid.addVelocity(swirlX, swirlY, Math.cos(step * 0.17) * 1.5, Math.sin(step * 0.13) * 1.5, 8);

    let maxFluidSampleSpeed = 0;

    for (let i = 0; i < sim.bodies.rigid.length; i++) {
      const rb = sim.bodies.rigid[i];
      const flow = sim.fluid.getVelocityAtWorld(rb.x, rb.y);
      maxFluidSampleSpeed = Math.max(maxFluidSampleSpeed, Math.hypot(flow.vx, flow.vy));
      rb.vx += (flow.vx - rb.vx) * 0.045;
      rb.vy += (flow.vy - rb.vy) * 0.045;
      rb.vx += Math.cos(step * 0.08 + i * 1.37) * 0.01;
      rb.vy += Math.sin(step * 0.06 + i * 0.91) * 0.009;
      rb.vx *= 0.992;
      rb.vy *= 0.992;
      rb.omega = (rb.omega || 0) * 0.992;
    }
    for (let i = 0; i < sim.bodies.soft.nodes.length; i++) {
      const node = sim.bodies.soft.nodes[i];
      const flow = sim.fluid.getVelocityAtWorld(node.x, node.y);
      maxFluidSampleSpeed = Math.max(maxFluidSampleSpeed, Math.hypot(flow.vx, flow.vy));
      node.vx += (flow.vx - node.vx) * 0.05;
      node.vy += (flow.vy - node.vy) * 0.05;
      node.vx += Math.cos(step * 0.07 + i * 0.5) * 0.004;
      node.vy += Math.sin(step * 0.05 + i * 0.73) * 0.004;
      node.vx *= 0.99;
      node.vy *= 0.99;
    }

    const dtPos = Math.max(1e-4, sim.dt * INTEGRATION_SCALE);
    sim.springLambda.fill(0);
    applyXPBDSprings(sim.bodies.soft.nodes, sim.bodies.soft.springs, dtPos, sim.springLambda, 3.0);

    const loops = clusterLoops(sim.bodies.soft.nodes);
    applyXPBDArea(sim.bodies.soft.nodes, loops, dtPos, sim.areaRest, sim.areaLambda, 3.0);

    for (const rb of sim.bodies.rigid) {
      rb.x += rb.vx * sim.dt * 22;
      rb.y += rb.vy * sim.dt * 28;
      rb.theta = (rb.theta || 0) + (rb.omega || 0) * sim.dt * 60;
      applyBounceBoundary(rb, sim.n, 0.84);
    }
    for (const node of sim.bodies.soft.nodes) {
      node.x += node.vx * sim.dt * INTEGRATION_SCALE;
      node.y += node.vy * sim.dt * INTEGRATION_SCALE;
      applyBounceBoundary(node, sim.n, 0.78);
    }

    for (let iter = 0; iter < 2; iter++) {
      for (let i = 0; i < sim.bodies.rigid.length; i++) {
        for (let j = i + 1; j < sim.bodies.rigid.length; j++) {
          resolveRigidVsRigidPolygonCollision(sim.bodies.rigid[i], sim.bodies.rigid[j], 0.28);
        }
      }
      for (const rb of sim.bodies.rigid) {
        for (const node of sim.bodies.soft.nodes) {
          resolveRigidVsSoftNodeCollision(rb, node, null, 0.28);
        }
      }
      for (let i = 0; i < sim.bodies.soft.nodes.length; i++) {
        for (let j = i + 1; j < sim.bodies.soft.nodes.length; j++) {
          if ((sim.bodies.soft.nodes[i].clusterId ?? 0) === (sim.bodies.soft.nodes[j].clusterId ?? 0)) continue;
          resolveSoftNodeVsSoftNode(sim.bodies.soft.nodes[i], sim.bodies.soft.nodes[j], 0.1);
        }
      }
      for (const rb of sim.bodies.rigid) applyBounceBoundary(rb, sim.n, 0.84);
      for (const node of sim.bodies.soft.nodes) applyBounceBoundary(node, sim.n, 0.78);
    }

    // Two-way body feedback into the fluid shadow field, then advect one step.
    for (const rb of sim.bodies.rigid) {
      sim.fluid.addVelocity(rb.x, rb.y, rb.vx * 0.38, rb.vy * 0.38, Math.max(5, rb.r * 0.9));
    }
    for (const node of sim.bodies.soft.nodes) {
      sim.fluid.addVelocity(node.x, node.y, node.vx * 0.18, node.vy * 0.18, 2.6);
    }
    sim.fluid.step();

    let maxActorDelta = 0;
    let maxSpeed = 0;
    for (let i = 0; i < sim.bodies.rigid.length; i++) {
      const rb = sim.bodies.rigid[i];
      maxSpeed = Math.max(maxSpeed, Math.hypot(rb.vx, rb.vy));
      const key = `rigid:${i}`;
      const prev = actorPrevCenter.get(key);
      if (prev) maxActorDelta = Math.max(maxActorDelta, Math.hypot(rb.x - prev.x, rb.y - prev.y));
      actorPrevCenter.set(key, { x: rb.x, y: rb.y });
    }

    const byCluster = new Map();
    for (const node of sim.bodies.soft.nodes) {
      maxSpeed = Math.max(maxSpeed, Math.hypot(node.vx, node.vy));
      const cid = node.clusterId ?? 0;
      if (!byCluster.has(cid)) byCluster.set(cid, []);
      byCluster.get(cid).push(node);
    }
    for (const [cid, nodes] of byCluster.entries()) {
      let cx = 0; let cy = 0;
      for (const node of nodes) { cx += node.x; cy += node.y; }
      cx /= nodes.length;
      cy /= nodes.length;
      const key = `soft:${cid}`;
      const prev = actorPrevCenter.get(key);
      if (prev) maxActorDelta = Math.max(maxActorDelta, Math.hypot(cx - prev.x, cy - prev.y));
      actorPrevCenter.set(key, { x: cx, y: cy });
    }

    let maxRigidOverlap = 0;
    for (let i = 0; i < sim.bodies.rigid.length; i++) {
      for (let j = i + 1; j < sim.bodies.rigid.length; j++) {
        const pa = rigidVerticesWorld(sim.bodies.rigid[i]);
        const pb = rigidVerticesWorld(sim.bodies.rigid[j]);
        maxRigidOverlap = Math.max(maxRigidOverlap, convexSATDepth(pa, pb));
      }
    }

    let maxSoftPenetration = 0;
    const nodes = sim.bodies.soft.nodes;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if ((nodes[i].clusterId ?? 0) === (nodes[j].clusterId ?? 0)) continue;
        const d = Math.hypot(nodes[j].x - nodes[i].x, nodes[j].y - nodes[i].y);
        maxSoftPenetration = Math.max(maxSoftPenetration, (nodes[i].r + nodes[j].r) - d);
      }
    }

    let maxSoftInsideRigidDepth = 0;
    for (const rb of sim.bodies.rigid) {
      const poly = rigidVerticesWorld(rb);
      for (const node of sim.bodies.soft.nodes) {
        if (!pointInPolygonInclusive(node.x, node.y, poly)) continue;
        let minEdgeDist = Infinity;
        for (let i = 0; i < poly.length; i++) {
          const a = poly[i];
          const b = poly[(i + 1) % poly.length];
          const ex = b.x - a.x;
          const ey = b.y - a.y;
          const t = clamp(((node.x - a.x) * ex + (node.y - a.y) * ey) / Math.max(1e-8, ex * ex + ey * ey), 0, 1);
          const px = a.x + ex * t;
          const py = a.y + ey * t;
          minEdgeDist = Math.min(minEdgeDist, Math.hypot(node.x - px, node.y - py));
        }
        maxSoftInsideRigidDepth = Math.max(maxSoftInsideRigidDepth, minEdgeDist + node.r);
      }
    }

    const areaRatios = softClusterAreaRatios(sim.bodies.soft.nodes, sim.areaRest);
    const maxAreaDeviation = areaRatios.reduce((m, r) => Math.max(m, Math.abs(1 - r.ratio)), 0);

    const finite = [...sim.bodies.rigid, ...sim.bodies.soft.nodes].every((b) => [b.x, b.y, b.vx, b.vy].every(Number.isFinite));

    const checks = {
      finite,
      maxActorDelta,
      maxSpeed,
      maxFluidSampleSpeed,
      maxRigidOverlap,
      maxSoftPenetration,
      maxSoftInsideRigidDepth,
      maxAreaDeviation,
    };

    telemetry.push({ step, ...checks });

    const stepViolations = [];
    if (!finite) stepViolations.push('non-finite state');
    if (maxActorDelta > 7.5) stepViolations.push(`jerk spike ${maxActorDelta.toFixed(3)}`);
    if (maxSpeed > 8.5) stepViolations.push(`velocity spike ${maxSpeed.toFixed(3)}`);
    if (maxFluidSampleSpeed > 6.5) stepViolations.push(`fluid speed spike ${maxFluidSampleSpeed.toFixed(3)}`);
    if (maxRigidOverlap > 0.9) stepViolations.push(`rigid overlap ${maxRigidOverlap.toFixed(3)}`);
    if (maxSoftPenetration > 0.7) stepViolations.push(`soft-soft penetration ${maxSoftPenetration.toFixed(3)}`);
    if (maxSoftInsideRigidDepth > 0.9) stepViolations.push(`soft-rigid penetration ${maxSoftInsideRigidDepth.toFixed(3)}`);
    if (maxAreaDeviation > 0.5) stepViolations.push(`soft area drift ${maxAreaDeviation.toFixed(3)}`);

    if (stepViolations.length) {
      violations.push({ step, issues: stepViolations, checks });
      if (violations.length >= 4) break;
    }
  }

  const final = telemetry[telemetry.length - 1] || null;
  return {
    pass: violations.length === 0,
    violations,
    final,
    sampledTelemetry: telemetry.filter((_, i) => i % 10 === 0 || i === telemetry.length - 1),
  };
}

async function loadCatalog() {
  try {
    const text = await fs.readFile(CATALOG_PATH, 'utf8');
    const json = JSON.parse(text);
    if (!Array.isArray(json?.scenarios)) return { version: 1, updatedAt: null, scenarios: [] };
    return json;
  } catch {
    return { version: 1, updatedAt: null, scenarios: [] };
  }
}

async function writeCatalog(catalog, scenario) {
  const scenarios = [
    ...(Array.isArray(catalog.scenarios) ? catalog.scenarios : []).filter((s) => s && s.id !== scenario.id),
    scenario,
  ];
  scenarios.sort((a, b) => String(a.createdAt || '').localeCompare(String(b.createdAt || '')));
  const trimmed = scenarios.slice(-MAX_SCENARIOS);
  const next = {
    version: 1,
    updatedAt: new Date().toISOString(),
    scenarios: trimmed,
  };
  await fs.mkdir(path.dirname(CATALOG_PATH), { recursive: true });
  await fs.writeFile(CATALOG_PATH, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
}

async function writeReport(report) {
  await fs.mkdir(REPORT_DIR, { recursive: true });
  await fs.writeFile(REPORT_LATEST, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  await fs.appendFile(REPORT_HISTORY, `${JSON.stringify(report)}\n`, 'utf8');
}

async function main() {
  const epoch30 = Math.floor(Date.now() / (30 * 60 * 1000));
  const seed = (epoch30 * 2654435761) >>> 0;
  const scenario = makeScenario(seed);
  const result = runScenario(scenario);

  const report = {
    ok: result.pass,
    generatedAt: new Date().toISOString(),
    scenarioId: scenario.id,
    combo: scenario.combo,
    seed: scenario.seed,
    grid: scenario.grid,
    steps: scenario.steps,
    final: result.final,
    violations: result.violations,
    sampledTelemetry: result.sampledTelemetry,
  };

  const catalog = await loadCatalog();
  await writeCatalog(catalog, scenario);
  await writeReport(report);

  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!result.pass) process.exitCode = 1;
}

main().catch((err) => {
  const msg = err && err.stack ? err.stack : String(err);
  process.stderr.write(`${msg}\n`);
  process.exitCode = 1;
});
