import test from 'node:test';
import assert from 'node:assert/strict';
import {
  resolveRigidVsRigidPolygonCollision,
  getRigidCollisionPolysWorld,
  rigidVerticesWorld,
  pointInPolygonInclusive,
  buildRigidRigidSpatialHashCandidates,
  buildRigidSoftNodeCollisionCache,
  buildRigidSoftSpatialHashCandidates,
  buildCollisionPhaseSceneCache,
} from '../../sim-server/public/rigid-collision.js';

function makeConcaveL(x, y) {
  return {
    x,
    y,
    vx: 0,
    vy: 0,
    theta: 0,
    omega: 0,
    mass: 8,
    inertia: 220,
    r: 12,
    verticesLocal: [
      { x: -10, y: -10 },
      { x: 10, y: -10 },
      { x: 10, y: -4 },
      { x: -4, y: -4 },
      { x: -4, y: 10 },
      { x: -10, y: 10 },
    ],
  };
}

function makeBox(x, y, half = 2) {
  return {
    x,
    y,
    vx: 0,
    vy: 0,
    theta: 0,
    omega: 0,
    mass: 2,
    inertia: 20,
    r: half,
    verticesLocal: [
      { x: -half, y: -half },
      { x: half, y: -half },
      { x: half, y: half },
      { x: -half, y: half },
    ],
  };
}

test('rigid-rigid polygon collision: concave notch does not false-collide', () => {
  const a = makeConcaveL(32, 32);
  const b = makeBox(36, 36, 1.8); // in L notch void (outside solid)

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3);
  assert.equal(hit, false, 'notch occupancy should not produce rigid-rigid collision');
  assert.equal(a.x, 32);
  assert.equal(b.x, 36);
});

test('rigid-rigid polygon collision: overlapping boxes separate and exchange momentum', () => {
  const a = makeBox(20, 20, 3);
  const b = makeBox(24, 20, 3);
  a.vx = 1.5;
  b.vx = -0.5;

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.25);
  assert.equal(hit, true, 'expected overlap collision');
  assert.ok(a.x < 20.1, `A should be pushed left, got ${a.x}`);
  assert.ok(b.x > 23.9, `B should be pushed right, got ${b.x}`);
  assert.ok(a.vx < 1.5, `A should lose rightward speed, got ${a.vx}`);
  assert.ok(b.vx > -0.5, `B should gain rightward speed, got ${b.vx}`);
});

test('rigid-rigid polygon collision keeps finite state under high-speed impact', () => {
  const a = makeBox(20, 20, 4);
  const b = makeBox(26, 20, 4);
  a.vx = 60;
  b.vx = -40;

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3);
  assert.equal(hit, true, 'expected collision under overlap');
  for (const v of [a.vx, a.vy, b.vx, b.vy, a.omega, b.omega, a.x, b.x]) {
    assert.ok(Number.isFinite(v), `state should remain finite, got ${v}`);
  }
  assert.ok(Math.abs(a.vx) < 80, `guardrail should bound extreme velocity response, got ${a.vx}`);
  assert.ok(Math.abs(b.vx) < 80, `guardrail should bound extreme velocity response, got ${b.vx}`);
});

test('rigid-rigid polygon collision: no force when bodies are clearly separated', () => {
  const a = makeConcaveL(20, 20);
  const b = makeBox(58, 20, 3);
  a.vx = 0.7;
  b.vx = -0.3;

  const ax = a.x; const ay = a.y;
  const bx = b.x; const by = b.y;
  const avx = a.vx; const bvx = b.vx;

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3);
  assert.equal(hit, false, 'separated bodies must not collide');
  assert.equal(a.x, ax);
  assert.equal(a.y, ay);
  assert.equal(b.x, bx);
  assert.equal(b.y, by);
  assert.equal(a.vx, avx);
  assert.equal(b.vx, bvx);
});

test('rigid-rigid polygon collision: tiny overlap stays in slop band (no jitter correction)', () => {
  const a = makeBox(20, 20, 3);
  const b = makeBox(25.99, 20, 3); // overlap 0.01 < rigid contact slop
  a.vx = 0.2;
  b.vx = -0.1;

  const ax = a.x;
  const bx = b.x;
  const avx = a.vx;
  const bvx = b.vx;

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3);
  assert.equal(hit, false, 'tiny overlap should be ignored to prevent jitter at rest');
  assert.equal(a.x, ax);
  assert.equal(b.x, bx);
  assert.equal(a.vx, avx);
  assert.equal(b.vx, bvx);
});

test('rigid-rigid proxy decomposition stays inside concave hull and logs triggering proxy pair', () => {
  const a = makeConcaveL(32, 32);
  const b = makeBox(26, 26, 2.5);

  const hullA = rigidVerticesWorld(a);
  const polysA = getRigidCollisionPolysWorld(a);
  for (const poly of polysA) {
    let cx = 0; let cy = 0;
    for (const p of poly) { cx += p.x; cy += p.y; }
    cx /= Math.max(1, poly.length);
    cy /= Math.max(1, poly.length);
    assert.equal(pointInPolygonInclusive(cx, cy, hullA), true, 'proxy centroid should remain inside source concave hull');
  }

  const debug = { contacts: [], aIndex: 3, bIndex: 7, iter: 0, phase: 'test' };
  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3, debug);
  assert.equal(hit, true, 'expected overlap for proxy-pair logging case');
  assert.ok(debug.contacts.length >= 1, 'expected proxy contact log entry');
  const c = debug.contacts[0];
  assert.equal(c.a, 3);
  assert.equal(c.b, 7);
  assert.ok(Number.isInteger(c.proxyA) && c.proxyA >= 0);
  assert.ok(Number.isInteger(c.proxyB) && c.proxyB >= 0);
});

test('rigid collision proxy cache invalidates when local hull changes', () => {
  const body = makeBox(0, 0, 1);
  const first = getRigidCollisionPolysWorld(body);
  assert.equal(first.length, 1);
  assert.deepEqual(first[0][1], { x: 1, y: -1 });

  body.verticesLocal = [
    { x: -2, y: -2 },
    { x: 2, y: -2 },
    { x: 2, y: 2 },
    { x: -2, y: 2 },
  ];

  const second = getRigidCollisionPolysWorld(body);
  assert.equal(second.length, 1);
  assert.deepEqual(second[0][1], { x: 2, y: -2 }, 'world collision polys should refresh after topology edit');
});

test('rigid-rigid collision ignores non-finite local vertices deterministically', () => {
  const malformed = makeBox(14, 10, 2);
  malformed.verticesLocal = [
    { x: -2, y: -2 },
    { x: 2, y: -2 },
    { x: NaN, y: 0 },
    { x: 2, y: 2 },
    { x: -2, y: 2 },
    { x: -2, y: -2 },
  ];

  const b = makeBox(20, 10, 2);
  const ax = malformed.x;
  const bx = b.x;

  const hit = resolveRigidVsRigidPolygonCollision(malformed, b, 0.3);
  assert.equal(hit, false, 'non-overlapping malformed hull should not spuriously collide');
  assert.equal(malformed.x, ax, 'malformed body should not be moved by invalid geometry');
  assert.equal(b.x, bx, 'other body should remain unchanged');

  const polys = getRigidCollisionPolysWorld(malformed);
  assert.equal(polys.length, 1, 'sanitized convex proxy should still be available');
  assert.ok(polys[0].every((v) => Number.isFinite(v.x) && Number.isFinite(v.y)), 'cached proxy should be finite');
});

test('pointInPolygonInclusive is orientation-invariant on descending edges', () => {
  const poly = [
    { x: -3.70121809708695, y: -2.407090008712247 },
    { x: 0.7885652164026515, y: -1.423716793386125 },
    { x: 3.9016137324883324, y: -4.802757318032596 },
    { x: 4.423244291069521, y: -4.319993574872287 },
    { x: 4.656174669804701, y: 4.372090947773785 },
    { x: 1.4558556911874856, y: 2.366994659922299 },
    { x: 0.9095704001006224, y: 0.6897309632088007 },
    { x: -3.019057310101797, y: 1.7610003204523466 },
  ];
  const px = 2.9182181336383675;
  const py = -2.0885609761779342;

  assert.equal(pointInPolygonInclusive(px, py, poly), true, 'point should classify inside polygon');
  assert.equal(pointInPolygonInclusive(px, py, [...poly].reverse()), true, 'classification should be stable for reversed winding');
});

test('pointInPolygonInclusive keeps inside/outside classification stable across windings', () => {
  const poly = [
    { x: -4.889825200151876, y: -1.1713868636523896 },
    { x: -0.9143948169659533, y: -1.0382849723255982 },
    { x: -2.094022655840809, y: -3.958326780597795 },
    { x: 3.960405762190902, y: -3.4220954006394844 },
    { x: 2.065864045246874, y: 0.16777584034944137 },
    { x: 3.618644558107338, y: 4.946060228851987 },
    { x: -1.3380905228651918, y: 1.1529191086827435 },
    { x: -2.7627442987497997, y: 0.949753967738153 },
  ];

  const samples = [
    { x: 4.1401389656467344, y: 1.327161566969819, inside: false },
    { x: 0.0, y: 0.0, inside: true },
    { x: -4.7, y: -1.3, inside: false },
  ];

  for (const sample of samples) {
    const cw = pointInPolygonInclusive(sample.x, sample.y, poly);
    const ccw = pointInPolygonInclusive(sample.x, sample.y, [...poly].reverse());
    assert.equal(cw, sample.inside, `expected cw classification for (${sample.x}, ${sample.y})`);
    assert.equal(ccw, sample.inside, `expected ccw classification for (${sample.x}, ${sample.y})`);
  }
});

test('rigid collision world polys stay finite when pose contains non-finite values', () => {
  const malformed = makeBox(5, 5, 2);
  malformed.theta = Number.POSITIVE_INFINITY;
  malformed.x = Number.NaN;
  malformed.y = Number.NEGATIVE_INFINITY;

  const polys = getRigidCollisionPolysWorld(malformed);
  assert.equal(polys.length, 1, 'finite fallback should preserve one polygon proxy');
  for (const p of polys[0]) {
    assert.ok(Number.isFinite(p.x), `x should be finite, got ${p.x}`);
    assert.ok(Number.isFinite(p.y), `y should be finite, got ${p.y}`);
  }

  const verts = rigidVerticesWorld(malformed);
  assert.equal(verts.length, 4, 'finite fallback should preserve local-vertex transform');
  for (const v of verts) {
    assert.ok(Number.isFinite(v.x), `x should be finite, got ${v.x}`);
    assert.ok(Number.isFinite(v.y), `y should be finite, got ${v.y}`);
  }
});

test('rigid-rigid spatial hash broadphase prunes pair checks while keeping overlap candidate', () => {
  const collidingA = makeBox(20, 20, 3);
  const collidingB = makeBox(24.5, 20, 3);
  const farA = makeBox(120, 120, 3);
  const farB = makeBox(180, 180, 3);
  const farC = makeBox(220, 60, 3);
  const farD = makeBox(40, 210, 3);
  const bodies = [collidingA, collidingB, farA, farB, farC, farD];

  const result = buildRigidRigidSpatialHashCandidates(bodies, { cellSize: 12 });
  const pairSet = new Set(result.pairs.map(([i, j]) => `${i},${j}`));

  assert.equal(result.stats.bodyCount, 6);
  assert.equal(result.stats.bruteForcePairs, 15);
  assert.ok(result.stats.checkedPairs < result.stats.bruteForcePairs, 'broadphase should prune candidates');
  assert.ok(result.stats.prunedPairs > 0, 'expected some pair pruning');
  assert.ok(pairSet.has('0,1'), 'known overlapping pair must survive broadphase');
  assert.ok(Number(result.stats.emitAttempts) >= Number(result.stats.checkedPairs), 'emit attempts should cover output pairs');
  assert.ok(Number(result.stats.duplicatesRejected) >= 0, 'duplicates should be non-negative');
  assert.equal(Number(result.stats.pairsOut) || 0, Number(result.stats.checkedPairs) || 0, 'pairsOut should mirror checked pairs');
  assert.ok(Number(result.stats.dedupeMs) >= 0, 'dedupe timing should be non-negative');
  assert.ok(Number(result.stats.sortMs) >= 0, 'sort timing should be non-negative');
  assert.ok(Number(result.stats.totalBuildMs) >= Number(result.stats.dedupeMs), 'total build timing should include dedupe stage');
  assert.equal(result.stats.reuseHit, false, 'standalone broadphase should not report reuse hit by default');
});

test('rigid-rigid spatial hash broadphase candidate order is deterministic', () => {
  const bodies = [
    makeBox(12, 12, 2),
    makeBox(14, 12, 2),
    makeBox(60, 12, 2),
    makeBox(62, 12, 2),
    makeBox(100, 100, 2),
  ];

  const a = buildRigidRigidSpatialHashCandidates(bodies, { cellSize: 8 });
  const b = buildRigidRigidSpatialHashCandidates(bodies, { cellSize: 8 });

  assert.deepEqual(a.pairs, b.pairs, 'candidate pair ordering should be stable across identical runs');
  assert.deepEqual(a.stats.checkedPairs, b.stats.checkedPairs);
  assert.deepEqual(a.stats.prunedPairs, b.stats.prunedPairs);
  for (let i = 1; i < a.pairs.length; i++) {
    const prev = a.pairs[i - 1];
    const next = a.pairs[i];
    assert.ok((prev[0] < next[0]) || (prev[0] === next[0] && prev[1] <= next[1]), 'pairs should remain in deterministic i-then-j order');
  }
});

test('phase scene cache reuses shared rigid broadphase state for rigid-rigid and rigid-soft candidate sets', () => {
  const rigids = [
    makeBox(20, 20, 3),
    makeBox(24.5, 20, 3),
    makeBox(120, 120, 3),
  ];
  const nodes = [
    { x: 22, y: 20, r: 1.2 },
    { x: 19, y: 21, r: 1.2 },
    { x: 118, y: 120, r: 1.2 },
  ];
  const springs = [
    [0, 1, 1, 1],
    [1, 2, 1, 1],
  ];

  const scene = buildCollisionPhaseSceneCache(rigids, nodes, springs, {
    cellSize: 12,
    edgeBodyModeBlock: 1,
    includeRigidRigid: true,
    includeRigidSoft: true,
  });
  const rr = buildRigidRigidSpatialHashCandidates(rigids, { cellSize: 12 });
  const rs = buildRigidSoftSpatialHashCandidates(rigids, nodes, springs, { cellSize: 12, edgeBodyModeBlock: 1 });

  assert.deepEqual(scene.rigidRigid.pairs, rr.pairs, 'scene cache rigid-rigid pairs should match standalone broadphase');
  assert.deepEqual(scene.rigidSoft.nodeCandidatesByRigid, rs.nodeCandidatesByRigid, 'scene cache rigid-soft node candidates should match standalone broadphase');
  assert.deepEqual(scene.rigidSoft.edgeCandidatesByRigid, rs.edgeCandidatesByRigid, 'scene cache rigid-soft edge candidates should match standalone broadphase');
});

test('phase scene cache reuses rigid-rigid candidates when rigid cell spans are unchanged', () => {
  const rigids = [
    makeBox(20, 20, 3),
    makeBox(24.5, 20, 3),
    makeBox(120, 120, 3),
  ];
  const reuseCache = {};

  const first = buildCollisionPhaseSceneCache(rigids, [], [], {
    cellSize: 12,
    includeRigidRigid: true,
    includeRigidSoft: false,
    rigidRigidReuseCache: reuseCache,
  });
  const second = buildCollisionPhaseSceneCache(rigids, [], [], {
    cellSize: 12,
    includeRigidRigid: true,
    includeRigidSoft: false,
    rigidRigidReuseCache: reuseCache,
  });

  assert.equal(first.rigidRigid.stats.reuseHit, false, 'first call should build candidates');
  assert.equal(second.rigidRigid.stats.reuseHit, true, 'second call should reuse candidates when spans are unchanged');
  assert.equal(second.rigidRigid.stats.reuseMiss, false);
  assert.equal(Number(second.rigidRigid.stats.dedupeMs) || 0, 0, 'reused call should not spend dedupe time');
  assert.equal(Number(second.rigidRigid.stats.sortMs) || 0, 0, 'reused call should not spend sort time');
  assert.deepEqual(first.rigidRigid.pairs, second.rigidRigid.pairs, 'reused pair list should match built list');

  rigids[2].x += 30;
  const third = buildCollisionPhaseSceneCache(rigids, [], [], {
    cellSize: 12,
    includeRigidRigid: true,
    includeRigidSoft: false,
    rigidRigidReuseCache: reuseCache,
  });
  assert.equal(third.rigidRigid.stats.reuseHit, false, 'changed cell spans should invalidate reuse');
  assert.equal(third.rigidRigid.stats.reuseMiss, true);
});

test('rigid-rigid SAT prefilter rejects far proxy pairs before SAT dispatch', () => {
  const a = makeBox(20, 20, 2);
  const b = makeBox(120, 120, 2);
  const stats = {};

  const hit = resolveRigidVsRigidPolygonCollision(a, b, 0.3, null, { stats });
  assert.equal(hit, false, 'far-separated boxes should not collide');
  assert.ok((Number(stats.rigidRigidProxyPairChecks) || 0) >= 1, 'expected proxy prefilter checks');
  assert.ok((Number(stats.rigidRigidAabbRejected) || 0) >= 1, 'expected AABB prefilter rejection');
  assert.equal(Number(stats.rigidRigidSatCalls) || 0, 0, 'SAT should be skipped for far-separated proxy pair');
});

test('rigid-soft spatial hash broadphase prunes rigid-soft checks while preserving near candidates', () => {
  const rigids = [
    makeBox(20, 20, 3),
    makeBox(110, 110, 3),
    makeBox(220, 40, 3),
  ];
  const nodes = [
    { x: 22, y: 20, r: 1.2 },
    { x: 18, y: 21, r: 1.2 },
    { x: 108, y: 110, r: 1.2 },
    { x: 210, y: 200, r: 1.2 },
    { x: 240, y: 220, r: 1.2 },
  ];
  const springs = [
    [0, 1, 1, 1],
    [2, 3, 1, 1],
    [3, 4, 1, 0],
  ];

  const result = buildRigidSoftSpatialHashCandidates(rigids, nodes, springs, {
    cellSize: 12,
    edgeBodyModeBlock: 1,
  });

  assert.equal(result.stats.rigidCount, 3);
  assert.equal(result.stats.nodeCount, 5);
  assert.equal(result.stats.blockedEdgeCount, 2);
  assert.equal(result.stats.rawNodeChecks, 15);
  assert.equal(result.stats.rawEdgeChecks, 6);
  assert.ok(result.stats.candidateNodeChecks < result.stats.rawNodeChecks, 'node candidates should be pruned');
  assert.ok(result.stats.candidateEdgeChecks < result.stats.rawEdgeChecks, 'edge candidates should be pruned');

  const rigid0Nodes = result.nodeCandidatesByRigid[0] || [];
  assert.ok(rigid0Nodes.includes(0), 'near node candidate should survive for rigid[0]');
  assert.ok(rigid0Nodes.includes(1), 'second near node candidate should survive for rigid[0]');
  assert.ok(!rigid0Nodes.includes(4), 'far node should be pruned for rigid[0]');

  const rigid2Edges = result.edgeCandidatesByRigid[2] || [];
  assert.ok(!rigid2Edges.includes(0), 'far blocked edge should not be a candidate for rigid[2]');
});

test('rigid-soft spatial hash broadphase candidate order is deterministic', () => {
  const rigids = [
    makeBox(24, 24, 3),
    makeBox(56, 24, 3),
  ];
  const nodes = [
    { x: 25, y: 24, r: 1.2 },
    { x: 27, y: 24, r: 1.2 },
    { x: 55, y: 24, r: 1.2 },
    { x: 57, y: 24, r: 1.2 },
  ];
  const springs = [
    [0, 1, 1, 1],
    [2, 3, 1, 1],
  ];

  const a = buildRigidSoftSpatialHashCandidates(rigids, nodes, springs, { cellSize: 10, edgeBodyModeBlock: 1 });
  const b = buildRigidSoftSpatialHashCandidates(rigids, nodes, springs, { cellSize: 10, edgeBodyModeBlock: 1 });

  assert.deepEqual(a.nodeCandidatesByRigid, b.nodeCandidatesByRigid, 'node candidate ordering should be stable');
  assert.deepEqual(a.edgeCandidatesByRigid, b.edgeCandidatesByRigid, 'edge candidate ordering should be stable');
  assert.equal(a.stats.candidateNodeChecks, b.stats.candidateNodeChecks);
  assert.equal(a.stats.candidateEdgeChecks, b.stats.candidateEdgeChecks);
});

test('rigid-soft node cache build returns stable finite edge metadata', () => {
  const rigid = makeConcaveL(30, 30);
  const cache = buildRigidSoftNodeCollisionCache(rigid);
  assert.ok(cache, 'expected cache object');
  assert.ok(Array.isArray(cache.verts) && cache.verts.length >= 3);
  assert.ok(Array.isArray(cache.edges) && cache.edges.length === cache.verts.length);
  for (const e of cache.edges) {
    assert.ok(Number.isFinite(e.ax) && Number.isFinite(e.ay));
    assert.ok(Number.isFinite(e.bx) && Number.isFinite(e.by));
    assert.ok(Number.isFinite(e.nx) && Number.isFinite(e.ny));
  }
});
