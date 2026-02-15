import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveRigidVsRigidPolygonCollision, getRigidCollisionPolysWorld, rigidVerticesWorld, pointInPolygonInclusive } from '../../sim-server/public/rigid-collision.js';

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
