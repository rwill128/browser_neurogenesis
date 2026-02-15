import test from 'node:test';
import assert from 'node:assert/strict';
import { compileFieldToMesh } from '../../sim-server/public/field-to-structure-core.js';

function pointInPolygon(px, py, poly, eps = 1e-6) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;

    if (pointOnSegment(px, py, xi, yi, xj, yj, eps)) return true;

    const intersect = ((yi > py) !== (yj > py))
      && (px < ((xj - xi) * (py - yi)) / Math.max(1e-9, (yj - yi)) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function pointOnSegment(px, py, ax, ay, bx, by, eps = 1e-6) {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const cross = Math.abs(abx * apy - aby * apx);
  if (cross > eps) return false;

  const dot = apx * abx + apy * aby;
  if (dot < -eps) return false;

  const len2 = abx * abx + aby * aby;
  if (dot > len2 + eps) return false;

  return true;
}

test('field-to-structure compiler creates rigid and soft triangles from painted regions', () => {
  const w = 16, h = 16;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // rigid patch on left
  for (let y = 3; y <= 10; y++) {
    for (let x = 2; x <= 6; x++) rigid[y * w + x] = 1;
  }
  // soft patch on right
  for (let y = 4; y <= 11; y++) {
    for (let x = 9; x <= 13; x++) soft[y * w + x] = 1;
  }

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.2,
    density: 2,
  });

  assert.ok(mesh.nodes.length > 0);
  assert.ok(mesh.triangles.length > 0);
  assert.ok(mesh.meta.rigidTriangles > 0);
  assert.ok(mesh.meta.softTriangles > 0);
});

test('higher threshold reduces generated triangles', () => {
  const w = 16, h = 16;
  const rigid = new Float32Array(w * h).fill(0.4);
  const soft = new Float32Array(w * h).fill(0.0);

  const low = compileFieldToMesh({ width: w, height: h, rigidField: rigid, softField: soft, threshold: 0.2, density: 2 });
  const high = compileFieldToMesh({ width: w, height: h, rigidField: rigid, softField: soft, threshold: 0.6, density: 2 });

  assert.ok(low.triangles.length > high.triangles.length);
});

test('compiler emits soft cross-beams for square soft cells', () => {
  const w = 16, h = 16;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  for (let y = 4; y <= 11; y++) {
    for (let x = 4; x <= 11; x++) soft[y * w + x] = 1;
  }

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 2,
    connectivityMode: 'largest',
  });

  assert.ok(Array.isArray(mesh.softCrossBeams));
  assert.ok(mesh.softCrossBeams.length > 0, 'expected diagonal cross-beams for square soft lattice');
  assert.ok((mesh.meta.softCrossBeams || 0) === mesh.softCrossBeams.length);
  assert.equal(mesh.meta.softInfillMode, 'triangles+cross');
});

test('compiler can use triangle-only soft in-fill mode (no cross-beams)', () => {
  const w = 16, h = 16;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  for (let y = 4; y <= 11; y++) {
    for (let x = 4; x <= 11; x++) soft[y * w + x] = 1;
  }

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 2,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
  });

  assert.ok(Array.isArray(mesh.softCrossBeams));
  assert.equal(mesh.softCrossBeams.length, 0, 'triangle-only mode should not emit cross-beams');
  assert.equal(mesh.meta.softCrossBeams, 0);
  assert.equal(mesh.meta.softInfillMode, 'triangles');
  assert.ok(mesh.meta.softTriangles > 0, 'triangle-only mode should still produce soft triangles');
});

test('triangles+cross differs from triangles even under resolution-map adaptive cells', () => {
  const w = 64, h = 64;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);
  const resolution = new Float32Array(w * h).fill(0.5);

  for (let y = 8; y <= 56; y++) {
    for (let x = 8; x <= 56; x++) {
      soft[y * w + x] = 1;
      // Create local primitive-size variation so adaptive meshing is active.
      if (x < 28 && y < 36) resolution[y * w + x] = 0.1;
      if (x > 36 && y > 24) resolution[y * w + x] = 0.9;
    }
  }

  const triOnly = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    softDensityField: resolution,
    threshold: 0.35,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
  });

  const withCross = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    softDensityField: resolution,
    threshold: 0.35,
    connectivityMode: 'largest',
    softInfillMode: 'triangles+cross',
  });

  assert.equal(triOnly.softCrossBeams.length, 0, 'triangle-only must not emit cross-beams');
  assert.ok(withCross.softCrossBeams.length > 0, 'triangles+cross should emit additional beams');
});


test('compiler applies soft density map as local primitive-size control with seam continuity', () => {
  const w = 64, h = 64;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);
  const softDensity = new Float32Array(w * h).fill(0.0);

  for (let y = 8; y <= 56; y++) {
    for (let x = 8; x <= 56; x++) {
      soft[y * w + x] = 1;
      if (x >= 32) softDensity[y * w + x] = 1.0; // brighter half => coarser primitives
    }
  }

  const baseline = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 2,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
  });

  const mapped = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    softDensityField: softDensity,
    threshold: 0.35,
    density: 2,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
  });

  assert.equal(mapped.meta.densitySource, 'map', 'density map should drive primitive-size mode');
  assert.equal(mapped.meta.softDensityCulled, 0, 'size-control map should not sparse-cull soft triangles');
  assert.equal(mapped.meta.rigidDensityCulled, 0, 'no rigid fill present, so rigid cull count should remain zero');

  const softLeftFine = mapped.triangles.filter((t) => {
    if (t.kind !== 'soft') return false;
    const a = mapped.nodes[t.a], b = mapped.nodes[t.b], c = mapped.nodes[t.c];
    const cx = (a.x + b.x + c.x) / 3;
    return cx < 30;
  }).length;
  const softRightCoarse = mapped.triangles.filter((t) => {
    if (t.kind !== 'soft') return false;
    const a = mapped.nodes[t.a], b = mapped.nodes[t.b], c = mapped.nodes[t.c];
    const cx = (a.x + b.x + c.x) / 3;
    return cx > 34;
  }).length;
  assert.ok(softLeftFine > softRightCoarse, `expected darker/finer side to contain more primitives (left=${softLeftFine}, right=${softRightCoarse})`);

  const seamBand = mapped.triangles.filter((t) => {
    if (t.kind !== 'soft') return false;
    const a = mapped.nodes[t.a], b = mapped.nodes[t.b], c = mapped.nodes[t.c];
    const cx = (a.x + b.x + c.x) / 3;
    const cy = (a.y + b.y + c.y) / 3;
    return cx >= 30 && cx <= 34 && cy >= 12 && cy <= 52;
  });
  assert.ok(seamBand.length > 0, 'expected seam transition belt triangles to remain across density shift boundary');

  const outerSkin = mapped.triangles.filter((t) => {
    if (t.kind !== 'soft') return false;
    const a = mapped.nodes[t.a], b = mapped.nodes[t.b], c = mapped.nodes[t.c];
    const cx = (a.x + b.x + c.x) / 3;
    const cy = (a.y + b.y + c.y) / 3;
    const nearOuter = (cx <= 11 || cx >= 53 || cy <= 11 || cy >= 53);
    return nearOuter;
  });
  assert.ok(outerSkin.length > 0, 'expected outer boundary skin triangles to remain preserved');
});

test('soft minimum primitive size clamps adaptive tiny triangles', () => {
  const w = 64, h = 64;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);
  const resolution = new Float32Array(w * h).fill(0.0); // darkest => finest requested

  for (let y = 10; y <= 54; y++) {
    for (let x = 10; x <= 54; x++) soft[y * w + x] = 1;
  }

  const tinyAllowed = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    softDensityField: resolution,
    threshold: 0.35,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
    softMinCellSize: 1,
  });

  const clamped = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    softDensityField: resolution,
    threshold: 0.35,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
    softMinCellSize: 4,
  });

  assert.equal(tinyAllowed.meta.softMinCellSize, 1);
  assert.equal(clamped.meta.softMinCellSize, 4);
  assert.ok(
    clamped.meta.softTriangles < tinyAllowed.meta.softTriangles,
    `expected soft primitive floor to reduce triangle count (tiny=${tinyAllowed.meta.softTriangles}, clamped=${clamped.meta.softTriangles})`,
  );
});

test('density map also modulates rigid infill resolution (not soft-only)', () => {
  const w = 64, h = 64;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);
  const softDensity = new Float32Array(w * h).fill(0.0);

  for (let y = 8; y <= 56; y++) {
    for (let x = 8; x <= 56; x++) {
      rigid[y * w + x] = 1;
      if (x >= 32) softDensity[y * w + x] = 1.0; // brighter => coarser primitives
    }
  }

  const baseline = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 2,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
  });

  const mapped = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    softDensityField: softDensity,
    threshold: 0.35,
    density: 2,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
  });

  assert.equal(mapped.meta.densitySource, 'map', 'density map should own effective infill step source');
  assert.equal(mapped.meta.rigidDensityCulled, 0, 'size-control map should not sparse-cull rigid triangles');
  assert.ok(mapped.meta.rigidPieces > 0, 'rigid contour extraction should remain available/authoritative');

  const rigidLeftFine = mapped.triangles.filter((t) => {
    if (t.kind !== 'rigid') return false;
    const a = mapped.nodes[t.a], b = mapped.nodes[t.b], c = mapped.nodes[t.c];
    const cx = (a.x + b.x + c.x) / 3;
    return cx < 30;
  }).length;
  const rigidRightCoarse = mapped.triangles.filter((t) => {
    if (t.kind !== 'rigid') return false;
    const a = mapped.nodes[t.a], b = mapped.nodes[t.b], c = mapped.nodes[t.c];
    const cx = (a.x + b.x + c.x) / 3;
    return cx > 34;
  }).length;
  assert.ok(rigidLeftFine > rigidRightCoarse, `expected darker/finer rigid side to contain more primitives (left=${rigidLeftFine}, right=${rigidRightCoarse})`);
});

test('density map controls effective infill step (darker=>smaller, brighter=>larger)', () => {
  const w = 64, h = 64;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);
  const dark = new Float32Array(w * h).fill(0.05);
  const bright = new Float32Array(w * h).fill(0.95);

  for (let y = 10; y <= 54; y++) {
    for (let x = 10; x <= 54; x++) {
      rigid[y * w + x] = 1;
    }
  }

  const denseMesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    softDensityField: dark,
    threshold: 0.35,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
  });

  const sparseMesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    softDensityField: bright,
    threshold: 0.35,
    connectivityMode: 'largest',
    softInfillMode: 'triangles',
  });

  assert.ok(denseMesh.meta.density < sparseMesh.meta.density,
    `expected darker map to produce smaller step, got dense=${denseMesh.meta.density} sparse=${sparseMesh.meta.density}`);
  assert.ok(denseMesh.meta.rigidTriangles > sparseMesh.meta.rigidTriangles,
    'smaller step should produce denser rigid triangle coverage than brighter/sparser map');
});

test('connectivity largest mode drops disconnected islands', () => {
  const w = 20, h = 20;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // big island
  for (let y = 3; y <= 10; y++) for (let x = 3; x <= 10; x++) rigid[y * w + x] = 1;
  // tiny island
  for (let y = 14; y <= 15; y++) for (let x = 14; x <= 15; x++) rigid[y * w + x] = 1;

  const raw = compileFieldToMesh({ width: w, height: h, rigidField: rigid, softField: soft, threshold: 0.2, density: 2, connectivityMode: 'none' });
  const connected = compileFieldToMesh({ width: w, height: h, rigidField: rigid, softField: soft, threshold: 0.2, density: 2, connectivityMode: 'largest' });

  assert.ok(raw.triangles.length > connected.triangles.length);
  assert.ok(connected.meta.components >= 2);
  assert.equal(connected.meta.keptComponents, 1);
});

test('connectivity largest mode enforces single connected body across rigid+soft', () => {
  const w = 28, h = 20;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // dominant rigid component
  for (let y = 2; y <= 10; y++) for (let x = 2; x <= 12; x++) rigid[y * w + x] = 1;

  // disconnected soft component should be dropped under strict single-body policy
  for (let y = 3; y <= 10; y++) for (let x = 20; x <= 25; x++) soft[y * w + x] = 1;

  const connected = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.2,
    density: 2,
    connectivityMode: 'largest',
  });

  assert.equal(connected.meta.keptComponents, 1);
  assert.ok(connected.meta.rigidTriangles > 0);
  assert.equal(connected.meta.softTriangles, 0);
});

test('compiler exposes rigid decomposition pieces at compile time', () => {
  const w = 24, h = 24;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // concave-ish rigid "L" shape
  for (let y = 4; y <= 16; y++) for (let x = 4; x <= 9; x++) rigid[y * w + x] = 1;
  for (let y = 12; y <= 16; y++) for (let x = 4; x <= 16; x++) rigid[y * w + x] = 1;

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.2,
    density: 2,
    connectivityMode: 'largest',
  });

  assert.ok(Array.isArray(mesh.rigidPieces));
  assert.ok(mesh.rigidPieces.length >= 1);
  for (const p of mesh.rigidPieces) {
    assert.ok(Array.isArray(p.hull));
    assert.ok(p.hull.length >= 3);
  }
});

test('soft triangles do not overlap rigid contour in rigid+soft overlap zones', () => {
  const w = 64, h = 64;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // rigid L
  for (let y = 10; y <= 48; y++) for (let x = 10; x <= 20; x++) rigid[y * w + x] = 1;
  for (let y = 36; y <= 48; y++) for (let x = 10; x <= 48; x++) rigid[y * w + x] = 1;

  // soft patch intentionally overlapping lower-right interior of rigid L
  for (let y = 28; y <= 48; y++) for (let x = 18; x <= 42; x++) soft[y * w + x] = 1;

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 6,
    connectivityMode: 'largest',
  });

  assert.ok(mesh.rigidPieces.length >= 1);
  const rigidHulls = mesh.rigidPieces.map((p) => p.hull).filter((h) => Array.isArray(h) && h.length >= 3);

  for (const t of mesh.triangles) {
    if (t.kind !== 'soft') continue;
    const a = mesh.nodes[t.a];
    const b = mesh.nodes[t.b];
    const c = mesh.nodes[t.c];
    const probes = [
      { x: a.x, y: a.y },
      { x: b.x, y: b.y },
      { x: c.x, y: c.y },
      { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 },
      { x: (b.x + c.x) * 0.5, y: (b.y + c.y) * 0.5 },
      { x: (c.x + a.x) * 0.5, y: (c.y + a.y) * 0.5 },
      { x: (a.x + b.x + c.x) / 3, y: (a.y + b.y + c.y) / 3 },
    ];

    for (const hull of rigidHulls) {
      for (const p of probes) {
        assert.equal(pointInPolygon(p.x, p.y, hull), false, 'soft triangle probe overlapped rigid contour');
      }
    }
  }

  assert.ok((mesh.meta.softOverlapTrimmed || 0) > 0, 'expected overlap-trim guardrail to remove some soft triangles');
});

test('soft overlap trim treats rigid hull boundary as blocking (no seam-hugging soft triangles)', () => {
  const w = 48, h = 48;
  const rigid = new Float32Array(w * h);
  const soft = new Float32Array(w * h);

  // Compact rigid square.
  for (let y = 14; y <= 34; y++) for (let x = 14; x <= 34; x++) rigid[y * w + x] = 1;

  // Soft strip that exactly rides along the rigid top edge + slightly inside.
  for (let y = 10; y <= 16; y++) for (let x = 12; x <= 36; x++) soft[y * w + x] = 1;

  const mesh = compileFieldToMesh({
    width: w,
    height: h,
    rigidField: rigid,
    softField: soft,
    threshold: 0.35,
    density: 4,
    connectivityMode: 'largest',
  });

  assert.ok(mesh.rigidPieces.length >= 1, 'expected rigid contour to exist');
  const hulls = mesh.rigidPieces.map((p) => p.hull).filter((hull) => Array.isArray(hull) && hull.length >= 3);

  for (const t of mesh.triangles) {
    if (t.kind !== 'soft') continue;
    const a = mesh.nodes[t.a];
    const b = mesh.nodes[t.b];
    const c = mesh.nodes[t.c];
    const probes = [
      { x: a.x, y: a.y },
      { x: b.x, y: b.y },
      { x: c.x, y: c.y },
      { x: (a.x + b.x + c.x) / 3, y: (a.y + b.y + c.y) / 3 },
    ];

    for (const hull of hulls) {
      for (const p of probes) {
        assert.equal(pointInPolygon(p.x, p.y, hull), false, 'soft triangle is inside or on rigid hull boundary');
      }
    }
  }

  assert.ok((mesh.meta.softOverlapTrimmed || 0) > 0, 'expected seam-hugging overlap to be trimmed');
});
