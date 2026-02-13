import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';

test('camera config includes a finite MAX_ZOOM guard rail', () => {
  assert.equal(Number.isFinite(Number(config.MAX_ZOOM)), true);
  assert.ok(Number(config.MAX_ZOOM) > 0);
});

test('particle life decay is editable by default (not forced infinite)', () => {
  assert.equal(config.IS_PARTICLE_LIFE_INFINITE, false);
});

test('initial triangulated primitive defaults are enabled and weighted', () => {
  assert.equal(config.INITIAL_TRIANGULATED_PRIMITIVES_ENABLED, true);
  assert.ok(Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_TRIANGLE) > 0);
  assert.ok(Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_DIAMOND) > 0);
  assert.ok(Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_HEXAGON) > 0);
});

test('photosynth topology hard constraints are relaxed by default', () => {
  assert.equal(config.PHOTOSYNTH_FORCE_RIGID_CONNECTED_SPRINGS, false);
  assert.equal(config.PHOTOSYNTH_NEUTRALIZE_NON_PHOTOSYNTH_NEIGHBORS, false);
});

test('growth triangulated primitive mode is enabled by default', () => {
  assert.equal(config.GROWTH_TRIANGULATED_PRIMITIVES_ENABLED, true);
});

test('growth stage genetics are enabled with sensible defaults', () => {
  assert.equal(config.GROWTH_STAGE_GENETICS_ENABLED, true);
  assert.ok(Number(config.GROWTH_STAGE_COUNT_MIN) >= 1);
  assert.ok(Number(config.GROWTH_STAGE_COUNT_MAX) >= Number(config.GROWTH_STAGE_COUNT_MIN));
});

test('intra-body repulsion is enabled with slight default strength', () => {
  assert.equal(config.INTRA_BODY_REPULSION_ENABLED, true);
  assert.ok(Number(config.INTRA_BODY_REPULSION_STRENGTH) > 0);
  assert.ok(Number(config.INTRA_BODY_REPULSION_RADIUS_FACTOR) > 0);
  assert.equal(config.INTRA_BODY_REPULSION_SKIP_CONNECTED, true);
});

test('landscape dye emitter defaults are available and disabled by default', () => {
  assert.equal(config.LANDSCAPE_DYE_EMITTERS_ENABLED, false);
  assert.equal(Number(config.LANDSCAPE_DYE_EMITTER_COUNT), 0);
  assert.ok(Number(config.LANDSCAPE_DYE_EMITTER_STRENGTH_MIN) >= 0);
  assert.ok(Number(config.LANDSCAPE_DYE_EMITTER_STRENGTH_MAX) >= Number(config.LANDSCAPE_DYE_EMITTER_STRENGTH_MIN));
  assert.ok(Number(config.FLUID_FADE_RATE) > 0);
});

test('viscosity landscape defaults expose stronger-shape controls', () => {
  assert.ok(Number(config.VISCOSITY_LANDSCAPE_NOISE_SCALE) > 0);
  assert.ok(Number(config.VISCOSITY_LANDSCAPE_OCTAVES) >= 1);
  assert.ok(Number(config.VISCOSITY_LANDSCAPE_LACUNARITY) >= 1);
  assert.ok(Number(config.VISCOSITY_LANDSCAPE_GAIN) > 0);
  assert.ok(Number(config.VISCOSITY_LANDSCAPE_CONTRAST) >= 0);
  assert.ok(Number(config.VISCOSITY_LANDSCAPE_BANDS) >= 0);
});

test('edge-length telemetry defaults are enabled and hard cap defaults on at factor 6+', () => {
  assert.equal(config.EDGE_LENGTH_TELEMETRY_ENABLED, true);
  assert.ok(Number(config.EDGE_LENGTH_TELEMETRY_SAMPLE_EVERY_N_STEPS) >= 1);
  assert.ok(Number(config.EDGE_LENGTH_TELEMETRY_MODE_BIN_SIZE) > 0);
  assert.equal(config.EDGE_LENGTH_HARD_CAP_ENABLED, true);
  assert.ok(Number(config.EDGE_LENGTH_HARD_CAP_FACTOR) >= 6);
});

test('physics motion guard defaults are enabled with finite caps', () => {
  assert.equal(config.PHYSICS_MOTION_GUARD_ENABLED, true);
  assert.equal(config.PHYSICS_NONFINITE_FORCE_ZERO, true);
  assert.ok(Number(config.PHYSICS_MAX_ACCELERATION_MAGNITUDE) > 0);
  assert.ok(Number(config.PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP) > 0);
  assert.ok(Number(config.PHYSICS_MAX_IMPLICIT_VELOCITY_PER_STEP) < Number(config.MAX_PIXELS_PER_FRAME_DISPLACEMENT));
});
