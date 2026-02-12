import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';
import { createViscosityField } from '../../js/engine/environmentFields.js';

function withViscosityConfig(patch, fn) {
  const keys = [
    'MIN_VISCOSITY_MULTIPLIER',
    'MAX_VISCOSITY_MULTIPLIER',
    'VISCOSITY_LANDSCAPE_NOISE_SCALE',
    'VISCOSITY_LANDSCAPE_OCTAVES',
    'VISCOSITY_LANDSCAPE_LACUNARITY',
    'VISCOSITY_LANDSCAPE_GAIN',
    'VISCOSITY_LANDSCAPE_CONTRAST',
    'VISCOSITY_LANDSCAPE_BANDS'
  ];

  const backup = {};
  for (const k of keys) backup[k] = config[k];

  try {
    Object.assign(config, patch);
    return fn();
  } finally {
    Object.assign(config, backup);
  }
}

test('viscosity field keeps values inside configured min/max range', () => {
  const noise = {
    noise(x, y) {
      return Math.sin(x * 0.37 + y * 0.19);
    }
  };

  withViscosityConfig(
    {
      MIN_VISCOSITY_MULTIPLIER: 0.1,
      MAX_VISCOSITY_MULTIPLIER: 12,
      VISCOSITY_LANDSCAPE_NOISE_SCALE: 0.05,
      VISCOSITY_LANDSCAPE_OCTAVES: 3,
      VISCOSITY_LANDSCAPE_LACUNARITY: 2,
      VISCOSITY_LANDSCAPE_GAIN: 0.55,
      VISCOSITY_LANDSCAPE_CONTRAST: 0.9,
      VISCOSITY_LANDSCAPE_BANDS: 10
    },
    () => {
      const field = createViscosityField(24, () => 0.123, noise);
      for (const v of field) {
        assert.ok(v >= 0.1 - 1e-9, `value below min: ${v}`);
        assert.ok(v <= 12 + 1e-9, `value above max: ${v}`);
      }
    }
  );
});

test('viscosity landscape banding yields more distinct plateaus', () => {
  const noise = {
    noise(x, y) {
      return Math.sin(x * 0.31 + y * 0.17);
    }
  };

  withViscosityConfig(
    {
      MIN_VISCOSITY_MULTIPLIER: 0,
      MAX_VISCOSITY_MULTIPLIER: 1,
      VISCOSITY_LANDSCAPE_NOISE_SCALE: 0.06,
      VISCOSITY_LANDSCAPE_OCTAVES: 1,
      VISCOSITY_LANDSCAPE_LACUNARITY: 2,
      VISCOSITY_LANDSCAPE_GAIN: 0.5,
      VISCOSITY_LANDSCAPE_CONTRAST: 0,
      VISCOSITY_LANDSCAPE_BANDS: 0
    },
    () => {
      const smooth = createViscosityField(16, () => 0.2, noise);

      config.VISCOSITY_LANDSCAPE_BANDS = 4;
      const banded = createViscosityField(16, () => 0.2, noise);

      const uniqueSmooth = new Set(Array.from(smooth, (v) => Number(v.toFixed(4)))).size;
      const uniqueBanded = new Set(Array.from(banded, (v) => Number(v.toFixed(4)))).size;

      assert.ok(uniqueSmooth > uniqueBanded, `expected fewer distinct levels with banding (${uniqueSmooth} vs ${uniqueBanded})`);
      assert.ok(uniqueBanded <= 4, `expected <= 4 plateaus, got ${uniqueBanded}`);
    }
  );
});

test('viscosity contrast pushes values toward highs and lows', () => {
  const noise = {
    noise(x, y) {
      return Math.sin(x * 0.29 + y * 0.11);
    }
  };

  withViscosityConfig(
    {
      MIN_VISCOSITY_MULTIPLIER: 0,
      MAX_VISCOSITY_MULTIPLIER: 1,
      VISCOSITY_LANDSCAPE_NOISE_SCALE: 0.05,
      VISCOSITY_LANDSCAPE_OCTAVES: 2,
      VISCOSITY_LANDSCAPE_LACUNARITY: 2,
      VISCOSITY_LANDSCAPE_GAIN: 0.55,
      VISCOSITY_LANDSCAPE_BANDS: 0
    },
    () => {
      config.VISCOSITY_LANDSCAPE_CONTRAST = 0;
      const flat = createViscosityField(20, () => 0.4, noise);

      config.VISCOSITY_LANDSCAPE_CONTRAST = 0.9;
      const punchy = createViscosityField(20, () => 0.4, noise);

      const avgDistanceFromMid = (arr) => {
        let sum = 0;
        for (const v of arr) sum += Math.abs(v - 0.5);
        return sum / arr.length;
      };

      const flatSpread = avgDistanceFromMid(flat);
      const punchySpread = avgDistanceFromMid(punchy);

      assert.ok(punchySpread > flatSpread, `expected higher contrast spread (${punchySpread} <= ${flatSpread})`);
    }
  );
});
