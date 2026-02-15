import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

test('pressure solve shader samples vertical neighbors with dy (non-square-safe projection)', () => {
  const sourcePath = resolve(process.cwd(), 'js/gpuFluidField.js');
  const src = readFileSync(sourcePath, 'utf8');

  const start = src.indexOf('this.pressureSolveFragmentShaderSource = `');
  assert.ok(start >= 0, 'pressure solve shader source should exist');
  const end = src.indexOf('`;', start);
  assert.ok(end > start, 'pressure solve shader should be a template literal');

  const shader = src.slice(start, end);

  assert.match(shader, /float\s+dy\s*=\s*1\.0\s*\/\s*u_resolution\.y\s*;/, 'shader should define dy from Y resolution');
  assert.match(shader, /v_texCoord\s*\+\s*vec2\(0\.0,\s*dy\)/, 'p_up should use dy for vertical offset');
  assert.match(shader, /v_texCoord\s*-\s*vec2\(0\.0,\s*dy\)/, 'p_down should use dy for vertical offset');
  assert.doesNotMatch(shader, /v_texCoord\s*[+-]\s*vec2\(0\.0,\s*dx\)/, 'vertical pressure neighbors must not use dx');
});
