import test from 'node:test';
import assert from 'node:assert/strict';

import {
  parseConfigOverrideToken,
  applyConfigOverrides
} from '../../js/engine/configOverride.mjs';

test('parseConfigOverrideToken coerces booleans and numbers', () => {
  assert.deepEqual(parseConfigOverrideToken('A=true'), { key: 'A', value: true });
  assert.deepEqual(parseConfigOverrideToken('B=false'), { key: 'B', value: false });
  assert.deepEqual(parseConfigOverrideToken('C=12.5'), { key: 'C', value: 12.5 });
  assert.deepEqual(parseConfigOverrideToken('D=hello'), { key: 'D', value: 'hello' });
});

test('applyConfigOverrides updates known keys and reports unknown keys', () => {
  const cfg = { X: 1, Y: false };
  const out = applyConfigOverrides(cfg, ['X=5', 'Y=true', 'NOPE=1']);

  assert.equal(cfg.X, 5);
  assert.equal(cfg.Y, true);
  assert.deepEqual(out.unknown, ['NOPE']);
  assert.equal(out.applied.length, 2);
});
