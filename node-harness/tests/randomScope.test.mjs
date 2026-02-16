import test from 'node:test';
import assert from 'node:assert/strict';

import { withRandomSource } from '../../js/engine/randomScope.mjs';

test('withRandomSource swaps Math.random for callback duration and restores afterward', () => {
  const original = Math.random;
  let calls = 0;
  const rng = () => {
    calls += 1;
    return 0.123456;
  };

  const value = withRandomSource(rng, () => Math.random());

  assert.equal(value, 0.123456);
  assert.equal(calls, 1);
  assert.equal(Math.random, original);
});

test('withRandomSource restores Math.random even when callback throws', () => {
  const original = Math.random;

  assert.throws(() => withRandomSource(() => 0.5, () => {
    throw new Error('boom');
  }), /boom/);

  assert.equal(Math.random, original);
});

test('withRandomSource supports nested deterministic scopes', () => {
  const original = Math.random;

  const outer = () => 0.9;
  const inner = () => 0.1;

  const sampled = withRandomSource(outer, () => {
    const a = Math.random();
    const b = withRandomSource(inner, () => Math.random());
    const c = Math.random();
    return [a, b, c];
  });

  assert.deepEqual(sampled, [0.9, 0.1, 0.9]);
  assert.equal(Math.random, original);
});

test('withRandomSource bypasses patching for non-function rng values', () => {
  const original = Math.random;
  const sentinel = withRandomSource(null, () => Math.random);
  assert.equal(sentinel, original);
  assert.equal(Math.random, original);
});
