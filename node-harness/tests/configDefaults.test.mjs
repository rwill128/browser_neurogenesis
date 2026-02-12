import test from 'node:test';
import assert from 'node:assert/strict';

import config from '../../js/config.js';

test('camera config includes a finite MAX_ZOOM guard rail', () => {
  assert.equal(Number.isFinite(Number(config.MAX_ZOOM)), true);
  assert.ok(Number(config.MAX_ZOOM) > 0);
});
