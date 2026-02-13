import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = '/Users/richardwilliams/browser_neurogenesis';

/**
 * Guardrail test for browser config UI wiring:
 * every static getElementById() binding in js/ui.js should exist in index.html.
 *
 * This catches regressions where controls silently stop working due to missing IDs.
 */
test('ui.js DOM id bindings exist in index.html', () => {
  const uiSource = readFileSync(resolve(ROOT, 'js/ui.js'), 'utf8');
  const htmlSource = readFileSync(resolve(ROOT, 'index.html'), 'utf8');

  const idRegex = /document\.getElementById\('([^']+)'\)/g;
  const referenced = new Set();

  let match;
  while ((match = idRegex.exec(uiSource)) !== null) {
    referenced.add(match[1]);
  }

  // IDs that are generated dynamically or not expected in the static index shell.
  const optionalIds = new Set([
    // Dynamically inserted by ui.js when missing from static markup.
    'infoBodyReproCooldownGeneVal',
    'infoBodyReproCooldownGeneP',
    'infoBodyEffectiveReproCooldownVal',
    'infoBodyEffectiveReproCooldownP'
  ]);

  const missing = [...referenced]
    .filter((id) => !optionalIds.has(id))
    .filter((id) => !htmlSource.includes(`id="${id}"`));

  assert.deepEqual(
    missing,
    [],
    `Missing DOM ids referenced by js/ui.js: ${missing.join(', ')}`
  );
});
