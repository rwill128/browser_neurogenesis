/**
 * Small helpers to parse/apply runtime config overrides from CLI strings.
 */

/**
 * Parse one override token in KEY=VALUE format.
 *
 * Numeric and boolean values are auto-coerced.
 */
export function parseConfigOverrideToken(token) {
  const raw = String(token || '').trim();
  const eq = raw.indexOf('=');
  if (eq <= 0 || eq === raw.length - 1) {
    throw new Error(`Invalid override token: ${token}`);
  }

  const key = raw.slice(0, eq).trim();
  const valueRaw = raw.slice(eq + 1).trim();

  if (!key) throw new Error(`Invalid override key in token: ${token}`);

  const lower = valueRaw.toLowerCase();
  if (lower === 'true') return { key, value: true };
  if (lower === 'false') return { key, value: false };

  const num = Number(valueRaw);
  if (Number.isFinite(num)) return { key, value: num };

  return { key, value: valueRaw };
}

/**
 * Apply parsed overrides to a config-like object.
 *
 * Unknown keys are ignored and reported back.
 */
export function applyConfigOverrides(configObj, overrides = []) {
  const applied = [];
  const unknown = [];

  for (const item of overrides) {
    const parsed = typeof item === 'string' ? parseConfigOverrideToken(item) : item;
    if (!parsed || typeof parsed.key !== 'string') continue;

    if (!(parsed.key in configObj)) {
      unknown.push(parsed.key);
      continue;
    }

    configObj[parsed.key] = parsed.value;
    applied.push({ key: parsed.key, value: parsed.value });
  }

  return { applied, unknown };
}
