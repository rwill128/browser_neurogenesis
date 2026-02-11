export const DEFAULT_IMMUTABLE_CONFIG_KEYS = [
  'GRID_CELL_SIZE'
];

export function createConfigViews(config, {
  immutableKeys = DEFAULT_IMMUTABLE_CONFIG_KEYS
} = {}) {
  const constants = {};
  for (const key of immutableKeys) {
    constants[key] = config[key];
  }

  return {
    runtime: config,
    constants: Object.freeze(constants),
    immutableKeys: Object.freeze([...immutableKeys])
  };
}

export function resolveConfigViews(configOrViews) {
  if (!configOrViews) {
    throw new Error('resolveConfigViews requires a config object or config views');
  }

  if (configOrViews.runtime && configOrViews.constants) {
    return configOrViews;
  }

  return createConfigViews(configOrViews);
}
