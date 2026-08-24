export const runtimeState = {
  fluidField: null,
  softBodyPopulation: [],
  mutationStats: {}
};

export function syncRuntimeState(patch = {}) {
  Object.assign(runtimeState, patch);
}
