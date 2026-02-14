// Placeholder GPU backend for milestone scaffolding.
// Intentionally no-op for now; real implementation will use a compute path.

export function createGpuBackend() {
  return {
    name: 'gpu-stub',
    step(state, cfg) {
      // no-op step (placeholder)
      return state;
    },
    dispose() {}
  };
}
