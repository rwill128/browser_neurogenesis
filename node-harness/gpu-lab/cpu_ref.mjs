import { FluidField } from '../../js/classes/FluidField.js';

export function createCpuBackend({ size, diffusion, viscosity, dt, worldWidth = 80000, worldHeight = 64000 }) {
  const fluid = new FluidField(size, diffusion, viscosity, dt, worldWidth / size, worldHeight / size);
  return {
    name: 'cpu-ref',
    fluid,
    step(worldTick = 0) {
      fluid.step(worldTick);
    }
  };
}
