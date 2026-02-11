import { getScenarioDef } from '../js/engine/scenarioDefs.mjs';

export function getScenario(name) {
  const def = getScenarioDef(name);
  return { name: def.name, ...def.nodeConfig };
}
