export const scenarios = {
  micro_stability: {
    name: 'micro_stability',
    world: { width: 120, height: 80 },
    creatures: 8,
    particles: 200,
    dt: 1 / 30,
    steps: 300
  },
  micro_predation: {
    name: 'micro_predation',
    world: { width: 100, height: 70 },
    creatures: 12,
    particles: 260,
    dt: 1 / 30,
    steps: 300
  }
};

export function getScenario(name) {
  return scenarios[name] || scenarios.micro_stability;
}
