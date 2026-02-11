import { mulberry32 } from '../js/engine/random.mjs';

export class MiniWorld {
  constructor(config, seed = 42) {
    this.config = config;
    this.seed = seed >>> 0;
    this.rand = mulberry32(this.seed);
    this.tick = 0;
    this.time = 0;
    this.creatures = [];
    this.particles = config.particles || 0;

    for (let i = 0; i < config.creatures; i++) {
      this.creatures.push({
        id: i,
        x: this.rand() * config.world.width,
        y: this.rand() * config.world.height,
        vx: (this.rand() - 0.5) * 6,
        vy: (this.rand() - 0.5) * 6,
        energy: 80 + this.rand() * 40,
        unstable: false
      });
    }
  }

  step(dt) {
    this.tick += 1;
    this.time += dt;

    for (const c of this.creatures) {
      if (c.unstable) continue;

      // Toy fluid influence + jitter (placeholder for future extracted real logic)
      const fx = Math.sin((c.y + this.time * 4) * 0.07) * 0.4 + (this.rand() - 0.5) * 0.1;
      const fy = Math.cos((c.x + this.time * 3) * 0.06) * 0.4 + (this.rand() - 0.5) * 0.1;

      c.vx += fx * dt * 30;
      c.vy += fy * dt * 30;
      c.vx *= 0.985;
      c.vy *= 0.985;

      c.x += c.vx * dt;
      c.y += c.vy * dt;

      if (c.x < 0 || c.x > this.config.world.width) {
        c.vx *= -0.9;
        c.x = Math.max(0, Math.min(this.config.world.width, c.x));
      }
      if (c.y < 0 || c.y > this.config.world.height) {
        c.vy *= -0.9;
        c.y = Math.max(0, Math.min(this.config.world.height, c.y));
      }

      c.energy -= 0.03 + Math.abs(c.vx) * 0.001 + Math.abs(c.vy) * 0.001;
      if (c.energy <= 0) c.unstable = true;
    }
  }

  snapshot() {
    const live = this.creatures.filter(c => !c.unstable);
    return {
      tick: this.tick,
      time: Number(this.time.toFixed(3)),
      seed: this.seed,
      populations: {
        creatures: this.creatures.length,
        liveCreatures: live.length,
        particles: this.particles
      },
      sampleCreatures: live.slice(0, 5).map(c => ({
        id: c.id,
        energy: Number(c.energy.toFixed(2)),
        center: { x: Number(c.x.toFixed(2)), y: Number(c.y.toFixed(2)) },
        vel: { x: Number(c.vx.toFixed(2)), y: Number(c.vy.toFixed(2)) }
      }))
    };
  }
}
