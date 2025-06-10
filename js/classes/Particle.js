import config from '../config.js';

// --- Particle Class (for fluid visualization) ---
export class Particle {
    constructor(x, y, fluidFieldRef) {
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(Math.random() * 0.2 - 0.1, Math.random() * 0.2 - 0.1);
        this.fluidField = fluidFieldRef;
        this.life = 1.0;
        this.lifeDecay = config.PARTICLE_BASE_LIFE_DECAY + Math.random() * config.PARTICLE_LIFE_DECAY_RANDOM_FACTOR;
        this.size = Math.random() * 1.5 + 0.5;
        this.isEaten = false;
    }

    update(dt) {
        if (this.isEaten) {
             // Particle is fading out after being eaten
             this.life -= 0.2 * dt * 60; // Faster fade for eaten particles
             if (this.life <= 0) {
                 // To be removed in updatePhysics
             }
            return;
        }

        if (!this.fluidField) return;
        const fluidGridX = Math.floor(this.pos.x / this.fluidField.scaleX); // Use scaleX
        const fluidGridY = Math.floor(this.pos.y / this.fluidField.scaleY); // Use scaleY
        const idx = this.fluidField.IX(fluidGridX, fluidGridY);

        const fluidVelX = this.fluidField.Vx[idx];
        const fluidVelY = this.fluidField.Vy[idx];

        this.vel.x = this.vel.x * (1.0 - config.PARTICLE_FLUID_INFLUENCE) + fluidVelX * config.PARTICLE_FLUID_INFLUENCE;
        this.vel.y = this.vel.y * (1.0 - config.PARTICLE_FLUID_INFLUENCE) + fluidVelY * config.PARTICLE_FLUID_INFLUENCE;
        this.vel.x += (Math.random() - 0.5) * 0.05;
        this.vel.y += (Math.random() - 0.5) * 0.05;

        this.pos = this.pos.add(this.vel.mul(dt * 100));

        if (config.IS_WORLD_WRAPPING) {
            if (this.pos.x < 0) this.pos.x += config.WORLD_WIDTH;
            if (this.pos.x > config.WORLD_WIDTH) this.pos.x -= config.WORLD_WIDTH;
            if (this.pos.y < 0) this.pos.y += config.WORLD_HEIGHT;
            if (this.pos.y > config.WORLD_HEIGHT) this.pos.y -= config.WORLD_HEIGHT;
        } else {
            if (this.pos.x - this.radius < 0) {
                this.pos.x = this.radius;
                this.vel.x = 0;
            } else if (this.pos.x + this.radius > config.WORLD_WIDTH) {
                this.pos.x = config.WORLD_WIDTH - this.radius;
                this.vel.x = 0;
            }
            if (this.pos.y - this.radius < 0) {
                this.pos.y = this.radius;
                this.vel.y = 0;
            } else if (this.pos.y + this.radius > config.WORLD_HEIGHT) {
                this.pos.y = config.WORLD_HEIGHT - this.radius;
                this.vel.y = 0;
            }
        }

        if (!config.IS_PARTICLE_LIFE_INFINITE) {
            this.life -= (config.PARTICLE_BASE_LIFE_DECAY + Math.random() * config.PARTICLE_LIFE_DECAY_RANDOM_FACTOR) * dt * 60;
            if (this.life <=0) {
               // To be removed in updatePhysics
            }
        } else {
            this.life = 1.0;
        }
    }

    // respawn() removed as particles are now emitted and removed

    draw(ctx) {
        const alpha = config.IS_PARTICLE_LIFE_INFINITE ? 0.7 : Math.max(0, this.life * 0.7);
        if (alpha <= 0.01 && !config.IS_PARTICLE_LIFE_INFINITE) return;
        ctx.fillStyle = `rgba(220, 220, 250, ${alpha})`;
        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
} 