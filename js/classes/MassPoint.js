import config from '../config.js';
import { Vec2 } from "../utils.js";
import {EyeTargetType, MovementType, NodeType} from "./constants.js";

// --- MassPoint Class (Soft Body with Verlet Integration) ---
export class MassPoint {
    constructor(x, y, mass = 0.5, radius = 5, color = 'rgba(0,150,255,0.8)') {
        this.pos = new Vec2(x, y);
        this.prevPos = new Vec2(x, y);
        this.force = new Vec2(); // Initialized as a Vec2 object
        this.mass = mass;
        this.invMass = this.mass !== 0 ? 1 / this.mass : 0;
        this.radius = radius;
        this.color = color;
        this.nodeType = NodeType.EATER; // Default to a base type, will be set in createShape
        this.movementType = MovementType.NEUTRAL; // Default movement type
        this.dyeColor = [0,0,0]; // Still needed for Emitter type
        this.neuronData = null;
        this.currentExertionLevel = 0; // New: For dynamic energy costs
        this.isGrabbing = false; // New: For NN-controlled grabbing state
        this.isDesignatedEye = false; // New: To identify the creature's primary eye point
        this.canBeGrabber = false; // New: Gene, false by default
        this.eyeTargetType = EyeTargetType.PARTICLE; // New: What this eye targets
        this.seesTarget = false;       // Renamed from seesParticle
        this.nearestTargetMagnitude = 0; // Renamed from nearestParticleMagnitude
        this.nearestTargetDirection = 0; // Renamed from nearestParticleDirection
        this.maxEffectiveJetVelocity = 0; // New: For JET type
        this.sensedFluidVelocity = new Vec2(); // For JET and SWIMMER
        this.jetData = { currentMagnitude: 0, currentAngle: 0 }; // For JET type
    }
    applyForce(f) { this.force.addInPlace(f); } // Use addInPlace

    get isFixed() { // Getter for convenience, based on movementType OR if grabbing
        return this.movementType === MovementType.FIXED || this.isGrabbing;
    }

    update(dt) {
        if (this.isFixed || this.invMass === 0) { // isFixed getter now includes isGrabbing
            this.force = new Vec2();
            return;
        }

        const acceleration = this.force.mul(this.invMass);

        const tempX = this.pos.x;
        const tempY = this.pos.y;

        this.pos.x = 2 * this.pos.x - this.prevPos.x + acceleration.x * dt * dt;
        this.pos.y = 2 * this.pos.y - this.prevPos.y + acceleration.y * dt * dt;

        this.prevPos.x = tempX;
        this.prevPos.y = tempY;

        this.force = new Vec2();
    }

    draw(ctx) {
        // Draw interaction radii first (underneath the point)
        const exertion = this.currentExertionLevel || 0; // Default to 0 if undefined

        if (this.nodeType === NodeType.EATER) {
            const effectiveEatingRadiusMultiplier = config.EATING_RADIUS_MULTIPLIER_BASE + (config.EATING_RADIUS_MULTIPLIER_MAX_BONUS * exertion);
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.radius * effectiveEatingRadiusMultiplier, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 165, 0, ${0.15 + exertion * 0.2})`; // Increased base opacity
            ctx.fill();
        }
        if (this.nodeType === NodeType.PREDATOR) {
            const effectivePredationRadiusMultiplier = config.PREDATION_RADIUS_MULTIPLIER_BASE + (config.PREDATION_RADIUS_MULTIPLIER_MAX_BONUS * exertion);
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.radius * effectivePredationRadiusMultiplier, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 50, 50, ${0.15 + exertion * 0.2})`; // Increased base opacity
            ctx.fill();
        }
        if (this.nodeType === NodeType.ATTRACTOR) {
            const effectiveAttractionRadiusMultiplier = config.ATTRACTION_RADIUS_MULTIPLIER_BASE + (config.ATTRACTION_RADIUS_MULTIPLIER_MAX_BONUS * exertion);
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.radius * effectiveAttractionRadiusMultiplier, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 105, 180, ${0.1 + exertion * 0.2})`;
            ctx.fill();
        }
        if (this.nodeType === NodeType.REPULSOR) {
            const effectiveRepulsionRadiusMultiplier = config.REPULSION_RADIUS_MULTIPLIER_BASE + (config.REPULSION_RADIUS_MULTIPLIER_MAX_BONUS * exertion);
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.radius * effectiveRepulsionRadiusMultiplier, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(128, 0, 128, ${0.1 + exertion * 0.2})`;
            ctx.fill();
        }


        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, this.radius, 0, Math.PI * 2);
        let mainColor = this.isFixed ? 'rgba(255,0,0,0.9)' : this.color;
        if (this.nodeType === NodeType.NEURON) {
            mainColor = 'rgba(200, 100, 255, 0.9)';
        } else if (this.nodeType === NodeType.PREDATOR) {
            mainColor = 'rgba(255, 50, 50, 0.9)';
        } else if (this.nodeType === NodeType.SWIMMER) {
             mainColor = 'rgba(0,200,255,0.9)'; // Bright blue for Swimmer
        } else if (this.nodeType === NodeType.PHOTOSYNTHETIC) { 
            mainColor = 'rgba(60, 179, 113, 0.9)'; // MediumSeaGreen for photosynthesis
        } else if (this.nodeType === NodeType.EMITTER) {
            mainColor = 'rgba(0,255,100,0.9)';
        } else if (this.nodeType === NodeType.EATER) {
            mainColor = 'rgba(255,165,0,0.9)';
        } else if (this.nodeType === NodeType.EYE) { // New Eye color
            mainColor = 'rgba(180, 180, 250, 0.9)'; // Light purple/blue for Eye
        } else if (this.nodeType === NodeType.JET) { // New Jet color
            mainColor = 'rgba(255, 255, 100, 0.9)'; // Yellow for Jet
        } else if (this.nodeType === NodeType.ATTRACTOR) {
            mainColor = 'rgba(255, 105, 180, 0.9)'; // Hot Pink for Attractor
        } else if (this.nodeType === NodeType.REPULSOR) {
            mainColor = 'rgba(128, 0, 128, 0.9)';   // Purple for Repulsor
        }
        ctx.fillStyle = mainColor;
        ctx.fill();

        // Glow effect using shadow
        ctx.save(); // Save context state before applying shadow
        ctx.shadowColor = mainColor;
        ctx.shadowBlur = 7;
        ctx.fill(); // Re-fill to apply shadow as a glow
        ctx.restore(); // Restore context state to remove shadow for subsequent drawings


        if (this.nodeType === NodeType.NEURON) {
            ctx.beginPath(); // Inner dot for neuron
            ctx.arc(this.pos.x, this.pos.y, this.radius * 0.5, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(250, 200, 255, 0.9)';
            ctx.fill();
        }
        // ctx.closePath(); // Already closed by fill()
    }
} 