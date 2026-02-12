import config from '../config.js';
import {convexHull, Vec2} from '../utils.js';
import { NodeType, RLRewardStrategy, RLAlgorithmType, EyeTargetType, MovementType } from './constants.js';
import {MassPoint} from "./MassPoint.js";
import {Spring} from "./Spring.js";
import {Brain} from "./Brain.js";
import { runtimeState } from "../engine/runtimeState.js";
import {
    computeGrowthPopulationThrottle,
    computeGrowthSizeCostMultiplier
} from '../engine/growthControls.mjs';
import {
    computeDensityFertilityScale,
    evaluateResourceCoupling,
    applyReproductionResourceDebit
} from '../engine/reproductionControls.mjs';

const DEFAULT_GROWTH_NODE_TYPES = [
    NodeType.PREDATOR,
    NodeType.EATER,
    NodeType.PHOTOSYNTHETIC,
    NodeType.NEURON,
    NodeType.EMITTER,
    NodeType.SWIMMER,
    NodeType.EYE,
    NodeType.JET,
    NodeType.ATTRACTOR,
    NodeType.REPULSOR
];

const DYE_AFFINITY_KEYS = ['EATER', 'PREDATOR', 'PHOTOSYNTHETIC', 'SWIMMER', 'JET', 'EMITTER'];

function clampNumber(value, lo, hi) {
    const n = Number(value);
    if (!Number.isFinite(n)) return lo;
    return Math.max(lo, Math.min(hi, n));
}

function computeHueFromRgb(r, g, b) {
    const rn = clampNumber(r, 0, 255) / 255;
    const gn = clampNumber(g, 0, 255) / 255;
    const bn = clampNumber(b, 0, 255) / 255;

    const max = Math.max(rn, gn, bn);
    const min = Math.min(rn, gn, bn);
    const d = max - min;
    if (d <= 1e-8) return 0;

    let h = 0;
    if (max === rn) h = ((gn - bn) / d) % 6;
    else if (max === gn) h = ((bn - rn) / d) + 2;
    else h = ((rn - gn) / d) + 4;

    h /= 6;
    if (h < 0) h += 1;
    return h;
}

function circularHueDistance(a, b) {
    const da = Math.abs((Number(a) || 0) - (Number(b) || 0));
    return Math.min(da, 1 - da);
}

// --- SoftBody Class ---
export class SoftBody {
    constructor(id, initialX, initialY, creationData = null, isBlueprint = false) {
        this.id = id;
        this.massPoints = [];
        this.springs = [];
        this.nutrientField = null;
        this.lightField = null;
        this.particles = null;
        this.spatialGrid = null;

        // Genetic blueprint
        this.blueprintPoints = []; // Array of { relX, relY, radius, mass, nodeType, movementType, dyeColor, canBeGrabber, neuronDataBlueprint, activationIntervalGene, predatorRadiusGene }
        this.blueprintSprings = []; // Array of { p1Index, p2Index, restLength, isRigid, activationIntervalGene } (indices refer to blueprintPoints)

        this.isUnstable = false;
        // First fatal condition assigned during lifetime (used by instability telemetry).
        this.unstableReason = null;
        this.unstableReasonDetails = null;
        this.ticksSinceBirth = 0;
        this.absoluteAgeTicks = 0;
        this.canReproduce = false;
        this.shapeType = creationData ? creationData.shapeType : Math.floor(Math.random() * 3);
        this.justReproduced = false; // New: Flag for reproduction reward

        // Lifecycle provenance metadata for instability diagnostics.
        this.birthOrigin = 'unknown';
        this.parentBodyId = null;
        this.lineageRootId = Number.isFinite(Number(id)) ? Number(id) : null;
        this.generation = 0;
        this.reproductionEventsCompleted = 0;
        this.ticksSinceLastReproduction = null;

        this.energyGainedFromPhotosynthesis = 0;
        this.energyGainedFromEating = 0;
        this.energyGainedFromPredation = 0;

        // New: Energy cost accumulators
        this.energyCostFromBaseNodes = 0;
        this.energyCostFromEmitterNodes = 0;
        this.energyCostFromEaterNodes = 0;
        this.energyCostFromPredatorNodes = 0;
        this.energyCostFromNeuronNodes = 0;
        this.energyCostFromSwimmerNodes = 0;
        this.energyCostFromPhotosyntheticNodes = 0;
        this.energyCostFromGrabbingNodes = 0;
        this.energyCostFromEyeNodes = 0;
        this.energyCostFromJetNodes = 0;
        this.energyCostFromAttractorNodes = 0;
        this.energyCostFromRepulsorNodes = 0;

        this.currentMaxEnergy = config.BASE_MAX_CREATURE_ENERGY; // Initial placeholder
        this.blueprintRadius = 0; // New: Approximate radius based on blueprint points

        // Node counts for brain initialization optimization
        this.numEmitterNodes = 0;
        this.numSwimmerNodes = 0;
        this.numEaterNodes = 0;
        this.numPredatorNodes = 0;
        this.numEyeNodes = 0;
        this.numJetNodes = 0;
        this.numPotentialGrabberNodes = 0;
        this.numAttractorNodes = 0;
        this.numRepulsorNodes = 0;

        this.failedReproductionCooldown = 0; // New: Cooldown after a failed reproduction attempt
        this.energyGainedFromPhotosynthesisThisTick = 0; // New: Photosynthesis gain in the current tick

        // Growth/development state (new): creatures can probabilistically add nodes over lifetime.
        this.growthGenome = null;
        this.growthCooldownRemaining = 0;
        this.growthEventsCompleted = 0;
        this.growthNodesAdded = 0;
        this.totalGrowthEnergySpent = 0;
        this.growthSuppressedByPopulation = 0;
        this.growthSuppressedByEnergy = 0;
        this.growthSuppressedByCooldown = 0;
        // Growth telemetry (new): make non-growth outcomes observable in soak/UI diagnostics.
        this.growthSuppressedByMaxPoints = 0;
        this.growthSuppressedByNoCapacity = 0;
        this.growthSuppressedByChanceRoll = 0;
        this.growthSuppressedByPlacement = 0;
        this.nnTopologyVersion = 0;
        this.rlBufferResetsDueToTopology = 0;

        // Reproduction control telemetry (new): observability for density/resource coupling.
        this.reproductionSuppressedByDensity = 0;
        this.reproductionSuppressedByResources = 0;
        this.reproductionSuppressedByFertilityRoll = 0;
        this.reproductionSuppressedByDye = 0;
        this.reproductionResourceDebitApplied = 0;

        // Heritable dye ecology genes/state (new): color niche + response profile.
        this.dyePreferredHue = Math.random();
        this.dyeHueTolerance = 0.2;
        this.dyeResponseGain = 1.0;
        this.dyeResponseSign = 1;
        this.dyeNodeTypeAffinity = {};
        this.lastDyeEcologyState = null;

        // Growth telemetry (new): track dye-driven suppression independently.
        this.growthSuppressedByDye = 0;

        // Actuation telemetry (new): evaluative throttling observability.
        this.actuationEvaluations = 0;
        this.actuationSkips = 0;
        this.actuationEvaluationsByNodeType = {};
        this.actuationSkipsByNodeType = {};
        this.actuationIntervalSamples = 0;
        this.actuationIntervalTotal = 0;
        this.energyCostFromActuationUpkeep = 0;
        this.energyCostFromActuationEvents = 0;

        // Initialize heritable/mutable properties
        if (isBlueprint && creationData) {
            // --- CREATION FROM IMPORTED BLUEPRINT ---
            const blueprint = creationData;
            this.birthOrigin = 'imported_blueprint';
            this.parentBodyId = null;
            this.lineageRootId = Number.isFinite(Number(this.id)) ? Number(this.id) : null;
            this.generation = 0;
            this.reproductionEventsCompleted = 0;
            this.ticksSinceLastReproduction = null;
            this.stiffness = blueprint.stiffness;
            this.springDamping = blueprint.springDamping;
            this.motorImpulseInterval = blueprint.motorImpulseInterval;
            this.motorImpulseMagnitudeCap = blueprint.motorImpulseMagnitudeCap;
            this.emitterStrength = blueprint.emitterStrength;
            this.emitterDirection = new Vec2(blueprint.emitterDirection.x, blueprint.emitterDirection.y);
            this.numOffspring = blueprint.numOffspring;
            this.offspringSpawnRadius = blueprint.offspringSpawnRadius;
            this.pointAddChance = blueprint.pointAddChance;
            this.springConnectionRadius = blueprint.springConnectionRadius;
            this.jetMaxVelocityGene = blueprint.jetMaxVelocityGene;
            this.reproductionEnergyThreshold = blueprint.reproductionEnergyThreshold;
            this.reproductionCooldownGene = blueprint.reproductionCooldownGene;
            this.defaultActivationPattern = blueprint.defaultActivationPattern;
            this.defaultActivationLevel = blueprint.defaultActivationLevel;
            this.defaultActivationPeriod = blueprint.defaultActivationPeriod;
            this.defaultActivationPhaseOffset = blueprint.defaultActivationPhaseOffset;
            this.rlAlgorithmType = blueprint.rlAlgorithmType;
            this.rewardStrategy = blueprint.rewardStrategy;
            this.growthGenome = this._sanitizeGrowthGenome(blueprint.growthGenome || this._createRandomGrowthGenome());
            this.dyePreferredHue = Number(blueprint.dyePreferredHue);
            this.dyeHueTolerance = Number(blueprint.dyeHueTolerance);
            this.dyeResponseGain = Number(blueprint.dyeResponseGain);
            this.dyeResponseSign = Number(blueprint.dyeResponseSign);
            this.dyeNodeTypeAffinity = blueprint.dyeNodeTypeAffinity || {};
            this._sanitizeDyeEcologyGenes();

            // Directly use the blueprint's structure
            this.blueprintPoints = JSON.parse(JSON.stringify(blueprint.blueprintPoints));
            this.blueprintSprings = JSON.parse(JSON.stringify(blueprint.blueprintSprings));
            this.blueprintPoints.forEach((bp) => {
                bp.activationIntervalGene = this._sanitizeActivationIntervalGene(
                    bp.activationIntervalGene ?? this._randomActivationIntervalGene()
                );
                bp.predatorRadiusGene = this._sanitizePredatorRadiusGene(
                    bp.predatorRadiusGene ?? this._randomPredatorRadiusGene()
                );
            });
            this.blueprintSprings.forEach((bs) => {
                bs.activationIntervalGene = this._sanitizeActivationIntervalGene(
                    bs.activationIntervalGene ?? this._randomActivationIntervalGene()
                );
            });
            this._instantiatePhenotypeFromBlueprint(initialX, initialY);

        } else {
            // --- CREATION FROM PARENT (REPRODUCTION) OR FROM SCRATCH ---
            const parentBody = creationData; // In this case, creationData is the parentBody
            if (parentBody) {
                this.birthOrigin = 'reproduction_offspring';
                this.parentBodyId = Number.isFinite(Number(parentBody.id)) ? Number(parentBody.id) : null;
                this.lineageRootId = Number.isFinite(Number(parentBody.lineageRootId))
                    ? Number(parentBody.lineageRootId)
                    : (Number.isFinite(Number(parentBody.id)) ? Number(parentBody.id) : this.lineageRootId);
                this.generation = Math.max(0, Math.floor(Number(parentBody.generation) || 0) + 1);
                this.reproductionEventsCompleted = 0;
                this.ticksSinceLastReproduction = null;

                this.stiffness = parentBody.stiffness * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.stiffness !== parentBody.stiffness) runtimeState.mutationStats.springStiffness++;
                this.springDamping = parentBody.springDamping * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.springDamping !== parentBody.springDamping) runtimeState.mutationStats.springDamping++;

                let oldMotorInterval = parentBody.motorImpulseInterval;
                this.motorImpulseInterval = parentBody.motorImpulseInterval * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (Math.floor(this.motorImpulseInterval) !== Math.floor(oldMotorInterval)) runtimeState.mutationStats.motorInterval++;

                let oldMotorCap = parentBody.motorImpulseMagnitudeCap;
                this.motorImpulseMagnitudeCap = parentBody.motorImpulseMagnitudeCap * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.motorImpulseMagnitudeCap !== oldMotorCap) runtimeState.mutationStats.motorCap++;

                let oldEmitterStrength = parentBody.emitterStrength;
                this.emitterStrength = parentBody.emitterStrength * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.emitterStrength !== oldEmitterStrength) runtimeState.mutationStats.emitterStrength++;

                let oldJetMaxVel = parentBody.jetMaxVelocityGene;
                this.jetMaxVelocityGene = parentBody.jetMaxVelocityGene * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.jetMaxVelocityGene !== oldJetMaxVel) runtimeState.mutationStats.jetMaxVelocityGene++;

                let offspringNumChange = (Math.random() < Math.max(0, Math.min(1, config.MUTATION_CHANCE_BOOL * config.GLOBAL_MUTATION_RATE_MODIFIER))) ? (Math.random() < 0.5 ? -1 : 1) : 0;
                this.numOffspring = parentBody.numOffspring + offspringNumChange;
                if (offspringNumChange !== 0) runtimeState.mutationStats.numOffspring++;

                let oldOffspringSpawnRadius = parentBody.offspringSpawnRadius;
                this.offspringSpawnRadius = parentBody.offspringSpawnRadius * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.5));
                if (this.offspringSpawnRadius !== oldOffspringSpawnRadius) runtimeState.mutationStats.offspringSpawnRadius++;

                let oldPointAddChance = parentBody.pointAddChance;
                this.pointAddChance = parentBody.pointAddChance * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 2));
                if (this.pointAddChance !== oldPointAddChance) runtimeState.mutationStats.pointAddChanceGene++;

                let oldSpringConnectionRadius = parentBody.springConnectionRadius;
                this.springConnectionRadius = parentBody.springConnectionRadius * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER));
                if (this.springConnectionRadius !== oldSpringConnectionRadius) runtimeState.mutationStats.springConnectionRadiusGene++;

                if (parentBody.emitterDirection) {
                    const oldEmitterDirX = parentBody.emitterDirection.x;
                    const angleMutation = (Math.random() - 0.5) * Math.PI * 0.2 * config.GLOBAL_MUTATION_RATE_MODIFIER;
                    const cosA = Math.cos(angleMutation);
                    const sinA = Math.sin(angleMutation);
                    this.emitterDirection = new Vec2(parentBody.emitterDirection.x * cosA - parentBody.emitterDirection.y * sinA, parentBody.emitterDirection.x * sinA + parentBody.emitterDirection.y * cosA).normalize();
                    if (this.emitterDirection.x !== oldEmitterDirX) runtimeState.mutationStats.emitterDirection++; // Simplified check
                } else {
                    this.emitterDirection = new Vec2(Math.random()*2-1, Math.random()*2-1).normalize();
                    console.warn(`Parent body ${parentBody.id} was missing emitterDirection. Offspring ${this.id} gets random emitterDirection.`);
                    runtimeState.mutationStats.emitterDirection++; // Count as a change if parent was missing it
                }

                let oldReproThreshold = parentBody.reproductionEnergyThreshold;
                this.reproductionEnergyThreshold = parentBody.reproductionEnergyThreshold; 
                // Mutation of reproductionEnergyThreshold happens later, after currentMaxEnergy is set for the offspring

                // Inherit and mutate activation pattern properties
                if (Math.random() < config.ACTIVATION_PATTERN_MUTATION_CHANCE) {
                    const patterns = Object.values(config.ActivationPatternType);
                    this.defaultActivationPattern = patterns[Math.floor(Math.random() * patterns.length)];
                } else {
                    this.defaultActivationPattern = parentBody.defaultActivationPattern;
                }
                this.defaultActivationLevel = parentBody.defaultActivationLevel * (1 + (Math.random() - 0.5) * 2 * config.ACTIVATION_PARAM_MUTATION_MAGNITUDE);
                this.defaultActivationPeriod = parentBody.defaultActivationPeriod * (1 + (Math.random() - 0.5) * 2 * config.ACTIVATION_PARAM_MUTATION_MAGNITUDE);
                this.defaultActivationPhaseOffset = parentBody.defaultActivationPhaseOffset + (Math.random() - 0.5) * (parentBody.defaultActivationPeriod * 0.2); 

                // Inherit/Mutate Reward Strategy
                if (Math.random() < config.RLRewardStrategy_MUTATION_CHANCE) {
                    const strategies = Object.values(RLRewardStrategy);
                    let newStrategy = strategies[Math.floor(Math.random() * strategies.length)];
                    if (newStrategy !== parentBody.rewardStrategy) {
                        this.rewardStrategy = newStrategy;
                        runtimeState.mutationStats.rewardStrategyChange++;
                    } else {
                        if (strategies.length > 1) {
                            let tempStrategies = strategies.filter(s => s !== parentBody.rewardStrategy);
                            if (tempStrategies.length > 0) {
                                this.rewardStrategy = tempStrategies[Math.floor(Math.random() * tempStrategies.length)];
                                runtimeState.mutationStats.rewardStrategyChange++; 
                            } else {
                                this.rewardStrategy = parentBody.rewardStrategy;
                            }
                        } else {
                            this.rewardStrategy = parentBody.rewardStrategy;
                        }
                    }
                } else {
                    this.rewardStrategy = parentBody.rewardStrategy;
                }
                // RL Algorithm type inheritance (can add mutation if needed)
                this.rlAlgorithmType = parentBody.rlAlgorithmType || RLAlgorithmType.REINFORCE;

                // Inherit and mutate reproductionCooldownGene
                this.reproductionCooldownGene = parentBody.reproductionCooldownGene * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.2));
                this.reproductionCooldownGene = Math.max(50, Math.min(Math.floor(this.reproductionCooldownGene), 20000)); // Clamp
                if (this.reproductionCooldownGene !== parentBody.reproductionCooldownGene) {
                    runtimeState.mutationStats.reproductionCooldownGene = (runtimeState.mutationStats.reproductionCooldownGene || 0) + 1;
                }

                const growthMutation = this._mutateGrowthGenomeFromParent(parentBody.growthGenome || null);
                this.growthGenome = growthMutation.genome;
                if (growthMutation.didMutate) {
                    runtimeState.mutationStats.growthGenomeMutations = (runtimeState.mutationStats.growthGenomeMutations || 0) + 1;
                }

                this._inheritAndMutateDyeEcologyGenes(parentBody);

            } else {
                // Initial defaults for brand new creatures
                const edgeStiffnessMin = Number.isFinite(Number(config.NEW_EDGE_STIFFNESS_MIN)) ? Number(config.NEW_EDGE_STIFFNESS_MIN) : 500;
                const edgeStiffnessMax = Number.isFinite(Number(config.NEW_EDGE_STIFFNESS_MAX))
                    ? Math.max(edgeStiffnessMin, Number(config.NEW_EDGE_STIFFNESS_MAX))
                    : 3000;
                const edgeDampingMin = Number.isFinite(Number(config.NEW_EDGE_DAMPING_MIN)) ? Number(config.NEW_EDGE_DAMPING_MIN) : 5;
                const edgeDampingMax = Number.isFinite(Number(config.NEW_EDGE_DAMPING_MAX))
                    ? Math.max(edgeDampingMin, Number(config.NEW_EDGE_DAMPING_MAX))
                    : 25;
                const connectionRadiusMin = Number.isFinite(Number(config.INITIAL_SPRING_CONNECTION_RADIUS_MIN))
                    ? Number(config.INITIAL_SPRING_CONNECTION_RADIUS_MIN)
                    : 40;
                const connectionRadiusMax = Number.isFinite(Number(config.INITIAL_SPRING_CONNECTION_RADIUS_MAX))
                    ? Math.max(connectionRadiusMin, Number(config.INITIAL_SPRING_CONNECTION_RADIUS_MAX))
                    : 80;

                this.stiffness = edgeStiffnessMin + Math.random() * (edgeStiffnessMax - edgeStiffnessMin);
                this.springDamping = edgeDampingMin + Math.random() * (edgeDampingMax - edgeDampingMin);
                this.motorImpulseInterval = 30 + Math.floor(Math.random() * 90);
                this.motorImpulseMagnitudeCap = 0.5 + Math.random() * 2.0;
                this.emitterStrength = 0.2 + Math.random() * 1.0;
                this.emitterDirection = new Vec2(Math.random()*2-1, Math.random()*2-1).normalize();
                this.numOffspring = 1 + Math.floor(Math.random() * 3);
                this.offspringSpawnRadius = 50 + Math.random() * 50;
                this.pointAddChance = 0.02 + Math.random() * 0.06;
                this.springConnectionRadius = connectionRadiusMin + Math.random() * (connectionRadiusMax - connectionRadiusMin);
                this.jetMaxVelocityGene = config.JET_MAX_VELOCITY_GENE_DEFAULT * (0.8 + Math.random() * 0.4);
                this.reproductionEnergyThreshold = config.BASE_MAX_CREATURE_ENERGY; // Will be refined based on actual max energy

                // Default Activation Pattern Properties for new creature
                const patterns = Object.values(config.ActivationPatternType);
                this.defaultActivationPattern = patterns[Math.floor(Math.random() * patterns.length)];
                this.defaultActivationLevel = config.DEFAULT_ACTIVATION_LEVEL_MIN + Math.random() * (config.DEFAULT_ACTIVATION_LEVEL_MAX - config.DEFAULT_ACTIVATION_LEVEL_MIN);
                this.defaultActivationPeriod = config.DEFAULT_ACTIVATION_PERIOD_MIN_TICKS + Math.floor(Math.random() * (config.DEFAULT_ACTIVATION_PERIOD_MAX_TICKS - config.DEFAULT_ACTIVATION_PERIOD_MIN_TICKS + 1));
                this.defaultActivationPhaseOffset = Math.random() * this.defaultActivationPeriod;

                // Default RL Algorithm for brand new creatures
                this.rlAlgorithmType = RLAlgorithmType.REINFORCE; 

                // New: Default Reward Strategy for brand new creatures
                const strategies = Object.values(RLRewardStrategy);
                this.rewardStrategy = strategies[Math.floor(Math.random() * strategies.length)];

                // Initialize reproductionCooldownGene for new creatures (configurable range).
                const reproCooldownMin = Number.isFinite(Number(config.INITIAL_REPRODUCTION_COOLDOWN_GENE_MIN))
                    ? Math.max(1, Math.floor(Number(config.INITIAL_REPRODUCTION_COOLDOWN_GENE_MIN)))
                    : 100;
                const reproCooldownMax = Number.isFinite(Number(config.INITIAL_REPRODUCTION_COOLDOWN_GENE_MAX))
                    ? Math.max(reproCooldownMin, Math.floor(Number(config.INITIAL_REPRODUCTION_COOLDOWN_GENE_MAX)))
                    : 5000;
                this.reproductionCooldownGene = reproCooldownMin + Math.floor(Math.random() * (reproCooldownMax - reproCooldownMin + 1));
                this.growthGenome = this._createRandomGrowthGenome();
                this._initializeRandomDyeEcologyGenes();
            }

            // Clamp activation properties after they've been set/inherited/mutated
            this.defaultActivationLevel = Math.max(config.DEFAULT_ACTIVATION_LEVEL_MIN * 0.1, Math.min(this.defaultActivationLevel, config.DEFAULT_ACTIVATION_LEVEL_MAX * 2.0)); // Wider clamping for more variance
            this.defaultActivationPeriod = Math.max(config.DEFAULT_ACTIVATION_PERIOD_MIN_TICKS * 0.25, Math.min(this.defaultActivationPeriod, config.DEFAULT_ACTIVATION_PERIOD_MAX_TICKS * 2.0));
            this.defaultActivationPeriod = Math.floor(this.defaultActivationPeriod);
            if (this.defaultActivationPeriod === 0) this.defaultActivationPeriod = 1; // Avoid division by zero
            if (this.defaultActivationPhaseOffset < 0) this.defaultActivationPhaseOffset += this.defaultActivationPeriod;
            this.defaultActivationPhaseOffset %= this.defaultActivationPeriod;

            // Clamp other properties (already existing logic)
            this.stiffness = Math.max(100, Math.min(this.stiffness, 10000));
            this.springDamping = Math.max(0.1, Math.min(this.springDamping, 50));
            this.motorImpulseInterval = Math.max(10, Math.floor(this.motorImpulseInterval));
            this.motorImpulseMagnitudeCap = Math.max(0, Math.min(this.motorImpulseMagnitudeCap, 5.0));
            this.emitterStrength = Math.max(0, Math.min(this.emitterStrength, 3.0));
            this.numOffspring = Math.max(1, Math.min(this.numOffspring, 5));
            this.offspringSpawnRadius = Math.max(20, Math.min(this.offspringSpawnRadius, 150));
            this.pointAddChance = Math.max(0, Math.min(0.5, this.pointAddChance));
            const springConnectionRadiusMin = Number.isFinite(Number(config.SPRING_CONNECTION_RADIUS_MIN))
                ? Number(config.SPRING_CONNECTION_RADIUS_MIN)
                : 10;
            const springConnectionRadiusMax = Number.isFinite(Number(config.SPRING_CONNECTION_RADIUS_MAX))
                ? Math.max(springConnectionRadiusMin, Number(config.SPRING_CONNECTION_RADIUS_MAX))
                : 100;
            this.springConnectionRadius = Math.max(springConnectionRadiusMin, Math.min(this.springConnectionRadius, springConnectionRadiusMax));
            this.jetMaxVelocityGene = Math.max(0.1, Math.min(this.jetMaxVelocityGene, 50.0));
            this.growthGenome = this._sanitizeGrowthGenome(this.growthGenome || this._createRandomGrowthGenome());
            this._sanitizeDyeEcologyGenes();

            this.fluidEntrainment = config.BODY_FLUID_ENTRAINMENT_FACTOR;
            this.fluidCurrentStrength = config.FLUID_CURRENT_STRENGTH_ON_BODY;
            this.bodyPushStrength = config.SOFT_BODY_PUSH_STRENGTH;

            // 1. Create shape (points and initial springs)
            // If created from blueprint, this is already done.
            if (!isBlueprint) {
                this.createShape(initialX, initialY, parentBody);
            }

            // 2. Body Scale Mutation (if offspring)
            const parentBodyForMutation = isBlueprint ? null : creationData;
            if (parentBodyForMutation && Math.random() < config.BODY_SCALE_MUTATION_CHANCE) {
                const scaleFactor = 1.0 + (Math.random() - 0.5) * 2 * config.BODY_SCALE_MUTATION_MAGNITUDE;
                if (scaleFactor > 0.1 && Math.abs(scaleFactor - 1.0) > 0.001) { // Check if it actually scaled
                    this.springs.forEach(spring => {
                        spring.restLength *= scaleFactor;
                        spring.restLength = Math.max(1, spring.restLength); 
                    });
                    this.massPoints.forEach(point => {
                        point.radius *= scaleFactor;
                        point.radius = Math.max(0.5, point.radius); 
                    });
                    this.offspringSpawnRadius *= scaleFactor;
                    this.offspringSpawnRadius = Math.max(10, this.offspringSpawnRadius); 
                    runtimeState.mutationStats.bodyScale++;
                }
            }

            // 3. Calculate dynamic max energy based on final point count
            this.calculateCurrentMaxEnergy();

            // Calculate effective reproduction cooldown based on gene and point count
            this.effectiveReproductionCooldown = Math.floor(this.reproductionCooldownGene * (1 + (0.2 * Math.max(0, this.massPoints.length - 1))));
            if (this.massPoints.length === 0) { // Safety for empty body after blueprint instantiation (should not happen ideally)
                 this.effectiveReproductionCooldown = Math.floor(this.reproductionCooldownGene);
            }

            // 4. Set initial creature energy
            this.creatureEnergy = this.currentMaxEnergy * config.OFFSPRING_INITIAL_ENERGY_SHARE;

            let oldReproThresholdForStat = this.reproductionEnergyThreshold; // Capture before mutation
            if (!isBlueprint && parentBodyForMutation) {
                this.reproductionEnergyThreshold = this.reproductionEnergyThreshold * (1 + (Math.random() - 0.5) * 2 * (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.2));
            } else if (!parentBodyForMutation) { // This is for brand new from scratch or imported
                this.reproductionEnergyThreshold = this.currentMaxEnergy * (0.75 + Math.random() * 0.2);
            }
            // For blueprints, the threshold is already set, but we still clamp it.
            this.reproductionEnergyThreshold = Math.max(this.currentMaxEnergy * 0.05, Math.min(this.reproductionEnergyThreshold, this.currentMaxEnergy));
            this.reproductionEnergyThreshold = Math.round(this.reproductionEnergyThreshold);
            if (parentBodyForMutation && this.reproductionEnergyThreshold !== oldReproThresholdForStat) { // Only count if from parent and changed
                runtimeState.mutationStats.reproductionEnergyThreshold++;
            }

        }

        this.growthGenome = this._sanitizeGrowthGenome(this.growthGenome || this._createRandomGrowthGenome());

        this.primaryEyePoint = null; // New: For the creature's main eye
        if (this.massPoints.length > 0) {
            // Attempt to find an existing EYE node to designate as primary
            for (const point of this.massPoints) {
                if (point.nodeType === NodeType.EYE) {
                    this.primaryEyePoint = point;
                    point.isDesignatedEye = true;
                    break; 
                }
            }
            // If no EYE node found, and we want to ensure one, we could designate/change one here.
            // For now, it's only set if an EYE node is already present (e.g., from mutation or initial gen).
            // If still null, the NN input for eye will be default/neutral.
        }

        this.brain = new Brain(this); 

        // Temporary vectors for calculations to reduce allocations
        this._tempVec1 = new Vec2();
        this._tempVec2 = new Vec2();
    }

    calculateCurrentMaxEnergy() {
        if (this.massPoints.length === 0) {
            this.currentMaxEnergy = 0; 
        } else {
            this.currentMaxEnergy = this.massPoints.length * config.ENERGY_PER_MASS_POINT_BONUS;
            // Ensure currentMaxEnergy is at least 1 if points exist and ENERGY_PER_MASS_POINT_BONUS might be zero or very small.
            if (this.currentMaxEnergy < 1) { 
                this.currentMaxEnergy = 1;
            }
        }
    }

    _sanitizeActivationIntervalGene(value) {
        const min = Math.max(1, Math.floor(config.ACTUATION_INTERVAL_GENE_MIN || 1));
        const max = Math.max(min, Math.floor(config.ACTUATION_INTERVAL_GENE_MAX || min));
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return min;
        return Math.max(min, Math.min(max, Math.round(parsed)));
    }

    _randomActivationIntervalGene() {
        const min = Math.max(1, Math.floor(config.ACTUATION_INTERVAL_GENE_MIN || 1));
        const max = Math.max(min, Math.floor(config.ACTUATION_INTERVAL_GENE_MAX || min));
        return min + Math.floor(Math.random() * (max - min + 1));
    }

    _mutateActivationIntervalGene(parentGene) {
        const gene = this._sanitizeActivationIntervalGene(parentGene);
        if (Math.random() >= Math.max(0, Math.min(1, Number(config.ACTUATION_INTERVAL_GENE_MUTATION_CHANCE) || 0))) {
            return { gene, didMutate: false };
        }

        const step = Math.max(1, Math.floor(Number(config.ACTUATION_INTERVAL_GENE_MUTATION_STEP) || 1));
        const direction = Math.random() < 0.5 ? -1 : 1;
        const mutated = this._sanitizeActivationIntervalGene(gene + (direction * step));
        return { gene: mutated, didMutate: mutated !== gene };
    }

    _sanitizePredatorRadiusGene(value) {
        const min = Math.max(0.05, Number(config.PREDATOR_RADIUS_GENE_MIN) || 0.2);
        const max = Math.max(min, Number(config.PREDATOR_RADIUS_GENE_MAX) || min);
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return min;
        return Math.max(min, Math.min(max, parsed));
    }

    _randomPredatorRadiusGene() {
        const min = Math.max(0.05, Number(config.PREDATOR_RADIUS_GENE_MIN) || 0.2);
        const max = Math.max(min, Number(config.PREDATOR_RADIUS_GENE_MAX) || min);
        return min + Math.random() * (max - min);
    }

    _mutatePredatorRadiusGene(parentGene) {
        const gene = this._sanitizePredatorRadiusGene(parentGene);
        const chance = Math.max(0, Math.min(1, Number(config.PREDATOR_RADIUS_GENE_MUTATION_CHANCE) || 0));
        if (Math.random() >= chance) {
            return { gene, didMutate: false };
        }

        const magnitude = Math.max(0, Number(config.PREDATOR_RADIUS_GENE_MUTATION_MAGNITUDE) || 0);
        const jitter = (Math.random() - 0.5) * 2 * magnitude;
        const mutated = this._sanitizePredatorRadiusGene(gene * (1 + jitter));
        return { gene: mutated, didMutate: Math.abs(mutated - gene) > 1e-6 };
    }

    _computePredatorRadiusMultiplier(point) {
        const gene = this._sanitizePredatorRadiusGene(point?.predatorRadiusGene ?? this._randomPredatorRadiusGene());
        return gene;
    }

    _computePredatorRadius(point) {
        const exertion = Math.max(0, Math.min(1, Number(point?.currentExertionLevel) || 0));
        const multiplier = this._computePredatorRadiusMultiplier(point);
        return Math.max(0, Number(point?.radius) || 0) * multiplier * exertion;
    }

    _createRandomDyeAffinityMap() {
        const out = {};
        for (const key of DYE_AFFINITY_KEYS) {
            out[key] = 0.6 + Math.random() * 0.8;
        }
        return out;
    }

    _sanitizeDyeAffinityMap(rawMap = {}) {
        const out = {};
        for (const key of DYE_AFFINITY_KEYS) {
            const raw = Number(rawMap?.[key]);
            out[key] = Number.isFinite(raw) ? Math.max(0.05, Math.min(3.0, raw)) : 1.0;
        }
        return out;
    }

    _initializeRandomDyeEcologyGenes() {
        this.dyePreferredHue = Math.random();
        this.dyeHueTolerance = (Number(config.DYE_RECEPTOR_HUE_TOLERANCE_MIN) || 0.04)
            + Math.random() * Math.max(0.001, (Number(config.DYE_RECEPTOR_HUE_TOLERANCE_MAX) || 0.35) - (Number(config.DYE_RECEPTOR_HUE_TOLERANCE_MIN) || 0.04));
        this.dyeResponseGain = (Number(config.DYE_RECEPTOR_RESPONSE_GAIN_MIN) || 0.25)
            + Math.random() * Math.max(0.001, (Number(config.DYE_RECEPTOR_RESPONSE_GAIN_MAX) || 1.5) - (Number(config.DYE_RECEPTOR_RESPONSE_GAIN_MIN) || 0.25));
        this.dyeResponseSign = Math.random() < 0.5 ? -1 : 1;
        this.dyeNodeTypeAffinity = this._createRandomDyeAffinityMap();
        this._sanitizeDyeEcologyGenes();
    }

    _sanitizeDyeEcologyGenes() {
        const tolMin = Math.max(0.005, Number(config.DYE_RECEPTOR_HUE_TOLERANCE_MIN) || 0.04);
        const tolMax = Math.max(tolMin, Number(config.DYE_RECEPTOR_HUE_TOLERANCE_MAX) || 0.35);
        const gainMin = Math.max(0, Number(config.DYE_RECEPTOR_RESPONSE_GAIN_MIN) || 0.25);
        const gainMax = Math.max(gainMin, Number(config.DYE_RECEPTOR_RESPONSE_GAIN_MAX) || 1.5);

        this.dyePreferredHue = ((Number(this.dyePreferredHue) || 0) % 1 + 1) % 1;
        this.dyeHueTolerance = clampNumber(this.dyeHueTolerance, tolMin, tolMax);
        this.dyeResponseGain = clampNumber(this.dyeResponseGain, gainMin, gainMax);
        this.dyeResponseSign = Number(this.dyeResponseSign) < 0 ? -1 : 1;
        this.dyeNodeTypeAffinity = this._sanitizeDyeAffinityMap(this.dyeNodeTypeAffinity || {});
    }

    _inheritAndMutateDyeEcologyGenes(parentBody) {
        if (!parentBody) {
            this._initializeRandomDyeEcologyGenes();
            return;
        }

        this.dyePreferredHue = Number(parentBody.dyePreferredHue);
        this.dyeHueTolerance = Number(parentBody.dyeHueTolerance);
        this.dyeResponseGain = Number(parentBody.dyeResponseGain);
        this.dyeResponseSign = Number(parentBody.dyeResponseSign) < 0 ? -1 : 1;
        this.dyeNodeTypeAffinity = this._sanitizeDyeAffinityMap(parentBody.dyeNodeTypeAffinity || {});

        const mutationChance = Math.max(0, Math.min(1, Number(config.DYE_RECEPTOR_MUTATION_CHANCE) || 0));
        const mutationMagnitude = Math.max(0, Number(config.DYE_RECEPTOR_MUTATION_MAGNITUDE) || 0);

        if (Math.random() < mutationChance) {
            this.dyePreferredHue = ((this.dyePreferredHue + ((Math.random() - 0.5) * 2 * mutationMagnitude)) % 1 + 1) % 1;
            this._bumpMutationStat('dyePreferredHue');
        }
        if (Math.random() < mutationChance) {
            this.dyeHueTolerance += (Math.random() - 0.5) * 2 * mutationMagnitude * 0.2;
            this._bumpMutationStat('dyeHueTolerance');
        }
        if (Math.random() < mutationChance) {
            this.dyeResponseGain += (Math.random() - 0.5) * 2 * mutationMagnitude;
            this._bumpMutationStat('dyeResponseGain');
        }
        if (Math.random() < mutationChance * 0.5) {
            this.dyeResponseSign = this.dyeResponseSign < 0 ? 1 : -1;
            this._bumpMutationStat('dyeResponseSign');
        }

        for (const key of DYE_AFFINITY_KEYS) {
            if (Math.random() < mutationChance) {
                this.dyeNodeTypeAffinity[key] = (Number(this.dyeNodeTypeAffinity[key]) || 1)
                    + ((Math.random() - 0.5) * 2 * mutationMagnitude);
                this._bumpMutationStat('dyeNodeTypeAffinity');
            }
        }

        this._sanitizeDyeEcologyGenes();
    }

    _sampleLocalDyeAt(worldX, worldY) {
        const fluid = runtimeState.fluidField;
        if (!fluid || !fluid.scaleX || !fluid.scaleY) {
            return { r: 0, g: 0, b: 0, intensity: 0, hue: 0, saturation: 0 };
        }

        const gx = Math.floor(worldX / fluid.scaleX);
        const gy = Math.floor(worldY / fluid.scaleY);
        const idx = fluid.IX(gx, gy);

        const r = Number(fluid.densityR?.[idx]) || 0;
        const g = Number(fluid.densityG?.[idx]) || 0;
        const b = Number(fluid.densityB?.[idx]) || 0;

        const maxCh = Math.max(r, g, b) / 255;
        const minCh = Math.min(r, g, b) / 255;
        const intensity = Math.max(0, Math.min(1, (r + g + b) / (255 * 3)));
        const saturation = maxCh <= 1e-8 ? 0 : Math.max(0, Math.min(1, (maxCh - minCh) / Math.max(maxCh, 1e-8)));
        const hue = computeHueFromRgb(r, g, b);

        return { r, g, b, intensity, hue, saturation };
    }

    _computeDyeEcologyStateAt(worldX, worldY) {
        if (!config.DYE_ECOLOGY_ENABLED) {
            return {
                ...this._sampleLocalDyeAt(worldX, worldY),
                match: 0.5,
                preferredMatch: 0.5,
                response: 0,
                overexposure: 0,
                emitterInhibitionScale: 1
            };
        }

        const sample = this._sampleLocalDyeAt(worldX, worldY);
        const hueDist = circularHueDistance(sample.hue, this.dyePreferredHue);
        const tol = Math.max(0.005, Number(this.dyeHueTolerance) || 0.1);
        const match = Math.exp(-(hueDist * hueDist) / Math.max(1e-6, 2 * tol * tol));
        const preferredMatch = this.dyeResponseSign >= 0 ? match : (1 - match);
        const centered = (preferredMatch - 0.5) * 2;
        const response = centered * (Number(this.dyeResponseGain) || 1);

        const overThresh = Math.max(0, Math.min(1, Number(config.DYE_OVEREXPOSURE_THRESHOLD) || 0.82));
        const overexposure = sample.intensity > overThresh
            ? Math.max(0, Math.min(1, (sample.intensity - overThresh) / Math.max(1e-6, 1 - overThresh)))
            : 0;

        const inhibThresh = Math.max(0, Math.min(1, Number(config.DYE_EMITTER_SELF_INHIBITION_THRESHOLD) || 0.6));
        const inhibStrength = Math.max(0, Number(config.DYE_EMITTER_SELF_INHIBITION_STRENGTH) || 0.55);
        const inhibNorm = sample.intensity > inhibThresh
            ? Math.max(0, Math.min(1, (sample.intensity - inhibThresh) / Math.max(1e-6, 1 - inhibThresh)))
            : 0;
        const emitterInhibitionScale = Math.max(0.1, 1 - inhibNorm * inhibStrength);

        return {
            ...sample,
            match,
            preferredMatch,
            response,
            overexposure,
            emitterInhibitionScale
        };
    }

    _resolveDyeEffectScale(state, { weight = 1, affinityKey = null, includeEmitterInhibition = false } = {}) {
        if (!config.DYE_ECOLOGY_ENABLED) return 1;

        const minScale = Math.max(0.05, Number(config.DYE_EFFECT_MIN_SCALE) || 0.35);
        const maxScale = Math.max(minScale, Number(config.DYE_EFFECT_MAX_SCALE) || 1.9);

        let affinity = 1;
        if (affinityKey) {
            affinity = Number(this.dyeNodeTypeAffinity?.[affinityKey]);
            if (!Number.isFinite(affinity)) affinity = 1;
        }

        const base = 1 + ((Number(state?.response) || 0) * weight * affinity);
        let out = Math.max(minScale, Math.min(maxScale, base));

        if (includeEmitterInhibition) {
            out *= Math.max(0.1, Number(state?.emitterInhibitionScale) || 1);
            out = Math.max(0.05, Math.min(maxScale, out));
        }

        return out;
    }

    _getDyeEcologyStateAtBodyCenter() {
        const c = this.getAveragePosition();
        const state = this._computeDyeEcologyStateAt(c.x, c.y);
        this.lastDyeEcologyState = state;
        return state;
    }

    _resolveActuationCooldownMultiplier(point, channel = 'node') {
        if (channel === 'grabber') {
            return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_GRABBER) || 1);
        }
        if (channel === 'default_pattern') {
            return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_DEFAULT_PATTERN) || 1);
        }

        switch (point?.nodeType) {
            case NodeType.EMITTER:
                return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_EMITTER) || 1);
            case NodeType.SWIMMER:
                return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_SWIMMER) || 1);
            case NodeType.EATER:
                return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_EATER) || 1);
            case NodeType.PREDATOR:
                return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_PREDATOR) || 1);
            case NodeType.JET:
                return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_JET) || 1);
            case NodeType.ATTRACTOR:
                return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_ATTRACTOR) || 1);
            case NodeType.REPULSOR:
                return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_REPULSOR) || 1);
            default:
                return Math.max(0.05, Number(config.ACTUATION_COOLDOWN_MULTIPLIER_DEFAULT) || 1);
        }
    }

    _computeEffectiveActuationInterval(point, channel = 'node') {
        const baseGene = this._sanitizeActivationIntervalGene(point?.activationIntervalGene);
        const multiplier = this._resolveActuationCooldownMultiplier(point, channel);
        const scaled = this._sanitizeActivationIntervalGene(baseGene * multiplier);
        return Math.max(1, scaled);
    }

    _getActuationTelemetryKey(point, channel = 'node') {
        if (channel === 'grabber') return 'grabber';
        if (channel === 'default_pattern') return 'default_pattern';
        return `node_${Number.isFinite(Number(point?.nodeType)) ? Number(point.nodeType) : -1}`;
    }

    _recordActuationDecision(point, channel, evaluated, effectiveInterval) {
        const key = this._getActuationTelemetryKey(point, channel);

        this.actuationIntervalSamples += 1;
        this.actuationIntervalTotal += effectiveInterval;

        if (evaluated) {
            this.actuationEvaluations += 1;
            this.actuationEvaluationsByNodeType[key] = (this.actuationEvaluationsByNodeType[key] || 0) + 1;
        } else {
            this.actuationSkips += 1;
            this.actuationSkipsByNodeType[key] = (this.actuationSkipsByNodeType[key] || 0) + 1;
        }
    }

    _prepareActuationStateForTick() {
        for (const point of this.massPoints) {
            point.__actuationEvaluatedThisTick = false;
            if (!Number.isFinite(point.activationIntervalGene)) {
                point.activationIntervalGene = this._randomActivationIntervalGene();
            }
            point.activationIntervalGene = this._sanitizeActivationIntervalGene(point.activationIntervalGene);

            if (!Number.isFinite(point.predatorRadiusGene)) {
                point.predatorRadiusGene = this._randomPredatorRadiusGene();
            }
            point.predatorRadiusGene = this._sanitizePredatorRadiusGene(point.predatorRadiusGene);

            if (!point.actuationCooldownByChannel || typeof point.actuationCooldownByChannel !== 'object') {
                point.actuationCooldownByChannel = {};
            }
            // Legacy compatibility: migrate single cooldown into node channel once.
            if (
                Number.isFinite(point.actuationCooldownRemaining) &&
                !Number.isFinite(point.actuationCooldownByChannel.node)
            ) {
                point.actuationCooldownByChannel.node = Number(point.actuationCooldownRemaining);
            }

            if (!point.swimmerActuation) {
                point.swimmerActuation = { magnitude: 0, angle: 0 };
            }
        }
    }

    _shouldEvaluatePointActuation(point, channel = 'node') {
        const effectiveInterval = this._computeEffectiveActuationInterval(point, channel);
        const key = String(channel || 'node');

        if (!point.actuationCooldownByChannel || typeof point.actuationCooldownByChannel !== 'object') {
            point.actuationCooldownByChannel = {};
        }

        if (!Number.isFinite(point.actuationCooldownByChannel[key])) {
            point.actuationCooldownByChannel[key] = 0;
        }

        if (point.actuationCooldownByChannel[key] <= 0) {
            point.actuationCooldownByChannel[key] = Math.max(0, effectiveInterval - 1);
            point.__actuationEvaluatedThisTick = true;
            this._recordActuationDecision(point, channel, true, effectiveInterval);
            return true;
        }

        point.actuationCooldownByChannel[key] -= 1;
        this._recordActuationDecision(point, channel, false, effectiveInterval);
        return false;
    }


    /**
     * Create a random heritable growth genome for new creatures.
     *
     * The genome controls growth probability, size, anchor preferences,
     * node-type preferences, edge preferences, and target distance bands.
     */
    _createRandomGrowthGenome() {
        const randomWeight = () => config.GROWTH_MIN_WEIGHT + Math.random();
        const midStart = Math.max(config.GROWTH_DISTANCE_MIN + 1, config.GROWTH_DISTANCE_MID);
        const maxDist = Math.max(midStart + 1, config.GROWTH_DISTANCE_MAX);
        const farStart = Math.max(midStart + 1, Math.floor((midStart + maxDist) * 0.5));

        return this._sanitizeGrowthGenome({
            growthChancePerTick: config.GROWTH_BASE_CHANCE_MIN + Math.random() * Math.max(0.0001, (config.GROWTH_BASE_CHANCE_MAX - config.GROWTH_BASE_CHANCE_MIN)),
            minEnergyRatioToGrow: config.GROWTH_MIN_ENERGY_RATIO_MIN + Math.random() * Math.max(0.0001, (config.GROWTH_MIN_ENERGY_RATIO_MAX - config.GROWTH_MIN_ENERGY_RATIO_MIN)),
            growthCooldownTicks: Math.floor(config.GROWTH_COOLDOWN_MIN + Math.random() * Math.max(1, (config.GROWTH_COOLDOWN_MAX - config.GROWTH_COOLDOWN_MIN + 1))),
            nodesPerGrowthWeights: [
                { count: 1, weight: randomWeight() },
                { count: 2, weight: randomWeight() },
                { count: 3, weight: randomWeight() }
            ],
            newNodeTypeWeights: DEFAULT_GROWTH_NODE_TYPES.map((nodeType) => ({ nodeType, weight: randomWeight() })),
            anchorNodeTypeWeights: DEFAULT_GROWTH_NODE_TYPES.map((nodeType) => ({ nodeType, weight: randomWeight() })),
            distanceRangeWeights: [
                { key: 'near', min: config.GROWTH_DISTANCE_MIN, max: midStart, weight: randomWeight() },
                { key: 'mid', min: midStart, max: farStart, weight: randomWeight() },
                { key: 'far', min: farStart, max: maxDist, weight: randomWeight() }
            ],
            edgeTypeWeights: [
                { type: 'soft', weight: randomWeight() },
                { type: 'rigid', weight: randomWeight() }
            ],
            edgeStiffnessScale: 0.6 + Math.random() * 1.2,
            edgeDampingScale: 0.6 + Math.random() * 1.2,
            nodeActivationIntervalBias: (Math.random() - 0.5) * 2,
            edgeActivationIntervalBias: (Math.random() - 0.5) * 2,
            activationIntervalJitter: Math.random() * 1.5
        });
    }

    /**
     * Normalize and clamp growth-gene values to safe runtime bounds.
     */
    _sanitizeGrowthGenome(genome) {
        if (!genome || typeof genome !== 'object') {
            // Avoid recursive call loops if fallback itself fails to return object.
            const seeded = {
                growthChancePerTick: config.GROWTH_BASE_CHANCE_MIN,
                minEnergyRatioToGrow: config.GROWTH_MIN_ENERGY_RATIO_MIN,
                growthCooldownTicks: config.GROWTH_COOLDOWN_MIN,
                nodesPerGrowthWeights: [{ count: 1, weight: 1 }],
                newNodeTypeWeights: DEFAULT_GROWTH_NODE_TYPES.map((nodeType) => ({ nodeType, weight: 1 })),
                anchorNodeTypeWeights: DEFAULT_GROWTH_NODE_TYPES.map((nodeType) => ({ nodeType, weight: 1 })),
                distanceRangeWeights: [{ key: 'near', min: config.GROWTH_DISTANCE_MIN, max: config.GROWTH_DISTANCE_MAX, weight: 1 }],
                edgeTypeWeights: [{ type: 'soft', weight: 1 }],
                edgeStiffnessScale: 1,
                edgeDampingScale: 1,
                nodeActivationIntervalBias: 0,
                edgeActivationIntervalBias: 0,
                activationIntervalJitter: 0.5
            };
            genome = seeded;
        }

        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
        const normalizeWeights = (entries, mapFn) => {
            const out = (entries || []).map(mapFn).filter(Boolean);
            if (out.length === 0) return [];
            let total = 0;
            for (const e of out) {
                e.weight = Math.max(config.GROWTH_MIN_WEIGHT, Number(e.weight) || config.GROWTH_MIN_WEIGHT);
                total += e.weight;
            }
            if (total <= 0) {
                const equal = 1 / out.length;
                out.forEach((e) => { e.weight = equal; });
                return out;
            }
            out.forEach((e) => { e.weight /= total; });
            return out;
        };

        const sanitized = {
            growthChancePerTick: clamp(Number(genome.growthChancePerTick) || config.GROWTH_BASE_CHANCE_MIN, config.GROWTH_BASE_CHANCE_MIN, config.GROWTH_BASE_CHANCE_MAX),
            minEnergyRatioToGrow: clamp(Number(genome.minEnergyRatioToGrow) || config.GROWTH_MIN_ENERGY_RATIO_MIN, config.GROWTH_MIN_ENERGY_RATIO_MIN, config.GROWTH_MIN_ENERGY_RATIO_MAX),
            growthCooldownTicks: Math.floor(clamp(Number(genome.growthCooldownTicks) || config.GROWTH_COOLDOWN_MIN, config.GROWTH_COOLDOWN_MIN, config.GROWTH_COOLDOWN_MAX)),
            nodesPerGrowthWeights: normalizeWeights(genome.nodesPerGrowthWeights, (e) => {
                const count = Math.max(1, Math.min(5, Math.floor(Number(e?.count) || 1)));
                return { count, weight: Number(e?.weight) || 0 };
            }),
            newNodeTypeWeights: normalizeWeights(genome.newNodeTypeWeights, (e) => {
                const nodeType = Math.floor(Number(e?.nodeType));
                if (!Number.isInteger(nodeType)) return null;
                return { nodeType, weight: Number(e?.weight) || 0 };
            }),
            anchorNodeTypeWeights: normalizeWeights(genome.anchorNodeTypeWeights, (e) => {
                const nodeType = Math.floor(Number(e?.nodeType));
                if (!Number.isInteger(nodeType)) return null;
                return { nodeType, weight: Number(e?.weight) || 0 };
            }),
            distanceRangeWeights: normalizeWeights(genome.distanceRangeWeights, (e) => {
                let min = Number(e?.min);
                let max = Number(e?.max);
                if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
                min = clamp(min, config.GROWTH_DISTANCE_MIN, config.GROWTH_DISTANCE_MAX);
                max = clamp(max, config.GROWTH_DISTANCE_MIN, config.GROWTH_DISTANCE_MAX);
                if (max <= min) max = Math.min(config.GROWTH_DISTANCE_MAX, min + 1);
                return { key: e?.key || 'custom', min, max, weight: Number(e?.weight) || 0 };
            }),
            edgeTypeWeights: normalizeWeights(genome.edgeTypeWeights, (e) => {
                const type = e?.type === 'rigid' ? 'rigid' : 'soft';
                return { type, weight: Number(e?.weight) || 0 };
            }),
            edgeStiffnessScale: clamp(Number(genome.edgeStiffnessScale) || 1, 0.1, 5),
            edgeDampingScale: clamp(Number(genome.edgeDampingScale) || 1, 0.1, 5),
            nodeActivationIntervalBias: clamp(Number(genome.nodeActivationIntervalBias) || 0, -3, 3),
            edgeActivationIntervalBias: clamp(Number(genome.edgeActivationIntervalBias) || 0, -3, 3),
            activationIntervalJitter: clamp(Number(genome.activationIntervalJitter) || 0.5, 0, 3)
        };

        if (sanitized.nodesPerGrowthWeights.length === 0) sanitized.nodesPerGrowthWeights = [{ count: 1, weight: 1 }];
        if (sanitized.newNodeTypeWeights.length === 0) sanitized.newNodeTypeWeights = DEFAULT_GROWTH_NODE_TYPES.map((nodeType) => ({ nodeType, weight: 1 / DEFAULT_GROWTH_NODE_TYPES.length }));
        if (sanitized.anchorNodeTypeWeights.length === 0) sanitized.anchorNodeTypeWeights = DEFAULT_GROWTH_NODE_TYPES.map((nodeType) => ({ nodeType, weight: 1 / DEFAULT_GROWTH_NODE_TYPES.length }));
        if (sanitized.distanceRangeWeights.length === 0) sanitized.distanceRangeWeights = [{ key: 'near', min: config.GROWTH_DISTANCE_MIN, max: config.GROWTH_DISTANCE_MAX, weight: 1 }];
        if (sanitized.edgeTypeWeights.length === 0) sanitized.edgeTypeWeights = [{ type: 'soft', weight: 1 }];

        return sanitized;
    }

    /**
     * Inherit growth genes from parent and mutate them probabilistically.
     */
    _mutateGrowthGenomeFromParent(parentGenome) {
        if (!parentGenome || typeof parentGenome !== 'object') {
            return { genome: this._createRandomGrowthGenome(), didMutate: false };
        }

        const genome = JSON.parse(JSON.stringify(this._sanitizeGrowthGenome(parentGenome)));
        const mutationChance = Math.max(0, Math.min(1, config.GROWTH_GENE_MUTATION_CHANCE * config.GLOBAL_MUTATION_RATE_MODIFIER));
        const mutationMagnitude = Math.max(0, config.GROWTH_GENE_MUTATION_MAGNITUDE * config.GLOBAL_MUTATION_RATE_MODIFIER);
        let didMutate = false;

        const mutateScalar = (key, lo, hi) => {
            if (Math.random() >= mutationChance) return;
            const old = Number(genome[key]) || lo;
            const mutated = old * (1 + (Math.random() - 0.5) * 2 * mutationMagnitude);
            genome[key] = Math.max(lo, Math.min(hi, mutated));
            didMutate = didMutate || Math.abs(genome[key] - old) > 1e-6;
        };

        mutateScalar('growthChancePerTick', config.GROWTH_BASE_CHANCE_MIN, config.GROWTH_BASE_CHANCE_MAX);
        mutateScalar('minEnergyRatioToGrow', config.GROWTH_MIN_ENERGY_RATIO_MIN, config.GROWTH_MIN_ENERGY_RATIO_MAX);
        if (Math.random() < mutationChance) {
            const delta = Math.round((Math.random() - 0.5) * 2 * mutationMagnitude * 20);
            genome.growthCooldownTicks = Math.max(config.GROWTH_COOLDOWN_MIN, Math.min(config.GROWTH_COOLDOWN_MAX, Math.floor(genome.growthCooldownTicks + delta)));
            didMutate = didMutate || delta !== 0;
        }

        mutateScalar('edgeStiffnessScale', 0.1, 5);
        mutateScalar('edgeDampingScale', 0.1, 5);
        mutateScalar('nodeActivationIntervalBias', -3, 3);
        mutateScalar('edgeActivationIntervalBias', -3, 3);
        mutateScalar('activationIntervalJitter', 0, 3);

        const mutateWeights = (entries, onEntry = null) => {
            for (const e of entries || []) {
                if (Math.random() < mutationChance) {
                    e.weight = Math.max(config.GROWTH_MIN_WEIGHT, e.weight * (1 + (Math.random() - 0.5) * 2 * mutationMagnitude));
                    didMutate = true;
                }
                if (onEntry) onEntry(e);
            }
        };

        mutateWeights(genome.nodesPerGrowthWeights);
        mutateWeights(genome.newNodeTypeWeights);
        mutateWeights(genome.anchorNodeTypeWeights);
        mutateWeights(genome.edgeTypeWeights);
        mutateWeights(genome.distanceRangeWeights, (entry) => {
            if (Math.random() < mutationChance) {
                const span = Math.max(1, entry.max - entry.min);
                const shift = span * (Math.random() - 0.5) * mutationMagnitude;
                entry.min = Math.max(config.GROWTH_DISTANCE_MIN, Math.min(config.GROWTH_DISTANCE_MAX - 1, entry.min + shift));
                entry.max = Math.max(entry.min + 1, Math.min(config.GROWTH_DISTANCE_MAX, entry.max + shift));
                didMutate = true;
            }
        });

        return {
            genome: this._sanitizeGrowthGenome(genome),
            didMutate
        };
    }

    /**
     * Weighted random sampler for gene-defined categorical choices.
     */
    _sampleWeightedEntry(entries, fallback = null) {
        if (!Array.isArray(entries) || entries.length === 0) return fallback;
        let total = 0;
        for (const e of entries) {
            total += Math.max(0, Number(e.weight) || 0);
        }
        if (total <= 0) return entries[0] || fallback;

        let r = Math.random() * total;
        for (const e of entries) {
            r -= Math.max(0, Number(e.weight) || 0);
            if (r <= 0) return e;
        }
        return entries[entries.length - 1] || fallback;
    }

    _bumpMutationStat(key, amount = 1) {
        if (!runtimeState.mutationStats || typeof runtimeState.mutationStats !== 'object') return;
        runtimeState.mutationStats[key] = (runtimeState.mutationStats[key] || 0) + amount;
    }

    _cloneBlueprintSnapshot(points = this.blueprintPoints, springs = this.blueprintSprings) {
        return {
            points: JSON.parse(JSON.stringify(Array.isArray(points) ? points : [])),
            springs: JSON.parse(JSON.stringify(Array.isArray(springs) ? springs : []))
        };
    }

    _sanitizeBlueprintDataInPlace() {
        const sanitizedPoints = [];
        for (const rawPoint of this.blueprintPoints || []) {
            if (!rawPoint || typeof rawPoint !== 'object') continue;

            const nodeType = Number.isFinite(Number(rawPoint.nodeType))
                ? Math.max(NodeType.PREDATOR, Math.min(NodeType.REPULSOR, Math.floor(Number(rawPoint.nodeType))))
                : NodeType.EATER;

            let movementType = Number.isFinite(Number(rawPoint.movementType))
                ? Math.max(MovementType.FIXED, Math.min(MovementType.NEUTRAL, Math.floor(Number(rawPoint.movementType))))
                : MovementType.NEUTRAL;
            if (nodeType === NodeType.SWIMMER) movementType = MovementType.NEUTRAL;

            const dyeColor = Array.isArray(rawPoint.dyeColor) && rawPoint.dyeColor.length >= 3
                ? rawPoint.dyeColor.slice(0, 3).map((v) => Math.max(0, Math.min(255, Math.floor(Number(v) || 0))))
                : [200, 50, 50];

            const isNeuron = nodeType === NodeType.NEURON;
            let neuronDataBlueprint = null;
            if (isNeuron) {
                const hiddenLayerSize = Number(rawPoint?.neuronDataBlueprint?.hiddenLayerSize);
                neuronDataBlueprint = {
                    hiddenLayerSize: Math.max(
                        config.DEFAULT_HIDDEN_LAYER_SIZE_MIN,
                        Math.min(
                            config.DEFAULT_HIDDEN_LAYER_SIZE_MAX,
                            Number.isFinite(hiddenLayerSize) ? Math.floor(hiddenLayerSize) : config.DEFAULT_HIDDEN_LAYER_SIZE_MIN
                        )
                    )
                };
            }

            sanitizedPoints.push({
                relX: Number(rawPoint.relX) || 0,
                relY: Number(rawPoint.relY) || 0,
                radius: Math.max(0.5, Math.min(12, Number(rawPoint.radius) || 1)),
                mass: Math.max(0.1, Math.min(2.5, Number(rawPoint.mass) || 0.5)),
                nodeType,
                movementType,
                dyeColor,
                canBeGrabber: Boolean(rawPoint.canBeGrabber),
                neuronDataBlueprint,
                activationIntervalGene: this._sanitizeActivationIntervalGene(rawPoint.activationIntervalGene ?? this._randomActivationIntervalGene()),
                predatorRadiusGene: this._sanitizePredatorRadiusGene(rawPoint.predatorRadiusGene ?? this._randomPredatorRadiusGene()),
                eyeTargetType: nodeType === NodeType.EYE
                    ? (Number(rawPoint.eyeTargetType) === EyeTargetType.FOREIGN_BODY_POINT
                        ? EyeTargetType.FOREIGN_BODY_POINT
                        : EyeTargetType.PARTICLE)
                    : undefined
            });
        }

        const sanitizedSprings = [];
        const pointCount = sanitizedPoints.length;
        const seenEdgeKeys = new Set();

        for (const rawSpring of this.blueprintSprings || []) {
            const p1Index = Math.floor(Number(rawSpring?.p1Index));
            const p2Index = Math.floor(Number(rawSpring?.p2Index));
            if (!Number.isInteger(p1Index) || !Number.isInteger(p2Index)) continue;
            if (p1Index < 0 || p2Index < 0 || p1Index >= pointCount || p2Index >= pointCount) continue;
            if (p1Index === p2Index) continue;

            const edgeKey = p1Index < p2Index ? `${p1Index}:${p2Index}` : `${p2Index}:${p1Index}`;
            if (seenEdgeKeys.has(edgeKey)) continue;
            seenEdgeKeys.add(edgeKey);

            const p1 = sanitizedPoints[p1Index];
            const p2 = sanitizedPoints[p2Index];
            const dx = p1.relX - p2.relX;
            const dy = p1.relY - p2.relY;
            const geometricLength = Math.sqrt(dx * dx + dy * dy);

            sanitizedSprings.push({
                p1Index,
                p2Index,
                restLength: Math.max(1, Number(rawSpring?.restLength) || geometricLength || 1),
                isRigid: Boolean(rawSpring?.isRigid),
                stiffness: Math.max(100, Math.min(10000, Number(rawSpring?.stiffness) || this.stiffness)),
                damping: Math.max(0.1, Math.min(50, Number(rawSpring?.damping) || this.springDamping)),
                activationIntervalGene: this._sanitizeActivationIntervalGene(rawSpring?.activationIntervalGene ?? this._randomActivationIntervalGene())
            });
        }

        this.blueprintPoints = sanitizedPoints;
        this.blueprintSprings = sanitizedSprings;
        this._enforcePhotosyntheticBlueprintConstraints();
    }

    _enforcePhotosyntheticBlueprintConstraints() {
        const points = Array.isArray(this.blueprintPoints) ? this.blueprintPoints : [];
        const springs = Array.isArray(this.blueprintSprings) ? this.blueprintSprings : [];
        if (points.length === 0 || springs.length === 0) return;

        const adjacency = Array.from({ length: points.length }, () => new Set());

        for (const spring of springs) {
            const a = Math.floor(Number(spring?.p1Index));
            const b = Math.floor(Number(spring?.p2Index));
            if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
            if (a < 0 || b < 0 || a >= points.length || b >= points.length) continue;

            adjacency[a].add(b);
            adjacency[b].add(a);

            const aPhoto = points[a]?.nodeType === NodeType.PHOTOSYNTHETIC;
            const bPhoto = points[b]?.nodeType === NodeType.PHOTOSYNTHETIC;
            if (aPhoto || bPhoto) spring.isRigid = true;
        }

        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            if (!p || p.nodeType !== NodeType.PHOTOSYNTHETIC) continue;
            for (const nIdx of adjacency[i]) {
                const n = points[nIdx];
                if (!n) continue;
                if (n.nodeType !== NodeType.PHOTOSYNTHETIC) {
                    n.movementType = MovementType.NEUTRAL;
                }
            }
        }
    }

    _enforcePhotosyntheticPhenotypeConstraints() {
        const points = Array.isArray(this.massPoints) ? this.massPoints : [];
        const springs = Array.isArray(this.springs) ? this.springs : [];
        if (points.length === 0 || springs.length === 0) return;

        const pointIndex = new Map();
        points.forEach((p, idx) => pointIndex.set(p, idx));
        const adjacency = Array.from({ length: points.length }, () => new Set());

        for (const spring of springs) {
            const a = pointIndex.get(spring?.p1);
            const b = pointIndex.get(spring?.p2);
            if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;

            adjacency[a].add(b);
            adjacency[b].add(a);

            const aPhoto = points[a]?.nodeType === NodeType.PHOTOSYNTHETIC;
            const bPhoto = points[b]?.nodeType === NodeType.PHOTOSYNTHETIC;
            if (aPhoto || bPhoto) {
                spring.isRigid = true;
                spring.stiffness = config.RIGID_SPRING_STIFFNESS;
                spring.dampingFactor = config.RIGID_SPRING_DAMPING;
            }
        }

        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            if (!p || p.nodeType !== NodeType.PHOTOSYNTHETIC) continue;
            for (const nIdx of adjacency[i]) {
                const n = points[nIdx];
                if (!n) continue;
                if (n.nodeType !== NodeType.PHOTOSYNTHETIC) {
                    n.movementType = MovementType.NEUTRAL;
                }
            }
        }
    }

    _buildBlueprintAdjacency() {
        const adjacency = Array.from({ length: this.blueprintPoints.length }, () => []);
        for (const spring of this.blueprintSprings) {
            if (!spring) continue;
            const a = Math.floor(Number(spring.p1Index));
            const b = Math.floor(Number(spring.p2Index));
            if (!Number.isInteger(a) || !Number.isInteger(b)) continue;
            if (a < 0 || b < 0 || a >= adjacency.length || b >= adjacency.length || a === b) continue;
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
        return adjacency;
    }

    _countConnectedBlueprintPoints() {
        if (!Array.isArray(this.blueprintPoints) || this.blueprintPoints.length === 0) return 0;
        const adjacency = this._buildBlueprintAdjacency();
        const visited = new Set();
        const stack = [0];
        visited.add(0);

        while (stack.length) {
            const current = stack.pop();
            for (const next of adjacency[current] || []) {
                if (visited.has(next)) continue;
                visited.add(next);
                stack.push(next);
            }
        }

        return visited.size;
    }

    _evaluateBlueprintViability() {
        const points = this.blueprintPoints || [];
        const springs = this.blueprintSprings || [];

        const reasons = {
            structure: false,
            diversity: false,
            harvest: false,
            actuator: false
        };

        const pointCount = points.length;
        const minPoints = Math.max(1, Math.floor(Number(config.OFFSPRING_MIN_BLUEPRINT_POINTS) || 1));
        const springRatio = Math.max(0, Number(config.OFFSPRING_MIN_SPRING_TO_POINT_RATIO) || 0);
        const minSprings = Math.max(Math.max(0, pointCount - 1), Math.floor(pointCount * springRatio));

        if (pointCount < minPoints || springs.length < minSprings) {
            reasons.structure = true;
        }

        if (!reasons.structure) {
            const connectedCount = this._countConnectedBlueprintPoints();
            if (connectedCount < pointCount) reasons.structure = true;
        }

        this._calculateBlueprintRadius();
        const maxWorldDim = Math.max(1, Math.max(config.WORLD_WIDTH, config.WORLD_HEIGHT));
        const maxRadiusFraction = Math.max(0.05, Math.min(1, Number(config.OFFSPRING_MAX_BLUEPRINT_RADIUS_WORLD_FRACTION) || 0.45));
        const maxRadius = maxWorldDim * maxRadiusFraction;
        if (!Number.isFinite(this.blueprintRadius) || this.blueprintRadius <= 0 || this.blueprintRadius > maxRadius) {
            reasons.structure = true;
        }

        const nodeTypeSet = new Set(points.map((p) => p?.nodeType));
        const minDiversity = Math.max(1, Math.floor(Number(config.OFFSPRING_MIN_NODE_TYPE_DIVERSITY) || 1));
        if (nodeTypeSet.size < minDiversity) {
            reasons.diversity = true;
        }

        const harvestTypes = new Set([NodeType.EATER, NodeType.PHOTOSYNTHETIC, NodeType.PREDATOR]);
        const actuatorTypes = new Set([NodeType.SWIMMER, NodeType.EMITTER, NodeType.JET, NodeType.ATTRACTOR, NodeType.REPULSOR]);

        const hasHarvestNode = points.some((p) => harvestTypes.has(p?.nodeType));
        const hasActuatorNode = points.some((p) => actuatorTypes.has(p?.nodeType));

        if (config.OFFSPRING_REQUIRE_HARVESTER_NODE && !hasHarvestNode) reasons.harvest = true;
        if (config.OFFSPRING_REQUIRE_ACTUATOR_NODE && !hasActuatorNode) reasons.actuator = true;

        return {
            ok: !reasons.structure && !reasons.diversity && !reasons.harvest && !reasons.actuator,
            reasons
        };
    }

    _sampleConnectedDonorModuleIndices(donor) {
        const donorPoints = Array.isArray(donor?.blueprintPoints) ? donor.blueprintPoints : [];
        const donorSprings = Array.isArray(donor?.blueprintSprings) ? donor.blueprintSprings : [];
        if (donorPoints.length < 2) return [];

        const minTake = Math.max(1, Math.floor(Number(config.HGT_GRAFT_MIN_POINTS) || 2));
        const maxTake = Math.max(minTake, Math.floor(Number(config.HGT_GRAFT_MAX_POINTS) || minTake));
        const target = Math.min(donorPoints.length, minTake + Math.floor(Math.random() * (maxTake - minTake + 1)));

        const adjacency = Array.from({ length: donorPoints.length }, () => []);
        for (const spring of donorSprings) {
            const a = Math.floor(Number(spring?.p1Index));
            const b = Math.floor(Number(spring?.p2Index));
            if (!Number.isInteger(a) || !Number.isInteger(b) || a === b) continue;
            if (a < 0 || b < 0 || a >= donorPoints.length || b >= donorPoints.length) continue;
            adjacency[a].push(b);
            adjacency[b].push(a);
        }

        const seed = Math.floor(Math.random() * donorPoints.length);
        const visited = new Set([seed]);
        const queue = [seed];

        while (queue.length > 0 && visited.size < target) {
            const current = queue.shift();
            const neighbors = [...(adjacency[current] || [])];
            for (let i = neighbors.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [neighbors[i], neighbors[j]] = [neighbors[j], neighbors[i]];
            }
            for (const next of neighbors) {
                if (visited.has(next)) continue;
                visited.add(next);
                queue.push(next);
                if (visited.size >= target) break;
            }
        }

        return [...visited];
    }

    _attemptDonorModuleGraftMutation(parentBody) {
        this._bumpMutationStat('hgtDonorGraftAttempt');

        const population = Array.isArray(runtimeState.softBodyPopulation) ? runtimeState.softBodyPopulation : [];
        const parentCenter = parentBody?.getAveragePosition ? parentBody.getAveragePosition() : this.getAveragePosition();
        const donorSearchRadius = Math.max(0, Number(config.HGT_GRAFT_DONOR_SEARCH_RADIUS) || 0);

        const donorCandidates = population.filter((candidate) => {
            if (!candidate || candidate === parentBody || candidate === this || candidate.isUnstable) return false;
            if (!Array.isArray(candidate.blueprintPoints) || candidate.blueprintPoints.length < 2) return false;
            if (!Array.isArray(candidate.blueprintSprings) || candidate.blueprintSprings.length < 1) return false;
            if (donorSearchRadius <= 0 || !parentCenter || typeof candidate.getAveragePosition !== 'function') return true;
            const c = candidate.getAveragePosition();
            const dx = c.x - parentCenter.x;
            const dy = c.y - parentCenter.y;
            return (dx * dx + dy * dy) <= donorSearchRadius * donorSearchRadius;
        });

        if (donorCandidates.length === 0) {
            this._bumpMutationStat('hgtDonorGraftRejectedNoDonor');
            return false;
        }

        const donor = donorCandidates[Math.floor(Math.random() * donorCandidates.length)];
        const donorIndices = this._sampleConnectedDonorModuleIndices(donor);
        if (donorIndices.length < Math.max(1, Math.floor(Number(config.HGT_GRAFT_MIN_POINTS) || 2))) {
            this._bumpMutationStat('hgtDonorGraftRejectedInvalid');
            return false;
        }

        const maxTotalPoints = Math.max(1, Math.floor(Number(config.HGT_GRAFT_MAX_TOTAL_POINTS) || config.GROWTH_MAX_POINTS_PER_CREATURE));
        if (this.blueprintPoints.length + donorIndices.length > maxTotalPoints) {
            this._bumpMutationStat('hgtDonorGraftRejectedCapacity');
            return false;
        }

        const startPointCount = this.blueprintPoints.length;
        const startSpringCount = this.blueprintSprings.length;

        const mapOldToNew = new Map();
        const modulePoints = donorIndices.map((idx) => donor.blueprintPoints[idx]).filter(Boolean);
        if (modulePoints.length < 2) {
            this._bumpMutationStat('hgtDonorGraftRejectedInvalid');
            return false;
        }

        const moduleCentroid = modulePoints.reduce((acc, p) => {
            acc.x += Number(p.relX) || 0;
            acc.y += Number(p.relY) || 0;
            return acc;
        }, { x: 0, y: 0 });
        moduleCentroid.x /= modulePoints.length;
        moduleCentroid.y /= modulePoints.length;

        const anchorIndex = this.blueprintPoints.length > 0
            ? Math.floor(Math.random() * this.blueprintPoints.length)
            : -1;
        const anchorPoint = anchorIndex >= 0 ? this.blueprintPoints[anchorIndex] : null;

        let moduleRadius = 1;
        for (const p of modulePoints) {
            const dx = (Number(p.relX) || 0) - moduleCentroid.x;
            const dy = (Number(p.relY) || 0) - moduleCentroid.y;
            moduleRadius = Math.max(moduleRadius, Math.sqrt(dx * dx + dy * dy) + Math.max(0.5, Number(p.radius) || 0));
        }

        const angle = Math.random() * Math.PI * 2;
        const anchorX = anchorPoint ? Number(anchorPoint.relX) || 0 : 0;
        const anchorY = anchorPoint ? Number(anchorPoint.relY) || 0 : 0;
        const anchorRadius = anchorPoint ? Math.max(0.5, Number(anchorPoint.radius) || 0.5) : 0.5;
        const placementDistance = anchorRadius + moduleRadius + 2 + Math.random() * 6;
        const tx = anchorX + Math.cos(angle) * placementDistance - moduleCentroid.x;
        const ty = anchorY + Math.sin(angle) * placementDistance - moduleCentroid.y;

        for (const oldIndex of donorIndices) {
            const src = donor.blueprintPoints[oldIndex];
            if (!src) continue;

            const nodeType = Number.isFinite(Number(src.nodeType))
                ? Math.max(NodeType.PREDATOR, Math.min(NodeType.REPULSOR, Math.floor(Number(src.nodeType))))
                : NodeType.EATER;
            let movementType = Number.isFinite(Number(src.movementType))
                ? Math.max(MovementType.FIXED, Math.min(MovementType.NEUTRAL, Math.floor(Number(src.movementType))))
                : MovementType.NEUTRAL;
            if (nodeType === NodeType.SWIMMER) movementType = MovementType.NEUTRAL;

            const point = {
                relX: (Number(src.relX) || 0) + tx,
                relY: (Number(src.relY) || 0) + ty,
                radius: Math.max(0.5, Math.min(12, Number(src.radius) || 1)),
                mass: Math.max(0.1, Math.min(2.5, Number(src.mass) || 0.5)),
                nodeType,
                movementType,
                dyeColor: Array.isArray(src.dyeColor) ? src.dyeColor.slice(0, 3) : [200, 50, 50],
                canBeGrabber: Boolean(src.canBeGrabber),
                neuronDataBlueprint: nodeType === NodeType.NEURON
                    ? {
                        hiddenLayerSize: Math.max(
                            config.DEFAULT_HIDDEN_LAYER_SIZE_MIN,
                            Math.min(
                                config.DEFAULT_HIDDEN_LAYER_SIZE_MAX,
                                Number(src?.neuronDataBlueprint?.hiddenLayerSize) || config.DEFAULT_HIDDEN_LAYER_SIZE_MIN
                            )
                        )
                    }
                    : null,
                activationIntervalGene: this._sanitizeActivationIntervalGene(src.activationIntervalGene ?? this._randomActivationIntervalGene()),
                predatorRadiusGene: this._sanitizePredatorRadiusGene(src.predatorRadiusGene ?? this._randomPredatorRadiusGene()),
                eyeTargetType: nodeType === NodeType.EYE
                    ? (Number(src.eyeTargetType) === EyeTargetType.FOREIGN_BODY_POINT
                        ? EyeTargetType.FOREIGN_BODY_POINT
                        : EyeTargetType.PARTICLE)
                    : undefined
            };

            const newIndex = this.blueprintPoints.length;
            this.blueprintPoints.push(point);
            mapOldToNew.set(oldIndex, newIndex);
        }

        let addedSprings = 0;

        for (const donorSpring of donor.blueprintSprings || []) {
            const a = Math.floor(Number(donorSpring?.p1Index));
            const b = Math.floor(Number(donorSpring?.p2Index));
            if (!mapOldToNew.has(a) || !mapOldToNew.has(b)) continue;

            const p1Index = mapOldToNew.get(a);
            const p2Index = mapOldToNew.get(b);
            if (p1Index === p2Index) continue;

            this.blueprintSprings.push({
                p1Index,
                p2Index,
                restLength: Math.max(1, Number(donorSpring?.restLength) || 1),
                isRigid: Boolean(donorSpring?.isRigid),
                stiffness: Math.max(100, Math.min(10000, Number(donorSpring?.stiffness) || this.stiffness)),
                damping: Math.max(0.1, Math.min(50, Number(donorSpring?.damping) || this.springDamping)),
                activationIntervalGene: this._sanitizeActivationIntervalGene(donorSpring?.activationIntervalGene ?? this._randomActivationIntervalGene())
            });
            addedSprings += 1;
        }

        const attachmentTargetCount = Math.max(1, Math.floor(Number(config.HGT_GRAFT_ATTACHMENT_SPRINGS) || 1));
        const preGraftIndices = Array.from({ length: startPointCount }, (_, i) => i);
        const graftIndices = Array.from(mapOldToNew.values());

        const usedAttachmentPairs = new Set();
        for (let i = 0; i < attachmentTargetCount && preGraftIndices.length > 0 && graftIndices.length > 0; i++) {
            const baseIdx = preGraftIndices[Math.floor(Math.random() * preGraftIndices.length)];
            const graftIdx = graftIndices[Math.floor(Math.random() * graftIndices.length)];
            if (baseIdx === graftIdx) continue;

            const edgeKey = baseIdx < graftIdx ? `${baseIdx}:${graftIdx}` : `${graftIdx}:${baseIdx}`;
            if (usedAttachmentPairs.has(edgeKey)) continue;
            usedAttachmentPairs.add(edgeKey);

            const pA = this.blueprintPoints[baseIdx];
            const pB = this.blueprintPoints[graftIdx];
            if (!pA || !pB) continue;
            const dx = (Number(pA.relX) || 0) - (Number(pB.relX) || 0);
            const dy = (Number(pA.relY) || 0) - (Number(pB.relY) || 0);
            const dist = Math.sqrt(dx * dx + dy * dy);

            const pAGene = this._sanitizeActivationIntervalGene(pA.activationIntervalGene ?? this._randomActivationIntervalGene());
            const pBGene = this._sanitizeActivationIntervalGene(pB.activationIntervalGene ?? this._randomActivationIntervalGene());

            this.blueprintSprings.push({
                p1Index: baseIdx,
                p2Index: graftIdx,
                restLength: Math.max(1, dist || 1),
                isRigid: Math.random() < (config.CHANCE_FOR_RIGID_SPRING * 0.5),
                stiffness: this.stiffness,
                damping: this.springDamping,
                activationIntervalGene: this._sanitizeActivationIntervalGene((pAGene + pBGene) * 0.5)
            });
            addedSprings += 1;
        }

        if (addedSprings <= 0) {
            this.blueprintPoints.length = startPointCount;
            this.blueprintSprings.length = startSpringCount;
            this._bumpMutationStat('hgtDonorGraftRejectedInvalid');
            return false;
        }

        this._bumpMutationStat('hgtDonorGraftApplied');
        this._bumpMutationStat('pointAddActual', this.blueprintPoints.length - startPointCount);
        this._bumpMutationStat('springAddition', this.blueprintSprings.length - startSpringCount);
        return true;
    }

    /**
     * Recompute cached node-type counters used by brain sizing and diagnostics.
     */
    _recountNodeTypeCaches() {
        this.numEmitterNodes = 0;
        this.numSwimmerNodes = 0;
        this.numEaterNodes = 0;
        this.numPredatorNodes = 0;
        this.numEyeNodes = 0;
        this.numJetNodes = 0;
        this.numPotentialGrabberNodes = 0;
        this.numAttractorNodes = 0;
        this.numRepulsorNodes = 0;

        this.primaryEyePoint = null;
        for (const p of this.massPoints) {
            p.isDesignatedEye = false;

            if (p.nodeType === NodeType.EMITTER) this.numEmitterNodes++;
            else if (p.nodeType === NodeType.SWIMMER) this.numSwimmerNodes++;
            else if (p.nodeType === NodeType.EATER) this.numEaterNodes++;
            else if (p.nodeType === NodeType.PREDATOR) this.numPredatorNodes++;
            else if (p.nodeType === NodeType.EYE) {
                this.numEyeNodes++;
                if (!this.primaryEyePoint) this.primaryEyePoint = p;
            }
            else if (p.nodeType === NodeType.JET) this.numJetNodes++;
            else if (p.nodeType === NodeType.ATTRACTOR) this.numAttractorNodes++;
            else if (p.nodeType === NodeType.REPULSOR) this.numRepulsorNodes++;

            if (p.canBeGrabber) this.numPotentialGrabberNodes++;
        }

        if (this.primaryEyePoint) {
            this.primaryEyePoint.isDesignatedEye = true;
        }
    }

    /**
     * Mark this body as unstable once, preserving first failure reason for telemetry.
     */
    _markUnstable(reason = 'unknown', details = null) {
        if (this.isUnstable) return;
        this.isUnstable = true;
        this.unstableReason = reason;
        this.unstableReasonDetails = details ? JSON.parse(JSON.stringify(details)) : null;
    }

    /**
     * Keep blueprintRadius conservative when phenotype grows at runtime.
     */
    _updateBlueprintRadiusFromCurrentPhenotype() {
        if (!this.massPoints || this.massPoints.length === 0) return;
        const center = this.getAveragePosition();
        let maxDistSq = 0;
        for (const p of this.massPoints) {
            const dx = p.pos.x - center.x;
            const dy = p.pos.y - center.y;
            const d = Math.sqrt(dx * dx + dy * dy) + (p.radius || 0);
            const dSq = d * d;
            if (dSq > maxDistSq) maxDistSq = dSq;
        }
        this.blueprintRadius = Math.max(this.blueprintRadius || 0, Math.sqrt(maxDistSq));
    }

    /**
     * Growth pass: probabilistically add one or more nodes with heritable preferences.
     */
    _attemptGrowthStep(dt) {
        if (!config.GROWTH_ENABLED || this.isUnstable || this.massPoints.length === 0) return false;

        // Telemetry: distinguish hard point-cap suppression from other non-growth outcomes.
        if (this.massPoints.length >= config.GROWTH_MAX_POINTS_PER_CREATURE) {
            this.growthSuppressedByMaxPoints += 1;
            return false;
        }

        if (this.growthCooldownRemaining > 0) {
            this.growthCooldownRemaining--;
            this.growthSuppressedByCooldown += 1;
            return false;
        }

        const genome = this._sanitizeGrowthGenome(this.growthGenome || this._createRandomGrowthGenome());
        this.growthGenome = genome;

        const energyRatio = this.currentMaxEnergy > 0 ? this.creatureEnergy / this.currentMaxEnergy : 0;
        if (energyRatio < genome.minEnergyRatioToGrow) {
            this.growthSuppressedByEnergy += 1;
            return false;
        }

        const population = Array.isArray(runtimeState.softBodyPopulation)
            ? runtimeState.softBodyPopulation.length
            : config.CREATURE_POPULATION_FLOOR;
        const popThrottle = computeGrowthPopulationThrottle({
            population,
            floor: config.CREATURE_POPULATION_FLOOR,
            ceiling: config.CREATURE_POPULATION_CEILING,
            softLimitMultiplier: config.GROWTH_POP_SOFT_LIMIT_MULTIPLIER,
            hardLimitMultiplier: config.GROWTH_POP_HARD_LIMIT_MULTIPLIER,
            minThrottleScale: config.GROWTH_MIN_THROTTLE_SCALE
        });

        if (!popThrottle.allowGrowth) {
            this.growthSuppressedByPopulation += 1;
            return false;
        }

        const dtScale = Math.max(0.1, dt * 60);
        const baseChance = Math.max(0, Math.min(1, genome.growthChancePerTick));
        const dyeState = this._getDyeEcologyStateAtBodyCenter();
        const dyeGrowthScale = this._resolveDyeEffectScale(dyeState, {
            weight: Math.max(0, Number(config.DYE_GROWTH_EFFECT_WEIGHT) || 0.7)
        });
        const growthChance = (1 - Math.pow(1 - baseChance, dtScale)) * popThrottle.scale * dyeGrowthScale;
        if (Math.random() >= growthChance) {
            this.growthSuppressedByChanceRoll += 1;
            if (dyeGrowthScale < 1) this.growthSuppressedByDye += 1;
            return false;
        }

        const nodesPlan = this._sampleWeightedEntry(genome.nodesPerGrowthWeights, { count: 1 });
        const maxNodesByPlan = Math.max(1, Math.floor(nodesPlan?.count || 1));
        const maxNodesByCapacity = Math.max(0, config.GROWTH_MAX_POINTS_PER_CREATURE - this.massPoints.length);
        const targetNodeAdds = Math.min(maxNodesByPlan, maxNodesByCapacity);
        if (targetNodeAdds <= 0) {
            this.growthSuppressedByNoCapacity += 1;
            return false;
        }

        const preGrowthPoints = this.massPoints.slice();
        const startPointCount = this.massPoints.length;
        const startSpringCount = this.springs.length;
        let totalEdgeLength = 0;
        let nodesAdded = 0;

        const movementChoices = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];

        for (let i = 0; i < targetNodeAdds; i++) {
            let placed = false;

            for (let attempt = 0; attempt < config.GROWTH_PLACEMENT_ATTEMPTS_PER_NODE; attempt++) {
                const anchorType = this._sampleWeightedEntry(genome.anchorNodeTypeWeights, null)?.nodeType;
                const typedAnchors = preGrowthPoints.filter((p) => p.nodeType === anchorType);
                const anchorPool = typedAnchors.length > 0 ? typedAnchors : preGrowthPoints;
                const anchor = anchorPool[Math.floor(Math.random() * anchorPool.length)];
                if (!anchor) break;

                const distanceBucket = this._sampleWeightedEntry(genome.distanceRangeWeights, null) || { min: config.GROWTH_DISTANCE_MIN, max: config.GROWTH_DISTANCE_MAX };
                const distance = distanceBucket.min + Math.random() * Math.max(0.001, (distanceBucket.max - distanceBucket.min));
                const angle = Math.random() * Math.PI * 2;

                let x = anchor.pos.x + Math.cos(angle) * distance;
                let y = anchor.pos.y + Math.sin(angle) * distance;

                if (config.IS_WORLD_WRAPPING) {
                    x = ((x % config.WORLD_WIDTH) + config.WORLD_WIDTH) % config.WORLD_WIDTH;
                    y = ((y % config.WORLD_HEIGHT) + config.WORLD_HEIGHT) % config.WORLD_HEIGHT;
                } else {
                    if (x < 0 || x > config.WORLD_WIDTH || y < 0 || y > config.WORLD_HEIGHT) continue;
                }

                const nodeType = this._sampleWeightedEntry(genome.newNodeTypeWeights, null)?.nodeType;
                if (nodeType === undefined || nodeType === null) continue;

                const radius = Math.max(0.5, Math.min(anchor.radius * (0.75 + Math.random() * 0.6), anchor.radius * 1.5));
                const mass = Math.max(0.1, Math.min(anchor.mass * (0.75 + Math.random() * 0.6), 1.5));

                const minClearanceFactor = Math.max(1.0, config.GROWTH_MIN_POINT_CLEARANCE_FACTOR);
                let collides = false;
                for (const existing of this.massPoints) {
                    const dx = x - existing.pos.x;
                    const dy = y - existing.pos.y;
                    const minDist = (radius + existing.radius) * minClearanceFactor;
                    if (dx * dx + dy * dy < minDist * minDist) {
                        collides = true;
                        break;
                    }
                }
                if (collides) continue;

                const newPoint = new MassPoint(x, y, mass, radius);
                newPoint.nodeType = nodeType;
                newPoint.movementType = nodeType === NodeType.SWIMMER
                    ? MovementType.NEUTRAL
                    : movementChoices[Math.floor(Math.random() * movementChoices.length)];
                newPoint.dyeColor = [
                    Math.floor(Math.random() * 255),
                    Math.floor(Math.random() * 255),
                    Math.floor(Math.random() * 255)
                ];
                newPoint.canBeGrabber = Math.random() < 0.15;

                // Growth-genome driven actuation interval inheritance:
                // anchor gene + heritable bias + jitter.
                const intervalJitter = (Math.random() - 0.5) * 2 * (genome.activationIntervalJitter || 0);
                const inheritedNodeInterval = this._sanitizeActivationIntervalGene(
                    (anchor.activationIntervalGene ?? this._randomActivationIntervalGene()) +
                    (genome.nodeActivationIntervalBias || 0) +
                    intervalJitter
                );
                newPoint.activationIntervalGene = inheritedNodeInterval;
                const parentPredatorRadiusGene = this._sanitizePredatorRadiusGene(anchor.predatorRadiusGene ?? this._randomPredatorRadiusGene());
                newPoint.predatorRadiusGene = this._sanitizePredatorRadiusGene(
                    parentPredatorRadiusGene * (0.9 + Math.random() * 0.2)
                );
                newPoint.actuationCooldownByChannel = { node: 0, grabber: 0, default_pattern: 0 };
                newPoint.swimmerActuation = { magnitude: 0, angle: 0 };
                newPoint.eyeTargetType = Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT;
                newPoint.maxEffectiveJetVelocity = this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4);

                if (nodeType === NodeType.NEURON) {
                    newPoint.neuronData = {
                        isBrain: false,
                        hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * Math.max(1, (config.DEFAULT_HIDDEN_LAYER_SIZE_MAX - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))),
                        sensorPointIndex: -1,
                        effectorPointIndex: -1
                    };
                } else {
                    newPoint.neuronData = null;
                }

                let closest = null;
                let closestDistSq = Infinity;
                for (const p of preGrowthPoints) {
                    const dx = newPoint.pos.x - p.pos.x;
                    const dy = newPoint.pos.y - p.pos.y;
                    const dSq = dx * dx + dy * dy;
                    if (dSq < closestDistSq) {
                        closestDistSq = dSq;
                        closest = p;
                    }
                }
                if (!closest || !Number.isFinite(closestDistSq) || closestDistSq <= 0) continue;

                this.massPoints.push(newPoint);

                const edgeType = this._sampleWeightedEntry(genome.edgeTypeWeights, { type: 'soft' })?.type || 'soft';
                const restLength = Math.sqrt(closestDistSq);
                const stiffness = Math.max(100, this.stiffness * genome.edgeStiffnessScale * (0.8 + Math.random() * 0.4));
                const damping = Math.max(0.1, this.springDamping * genome.edgeDampingScale * (0.8 + Math.random() * 0.4));
                const spring = new Spring(closest, newPoint, stiffness, damping, restLength, edgeType === 'rigid');
                const parentIntervalA = closest.activationIntervalGene ?? this._randomActivationIntervalGene();
                const parentIntervalB = newPoint.activationIntervalGene ?? this._randomActivationIntervalGene();
                const edgeIntervalBase = (parentIntervalA + parentIntervalB) * 0.5;
                const edgeIntervalJitter = (Math.random() - 0.5) * 2 * (genome.activationIntervalJitter || 0);
                spring.activationIntervalGene = this._sanitizeActivationIntervalGene(
                    edgeIntervalBase + (genome.edgeActivationIntervalBias || 0) + edgeIntervalJitter
                );
                spring.actuationCooldownByChannel = { edge: 0 };
                this.springs.push(spring);

                totalEdgeLength += restLength;
                nodesAdded++;
                placed = true;
                break;
            }

            if (!placed) {
                // Continue attempting remaining nodes; partial growth is allowed if at least one node lands.
                continue;
            }
        }

        if (nodesAdded <= 0) {
            this.growthSuppressedByPlacement += 1;
            return false;
        }

        const edgesAdded = this.springs.length - startSpringCount;
        const sizeCostMultiplier = computeGrowthSizeCostMultiplier({
            currentPoints: this.massPoints.length,
            maxPoints: config.GROWTH_MAX_POINTS_PER_CREATURE,
            exponent: config.GROWTH_SIZE_COST_EXPONENT,
            maxMultiplier: config.GROWTH_SIZE_COST_MAX_MULTIPLIER
        });
        const growthCost = config.GROWTH_ENERGY_COST_SCALAR * sizeCostMultiplier * (
            nodesAdded * config.GROWTH_COST_PER_NODE +
            edgesAdded * config.GROWTH_COST_PER_EDGE +
            totalEdgeLength * config.GROWTH_COST_PER_EDGE_LENGTH
        );

        if (this.creatureEnergy < growthCost) {
            // Roll back fully if growth budget was insufficient.
            this.massPoints.length = startPointCount;
            this.springs.length = startSpringCount;
            this.growthSuppressedByEnergy += 1;
            return false;
        }

        this.creatureEnergy -= growthCost;
        this.totalGrowthEnergySpent += growthCost;
        this.growthCooldownRemaining = Math.max(1, Math.floor(genome.growthCooldownTicks));
        this.growthEventsCompleted += 1;
        this.growthNodesAdded += nodesAdded;

        this._enforcePhotosyntheticPhenotypeConstraints();
        this.calculateCurrentMaxEnergy();
        this.effectiveReproductionCooldown = Math.floor(this.reproductionCooldownGene * (1 + (0.2 * Math.max(0, this.massPoints.length - 1))));
        this._recountNodeTypeCaches();
        this.initializeBrain();
        this._updateBlueprintRadiusFromCurrentPhenotype();

        return true;
    }

    createShape(startX, startY, parentBody = null) {
        this.massPoints = []; // Clear actual points
        this.springs = [];  // Clear actual springs
        this.blueprintPoints = []; // Clear any previous blueprint
        this.blueprintSprings = [];// Clear any previous blueprint

        const baseRadius = 1 + Math.random() * 1; 
        const availableFunctionalNodeTypes = [NodeType.PREDATOR, NodeType.EATER, NodeType.PHOTOSYNTHETIC, NodeType.NEURON, NodeType.EMITTER, NodeType.SWIMMER, NodeType.EYE, NodeType.JET, NodeType.ATTRACTOR, NodeType.REPULSOR];
        const dyeColorChoices = [config.DYE_COLORS.RED, config.DYE_COLORS.GREEN, config.DYE_COLORS.BLUE];

        const edgeStiffnessMin = Number.isFinite(Number(config.NEW_EDGE_STIFFNESS_MIN)) ? Number(config.NEW_EDGE_STIFFNESS_MIN) : 500;
        const edgeStiffnessMax = Number.isFinite(Number(config.NEW_EDGE_STIFFNESS_MAX))
            ? Math.max(edgeStiffnessMin, Number(config.NEW_EDGE_STIFFNESS_MAX))
            : 3000;
        const edgeDampingMin = Number.isFinite(Number(config.NEW_EDGE_DAMPING_MIN)) ? Number(config.NEW_EDGE_DAMPING_MIN) : 5;
        const edgeDampingMax = Number.isFinite(Number(config.NEW_EDGE_DAMPING_MAX))
            ? Math.max(edgeDampingMin, Number(config.NEW_EDGE_DAMPING_MAX))
            : 25;
        const edgeRestLengthVariation = Math.max(0, Number(config.NEW_SPRING_REST_LENGTH_VARIATION) || 0);
        const edgeRestLengthBias = Number.isFinite(Number(config.NEW_SPRING_REST_LENGTH_BIAS))
            ? Math.max(0.2, Number(config.NEW_SPRING_REST_LENGTH_BIAS))
            : 1;
        const edgeRestLengthMax = Number.isFinite(Number(config.NEW_SPRING_REST_LENGTH_MAX))
            ? Math.max(1, Number(config.NEW_SPRING_REST_LENGTH_MAX))
            : Infinity;

        const drawRandomEdgeStiffness = (scale = 1) => {
            const s = Math.max(0.1, Number(scale) || 1);
            return (edgeStiffnessMin + Math.random() * (edgeStiffnessMax - edgeStiffnessMin)) * s;
        };
        const drawRandomEdgeDamping = () => edgeDampingMin + Math.random() * (edgeDampingMax - edgeDampingMin);
        const resolveEdgeRestLength = (geometricDistance, jitterScale = 1) => {
            const dist = Number(geometricDistance);
            if (!Number.isFinite(dist) || dist <= 0) return 1;
            const jitter = 1 + (Math.random() - 0.5) * 2 * edgeRestLengthVariation * Math.max(0, Number(jitterScale) || 0);
            const biased = dist * edgeRestLengthBias * jitter;
            return Math.max(1, Math.min(edgeRestLengthMax, biased));
        };

        if (parentBody) {
            // --- Reproduction: Inherit and Mutate Blueprint ---

            // 1. Deep copy blueprint from parent
            this.blueprintPoints = JSON.parse(JSON.stringify(parentBody.blueprintPoints));
            this.blueprintSprings = JSON.parse(JSON.stringify(parentBody.blueprintSprings));
            const parentBlueprintSnapshot = this._cloneBlueprintSnapshot(this.blueprintPoints, this.blueprintSprings);

            // 2. Mutate blueprint points (coordinates, types, properties)
            this.blueprintPoints.forEach(bp => {
                bp.activationIntervalGene = this._sanitizeActivationIntervalGene(
                    bp.activationIntervalGene ?? this._randomActivationIntervalGene()
                );
                bp.predatorRadiusGene = this._sanitizePredatorRadiusGene(
                    bp.predatorRadiusGene ?? this._randomPredatorRadiusGene()
                );

                // Mutate relative coordinates
                if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.5)) {
                    bp.relX += (Math.random() - 0.5) * 2; // Smaller jitter for blueprint stability
                    runtimeState.mutationStats.blueprintCoordinateChange = (runtimeState.mutationStats.blueprintCoordinateChange || 0) + 1;
                }
                if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER * 0.5)) {
                    bp.relY += (Math.random() - 0.5) * 2;
                    runtimeState.mutationStats.blueprintCoordinateChange = (runtimeState.mutationStats.blueprintCoordinateChange || 0) + 1;
                }

                // Mutate mass & radius
                if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.mass = Math.max(0.1, Math.min(bp.mass * (1 + (Math.random() - 0.5) * 0.4), 1.0));
                    runtimeState.mutationStats.blueprintMassRadiusChange = (runtimeState.mutationStats.blueprintMassRadiusChange || 0) + 1;
                }
                if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.radius = Math.max(0.5, Math.min(bp.radius * (1 + (Math.random() - 0.5) * 0.4), baseRadius * 2.5)); // Max based on baseRadius
                     runtimeState.mutationStats.blueprintMassRadiusChange = (runtimeState.mutationStats.blueprintMassRadiusChange || 0) + 1;
                }

                // Mutate nodeType
                if (Math.random() < (config.MUTATION_CHANCE_NODE_TYPE * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    const oldNodeType = bp.nodeType;
                    if (Math.random() < config.NEURON_CHANCE) {
                        bp.nodeType = NodeType.NEURON;
                    } else {
                        const otherNodeTypes = availableFunctionalNodeTypes.filter(t => t !== NodeType.NEURON);
                        bp.nodeType = otherNodeTypes[Math.floor(Math.random() * otherNodeTypes.length)];
                    }
                    if (bp.nodeType !== oldNodeType) runtimeState.mutationStats.nodeTypeChange++;

                    // If it becomes an EYE, initialize eyeTargetType randomly
                    if (bp.nodeType === NodeType.EYE && bp.eyeTargetType === undefined) {
                        bp.eyeTargetType = Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT;
                    }
                }

                // Mutate movementType
                if (Math.random() < (config.MUTATION_CHANCE_NODE_TYPE * config.GLOBAL_MUTATION_RATE_MODIFIER)) { 
                    const oldMovementType = bp.movementType;
                    const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                    bp.movementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                    if (bp.movementType !== oldMovementType) runtimeState.mutationStats.movementTypeChange++;
                }
                // Ensure Swimmer nodes are always Neutral type
                if (bp.nodeType === NodeType.SWIMMER) {
                    bp.movementType = MovementType.NEUTRAL;
                }

                // Mutate canBeGrabber gene
                if (Math.random() < config.GRABBER_GENE_MUTATION_CHANCE) {
                    bp.canBeGrabber = !bp.canBeGrabber;
                    runtimeState.mutationStats.grabberGeneChange++;
                }

                const intervalMutation = this._mutateActivationIntervalGene(bp.activationIntervalGene);
                bp.activationIntervalGene = intervalMutation.gene;
                if (intervalMutation.didMutate) {
                    runtimeState.mutationStats.activationIntervalGene = (runtimeState.mutationStats.activationIntervalGene || 0) + 1;
                }

                const predatorRadiusMutation = this._mutatePredatorRadiusGene(bp.predatorRadiusGene);
                bp.predatorRadiusGene = predatorRadiusMutation.gene;
                if (predatorRadiusMutation.didMutate) {
                    runtimeState.mutationStats.predatorRadiusGene = (runtimeState.mutationStats.predatorRadiusGene || 0) + 1;
                }

                // Mutate dyeColor
                if (Math.random() < (config.MUTATION_CHANCE_BOOL * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    bp.dyeColor = dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)];
                    runtimeState.mutationStats.blueprintDyeColorChange = (runtimeState.mutationStats.blueprintDyeColorChange || 0) + 1;
                }

                // Mutate neuronDataBlueprint (specifically hiddenLayerSize if neuron)
                if (bp.nodeType === NodeType.NEURON) {
                    if (!bp.neuronDataBlueprint) { // Ensure it exists
                        bp.neuronDataBlueprint = { hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MAX - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) };
                    }
                    if (Math.random() < (config.MUTATION_RATE_PERCENT * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                        let newSize = bp.neuronDataBlueprint.hiddenLayerSize + Math.floor((Math.random() * 6) - 3); // Mutate by +/- up to 3
                        bp.neuronDataBlueprint.hiddenLayerSize = Math.max(config.DEFAULT_HIDDEN_LAYER_SIZE_MIN, Math.min(newSize, config.DEFAULT_HIDDEN_LAYER_SIZE_MAX));
                        runtimeState.mutationStats.blueprintNeuronHiddenSizeChange = (runtimeState.mutationStats.blueprintNeuronHiddenSizeChange || 0) + 1;
                    }
                } else {
                    bp.neuronDataBlueprint = null; // Crucial: ensure non-neurons have null neuronDataBlueprint
                }

                // Mutate eyeTargetType if it's an EYE node
                if (bp.nodeType === NodeType.EYE && bp.eyeTargetType !== undefined && Math.random() < config.EYE_TARGET_TYPE_MUTATION_CHANCE) {
                    const oldEyeTargetType = bp.eyeTargetType;
                    bp.eyeTargetType = (bp.eyeTargetType === EyeTargetType.PARTICLE) ? EyeTargetType.FOREIGN_BODY_POINT : EyeTargetType.PARTICLE;
                    if (bp.eyeTargetType !== oldEyeTargetType) {
                        runtimeState.mutationStats.eyeTargetTypeChange = (runtimeState.mutationStats.eyeTargetTypeChange || 0) + 1;
                    }
                }
            });

            // 3. Mutate blueprint springs (restLength, isRigid)
            this.blueprintSprings.forEach(bs => {
                bs.activationIntervalGene = this._sanitizeActivationIntervalGene(
                    bs.activationIntervalGene ?? this._randomActivationIntervalGene()
                );
                const edgeIntervalMutation = this._mutateActivationIntervalGene(bs.activationIntervalGene);
                bs.activationIntervalGene = edgeIntervalMutation.gene;
                if (edgeIntervalMutation.didMutate) {
                    runtimeState.mutationStats.edgeActivationIntervalGene = (runtimeState.mutationStats.edgeActivationIntervalGene || 0) + 1;
                }

                if (Math.random() < (config.SPRING_PROP_MUTATION_MAGNITUDE * config.GLOBAL_MUTATION_RATE_MODIFIER)) { // Use magnitude as chance here
                    const oldRestLength = bs.restLength;
                    bs.restLength = Math.max(1, bs.restLength * (1 + (Math.random() - 0.5) * 2 * config.SPRING_PROP_MUTATION_MAGNITUDE));
                    if (Math.abs(bs.restLength - oldRestLength) > 0.01) runtimeState.mutationStats.springRestLength++;
                }
                if (Math.random() < (config.MUTATION_CHANCE_BOOL * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    const oldRigid = bs.isRigid;
                    bs.isRigid = !bs.isRigid; // Simple flip for now
                    if (bs.isRigid !== oldRigid) runtimeState.mutationStats.springRigidityFlip++;
                }
                 if (Math.random() < (config.SPRING_PROP_MUTATION_MAGNITUDE * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    if (typeof bs.stiffness === 'undefined') bs.stiffness = this.stiffness;
                    const oldStiffness = bs.stiffness;
                    bs.stiffness = Math.max(100, Math.min(bs.stiffness * (1 + (Math.random() - 0.5) * 2 * config.SPRING_PROP_MUTATION_MAGNITUDE), 10000));
                    if (Math.abs(bs.stiffness - oldStiffness) > 0.01) runtimeState.mutationStats.springStiffness++;
                }
                if (Math.random() < (config.SPRING_PROP_MUTATION_MAGNITUDE * config.GLOBAL_MUTATION_RATE_MODIFIER)) {
                    if (typeof bs.damping === 'undefined') bs.damping = this.springDamping;
                    const oldDamping = bs.damping;
                    bs.damping = Math.max(0.1, Math.min(bs.damping * (1 + (Math.random() - 0.5) * 2 * config.SPRING_PROP_MUTATION_MAGNITUDE), 50));
                    if (Math.abs(bs.damping - oldDamping) > 0.01) runtimeState.mutationStats.springDamping++;
                }
            });

            // Step 4: Structural Blueprint Mutations (Point Add, Spring Add/Delete, Subdivision, Scale, etc.)
            // --- Point Addition Mutation (Blueprint) ---
            if (Math.random() < this.pointAddChance * config.GLOBAL_MUTATION_RATE_MODIFIER && this.blueprintPoints.length > 0) {
                const lastBp = this.blueprintPoints[this.blueprintPoints.length - 1];
                const newRelX = lastBp.relX + (Math.random() - 0.5) * config.NEW_POINT_OFFSET_RADIUS * 0.5; // Smaller offset for blueprint
                const newRelY = lastBp.relY + (Math.random() - 0.5) * config.NEW_POINT_OFFSET_RADIUS * 0.5;
                const newMass = 0.1 + Math.random() * 0.9;
                const newRadius = baseRadius * (0.8 + Math.random() * 0.4);

                let newNodeType;
                if (Math.random() < config.NEURON_CHANCE) {
                    newNodeType = NodeType.NEURON;
                } else {
                    const otherNodeTypes = availableFunctionalNodeTypes.filter(t => t !== NodeType.NEURON);
                    newNodeType = otherNodeTypes[Math.floor(Math.random() * otherNodeTypes.length)];
                }

                const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                let newMovementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                if (newNodeType === NodeType.SWIMMER) {
                    newMovementType = MovementType.NEUTRAL;
                }
                const newBp = {
                    relX: newRelX, relY: newRelY, radius: newRadius, mass: newMass,
                    nodeType: newNodeType, movementType: newMovementType,
                    dyeColor: dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)],
                    canBeGrabber: Math.random() < config.GRABBER_GENE_MUTATION_CHANCE,
                    neuronDataBlueprint: newNodeType === NodeType.NEURON ? { hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MAX - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) } : null,
                    activationIntervalGene: this._randomActivationIntervalGene(),
                    predatorRadiusGene: this._randomPredatorRadiusGene(),
                    eyeTargetType: newNodeType === NodeType.EYE ? (Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT) : undefined,
                    maxEffectiveJetVelocity: this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4)
                };
                this.blueprintPoints.push(newBp);
                const newPointIndex = this.blueprintPoints.length - 1;
                runtimeState.mutationStats.pointAddActual++;

                // Connect new blueprint point with springs
                const numSpringsToAddNewPoint = config.MIN_SPRINGS_PER_NEW_NODE + Math.floor(Math.random() * (config.MAX_SPRINGS_PER_NEW_NODE - config.MIN_SPRINGS_PER_NEW_NODE + 1));
                const existingBpIndices = this.blueprintPoints.map((_, i) => i).filter(i => i !== newPointIndex);
                const shuffledExistingBpIndices = existingBpIndices.sort(() => 0.5 - Math.random());

                for (let k = 0; k < Math.min(numSpringsToAddNewPoint, shuffledExistingBpIndices.length); k++) {
                    const connectToBpIndex = shuffledExistingBpIndices[k];
                    const connectToBp = this.blueprintPoints[connectToBpIndex];
                    const dist = Math.sqrt((newBp.relX - connectToBp.relX)**2 + (newBp.relY - connectToBp.relY)**2);
                    const newRestLength = resolveEdgeRestLength(dist, 1);
                    const becomeRigid = Math.random() < config.CHANCE_FOR_RIGID_SPRING;
                    const newStiffness = drawRandomEdgeStiffness();
                    const newDamping = drawRandomEdgeDamping();
                    this.blueprintSprings.push({ p1Index: newPointIndex, p2Index: connectToBpIndex, restLength: newRestLength, isRigid: becomeRigid, stiffness: newStiffness, damping: newDamping, activationIntervalGene: this._randomActivationIntervalGene() });
                }
            }

            // --- Spring Deletion Mutation (Blueprint) ---
            if (this.blueprintSprings.length > this.blueprintPoints.length -1 && Math.random() < config.SPRING_DELETION_CHANCE) { // Ensure min connectivity
            // Ensure min connectivity
                // Complex check to avoid orphaning needed here for blueprint springs
                // For now, simple random deletion if enough springs exist
                const springToDeleteIndex = Math.floor(Math.random() * this.blueprintSprings.length);
                this.blueprintSprings.splice(springToDeleteIndex, 1);
                runtimeState.mutationStats.springDeletion++;
            }

            // --- Spring Addition Mutation (Blueprint) ---
            if (this.blueprintPoints.length >= 2 && Math.random() < config.SPRING_ADDITION_CHANCE) {
                let attempts = 0;
                while (attempts < 10) {
                    const idx1 = Math.floor(Math.random() * this.blueprintPoints.length);
                    let idx2 = Math.floor(Math.random() * this.blueprintPoints.length);
                    if (idx1 === idx2 && this.blueprintPoints.length > 1) {
                        idx2 = (idx1 + 1) % this.blueprintPoints.length;
                    }
                    if (idx1 === idx2) break;

                    const pA_bp = this.blueprintPoints[idx1];
                    const pB_bp = this.blueprintPoints[idx2];
                    let alreadyConnected = this.blueprintSprings.some(bs => 
                        (bs.p1Index === idx1 && bs.p2Index === idx2) || (bs.p1Index === idx2 && bs.p2Index === idx1)
                    );
                    if (!alreadyConnected) {
                        const dist = Math.sqrt((pA_bp.relX - pB_bp.relX)**2 + (pA_bp.relY - pB_bp.relY)**2);
                        const newRestLength = resolveEdgeRestLength(dist, 1);
                        const becomeRigid = Math.random() < config.CHANCE_FOR_RIGID_SPRING;
                        const newStiffness = drawRandomEdgeStiffness();
                        const newDamping = drawRandomEdgeDamping();
                        this.blueprintSprings.push({ p1Index: idx1, p2Index: idx2, restLength: newRestLength, isRigid: becomeRigid, stiffness: newStiffness, damping: newDamping, activationIntervalGene: this._randomActivationIntervalGene() });
                        runtimeState.mutationStats.springAddition++; 
                        break;
                    }
                    attempts++;
                }
            }

            // --- Spring Subdivision Mutation (Blueprint) ---
            if (this.blueprintSprings.length > 0 && Math.random() < config.SPRING_SUBDIVISION_MUTATION_CHANCE) {
                const springToSubdivideIndex = Math.floor(Math.random() * this.blueprintSprings.length);
                const originalBs = this.blueprintSprings[springToSubdivideIndex];
                    const bp1 = this.blueprintPoints[originalBs.p1Index];
                    const bp2 = this.blueprintPoints[originalBs.p2Index];

                    const midRelX = (bp1.relX + bp2.relX) / 2;
                    const midRelY = (bp1.relY + bp2.relY) / 2;
                    const newRadius = ((bp1.radius || baseRadius) + (bp2.radius || baseRadius)) / 2 * (0.8 + Math.random() * 0.4);
                    const newMass = ((bp1.mass || 0.5) + (bp2.mass || 0.5)) / 2 * (0.8 + Math.random() * 0.4);
                    let newNodeType = availableFunctionalNodeTypes[Math.floor(Math.random() * availableFunctionalNodeTypes.length)];
                    const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                    let newMovementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                    if (newNodeType === NodeType.SWIMMER) {
                        newMovementType = MovementType.NEUTRAL;
                    }

                    const newMidBp = {
                        relX: midRelX, relY: midRelY, radius: Math.max(0.5, newRadius), mass: Math.max(0.1, newMass),
                        nodeType: newNodeType, movementType: newMovementType,
                        dyeColor: dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)],
                        canBeGrabber: Math.random() < config.GRABBER_GENE_MUTATION_CHANCE,
                        neuronDataBlueprint: newNodeType === NodeType.NEURON ? { hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MIN - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1)) } : null,
                        activationIntervalGene: this._randomActivationIntervalGene(),
                        predatorRadiusGene: this._randomPredatorRadiusGene(),
                        eyeTargetType: newNodeType === NodeType.EYE ? (Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT) : undefined,
                        maxEffectiveJetVelocity: this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4)
                    };
                    this.blueprintPoints.push(newMidBp);
                    const newMidPointIndex = this.blueprintPoints.length - 1;

                    this.blueprintSprings.splice(springToSubdivideIndex, 1); // Remove original spring

                    let restLength1 = Math.sqrt((bp1.relX - midRelX)**2 + (bp1.relY - midRelY)**2) * (1 + (Math.random() - 0.5) * 0.1);
                    this.blueprintSprings.push({ p1Index: originalBs.p1Index, p2Index: newMidPointIndex, restLength: Math.max(1, restLength1), isRigid: originalBs.isRigid, stiffness: originalBs.stiffness, damping: originalBs.damping, activationIntervalGene: this._sanitizeActivationIntervalGene(originalBs.activationIntervalGene ?? this._randomActivationIntervalGene()) });

                    let restLength2 = Math.sqrt((midRelX - bp2.relX)**2 + (midRelY - bp2.relY)**2) * (1 + (Math.random() - 0.5) * 0.1);
                    this.blueprintSprings.push({ p1Index: newMidPointIndex, p2Index: originalBs.p2Index, restLength: Math.max(1, restLength2), isRigid: originalBs.isRigid, stiffness: originalBs.stiffness, damping: originalBs.damping, activationIntervalGene: this._sanitizeActivationIntervalGene(originalBs.activationIntervalGene ?? this._randomActivationIntervalGene()) });

                    runtimeState.mutationStats.springSubdivision++; 
                    runtimeState.mutationStats.pointAddActual++; 
            }

            // --- Body Scale Mutation (Blueprint) ---
            if (Math.random() < config.BODY_SCALE_MUTATION_CHANCE) {
                const scaleFactor = 1.0 + (Math.random() - 0.5) * 2 * config.BODY_SCALE_MUTATION_MAGNITUDE;
                if (scaleFactor > 0.1 && Math.abs(scaleFactor - 1.0) > 0.001) {
                    this.blueprintPoints.forEach(bp => {
                        bp.relX *= scaleFactor;
                        bp.relY *= scaleFactor;
                        bp.radius = Math.max(0.5, bp.radius * scaleFactor); 
                    });
                    this.blueprintSprings.forEach(bs => {
                        bs.restLength = Math.max(1, bs.restLength * scaleFactor); 
                    });
                    this.offspringSpawnRadius *= scaleFactor; // This is a creature property, not blueprint, but scale it too
                    this.offspringSpawnRadius = Math.max(10, this.offspringSpawnRadius); 
                    runtimeState.mutationStats.bodyScale++;
                }
            }
            // TODO: Add blueprint versions of Segment Duplication, Symmetrical Duplication etc.

            // --- Shape Addition Mutation (Blueprint) ---
            if (this.blueprintPoints.length >= 2 && Math.random() < 0.02) { // 2% chance
                const blueprintPointsForHull = this.blueprintPoints.map(bp => ({ x: bp.relX, y: bp.relY, originalBlueprintPoint: bp }));
                if (blueprintPointsForHull.length >= 3) {
                    const hullPointsWithOriginal = convexHull(blueprintPointsForHull);
                    if (hullPointsWithOriginal && hullPointsWithOriginal.length >= 2) {
                        const hullIndices = hullPointsWithOriginal.map(p => this.blueprintPoints.indexOf(p.originalBlueprintPoint));

                        let externalEdge = null;
                        let p1_hull_idx, p2_hull_idx;
                        let attempts = 0;
                        let foundValidEdge = false;

                        while (attempts < hullIndices.length * 2 && !foundValidEdge) {
                            let randHullStartIdx = Math.floor(Math.random() * hullIndices.length);
                            p1_hull_idx = hullIndices[randHullStartIdx];
                            p2_hull_idx = hullIndices[(randHullStartIdx + 1) % hullIndices.length];

                            if (this.blueprintSprings.some(bs => (bs.p1Index === p1_hull_idx && bs.p2Index === p2_hull_idx) || (bs.p1Index === p2_hull_idx && bs.p2Index === p1_hull_idx))) {
                                foundValidEdge = true;
                                externalEdge = {
                                    p1: this.blueprintPoints[p1_hull_idx],
                                    p2: this.blueprintPoints[p2_hull_idx],
                                };
                            }
                            attempts++;
                        }

                        if (externalEdge) {
                            const P1_bp = externalEdge.p1;
                            const P2_bp = externalEdge.p2;
                            const p1_idx = this.blueprintPoints.indexOf(P1_bp);
                            const p2_idx = this.blueprintPoints.indexOf(P2_bp);
                            const P1 = new Vec2(P1_bp.relX, P1_bp.relY);
                            const P2 = new Vec2(P2_bp.relX, P2_bp.relY);
                            const sideVector = P2.sub(P1);
                            const sideLength = sideVector.mag();

                            if (sideLength > 1) {
                                let normal = new Vec2(-sideVector.y, sideVector.x).normalize();
                                const midPoint = P1.add(sideVector.mul(0.5));
                                if (Vec2.dot(normal, midPoint) > 0) {
                                    normal.mulInPlace(-1);
                                }

                                const newPoints = [];
                                const shapeType = Math.random() < 0.5 ? 'square' : 'pentagon';
                                let newSpringsInfo = [];

                                if (shapeType === 'square') {
                                    const P4 = P1.add(normal.mul(sideLength));
                                    const P3 = P2.add(normal.mul(sideLength));
                                    newPoints.push(P4, P3);
                                    newSpringsInfo = [
                                        { from: p2_idx, toNew: 1, len: sideLength },
                                        { new1: 1, toNew: 0, len: sideLength },
                                        { new1: 0, to: p1_idx, len: sideLength },
                                        { from: p1_idx, toNew: 1, len: sideLength * Math.SQRT2 }
                                    ];
                                } else { // Pentagon
                                    const angle = -72 * Math.PI / 180;
                                    const P1_minus_P2 = P1.sub(P2);
                                    let P3_vec = new Vec2(P1_minus_P2.x * Math.cos(angle) - P1_minus_P2.y * Math.sin(angle), P1_minus_P2.x * Math.sin(angle) + P1_minus_P2.y * Math.cos(angle));
                                    const P3 = P2.add(P3_vec);

                                    const P2_minus_P3 = P2.sub(P3);
                                    let P4_vec = new Vec2(P2_minus_P3.x * Math.cos(angle) - P2_minus_P3.y * Math.sin(angle), P2_minus_P3.x * Math.sin(angle) + P2_minus_P3.y * Math.cos(angle));
                                    const P4 = P3.add(P4_vec);

                                    const P2_minus_P1 = P2.sub(P1);
                                    let P5_vec = new Vec2(P2_minus_P1.x * Math.cos(-angle) - P2_minus_P1.y * Math.sin(-angle), P2_minus_P1.x * Math.sin(-angle) + P2_minus_P1.y * Math.sin(-angle));
                                    const P5 = P1.add(P5_vec);

                                    newPoints.push(P3, P4, P5);
                                    newSpringsInfo = [
                                        { from: p2_idx, toNew: 0, len: sideLength },
                                        { new1: 0, toNew: 1, len: sideLength },
                                        { new1: 1, toNew: 2, len: sideLength },
                                        { new1: 2, to: p1_idx, len: sideLength },
                                        { from: p1_idx, toNew: 0, len: P3.sub(P1).mag() },
                                        { from: p2_idx, toNew: 2, len: P5.sub(P2).mag() }
                                    ];
                                }

                                const firstNewPointIndex = this.blueprintPoints.length;
                                newPoints.forEach(p_vec => {
                                    this.blueprintPoints.push({
                                        relX: p_vec.x, relY: p_vec.y,
                                        radius: (P1_bp.radius + P2_bp.radius) / 2,
                                        mass: (P1_bp.mass + P2_bp.mass) / 2,
                                        nodeType: P1_bp.nodeType, movementType: P1_bp.movementType,
                                        dyeColor: P1_bp.dyeColor, canBeGrabber: P1_bp.canBeGrabber,
                                        activationIntervalGene: this._sanitizeActivationIntervalGene(P1_bp.activationIntervalGene ?? this._randomActivationIntervalGene()),
                                        predatorRadiusGene: this._sanitizePredatorRadiusGene(P1_bp.predatorRadiusGene ?? this._randomPredatorRadiusGene()),
                                        neuronDataBlueprint: null
                                    });
                                });

                                newSpringsInfo.forEach(info => {
                                    const p1 = info.from !== undefined ? info.from : firstNewPointIndex + info.new1;
                                    const p2 = info.to !== undefined ? info.to : firstNewPointIndex + info.toNew;
                                    this.blueprintSprings.push({ p1Index: p1, p2Index: p2, restLength: info.len, isRigid: false, stiffness: this.stiffness, damping: this.springDamping, activationIntervalGene: this._randomActivationIntervalGene() });
                                });

                                runtimeState.mutationStats.shapeAddition++;
                            }
                        }
                    }
                }
            }
            // TODO: Add blueprint versions of Segment Duplication, Symmetrical Duplication etc.

            // Optional HGT-like donor graft: pull a connected module from a nearby donor blueprint.
            if (Math.random() < Math.max(0, Math.min(1, Number(config.HGT_GRAFT_MUTATION_CHANCE) || 0))) {
                this._attemptDonorModuleGraftMutation(parentBody);
            }

            // Finalize and validate mutated blueprint before instantiation.
            this._sanitizeBlueprintDataInPlace();
            const viability = this._evaluateBlueprintViability();
            if (!viability.ok) {
                if (viability.reasons.structure) this._bumpMutationStat('offspringViabilityRejectedStructure');
                if (viability.reasons.diversity) this._bumpMutationStat('offspringViabilityRejectedDiversity');
                if (viability.reasons.harvest) this._bumpMutationStat('offspringViabilityRejectedHarvest');
                if (viability.reasons.actuator) this._bumpMutationStat('offspringViabilityRejectedActuator');

                this.blueprintPoints = JSON.parse(JSON.stringify(parentBlueprintSnapshot.points));
                this.blueprintSprings = JSON.parse(JSON.stringify(parentBlueprintSnapshot.springs));
                this._sanitizeBlueprintDataInPlace();
                this._bumpMutationStat('offspringViabilityFallbackToParent');
            }

            // Step 5: Instantiate Phenotype from the finalized blueprint
            this._instantiatePhenotypeFromBlueprint(startX, startY);

        } else { 
            // --- Initial Generation: Create Blueprint from Geometric Primitives ---
            let initialTempMassPoints = []; // Temporary MassPoint objects to get initial geometry
            let initialTempSprings = [];  // Temporary Spring objects

            // Create initial geometric shape using stable triangle-mesh primitives when enabled.
            const pointDistanceMin = Number.isFinite(Number(config.INITIAL_BODY_POINT_DISTANCE_MIN))
                ? Number(config.INITIAL_BODY_POINT_DISTANCE_MIN)
                : 5;
            const pointDistanceMax = Number.isFinite(Number(config.INITIAL_BODY_POINT_DISTANCE_MAX))
                ? Math.max(pointDistanceMin, Number(config.INITIAL_BODY_POINT_DISTANCE_MAX))
                : 8;
            const basePointDist = pointDistanceMin + Math.random() * (pointDistanceMax - pointDistanceMin);
            const useTriangulatedPrimitives = config.INITIAL_TRIANGULATED_PRIMITIVES_ENABLED !== false;
            let initialPrimitiveUsesUniformEdges = false;

            const selectWeightedTemplate = (templates) => {
                let total = 0;
                for (const t of templates) total += Math.max(0, Number(t.weight) || 0);
                if (!(total > 0)) return templates[0];
                let r = Math.random() * total;
                for (const t of templates) {
                    r -= Math.max(0, Number(t.weight) || 0);
                    if (r <= 0) return t;
                }
                return templates[templates.length - 1];
            };

            const buildTriangulatedPrimitive = () => {
                const SQRT3_OVER_2 = Math.sqrt(3) / 2;
                const templates = [
                    {
                        name: 'triangle',
                        weight: Number.isFinite(Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_TRIANGLE))
                            ? Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_TRIANGLE)
                            : 1,
                        points: [
                            { x: 0, y: 0 },
                            { x: 1, y: 0 },
                            { x: 0.5, y: SQRT3_OVER_2 }
                        ],
                        triangles: [[0, 1, 2]]
                    },
                    {
                        name: 'diamond',
                        weight: Number.isFinite(Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_DIAMOND))
                            ? Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_DIAMOND)
                            : 1,
                        points: [
                            { x: 0, y: 0 },
                            { x: 1, y: 0 },
                            { x: 0.5, y: SQRT3_OVER_2 },
                            { x: 1.5, y: SQRT3_OVER_2 }
                        ],
                        triangles: [[0, 1, 2], [1, 3, 2]]
                    },
                    {
                        name: 'hexagon',
                        weight: Number.isFinite(Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_HEXAGON))
                            ? Number(config.INITIAL_TRI_TEMPLATE_WEIGHT_HEXAGON)
                            : 1,
                        points: [
                            { x: 0, y: 0 },
                            { x: 1, y: 0 },
                            { x: 0.5, y: SQRT3_OVER_2 },
                            { x: -0.5, y: SQRT3_OVER_2 },
                            { x: -1, y: 0 },
                            { x: -0.5, y: -SQRT3_OVER_2 },
                            { x: 0.5, y: -SQRT3_OVER_2 }
                        ],
                        triangles: [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [0, 6, 1]]
                    }
                ];

                const selected = selectWeightedTemplate(templates);
                const rigidChance = Math.max(0, Math.min(1, Number(config.INITIAL_TRI_MESH_EDGE_RIGID_CHANCE) || 0));
                const sharedEdgeStiffness = drawRandomEdgeStiffness(1.1);
                const sharedEdgeDamping = drawRandomEdgeDamping();

                for (const p of selected.points) {
                    initialTempMassPoints.push(new MassPoint(
                        p.x * basePointDist,
                        p.y * basePointDist,
                        0.3 + Math.random() * 0.4,
                        baseRadius
                    ));
                }

                const edgeSet = new Set();
                const addEdge = (a, b) => {
                    const i = Math.min(a, b);
                    const j = Math.max(a, b);
                    edgeSet.add(`${i}:${j}`);
                };

                for (const tri of selected.triangles) {
                    if (!Array.isArray(tri) || tri.length !== 3) continue;
                    addEdge(tri[0], tri[1]);
                    addEdge(tri[1], tri[2]);
                    addEdge(tri[2], tri[0]);
                }

                for (const key of edgeSet) {
                    const [aRaw, bRaw] = key.split(':');
                    const a = Number(aRaw);
                    const b = Number(bRaw);
                    const p1 = initialTempMassPoints[a];
                    const p2 = initialTempMassPoints[b];
                    if (!p1 || !p2 || p1 === p2) continue;

                    initialTempSprings.push(new Spring(
                        p1,
                        p2,
                        sharedEdgeStiffness,
                        sharedEdgeDamping,
                        basePointDist,
                        Math.random() < rigidChance
                    ));
                }

                this.shapeType = 3;
                initialPrimitiveUsesUniformEdges = true;
            };

            if (useTriangulatedPrimitives) {
                buildTriangulatedPrimitive();
            } else if (this.shapeType === 0) { // Grid
                const numPointsX = 3; const numPointsY = 3; let gridPoints = [];
                for (let i = 0; i < numPointsY; i++) { gridPoints[i] = []; for (let j = 0; j < numPointsX; j++) {
                    // Position points relative to an arbitrary origin (0,0) for now, will adjust with centroid
                    const point = new MassPoint(j * basePointDist, i * basePointDist, 0.3 + Math.random() * 0.4, baseRadius);
                    initialTempMassPoints.push(point); gridPoints[i][j] = point;
                }}
                for (let i=0; i<numPointsY; i++) for (let j=0; j<numPointsX-1; j++) initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i][j+1], drawRandomEdgeStiffness(), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                for (let j=0; j<numPointsX; j++) for (let i=0; i<numPointsY-1; i++) initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i+1][j], drawRandomEdgeStiffness(), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                for (let i=0; i<numPointsY-1; i++) for (let j=0; j<numPointsX-1; j++) {
                    initialTempSprings.push(new Spring(gridPoints[i][j], gridPoints[i+1][j+1], drawRandomEdgeStiffness(0.85), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                    initialTempSprings.push(new Spring(gridPoints[i+1][j], gridPoints[i][j+1], drawRandomEdgeStiffness(0.85), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                }
            } else if (this.shapeType === 1) { // Line
                const numLinePoints = Math.floor(3 + Math.random() * 3); const isHorizontal = Math.random() < 0.5; let linePoints = [];
                for (let i=0; i<numLinePoints; i++) {
                    const x = (isHorizontal ? i * basePointDist : 0);
                    const y = (isHorizontal ? 0 : i * basePointDist);
                    const point = new MassPoint(x,y, 0.3+Math.random()*0.4, baseRadius);
                    initialTempMassPoints.push(point); linePoints.push(point);
                }
                for (let i=0; i<numLinePoints-1; i++) initialTempSprings.push(new Spring(linePoints[i], linePoints[i+1], drawRandomEdgeStiffness(), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                if (numLinePoints > 2) initialTempSprings.push(new Spring(linePoints[0], linePoints[numLinePoints-1], drawRandomEdgeStiffness(0.75), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
            } else { // Star
                const numOuterPoints = Math.floor(4 + Math.random()*3);
                const centralPoint = new MassPoint(0, 0, (0.3+Math.random()*0.4)*1.5, baseRadius*1.2); // Center at (0,0) for now
                initialTempMassPoints.push(centralPoint);
                const circleRadius = basePointDist * 1.5;
                for (let i=0; i<numOuterPoints; i++) {
                    const angle = (i / numOuterPoints) * Math.PI * 2;
                    const x = Math.cos(angle)*circleRadius;
                    const y = Math.sin(angle)*circleRadius;
                    const point = new MassPoint(x,y, 0.3+Math.random()*0.4, baseRadius);
                    initialTempMassPoints.push(point);
                    initialTempSprings.push(new Spring(centralPoint, point, drawRandomEdgeStiffness(), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                    if (i>0) initialTempSprings.push(new Spring(initialTempMassPoints[initialTempMassPoints.length-2], point, drawRandomEdgeStiffness(0.9), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
                }
                if (numOuterPoints > 1) initialTempSprings.push(new Spring(initialTempMassPoints[1], initialTempMassPoints[initialTempMassPoints.length-1], drawRandomEdgeStiffness(0.9), drawRandomEdgeDamping(), null, Math.random() < config.CHANCE_FOR_RIGID_SPRING));
            }

            // Calculate centroid of these initial temporary points (which were created around a local 0,0)
            let centroidX = 0, centroidY = 0;
            if (initialTempMassPoints.length > 0) {
                initialTempMassPoints.forEach(p => { centroidX += p.pos.x; centroidY += p.pos.y; });
                centroidX /= initialTempMassPoints.length;
                centroidY /= initialTempMassPoints.length;
            } 

            // Populate blueprintPoints from initialTempMassPoints, making coordinates relative to their own centroid
            initialTempMassPoints.forEach(p_temp => {
                let chosenNodeType;
                if (Math.random() < config.NEURON_CHANCE) {
                    chosenNodeType = NodeType.NEURON;
                } else {
                    const otherNodeTypes = availableFunctionalNodeTypes.filter(t => t !== NodeType.NEURON);
                    chosenNodeType = otherNodeTypes[Math.floor(Math.random() * otherNodeTypes.length)];
                }

                const availableMovementTypes = [MovementType.FIXED, MovementType.FLOATING, MovementType.NEUTRAL];
                let chosenMovementType = availableMovementTypes[Math.floor(Math.random() * availableMovementTypes.length)];
                if (chosenNodeType === NodeType.SWIMMER) {
                    chosenMovementType = MovementType.NEUTRAL;
                }
                const canBeGrabberInitial = Math.random() < config.GRABBER_GENE_MUTATION_CHANCE;
                const neuronDataBp = chosenNodeType === NodeType.NEURON ? {
                    hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MIN - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                } : null;

                this.blueprintPoints.push({
                    relX: p_temp.pos.x - centroidX, // Store relative to calculated centroid of the initial shape
                    relY: p_temp.pos.y - centroidY,
                    radius: p_temp.radius,
                    mass: p_temp.mass,
                    nodeType: chosenNodeType,
                    movementType: chosenMovementType,
                    dyeColor: dyeColorChoices[Math.floor(Math.random() * dyeColorChoices.length)],
                    canBeGrabber: canBeGrabberInitial,
                    neuronDataBlueprint: neuronDataBp,
                    activationIntervalGene: this._randomActivationIntervalGene(),
                    predatorRadiusGene: this._randomPredatorRadiusGene(),
                    eyeTargetType: chosenNodeType === NodeType.EYE ? (Math.random() < 0.5 ? EyeTargetType.PARTICLE : EyeTargetType.FOREIGN_BODY_POINT) : undefined,
                    maxEffectiveJetVelocity: this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4)
                });
            });

            // Populate blueprintSprings from initialTempSprings
            initialTempSprings.forEach(s_temp => {
                const p1Index = initialTempMassPoints.indexOf(s_temp.p1);
                const p2Index = initialTempMassPoints.indexOf(s_temp.p2);
                if (p1Index !== -1 && p2Index !== -1) {
                    this.blueprintSprings.push({
                        p1Index: p1Index,
                        p2Index: p2Index,
                        restLength: initialPrimitiveUsesUniformEdges
                            ? Math.max(1, Number(s_temp.restLength) || basePointDist)
                            : resolveEdgeRestLength(s_temp.restLength, 0.5),
                        isRigid: s_temp.isRigid,
                        stiffness: s_temp.stiffness,
                        damping: s_temp.dampingFactor,
                        activationIntervalGene: this._randomActivationIntervalGene()
                    });
                }
            });

            // Instantiate actual phenotype using the new blueprint.
            // The `startX, startY` provided to createShape becomes the target world position for the *centroid* of this new creature.
            this._instantiatePhenotypeFromBlueprint(startX, startY); 
        }

        // Common post-creation neuron linking (for both initial and reproduced)
        this.massPoints.forEach((p, idx) => {
            if (p.nodeType === NodeType.NEURON && p.neuronData) {
                if (this.massPoints.length > 1) {
                    let newSensorIndex;
                    do { newSensorIndex = Math.floor(Math.random() * this.massPoints.length); } while (newSensorIndex === idx);
                    p.neuronData.sensorPointIndex = newSensorIndex;
                    } else {
                    p.neuronData.sensorPointIndex = -1;
                }
                const effectorCandidates = this.massPoints.map((ep, epIdx) => ep.nodeType === NodeType.EMITTER ? epIdx : -1).filter(epIdx => epIdx !== -1 && epIdx !== idx);
                if (effectorCandidates.length > 0) {
                     p.neuronData.effectorPointIndex = effectorCandidates[Math.floor(Math.random() * effectorCandidates.length)];
                } else {
                    p.neuronData.effectorPointIndex = -1;
                }
            }
        });
        this._enforcePhotosyntheticBlueprintConstraints();
        this._calculateBlueprintRadius(); // Calculate after all blueprint points are finalized
    }

    _calculateBlueprintRadius() {
        if (!this.blueprintPoints || this.blueprintPoints.length === 0) {
            this.blueprintRadius = 5; // Default small radius if no blueprint points
            return;
        }
        let maxDistSq = 0;
        this.blueprintPoints.forEach(bp => {
            // relX, relY are distances from the blueprint's own centroid (0,0)
            const distToPointCenterSq = bp.relX * bp.relX + bp.relY * bp.relY;
            // Add the point's own radius to get distance to its edge
            const effectiveDistToEdge = Math.sqrt(distToPointCenterSq) + (bp.radius || 0);
            if (effectiveDistToEdge * effectiveDistToEdge > maxDistSq) {
                maxDistSq = effectiveDistToEdge * effectiveDistToEdge;
            }
        });
        this.blueprintRadius = Math.sqrt(maxDistSq);
        if (isNaN(this.blueprintRadius) || this.blueprintRadius === 0) {
            // Fallback if calculation results in NaN or 0 (e.g. single point at origin with 0 radius)
            this.blueprintRadius = 5; 
        }
    }

    _instantiatePhenotypeFromBlueprint(spawnX, spawnY) {
        this.massPoints = [];
        this.springs = [];

        // 1. Instantiate MassPoints from blueprintPoints
        for (const bp of this.blueprintPoints) {
            // Calculate absolute world position
            const worldX = spawnX + bp.relX;
            const worldY = spawnY + bp.relY;

            const newPoint = new MassPoint(worldX, worldY, bp.mass, bp.radius);
            newPoint.nodeType = bp.nodeType;
            newPoint.movementType = bp.movementType;
            newPoint.dyeColor = [...bp.dyeColor]; // Ensure deep copy for array
            newPoint.canBeGrabber = bp.canBeGrabber;
            newPoint.activationIntervalGene = this._sanitizeActivationIntervalGene(
                bp.activationIntervalGene ?? this._randomActivationIntervalGene()
            );
            newPoint.predatorRadiusGene = this._sanitizePredatorRadiusGene(
                bp.predatorRadiusGene ?? this._randomPredatorRadiusGene()
            );
            newPoint.actuationCooldownByChannel = { node: 0, grabber: 0, default_pattern: 0 };
            newPoint.swimmerActuation = { magnitude: 0, angle: 0 };
            newPoint.eyeTargetType = bp.eyeTargetType === undefined ? EyeTargetType.PARTICLE : bp.eyeTargetType;
            newPoint.maxEffectiveJetVelocity = this.jetMaxVelocityGene * (0.8 + Math.random() * 0.4);

            if (bp.neuronDataBlueprint) {
                newPoint.neuronData = {
                    // Copy properties from neuronDataBlueprint
                    // isBrain will be determined later in initializeBrain
                    isBrain: false, // Default, will be set by initializeBrain if this becomes the brain
                    hiddenLayerSize: bp.neuronDataBlueprint.hiddenLayerSize,
                    // sensorPointIndex and effectorPointIndex will be linked later
                    sensorPointIndex: -1, 
                    effectorPointIndex: -1
                };
            } else {
                newPoint.neuronData = null;
            }
            this.massPoints.push(newPoint);
        }

        // 2. Instantiate Springs from blueprintSprings
        for (const bs of this.blueprintSprings) {
            if (bs.p1Index >= 0 && bs.p1Index < this.massPoints.length &&
                bs.p2Index >= 0 && bs.p2Index < this.massPoints.length) {

                const p1 = this.massPoints[bs.p1Index];
                const p2 = this.massPoints[bs.p2Index];

                const springStiffness = bs.stiffness === undefined ? this.stiffness : bs.stiffness;
                const springDamping = bs.damping === undefined ? this.springDamping : bs.damping;

                // Use the creature's overall stiffness and damping
                // The blueprint spring carries restLength and isRigid
                const spring = new Spring(p1, p2, springStiffness, springDamping, bs.restLength, bs.isRigid);
                spring.activationIntervalGene = this._sanitizeActivationIntervalGene(
                    bs.activationIntervalGene ?? this._randomActivationIntervalGene()
                );
                spring.actuationCooldownByChannel = { edge: 0 };
                this.springs.push(spring);
            } else {
                console.warn(`Body ${this.id}: Invalid spring blueprint indices ${bs.p1Index}, ${bs.p2Index} for ${this.massPoints.length} points.`);
            }
        }

        this._enforcePhotosyntheticPhenotypeConstraints();

        // Recalculate things that depend on the final massPoints
        this.calculateCurrentMaxEnergy(); 

        // Refresh cached counters/eye designation used by brain sizing and UI.
        this._recountNodeTypeCaches();

        // Note: initializeBrain() can still be finalized by constructor flow after createShape().
    }

    // --- Main Update Method ---
    updateSelf(dt, fluidFieldRef) {
        if (this.isUnstable) return;

        this._enforcePhotosyntheticPhenotypeConstraints();
        this._prepareActuationStateForTick();

        let brainNode = this.massPoints.find(p => p.neuronData && p.neuronData.isBrain);
        this._updateSensoryInputsAndDefaultActivations(
            fluidFieldRef,
            this.nutrientField,
            this.lightField,
            { applyDefaultActivations: !brainNode }
        ); // Removed particles argument

        if (brainNode) {
            this._processBrain(brainNode, dt, fluidFieldRef, this.nutrientField, this.lightField); // Removed particles argument
        } else {
            this._applyFallbackBehaviors(dt, fluidFieldRef);
        }

        this._updateEnergyBudget(dt, fluidFieldRef, this.nutrientField, this.lightField); // nutrientField and lightField are class properties
        if (this.isUnstable) return; // Death from energy budget check

        this._performPhysicalUpdates(dt, fluidFieldRef);
        if (this.isUnstable) return; // Instability from physical updates

        this._finalizeUpdateAndCheckStability(dt); // dt might be needed for some interaction logic if it moves here
        if (this.isUnstable) return;

        // Developmental growth pass (new): may add genetically guided nodes/edges.
        this._attemptGrowthStep(dt);
    }

    // --- Refactored Helper Methods (Shells) ---
    _updateSensoryInputsAndDefaultActivations(fluidFieldRef, nutrientField, lightField, { applyDefaultActivations = true } = {}) { // Removed particles argument
        if (applyDefaultActivations) {
            this._applyDefaultActivationPatterns();
        }
        this._updateEyeNodes(); // Removed particles argument
        this._updateJetAndSwimmerFluidSensor(fluidFieldRef);
    }

    setNutrientField(field) {
        this.nutrientField = field;
    }

    setLightField(field) {
        this.lightField = field;
    }

    setParticles(particles) {
        this.particles = particles;
    }

    setSpatialGrid(grid) {
        this.spatialGrid = grid;
    }

    _applyDefaultActivationPatterns() {
        this.massPoints.forEach(point => {
            // If the point is an EATER or PREDATOR, default-pattern fallback keeps exertion at zero.
            if (point.nodeType === NodeType.EATER || point.nodeType === NodeType.PREDATOR) {
                point.currentExertionLevel = 0;
                return;
            }

            // Throttle default-pattern actuation updates with heritable interval genes.
            if (!this._shouldEvaluatePointActuation(point, 'default_pattern')) {
                return;
            }

            let baseActivation = 0;
            const timeFactor = (this.ticksSinceBirth + this.defaultActivationPhaseOffset) / Math.max(1, this.defaultActivationPeriod);

            switch (this.defaultActivationPattern) {
                case config.ActivationPatternType.FLAT:
                    baseActivation = this.defaultActivationLevel;
                    break;
                case config.ActivationPatternType.SINE:
                    baseActivation = this.defaultActivationLevel * (Math.sin(2 * Math.PI * timeFactor) * 0.5 + 0.5); // Ranges 0 to level
                    break;
                case config.ActivationPatternType.PULSE:
                    // Simple pulse: on for 10% of period, uses defaultActivationLevel as max
                    baseActivation = (timeFactor % 1.0 < 0.1) ? this.defaultActivationLevel : 0;
                    break;
            }
            point.currentExertionLevel = Math.max(0, Math.min(1, baseActivation)); // Clamp to 0-1
        });
    }

    _updateEyeNodes() { // Removed particles argument
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.EYE) {
                point.seesTarget = false; // Reset first
                point.nearestTargetMagnitude = 0;
                point.nearestTargetDirection = 0;
                let closestDistSq = config.EYE_DETECTION_RADIUS * config.EYE_DETECTION_RADIUS;

                // Determine the grid cell range to check based on config.EYE_DETECTION_RADIUS
                const eyeGxMin = Math.max(0, Math.floor((point.pos.x - config.EYE_DETECTION_RADIUS) / config.GRID_CELL_SIZE));
                const eyeGxMax = Math.min(config.GRID_COLS - 1, Math.floor((point.pos.x + config.EYE_DETECTION_RADIUS) / config.GRID_CELL_SIZE));
                const eyeGyMin = Math.max(0, Math.floor((point.pos.y - config.EYE_DETECTION_RADIUS) / config.GRID_CELL_SIZE));
                const eyeGyMax = Math.min(config.GRID_ROWS - 1, Math.floor((point.pos.y + config.EYE_DETECTION_RADIUS) / config.GRID_CELL_SIZE));

                if (point.eyeTargetType === EyeTargetType.PARTICLE) {
                    let nearestParticleFound = null;
                    for (let gy = eyeGyMin; gy <= eyeGyMax; gy++) {
                        for (let gx = eyeGxMin; gx <= eyeGxMax; gx++) {
                            const cellIndex = gx + gy * config.GRID_COLS;
                            if (this.spatialGrid[cellIndex] && this.spatialGrid[cellIndex].length > 0) {
                                const cellBucket = this.spatialGrid[cellIndex];
                                for (const item of cellBucket) {
                                    if (item.type === 'particle') {
                                        const particle = item.particleRef;
                                        if (particle.life <= 0) continue;

                                        const distSq = point.pos.sub(particle.pos).magSq();
                                        if (distSq < closestDistSq) {
                                            closestDistSq = distSq;
                                            nearestParticleFound = particle;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (nearestParticleFound) {
                        point.seesTarget = true;
                        const vecToTarget = nearestParticleFound.pos.sub(point.pos);
                        point.nearestTargetMagnitude = vecToTarget.mag() / config.EYE_DETECTION_RADIUS; 
                        point.nearestTargetDirection = Math.atan2(vecToTarget.y, vecToTarget.x);
                    }
                } else if (point.eyeTargetType === EyeTargetType.FOREIGN_BODY_POINT) {
                    let nearestForeignPointFound = null;
                    for (let gy = eyeGyMin; gy <= eyeGyMax; gy++) {
                        for (let gx = eyeGxMin; gx <= eyeGxMax; gx++) {
                            const cellIndex = gx + gy * config.GRID_COLS;
                            if (this.spatialGrid[cellIndex] && this.spatialGrid[cellIndex].length > 0) {
                                const cellBucket = this.spatialGrid[cellIndex];
                                for (const item of cellBucket) {
                                    // Check for softbody_point, ensure it's not from the current body, and body is not unstable
                                    if (item.type === 'softbody_point' && item.bodyRef !== this && !item.bodyRef.isUnstable) {
                                        const foreignPoint = item.pointRef;
                                        const distSq = point.pos.sub(foreignPoint.pos).magSq();
                                        if (distSq < closestDistSq) {
                                            closestDistSq = distSq;
                                            nearestForeignPointFound = foreignPoint;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (nearestForeignPointFound) {
                        point.seesTarget = true;
                        const vecToTarget = nearestForeignPointFound.pos.sub(point.pos);
                        point.nearestTargetMagnitude = vecToTarget.mag() / config.EYE_DETECTION_RADIUS;
                        point.nearestTargetDirection = Math.atan2(vecToTarget.y, vecToTarget.x);
                    }
                }
            }
        });
    }

    _processBrain(brainNode, dt, fluidFieldRef, nutrientField, lightField) {
        // Delegate to the Brain class for brain processing
        this.brain.process(dt, fluidFieldRef, nutrientField, lightField);
    }

    _applyFallbackBehaviors(dt, fluidFieldRef) {
        // This is called if brainNode is null or has incomplete data.
        // Apply default activation patterns (already done in _updateSensoryInputsAndDefaultActivations)
        // Apply random motor impulses for non-NN controlled movement
        if (this.motorImpulseMagnitudeCap > 0.0001 && (this.ticksSinceBirth % this.motorImpulseInterval === 0)) {
            for (let point of this.massPoints) {
                if (!point.isFixed && point.movementType !== MovementType.FLOATING) { // Swimmers (not floating) can still have motor impulses
                    const randomAngle = Math.random() * Math.PI * 2;
                    const impulseDir = new Vec2(Math.cos(randomAngle), Math.sin(randomAngle));
                    const impulseMag = Math.random() * this.motorImpulseMagnitudeCap;
                    point.applyForce(impulseDir.mul(impulseMag / dt));
                }
            }
        }
        // Note: Emitter dye emission and Swimmer fluid push in fallback mode are currently handled
        // by the fluid interaction logic in _performPhysicalUpdates, based on point.currentExertionLevel
        // which would have been set by _applyDefaultActivationPatterns.
    }

    _updateEnergyBudget(dt, fluidFieldRef, nutrientField, lightField) {
        // console.log("Body", this.id, "_updateEnergyBudget called. fluidFieldRef type:", fluidFieldRef ? fluidFieldRef.constructor.name : 'undefined');
        let currentFrameEnergyCost = 0;
        let currentFrameEnergyGain = 0;
        let poisonDamageThisFrame = 0; // Initialize here
        this.energyGainedFromPhotosynthesisThisTick = 0; // Reset for the current tick

        const hasFluidField = fluidFieldRef !== null && typeof fluidFieldRef !== 'undefined';
        const hasNutrientField = this.nutrientField !== null && typeof this.nutrientField !== 'undefined';
        const hasLightField = this.lightField !== null && typeof this.lightField !== 'undefined';

        const dyeState = this._getDyeEcologyStateAtBodyCenter();
        const photosynthesisDyeScale = this._resolveDyeEffectScale(dyeState, {
            affinityKey: 'PHOTOSYNTHETIC',
            weight: Math.max(0, Number(config.DYE_PHOTOSYNTHESIS_EFFECT_WEIGHT) || 0.55)
        });

        const scaleX = hasFluidField ? fluidFieldRef.scaleX : 0;
        const scaleY = hasFluidField ? fluidFieldRef.scaleY : 0;

        for (const point of this.massPoints) {
            // --- Red Dye Poison Effect (Moved inside main loop) ---
            if (hasFluidField && config.RED_DYE_POISON_STRENGTH > 0) {
                // const fluidGridX = Math.floor(point.pos.x / scaleX); // gx already calculated below
                // const fluidGridY = Math.floor(point.pos.y / scaleY); // gy already calculated below
                // let tempMapIdxForPoison = fluidFieldRef.IX(fluidGridX, fluidGridY); // mapIdx used below is the same
                // const redDensity = (fluidFieldRef.densityR[tempMapIdxForPoison] || 0) / 255;
                // Calculation will use mapIdx determined below if hasFluidField
            }
            // --- End of Moved Red Dye Poison Effect ---

            let costMultiplier = 1.0;
            let mapIdx = -1;

            if (hasFluidField) {
                const gx = Math.floor(point.pos.x / scaleX);
                const gy = Math.floor(point.pos.y / scaleY);
                mapIdx = fluidFieldRef.IX(gx, gy);

                // --- Red Dye Poison Calculation (using mapIdx) ---
                if (config.RED_DYE_POISON_STRENGTH > 0) {
                    const redDensity = (fluidFieldRef.densityR[mapIdx] || 0) / 255;
                    if (redDensity > 0.01) {
                        poisonDamageThisFrame += redDensity * config.RED_DYE_POISON_STRENGTH * (point.radius / 5);
                    }
                }
                // --- End of Red Dye Poison Calculation ---

                if (hasNutrientField) {
                    const baseNutrientValue = this.nutrientField[mapIdx] !== undefined ? this.nutrientField[mapIdx] : 1.0;
                    const effectiveNutrientValue = baseNutrientValue * config.globalNutrientMultiplier;
                    costMultiplier = 1.0 / Math.max(config.MIN_NUTRIENT_VALUE, effectiveNutrientValue);
                }
            }


            const baseNodeCostThisFrame = config.BASE_NODE_EXISTENCE_COST * costMultiplier;
            currentFrameEnergyCost += baseNodeCostThisFrame;
            this.energyCostFromBaseNodes += baseNodeCostThisFrame * dt;

            const exertion = point.currentExertionLevel || 0;
            const exertionSq = exertion * exertion; // Calculate once if used multiple times
            const wasActuationEvaluated = point.__actuationEvaluatedThisTick === true;
            const upkeepFraction = Math.max(0, Math.min(1, Number(config.ACTUATION_UPKEEP_COST_FRACTION) || 0));
            const activationMultiplier = Math.max(0, Number(config.ACTUATION_ACTIVATION_COST_MULTIPLIER) || 0);

            const splitActuationCost = (baseCost) => {
                const upkeepCost = baseCost * upkeepFraction;
                const activationCost = wasActuationEvaluated ? (baseCost * activationMultiplier) : 0;
                this.energyCostFromActuationUpkeep += upkeepCost * dt;
                this.energyCostFromActuationEvents += activationCost * dt;
                return upkeepCost + activationCost;
            };

            // NodeType specific costs
            switch (point.nodeType) {
                case NodeType.EMITTER: {
                    const emitterCostThisFrame = splitActuationCost(config.EMITTER_NODE_ENERGY_COST * exertionSq * costMultiplier);
                    currentFrameEnergyCost += emitterCostThisFrame;
                    this.energyCostFromEmitterNodes += emitterCostThisFrame * dt;
                    break;
                }
                case NodeType.SWIMMER: {
                    const swimmerCostThisFrame = splitActuationCost(config.SWIMMER_NODE_ENERGY_COST * exertionSq * costMultiplier);
                    currentFrameEnergyCost += swimmerCostThisFrame;
                    this.energyCostFromSwimmerNodes += swimmerCostThisFrame * dt;
                    break;
                }
                case NodeType.JET: {
                    const jetCostThisFrame = splitActuationCost(config.JET_NODE_ENERGY_COST * exertionSq * costMultiplier);
                    currentFrameEnergyCost += jetCostThisFrame;
                    this.energyCostFromJetNodes += jetCostThisFrame * dt;
                    break;
                }
                case NodeType.EATER: {
                    const eaterCostThisFrame = splitActuationCost(config.EATER_NODE_ENERGY_COST * exertionSq * costMultiplier);
                    currentFrameEnergyCost += eaterCostThisFrame;
                    this.energyCostFromEaterNodes += eaterCostThisFrame * dt;
                    break;
                }
                case NodeType.PREDATOR: {
                    const predatorCostThisFrame = splitActuationCost(config.PREDATOR_NODE_ENERGY_COST * exertionSq * costMultiplier);
                    currentFrameEnergyCost += predatorCostThisFrame;
                    this.energyCostFromPredatorNodes += predatorCostThisFrame * dt;
                    break;
                }
                case NodeType.PHOTOSYNTHETIC: {
                    const photosyntheticCostThisFrame = config.PHOTOSYNTHETIC_NODE_ENERGY_COST * costMultiplier;
                    currentFrameEnergyCost += photosyntheticCostThisFrame;
                    this.energyCostFromPhotosyntheticNodes += photosyntheticCostThisFrame * dt;

                    if (hasLightField && hasFluidField && mapIdx !== -1) { // mapIdx would have been calculated if hasFluidField
                        const baseLightValue = this.lightField[mapIdx] !== undefined ? this.lightField[mapIdx] : 0.0;
                        const effectiveLightValue = baseLightValue * config.globalLightMultiplier;
                        const energyGainThisPoint = effectiveLightValue * config.PHOTOSYNTHESIS_EFFICIENCY * (point.radius / 5) * dt * photosynthesisDyeScale;
                        currentFrameEnergyGain += energyGainThisPoint;
                        this.energyGainedFromPhotosynthesis += energyGainThisPoint; // Lifetime total
                        this.energyGainedFromPhotosynthesisThisTick += energyGainThisPoint; // Current tick total
                    }
                    break;
                }
                case NodeType.ATTRACTOR: {
                    const attractorCostThisFrame = splitActuationCost(config.ATTRACTOR_NODE_ENERGY_COST * exertionSq * costMultiplier);
                    currentFrameEnergyCost += attractorCostThisFrame;
                    this.energyCostFromAttractorNodes += attractorCostThisFrame * dt;
                    break;
                }
                case NodeType.REPULSOR: {
                    const repulsorCostThisFrame = splitActuationCost(config.REPULSOR_NODE_ENERGY_COST * exertionSq * costMultiplier);
                    currentFrameEnergyCost += repulsorCostThisFrame;
                    this.energyCostFromRepulsorNodes += repulsorCostThisFrame * dt;
                    break;
                }
                // Note: Neuron, Grabbing, Eye costs are handled by separate 'if' statements below
                // as they can co-exist or have different conditions than the primary functional type.
            }

            // Neuron cost (can be any type of point, but has neuronData)
            if (point.nodeType === NodeType.NEURON) { // This check is okay, as NodeType is exclusive.
                let neuronCostThisFrame = 0;
                if (point.neuronData && point.neuronData.isBrain) {
                    neuronCostThisFrame = config.NEURON_NODE_ENERGY_COST * 5 * costMultiplier;
                    neuronCostThisFrame += (point.neuronData.hiddenLayerSize || 0) * config.NEURON_NODE_ENERGY_COST * 0.1 * costMultiplier;
                } else {
                    neuronCostThisFrame = config.NEURON_NODE_ENERGY_COST * costMultiplier;
                }
                currentFrameEnergyCost += neuronCostThisFrame;
                this.energyCostFromNeuronNodes += neuronCostThisFrame * dt;
            }

            // Grabbing cost (independent of NodeType, depends on isGrabbing state)
            if (point.isGrabbing) {
                const grabbingCostThisFrame = splitActuationCost(config.GRABBING_NODE_ENERGY_COST * costMultiplier);
                currentFrameEnergyCost += grabbingCostThisFrame;
                this.energyCostFromGrabbingNodes += grabbingCostThisFrame * dt;
            }

            // Eye cost (independent of NodeType, depends on isDesignatedEye state)
            if (point.isDesignatedEye) {
                 const eyeCostThisFrame = config.EYE_NODE_ENERGY_COST * costMultiplier;
                 currentFrameEnergyCost += eyeCostThisFrame;
                 this.energyCostFromEyeNodes += eyeCostThisFrame * dt;
            }
        }

        // Apply poison damage after the loop, before other adjustments
        if (poisonDamageThisFrame > 0) {
            this.creatureEnergy -= poisonDamageThisFrame * dt * 60; // dt is in seconds, scale strength to be per-second
        }

        if (config.DYE_ECOLOGY_ENABLED) {
            const overexposureDrain = (Number(config.DYE_OVEREXPOSURE_ENERGY_DRAIN) || 0)
                * (Number(dyeState.overexposure) || 0)
                * Math.max(0, Number(this.dyeResponseGain) || 0);
            if (overexposureDrain > 0) {
                this.creatureEnergy -= overexposureDrain * dt * 60;
            }
        }

        this.creatureEnergy += currentFrameEnergyGain; // Gains are already dt-scaled
        this.creatureEnergy -= currentFrameEnergyCost * dt; // Costs are per-frame, so scale by dt here
        this.creatureEnergy = Math.min(this.currentMaxEnergy, Math.max(0, this.creatureEnergy));

        if (this.creatureEnergy <= 0) {
            this._markUnstable('energy_depleted');
        }
    }

    _performPhysicalUpdates(dt, fluidFieldRef) {
        const restitution = 0.4; // Local constant for boundary collision restitution
        const dyeState = this._getDyeEcologyStateAtBodyCenter();
        const swimmerDyeScale = this._resolveDyeEffectScale(dyeState, {
            affinityKey: 'SWIMMER',
            weight: Math.max(0, Number(config.DYE_SWIMMER_EFFECT_WEIGHT) || 0.35)
        });
        const jetDyeScale = this._resolveDyeEffectScale(dyeState, {
            affinityKey: 'JET',
            weight: Math.max(0, Number(config.DYE_JET_EFFECT_WEIGHT) || 0.35)
        });
        const emitterDyeScale = this._resolveDyeEffectScale(dyeState, {
            affinityKey: 'EMITTER',
            weight: Math.max(0, Number(config.DYE_EMITTER_EFFECT_WEIGHT) || 0.4),
            includeEmitterInhibition: true
        });

        if (fluidFieldRef) {
            for (let point of this.massPoints) {
                const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);

                if (!isFinite(fluidGridX) || !isFinite(fluidGridY)) continue;

                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);
                // Check index validity once, for all interactions.
                // Assuming Vx, Vy, density arrays all have the same length.
                if (idx < 0 || idx >= (fluidFieldRef.Vx ? fluidFieldRef.Vx.length : (fluidFieldRef.textures.velocityPing ? fluidFieldRef.size * fluidFieldRef.size : 0))) continue;


                // --- Fluid Interactions (should occur even if fixed) ---
                if (point.nodeType === NodeType.EMITTER) {
                    let dyeEmissionStrength = 50 * point.currentExertionLevel * emitterDyeScale;
                    fluidFieldRef.addDensity(fluidGridX, fluidGridY, point.dyeColor[0], point.dyeColor[1], point.dyeColor[2], dyeEmissionStrength);
                }

                if (point.nodeType === NodeType.JET) {
                    const exertion = point.currentExertionLevel || 0;
                    if (exertion > 0.01) {
                        const currentFluidVelX = fluidFieldRef.Vx ? fluidFieldRef.Vx[idx] : 0; // Placeholder for GPU
                        const currentFluidVelY = fluidFieldRef.Vy ? fluidFieldRef.Vy[idx] : 0; // Placeholder for GPU
                        const currentFluidSpeedSq = currentFluidVelX ** 2 + currentFluidVelY ** 2;

                        if (currentFluidSpeedSq < point.maxEffectiveJetVelocity ** 2) {
                            const finalMagnitude = point.jetData.currentMagnitude * jetDyeScale;
                            const angle = point.jetData.currentAngle;
                            const appliedForceX = finalMagnitude * Math.cos(angle);
                            const appliedForceY = finalMagnitude * Math.sin(angle);
                            fluidFieldRef.addVelocity(fluidGridX, fluidGridY, appliedForceX, appliedForceY);
                        }
                    }
                }

                if (point.nodeType === NodeType.SWIMMER && !point.isFixed) {
                    const magnitude = Math.max(0, (Number(point.swimmerActuation?.magnitude) || 0) * swimmerDyeScale);
                    const angle = Number(point.swimmerActuation?.angle) || 0;
                    if (magnitude > 0.0001) {
                        point.applyForce(new Vec2(Math.cos(angle) * (magnitude / dt), Math.sin(angle) * (magnitude / dt)));
                    }
                }

                // --- Physics Updates for Mobile Points Only ---
                if (point.isFixed) continue;

                if (point.movementType === MovementType.FLOATING) {
                    const rawFluidVx = fluidFieldRef.Vx ? fluidFieldRef.Vx[idx] : 0;
                    const rawFluidVy = fluidFieldRef.Vy ? fluidFieldRef.Vy[idx] : 0;
                    this._tempVec1.copyFrom(point.pos).subInPlace(point.prevPos).mulInPlace(1.0 - this.fluidEntrainment);
                    this._tempVec2.x = rawFluidVx * fluidFieldRef.scaleX * dt;
                    this._tempVec2.y = rawFluidVy * fluidFieldRef.scaleY * dt;
                    this._tempVec2.mulInPlace(this.fluidCurrentStrength).mulInPlace(this.fluidEntrainment);
                    this._tempVec1.addInPlace(this._tempVec2);

                    point.prevPos.copyFrom(point.pos).subInPlace(this._tempVec1);
                }
            }
        }

        for (let spring of this.springs) {
            spring.applyForce();
        }

        for (let point of this.massPoints) {
            point.update(dt);
        }

        const MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ = (config.MAX_PIXELS_PER_FRAME_DISPLACEMENT) ** 2;
        for (let i = 0; i < this.massPoints.length; i++) {
            const point = this.massPoints[i];
            if (point.isFixed) continue;

            const displacementX = point.pos.x - point.prevPos.x;
            const displacementY = point.pos.y - point.prevPos.y;
            const displacementSq = displacementX ** 2 + displacementY ** 2;

            const posXNaN = Number.isNaN(point.pos.x);
            const posYNaN = Number.isNaN(point.pos.y);
            const posXFinite = Number.isFinite(point.pos.x);
            const posYFinite = Number.isFinite(point.pos.y);

            if (displacementSq > MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ) {
                this._markUnstable('physics_invalid_motion', {
                    pointIndex: i,
                    displacementX,
                    displacementY,
                    displacementSq,
                    displacementLimitSq: MAX_PIXELS_PER_FRAME_DISPLACEMENT_SQ,
                    pos: { x: point.pos.x, y: point.pos.y },
                    prevPos: { x: point.prevPos.x, y: point.prevPos.y },
                    nodeType: point.nodeType,
                    movementType: point.movementType,
                    ticksSinceBirth: this.ticksSinceBirth
                });
                return;
            }

            if (posXNaN || posYNaN) {
                this._markUnstable('physics_nan_position', {
                    pointIndex: i,
                    pos: { x: point.pos.x, y: point.pos.y },
                    prevPos: { x: point.prevPos.x, y: point.prevPos.y },
                    nodeType: point.nodeType,
                    movementType: point.movementType,
                    ticksSinceBirth: this.ticksSinceBirth
                });
                return;
            }

            if (!posXFinite || !posYFinite) {
                this._markUnstable('physics_non_finite_position', {
                    pointIndex: i,
                    pos: { x: point.pos.x, y: point.pos.y },
                    prevPos: { x: point.prevPos.x, y: point.prevPos.y },
                    nodeType: point.nodeType,
                    movementType: point.movementType,
                    ticksSinceBirth: this.ticksSinceBirth
                });
                return;
            }

            const implicitVelX = point.pos.x - point.prevPos.x;
            const implicitVelY = point.pos.y - point.prevPos.y;

            const outOfBounds = (
                point.pos.x < 0 ||
                point.pos.x > config.WORLD_WIDTH ||
                point.pos.y < 0 ||
                point.pos.y > config.WORLD_HEIGHT
            );

            if (outOfBounds) {
                if (config.IS_WORLD_WRAPPING) {
                    if (point.pos.x < 0) { point.pos.x += config.WORLD_WIDTH; point.prevPos.x += config.WORLD_WIDTH; }
                    else if (point.pos.x > config.WORLD_WIDTH) { point.pos.x -= config.WORLD_WIDTH; point.prevPos.x -= config.WORLD_WIDTH; }
                    if (point.pos.y < 0) { point.pos.y += config.WORLD_HEIGHT; point.prevPos.y += config.WORLD_HEIGHT; }
                    else if (point.pos.y > config.WORLD_HEIGHT) { point.pos.y -= config.WORLD_HEIGHT; point.prevPos.y -= config.WORLD_HEIGHT; }
                } else {
                    if (point.pos.x - point.radius < 0) {
                        point.pos.x = point.radius;
                        point.prevPos.x = point.pos.x - implicitVelX * restitution;
                    } else if (point.pos.x + point.radius > config.WORLD_WIDTH) {
                        point.pos.x = config.WORLD_WIDTH - point.radius;
                        point.prevPos.x = point.pos.x - implicitVelX * restitution;
                    }
                    if (point.pos.y - point.radius < 0) {
                        point.pos.y = point.radius;
                        point.prevPos.y = point.pos.y - implicitVelY * restitution;
                    } else if (point.pos.y + point.radius > config.WORLD_HEIGHT) {
                        point.pos.y = config.WORLD_HEIGHT - point.radius;
                        point.prevPos.y = point.pos.y - implicitVelY * restitution;
                    }
                }

                if (config.KILL_ON_OUT_OF_BOUNDS) {
                    this._markUnstable('physics_out_of_bounds');
                    return;
                }
            }
        }
    }

    _finalizeUpdateAndCheckStability(dt) { 
        this.preyPredatedThisTick = new Set(); // Keyed by "predatorPointIndex:preyBodyId" for this body's current tick
        if (this.isUnstable) return; 

        const dyeState = this._getDyeEcologyStateAtBodyCenter();
        const predatorDyeScale = this._resolveDyeEffectScale(dyeState, {
            affinityKey: 'PREDATOR',
            weight: Math.max(0, Number(config.DYE_PREDATOR_EFFECT_WEIGHT) || 0.65)
        });
        const eaterDyeScale = this._resolveDyeEffectScale(dyeState, {
            affinityKey: 'EATER',
            weight: Math.max(0, Number(config.DYE_EATER_EFFECT_WEIGHT) || 0.65)
        });

        // Inter-body repulsion & Predation
        // Use this._tempVec1 for diff, and this._tempVec2 for forceDir/repulsionForce
        const tempDiffVec = this._tempVec1; 
        const tempForceVec = this._tempVec2;

        for (let i_p1 = 0; i_p1 < this.massPoints.length; i_p1++) {
            const p1 = this.massPoints[i_p1];

            // Attractor/Repulsor logic: find nearest foreign point and apply force
            if ((p1.nodeType === NodeType.ATTRACTOR || p1.nodeType === NodeType.REPULSOR) && (p1.currentExertionLevel || 0) > 0.01) {
                let nearestForeignPoint = null;
                const isAttractor = p1.nodeType === NodeType.ATTRACTOR;
                const p1Exertion = p1.currentExertionLevel;

                const radiusMultiplierConfig = isAttractor ?
                    { base: config.ATTRACTION_RADIUS_MULTIPLIER_BASE, bonus: config.ATTRACTION_RADIUS_MULTIPLIER_MAX_BONUS } :
                    { base: config.REPULSION_RADIUS_MULTIPLIER_BASE, bonus: config.REPULSION_RADIUS_MULTIPLIER_MAX_BONUS };
                const radiusMultiplier = radiusMultiplierConfig.base + (radiusMultiplierConfig.bonus * p1Exertion);
                const interactionRadius = p1.radius * radiusMultiplier;
                let min_dist_sq = interactionRadius * interactionRadius; // Start search radius at max interaction radius

                const p1Gx_force = Math.max(0, Math.min(config.GRID_COLS - 1, Math.floor(p1.pos.x / config.GRID_CELL_SIZE)));
                const p1Gy_force = Math.max(0, Math.min(config.GRID_ROWS - 1, Math.floor(p1.pos.y / config.GRID_CELL_SIZE)));
                const searchRadiusInCells = Math.ceil(interactionRadius / config.GRID_CELL_SIZE);

                for (let dy = -searchRadiusInCells; dy <= searchRadiusInCells; dy++) {
                    for (let dx = -searchRadiusInCells; dx <= searchRadiusInCells; dx++) {
                        const checkGx = p1Gx_force + dx;
                        const checkGy = p1Gy_force + dy;
                        if (checkGx >= 0 && checkGx < config.GRID_COLS && checkGy >= 0 && checkGy < config.GRID_ROWS) {
                            const cellIndex = checkGx + checkGy * config.GRID_COLS;
                            if (Array.isArray(this.spatialGrid[cellIndex])) {
                                const cellBucket = this.spatialGrid[cellIndex];
                                for (const otherItem of cellBucket) {
                                    if (otherItem.type === 'softbody_point' && otherItem.bodyRef !== this && !otherItem.bodyRef.isUnstable) {
                                        const p2_candidate = otherItem.pointRef;
                                        tempDiffVec.copyFrom(p1.pos).subInPlace(p2_candidate.pos);
                                        const distSq = tempDiffVec.magSq();
                                        if (distSq < min_dist_sq) {
                                            min_dist_sq = distSq;
                                            nearestForeignPoint = p2_candidate;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (nearestForeignPoint) {
                    const maxForce = isAttractor ? config.ATTRACTOR_MAX_FORCE : config.REPULSOR_MAX_FORCE;
                    const forceMagnitude = p1Exertion * maxForce * (1.0 - (Math.sqrt(min_dist_sq) / interactionRadius));
                    const finalForceMagnitude = isAttractor ? -forceMagnitude : forceMagnitude;

                    const forceDir = this._tempVec2.copyFrom(p1.pos).subInPlace(nearestForeignPoint.pos).normalizeInPlace();
                    const forceOnP1 = forceDir.mulInPlace(finalForceMagnitude);

                    if (!p1.isFixed) {
                        p1.applyForce(forceOnP1);
                    }
                    if (!nearestForeignPoint.isFixed) {
                        nearestForeignPoint.applyForce(forceOnP1.clone().mulInPlace(-1));
                    }
                }
            }


            if (p1.isFixed) continue;

            const p1ExertionClamped = Math.max(0, Math.min(1, Number(p1.currentExertionLevel) || 0));
            const eaterPenaltyArmed = p1.nodeType === NodeType.EATER && p1ExertionClamped > 0.01;
            const eaterTouchedForeignPoints = eaterPenaltyArmed ? new Set() : null;

            const p1Gx = Math.max(0, Math.min(config.GRID_COLS - 1, Math.floor(p1.pos.x / config.GRID_CELL_SIZE)));
            const p1Gy = Math.max(0, Math.min(config.GRID_ROWS - 1, Math.floor(p1.pos.y / config.GRID_CELL_SIZE)));

            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const checkGx = p1Gx + dx;
                    const checkGy = p1Gy + dy;

                    if (checkGx >= 0 && checkGx < config.GRID_COLS && checkGy >= 0 && checkGy < config.GRID_ROWS) {
                        const cellIndex = checkGx + checkGy * config.GRID_COLS;
                        if (Array.isArray(this.spatialGrid[cellIndex])) {
                            const cellBucket = this.spatialGrid[cellIndex];
                            for (const otherItem of cellBucket) {
                                if (otherItem.type === 'softbody_point') {
                                    if (otherItem.bodyRef === this) continue;
                                    const p2 = otherItem.pointRef;
                                    if (p2.isFixed) continue;

                                    // const diff = p1.pos.sub(p2.pos);
                                    tempDiffVec.copyFrom(p1.pos).subInPlace(p2.pos);
                                    const distSq = tempDiffVec.magSq();
                                    const interactionRadius = (p1.radius + p2.radius) * config.BODY_REPULSION_RADIUS_FACTOR;

                                    if (distSq < interactionRadius * interactionRadius && distSq > 0.0001) {
                                        const dist = Math.sqrt(distSq);
                                        const overlap = interactionRadius - dist;
                                        // const forceDir = diff.normalize();
                                        // const repulsionForce = forceDir.mul(repulsionForceMag);
                                        // p1.applyForce(repulsionForce);

                                        tempForceVec.copyFrom(tempDiffVec).normalizeInPlace();
                                        const repulsionForceMag = config.BODY_REPULSION_STRENGTH * overlap * 0.5;
                                        tempForceVec.mulInPlace(repulsionForceMag);
                                        p1.applyForce(tempForceVec);
                                    }

                                    if (eaterPenaltyArmed && distSq < interactionRadius * interactionRadius) {
                                        eaterTouchedForeignPoints.add(p2);
                                    }

                                    if (p1.nodeType === NodeType.PREDATOR) {
                                        const p1Exertion = Math.max(0, Math.min(1, Number(p1.currentExertionLevel) || 0));
                                        const predationRadius = this._computePredatorRadius(p1);
                                        if (predationRadius <= 0) continue;

                                        if (distSq < predationRadius * predationRadius) {
                                            // Allow multi-node predation pressure: each predator node may sap a given prey once per tick.
                                            const predationKey = `${i_p1}:${otherItem.bodyRef.id}`;
                                            if (!this.preyPredatedThisTick.has(predationKey)) {
                                                const effectiveEnergySapped = (config.ENERGY_SAPPED_PER_PREDATION_BASE + (config.ENERGY_SAPPED_PER_PREDATION_MAX_BONUS * p1Exertion)) * predatorDyeScale;
                                                const energyToSap = Math.min(otherItem.bodyRef.creatureEnergy, effectiveEnergySapped); 
                                                if (energyToSap > 0) {
                                                    otherItem.bodyRef.creatureEnergy -= energyToSap;
                                                    this.creatureEnergy = Math.min(this.currentMaxEnergy, this.creatureEnergy + energyToSap); 
                                                    this.energyGainedFromPredation += energyToSap;
                                                    this.preyPredatedThisTick.add(predationKey);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (p1.nodeType === NodeType.PREDATOR) {
                const predationRadius = this._computePredatorRadius(p1);
                if (predationRadius > 0) {
                    const predationRadiusSq = predationRadius * predationRadius;
                    let selfOverlaps = 0;
                    for (let selfIdx = 0; selfIdx < this.massPoints.length; selfIdx++) {
                        if (selfIdx === i_p1) continue;
                        const selfPoint = this.massPoints[selfIdx];
                        if (!selfPoint) continue;

                        tempDiffVec.copyFrom(p1.pos).subInPlace(selfPoint.pos);
                        if (tempDiffVec.magSq() < predationRadiusSq) {
                            selfOverlaps += 1;
                        }
                    }

                    if (selfOverlaps > 0) {
                        const exertion = Math.max(0, Math.min(1, Number(p1.currentExertionLevel) || 0));
                        const maxOverlaps = Math.max(1, Math.floor(Number(config.PREDATOR_SELF_DAMAGE_MAX_OVERLAPS_PER_TICK) || 1));
                        const overlapFactor = Math.min(selfOverlaps, maxOverlaps);
                        const damagePerOverlap = (Number(config.PREDATOR_SELF_DAMAGE_BASE) || 0)
                            + ((Number(config.PREDATOR_SELF_DAMAGE_MAX_BONUS) || 0) * exertion);
                        const totalSelfDamage = Math.min(this.creatureEnergy, damagePerOverlap * overlapFactor);
                        if (totalSelfDamage > 0) {
                            this.creatureEnergy -= totalSelfDamage;
                        }
                    }
                }
            }

            if (eaterPenaltyArmed) {
                let eaterSelfTouches = 0;
                for (let selfIdx = 0; selfIdx < this.massPoints.length; selfIdx++) {
                    if (selfIdx === i_p1) continue;
                    const selfPoint = this.massPoints[selfIdx];
                    if (!selfPoint) continue;

                    tempDiffVec.copyFrom(p1.pos).subInPlace(selfPoint.pos);
                    const selfDistSq = tempDiffVec.magSq();
                    const selfInteractionRadius = (p1.radius + selfPoint.radius) * config.BODY_REPULSION_RADIUS_FACTOR;
                    if (selfDistSq < selfInteractionRadius * selfInteractionRadius) {
                        eaterSelfTouches += 1;
                    }
                }

                const foreignTouches = eaterTouchedForeignPoints ? eaterTouchedForeignPoints.size : 0;
                const totalTouches = eaterSelfTouches + foreignTouches;
                if (totalTouches > 0) {
                    const maxOverlaps = Math.max(1, Math.floor(Number(config.PREDATOR_SELF_DAMAGE_MAX_OVERLAPS_PER_TICK) || 1));
                    const overlapFactor = Math.min(totalTouches, maxOverlaps);
                    const damagePerOverlap = (Number(config.PREDATOR_SELF_DAMAGE_BASE) || 0)
                        + ((Number(config.PREDATOR_SELF_DAMAGE_MAX_BONUS) || 0) * p1ExertionClamped);
                    const totalContactPenalty = Math.min(this.creatureEnergy, damagePerOverlap * overlapFactor);
                    if (totalContactPenalty > 0) {
                        this.creatureEnergy -= totalContactPenalty;
                    }
                }
            }
        }

        // Eating Logic
        for (let point of this.massPoints) {
            if (point.isFixed) continue; 
            if (point.nodeType === NodeType.EATER) { 
                const pointExertion = point.currentExertionLevel || 0;
                const effectiveEatingRadiusMultiplier = config.EATING_RADIUS_MULTIPLIER_BASE + (config.EATING_RADIUS_MULTIPLIER_MAX_BONUS * pointExertion);
                const eatingRadius = point.radius * effectiveEatingRadiusMultiplier;
                const eatingRadiusSq = eatingRadius * eatingRadius;

                // Determine the grid cell range to check based on eatingRadius
                const eaterGxMin = Math.max(0, Math.floor((point.pos.x - eatingRadius) / config.GRID_CELL_SIZE));
                const eaterGxMax = Math.min(config.GRID_COLS - 1, Math.floor((point.pos.x + eatingRadius) / config.GRID_CELL_SIZE));
                const eaterGyMin = Math.max(0, Math.floor((point.pos.y - eatingRadius) / config.GRID_CELL_SIZE));
                const eaterGyMax = Math.min(config.GRID_ROWS - 1, Math.floor((point.pos.y + eatingRadius) / config.GRID_CELL_SIZE));

                for (let gy = eaterGyMin; gy <= eaterGyMax; gy++) {
                    for (let gx = eaterGxMin; gx <= eaterGxMax; gx++) {
                        const cellIndex = gx + gy * config.GRID_COLS;
                        if (Array.isArray(this.spatialGrid[cellIndex])) {
                            const cellBucket = this.spatialGrid[cellIndex];
                            for (let k = cellBucket.length - 1; k >= 0; k--) { 
                                const item = cellBucket[k];
                                if (item.type === 'particle') {
                                    const particle = item.particleRef;
                                    if (particle.life > 0 && !particle.isEaten) {
                                        // const distSq = point.pos.sub(particle.pos).magSq();
                                        this._tempVec1.copyFrom(point.pos).subInPlace(particle.pos);
                                        const distSq = this._tempVec1.magSq();
                                        if (distSq < eatingRadiusSq) {
                                            particle.isEaten = true;
                                            particle.life = 0; 

                                            let energyGain = config.ENERGY_PER_PARTICLE;
                                            if (this.nutrientField && runtimeState.fluidField) { // fluidFieldRef is runtimeState.fluidField in this context
                                                const particleGx = Math.floor(particle.pos.x / runtimeState.fluidField.scaleX);
                                                const particleGy = Math.floor(particle.pos.y / runtimeState.fluidField.scaleY);
                                                const nutrientIdxAtParticle = runtimeState.fluidField.IX(particleGx, particleGy);
                                                const baseNutrientValueAtParticle = this.nutrientField[nutrientIdxAtParticle] !== undefined ? this.nutrientField[nutrientIdxAtParticle] : 1.0;
                                                const effectiveNutrientAtParticle = baseNutrientValueAtParticle * config.globalNutrientMultiplier;
                                                energyGain *= Math.max(config.MIN_NUTRIENT_VALUE, effectiveNutrientAtParticle);
                                            }
                                            energyGain *= eaterDyeScale;
                                            this.creatureEnergy = Math.min(this.currentMaxEnergy, this.creatureEnergy + energyGain); // Use currentMaxEnergy
                                            this.energyGainedFromEating += energyGain;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (this.isUnstable) return; 

        // Final Instability Checks: Springs and Span
        const localMaxSpringStretchFactor = config.MAX_SPRING_STRETCH_FACTOR;
        const localMaxSpanPerPointFactor = config.MAX_SPAN_PER_POINT_FACTOR;

        for (const spring of this.springs) {
            // const currentLength = spring.p1.pos.sub(spring.p2.pos).mag();
            this._tempVec1.copyFrom(spring.p1.pos).subInPlace(spring.p2.pos);
            const currentLength = this._tempVec1.mag();
            if (currentLength > spring.restLength * localMaxSpringStretchFactor) {
                this._markUnstable('physics_spring_overstretch');
                // console.warn(...)
                return;
            }
        }

        if (this.massPoints.length > 2) { 
            const bbox = this.getBoundingBox();
            if (bbox.width > this.massPoints.length * localMaxSpanPerPointFactor ||
                bbox.height > this.massPoints.length * localMaxSpanPerPointFactor) {
                this._markUnstable('physics_span_exceeded');
                // console.warn(...)
                return;
            }
        }

        this.ticksSinceBirth++;
        this.absoluteAgeTicks = Math.max(0, Math.floor(Number(this.absoluteAgeTicks) || 0) + 1);
        if (Number.isFinite(Number(this.ticksSinceLastReproduction))) {
            this.ticksSinceLastReproduction = Math.max(0, Math.floor(Number(this.ticksSinceLastReproduction) + 1));
        }

        // Check for max age
        if (this.ticksSinceBirth > config.MAX_CREATURE_AGE_TICKS) {
            this._markUnstable('age_limit');
            return; // Creature dies of old age
        }

        if (this.ticksSinceBirth > this.effectiveReproductionCooldown) { // Use effective cooldown
            this.canReproduce = true;
        }
    }

    getAverageStiffness() {
        if (this.springs.length === 0) return 0;
        const nonRigidSprings = this.springs.filter(s => !s.isRigid);
        if (nonRigidSprings.length === 0) return config.RIGID_SPRING_STIFFNESS;
        const totalStiffness = nonRigidSprings.reduce((sum, spring) => sum + spring.stiffness, 0);
        return totalStiffness / nonRigidSprings.length;
    }

    getAverageDamping() {
        if (this.springs.length === 0) return 0;
        const nonRigidSprings = this.springs.filter(s => !s.isRigid);
        if (nonRigidSprings.length === 0) return config.RIGID_SPRING_DAMPING;
        const totalDamping = nonRigidSprings.reduce((sum, spring) => sum + spring.dampingFactor, 0);
        return totalDamping / nonRigidSprings.length;
    }

    _updateJetAndSwimmerFluidSensor(fluidFieldRef) {
        if (!fluidFieldRef) return;
        this.massPoints.forEach(point => {
            if (point.nodeType === NodeType.JET || point.nodeType === NodeType.SWIMMER) {
                const fluidGridX = Math.floor(point.pos.x / fluidFieldRef.scaleX);
                const fluidGridY = Math.floor(point.pos.y / fluidFieldRef.scaleY);
                const idx = fluidFieldRef.IX(fluidGridX, fluidGridY);
                const fluidVelX = fluidFieldRef.Vx[idx];
                const fluidVelY = fluidFieldRef.Vy[idx];
                point.sensedFluidVelocity.x = fluidVelX;
                point.sensedFluidVelocity.y = fluidVelY;
            }
        });
    }

    translateBody(dx, dy) {
        this.massPoints.forEach(p => {
            p.pos.x += dx;
            p.pos.y += dy;
            p.prevPos.x += dx;
            p.prevPos.y += dy;
        });
    }
    getMinX() { return Math.min(...this.massPoints.map(p => p.pos.x - p.radius)); }
    getMaxX() { return Math.max(...this.massPoints.map(p => p.pos.x + p.radius)); }
    getMinY() { return Math.min(...this.massPoints.map(p => p.pos.y - p.radius)); }
    getMaxY() { return Math.max(...this.massPoints.map(p => p.pos.y + p.radius)); }

    /**
     * Count nearby living creatures around a point for local density pressure.
     */
    _countNearbyCreatures(center, radius) {
        const crowd = Array.isArray(runtimeState.softBodyPopulation) ? runtimeState.softBodyPopulation : [];
        const r = Math.max(1, Number(radius) || 1);
        const rSq = r * r;
        let count = 0;

        for (const other of crowd) {
            if (!other || other === this || other.isUnstable) continue;
            const otherCenter = other.getAveragePosition ? other.getAveragePosition() : null;
            if (!otherCenter) continue;
            const dx = otherCenter.x - center.x;
            const dy = otherCenter.y - center.y;
            if (dx * dx + dy * dy <= rSq) count++;
        }
        return count;
    }

    /**
     * Sample local nutrient/light values at a world position.
     */
    _sampleReproductionResourcesAt(worldX, worldY) {
        if (!runtimeState.fluidField) return null;
        const fluid = runtimeState.fluidField;
        const gx = Math.floor(worldX / fluid.scaleX);
        const gy = Math.floor(worldY / fluid.scaleY);
        const idx = fluid.IX(gx, gy);

        const nutrient = this.nutrientField && idx >= 0 && idx < this.nutrientField.length
            ? (this.nutrientField[idx] * config.globalNutrientMultiplier)
            : null;
        const light = this.lightField && idx >= 0 && idx < this.lightField.length
            ? (this.lightField[idx] * config.globalLightMultiplier)
            : null;

        return { idx, nutrient, light };
    }

    /**
     * Debit local nutrient/light resources when offspring are produced.
     */
    _applyReproductionResourceDebitAt(worldX, worldY) {
        const sample = this._sampleReproductionResourcesAt(worldX, worldY);
        if (!sample) return;

        applyReproductionResourceDebit({
            nutrientField: this.nutrientField,
            lightField: this.lightField,
            index: sample.idx,
            nutrientDebit: config.REPRO_RESOURCE_NUTRIENT_DEBIT_PER_OFFSPRING,
            lightDebit: config.REPRO_RESOURCE_LIGHT_DEBIT_PER_OFFSPRING,
            nutrientMin: config.REPRO_RESOURCE_FIELD_MIN_CLAMP,
            lightMin: config.REPRO_RESOURCE_FIELD_MIN_CLAMP
        });

        this.reproductionResourceDebitApplied += 1;
    }

    /**
     * Attempt reproduction with density-dependent fertility and local resource coupling.
     */
    reproduce({ maxOffspring = null } = {}) {
        if (this.failedReproductionCooldown > 0) {
            this.failedReproductionCooldown--;
            return []; // On cooldown from a previous failed attempt
        }

        if (this.isUnstable || !this.canReproduce || !config.canCreaturesReproduceGlobally) return []; // Check global flag

        const parentAvgPos = this.getAveragePosition();
        const nearbyCreatures = this._countNearbyCreatures(parentAvgPos, config.REPRO_LOCAL_DENSITY_RADIUS);
        const densityScale = computeDensityFertilityScale({
            population: Array.isArray(runtimeState.softBodyPopulation) ? runtimeState.softBodyPopulation.length : 0,
            floor: config.CREATURE_POPULATION_FLOOR,
            ceiling: config.CREATURE_POPULATION_CEILING,
            globalSoftMultiplier: config.REPRO_FERTILITY_GLOBAL_SOFT_MULTIPLIER,
            globalHardMultiplier: config.REPRO_FERTILITY_GLOBAL_HARD_MULTIPLIER,
            globalMinScale: config.REPRO_FERTILITY_GLOBAL_MIN_SCALE,
            localNeighbors: nearbyCreatures,
            localSoftNeighbors: config.REPRO_FERTILITY_LOCAL_SOFT_NEIGHBORS,
            localHardNeighbors: config.REPRO_FERTILITY_LOCAL_HARD_NEIGHBORS,
            localMinScale: config.REPRO_FERTILITY_LOCAL_MIN_SCALE
        });

        let resourceFertilityScale = 1;
        const resourceSample = this._sampleReproductionResourcesAt(parentAvgPos.x, parentAvgPos.y);
        if (resourceSample && resourceSample.nutrient !== null && resourceSample.light !== null) {
            const resourceCoupling = evaluateResourceCoupling({
                nutrientValue: resourceSample.nutrient,
                lightValue: resourceSample.light,
                minNutrient: config.REPRO_RESOURCE_MIN_NUTRIENT,
                minLight: config.REPRO_RESOURCE_MIN_LIGHT
            });

            if (!resourceCoupling.allow) {
                this.reproductionSuppressedByResources += 1;
                return [];
            }

            resourceFertilityScale = resourceCoupling.fertilityScale;
        }

        const dyeState = this._computeDyeEcologyStateAt(parentAvgPos.x, parentAvgPos.y);
        this.lastDyeEcologyState = dyeState;
        const dyeFertilityScale = this._resolveDyeEffectScale(dyeState, {
            weight: Math.max(0, Number(config.DYE_REPRO_EFFECT_WEIGHT) || 0.7)
        });

        const fertilityScale = Math.max(
            config.REPRO_MIN_FERTILITY_SCALE,
            Math.max(0, Math.min(1, densityScale.scale * resourceFertilityScale * dyeFertilityScale))
        );

        if (Math.random() > fertilityScale) {
            if (densityScale.scale < 1) this.reproductionSuppressedByDensity += 1;
            if (dyeFertilityScale < 1) this.reproductionSuppressedByDye += 1;
            this.reproductionSuppressedByFertilityRoll += 1;
            return [];
        }

        const energyForOneOffspring = this.currentMaxEnergy * config.OFFSPRING_INITIAL_ENERGY_SHARE; // Use currentMaxEnergy for cost basis
        const hadEnoughEnergyForAttempt = this.creatureEnergy >= energyForOneOffspring;

        let successfullyPlacedOffspring = 0;
        const offspring = [];

        // Pre-calculate spatial info for existing bodies to optimize collision checks
        const existingBodiesSpatialInfo = [];
        for (const body of runtimeState.softBodyPopulation) {
            if (body !== this && !body.isUnstable) { // Don't include self or unstable bodies
                existingBodiesSpatialInfo.push({
                    center: body.getAveragePosition(),
                    radius: body.blueprintRadius
                });
            }
        }

        const maxOffspringInt = Number.isFinite(Number(maxOffspring)) ? Math.max(0, Math.floor(Number(maxOffspring))) : null;
        const targetOffspring = (maxOffspringInt === null) ? this.numOffspring : Math.min(this.numOffspring, maxOffspringInt);

        for (let i = 0; i < targetOffspring; i++) {
            if (this.creatureEnergy < energyForOneOffspring) break; // Not enough energy for this one

            let placedThisOffspring = false;
            for (let attempt = 0; attempt < config.OFFSPRING_PLACEMENT_ATTEMPTS; attempt++) {
                const angle = Math.random() * Math.PI * 2;
                const radiusOffset = this.offspringSpawnRadius * (0.5 + Math.random() * 0.5); // offspringSpawnRadius is a gene
                const offsetX = Math.cos(angle) * radiusOffset;
                const offsetY = Math.sin(angle) * radiusOffset;

                const spawnX = parentAvgPos.x + offsetX;
                const spawnY = parentAvgPos.y + offsetY;

                // Create the potential child. Its blueprintRadius will be calculated in its constructor.
                // We will assign a proper ID only if placement is successful.
                const potentialChild = new SoftBody(-1, spawnX, spawnY, this); // Use -1 or a temporary ID marker
                potentialChild.setNutrientField(this.nutrientField);
                potentialChild.setLightField(this.lightField);
                potentialChild.setParticles(this.particles);
                potentialChild.setSpatialGrid(this.spatialGrid);

                if (potentialChild.massPoints.length === 0 || potentialChild.blueprintRadius === 0) continue;

                const childViability = potentialChild._evaluateBlueprintViability();
                if (!childViability.ok) {
                    if (childViability.reasons.structure) this._bumpMutationStat('offspringViabilityRejectedStructure');
                    if (childViability.reasons.diversity) this._bumpMutationStat('offspringViabilityRejectedDiversity');
                    if (childViability.reasons.harvest) this._bumpMutationStat('offspringViabilityRejectedHarvest');
                    if (childViability.reasons.actuator) this._bumpMutationStat('offspringViabilityRejectedActuator');
                    continue;
                }

                let isSpotClear = true;
                // Check against existing population using blueprintRadius and cached positions
                for (const otherBodyInfo of existingBodiesSpatialInfo) {
                    const distSq = (spawnX - otherBodyInfo.center.x)**2 + (spawnY - otherBodyInfo.center.y)**2;
                    // Use the sum of blueprint radii plus a clearance value for the check
                    const combinedRadii = potentialChild.blueprintRadius + otherBodyInfo.radius + config.OFFSPRING_PLACEMENT_CLEARANCE_RADIUS;
                    if (distSq < combinedRadii * combinedRadii) {
                        isSpotClear = false;
                        break;
                    }
                }
                // Check against already spawned new offspring in this cycle
                if (isSpotClear) {
                    for (const newBorn of offspring) { // offspring contains fully created children
                        const newBornCenter = newBorn.getAveragePosition();
                        const distSq = (spawnX - newBornCenter.x)**2 + (spawnY - newBornCenter.y)**2;
                        const combinedRadii = potentialChild.blueprintRadius + newBorn.blueprintRadius + config.OFFSPRING_PLACEMENT_CLEARANCE_RADIUS;
                        if (distSq < combinedRadii * combinedRadii) {
                            isSpotClear = false;
                            break;
                        }
                    }
                }

                if (isSpotClear) {
                    this.creatureEnergy -= energyForOneOffspring;
                    this._applyReproductionResourceDebitAt(spawnX, spawnY);

                    // Optimization: tempChild becomes the finalChild
                    potentialChild.id = this.id + successfullyPlacedOffspring; // Assign final ID and increment global counter
                    potentialChild.birthOrigin = 'reproduction_offspring';
                    potentialChild.parentBodyId = Number.isFinite(Number(this.id)) ? Number(this.id) : null;
                    potentialChild.lineageRootId = Number.isFinite(Number(this.lineageRootId))
                        ? Number(this.lineageRootId)
                        : (Number.isFinite(Number(this.id)) ? Number(this.id) : null);
                    potentialChild.generation = Math.max(0, Math.floor(Number(this.generation) || 0) + 1);
                    potentialChild.absoluteAgeTicks = 0;
                    potentialChild.reproductionEventsCompleted = 0;
                    potentialChild.ticksSinceLastReproduction = null;
                    potentialChild.creatureEnergy = energyForOneOffspring;
                    offspring.push(potentialChild);

                    successfullyPlacedOffspring++;
                    placedThisOffspring = true;
                    break; // Break from placement attempts for this offspring
                }
            }
            if (!placedThisOffspring && this.creatureEnergy < energyForOneOffspring) {
                break; // Not enough energy for further attempts for other offspring either
            }
        }

        if (successfullyPlacedOffspring > 0) {
            this.creatureEnergy *= (1 - config.REPRODUCTION_ADDITIONAL_COST_FACTOR);
            if (this.creatureEnergy < 0) this.creatureEnergy = 0;
            this.ticksSinceBirth = 0;
            this.reproductionEventsCompleted = Math.max(0, Math.floor(Number(this.reproductionEventsCompleted) || 0) + 1);
            this.ticksSinceLastReproduction = 0;
            this.canReproduce = false;
            this.justReproduced = true; // Set the flag here
        } else if (hadEnoughEnergyForAttempt && successfullyPlacedOffspring === 0) {
            // If had enough energy but couldn't place any offspring (e.g., due to space)
            this.failedReproductionCooldown = config.FAILED_REPRODUCTION_COOLDOWN_TICKS;
        }
        return offspring;
    }

    /**
     * Compute axis-aligned bounds over all mass points (including radius).
     */
    getBoundingBox() {
        if (this.massPoints.length === 0) {
            return { minX: 0, minY: 0, maxX: 0, maxY: 0, width: 0, height: 0 };
        }
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.massPoints.forEach(p => {
            minX = Math.min(minX, p.pos.x - p.radius);
            minY = Math.min(minY, p.pos.y - p.radius);
            maxX = Math.max(maxX, p.pos.x + p.radius);
            maxY = Math.max(maxY, p.pos.y + p.radius);
        });
        return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
    }

    /**
     * Return center-of-mass position approximation from current point locations.
     */
    getAveragePosition() {
        if (this.massPoints.length === 0) return new Vec2(config.WORLD_WIDTH/2, config.WORLD_HEIGHT/2);
        let sumX = 0, sumY = 0;
        this.massPoints.forEach(p => { sumX += p.pos.x; sumY += p.pos.y; });
        return new Vec2(sumX / this.massPoints.length, sumY / this.massPoints.length);
    }

    /**
     * Return center-of-mass velocity approximation from Verlet positions.
     */
    getAverageVelocity() {
        if (this.massPoints.length === 0) return new Vec2(0,0);
        let sumVelX = 0, sumVelY = 0;
        this.massPoints.forEach(p => {
            const velX = p.pos.x - p.prevPos.x;
            const velY = p.pos.y - p.prevPos.y;
            sumVelX += velX;
            sumVelY += velY;
        });
        return new Vec2(sumVelX / this.massPoints.length, sumVelY / this.massPoints.length);
    }

    /**
     * Re-initialize/resize the creature brain after topology changes.
     *
     * This delegates to the dedicated Brain class implementation, which can
     * preserve existing weights where possible and only initialize new slices.
     */
    initializeBrain() {
        if (!this.brain) {
            this.brain = new Brain(this);
            return;
        }
        this.brain.initialize();
    }

    _findOrCreateBrainNode() {
        let brainNode = null;
        // First, try to find an already designated brain
        for (const point of this.massPoints) {
            if (point.neuronData && point.neuronData.isBrain) {
                brainNode = point;
                break;
            }
        }

        // If no brain was pre-designated, find the first NEURON type node
        if (!brainNode) {
            for (const point of this.massPoints) {
                if (point.nodeType === NodeType.NEURON) {
                    if (!point.neuronData) { // Ensure neuronData exists
                        point.neuronData = { 
                            hiddenLayerSize: config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MIN - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1))
                        };
                    }
                    point.neuronData.isBrain = true;
                    brainNode = point; 
                    // Ensure other neurons are not brains
                    this.massPoints.forEach(otherP => {
                        if (otherP !== point && otherP.nodeType === NodeType.NEURON && otherP.neuronData) {
                            otherP.neuronData.isBrain = false;
                        }
                    });
                    break; 
                }
            }
        }
        return brainNode;
    }

    _calculateBrainVectorSizes(brainNode) {
        const nd = brainNode.neuronData;
        const numEmitterPoints = this.numEmitterNodes;
        const numSwimmerPoints = this.numSwimmerNodes;
        const numEaterPoints = this.numEaterNodes;
        const numPredatorPoints = this.numPredatorNodes;
        const numEyeNodes = this.numEyeNodes;
        const numJetNodes = this.numJetNodes;
        const numPotentialGrabberPoints = this.numPotentialGrabberNodes;
        const numAttractorPoints = this.numAttractorNodes;
        const numRepulsorPoints = this.numRepulsorNodes;

        // console.log(`Body ${this.id} _calculateBrainVectorSizes: Using Stored Counts: E:${numEmitterPoints}, S:${numSwimmerPoints}, Ea:${numEaterPoints}, P:${numPredatorPoints}, G:${numPotentialGrabberPoints}, Ey:${numEyeNodes}`);
        nd.inputVectorSize = config.NEURAL_INPUT_SIZE_BASE +
                             (numEyeNodes * config.NEURAL_INPUTS_PER_EYE) +
                             (numSwimmerPoints * config.NEURAL_INPUTS_PER_FLUID_SENSOR) +
                             (numJetNodes * config.NEURAL_INPUTS_PER_FLUID_SENSOR) +
                             (this.springs.length * config.NEURAL_INPUTS_PER_SPRING_SENSOR);
        // console.log(`Body ${this.id} _calculateBrainVectorSizes: Counts (from stored) directly before sum: E:${numEmitterPoints}, S:${numSwimmerPoints}, Ea:${numEaterPoints}, P:${predatorPoints}, G:${numPotentialGrabberPoints}`);
        nd.outputVectorSize = (numEmitterPoints * config.NEURAL_OUTPUTS_PER_EMITTER) +
                              (numSwimmerPoints * config.NEURAL_OUTPUTS_PER_SWIMMER) +
                              (numEaterPoints * config.NEURAL_OUTPUTS_PER_EATER) +
                              (numPredatorPoints * config.NEURAL_OUTPUTS_PER_PREDATOR) +
                              (numJetNodes * config.NEURAL_OUTPUTS_PER_JET) +
                              (numPotentialGrabberPoints * config.NEURAL_OUTPUTS_PER_GRABBER_TOGGLE) +
                              (numAttractorPoints * config.NEURAL_OUTPUTS_PER_ATTRACTOR) +
                              (numRepulsorPoints * config.NEURAL_OUTPUTS_PER_REPULSOR);
        // console.log(`Body ${this.id} _calculateBrainVectorSizes: Calculated nd.outputVectorSize = ${nd.outputVectorSize}`);
    }

    _initializeBrainWeightsAndBiases(brainNode) {
        const nd = brainNode.neuronData;
        if (typeof nd.hiddenLayerSize !== 'number' || nd.hiddenLayerSize < config.DEFAULT_HIDDEN_LAYER_SIZE_MIN || nd.hiddenLayerSize > config.DEFAULT_HIDDEN_LAYER_SIZE_MIN) {
            nd.hiddenLayerSize = config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + Math.floor(Math.random() * (config.DEFAULT_HIDDEN_LAYER_SIZE_MIN - config.DEFAULT_HIDDEN_LAYER_SIZE_MIN + 1));
        }

        if (!nd.weightsIH || nd.weightsIH.length !== nd.hiddenLayerSize || (nd.weightsIH.length > 0 && nd.weightsIH[0].length !== nd.inputVectorSize) ) {
            nd.weightsIH = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize);
            nd.biasesH = initializeVector(nd.hiddenLayerSize);
            // console.log(`Body ${this.id} brain: Initialized weightsIH/biasesH. Inputs: ${nd.inputVectorSize}, Hidden: ${nd.hiddenLayerSize}`);
        }

        if (!nd.weightsHO || nd.weightsHO.length !== nd.outputVectorSize || (nd.weightsHO.length > 0 && nd.weightsHO[0].length !== nd.hiddenLayerSize) ) {
            nd.weightsHO = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize);
            nd.biasesO = initializeVector(nd.outputVectorSize);
            // console.log(`Body ${this.id} brain: Initialized weightsHO/biasesO. Outputs: ${nd.outputVectorSize}, Hidden: ${nd.hiddenLayerSize}`);
        }
    }

    _initializeBrainRLComponents(brainNode) {
        const nd = brainNode.neuronData;
        if (!nd.experienceBuffer) nd.experienceBuffer = [];
        if (typeof nd.framesSinceLastTrain !== 'number') nd.framesSinceLastTrain = 0; 
        if (typeof nd.previousEnergyForReward !== 'number') nd.previousEnergyForReward = this.creatureEnergy;
        if (typeof nd.previousEnergyChangeForNN !== 'number') nd.previousEnergyChangeForNN = 0; // New: For 2nd derivative input
        if (typeof nd.lastAvgNormalizedReward !== 'number') nd.lastAvgNormalizedReward = 0;
        if (typeof nd.maxExperienceBufferSize !== 'number') nd.maxExperienceBufferSize = 10;
        nd.currentFrameInputVectorWithLabels = []; // New: For UI display
        nd.currentFrameActionDetails = []; // Already exists, just ensure it's initialized
    }

    /**
     * Compute trajectory reward-to-go for legacy in-class RL updater.
     */
    calculateDiscountedRewards(rewards, gamma) {
        const discountedRewards = new Array(rewards.length);
        let runningAdd = 0;
        for (let i = rewards.length - 1; i >= 0; i--) {
            runningAdd = rewards[i] + gamma * runningAdd;
            discountedRewards[i] = runningAdd;
        }
        return discountedRewards;
    }

    /**
     * Legacy policy-gradient updater retained for compatibility with older flows.
     */
    updateBrainPolicy() {
        let brainNode = null;
        for (const point of this.massPoints) {
            if (point.neuronData && point.neuronData.isBrain) {
                brainNode = point;
                break;
            }
        }

        if (!brainNode || !brainNode.neuronData) return;
        const nd = brainNode.neuronData;

        if (!nd.experienceBuffer || nd.experienceBuffer.length < nd.maxExperienceBufferSize) {
            return; // Not enough experiences to train yet
        }

        const states = nd.experienceBuffer.map(exp => exp.state);
        const actionDetailsBatch = nd.experienceBuffer.map(exp => exp.actionDetails); // Array of arrays of actionDetail objects
        const rewards = nd.experienceBuffer.map(exp => exp.reward);

        const discountedRewards = this.calculateDiscountedRewards(rewards, config.DISCOUNT_FACTOR_GAMMA);

        // Normalize discounted rewards (optional but good practice)
        let meanDiscountedReward = 0;
        for (const r of discountedRewards) meanDiscountedReward += r;
        meanDiscountedReward /= discountedRewards.length;

        let stdDevDiscountedReward = 0;
        for (const r of discountedRewards) stdDevDiscountedReward += (r - meanDiscountedReward) ** 2;
        stdDevDiscountedReward = Math.sqrt(stdDevDiscountedReward / discountedRewards.length);

        const normalizedDiscountedRewards = discountedRewards.map(
            r => (r - meanDiscountedReward) / (stdDevDiscountedReward + 1e-6) // Add epsilon for stability
        );

        // Store the average normalized reward for diagnostics
        nd.lastAvgNormalizedReward = meanDiscountedReward; // Using mean before normalization for a more direct sense of reward scale
        // If you prefer the normalized mean: nd.lastAvgNormalizedReward = normalizedDiscountedRewards.reduce((a, b) => a + b, 0) / normalizedDiscountedRewards.length;

        // Initialize gradients for weights and biases (accumulate over the batch)
        const gradWeightsHO_acc = initializeMatrix(nd.outputVectorSize, nd.hiddenLayerSize, 0); // Initialize with zeros
        const gradBiasesO_acc = initializeVector(nd.outputVectorSize, 0);
        const gradWeightsIH_acc = initializeMatrix(nd.hiddenLayerSize, nd.inputVectorSize, 0);
        const gradBiasesH_acc = initializeVector(nd.hiddenLayerSize, 0);


        // Iterate through each trajectory/experience in the batch
        for (let t = 0; t < nd.experienceBuffer.length; t++) {
            const state_t = states[t];
            const actionDetails_t = actionDetailsBatch[t]; // This is an array of actionDetail objects for this timestep
            const G_t_normalized = normalizedDiscountedRewards[t];

            // Re-run forward pass to get hidden activations for this state_t
            // (assuming weights haven't changed *during* this batch processing)
            const hiddenLayerInputs_t = multiplyMatrixVector(nd.weightsIH, state_t);
            const hiddenLayerBiasedInputs_t = addVectors(hiddenLayerInputs_t, nd.biasesH);
            const hiddenActivations_t = hiddenLayerBiasedInputs_t.map(val => Math.tanh(val));

            // Gradients for Output Layer (weightsHO, biasesO)
            let currentActionDetailIdx = 0;
            for (let i = 0; i < nd.outputVectorSize / 2; i++) { // Iterate through 6 logical actions
                const meanOutputIdx = i * 2;
                const stdDevOutputIdx = i * 2 + 1;

                const ad = actionDetails_t[currentActionDetailIdx]; // Access detail for the current logical action

                if (!ad) { 
                    // This implies that actionDetails_t (from experience buffer) is shorter 
                    // than what outputVectorSize/2 expects. This could happen if not all 
                    // controllable points could take actions in that specific past frame 
                    // (e.g., rawOutputs array was too short for them). 
                    // We can't calculate a gradient for a missing action detail.
                    // console.warn(`Body ${this.id} updateBrainPolicy: Missing action detail at index ${currentActionDetailIdx} for timestep ${t}. Skipping this action component's gradient.`);
                    currentActionDetailIdx++; // Still need to advance to the next logical action slot
                    continue; 
                }
                currentActionDetailIdx++; // Increment after successful access and use.

                const { sampledAction, mean, stdDev } = ad;
                const grad_logProb_d_mean = (sampledAction - mean) / (stdDev * stdDev + 1e-9); 
                const grad_logProb_d_stdDev_output = (((sampledAction - mean) ** 2) - (stdDev * stdDev)) / (stdDev * stdDev * stdDev + 1e-9);

                // Error for the mean output neuron
                const error_mean = G_t_normalized * grad_logProb_d_mean;
                // Error for the stdDev output neuron (raw output from NN, before exp())
                // actualStdDev = exp(rawStdDev) + eps --> d(actualStdDev)/d(rawStdDev) approx actualStdDev
                const error_stdDev = G_t_normalized * grad_logProb_d_stdDev_output * (stdDev - 1e-6); // (stdDev-eps) is approx exp(rawStdDev)

                for (let k = 0; k < nd.hiddenLayerSize; k++) {
                    gradWeightsHO_acc[meanOutputIdx][k] += error_mean * hiddenActivations_t[k];
                    gradWeightsHO_acc[stdDevOutputIdx][k] += error_stdDev * hiddenActivations_t[k];
                }
                gradBiasesO_acc[meanOutputIdx] += error_mean;
                gradBiasesO_acc[stdDevOutputIdx] += error_stdDev;
            }

            // Gradients for Hidden Layer (weightsIH, biasesH)
            const d_tanh = (x) => 1 - x * x; // Derivative of tanh(y) is 1 - tanh(y)^2 where x = tanh(y)
            for (let h = 0; h < nd.hiddenLayerSize; h++) {
                let sum_error_weighted_HO = 0;
                for (let j = 0; j < nd.outputVectorSize; j++) {
                    // Determine if j is a mean or stdDev output to get the correct error term
                    const actionDetailIndex = Math.floor(j/2);
                    if (actionDetailIndex >= actionDetails_t.length) {
                        // This experience was recorded with a different number of actions
                        // than the current network configuration. Skip this gradient component.
                        console.warn(`Skipping gradient for output ${j} in experience ${t}: actionDetailIndex ${actionDetailIndex} out of bounds for actionDetails_t length ${actionDetails_t.length}`);
                        continue; 
                    }

                    const currentActionDetail = actionDetails_t[actionDetailIndex];
                    if (!currentActionDetail) { // Should ideally not happen if the length check passes
                        console.warn(`Skipping gradient for output ${j} in experience ${t}: currentActionDetail is undefined at index ${actionDetailIndex}`);
                        continue;
                    }

                    const error_from_output_j = (j % 2 === 0) ? 
                        G_t_normalized * (currentActionDetail.sampledAction - currentActionDetail.mean) / (currentActionDetail.stdDev**2 + 1e-9) :
                        G_t_normalized * (((currentActionDetail.sampledAction - currentActionDetail.mean)**2 - currentActionDetail.stdDev**2) / (currentActionDetail.stdDev**3 + 1e-9)) * (currentActionDetail.stdDev - 1e-6);
                    sum_error_weighted_HO += error_from_output_j * nd.weightsHO[j][h];
                }
                const error_h = sum_error_weighted_HO * d_tanh(hiddenActivations_t[h]);

                for (let i_input = 0; i_input < nd.inputVectorSize; i_input++) {
                    gradWeightsIH_acc[h][i_input] += error_h * state_t[i_input];
                }
                gradBiasesH_acc[h] += error_h;
            }
        }

        // Apply accumulated gradients (averaged over batch size implicitly by not dividing here, or explicitly divide)
        const batchSize = nd.experienceBuffer.length;
        for (let j = 0; j < nd.outputVectorSize; j++) {
            for (let k = 0; k < nd.hiddenLayerSize; k++) {
                nd.weightsHO[j][k] += config.LEARNING_RATE * gradWeightsHO_acc[j][k] / batchSize;
            }
            nd.biasesO[j] += config.LEARNING_RATE * gradBiasesO_acc[j] / batchSize;
        }
        for (let h = 0; h < nd.hiddenLayerSize; h++) {
            for (let i = 0; i < nd.inputVectorSize; i++) {
                nd.weightsIH[h][i] += config.LEARNING_RATE * gradWeightsIH_acc[h][i] / batchSize;
            }
            nd.biasesH[h] += config.LEARNING_RATE * gradBiasesH_acc[h] / batchSize;
        }

        nd.experienceBuffer = []; // Clear buffer after training
        nd.framesSinceLastTrain = 0;
        // console.log(`Body ${this.id} brain updated. Normalized G_t example: ${normalizedDiscountedRewards[0]}`);
    }

    drawSelf(ctx) {
        if (this.isUnstable) return;
        for (let spring of this.springs) spring.draw(ctx);
        for (let point of this.massPoints) point.draw(ctx);
    }

    // Helper function to find linear segments of points
    findLinearSegments(minLength, maxLength) {
        const segments = [];
        if (this.massPoints.length < minLength) return segments;

        const adjacency = new Map(); // point_index -> [neighbor_indices]
        this.massPoints.forEach((_, i) => adjacency.set(i, []));

        this.springs.forEach(spring => {
            const p1Index = this.massPoints.indexOf(spring.p1);
            const p2Index = this.massPoints.indexOf(spring.p2);
            if (p1Index !== -1 && p2Index !== -1) {
                adjacency.get(p1Index).push(p2Index);
                adjacency.get(p2Index).push(p2Index);
            }
        });

        for (let i = 0; i < this.massPoints.length; i++) {
            // Try to start a segment from each point
            // Explore paths using DFS/BFS to find linear chains
            const stack = [[i, [i]]]; // [current_point_index, path_so_far_indices]
            const visitedInPath = new Set(); // To avoid cycles in the current path being built

            while (stack.length > 0) {
                const [currentIndex, currentPathIndices] = stack.pop();
                visitedInPath.add(currentIndex);

                if (currentPathIndices.length >= minLength) {
                    segments.push(currentPathIndices.map(idx => this.massPoints[idx]));
                }

                if (currentPathIndices.length < maxLength) {
                    const neighbors = adjacency.get(currentIndex) || [];
                    for (const neighborIndex of neighbors) {
                        if (!currentPathIndices.includes(neighborIndex)) { // Avoid immediate backtracking and simple cycles
                             // Check if adding this neighbor would still constitute a "linear-enough" extension.
                             // For this version, we are more permissive: any sequence of connected points.
                            stack.push([neighborIndex, [...currentPathIndices, neighborIndex]]);
                        }
                    }
                }
            }
        }

        // Post-process to remove duplicate or fully contained sub-segments if desired, but for now, allow overlaps.
        // Ensure segments are unique point sequences if multiple paths lead to same point sequence.
        const uniqueSegments = [];
        const segmentSignatures = new Set();
        for (const seg of segments) {
            const signature = seg.map(p => this.massPoints.indexOf(p)).sort((a,b)=>a-b).join('-'); // Create a unique signature
            if(!segmentSignatures.has(signature)){
                segmentSignatures.add(signature);
                uniqueSegments.push(seg);
            }
        }
        return uniqueSegments;
    }

    exportBlueprint() {
        const blueprint = {
            version: 2,
            stiffness: this.stiffness,
            springDamping: this.springDamping,
            motorImpulseInterval: this.motorImpulseInterval,
            motorImpulseMagnitudeCap: this.motorImpulseMagnitudeCap,
            emitterStrength: this.emitterStrength,
            emitterDirection: { x: this.emitterDirection.x, y: this.emitterDirection.y },
            numOffspring: this.numOffspring,
            offspringSpawnRadius: this.offspringSpawnRadius,
            pointAddChance: this.pointAddChance,
            springConnectionRadius: this.springConnectionRadius,
            reproductionEnergyThreshold: this.reproductionEnergyThreshold,
            jetMaxVelocityGene: this.jetMaxVelocityGene,
            reproductionCooldownGene: this.reproductionCooldownGene,
            defaultActivationPattern: this.defaultActivationPattern,
            defaultActivationLevel: this.defaultActivationLevel,
            defaultActivationPeriod: this.defaultActivationPeriod,
            defaultActivationPhaseOffset: this.defaultActivationPhaseOffset,
            rlAlgorithmType: this.rlAlgorithmType,
            rewardStrategy: this.rewardStrategy,
            growthGenome: this.growthGenome,
            dyePreferredHue: this.dyePreferredHue,
            dyeHueTolerance: this.dyeHueTolerance,
            dyeResponseGain: this.dyeResponseGain,
            dyeResponseSign: this.dyeResponseSign,
            dyeNodeTypeAffinity: this.dyeNodeTypeAffinity,
            blueprintPoints: this.blueprintPoints,
            blueprintSprings: this.blueprintSprings
        };
        return blueprint;
    }
} 
