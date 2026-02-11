import config from './config.js';
import {SoftBody} from './classes/SoftBody.js';
import {Particle} from './classes/Particle.js';
import {FluidField} from './classes/FluidField.js';
import {GPUFluidField} from './gpuFluidField.js';
import {isWebGpuSupported} from './gpuUtils.js';
import {perlin} from './utils.js';
import {
    canvas,
    ctx, updateInfoPanel,
    updateInstabilityIndicator,
    updatePopulationCount
} from "./ui.js";
import viewport from './viewport.js';
import { drawNutrientMap, drawLightMap, drawViscosityMap } from './environment.js';
import { syncRuntimeState } from './engine/runtimeState.js';

let offscreenFluidCanvas, offscreenFluidCtx;
let spatialGrid;
let softBodyPopulation = [];
let fluidField = null;
let particles = [];
let nextSoftBodyId = 0;
let nutrientField = null;
let lightField = null;
let viscosityField = null;

let globalEnergyGains = {
    photosynthesis: 0,
    eating: 0,
    predation: 0
};

let globalEnergyCosts = {
    baseNodes: 0,
    emitterNodes: 0,
    eaterNodes: 0,
    predatorNodes: 0,
    neuronNodes: 0,
    swimmerNodes: 0,
    photosyntheticNodes: 0,
    grabbingNodes: 0,
    eyeNodes: 0,
    jetNodes: 0,
    attractorNodes: 0,
    repulsorNodes: 0
};

let mutationStats = { // New: For tracking mutation occurrences
    springStiffness: 0,
    springDamping: 0,
    motorInterval: 0,
    motorCap: 0,
    emitterStrength: 0,
    emitterDirection: 0,
    numOffspring: 0,
    offspringSpawnRadius: 0,
    pointAddChanceGene: 0, // Mutation of the gene itself
    springConnectionRadiusGene: 0, // Mutation of the gene itself
    reproductionEnergyThreshold: 0,
    nodeTypeChange: 0,
    movementTypeChange: 0,
    springDeletion: 0,
    springAddition: 0,
    springRestLength: 0,
    springRigidityFlip: 0,
    pointAddActual: 0,
    springSubdivision: 0,
    segmentDuplication: 0,
    symmetricBodyDuplication: 0,
    bodyScale: 0,
    rewardStrategyChange: 0, // New stat for reward strategy mutations
    grabberGeneChange: 0,    // New stat for grabber gene mutations
    eyeTargetTypeChange: 0,   // New: For eye target type mutations
    jetMaxVelocityGene: 0,
    reproductionCooldownGene: 0,
    blueprintMassRadiusChange: 0,
    blueprintDyeColorChange: 0,
    blueprintCoordinateChange: 0,
    blueprintNeuronHiddenSizeChange: 0,
    shapeAddition: 0
};

syncRuntimeState({
    fluidField,
    softBodyPopulation,
    mutationStats
});

function initializeSpatialGrid() {
    spatialGrid = new Array(config.GRID_COLS * config.GRID_ROWS);
    for (let i = 0; i < config.GRID_COLS * config.GRID_ROWS; i++) {
        spatialGrid[i] = [];
    }
}

function updateSpatialGrid() {
    if (!spatialGrid) return;
    for (let i = 0; i < spatialGrid.length; i++) {
        spatialGrid[i] = [];
    }

    for (const body of softBodyPopulation) {
        if (body.isUnstable) continue;
        for (let i_p = 0; i_p < body.massPoints.length; i_p++) {
            const point = body.massPoints[i_p];
            const gx = Math.floor(point.pos.x / config.GRID_CELL_SIZE);
            const gy = Math.floor(point.pos.y / config.GRID_CELL_SIZE);
            const index = gx + gy * config.GRID_COLS;
            if (index >= 0 && index < spatialGrid.length) {
                spatialGrid[index].push({
                    type: 'softbody_point',
                    pointRef: point,
                    bodyRef: body,
                    originalIndex: i_p
                });
            }
        }
    }

    for (const particle of particles) {
        if (particle.life <= 0) continue;
        const gx = Math.floor(particle.pos.x / config.GRID_CELL_SIZE);
        const gy = Math.floor(particle.pos.y / config.GRID_CELL_SIZE);
        const index = gx + gy * config.GRID_COLS;
        if (index >= 0 && index < spatialGrid.length) {
            spatialGrid[index].push({
                type: 'particle',
                particleRef: particle
            });
        }
    }
}

// --- Simulation Setup ---
function initializePopulation() {
    softBodyPopulation = [];
    syncRuntimeState({ softBodyPopulation });
    nextSoftBodyId = 0;

    for (let i = 0; i < config.CREATURE_POPULATION_FLOOR; i++) { // Use floor for initial pop
        const margin = 50;
        const randX = margin + Math.random() * (config.WORLD_WIDTH - margin * 2);
        const randY = margin + Math.random() * (config.WORLD_HEIGHT - margin * 2);
        const softBody = new SoftBody(nextSoftBodyId++, randX, randY, null);
        softBody.setNutrientField(nutrientField);
        softBody.setLightField(lightField);
        softBody.setParticles(particles);
        softBody.setSpatialGrid(spatialGrid);
        softBodyPopulation.push(softBody);
    }
    config.isAnySoftBodyUnstable = false; // Reset the flag
    console.log(`Initialized population with ${softBodyPopulation.length} creatures.`);
}


async function initFluidSimulation(targetCanvas) {
    const dt_simulation = 1 / 60; // Assuming fixed timestep for simulation physics if needed separately
    const scaleX = config.WORLD_WIDTH / config.FLUID_GRID_SIZE_CONTROL;
    const scaleY = config.WORLD_HEIGHT / config.FLUID_GRID_SIZE_CONTROL;

    if (config.USE_GPU_FLUID_SIMULATION) {
        console.log("Attempting to initialize GPUFluidField...");
        // console.log("Canvas element being passed to GPUFluidField:", targetCanvas); 
        if (!targetCanvas || typeof targetCanvas.getContext !== 'function') {
            console.error("CRITICAL: Target canvas for GPUFluidField is invalid!");
            console.warn("Falling back to CPU FluidField due to invalid target canvas.");
            fluidField = new FluidField(config.FLUID_GRID_SIZE_CONTROL, config.FLUID_DIFFUSION, config.FLUID_VISCOSITY, dt_simulation, scaleX, scaleY);
            fluidField.setViscosityField(viscosityField);
        } else {
            fluidField = new GPUFluidField(targetCanvas, config.FLUID_GRID_SIZE_CONTROL, config.FLUID_DIFFUSION, config.FLUID_VISCOSITY, dt_simulation, scaleX, scaleY);
            await fluidField._initPromise;

            // console.log("GPUFluidField instance created and awaited in simulation.js.");

            // Check if GPUFluidField successfully initialized in WebGPU mode (fluidField.device would be set)
            // or if it fell back to WebGL (fluidField.gl would be set and fluidField.device would be null)
            // or if both failed (gpuEnabled would be false).
            if (!fluidField.gpuEnabled || (!fluidField.device && !fluidField.gl)) {
                console.warn("Fallback to CPU: GPUFluidField initialization failed (neither WebGPU nor WebGL succeeded), or gpuEnabled is false.");
                fluidField = new FluidField(config.FLUID_GRID_SIZE_CONTROL, config.FLUID_DIFFUSION, config.FLUID_VISCOSITY, dt_simulation, scaleX, scaleY);
                fluidField.setViscosityField(viscosityField);
            } else if (fluidField.device) {
                console.log("GPUFluidField successfully initialized with WebGPU in simulation.js.");
            } else if (fluidField.gl) {
                console.log("GPUFluidField successfully initialized with WebGL fallback in simulation.js.");
            }
        }
    } else {
        console.log("Initializing CPU FluidField...");
        fluidField = new FluidField(config.FLUID_GRID_SIZE_CONTROL, config.FLUID_DIFFUSION, config.FLUID_VISCOSITY, dt_simulation, scaleX, scaleY);
        fluidField.setViscosityField(viscosityField);
    }

    // Initialize offscreen canvas for CPU fluid rendering if not using GPU, or if GPU failed completely
    if (!fluidField || (fluidField instanceof FluidField && (!offscreenFluidCanvas || offscreenFluidCanvas.width !== Math.round(config.FLUID_GRID_SIZE_CONTROL) || offscreenFluidCanvas.height !== Math.round(config.FLUID_GRID_SIZE_CONTROL)))) {
        if (!(fluidField instanceof GPUFluidField) || !fluidField.gpuEnabled) { // Only if truly CPU or GPU totally failed
            offscreenFluidCanvas = document.createElement('canvas');
            offscreenFluidCanvas.width = Math.round(config.FLUID_GRID_SIZE_CONTROL);
            offscreenFluidCanvas.height = Math.round(config.FLUID_GRID_SIZE_CONTROL);
            offscreenFluidCtx = offscreenFluidCanvas.getContext('2d', {willReadFrequently: true});
            console.log("Offscreen canvas for CPU fluid rendering initialized or resized.");
        }
    } else if (fluidField instanceof GPUFluidField && fluidField.gpuEnabled) {
        // If GPU is active, we might not need the CPU offscreen canvas, or it serves a different purpose.
        // For now, let's nullify them if WebGPU/WebGL is active for fluid.
        offscreenFluidCanvas = null;
        offscreenFluidCtx = null;
        // console.log("GPU fluid active, CPU offscreen canvas resources released/nulled.");
    }

    fluidField.useWrapping = config.IS_WORLD_WRAPPING;
    fluidField.maxVelComponent = config.MAX_FLUID_VELOCITY_COMPONENT;
    config.velocityEmitters = []; // Clear any existing emitters when re-initializing
    syncRuntimeState({ fluidField });
}

function initParticles() {
    particles = [];
    config.particleEmissionDebt = 0;
}

function applyVelocityEmitters() {
    if (!fluidField || config.EMITTER_STRENGTH <= 0) return;
    for (const emitter of config.velocityEmitters) {
        fluidField.addVelocity(emitter.gridX, emitter.gridY,
            emitter.forceX * config.EMITTER_STRENGTH,
            emitter.forceY * config.EMITTER_STRENGTH);
    }
}

let followInfoRefreshCounter = 0;

function getValidFollowTarget() {
    if (config.selectedInspectBody && !config.selectedInspectBody.isUnstable && softBodyPopulation.includes(config.selectedInspectBody) && config.selectedInspectBody.massPoints.length > 0) {
        return config.selectedInspectBody;
    }
    for (const body of softBodyPopulation) {
        if (!body.isUnstable && body.massPoints && body.massPoints.length > 0) {
            return body;
        }
    }
    return null;
}

function updateAutoFollowCamera() {
    if (!config.AUTO_FOLLOW_CREATURE) return;

    const target = getValidFollowTarget();
    if (!target) {
        config.selectedInspectBody = null;
        config.selectedInspectPoint = null;
        config.selectedInspectPointIndex = -1;
        return;
    }

    if (config.selectedInspectBody !== target) {
        config.selectedInspectBody = target;
        config.selectedInspectPointIndex = 0;
        config.selectedInspectPoint = target.massPoints[0] || null;
        followInfoRefreshCounter = 0;
    } else if (!config.selectedInspectPoint || !target.massPoints.includes(config.selectedInspectPoint)) {
        config.selectedInspectPointIndex = 0;
        config.selectedInspectPoint = target.massPoints[0] || null;
    }

    const center = target.getAveragePosition();
    const bbox = target.getBoundingBox();
    const bodySize = Math.max(40, bbox.maxX - bbox.minX, bbox.maxY - bbox.minY);
    const desiredZoom = Math.min(
        config.AUTO_FOLLOW_ZOOM_MAX,
        Math.max(config.AUTO_FOLLOW_ZOOM_MIN, Math.min(canvas.clientWidth, canvas.clientHeight) / (bodySize * 6))
    );

    viewport.zoom = desiredZoom;
    config.viewZoom = desiredZoom;

    viewport.offsetX = center.x - (canvas.clientWidth / viewport.zoom / 2);
    viewport.offsetY = center.y - (canvas.clientHeight / viewport.zoom / 2);

    const maxPanX = Math.max(0, config.WORLD_WIDTH - (canvas.clientWidth / viewport.zoom));
    const maxPanY = Math.max(0, config.WORLD_HEIGHT - (canvas.clientHeight / viewport.zoom));
    viewport.offsetX = Math.max(0, Math.min(viewport.offsetX, maxPanX));
    viewport.offsetY = Math.max(0, Math.min(viewport.offsetY, maxPanY));

    config.viewOffsetX = viewport.offsetX;
    config.viewOffsetY = viewport.offsetY;

    followInfoRefreshCounter++;
    if (followInfoRefreshCounter >= 20) {
        updateInfoPanel();
        followInfoRefreshCounter = 0;
    }
}

// --- Physics Update ---
function updatePhysics(dt) {
    if (config.IS_SIMULATION_PAUSED) {
        // requestAnimationFrame(gameLoop); // gameLoop call handled in main.js
        return;
    }

    updateSpatialGrid();

    applyVelocityEmitters();

    // Particle Emission Logic with Floor and Ceiling
    if (particles.length < config.PARTICLE_POPULATION_FLOOR) {
        let particlesToSpawnToFloor = config.PARTICLE_POPULATION_FLOOR - particles.length;
        for (let i = 0; i < particlesToSpawnToFloor; i++) {
            if (particles.length < config.PARTICLE_POPULATION_CEILING) { // Double check ceiling
                particles.push(new Particle(Math.random() * config.WORLD_WIDTH, Math.random() * config.WORLD_HEIGHT, fluidField));
            } else {
                break;
            }
        }
        config.particleEmissionDebt = 0; // Reset debt as we've just topped up
    } else if (particles.length < config.PARTICLE_POPULATION_CEILING && config.PARTICLES_PER_SECOND > 0 && fluidField) {
        config.particleEmissionDebt += config.PARTICLES_PER_SECOND * dt;
        while (config.particleEmissionDebt >= 1 && particles.length < config.PARTICLE_POPULATION_CEILING) {
            particles.push(new Particle(Math.random() * config.WORLD_WIDTH, Math.random() * config.WORLD_HEIGHT, fluidField));
            config.particleEmissionDebt -= 1;
        }
    } // If particles.length >= PARTICLE_POPULATION_CEILING, do nothing for rate-based emission


    if (config.selectedSoftBodyPoint && config.selectedSoftBodyPoint.point.isFixed && fluidField) {
        const activeBody = config.selectedSoftBodyPoint.body;
        const point = config.selectedSoftBodyPoint.point;
        const displacementX = point.pos.x - point.prevPos.x;
        const displacementY = point.pos.y - point.prevPos.y;
        const movementMagnitudeSq = displacementX * displacementX + displacementY * displacementY;
        const movementThresholdSq = 0.01 * 0.01;

        if (movementMagnitudeSq > movementThresholdSq) {
            const fluidGridX = Math.floor(point.pos.x / fluidField.scaleX);
            const fluidGridY = Math.floor(point.pos.y / fluidField.scaleY);

            fluidField.addVelocity(fluidGridX, fluidGridY,
                displacementX * config.SOFT_BODY_PUSH_STRENGTH / fluidField.scaleX, // Scale to grid velocity
                displacementY * config.SOFT_BODY_PUSH_STRENGTH / fluidField.scaleY);
            fluidField.addDensity(fluidGridX, fluidGridY, 60, 60, 80, 15);
        }
    }

    if (fluidField) {
        fluidField.dt = dt;
        fluidField.step();
    }

    let canCreaturesReproduceGlobally = softBodyPopulation.length < config.CREATURE_POPULATION_CEILING;

    let currentAnyUnstable = false;
    let newOffspring = [];

    for (let i = softBodyPopulation.length - 1; i >= 0; i--) {
        const body = softBodyPopulation[i];
        if (!body.isUnstable) {
            body.updateSelf(dt, fluidField);
            if (body.isUnstable) {
                currentAnyUnstable = true;
            } else if (body.creatureEnergy >= body.reproductionEnergyThreshold &&
                body.canReproduce &&
                canCreaturesReproduceGlobally &&
                body.failedReproductionCooldown <= 0) {
                newOffspring.push(...body.reproduce());
            }
        }
    }
    softBodyPopulation.push(...newOffspring);

    // Update and remove dead particles
    for (let i = particles.length - 1; i >= 0; i--) {
        particles[i].update(dt);
        if (particles[i].life <= 0 && !particles[i].isEaten) {
            particles.splice(i, 1);
        } else if (particles[i].isEaten && particles[i].life <= 0) {
            particles.splice(i, 1);
        }
    }

    if (currentAnyUnstable && !config.isAnySoftBodyUnstable) {
        config.isAnySoftBodyUnstable = true;
    } else if (!currentAnyUnstable && config.isAnySoftBodyUnstable && !softBodyPopulation.some(b => b.isUnstable)) {
        config.isAnySoftBodyUnstable = false;
    }
    updateInstabilityIndicator();

    let removedCount = 0;
    for (let i = softBodyPopulation.length - 1; i >= 0; i--) {
        if (softBodyPopulation[i].isUnstable) {
            const body = softBodyPopulation[i];
            globalEnergyGains.photosynthesis += body.energyGainedFromPhotosynthesis;
            globalEnergyGains.eating += body.energyGainedFromEating;
            globalEnergyGains.predation += body.energyGainedFromPredation;

            // Accumulate global costs from the dying body
            globalEnergyCosts.baseNodes += body.energyCostFromBaseNodes;
            globalEnergyCosts.emitterNodes += body.energyCostFromEmitterNodes;
            globalEnergyCosts.eaterNodes += body.energyCostFromEaterNodes;
            globalEnergyCosts.predatorNodes += body.energyCostFromPredatorNodes;
            globalEnergyCosts.neuronNodes += body.energyCostFromNeuronNodes;
            globalEnergyCosts.swimmerNodes += body.energyCostFromSwimmerNodes;
            globalEnergyCosts.photosyntheticNodes += body.energyCostFromPhotosyntheticNodes;
            globalEnergyCosts.grabbingNodes += body.energyCostFromGrabbingNodes;
            globalEnergyCosts.eyeNodes += body.energyCostFromEyeNodes;
            globalEnergyCosts.jetNodes += body.energyCostFromJetNodes;
            globalEnergyCosts.attractorNodes += body.energyCostFromAttractorNodes;
            globalEnergyCosts.repulsorNodes += body.energyCostFromRepulsorNodes;

            softBodyPopulation.splice(i, 1);
            removedCount++;
        }
    }

    // Creature population floor maintenance
    const neededToMaintainFloor = config.CREATURE_POPULATION_FLOOR - softBodyPopulation.length;
    if (neededToMaintainFloor > 0) {
        for (let i = 0; i < neededToMaintainFloor; i++) {
            if (softBodyPopulation.length < config.CREATURE_POPULATION_CEILING) { // Also respect ceiling when topping up
                const margin = 50;
                const randX = margin + Math.random() * (config.WORLD_WIDTH - margin * 2);
                const randY = margin + Math.random() * (config.WORLD_HEIGHT - margin * 2);
                const softBody = new SoftBody(nextSoftBodyId++, randX, randY, null);
                softBody.setNutrientField(nutrientField);
                softBody.setLightField(lightField);
                softBody.setParticles(particles);
                softBody.setSpatialGrid(spatialGrid);
                softBodyPopulation.push(softBody);
            } else {
                break; // Stop if ceiling is reached during floor maintenance
            }
        }
    }

    updateAutoFollowCamera();
    updatePopulationCount();

    // draw(); // REMOVE draw() call from here
    // requestAnimationFrame(gameLoop); // gameLoop call handled in main.js
}

// --- Drawing --- (draw() function is now standalone)
export function draw() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();

    ctx.scale(viewport.zoom, viewport.zoom);
    ctx.translate(-viewport.offsetX, -viewport.offsetY);

    if (fluidField) {
        // Reverted: Direct blocky drawing
        const viewportWorldWidth = canvas.width / viewport.zoom;
        const viewportWorldHeight = canvas.height / viewport.zoom;
        fluidField.draw(ctx, canvas.width, canvas.height, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }
    if (config.SHOW_NUTRIENT_MAP && nutrientField && fluidField) {
        drawNutrientMap(ctx, canvas.width, canvas.height, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }
    if (config.SHOW_LIGHT_MAP && lightField && fluidField) {
        drawLightMap(ctx, canvas.width, canvas.height, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }
    if (config.SHOW_VISCOSITY_MAP && viscosityField && fluidField) {
        drawViscosityMap(ctx, canvas.width, canvas.height, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }

    //console.log("[Debug] In draw() function, about to check SHOW_FLUID_VELOCITY. Value:", config.SHOW_FLUID_VELOCITY);
    if (config.SHOW_FLUID_VELOCITY && fluidField) {
        drawFluidVelocities(ctx, fluidField, canvas.width, canvas.height, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }

    for (let particle of particles) {
        // Culling check (particles are small, so checking their center point + radius is good)
        const viewRightWorld = viewport.offsetX + canvas.width / viewport.zoom;
        const viewBottomWorld = viewport.offsetY + canvas.height / viewport.zoom;
        const particleRadius = particle.size; // Assuming particle.size is its radius

        if (particle.pos.x + particleRadius < viewport.offsetX || particle.pos.x - particleRadius > viewRightWorld ||
            particle.pos.y + particleRadius < viewport.offsetY || particle.pos.y - particleRadius > viewBottomWorld) {
            continue; // Skip drawing if particle is outside viewport
        }
        particle.draw(ctx);
    }
    for (let body of softBodyPopulation) {
        // Culling check for soft bodies
        const bbox = body.getBoundingBox(); // { minX, minY, maxX, maxY } in world coords
        const viewRightWorld = viewport.offsetX + canvas.width / viewport.zoom;
        const viewBottomWorld = viewport.offsetY + canvas.height / viewport.zoom;

        if (bbox.maxX < viewport.offsetX || bbox.minX > viewRightWorld ||
            bbox.maxY < viewport.offsetY || bbox.minY > viewBottomWorld) {
            continue; // Skip drawing if body is outside viewport
        }
        body.drawSelf(ctx);
    }

    if (config.IS_EMITTER_EDIT_MODE && currentEmitterPreview && fluidField) {
        ctx.beginPath();
        ctx.moveTo(currentEmitterPreview.startX, currentEmitterPreview.startY);
        ctx.lineTo(currentEmitterPreview.endX, currentEmitterPreview.endY);
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.7)';
        ctx.lineWidth = 2 / viewport.zoom;
        ctx.stroke();
        const angle = Math.atan2(currentEmitterPreview.endY - currentEmitterPreview.startY, currentEmitterPreview.endX - currentEmitterPreview.startX);
        const arrowSize = 10 / viewport.zoom;
        ctx.lineTo(currentEmitterPreview.endX - arrowSize * Math.cos(angle - Math.PI / 6), currentEmitterPreview.endY - arrowSize * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(currentEmitterPreview.endX, currentEmitterPreview.endY);
        ctx.lineTo(currentEmitterPreview.endX - arrowSize * Math.cos(angle + Math.PI / 6), currentEmitterPreview.endY - arrowSize * Math.sin(angle + Math.PI / 6));
        ctx.stroke();
    }

    if (fluidField && config.velocityEmitters.length > 0) {
        for (const emitter of config.velocityEmitters) {
            const startX = (emitter.gridX + 0.5) * fluidField.scaleX;
            const startY = (emitter.gridY + 0.5) * fluidField.scaleY;

            const forceMagnitude = Math.sqrt(emitter.forceX ** 2 + emitter.forceY ** 2);
            const displayLength = 20 * config.EMITTER_STRENGTH;
            const endX = startX + (emitter.forceX / (forceMagnitude || 1)) * displayLength;
            const endY = startY + (emitter.forceY / (forceMagnitude || 1)) * displayLength;


            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(endX, endY);
            ctx.strokeStyle = 'rgba(0, 200, 255, 0.5)';
            ctx.lineWidth = (1 + Math.min(5, forceMagnitude * config.EMITTER_STRENGTH * 0.5)) / viewport.zoom;
            ctx.stroke();

            const angle = Math.atan2(endY - startY, endX - startX);
            const arrowSize = 8 / viewport.zoom;
            if (Math.abs(endX - startX) > 0.01 || Math.abs(endY - startY) > 0.01) {
                ctx.lineTo(endX - arrowSize * Math.cos(angle - Math.PI / 6), endY - arrowSize * Math.sin(angle - Math.PI / 6));
                ctx.moveTo(endX, endY);
                ctx.lineTo(endX - arrowSize * Math.cos(angle + Math.PI / 6), endY - arrowSize * Math.sin(angle + Math.PI / 6));
                ctx.stroke();
            }
        }
    }

    if (config.selectedInspectBody && config.selectedInspectPoint) {
        updateInfoPanel(); // This function is in ui.js, ensure it's accessible
    }


    ctx.restore();
}

function drawFluidVelocities(ctx, fluidData, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
    if (!fluidData || !fluidData.Vx || !fluidData.Vy) return;
    console.log("[Debug] drawFluidVelocities called. SHOW_FLUID_VELOCITY:", config.SHOW_FLUID_VELOCITY);

    const N = Math.round(fluidData.size);
    if (N <= 0) {
        console.log("[Debug] Fluid grid size (N) is zero or negative.");
        return;
    }

    const worldCellWidth = config.WORLD_WIDTH / N; // Cell dimensions in world units
    const worldCellHeight = config.WORLD_HEIGHT / N;

    // Determine visible grid range
    const viewLeftWorld = viewOffsetXWorld;
    const viewTopWorld = viewOffsetYWorld;
    const viewRightWorld = viewOffsetXWorld + viewportCanvasWidth / currentZoom;
    const viewBottomWorld = viewOffsetYWorld + viewportCanvasHeight / currentZoom;

    const startCol = Math.max(0, Math.floor(viewLeftWorld / worldCellWidth));
    const endCol = Math.min(N - 1, Math.floor(viewRightWorld / worldCellWidth));
    const startRow = Math.max(0, Math.floor(viewTopWorld / worldCellHeight));
    const endRow = Math.min(N - 1, Math.floor(viewBottomWorld / worldCellHeight));

    // Log a sample of fluid velocities
    if (N > 0 && fluidData.Vx.length > 0 && (startCol <= endCol && startRow <= endRow)) {
        let sampleIndices = [];
        // Ensure we only sample valid indices within the visible range if possible
        for (let k = 0; k < Math.min(5, (endCol - startCol + 1) * (endRow - startRow + 1)); k++) {
            const randCol = startCol + Math.floor(Math.random() * (endCol - startCol + 1));
            const randRow = startRow + Math.floor(Math.random() * (endRow - startRow + 1));
            sampleIndices.push(fluidData.IX(randCol, randRow));
        }
        let sampleVelocities = sampleIndices.map(idx => `Vx[${idx}]: ${fluidData.Vx[idx]?.toFixed(3)}, Vy[${idx}]: ${fluidData.Vy[idx]?.toFixed(3)}`);
        console.log("[Debug] Sample fluid velocities (visible range):", sampleVelocities.join('; '));
        // This check for all zero might be too expensive if done every time on the full arrays.
        // Consider removing or restricting it if performance is an issue.
        // if (fluidData.Vx.every(v => Math.abs(v) < 0.001) && fluidData.Vy.every(v => Math.abs(v) < 0.001)) {
        //     console.log("[Debug] All fluid velocities are very close to zero.");
        // }
    }

    const arrowLengthScale = 0.8 * Math.min(worldCellWidth, worldCellHeight); // Adjusted for better visuals
    const maxVelocityDisplay = 5;

    ctx.strokeStyle = 'rgba(200, 200, 255, 0.6)';
    ctx.lineWidth = Math.max(0.5, 1.0 / currentZoom);

    for (let j = startRow; j <= endRow; j++) {
        for (let i = startCol; i <= endCol; i++) {
            const idx = fluidData.IX(i, j);
            let vx = fluidData.Vx[idx];
            let vy = fluidData.Vy[idx];

            const mag = Math.sqrt(vx * vx + vy * vy);
            if (mag > maxVelocityDisplay) {
                vx = (vx / mag) * maxVelocityDisplay;
                vy = (vy / mag) * maxVelocityDisplay;
            }

            // Cell center in world coordinates
            const startXWorld = (i + 0.5) * worldCellWidth;
            const startYWorld = (j + 0.5) * worldCellHeight;
            const endXWorld = startXWorld + vx * arrowLengthScale;
            const endYWorld = startYWorld + vy * arrowLengthScale;

            if (Math.abs(startXWorld - endXWorld) < 0.1 && Math.abs(startYWorld - endYWorld) < 0.1) continue;

            ctx.beginPath();
            ctx.moveTo(startXWorld, startYWorld); // Draw using world coordinates
            ctx.lineTo(endXWorld, endYWorld);   // The main canvas transform handles screen placement
            ctx.stroke();

            const angle = Math.atan2(endYWorld - startYWorld, endXWorld - startXWorld);
            const baseArrowHeadSize = 4;
            const arrowHeadScreenMinSize = 1.5;
            const arrowHeadScreenMaxSize = 8;
            let arrowHeadSize = baseArrowHeadSize / currentZoom;
            arrowHeadSize = Math.max(arrowHeadScreenMinSize, Math.min(arrowHeadSize, arrowHeadScreenMaxSize));

            ctx.beginPath();
            ctx.moveTo(endXWorld, endYWorld);
            ctx.lineTo(endXWorld - arrowHeadSize * Math.cos(angle - Math.PI / 6), endYWorld - arrowHeadSize * Math.sin(angle - Math.PI / 6));
            ctx.moveTo(endXWorld, endYWorld);
            ctx.lineTo(endXWorld - arrowHeadSize * Math.cos(angle + Math.PI / 6), endYWorld - arrowHeadSize * Math.sin(angle + Math.PI / 6));
            ctx.stroke();
        }
    }
}

function initNutrientMap() {
    const size = Math.round(config.FLUID_GRID_SIZE_CONTROL);
    if (size <= 0 || !Number.isFinite(size)) {
        console.error("Invalid size for nutrient map:", size);
        nutrientField = new Float32Array(0); // Empty array if size is invalid
        return;
    }
    // nutrientField = new Float32Array(size * size).fill(1.0); // Old: Default to 1.0 (neutral)
    nutrientField = new Float32Array(size * size);
    const noiseScale = 0.05; // Same scale as light map for consistency, can be different
    const noiseOffsetX = Math.random() * 1000 + 1000; // Different random offset seed
    const noiseOffsetY = Math.random() * 1000 + 1000; // Different random offset seed

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            let noiseValue = perlin.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
            // Map Perlin noise (-1 to 1) to our desired nutrient range (MIN_NUTRIENT_VALUE to MAX_NUTRIENT_VALUE)
            let mappedValue = ((noiseValue + 1) / 2) * (config.MAX_NUTRIENT_VALUE - config.MIN_NUTRIENT_VALUE) + config.MIN_NUTRIENT_VALUE;
            nutrientField[y * size + x] = Math.max(config.MIN_NUTRIENT_VALUE, Math.min(config.MAX_NUTRIENT_VALUE, mappedValue));
        }
    }
    console.log(`Nutrient map initialized to ${size}x${size} with Perlin noise pattern.`);
}

function initLightMap() {
    const size = Math.round(config.FLUID_GRID_SIZE_CONTROL);
    if (size <= 0 || !Number.isFinite(size)) {
        console.error("Invalid size for light map:", size);
        lightField = new Float32Array(0);
        return;
    }
    lightField = new Float32Array(size * size);
    const noiseScale = 0.05; // Adjust for patchiness; smaller = larger patches
    const noiseOffsetX = Math.random() * 1000;
    const noiseOffsetY = Math.random() * 1000;

    for (let y_coord = 0; y_coord < size; y_coord++) {
        for (let x = 0; x < size; x++) {
            let noiseValue = perlin.noise(x * noiseScale + noiseOffsetX, y_coord * noiseScale + noiseOffsetY);
            noiseValue = (noiseValue + 1) / 2; // Map to 0-1 range
            lightField[y_coord * size + x] = Math.max(config.MIN_LIGHT_VALUE, Math.min(config.MAX_LIGHT_VALUE, noiseValue));
        }
    }
    console.log(`Light map initialized to ${size}x${size} with Perlin noise pattern.`);
}

function initViscosityMap() {
    const size = Math.round(config.FLUID_GRID_SIZE_CONTROL);
    if (size <= 0 || !Number.isFinite(size)) {
        console.error("Invalid size for viscosity map:", size);
        viscosityField = new Float32Array(0);
        return;
    }
    viscosityField = new Float32Array(size * size);
    const noiseScale = 0.06; // Slightly different scale for variety, or keep same as others (0.05)
    const noiseOffsetX = Math.random() * 1000 + 2000; // Different random offset seed
    const noiseOffsetY = Math.random() * 1000 + 2000; // Different random offset seed

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            let noiseValue = perlin.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
            // Map Perlin noise (-1 to 1) to our desired viscosity range 
            let mappedValue = ((noiseValue + 1) / 2) * (config.MAX_VISCOSITY_MULTIPLIER - config.MIN_VISCOSITY_MULTIPLIER) + config.MIN_VISCOSITY_MULTIPLIER;
            viscosityField[y * size + x] = Math.max(config.MIN_VISCOSITY_MULTIPLIER, Math.min(config.MAX_VISCOSITY_MULTIPLIER, mappedValue));
        }
    }
    console.log(`Viscosity map initialized to ${size}x${size} with Perlin noise pattern.`);
}

export {
    initializeSpatialGrid,
    initializePopulation,
    updatePhysics,
    initFluidSimulation,
    initParticles,
    initNutrientMap,
    mutationStats,
    initLightMap,
    initViscosityMap,
    nutrientField,
    lightField,
    viscosityField,
    softBodyPopulation,
    fluidField,
    nextSoftBodyId,
    globalEnergyCosts,
    globalEnergyGains,
    particles,
    spatialGrid
};
