import config from './config.js';
import {SoftBody} from './classes/SoftBody.js';
import {Spring} from './classes/Spring.js';
import {Particle} from './classes/Particle.js';
import { stepWorld } from './engine/stepWorld.mjs';
import {FluidField} from './classes/FluidField.js';
import {GPUFluidField} from './gpuFluidField.js';
import {isWebGpuSupported} from './gpuUtils.js';
import {
    initializeSpatialGrid as initializeSharedSpatialGrid,
    initializePopulation as initializeSharedPopulation,
    initializeParticles as initializeSharedParticles,
    initializeNutrientMap as initializeSharedNutrientMap,
    initializeLightMap as initializeSharedLightMap,
    initializeViscosityMap as initializeSharedViscosityMap
} from './engine/initWorld.mjs';
import {
    canvas,
    ctx, updateInfoPanel,
    updateInstabilityIndicator,
    updatePopulationCount
} from "./ui.js";
import viewport from './viewport.js';
import { drawNutrientMap, drawLightMap, drawViscosityMap } from './environment.js';
import { syncRuntimeState } from './engine/runtimeState.js';
import { createWorldState } from './engine/worldState.mjs';
import { runBrowserStepAdapter } from './engine/browserStepAdapter.mjs';
import { createConfigViews } from './engine/configViews.mjs';
import { saveWorldStateSnapshot, loadWorldStateSnapshot } from './engine/worldPersistence.mjs';
import { resolveCanvasRenderMetrics } from './engine/cameraMath.mjs';

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
    shapeAddition: 0,
    growthGenomeMutations: 0
};

const simulationWorldState = createWorldState({
    spatialGrid,
    softBodyPopulation,
    fluidField,
    particles,
    nextSoftBodyId,
    nutrientField,
    lightField,
    viscosityField,
    mutationStats,
    globalEnergyGains,
    globalEnergyCosts
});
const simulationConfigViews = createConfigViews(config);

function syncModuleBindingsFromWorldState() {
    spatialGrid = simulationWorldState.spatialGrid;
    softBodyPopulation = simulationWorldState.softBodyPopulation;
    fluidField = simulationWorldState.fluidField;
    particles = simulationWorldState.particles;
    nextSoftBodyId = simulationWorldState.nextSoftBodyId;
    nutrientField = simulationWorldState.nutrientField;
    lightField = simulationWorldState.lightField;
    viscosityField = simulationWorldState.viscosityField;
    mutationStats = simulationWorldState.mutationStats;
    globalEnergyGains = simulationWorldState.globalEnergyGains;
    globalEnergyCosts = simulationWorldState.globalEnergyCosts;
}

syncRuntimeState({
    fluidField: simulationWorldState.fluidField,
    softBodyPopulation: simulationWorldState.softBodyPopulation,
    mutationStats: simulationWorldState.mutationStats
});
syncModuleBindingsFromWorldState();

function initializeSpatialGrid() {
    initializeSharedSpatialGrid(simulationWorldState, simulationConfigViews);
    syncModuleBindingsFromWorldState();
}


// --- Simulation Setup ---
function initializePopulation() {
    initializeSharedPopulation(simulationWorldState, {
        configViews: simulationConfigViews,
        config,
        SoftBodyClass: SoftBody,
        count: config.CREATURE_POPULATION_FLOOR,
        spawnMargin: 50,
        rng: Math.random
    });

    syncModuleBindingsFromWorldState();
    syncRuntimeState({ softBodyPopulation: simulationWorldState.softBodyPopulation });
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

    simulationWorldState.fluidField = fluidField;
    syncModuleBindingsFromWorldState();
    syncRuntimeState({ fluidField: simulationWorldState.fluidField });
}

function initParticles() {
    initializeSharedParticles(simulationWorldState, {
        configViews: simulationConfigViews,
        config,
        ParticleClass: null,
        count: 0,
        rng: Math.random
    });
    config.particleEmissionDebt = 0;
    syncModuleBindingsFromWorldState();
}

/**
 * Emit parse-friendly instability death lines to browser console for micro-run diagnostics.
 */
function logInstabilityDeathsToConsole(stepResult) {
    const deaths = Array.isArray(stepResult?.removedBodies) ? stepResult.removedBodies : [];
    if (!deaths.length) return;

    if (typeof window !== 'undefined') {
        if (!Array.isArray(window.__instabilityDeaths)) window.__instabilityDeaths = [];
        window.__instabilityDeaths.push(...deaths);
    }

    for (const death of deaths) {
        try {
            console.warn(`[UNSTABLE_DEATH] ${JSON.stringify(death)}`);
        } catch {
            console.warn('[UNSTABLE_DEATH]', death);
        }
    }
}

// --- Physics Update ---
function updatePhysics(dt) {
    if (config.IS_SIMULATION_PAUSED) {
        return;
    }

    const stepResult = runBrowserStepAdapter({
        worldState: simulationWorldState,
        dt,
        config,
        stepWorld,
        stepOptions: {
            configViews: simulationConfigViews,
            config,
            SoftBodyClass: SoftBody,
            ParticleClass: Particle,
            rng: Math.random,
            allowReproduction: true,
            maintainCreatureFloor: true,
            maintainParticleFloor: true,
            applyEmitters: true,
            applySelectedPointPush: true,
            creatureSpawnMargin: 50,
            captureInstabilityTelemetry: true,
            maxRecentInstabilityDeaths: 5000
        },
        viewport,
        canvas,
        updateInfoPanel,
        updateInstabilityIndicator,
        updatePopulationCount
    });

    logInstabilityDeathsToConsole(stepResult);
    syncModuleBindingsFromWorldState();
}


// --- Drawing --- (draw() function is now standalone)
export function draw() {
    // Keep camera math in CSS-pixel space while drawing into device-pixel backing buffers.
    const fallbackDpr = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1;
    const { cssWidth, cssHeight, dprX, dprY } = resolveCanvasRenderMetrics({
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        clientWidth: canvas.clientWidth,
        clientHeight: canvas.clientHeight,
        fallbackDpr
    });

    // Clear in backing-store pixels, then switch to CSS-pixel camera units.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.scale(dprX, dprY);
    ctx.scale(viewport.zoom, viewport.zoom);
    ctx.translate(-viewport.offsetX, -viewport.offsetY);

    if (fluidField) {
        // Draw in CSS-space viewport units; renderer transform handles world->screen mapping.
        fluidField.draw(ctx, cssWidth, cssHeight, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }
    if (config.SHOW_NUTRIENT_MAP && nutrientField && fluidField) {
        drawNutrientMap(ctx, cssWidth, cssHeight, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }
    if (config.SHOW_LIGHT_MAP && lightField && fluidField) {
        drawLightMap(ctx, cssWidth, cssHeight, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }
    if (config.SHOW_VISCOSITY_MAP && viscosityField && fluidField) {
        drawViscosityMap(ctx, cssWidth, cssHeight, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }

    //console.log("[Debug] In draw() function, about to check SHOW_FLUID_VELOCITY. Value:", config.SHOW_FLUID_VELOCITY);
    if (config.SHOW_FLUID_VELOCITY && fluidField) {
        drawFluidVelocities(ctx, fluidField, cssWidth, cssHeight, viewport.offsetX, viewport.offsetY, viewport.zoom);
    }

    for (let particle of particles) {
        // Culling check (particles are small, so checking their center point + radius is good)
        const viewRightWorld = viewport.offsetX + cssWidth / viewport.zoom;
        const viewBottomWorld = viewport.offsetY + cssHeight / viewport.zoom;
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
        const viewRightWorld = viewport.offsetX + cssWidth / viewport.zoom;
        const viewBottomWorld = viewport.offsetY + cssHeight / viewport.zoom;

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
    const field = initializeSharedNutrientMap(simulationWorldState, {
        configViews: simulationConfigViews,
        config,
        size,
        rng: Math.random
    });
    if (field.length === 0) {
        console.error("Invalid size for nutrient map:", size);
        return;
    }
    syncModuleBindingsFromWorldState();
    console.log(`Nutrient map initialized to ${size}x${size} with Perlin noise pattern.`);
}

function initLightMap() {
    const size = Math.round(config.FLUID_GRID_SIZE_CONTROL);
    const field = initializeSharedLightMap(simulationWorldState, {
        configViews: simulationConfigViews,
        config,
        size,
        rng: Math.random
    });
    if (field.length === 0) {
        console.error("Invalid size for light map:", size);
        return;
    }
    syncModuleBindingsFromWorldState();
    console.log(`Light map initialized to ${size}x${size} with Perlin noise pattern.`);
}

function initViscosityMap() {
    const size = Math.round(config.FLUID_GRID_SIZE_CONTROL);
    const field = initializeSharedViscosityMap(simulationWorldState, {
        configViews: simulationConfigViews,
        config,
        size,
        rng: Math.random
    });
    if (field.length === 0) {
        console.error("Invalid size for viscosity map:", size);
        return;
    }
    syncModuleBindingsFromWorldState();
    console.log(`Viscosity map initialized to ${size}x${size} with Perlin noise pattern.`);
}

/**
 * Capture a full real-path world snapshot.
 *
 * The payload is intentionally shared with node-harness so saves are portable
 * across browser and node runtimes.
 */
function saveCurrentWorldSnapshot(meta = {}) {
    return saveWorldStateSnapshot({
        worldState: simulationWorldState,
        configOrViews: simulationConfigViews,
        rng: Math.random,
        meta: {
            source: 'browser-real',
            totalSimulationTime: config.totalSimulationTime,
            ...meta
        }
    });
}

/**
 * Rehydrate the browser runtime from a previously saved world snapshot.
 */
function loadWorldFromSnapshot(snapshot) {
    const loadInfo = loadWorldStateSnapshot(snapshot, {
        worldState: simulationWorldState,
        configOrViews: simulationConfigViews,
        classes: {
            SoftBodyClass: SoftBody,
            ParticleClass: Particle,
            SpringClass: Spring,
            FluidFieldClass: FluidField
        },
        rng: Math.random
    });

    if (loadInfo?.meta && Number.isFinite(loadInfo.meta.totalSimulationTime)) {
        config.totalSimulationTime = Number(loadInfo.meta.totalSimulationTime);
    }

    syncModuleBindingsFromWorldState();
    syncRuntimeState({
        fluidField: simulationWorldState.fluidField,
        softBodyPopulation: simulationWorldState.softBodyPopulation,
        mutationStats: simulationWorldState.mutationStats
    });

    updateInstabilityIndicator();
    updatePopulationCount();
    updateInfoPanel();

    return loadInfo;
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
    spatialGrid,
    saveCurrentWorldSnapshot,
    loadWorldFromSnapshot
};
