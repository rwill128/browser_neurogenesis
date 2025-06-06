let spatialGrid = [];
let softBodyPopulation = [];
let fluidField = null;
let particles = [];
let nextSoftBodyId = 0;
const restitution = 0.4; // Moved from global constants as it's physics-specific

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
    jetNodes: 0
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
    blueprintNeuronHiddenSizeChange: 0
};

function initializeSpatialGrid() {
    GRID_COLS = Math.max(1, Math.ceil(WORLD_WIDTH / GRID_CELL_SIZE));
    GRID_ROWS = Math.max(1, Math.ceil(WORLD_HEIGHT / GRID_CELL_SIZE));
    spatialGrid = new Array(GRID_COLS * GRID_ROWS);
    for (let i = 0; i < spatialGrid.length; i++) {
        spatialGrid[i] = [];
    }
}


function populateSpatialGrid() {
    for (let i = 0; i < spatialGrid.length; i++) {
        spatialGrid[i] = [];
    }
    // Add soft body mass points to the grid
    softBodyPopulation.forEach(body => {
        if (body.isUnstable) return;
        body.massPoints.forEach(point => {
            if (isNaN(point.pos.x) || isNaN(point.pos.y) || !isFinite(point.pos.x) || !isFinite(point.pos.y)) {
                return;
            }
            const gx = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(point.pos.x / GRID_CELL_SIZE)));
            const gy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(point.pos.y / GRID_CELL_SIZE)));

            const gridIndex = gx + gy * GRID_COLS;
            if(spatialGrid[gridIndex] && Number.isFinite(gridIndex) && gridIndex < spatialGrid.length) {
                spatialGrid[gridIndex].push({ type: 'softbody_point', pointRef: point, bodyRef: body });
            }
        });
    });
    // Add particles to the grid
    particles.forEach(particle => {
        if (particle.life <= 0) return; // Don't add dead/eaten particles
         if (isNaN(particle.pos.x) || isNaN(particle.pos.y) || !isFinite(particle.pos.x) || !isFinite(particle.pos.y)) {
            return;
        }
        const gx = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(particle.pos.x / GRID_CELL_SIZE)));
        const gy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(particle.pos.y / GRID_CELL_SIZE)));
        const gridIndex = gx + gy * GRID_COLS;
         if(spatialGrid[gridIndex] && Number.isFinite(gridIndex) && gridIndex < spatialGrid.length) {
            spatialGrid[gridIndex].push({ type: 'particle', particleRef: particle });
        }
    });
}

// --- Simulation Setup ---
function initializePopulation() {
    softBodyPopulation = [];
    isAnySoftBodyUnstable = false;
    updateInstabilityIndicator();
    nextSoftBodyId = 0;

    for (let i = 0; i < CREATURE_POPULATION_FLOOR; i++) { // Use floor for initial pop
        const margin = 50;
        const randX = margin + Math.random() * (WORLD_WIDTH - margin * 2);
        const randY = margin + Math.random() * (WORLD_HEIGHT - margin * 2);
        softBodyPopulation.push(new SoftBody(nextSoftBodyId++, randX, randY, null));
    }
    // lastTime = performance.now(); // This will be handled in main.js
    updatePopulationCount();
}


async function initFluidSimulation(targetCanvas) {
    const dt_simulation = 1/60; // Assuming fixed timestep for simulation physics if needed separately
    const scaleX = WORLD_WIDTH / FLUID_GRID_SIZE_CONTROL;
    const scaleY = WORLD_HEIGHT / FLUID_GRID_SIZE_CONTROL;

    if (USE_GPU_FLUID_SIMULATION) {
        console.log("Attempting to initialize GPUFluidField...");
        // console.log("Canvas element being passed to GPUFluidField:", targetCanvas); 
        if (!targetCanvas || typeof targetCanvas.getContext !== 'function') {
            console.error("CRITICAL: Target canvas for GPUFluidField is invalid!");
            console.warn("Falling back to CPU FluidField due to invalid target canvas.");
            fluidField = new FluidField(FLUID_GRID_SIZE_CONTROL, FLUID_DIFFUSION, FLUID_VISCOSITY, dt_simulation, scaleX, scaleY);
        } else {
            fluidField = new GPUFluidField(targetCanvas, FLUID_GRID_SIZE_CONTROL, FLUID_DIFFUSION, FLUID_VISCOSITY, dt_simulation, scaleX, scaleY);
            await fluidField._initPromise; 
            
            // console.log("GPUFluidField instance created and awaited in simulation.js.");

            // Check if GPUFluidField successfully initialized in WebGPU mode (fluidField.device would be set)
            // or if it fell back to WebGL (fluidField.gl would be set and fluidField.device would be null)
            // or if both failed (gpuEnabled would be false).
            if (!fluidField.gpuEnabled || (!fluidField.device && !fluidField.gl)) { 
                console.warn("Fallback to CPU: GPUFluidField initialization failed (neither WebGPU nor WebGL succeeded), or gpuEnabled is false.");
                fluidField = new FluidField(FLUID_GRID_SIZE_CONTROL, FLUID_DIFFUSION, FLUID_VISCOSITY, dt_simulation, scaleX, scaleY);
            } else if (fluidField.device) {
                console.log("GPUFluidField successfully initialized with WebGPU in simulation.js.");
            } else if (fluidField.gl) {
                console.log("GPUFluidField successfully initialized with WebGL fallback in simulation.js.");
            }
        }
    } else {
        console.log("Initializing CPU FluidField...");
        fluidField = new FluidField(FLUID_GRID_SIZE_CONTROL, FLUID_DIFFUSION, FLUID_VISCOSITY, dt_simulation, scaleX, scaleY);
    }

    // Initialize offscreen canvas for CPU fluid rendering if not using GPU, or if GPU failed completely
    if (!fluidField || (fluidField instanceof FluidField && (!offscreenFluidCanvas || offscreenFluidCanvas.width !== Math.round(FLUID_GRID_SIZE_CONTROL) || offscreenFluidCanvas.height !== Math.round(FLUID_GRID_SIZE_CONTROL)))) {
        if (!(fluidField instanceof GPUFluidField) || !fluidField.gpuEnabled) { // Only if truly CPU or GPU totally failed
            offscreenFluidCanvas = document.createElement('canvas');
            offscreenFluidCanvas.width = Math.round(FLUID_GRID_SIZE_CONTROL);
            offscreenFluidCanvas.height = Math.round(FLUID_GRID_SIZE_CONTROL);
            offscreenFluidCtx = offscreenFluidCanvas.getContext('2d', { willReadFrequently: true });
            console.log("Offscreen canvas for CPU fluid rendering initialized or resized.");
        }
    } else if (fluidField instanceof GPUFluidField && fluidField.gpuEnabled) {
        // If GPU is active, we might not need the CPU offscreen canvas, or it serves a different purpose.
        // For now, let's nullify them if WebGPU/WebGL is active for fluid.
        offscreenFluidCanvas = null;
        offscreenFluidCtx = null;
        // console.log("GPU fluid active, CPU offscreen canvas resources released/nulled.");
    }

    fluidField.useWrapping = IS_WORLD_WRAPPING;
    fluidField.maxVelComponent = MAX_FLUID_VELOCITY_COMPONENT; 
    velocityEmitters = []; // Clear any existing emitters when re-initializing
}

function initParticles() {
    particles = [];
    particleEmissionDebt = 0;
}

function applyVelocityEmitters() {
    if (!fluidField || EMITTER_STRENGTH <= 0) return;
    for (const emitter of velocityEmitters) {
        fluidField.addVelocity(emitter.gridX, emitter.gridY,
                               emitter.forceX * EMITTER_STRENGTH,
                               emitter.forceY * EMITTER_STRENGTH);
    }
}


// --- Physics Update ---
function updatePhysics(dt) {
    if (IS_SIMULATION_PAUSED) {
        // requestAnimationFrame(gameLoop); // gameLoop call handled in main.js
        return;
    }

    populateSpatialGrid();

    applyVelocityEmitters();

    // Particle Emission Logic with Floor and Ceiling
    if (particles.length < PARTICLE_POPULATION_FLOOR) {
        let particlesToSpawnToFloor = PARTICLE_POPULATION_FLOOR - particles.length;
        for (let i = 0; i < particlesToSpawnToFloor; i++) {
            if (particles.length < PARTICLE_POPULATION_CEILING) { // Double check ceiling
                 particles.push(new Particle(Math.random() * WORLD_WIDTH, Math.random() * WORLD_HEIGHT, fluidField));
            } else {
                break;
            }
        }
        particleEmissionDebt = 0; // Reset debt as we've just topped up
    } else if (particles.length < PARTICLE_POPULATION_CEILING && PARTICLES_PER_SECOND > 0 && fluidField) {
        particleEmissionDebt += PARTICLES_PER_SECOND * dt;
        while (particleEmissionDebt >= 1 && particles.length < PARTICLE_POPULATION_CEILING) {
            particles.push(new Particle(Math.random() * WORLD_WIDTH, Math.random() * WORLD_HEIGHT, fluidField));
            particleEmissionDebt -= 1;
        }
    } // If particles.length >= PARTICLE_POPULATION_CEILING, do nothing for rate-based emission


    if (selectedSoftBodyPoint && selectedSoftBodyPoint.point.isFixed && fluidField) {
        const activeBody = selectedSoftBodyPoint.body;
        const point = selectedSoftBodyPoint.point;
        const displacementX = point.pos.x - point.prevPos.x;
        const displacementY = point.pos.y - point.prevPos.y;
        const movementMagnitudeSq = displacementX*displacementX + displacementY*displacementY;
        const movementThresholdSq = 0.01 * 0.01;

        if (movementMagnitudeSq > movementThresholdSq) {
            const fluidGridX = Math.floor(point.pos.x / fluidField.scaleX);
            const fluidGridY = Math.floor(point.pos.y / fluidField.scaleY);

            fluidField.addVelocity(fluidGridX, fluidGridY,
                                   displacementX * SOFT_BODY_PUSH_STRENGTH / fluidField.scaleX, // Scale to grid velocity
                                   displacementY * SOFT_BODY_PUSH_STRENGTH / fluidField.scaleY);
            fluidField.addDensity(fluidGridX, fluidGridY, 60, 60, 80, 15);
        }
    }

    if (fluidField) {
        fluidField.dt = dt;
        fluidField.step();
    }

    canCreaturesReproduceGlobally = softBodyPopulation.length < CREATURE_POPULATION_CEILING;

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
        } else if (particles[i].isEaten && particles[i].life <=0) {
            particles.splice(i,1);
        }
    }

    if(currentAnyUnstable && !isAnySoftBodyUnstable) {
        isAnySoftBodyUnstable = true;
    } else if (!currentAnyUnstable && isAnySoftBodyUnstable && !softBodyPopulation.some(b => b.isUnstable)) {
        isAnySoftBodyUnstable = false;
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

            softBodyPopulation.splice(i, 1);
            removedCount++;
        }
    }

    // Creature population floor maintenance
    const neededToMaintainFloor = CREATURE_POPULATION_FLOOR - softBodyPopulation.length;
    if (neededToMaintainFloor > 0) {
         for (let i = 0; i < neededToMaintainFloor; i++) {
             if (softBodyPopulation.length < CREATURE_POPULATION_CEILING) { // Also respect ceiling when topping up
                 const margin = 50;
                 const randX = margin + Math.random() * (WORLD_WIDTH - margin * 2);
                 const randY = margin + Math.random() * (WORLD_HEIGHT - margin * 2);
                 softBodyPopulation.push(new SoftBody(nextSoftBodyId++, randX, randY, null));
             } else {
                 break; // Stop if ceiling is reached during floor maintenance
             }
         }
    }

    updatePopulationCount();

    // draw(); // REMOVE draw() call from here
    // requestAnimationFrame(gameLoop); // gameLoop call handled in main.js
}

// --- Drawing --- (draw() function is now standalone)
function draw() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();

    ctx.scale(viewZoom, viewZoom);
    ctx.translate(-viewOffsetX, -viewOffsetY);

    if (fluidField) {
        // Reverted: Direct blocky drawing
        const viewportWorldWidth = canvas.width / viewZoom;
        const viewportWorldHeight = canvas.height / viewZoom;
        fluidField.draw(ctx, canvas.width, canvas.height, viewOffsetX, viewOffsetY, viewZoom);
    }
    if (SHOW_NUTRIENT_MAP && nutrientField && fluidField) {
        drawNutrientMap(ctx, canvas.width, canvas.height, viewOffsetX, viewOffsetY, viewZoom);
    }
    if (SHOW_LIGHT_MAP && lightField && fluidField) {
        drawLightMap(ctx, canvas.width, canvas.height, viewOffsetX, viewOffsetY, viewZoom);
    }
    if (SHOW_VISCOSITY_MAP && viscosityField && fluidField) {
        drawViscosityMap(ctx, canvas.width, canvas.height, viewOffsetX, viewOffsetY, viewZoom);
    }

    //console.log("[Debug] In draw() function, about to check SHOW_FLUID_VELOCITY. Value:", SHOW_FLUID_VELOCITY);
    if (SHOW_FLUID_VELOCITY && fluidField) {
        drawFluidVelocities(ctx, fluidField, canvas.width, canvas.height, viewOffsetX, viewOffsetY, viewZoom);
    }

    for (let particle of particles) {
        // Culling check (particles are small, so checking their center point + radius is good)
        const viewRightWorld = viewOffsetX + canvas.width / viewZoom;
        const viewBottomWorld = viewOffsetY + canvas.height / viewZoom;
        const particleRadius = particle.size; // Assuming particle.size is its radius

        if (particle.pos.x + particleRadius < viewOffsetX || particle.pos.x - particleRadius > viewRightWorld ||
            particle.pos.y + particleRadius < viewOffsetY || particle.pos.y - particleRadius > viewBottomWorld) {
            continue; // Skip drawing if particle is outside viewport
        }
        particle.draw(ctx);
    }
    for (let body of softBodyPopulation) {
        // Culling check for soft bodies
        const bbox = body.getBoundingBox(); // { minX, minY, maxX, maxY } in world coords
        const viewRightWorld = viewOffsetX + canvas.width / viewZoom;
        const viewBottomWorld = viewOffsetY + canvas.height / viewZoom;

        if (bbox.maxX < viewOffsetX || bbox.minX > viewRightWorld ||
            bbox.maxY < viewOffsetY || bbox.minY > viewBottomWorld) {
            continue; // Skip drawing if body is outside viewport
        }
        body.drawSelf(ctx);
    }

    if (IS_EMITTER_EDIT_MODE && currentEmitterPreview && fluidField) {
        ctx.beginPath();
        ctx.moveTo(currentEmitterPreview.startX, currentEmitterPreview.startY);
        ctx.lineTo(currentEmitterPreview.endX, currentEmitterPreview.endY);
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.7)';
        ctx.lineWidth = 2 / viewZoom;
        ctx.stroke();
        const angle = Math.atan2(currentEmitterPreview.endY - currentEmitterPreview.startY, currentEmitterPreview.endX - currentEmitterPreview.startX);
        const arrowSize = 10 / viewZoom;
        ctx.lineTo(currentEmitterPreview.endX - arrowSize * Math.cos(angle - Math.PI / 6), currentEmitterPreview.endY - arrowSize * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(currentEmitterPreview.endX, currentEmitterPreview.endY);
        ctx.lineTo(currentEmitterPreview.endX - arrowSize * Math.cos(angle + Math.PI / 6), currentEmitterPreview.endY - arrowSize * Math.sin(angle + Math.PI / 6));
        ctx.stroke();
    }

    if (fluidField && velocityEmitters.length > 0) {
         for (const emitter of velocityEmitters) {
            const startX = (emitter.gridX + 0.5) * fluidField.scaleX;
            const startY = (emitter.gridY + 0.5) * fluidField.scaleY;

            const forceMagnitude = Math.sqrt(emitter.forceX**2 + emitter.forceY**2);
            const displayLength = 20 * EMITTER_STRENGTH;
            const endX = startX + (emitter.forceX / (forceMagnitude || 1)) * displayLength ;
            const endY = startY + (emitter.forceY / (forceMagnitude || 1)) * displayLength ;


            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(endX, endY);
            ctx.strokeStyle = 'rgba(0, 200, 255, 0.5)';
            ctx.lineWidth = (1 + Math.min(5, forceMagnitude * EMITTER_STRENGTH * 0.5)) / viewZoom;
            ctx.stroke();

            const angle = Math.atan2(endY - startY, endX - startX);
            const arrowSize = 8 / viewZoom;
            if (Math.abs(endX-startX) > 0.01 || Math.abs(endY-startY) > 0.01) {
                ctx.lineTo(endX - arrowSize * Math.cos(angle - Math.PI / 6), endY - arrowSize * Math.sin(angle - Math.PI / 6));
                ctx.moveTo(endX, endY);
                ctx.lineTo(endX - arrowSize * Math.cos(angle + Math.PI / 6), endY - arrowSize * Math.sin(angle + Math.PI / 6));
                ctx.stroke();
            }
        }
    }

    if (selectedInspectBody && selectedInspectPoint) {
        updateInfoPanel(); // This function is in ui.js, ensure it's accessible
    }


    ctx.restore();
}

function drawFluidVelocities(ctx, fluidData, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
    if (!fluidData || !fluidData.Vx || !fluidData.Vy) return;
    console.log("[Debug] drawFluidVelocities called. SHOW_FLUID_VELOCITY:", SHOW_FLUID_VELOCITY);

    const N = Math.round(fluidData.size);
    if (N <= 0) {
        console.log("[Debug] Fluid grid size (N) is zero or negative.");
        return;
    }

    const worldCellWidth = WORLD_WIDTH / N; // Cell dimensions in world units
    const worldCellHeight = WORLD_HEIGHT / N;

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
        for(let k=0; k < Math.min(5, (endCol-startCol+1)*(endRow-startRow+1)); k++) { 
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