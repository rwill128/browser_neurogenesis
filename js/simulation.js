let spatialGrid = [];
let softBodyPopulation = [];
let fluidField;
let particles = [];
let nextSoftBodyId = 0;
const restitution = 0.4; // Moved from global constants as it's physics-specific

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
        softBodyPopulation.push(new SoftBody(nextSoftBodyId++, randX, randY));
    }
    // lastTime = performance.now(); // This will be handled in main.js
    updatePopulationCount();
}


function initFluidSimulation() {
    const scaleX = WORLD_WIDTH / FLUID_GRID_SIZE_CONTROL;
    const scaleY = WORLD_HEIGHT / FLUID_GRID_SIZE_CONTROL;
    fluidField = new FluidField(FLUID_GRID_SIZE_CONTROL, FLUID_DIFFUSION, FLUID_VISCOSITY, 1/60, scaleX, scaleY);
    fluidField.useWrapping = IS_WORLD_WRAPPING;
    fluidField.maxVelComponent = MAX_FLUID_VELOCITY_COMPONENT; // Pass the cap
    velocityEmitters = [];
    if (offscreenFluidCanvas) {
        offscreenFluidCanvas.width = Math.round(FLUID_GRID_SIZE_CONTROL); // Ensure integer
        offscreenFluidCanvas.height = Math.round(FLUID_GRID_SIZE_CONTROL); // Ensure integer
    }
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
            } else if (body.creatureEnergy >= MAX_CREATURE_ENERGY && body.canReproduce && canCreaturesReproduceGlobally) {
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
                 softBodyPopulation.push(new SoftBody(nextSoftBodyId++, randX, randY));
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
    ctx.fillRect(0, 0, WORLD_WIDTH, WORLD_HEIGHT);

    ctx.save();

    ctx.scale(viewZoom, viewZoom);
    ctx.translate(-viewOffsetX, -viewOffsetY);

    if (fluidField) {
        fluidField.draw(ctx, WORLD_WIDTH, WORLD_HEIGHT);
    }
    if (SHOW_NUTRIENT_MAP && nutrientField && fluidField) {
        drawNutrientMap(ctx);
    }
    if (SHOW_LIGHT_MAP && lightField && fluidField) {
        drawLightMap(ctx);
    }
    if (SHOW_VISCOSITY_MAP && viscosityField && fluidField) {
        drawViscosityMap(ctx);
    }

    for (let particle of particles) {
        particle.draw(ctx);
    }
    for (let body of softBodyPopulation) {
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