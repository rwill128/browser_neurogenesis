let lastTime = 0;
let deltaTime = 0;
let animationFrameId = null;

// --- Main Initialization Sequence ---
async function main() {
    // console.log('[Initial Sim State] Before any setup:', {
    //     WORLD_WIDTH_initial_input: worldWidthInput.value,
    //     WORLD_HEIGHT_initial_input: worldHeightInput.value,
    //     canvas_initial_W: canvas.width,
    //     canvas_initial_H: canvas.height,
    //     canvas_client_W: canvas.clientWidth,
    //     canvas_client_H: canvas.clientHeight,
    //     viewZoom_initial: viewZoom,
    //     viewOffsetX_initial: viewOffsetX,
    //     viewOffsetY_initial: viewOffsetY
    // });

    // Initialize ALL global config variables with hardcoded defaults first
    // This ensures they have a value even if JSON load fails or is incomplete.
    // These are already defined in config.js, so ensure they are loaded before this script.
    WORLD_WIDTH = parseInt(worldWidthInput.value) || 8000;
    WORLD_HEIGHT = parseInt(worldHeightInput.value) || 6000;
    canvas.width = WORLD_WIDTH; // Set canvas internal size
    canvas.height = WORLD_HEIGHT;

    // initializeDefaultSliderVariables(); // Removed - global vars in config.js are the defaults

    INITIAL_POPULATION_SIZE = CREATURE_POPULATION_FLOOR; 

    initializeSpatialGrid();
    initializeAllSliderDisplays(); // Syncs HTML sliders with JS global defaults and updates display spans
    initFluidSimulation();
    initNutrientMap(); 
    initLightMap(); 
    initViscosityMap(); 
    initParticles();
    initializePopulation();
    updateInstabilityIndicator();
    lastTime = performance.now(); // Initialize lastTime before starting the loop
    animationFrameId = requestAnimationFrame(gameLoop);
}

// --- Game Loop ---
function gameLoop(timestamp) {
    if (IS_SIMULATION_PAUSED) {
        animationFrameId = requestAnimationFrame(gameLoop);
        return;
    }

    deltaTime = (timestamp - lastTime) / 1000;
    if (isNaN(deltaTime) || deltaTime <= 0) deltaTime = 1/60;
    lastTime = timestamp;

    const currentMaxDeltaTime = MAX_DELTA_TIME_MS / 1000.0;
    deltaTime = Math.min(deltaTime, currentMaxDeltaTime);

    totalSimulationTime += deltaTime;

    // Calculate global nutrient multiplier
    if (nutrientCyclePeriodSeconds > 0) {
        globalNutrientMultiplier = nutrientCycleBaseAmplitude + nutrientCycleWaveAmplitude * Math.sin((totalSimulationTime * 2 * Math.PI) / nutrientCyclePeriodSeconds);
        globalNutrientMultiplier = Math.max(0.01, globalNutrientMultiplier); // Clamp to avoid zero or negative
    } else {
        globalNutrientMultiplier = nutrientCycleBaseAmplitude + nutrientCycleWaveAmplitude; // Effectively static if period is 0
    }
    currentNutrientMultiplierDisplay.textContent = globalNutrientMultiplier.toFixed(2);

    // Calculate global light multiplier
    if (lightCyclePeriodSeconds > 0) {
        globalLightMultiplier = (Math.sin((totalSimulationTime * 2 * Math.PI) / lightCyclePeriodSeconds) + 1) / 2; // Ranges 0.0 to 1.0
    } else {
        globalLightMultiplier = 0.5; // Static at mid-value if period is 0
    }
    currentLightMultiplierDisplay.textContent = globalLightMultiplier.toFixed(2);

    updatePhysics(deltaTime);
    animationFrameId = requestAnimationFrame(gameLoop); // Keep the loop going
}

main(); // Start the main sequence 