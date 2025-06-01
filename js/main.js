let lastTime = 0;
let deltaTime = 0;
let animationFrameId = null;
let statsUpdateCounter = 0; // New counter for throttling stats update
const STATS_UPDATE_INTERVAL = 60; // Update stats approx every 60 frames (e.g., once per second at 60fps)

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
    deltaTime = (timestamp - lastTime) / 1000;
    if (isNaN(deltaTime) || deltaTime <= 0) deltaTime = 1/60; // Ensure valid deltaTime
    lastTime = timestamp;

    const currentMaxDeltaTime = MAX_DELTA_TIME_MS / 1000.0;
    const effectiveDeltaTime = Math.min(deltaTime, currentMaxDeltaTime);

    if (!IS_SIMULATION_PAUSED) {
        totalSimulationTime += effectiveDeltaTime;

        // Calculate global nutrient multiplier
        if (nutrientCyclePeriodSeconds > 0) {
            globalNutrientMultiplier = nutrientCycleBaseAmplitude + nutrientCycleWaveAmplitude * Math.sin((totalSimulationTime * 2 * Math.PI) / nutrientCyclePeriodSeconds);
            globalNutrientMultiplier = Math.max(0.01, globalNutrientMultiplier); 
        } else {
            globalNutrientMultiplier = nutrientCycleBaseAmplitude + nutrientCycleWaveAmplitude;
        }
        currentNutrientMultiplierDisplay.textContent = globalNutrientMultiplier.toFixed(2);

        // Calculate global light multiplier
        if (lightCyclePeriodSeconds > 0) {
            globalLightMultiplier = (Math.sin((totalSimulationTime * 2 * Math.PI) / lightCyclePeriodSeconds) + 1) / 2; 
        } else {
            globalLightMultiplier = 0.5; 
        }
        currentLightMultiplierDisplay.textContent = globalLightMultiplier.toFixed(2);

        updatePhysics(effectiveDeltaTime); // Only update physics if not paused
    }

    draw(); // Draw on every frame, regardless of pause state

    // Update stats panel if open, but throttled
    statsUpdateCounter++;
    if (statsPanel && statsPanel.classList.contains('open') && statsUpdateCounter >= STATS_UPDATE_INTERVAL) {
        updateStatsPanel();
        statsUpdateCounter = 0;
    }

    animationFrameId = requestAnimationFrame(gameLoop); // Keep the loop going
}

main(); // Start the main sequence 