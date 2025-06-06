let lastTime = 0;
let deltaTime = 0;
let animationFrameId = null;
let statsUpdateCounter = 0; // New counter for throttling stats update
const STATS_UPDATE_INTERVAL = 60; // Update stats approx every 60 frames (e.g., once per second at 60fps)
let frameTimeDisplayElement = null; // To store the DOM element
let frameTimeAccumulator = 0;
let frameCountForAvg = 0;
const canvas = document.getElementById('simulationCanvas');
const ctx = canvas.getContext('2d');
const webgpuCanvas = document.getElementById('webgpuFluidCanvas');
let offscreenFluidCanvas, offscreenFluidCtx;


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

    if (!canvas) {
        console.error("CRITICAL: 'canvas' element not found!");
        return;
    }

    frameTimeDisplayElement = document.getElementById('frameTimeDisplay'); // Initialize here


    // Initialize ALL global config variables with hardcoded defaults first
    // This ensures they have a value even if JSON load fails or is incomplete.
    // These are already defined in config.js, so ensure they are loaded before this script.
    WORLD_WIDTH = parseInt(worldWidthInput.value) || 8000;
    WORLD_HEIGHT = parseInt(worldHeightInput.value) || 6000;
    canvas.width = 1920; // Set canvas internal size to fixed HD
    canvas.height = 1080; // Set canvas internal size to fixed HD

    // initializeDefaultSliderVariables(); // Removed - global vars in config.js are the defaults

    INITIAL_POPULATION_SIZE = CREATURE_POPULATION_FLOOR;

    initializeSpatialGrid();
    initializeAllSliderDisplays(); // Syncs HTML sliders with JS global defaults and updates display spans
    await initFluidSimulation(USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    initNutrientMap();
    initLightMap();
    initViscosityMap();
    initParticles();
    initializePopulation();
    updateInstabilityIndicator();
    lastTime = performance.now(); // Initialize lastTime before starting the loop
    frameTimeDisplayElement = document.getElementById('frameTimeDisplay'); // Get the element
    animationFrameId = requestAnimationFrame(gameLoop);
}

// --- Game Loop ---
function gameLoop(timestamp) {
    const loopStartTime = performance.now(); // Record start time of the loop

    deltaTime = (timestamp - lastTime) / 1000;
    if (isNaN(deltaTime) || deltaTime <= 0) deltaTime = 1 / 60; // Ensure valid deltaTime
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

    if (!IS_HEADLESS_MODE) {
        draw(); // Draw on every frame, unless in headless mode
    }

    // Update stats panel if open, but throttled
    statsUpdateCounter++;
    if (statsUpdateCounter >= STATS_UPDATE_INTERVAL) {
        if (statsPanel && statsPanel.classList.contains('open')) {
            updateStatsPanel();
        }
        if (frameTimeDisplayElement && frameCountForAvg > 0) {
            const avgFrameTime = frameTimeAccumulator / frameCountForAvg;
            frameTimeDisplayElement.textContent = `Frame Time: ${avgFrameTime.toFixed(2)} ms`;
            frameTimeAccumulator = 0;
            frameCountForAvg = 0;
        }
        statsUpdateCounter = 0;
    }

    const loopEndTime = performance.now();
    const currentFrameDuration = loopEndTime - loopStartTime;
    frameTimeAccumulator += currentFrameDuration;
    frameCountForAvg++;

    animationFrameId = requestAnimationFrame(gameLoop); // Keep the loop going
}

document.addEventListener('DOMContentLoaded', main);
