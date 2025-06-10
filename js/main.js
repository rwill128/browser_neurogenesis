import config from './config.js';
import './classes/constants.js';
import './classes/MassPoint.js';
import './classes/Spring.js';
import './classes/Brain.js';
import './classes/SoftBody.js';
import './classes/FluidField.js';
import './classes/Particle.js';
import { initializeSpatialGrid, initializePopulation, updatePhysics, initFluidSimulation, initParticles, initNutrientMap, initLightMap, initViscosityMap } from './simulation.js';
import { 
    canvas, webgpuCanvas, worldWidthInput, worldHeightInput,
    updateInstabilityIndicator, initializeAllSliderDisplays, updatePopulationCount, updateStatsPanel, updateInfoPanel
} from './ui.js';

let lastTime = 0;
let deltaTime = 0;
let animationFrameId = null;
let statsUpdateCounter = 0;
const STATS_UPDATE_INTERVAL = 60;
let frameTimeDisplayElement = null;
let frameTimeAccumulator = 0;
let frameCountForAvg = 0;

async function main() {
    config.WORLD_WIDTH = parseInt(worldWidthInput.value) || 8000;
    config.WORLD_HEIGHT = parseInt(worldHeightInput.value) || 6000;
    canvas.width = 1920;
    canvas.height = 1080;

    config.GRID_COLS = Math.ceil(config.WORLD_WIDTH / config.GRID_CELL_SIZE);
    config.GRID_ROWS = Math.ceil(config.WORLD_HEIGHT / config.GRID_CELL_SIZE);

    config.INITIAL_POPULATION_SIZE = config.CREATURE_POPULATION_FLOOR; 

    initializeSpatialGrid();
    initializeAllSliderDisplays();
    await initFluidSimulation(config.USE_GPU_FLUID_SIMULATION ? webgpuCanvas : canvas);
    initNutrientMap(); 
    initLightMap(); 
    initViscosityMap(); 
    initParticles();
    initializePopulation();
    updateInstabilityIndicator();
    lastTime = performance.now();
    frameTimeDisplayElement = document.getElementById('frameTimeDisplay');
    animationFrameId = requestAnimationFrame(gameLoop);
}

function gameLoop(timestamp) {
    const loopStartTime = performance.now();

    deltaTime = (timestamp - lastTime) / 1000;
    if (isNaN(deltaTime) || deltaTime <= 0) deltaTime = 1/60;
    lastTime = timestamp;

    const currentMaxDeltaTime = config.MAX_DELTA_TIME_MS / 1000.0;
    const effectiveDeltaTime = Math.min(deltaTime, currentMaxDeltaTime);

    if (!config.IS_SIMULATION_PAUSED) {
        config.totalSimulationTime += effectiveDeltaTime;

        if (config.nutrientCyclePeriodSeconds > 0) {
            config.globalNutrientMultiplier = config.nutrientCycleBaseAmplitude + config.nutrientCycleWaveAmplitude * Math.sin((config.totalSimulationTime * 2 * Math.PI) / config.nutrientCyclePeriodSeconds);
            config.globalNutrientMultiplier = Math.max(0.01, config.globalNutrientMultiplier); 
        } else {
            config.globalNutrientMultiplier = config.nutrientCycleBaseAmplitude + config.nutrientCycleWaveAmplitude;
        }
        currentNutrientMultiplierDisplay.textContent = config.globalNutrientMultiplier.toFixed(2);

        if (config.lightCyclePeriodSeconds > 0) {
            config.globalLightMultiplier = (Math.sin((config.totalSimulationTime * 2 * Math.PI) / config.lightCyclePeriodSeconds) + 1) / 2; 
        } else {
            config.globalLightMultiplier = 0.5; 
        }
        currentLightMultiplierDisplay.textContent = config.globalLightMultiplier.toFixed(2);

        updatePhysics(effectiveDeltaTime);
    }

    if (!config.IS_HEADLESS_MODE) {
        draw();
    }

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

    animationFrameId = requestAnimationFrame(gameLoop);
}

document.addEventListener('DOMContentLoaded', main);
