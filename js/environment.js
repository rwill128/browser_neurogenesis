let nutrientField = null;
let lightField = null;
let viscosityField = null;

function initNutrientMap() {
    const size = Math.round(FLUID_GRID_SIZE_CONTROL);
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
            let mappedValue = ((noiseValue + 1) / 2) * (MAX_NUTRIENT_VALUE - MIN_NUTRIENT_VALUE) + MIN_NUTRIENT_VALUE;
            nutrientField[y * size + x] = Math.max(MIN_NUTRIENT_VALUE, Math.min(MAX_NUTRIENT_VALUE, mappedValue));
        }
    }
    console.log(`Nutrient map initialized to ${size}x${size} with Perlin noise pattern.`);
}

function initLightMap() {
    const size = Math.round(FLUID_GRID_SIZE_CONTROL);
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
            lightField[y_coord * size + x] = Math.max(MIN_LIGHT_VALUE, Math.min(MAX_LIGHT_VALUE, noiseValue));
        }
    }
    console.log(`Light map initialized to ${size}x${size} with Perlin noise pattern.`);
}

function initViscosityMap() {
    const size = Math.round(FLUID_GRID_SIZE_CONTROL);
    if (size <= 0 || !Number.isFinite(size)) {
        console.error("Invalid size for viscosity map:", size);
        viscosityField = new Float32Array(0);
        return;
    }
    // viscosityField = new Float32Array(size * size).fill(1.0); // Old: Default to 1.0 (normal viscosity)
    viscosityField = new Float32Array(size * size);
    const noiseScale = 0.06; // Slightly different scale for variety, or keep same as others (0.05)
    const noiseOffsetX = Math.random() * 1000 + 2000; // Different random offset seed
    const noiseOffsetY = Math.random() * 1000 + 2000; // Different random offset seed

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            let noiseValue = perlin.noise(x * noiseScale + noiseOffsetX, y * noiseScale + noiseOffsetY);
            // Map Perlin noise (-1 to 1) to our desired viscosity range 
            let mappedValue = ((noiseValue + 1) / 2) * (MAX_VISCOSITY_MULTIPLIER - MIN_VISCOSITY_MULTIPLIER) + MIN_VISCOSITY_MULTIPLIER;
            viscosityField[y * size + x] = Math.max(MIN_VISCOSITY_MULTIPLIER, Math.min(MAX_VISCOSITY_MULTIPLIER, mappedValue));
        }
    }
    console.log(`Viscosity map initialized to ${size}x${size} with Perlin noise pattern.`);
}


function paintNutrientBrush(worldX, worldY) {
    if (!nutrientField || !fluidField || FLUID_GRID_SIZE_CONTROL <= 0) return;

    const N = Math.round(FLUID_GRID_SIZE_CONTROL);
    const gridX = Math.floor(worldX / fluidField.scaleX);
    const gridY = Math.floor(worldY / fluidField.scaleY);
    const brushRadius = Math.floor(NUTRIENT_BRUSH_SIZE / 2);

    for (let offsetY = -brushRadius; offsetY <= brushRadius; offsetY++) {
        for (let offsetX = -brushRadius; offsetX <= brushRadius; offsetX++) {
            // Optional: Circular brush shape check
            // if (offsetX * offsetX + offsetY * offsetY > brushRadius * brushRadius) {
            //     continue;
            // }

            const currentPaintX = gridX + offsetX;
            const currentPaintY = gridY + offsetY;

            if (currentPaintX >= 0 && currentPaintX < N && currentPaintY >= 0 && currentPaintY < N) {
                const idx = fluidField.IX(currentPaintX, currentPaintY);
                const currentValue = nutrientField[idx];
                const targetValue = NUTRIENT_BRUSH_VALUE;
                
                let newValue = currentValue + (targetValue - currentValue) * NUTRIENT_BRUSH_STRENGTH;
                newValue = Math.max(MIN_NUTRIENT_VALUE, Math.min(newValue, MAX_NUTRIENT_VALUE)); // Clamp
                nutrientField[idx] = newValue;
            }
        }
    }
}

function paintLightBrush(worldX, worldY) {
    if (!lightField || !fluidField || FLUID_GRID_SIZE_CONTROL <= 0) return;

    const N = Math.round(FLUID_GRID_SIZE_CONTROL);
    const gridX = Math.floor(worldX / fluidField.scaleX);
    const gridY = Math.floor(worldY / fluidField.scaleY);
    const brushRadius = Math.floor(LIGHT_BRUSH_SIZE / 2);

    for (let offsetY = -brushRadius; offsetY <= brushRadius; offsetY++) {
        for (let offsetX = -brushRadius; offsetX <= brushRadius; offsetX++) {
            const currentPaintX = gridX + offsetX;
            const currentPaintY = gridY + offsetY;

            if (currentPaintX >= 0 && currentPaintX < N && currentPaintY >= 0 && currentPaintY < N) {
                const idx = fluidField.IX(currentPaintX, currentPaintY);
                const currentValue = lightField[idx];
                const targetValue = LIGHT_BRUSH_VALUE;
                
                let newValue = currentValue + (targetValue - currentValue) * LIGHT_BRUSH_STRENGTH;
                newValue = Math.max(MIN_LIGHT_VALUE, Math.min(newValue, MAX_LIGHT_VALUE));
                lightField[idx] = newValue;
            }
        }
    }
}

function paintViscosityBrush(worldX, worldY) {
    if (!viscosityField || !fluidField || FLUID_GRID_SIZE_CONTROL <= 0) return;
    const N = Math.round(FLUID_GRID_SIZE_CONTROL);
    const gridX = Math.floor(worldX / fluidField.scaleX);
    const gridY = Math.floor(worldY / fluidField.scaleY);
    const brushRadius = Math.floor(VISCOSITY_BRUSH_SIZE / 2);

    for (let offsetY = -brushRadius; offsetY <= brushRadius; offsetY++) {
        for (let offsetX = -brushRadius; offsetX <= brushRadius; offsetX++) {
            const currentPaintX = gridX + offsetX;
            const currentPaintY = gridY + offsetY;
            if (currentPaintX >= 0 && currentPaintX < N && currentPaintY >= 0 && currentPaintY < N) {
                const idx = fluidField.IX(currentPaintX, currentPaintY);
                const currentValue = viscosityField[idx];
                const targetValue = VISCOSITY_BRUSH_VALUE;
                let newValue = currentValue + (targetValue - currentValue) * VISCOSITY_BRUSH_STRENGTH;
                newValue = Math.max(MIN_VISCOSITY_MULTIPLIER, Math.min(newValue, MAX_VISCOSITY_MULTIPLIER));
                viscosityField[idx] = newValue;
            }
        }
    }
}

// --- Drawing ---
function drawNutrientMap(ctxToDrawOn, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
    if (!SHOW_NUTRIENT_MAP || !nutrientField || !fluidField) return;

    const N = Math.round(FLUID_GRID_SIZE_CONTROL);
    if (N <= 0) return;

    const worldCellWidth = WORLD_WIDTH / N;
    const worldCellHeight = WORLD_HEIGHT / N;

    const viewLeftWorld = viewOffsetXWorld;
    const viewTopWorld = viewOffsetYWorld;
    const viewRightWorld = viewOffsetXWorld + viewportCanvasWidth / currentZoom;
    const viewBottomWorld = viewOffsetYWorld + viewportCanvasHeight / currentZoom;

    const startCol = Math.max(0, Math.floor(viewLeftWorld / worldCellWidth));
    const endCol = Math.min(N - 1, Math.floor(viewRightWorld / worldCellWidth));
    const startRow = Math.max(0, Math.floor(viewTopWorld / worldCellHeight));
    const endRow = Math.min(N - 1, Math.floor(viewBottomWorld / worldCellHeight));

    for (let j = startRow; j <= endRow; j++) {
        for (let i = startCol; i <= endCol; i++) {
            const baseNutrientValue = nutrientField[fluidField.IX(i, j)]; 
            const effectiveNutrientValue = baseNutrientValue * globalNutrientMultiplier;
            let r = 0, g = 0, b = 0, a = 0;

            if (effectiveNutrientValue < 1.0) { // Desert-like
                r = 150;
                g = 100 - (effectiveNutrientValue / 1.0) * 50;
                b = 50  - (effectiveNutrientValue / 1.0) * 25;
                a = 0.3 * (1.0 - effectiveNutrientValue / 1.0); 
            } else if (effectiveNutrientValue > 1.0) { // Oasis-like
                r = 100 - ((effectiveNutrientValue - 1.0) / 1.0) * 50;
                g = 150;
                b = 100 - ((effectiveNutrientValue - 1.0) / 1.0) * 50;
                a = 0.3 * ((effectiveNutrientValue - 1.0) / 1.0); 
            }
            a = Math.min(0.3, Math.max(0, a));

            if (a > 0.01) {
                const cellWorldX = i * worldCellWidth;
                const cellWorldY = j * worldCellHeight;
                ctxToDrawOn.fillStyle = `rgba(${Math.floor(r)},${Math.floor(g)},${Math.floor(b)},${a.toFixed(2)})`;
                ctxToDrawOn.fillRect(cellWorldX, cellWorldY, worldCellWidth, worldCellHeight);
            }
        }
    }
}

function drawLightMap(ctxToDrawOn, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
    if (!SHOW_LIGHT_MAP || !lightField || !fluidField) return;

    const N = Math.round(FLUID_GRID_SIZE_CONTROL);
    if (N <= 0) return;

    const worldCellWidth = WORLD_WIDTH / N;
    const worldCellHeight = WORLD_HEIGHT / N;

    const viewLeftWorld = viewOffsetXWorld;
    const viewTopWorld = viewOffsetYWorld;
    const viewRightWorld = viewOffsetXWorld + viewportCanvasWidth / currentZoom;
    const viewBottomWorld = viewOffsetYWorld + viewportCanvasHeight / currentZoom;

    const startCol = Math.max(0, Math.floor(viewLeftWorld / worldCellWidth));
    const endCol = Math.min(N - 1, Math.floor(viewRightWorld / worldCellWidth));
    const startRow = Math.max(0, Math.floor(viewTopWorld / worldCellHeight));
    const endRow = Math.min(N - 1, Math.floor(viewBottomWorld / worldCellHeight));

    for (let j = startRow; j <= endRow; j++) {
        for (let i = startCol; i <= endCol; i++) {
            const baseLightValue = lightField[fluidField.IX(i, j)]; 
            const effectiveLightValue = baseLightValue * globalLightMultiplier;
            const intensity = Math.floor(effectiveLightValue * 200); 
            const bluePart = Math.floor((1 - effectiveLightValue) * 50); 
            const alpha = effectiveLightValue * 0.15 + (1 - effectiveLightValue) * 0.05; 

            if (effectiveLightValue > 0.01 || (1-effectiveLightValue) > 0.01) { 
                const cellWorldX = i * worldCellWidth;
                const cellWorldY = j * worldCellHeight;
                ctxToDrawOn.fillStyle = `rgba(${intensity}, ${intensity}, ${bluePart}, ${alpha.toFixed(2)})`;
                ctxToDrawOn.fillRect(cellWorldX, cellWorldY, worldCellWidth, worldCellHeight);
            }
        }
    }
}

function drawViscosityMap(ctxToDrawOn, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
    if (!SHOW_VISCOSITY_MAP || !viscosityField || !fluidField) return;
    const N = Math.round(FLUID_GRID_SIZE_CONTROL);
    if (N <= 0) return;
    const worldCellWidth = WORLD_WIDTH / N;
    const worldCellHeight = WORLD_HEIGHT / N;

    const viewLeftWorld = viewOffsetXWorld;
    const viewTopWorld = viewOffsetYWorld;
    const viewRightWorld = viewOffsetXWorld + viewportCanvasWidth / currentZoom;
    const viewBottomWorld = viewOffsetYWorld + viewportCanvasHeight / currentZoom;

    const startCol = Math.max(0, Math.floor(viewLeftWorld / worldCellWidth));
    const endCol = Math.min(N - 1, Math.floor(viewRightWorld / worldCellWidth));
    const startRow = Math.max(0, Math.floor(viewTopWorld / worldCellHeight));
    const endRow = Math.min(N - 1, Math.floor(viewBottomWorld / worldCellHeight));

    for (let j = startRow; j <= endRow; j++) {
        for (let i = startCol; i <= endCol; i++) {
            const viscosityValue = viscosityField[fluidField.IX(i, j)];
            let r = 0, g = 0, b = 0, a = 0;

            if (viscosityValue > 1.0) { // Higher viscosity - cooler colors
                b = 150 + Math.min(105, (viscosityValue - 1.0) * 25);
                r = 50 - Math.min(50, (viscosityValue - 1.0) * 10);
                g = 50 - Math.min(50, (viscosityValue - 1.0) * 10);
                a = 0.05 + Math.min(0.25, (viscosityValue - 1.0) * 0.05);
            } else if (viscosityValue < 1.0) { // Lower viscosity - warmer colors
                r = 150 + Math.min(105, (1.0 - viscosityValue) * 25);
                g = 100 - Math.min(50, (1.0 - viscosityValue) * 20);
                b = 50 - Math.min(50, (1.0 - viscosityValue) * 10);
                a = 0.05 + Math.min(0.25, (1.0 - viscosityValue) * 0.05);
            }
            a = Math.min(0.3, Math.max(0, a));

            if (a > 0.01) {
                const cellWorldX = i * worldCellWidth;
                const cellWorldY = j * worldCellHeight;
                ctxToDrawOn.fillStyle = `rgba(${Math.floor(r)},${Math.floor(g)},${Math.floor(b)},${a.toFixed(2)})`;
                ctxToDrawOn.fillRect(cellWorldX, cellWorldY, worldCellWidth, worldCellHeight);
            }
        }
    }
} 