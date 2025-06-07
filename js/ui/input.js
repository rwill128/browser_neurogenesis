// --- Mouse Interaction State Variables ---
let selectedSoftBodyPoint = null;
let mouse = {x: 0, y: 0, prevX: 0, prevY: 0, isDown: false, dx: 0, dy: 0};
let isRightDragging = false;
let panStartMouseDisplayX = 0;
let panStartMouseDisplayY = 0;
let panInitialViewOffsetX = 0;
let panInitialViewOffsetY = 0;
let isPaintingNutrients = false;
let isPaintingLight = false;
let isPaintingViscosity = false;
let emitterDragStartCell = null;
let currentEmitterPreview = null;


// --- Mouse Interaction Logic ---
function updateMouse(e) {
    const rect = canvas.getBoundingClientRect();
    mouse.prevX = mouse.x;
    mouse.prevY = mouse.y;
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    mouse.dx = mouse.x - mouse.prevX;
    mouse.dy = mouse.y - mouse.prevY;
}

function getMouseWorldCoordinates(displayMouseX, displayMouseY) {
    const bitmapInternalWidth = canvas.width;
    const bitmapInternalHeight = canvas.height;
    const cssClientWidth = canvas.clientWidth;
    const cssClientHeight = canvas.clientHeight;
    const bitmapDisplayScale = Math.min(cssClientWidth / bitmapInternalWidth, cssClientHeight / bitmapInternalHeight);
    const displayedBitmapWidthInCss = bitmapInternalWidth * bitmapDisplayScale;
    const displayedBitmapHeightInCss = bitmapInternalHeight * bitmapDisplayScale;
    const letterboxOffsetXcss = (cssClientWidth - displayedBitmapWidthInCss) / 2;
    const letterboxOffsetYcss = (cssClientHeight - displayedBitmapHeightInCss) / 2;
    const mouseOnScaledBitmapX = displayMouseX - letterboxOffsetXcss;
    const mouseOnScaledBitmapY = displayMouseY - letterboxOffsetYcss;
    const mouseOnUnscaledBitmapX = mouseOnScaledBitmapX / bitmapDisplayScale;
    const mouseOnUnscaledBitmapY = mouseOnScaledBitmapY / bitmapDisplayScale;
    const worldX = (mouseOnUnscaledBitmapX / viewZoom) + viewOffsetX;
    const worldY = (mouseOnUnscaledBitmapY / viewZoom) + viewOffsetY;
    return {x: worldX, y: worldY};
}

// --- Event Listeners ---
function initializeInputListeners() {
    document.addEventListener('keydown', (e) => {
        if (e.key.toLowerCase() !== 'p' && e.key.toLowerCase() !== 'w' && e.key.toLowerCase() !== 'a' && e.key.toLowerCase() !== 's' && e.key.toLowerCase() !== 'd' && IS_SIMULATION_PAUSED) {
            return;
        }
        const panSpeed = VIEW_PAN_SPEED / viewZoom;
        const maxPanX = Math.max(0, WORLD_WIDTH - (canvas.clientWidth / viewZoom));
        const maxPanY = Math.max(0, WORLD_HEIGHT - (canvas.clientHeight / viewZoom));

        switch (e.key.toLowerCase()) {
            case 'w': viewOffsetY = Math.max(0, viewOffsetY - panSpeed); break;
            case 's': viewOffsetY = Math.min(maxPanY, viewOffsetY + panSpeed); break;
            case 'a': viewOffsetX = Math.max(0, viewOffsetX - panSpeed); break;
            case 'd': viewOffsetX = Math.min(maxPanX, viewOffsetX + panSpeed); break;
            case 'p':
                IS_SIMULATION_PAUSED = !IS_SIMULATION_PAUSED;
                pauseResumeButton.textContent = IS_SIMULATION_PAUSED ? "Resume" : "Pause";
                if (!IS_SIMULATION_PAUSED) {
                    lastTime = performance.now();
                    requestAnimationFrame(gameLoop);
                }
                break;
        }
    });

    canvas.addEventListener('mousedown', (e) => {
        updateMouse(e);

        if (e.button === 2) { // Right mouse button
            isRightDragging = true;
            mouse.isDown = false;
            panStartMouseDisplayX = mouse.x;
            panStartMouseDisplayY = mouse.y;
            panInitialViewOffsetX = viewOffsetX;
            panInitialViewOffsetY = viewOffsetY;
            e.preventDefault();
            return;
        } else if (e.button === 0) { // Left mouse button
            mouse.isDown = true;
            isRightDragging = false;

            const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
            const simMouseX = worldCoords.x;
            const simMouseY = worldCoords.y;

            if (IS_CREATURE_IMPORT_MODE && IMPORTED_CREATURE_DATA) {
                placeImportedCreature(simMouseX, simMouseY);
                return;
            }

            if (IS_EMITTER_EDIT_MODE && fluidField) {
                if (IS_SIMULATION_PAUSED) return;
                const gridX = Math.floor(simMouseX / fluidField.scaleX);
                const gridY = Math.floor(simMouseY / fluidField.scaleY);
                emitterDragStartCell = {gridX, gridY, mouseStartX: simMouseX, mouseStartY: simMouseY};
                currentEmitterPreview = {
                    startX: (gridX + 0.5) * fluidField.scaleX,
                    startY: (gridY + 0.5) * fluidField.scaleY,
                    endX: simMouseX,
                    endY: simMouseY
                };
                selectedInspectBody = null;
                selectedInspectPoint = null;
                updateInfoPanel();
                return;
            }

            if (IS_NUTRIENT_EDIT_MODE) {
                if (IS_SIMULATION_PAUSED) return;
                isPaintingNutrients = true;
                paintNutrientBrush(simMouseX, simMouseY);
                selectedInspectBody = null;
                updateInfoPanel();
                selectedSoftBodyPoint = null;
                return;
            } else if (IS_LIGHT_EDIT_MODE) {
                if (IS_SIMULATION_PAUSED) return;
                isPaintingLight = true;
                paintLightBrush(simMouseX, simMouseY);
                selectedInspectBody = null;
                updateInfoPanel();
                selectedSoftBodyPoint = null;
                return;
            } else if (IS_VISCOSITY_EDIT_MODE) {
                if (IS_SIMULATION_PAUSED) return;
                isPaintingViscosity = true;
                paintViscosityBrush(simMouseX, simMouseY);
                selectedInspectBody = null;
                updateInfoPanel();
                selectedSoftBodyPoint = null;
                return;
            }

            let clickedOnPoint = false;
            for (let body of softBodyPopulation) {
                if (body.isUnstable) continue;
                for (let i = 0; i < body.massPoints.length; i++) {
                    const point = body.massPoints[i];
                    const dist = Math.sqrt((point.pos.x - simMouseX) ** 2 + (point.pos.y - simMouseY) ** 2);
                    if (dist < point.radius * 2.5) {
                        selectedSoftBodyPoint = {body: body, point: point};
                        selectedInspectBody = body;
                        selectedInspectPoint = point;
                        selectedInspectPointIndex = i;
                        if (!IS_SIMULATION_PAUSED) {
                            point.isFixed = true;
                            point.prevPos.x = point.pos.x;
                            point.prevPos.y = point.pos.y;
                        }
                        clickedOnPoint = true;
                        break;
                    }
                }
                if (clickedOnPoint) break;
            }

            if (!clickedOnPoint) {
                selectedSoftBodyPoint = null;
            }
            updateInfoPanel();
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        updateMouse(e);

        if (isRightDragging) {
            const displayDx = mouse.x - panStartMouseDisplayX;
            const displayDy = mouse.y - panStartMouseDisplayY;
            const bitmapDisplayScale = Math.min(canvas.clientWidth / canvas.width, canvas.clientHeight / canvas.height);
            const panDeltaX_world = displayDx / (bitmapDisplayScale * viewZoom);
            const panDeltaY_world = displayDy / (bitmapDisplayScale * viewZoom);
            viewOffsetX = panInitialViewOffsetX - panDeltaX_world;
            viewOffsetY = panInitialViewOffsetY - panDeltaY_world;
            const maxPanX = Math.max(0, WORLD_WIDTH - (canvas.clientWidth / bitmapDisplayScale / viewZoom));
            const maxPanY = Math.max(0, WORLD_HEIGHT - (canvas.clientHeight / bitmapDisplayScale / viewZoom));
            viewOffsetX = Math.max(0, Math.min(viewOffsetX, maxPanX));
            viewOffsetY = Math.max(0, Math.min(viewOffsetY, maxPanY));
            return;
        }

        const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
        const simMouseX = worldCoords.x;
        const simMouseY = worldCoords.y;
        const worldPrevCoords = getMouseWorldCoordinates(mouse.prevX, mouse.prevY);
        const worldMouseDx = simMouseX - worldPrevCoords.x;
        const worldMouseDy = simMouseY - worldPrevCoords.y;

        if (mouse.isDown && !IS_SIMULATION_PAUSED) {
            if (IS_EMITTER_EDIT_MODE && emitterDragStartCell) {
                currentEmitterPreview.endX = simMouseX;
                currentEmitterPreview.endY = simMouseY;
            } else if (selectedSoftBodyPoint) {
                const point = selectedSoftBodyPoint.point;
                point.prevPos.x = point.pos.x;
                point.prevPos.y = point.pos.y;
                point.pos.x = simMouseX;
                point.pos.y = simMouseY;
            } else if (fluidField) {
                const fluidGridX = Math.floor(simMouseX / fluidField.scaleX);
                const fluidGridY = Math.floor(simMouseY / fluidField.scaleY);
                fluidField.addDensity(fluidGridX, fluidGridY, Math.random() * 100 + 155, Math.random() * 50 + 25, Math.random() * 100 + 100, 150 + Math.random() * 50);
                fluidField.addDensity(fluidGridX, fluidGridY, Math.random() * 50 + 25, Math.random() * 100 + 155, Math.random() * 100 + 155, 150 + Math.random() * 50);
                fluidField.addVelocity(fluidGridX, fluidGridY, worldMouseDx * FLUID_MOUSE_DRAG_VELOCITY_SCALE, worldMouseDy * FLUID_MOUSE_DRAG_VELOCITY_SCALE);
            }
        } else if (mouse.isDown && IS_EMITTER_EDIT_MODE && emitterDragStartCell) {
            currentEmitterPreview.endX = simMouseX;
            currentEmitterPreview.endY = simMouseY;
        }

        if (IS_NUTRIENT_EDIT_MODE && isPaintingNutrients && mouse.isDown && !IS_SIMULATION_PAUSED) {
            paintNutrientBrush(simMouseX, simMouseY);
        } else if (IS_LIGHT_EDIT_MODE && isPaintingLight && mouse.isDown && !IS_SIMULATION_PAUSED) {
            paintLightBrush(simMouseX, simMouseY);
        } else if (IS_VISCOSITY_EDIT_MODE && isPaintingViscosity && mouse.isDown && !IS_SIMULATION_PAUSED) {
            paintViscosityBrush(simMouseX, simMouseY);
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (e.button === 2) {
            isRightDragging = false;
            e.preventDefault();
        } else if (e.button === 0) {
            mouse.isDown = false;
            isPaintingNutrients = false;
            isPaintingLight = false;
            isPaintingViscosity = false;
            if (IS_EMITTER_EDIT_MODE && emitterDragStartCell && fluidField && !IS_SIMULATION_PAUSED) {
                const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
                const worldForceX = (worldCoords.x - emitterDragStartCell.mouseStartX) * EMITTER_MOUSE_DRAG_SCALE;
                const worldForceY = (worldCoords.y - emitterDragStartCell.mouseStartY) * EMITTER_MOUSE_DRAG_SCALE;
                const existingEmitter = velocityEmitters.find(em => em.gridX === emitterDragStartCell.gridX && em.gridY === emitterDragStartCell.gridY);
                if (existingEmitter) {
                    existingEmitter.forceX = worldForceX / fluidField.scaleX;
                    existingEmitter.forceY = worldForceY / fluidField.scaleY;
                } else {
                    velocityEmitters.push({ gridX: emitterDragStartCell.gridX, gridY: emitterDragStartCell.gridY, forceX: worldForceX / fluidField.scaleX, forceY: worldForceY / fluidField.scaleY });
                }
                emitterDragStartCell = null;
                currentEmitterPreview = null;
            }
            if (selectedSoftBodyPoint) {
                const point = selectedSoftBodyPoint.point;
                if (!IS_SIMULATION_PAUSED) {
                    point.isFixed = false;
                    const worldDx = (mouse.dx / Math.min(canvas.clientWidth / WORLD_WIDTH, canvas.clientHeight / WORLD_HEIGHT) / viewZoom);
                    const worldDy = (mouse.dy / Math.min(canvas.clientWidth / WORLD_WIDTH, canvas.clientHeight / WORLD_HEIGHT) / viewZoom);
                    point.prevPos.x = point.pos.x - worldDx * 1.0;
                    point.prevPos.y = point.pos.y - worldDy * 1.0;
                }
                selectedSoftBodyPoint = null;
            }
        }
    });

    canvas.addEventListener('mouseleave', () => {
        mouse.isDown = false;
        isRightDragging = false;
        isPaintingNutrients = false;
        isPaintingLight = false;
        isPaintingViscosity = false;

        if (IS_EMITTER_EDIT_MODE && emitterDragStartCell) {
             const worldCoords = getMouseWorldCoordinates(mouse.x, mouse.y);
             const worldForceX = (worldCoords.x - emitterDragStartCell.mouseStartX) * EMITTER_MOUSE_DRAG_SCALE;
             const worldForceY = (worldCoords.y - emitterDragStartCell.mouseStartY) * EMITTER_MOUSE_DRAG_SCALE;
             const existingEmitter = velocityEmitters.find(em => em.gridX === emitterDragStartCell.gridX && em.gridY === emitterDragStartCell.gridY);
             if (existingEmitter) {
                 existingEmitter.forceX = worldForceX / fluidField.scaleX;
                 existingEmitter.forceY = worldForceY / fluidField.scaleY;
             } else {
                 velocityEmitters.push({ gridX: emitterDragStartCell.gridX, gridY: emitterDragStartCell.gridY, forceX: worldForceX / fluidField.scaleX, forceY: worldForceY / fluidField.scaleY });
             }
        }
        emitterDragStartCell = null;
        currentEmitterPreview = null;

        if (selectedSoftBodyPoint) {
            const point = selectedSoftBodyPoint.point;
            point.isFixed = false;
            const worldDx = (mouse.dx / Math.min(canvas.clientWidth / WORLD_WIDTH, canvas.clientHeight / WORLD_HEIGHT) / viewZoom);
            const worldDy = (mouse.dy / Math.min(canvas.clientWidth / WORLD_WIDTH, canvas.clientHeight / WORLD_HEIGHT) / viewZoom);
            point.prevPos.x = point.pos.x - worldDx * 1.0;
            point.prevPos.y = point.pos.y - worldDy * 1.0;
            selectedSoftBodyPoint = null;
        }
    });

    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        if (IS_CREATURE_IMPORT_MODE) {
            IS_CREATURE_IMPORT_MODE = false;
            IMPORTED_CREATURE_DATA = null;
            creatureImportStatus.textContent = "";
            canvas.style.cursor = 'default';
        }
    });

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const worldMouseBeforeZoom = getMouseWorldCoordinates(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
        const scroll = e.deltaY < 0 ? 1 : -1;
        let newZoom = viewZoom * Math.pow(1 + ZOOM_SENSITIVITY * 10, scroll);
        const minZoomX = canvas.clientWidth / WORLD_WIDTH;
        const minZoomY = canvas.clientHeight / WORLD_HEIGHT;
        let dynamicMinZoom = Math.min(minZoomX, minZoomY);
        dynamicMinZoom = Math.max(0.01, dynamicMinZoom);
        viewZoom = Math.max(dynamicMinZoom, Math.min(newZoom, MAX_ZOOM));
        
        const bitmapDisplayScale = Math.min(canvas.clientWidth / canvas.width, canvas.clientHeight / canvas.height);
        const mouseOnUnscaledBitmapX = (e.clientX - canvas.getBoundingClientRect().left - (canvas.clientWidth - canvas.width * bitmapDisplayScale) / 2) / bitmapDisplayScale;
        const mouseOnUnscaledBitmapY = (e.clientY - canvas.getBoundingClientRect().top - (canvas.clientHeight - canvas.height * bitmapDisplayScale) / 2) / bitmapDisplayScale;

        viewOffsetX = worldMouseBeforeZoom.x - (mouseOnUnscaledBitmapX / viewZoom);
        viewOffsetY = worldMouseBeforeZoom.y - (mouseOnUnscaledBitmapY / viewZoom);

        const maxPanX = Math.max(0, WORLD_WIDTH - (canvas.clientWidth / bitmapDisplayScale / viewZoom));
        const maxPanY = Math.max(0, WORLD_HEIGHT - (canvas.clientHeight / bitmapDisplayScale / viewZoom));
        viewOffsetX = Math.max(0, Math.min(viewOffsetX, maxPanX));
        viewOffsetY = Math.max(0, Math.min(viewOffsetY, maxPanY));
    });
} 