import config from '../config.js';

export class FluidField {
    constructor(size, diffusion, viscosity, dt, scaleX, scaleY) {
        this.size = Math.round(size);
        this.dt = dt;
        this.diffusion = diffusion;
        this.viscosity = viscosity;
        this.scaleX = scaleX; 
        this.scaleY = scaleY; 
        this.useWrapping = false;
        this.maxVelComponent = config.MAX_FLUID_VELOCITY_COMPONENT;
        this.viscosityField = null;

        this.densityR = new Float32Array(this.size * this.size).fill(0);
        this.densityG = new Float32Array(this.size * this.size).fill(0);
        this.densityB = new Float32Array(this.size * this.size).fill(0);
        this.densityR0 = new Float32Array(this.size * this.size).fill(0);
        this.densityG0 = new Float32Array(this.size * this.size).fill(0);
        this.densityB0 = new Float32Array(this.size * this.size).fill(0);

        this.Vx = new Float32Array(this.size * this.size).fill(0);
        this.Vy = new Float32Array(this.size * this.size).fill(0);
        this.Vx0 = new Float32Array(this.size * this.size).fill(0);
        this.Vy0 = new Float32Array(this.size * this.size).fill(0);

        // Scratch coefficient buffers for variable-viscosity solver path (reused to avoid per-step allocation).
        this._linSolveEffectiveA = new Float32Array(this.size * this.size);
        this._linSolveEffectiveCRecip = new Float32Array(this.size * this.size);

        this.iterations = 4; 

        /**
         * Active-tile tracking scaffold (phase 1 for sparse fluid architecture).
         *
         * This does not yet restrict solver compute to active tiles; it tracks spatial activity so
         * we can quantify sparsity and progressively migrate solver work to active regions.
         */
        this.activeTileSize = Math.max(8, Math.floor(Number(config.FLUID_ACTIVE_TILE_SIZE_CELLS) || 32));
        this.activeTileHalo = Math.max(0, Math.floor(Number(config.FLUID_ACTIVE_TILE_HALO_TILES) || 1));
        this.activeTileTtlMax = Math.max(1, Math.floor(Number(config.FLUID_ACTIVE_TILE_TTL_STEPS) || 12));
        this.activeTileCols = Math.max(1, Math.ceil(this.size / this.activeTileSize));
        this.activeTileRows = Math.max(1, Math.ceil(this.size / this.activeTileSize));
        this.activeTiles = new Map(); // key: "tx:ty" => remaining ttl steps
        this.activeTilesTouchedThisStep = new Set();
        this.lastActiveTileTelemetry = {
            activeTiles: 0,
            totalTiles: this.activeTileCols * this.activeTileRows,
            activePct: 0,
            touchedTiles: 0,
            sleepingTiles: 0,
            wakes: 0,
            carried: 0
        };
    }

    IX(x, y) {
        if (this.useWrapping) {
            x = (Math.floor(x) % this.size + this.size) % this.size;
            y = (Math.floor(y) % this.size + this.size) % this.size;
        } else {
            x = Math.max(0, Math.min(x, this.size - 1));
            y = Math.max(0, Math.min(y, this.size - 1));
        }
        return Math.floor(x) + Math.floor(y) * this.size;
    }

    _tileKey(tx, ty) {
        return `${tx}:${ty}`;
    }

    _markActiveTileByCoord(tx, ty) {
        if (!Number.isFinite(tx) || !Number.isFinite(ty)) return;
        const clampedTx = Math.max(0, Math.min(this.activeTileCols - 1, Math.floor(tx)));
        const clampedTy = Math.max(0, Math.min(this.activeTileRows - 1, Math.floor(ty)));
        const key = this._tileKey(clampedTx, clampedTy);
        const prev = Number(this.activeTiles.get(key) || 0);
        this.activeTiles.set(key, Math.max(prev, this.activeTileTtlMax));
        this.activeTilesTouchedThisStep.add(key);
    }

    /**
     * Mark activity around a fluid-grid cell (gx,gy), including halo tiles.
     */
    markActiveCell(gx, gy) {
        const tx = Math.floor((Math.max(0, Math.min(this.size - 1, Math.floor(gx))) / this.activeTileSize));
        const ty = Math.floor((Math.max(0, Math.min(this.size - 1, Math.floor(gy))) / this.activeTileSize));
        for (let oy = -this.activeTileHalo; oy <= this.activeTileHalo; oy++) {
            for (let ox = -this.activeTileHalo; ox <= this.activeTileHalo; ox++) {
                this._markActiveTileByCoord(tx + ox, ty + oy);
            }
        }
    }

    /**
     * Seed activity around creature centers before each fluid step.
     */
    seedActiveTilesFromBodies(bodies) {
        if (!Array.isArray(bodies) || bodies.length === 0) return;
        for (const b of bodies) {
            if (!b || b.isUnstable) continue;
            const c = typeof b.getAveragePosition === 'function' ? b.getAveragePosition() : null;
            if (!c) continue;
            const gx = Math.floor((Number(c.x) || 0) / Math.max(1e-6, this.scaleX));
            const gy = Math.floor((Number(c.y) || 0) / Math.max(1e-6, this.scaleY));
            this.markActiveCell(gx, gy);
        }
    }

    _finalizeActiveTileTelemetry() {
        let sleepingTiles = 0;
        let carried = 0;
        for (const [key, ttlRaw] of Array.from(this.activeTiles.entries())) {
            const ttl = Math.max(0, Math.floor(Number(ttlRaw) || 0) - 1);
            if (ttl <= 0) {
                this.activeTiles.delete(key);
                sleepingTiles += 1;
            } else {
                this.activeTiles.set(key, ttl);
                carried += 1;
            }
        }

        const totalTiles = Math.max(1, this.activeTileCols * this.activeTileRows);
        const activeTiles = this.activeTiles.size;
        this.lastActiveTileTelemetry = {
            activeTiles,
            totalTiles,
            activePct: activeTiles / totalTiles,
            touchedTiles: this.activeTilesTouchedThisStep.size,
            sleepingTiles,
            wakes: Math.max(0, this.activeTilesTouchedThisStep.size - carried),
            carried
        };
        this.activeTilesTouchedThisStep.clear();
    }

    getActiveTileTelemetry() {
        return { ...this.lastActiveTileTelemetry };
    }

    addDensity(x, y, emitterR, emitterG, emitterB, emissionStrength) {
        const idx = this.IX(x, y);
        this.markActiveCell(x, y);
        const normalizedEmissionEffect = (emissionStrength / 50.0) * config.DYE_PULL_RATE;
        this.densityR[idx] = Math.max(0, Math.min(255, this.densityR[idx] + (emitterR - this.densityR[idx]) * normalizedEmissionEffect));
        this.densityG[idx] = Math.max(0, Math.min(255, this.densityG[idx] + (emitterG - this.densityG[idx]) * normalizedEmissionEffect));
        this.densityB[idx] = Math.max(0, Math.min(255, this.densityB[idx] + (emitterB - this.densityB[idx]) * normalizedEmissionEffect));
    }

    addVelocity(x, y, amountX, amountY) {
        const idx = this.IX(x, y);
        this.markActiveCell(x, y);
        this.Vx[idx] = Math.max(-this.maxVelComponent, Math.min(this.Vx[idx] + amountX, this.maxVelComponent));
        this.Vy[idx] = Math.max(-this.maxVelComponent, Math.min(this.Vy[idx] + amountY, this.maxVelComponent));
    }

    clampVelocityComponents(arr) {
        for(let i=0; i < arr.length; i++) {
            arr[i] = Math.max(-this.maxVelComponent, Math.min(arr[i], this.maxVelComponent));
        }
    }

    _getSolverIterationsForField(field_type) {
        const fallback = Math.max(1, Math.floor(Number(this.iterations) || 4));

        if (field_type === 'velX' || field_type === 'velY') {
            return Math.max(1, Math.floor(Number(config.FLUID_SOLVER_ITERATIONS_VELOCITY) || fallback));
        }
        if (field_type === 'pressure') {
            return Math.max(1, Math.floor(Number(config.FLUID_SOLVER_ITERATIONS_PRESSURE) || fallback));
        }
        if (field_type === 'density') {
            return Math.max(1, Math.floor(Number(config.FLUID_SOLVER_ITERATIONS_DENSITY) || fallback));
        }
        return fallback;
    }

    lin_solve(b, x, x0, a_global_param, c_global_param, field_type, base_diff_rate, dt_param, bounds = null) {
        const N = this.size;
        const NMinus1 = N - 1;
        const xMin = bounds ? Math.max(1, Math.min(NMinus1 - 1, Math.floor(bounds.xMin))) : 1;
        const xMax = bounds ? Math.max(xMin, Math.min(NMinus1 - 1, Math.floor(bounds.xMax))) : (NMinus1 - 1);
        const yMin = bounds ? Math.max(1, Math.min(NMinus1 - 1, Math.floor(bounds.yMin))) : 1;
        const yMax = bounds ? Math.max(yMin, Math.min(NMinus1 - 1, Math.floor(bounds.yMax))) : (NMinus1 - 1);
        const cRecipGlobal = 1.0 / c_global_param;
        const solverIterations = this._getSolverIterationsForField(field_type);

        const isVelocityField = (field_type === 'velX' || field_type === 'velY');
        const viscosityField = (isVelocityField && this.viscosityField) ? this.viscosityField : null;

        if (!viscosityField) {
            for (let k_iter = 0; k_iter < solverIterations; k_iter++) {
                for (let j = yMin; j <= yMax; j++) {
                    let idx = j * N + xMin;
                    const rowEnd = j * N + xMax + 1;
                    while (idx < rowEnd) {
                        x[idx] = (x0[idx] + a_global_param * (x[idx + 1] + x[idx - 1] + x[idx + N] + x[idx - N])) * cRecipGlobal;
                        idx++;
                    }
                }
                this.set_bnd(b, x);
            }
            return;
        }

        const minVisc = Number(config.MIN_VISCOSITY_MULTIPLIER);
        const maxVisc = Number(config.MAX_VISCOSITY_MULTIPLIER);
        const gridFactor = (N - 2) * (N - 2);
        const baseScale = dt_param * base_diff_rate * gridFactor;

        const effectiveAByIdx = this._linSolveEffectiveA;
        const effectiveCRecipByIdx = this._linSolveEffectiveCRecip;

        // Build local coefficients once; reuse across all Gauss-Seidel iterations in this solve.
        for (let j = yMin; j <= yMax; j++) {
            let idx = j * N + xMin;
            const rowEnd = j * N + xMax + 1;
            while (idx < rowEnd) {
                let localMultiplier = viscosityField[idx];
                if (!Number.isFinite(localMultiplier)) localMultiplier = 1;
                if (localMultiplier < minVisc) localMultiplier = minVisc;
                else if (localMultiplier > maxVisc) localMultiplier = maxVisc;

                const effectiveA = baseScale * localMultiplier;
                effectiveAByIdx[idx] = effectiveA;
                effectiveCRecipByIdx[idx] = 1.0 / (1 + 4 * effectiveA);
                idx++;
            }
        }

        for (let k_iter = 0; k_iter < solverIterations; k_iter++) {
            for (let j = yMin; j <= yMax; j++) {
                let idx = j * N + xMin;
                const rowEnd = j * N + xMax + 1;
                while (idx < rowEnd) {
                    const effectiveA = effectiveAByIdx[idx];
                    const effectiveCRecip = effectiveCRecipByIdx[idx];
                    x[idx] = (x0[idx] + effectiveA * (x[idx + 1] + x[idx - 1] + x[idx + N] + x[idx - N])) * effectiveCRecip;
                    idx++;
                }
            }
            this.set_bnd(b, x);
        }
    }

    diffuse(b, x_out, x_in, base_diff_rate, dt, field_type = 'density', bounds = null) {
        const a_global = dt * base_diff_rate * (this.size - 2) * (this.size - 2);
        this.lin_solve(b, x_out, x_in, a_global, 1 + 4 * a_global, field_type, base_diff_rate, dt, bounds);
    }

    project(velocX_in_out, velocY_in_out, p_temp, div_temp, bounds = null) {
        const N = this.size;
        const NMinus1 = N - 1;
        const xMin = bounds ? Math.max(1, Math.min(NMinus1 - 1, Math.floor(bounds.xMin))) : 1;
        const xMax = bounds ? Math.max(xMin, Math.min(NMinus1 - 1, Math.floor(bounds.xMax))) : (NMinus1 - 1);
        const yMin = bounds ? Math.max(1, Math.min(NMinus1 - 1, Math.floor(bounds.yMin))) : 1;
        const yMax = bounds ? Math.max(yMin, Math.min(NMinus1 - 1, Math.floor(bounds.yMax))) : (NMinus1 - 1);

        for (let j = yMin; j <= yMax; j++) {
            let idx = j * N + xMin;
            const rowEnd = j * N + xMax + 1;
            while (idx < rowEnd) {
                div_temp[idx] = -0.5 * (velocX_in_out[idx + 1] - velocX_in_out[idx - 1] + velocY_in_out[idx + N] - velocY_in_out[idx - N]) / N;
                p_temp[idx] = 0;
                idx++;
            }
        }

        this.set_bnd(0, div_temp);
        this.set_bnd(0, p_temp);
        this.lin_solve(0, p_temp, div_temp, 1, 4, 'pressure', 0, 0, bounds);

        for (let j = yMin; j <= yMax; j++) {
            let idx = j * N + xMin;
            const rowEnd = j * N + xMax + 1;
            while (idx < rowEnd) {
                velocX_in_out[idx] -= 0.5 * (p_temp[idx + 1] - p_temp[idx - 1]) * N;
                velocY_in_out[idx] -= 0.5 * (p_temp[idx + N] - p_temp[idx - N]) * N;
                idx++;
            }
        }

        this.set_bnd(1, velocX_in_out);
        this.set_bnd(2, velocY_in_out);
    }

    advect(b, d_out, d_in, velocX_source, velocY_source, dt, bounds = null) {
        const N = this.size;
        const NMinus1 = N - 1;
        const xMin = bounds ? Math.max(1, Math.min(NMinus1 - 1, Math.floor(bounds.xMin))) : 1;
        const xMax = bounds ? Math.max(xMin, Math.min(NMinus1 - 1, Math.floor(bounds.xMax))) : (NMinus1 - 1);
        const yMin = bounds ? Math.max(1, Math.min(NMinus1 - 1, Math.floor(bounds.yMin))) : 1;
        const yMax = bounds ? Math.max(yMin, Math.min(NMinus1 - 1, Math.floor(bounds.yMax))) : (NMinus1 - 1);
        const dtx_scaled = dt * N;
        const dty_scaled = dt * N;

        for (let j_cell = yMin; j_cell <= yMax; j_cell++) {
            let current_idx = j_cell * N + xMin;
            const rowEnd = j_cell * N + xMax + 1;

            for (let i_cell = 1; current_idx < rowEnd; i_cell++, current_idx++) {
                let x = i_cell - (dtx_scaled * velocX_source[current_idx]);
                let y = j_cell - (dty_scaled * velocY_source[current_idx]);

                let i0, i1, j0, j1;
                if (this.useWrapping) {
                    x = (x % N + N) % N;
                    y = (y % N + N) % N;
                    i0 = Math.floor(x);
                    j0 = Math.floor(y);
                    i1 = (i0 + 1) % N;
                    j1 = (j0 + 1) % N;
                } else {
                    if (x < 0.5) x = 0.5;
                    else if (x > N - 1.5) x = N - 1.5;
                    i0 = Math.floor(x);
                    i1 = i0 + 1;

                    if (y < 0.5) y = 0.5;
                    else if (y > N - 1.5) y = N - 1.5;
                    j0 = Math.floor(y);
                    j1 = j0 + 1;
                }

                const s1 = x - i0;
                const s0 = 1.0 - s1;
                const t1 = y - j0;
                const t0 = 1.0 - t1;

                const idx00 = i0 + j0 * N;
                const idx01 = i0 + j1 * N;
                const idx10 = i1 + j0 * N;
                const idx11 = i1 + j1 * N;

                d_out[current_idx] = s0 * (t0 * d_in[idx00] + t1 * d_in[idx01]) +
                                     s1 * (t0 * d_in[idx10] + t1 * d_in[idx11]);
            }
        }

        this.set_bnd(b, d_out);
    }

    set_bnd(b, x_arr) {
        const N = this.size;
        const NMinus1 = N - 1;
        const NMinus2 = N - 2;

        if (this.useWrapping) {
            for (let i = 1; i < NMinus1; i++) {
                x_arr[i] = x_arr[i + NMinus2 * N];
                x_arr[i + NMinus1 * N] = x_arr[i + N];
            }
            for (let j = 1; j < NMinus1; j++) {
                const row = j * N;
                x_arr[row] = x_arr[row + NMinus2];
                x_arr[row + NMinus1] = x_arr[row + 1];
            }

            x_arr[0] = 0.5 * (x_arr[1] + x_arr[N]);
            x_arr[NMinus1 * N] = 0.5 * (x_arr[NMinus1 * N + 1] + x_arr[NMinus2 * N]);
            x_arr[NMinus1] = 0.5 * (x_arr[NMinus2] + x_arr[NMinus1 + N]);
            x_arr[N * N - 1] = 0.5 * (x_arr[N * N - 2] + x_arr[N * (NMinus1 - 1) + NMinus1]);
            return;
        }

        const invertY = b === 2;
        const invertX = b === 1;

        for (let i = 1; i < NMinus1; i++) {
            x_arr[i] = invertY ? -x_arr[i + N] : x_arr[i + N];
            const bottomIdx = i + NMinus1 * N;
            x_arr[bottomIdx] = invertY ? -x_arr[i + NMinus2 * N] : x_arr[i + NMinus2 * N];
        }
        for (let j = 1; j < NMinus1; j++) {
            const row = j * N;
            x_arr[row] = invertX ? -x_arr[row + 1] : x_arr[row + 1];
            const rightIdx = row + NMinus1;
            x_arr[rightIdx] = invertX ? -x_arr[row + NMinus2] : x_arr[row + NMinus2];
        }

        x_arr[0] = 0.5 * (x_arr[1] + x_arr[N]);
        x_arr[NMinus1] = 0.5 * (x_arr[NMinus2] + x_arr[NMinus1 + N]);
        x_arr[NMinus1 * N] = 0.5 * (x_arr[NMinus1 * N + 1] + x_arr[NMinus2 * N]);
        x_arr[N * N - 1] = 0.5 * (x_arr[N * N - 2] + x_arr[N * (NMinus1 - 1) + NMinus1]);
    }

    /**
     * Compute active-cell bounds from currently active tiles (with safety halo).
     */
    _getActiveCellBounds() {
        if (!this.activeTiles || this.activeTiles.size === 0) return null;

        let minTx = Infinity;
        let minTy = Infinity;
        let maxTx = -Infinity;
        let maxTy = -Infinity;
        for (const key of this.activeTiles.keys()) {
            const [txRaw, tyRaw] = String(key).split(':');
            const tx = Math.floor(Number(txRaw));
            const ty = Math.floor(Number(tyRaw));
            if (!Number.isFinite(tx) || !Number.isFinite(ty)) continue;
            if (tx < minTx) minTx = tx;
            if (ty < minTy) minTy = ty;
            if (tx > maxTx) maxTx = tx;
            if (ty > maxTy) maxTy = ty;
        }
        if (!Number.isFinite(minTx) || !Number.isFinite(minTy) || !Number.isFinite(maxTx) || !Number.isFinite(maxTy)) {
            return null;
        }

        const pad = Math.max(1, this.activeTileSize);
        const xMin = Math.max(1, (minTx * this.activeTileSize) - pad);
        const yMin = Math.max(1, (minTy * this.activeTileSize) - pad);
        const xMax = Math.min(this.size - 2, ((maxTx + 1) * this.activeTileSize) + pad);
        const yMax = Math.min(this.size - 2, ((maxTy + 1) * this.activeTileSize) + pad);
        return { xMin, yMin, xMax, yMax };
    }

    step() {
        const bounds = this._getActiveCellBounds();
        this.diffuse(1, this.Vx0, this.Vx, this.viscosity, this.dt, 'velX', bounds);
        this.diffuse(2, this.Vy0, this.Vy, this.viscosity, this.dt, 'velY', bounds);
        this.clampVelocityComponents(this.Vx0);
        this.clampVelocityComponents(this.Vy0);
        this.project(this.Vx0, this.Vy0, this.Vx, this.Vy, bounds);
        this.advect(1, this.Vx, this.Vx0, this.Vx0, this.Vy0, this.dt, bounds);
        this.advect(2, this.Vy, this.Vy0, this.Vx0, this.Vy0, this.dt, bounds);
        this.clampVelocityComponents(this.Vx);
        this.clampVelocityComponents(this.Vy);
        this.project(this.Vx, this.Vy, this.Vx0, this.Vy0, bounds);
        this.diffuse(0, this.densityR0, this.densityR, this.diffusion, this.dt, 'density', bounds);
        this.diffuse(0, this.densityG0, this.densityG, this.diffusion, this.dt, 'density', bounds);
        this.diffuse(0, this.densityB0, this.densityB, this.diffusion, this.dt, 'density', bounds);
        this.advect(0, this.densityR, this.densityR0, this.Vx, this.Vy, this.dt, bounds);
        this.advect(0, this.densityG, this.densityG0, this.Vx, this.Vy, this.dt, bounds);
        this.advect(0, this.densityB, this.densityB0, this.Vx, this.Vy, this.dt, bounds);

        if (bounds) {
            const x0 = Math.max(0, Math.floor(bounds.xMin));
            const x1 = Math.min(this.size - 1, Math.floor(bounds.xMax));
            const y0 = Math.max(0, Math.floor(bounds.yMin));
            const y1 = Math.min(this.size - 1, Math.floor(bounds.yMax));
            for (let y = y0; y <= y1; y++) {
                let idx = y * this.size + x0;
                const rowEnd = y * this.size + x1 + 1;
                while (idx < rowEnd) {
                    this.densityR[idx] = Math.max(0, this.densityR[idx] - config.FLUID_FADE_RATE * 255 * this.dt);
                    this.densityG[idx] = Math.max(0, this.densityG[idx] - config.FLUID_FADE_RATE * 255 * this.dt);
                    this.densityB[idx] = Math.max(0, this.densityB[idx] - config.FLUID_FADE_RATE * 255 * this.dt);
                    idx++;
                }
            }
        } else {
            for (let i = 0; i < this.densityR.length; i++) {
                this.densityR[i] = Math.max(0, this.densityR[i] - config.FLUID_FADE_RATE * 255 * this.dt);
                this.densityG[i] = Math.max(0, this.densityG[i] - config.FLUID_FADE_RATE * 255 * this.dt);
                this.densityB[i] = Math.max(0, this.densityB[i] - config.FLUID_FADE_RATE * 255 * this.dt);
            }
        }
        this._finalizeActiveTileTelemetry();
    }

    draw(ctxToDrawOn, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
        console.log("Drawing with CPU FluidField"); // Ensure this log is active
        const N = Math.round(this.size);
        if (N <= 0 || !Number.isFinite(N)) {
            console.error("FluidField.draw: Invalid N size:", N);
            return;
        }

        const worldCellWidth = config.WORLD_WIDTH / N;
        const worldCellHeight = config.WORLD_HEIGHT / N;

        const viewportWorldWidth = viewportCanvasWidth / currentZoom;
        const viewportWorldHeight = viewportCanvasHeight / currentZoom;

        const viewLeftWorld = viewOffsetXWorld;
        const viewTopWorld = viewOffsetYWorld;
        const viewRightWorld = viewOffsetXWorld + viewportWorldWidth;
        const viewBottomWorld = viewOffsetYWorld + viewportWorldHeight;

        const startCol = Math.max(0, Math.floor(viewLeftWorld / worldCellWidth));
        const endCol = Math.min(N - 1, Math.floor(viewRightWorld / worldCellWidth));
        const startRow = Math.max(0, Math.floor(viewTopWorld / worldCellHeight));
        const endRow = Math.min(N - 1, Math.floor(viewBottomWorld / worldCellHeight));

        if (startCol > endCol || startRow > endRow) return; // No visible cells

        for (let j = startRow; j <= endRow; j++) {
            for (let i = startCol; i <= endCol; i++) {
                const idx = this.IX(i, j);
                const rVal = Math.min(255, Math.max(0, Math.floor(this.densityR[idx])));
                const gVal = Math.min(255, Math.max(0, Math.floor(this.densityG[idx])));
                const bVal = Math.min(255, Math.max(0, Math.floor(this.densityB[idx])));
                const alphaVal = (rVal > 1 || gVal > 1 || bVal > 1) ? 0.4 : 0; // Original alpha for blocky rendering

                if (alphaVal > 0) {
                    const cellWorldX = i * worldCellWidth;
                    const cellWorldY = j * worldCellHeight;
                    ctxToDrawOn.fillStyle = `rgba(${rVal},${gVal},${bVal},${alphaVal.toFixed(2)})`;
                    ctxToDrawOn.fillRect(cellWorldX, cellWorldY, worldCellWidth, worldCellHeight);
                }
            }
        }
    }

    clear() {
        this.densityR.fill(0); this.densityG.fill(0); this.densityB.fill(0);
        this.densityR0.fill(0); this.densityG0.fill(0); this.densityB0.fill(0);
        this.Vx.fill(0); this.Vy.fill(0);
        this.Vx0.fill(0); this.Vy0.fill(0);
    }

    setViscosityField(field) {
        this.viscosityField = field;
    }
}
