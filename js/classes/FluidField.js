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
        this.activeTileSize = Math.max(1, Math.floor(Number(config.FLUID_ACTIVE_TILE_SIZE_CELLS) || 32));
        this.activeTileHalo = Math.max(0, Math.floor(Number(config.FLUID_ACTIVE_TILE_HALO_TILES) || 1));
        this.activeTileTtlMax = Math.max(1, Math.floor(Number(config.FLUID_ACTIVE_TILE_TTL_STEPS) || 12));
        this.activeTileCols = Math.max(1, Math.ceil(this.size / this.activeTileSize));
        this.activeTileRows = Math.max(1, Math.ceil(this.size / this.activeTileSize));
        this.totalActiveTiles = this.activeTileCols * this.activeTileRows;
        this.carrierTiles = this._createTileTracker(); // dye/creature/particle carriers -> full-res priority
        this.momentumTiles = this._createTileTracker(); // velocity-only regions -> lower-frequency priority
        this.coarseMomentumBlockSize = 2; // phase A scaffold: 2x2 momentum-only macro blocks
        this.lastStepPerf = {
            totalMs: 0,
            seedMomentumMs: 0,
            diffuseVelMs: 0,
            projectMs: 0,
            advectVelMs: 0,
            diffuseDensityMs: 0,
            advectDensityMs: 0,
            fadeMs: 0
        };
        this.lastActiveTileTelemetry = {
            carrierActiveTiles: 0,
            momentumTilesTotal: 0,
            momentumTilesNonCarrier: 0,
            coarseMomentumBlockSize: this.coarseMomentumBlockSize,
            coarseMomentumBlockCount: 0,
            coarseMomentumCoveragePct: 0,
            totalTiles: this.activeTileCols * this.activeTileRows,
            carrierPct: 0,
            momentumPct: 0,
            momentumNonCarrierPct: 0,
            carrierTouchedTiles: 0,
            momentumTouchedTiles: 0,
            sleepingCarrierTiles: 0,
            sleepingMomentumTiles: 0
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
        const x = Math.max(0, Math.min(this.activeTileCols - 1, Math.floor(tx)));
        const y = Math.max(0, Math.min(this.activeTileRows - 1, Math.floor(ty)));
        return y * this.activeTileCols + x;
    }

    _tileTx(tileId) {
        return tileId % this.activeTileCols;
    }

    _tileTy(tileId) {
        return Math.floor(tileId / this.activeTileCols);
    }

    _createTileTracker() {
        return {
            ttl: new Uint16Array(this.totalActiveTiles),
            index: new Int32Array(this.totalActiveTiles).fill(-1),
            active: new Int32Array(this.totalActiveTiles),
            count: 0,
            touched: new Uint8Array(this.totalActiveTiles),
            touchedCount: 0
        };
    }

    _trackerHas(tracker, tileId) {
        return tracker.ttl[tileId] > 0;
    }

    _trackerActivate(tracker, tileId) {
        if (tracker.ttl[tileId] < this.activeTileTtlMax) tracker.ttl[tileId] = this.activeTileTtlMax;
        if (tracker.index[tileId] < 0) {
            const pos = tracker.count;
            tracker.active[pos] = tileId;
            tracker.index[tileId] = pos;
            tracker.count = pos + 1;
        }
        if (tracker.touched[tileId] === 0) {
            tracker.touched[tileId] = 1;
            tracker.touchedCount += 1;
        }
    }

    _trackerClearTouched(tracker) {
        const active = tracker.active;
        for (let i = 0; i < tracker.count; i++) {
            const tileId = active[i];
            tracker.touched[tileId] = 0;
        }
        tracker.touchedCount = 0;
    }

    _markTileByCoord(tileTracker, tx, ty) {
        if (!Number.isFinite(tx) || !Number.isFinite(ty)) return;
        const clampedTx = Math.max(0, Math.min(this.activeTileCols - 1, Math.floor(tx)));
        const clampedTy = Math.max(0, Math.min(this.activeTileRows - 1, Math.floor(ty)));
        const tileId = this._tileKey(clampedTx, clampedTy);
        this._trackerActivate(tileTracker, tileId);
    }

    _markTileAroundCell(tileTracker, gx, gy) {
        const tx = Math.floor((Math.max(0, Math.min(this.size - 1, Math.floor(gx))) / this.activeTileSize));
        const ty = Math.floor((Math.max(0, Math.min(this.size - 1, Math.floor(gy))) / this.activeTileSize));
        for (let oy = -this.activeTileHalo; oy <= this.activeTileHalo; oy++) {
            for (let ox = -this.activeTileHalo; ox <= this.activeTileHalo; ox++) {
                this._markTileByCoord(tileTracker, tx + ox, ty + oy);
            }
        }
    }

    /**
     * Mark carrier activity around a fluid-grid cell (gx,gy), including halo tiles.
     */
    markCarrierCell(gx, gy) {
        this._markTileAroundCell(this.carrierTiles, gx, gy);
    }

    /**
     * Mark momentum-only activity around a fluid-grid cell (gx,gy), including halo tiles.
     */
    markMomentumCell(gx, gy) {
        this._markTileAroundCell(this.momentumTiles, gx, gy);
    }

    /**
     * Seed carrier activity around creature centers before each fluid step.
     */
    seedCarrierTilesFromBodies(bodies) {
        if (!Array.isArray(bodies) || bodies.length === 0) return;
        for (const b of bodies) {
            if (!b || b.isUnstable) continue;
            const c = typeof b.getAveragePosition === 'function' ? b.getAveragePosition() : null;
            if (!c) continue;
            const gx = Math.floor((Number(c.x) || 0) / Math.max(1e-6, this.scaleX));
            const gy = Math.floor((Number(c.y) || 0) / Math.max(1e-6, this.scaleY));
            this.markCarrierCell(gx, gy);
        }
    }

    /**
     * Seed carrier activity around particles before each fluid step.
     */
    seedCarrierTilesFromParticles(particles) {
        if (!Array.isArray(particles) || particles.length === 0) return;
        for (const p of particles) {
            if (!p || p.life <= 0 || !p.pos) continue;
            const gx = Math.floor((Number(p.pos.x) || 0) / Math.max(1e-6, this.scaleX));
            const gy = Math.floor((Number(p.pos.y) || 0) / Math.max(1e-6, this.scaleY));
            this.markCarrierCell(gx, gy);
        }
    }

    /**
     * Mark momentum tiles from the evolved velocity field so "invisible currents" are represented
     * even when no fresh addVelocity() impulse happened this tick.
     */
    seedMomentumTilesFromVelocityField() {
        const threshold = Math.max(0, Number(config.FLUID_MOMENTUM_ACTIVITY_SPEED_THRESHOLD) || 0);
        if (threshold <= 0) return;
        const thresholdSq = threshold * threshold;
        const N = this.size;
        const NMinus1 = N - 1;
        for (let y = 1; y < NMinus1; y++) {
            let idx = y * N + 1;
            const rowEnd = y * N + NMinus1;
            while (idx < rowEnd) {
                const vx = this.Vx[idx];
                const vy = this.Vy[idx];
                if ((vx * vx + vy * vy) >= thresholdSq) {
                    const gx = idx % N;
                    this.markMomentumCell(gx, y);
                }
                idx++;
            }
        }
    }

    _decayTileMap(tileTracker) {
        let sleeping = 0;
        let i = 0;
        while (i < tileTracker.count) {
            const tileId = tileTracker.active[i];
            const ttl = tileTracker.ttl[tileId];
            if (ttl <= 1) {
                tileTracker.ttl[tileId] = 0;
                tileTracker.index[tileId] = -1;
                const lastIdx = tileTracker.count - 1;
                const lastTileId = tileTracker.active[lastIdx];
                tileTracker.active[i] = lastTileId;
                tileTracker.index[lastTileId] = i;
                tileTracker.count = lastIdx;
                sleeping += 1;
                continue;
            }
            tileTracker.ttl[tileId] = ttl - 1;
            i += 1;
        }
        return sleeping;
    }

    /**
     * Phase A scaffold for coarse momentum-only processing:
     * group momentum-non-carrier tiles into 2x2 macro blocks and report coverage telemetry.
     */
    _collectCoarseMomentumBlocks() {
        const blockSize = Math.max(2, Math.floor(Number(this.coarseMomentumBlockSize) || 2));
        const blocks = new Set();
        for (let i = 0; i < this.momentumTiles.count; i++) {
            const key = this.momentumTiles.active[i];
            if (this._trackerHas(this.carrierTiles, key)) continue;
            const tx = this._tileTx(key);
            const ty = this._tileTy(key);
            const bx = Math.floor(tx / blockSize);
            const by = Math.floor(ty / blockSize);
            blocks.add(this._tileKey(bx, by));
        }

        const coarseMomentumBlockCount = blocks.size;
        const totalTiles = Math.max(1, this.activeTileCols * this.activeTileRows);
        const approxCoveredTiles = Math.min(totalTiles, coarseMomentumBlockCount * blockSize * blockSize);
        return {
            coarseMomentumBlockSize: blockSize,
            coarseMomentumBlockCount,
            coarseMomentumCoveragePct: approxCoveredTiles / totalTiles
        };
    }

    _finalizeActiveTileTelemetry() {
        const sleepingCarrierTiles = this._decayTileMap(this.carrierTiles);
        const sleepingMomentumTiles = this._decayTileMap(this.momentumTiles);

        const totalTiles = Math.max(1, this.activeTileCols * this.activeTileRows);
        const carrierActiveTiles = this.carrierTiles.count;
        const momentumTilesTotal = this.momentumTiles.count;
        let momentumTilesNonCarrier = 0;
        for (let i = 0; i < this.momentumTiles.count; i++) {
            const key = this.momentumTiles.active[i];
            if (!this._trackerHas(this.carrierTiles, key)) momentumTilesNonCarrier++;
        }
        const coarse = this._collectCoarseMomentumBlocks();
        this.lastActiveTileTelemetry = {
            carrierActiveTiles,
            momentumTilesTotal,
            momentumTilesNonCarrier,
            coarseMomentumBlockSize: coarse.coarseMomentumBlockSize,
            coarseMomentumBlockCount: coarse.coarseMomentumBlockCount,
            coarseMomentumCoveragePct: coarse.coarseMomentumCoveragePct,
            totalTiles,
            carrierPct: carrierActiveTiles / totalTiles,
            momentumPct: momentumTilesTotal / totalTiles,
            momentumNonCarrierPct: momentumTilesNonCarrier / totalTiles,
            carrierTouchedTiles: this.carrierTiles.touchedCount,
            momentumTouchedTiles: this.momentumTiles.touchedCount,
            sleepingCarrierTiles,
            sleepingMomentumTiles
        };
        this._trackerClearTouched(this.carrierTiles);
        this._trackerClearTouched(this.momentumTiles);
    }

    getActiveTileTelemetry() {
        return { ...this.lastActiveTileTelemetry };
    }

    getLastStepPerf() {
        return { ...this.lastStepPerf };
    }

    getActiveTileDebugCells(maxCells = 30000) {
        const out = [];
        const limit = Math.max(0, Math.floor(Number(maxCells) || 0));
        const seen = new Set();

        for (let i = 0; i < this.carrierTiles.count; i++) {
            if (limit > 0 && out.length >= limit) break;
            const key = this.carrierTiles.active[i];
            const tx = this._tileTx(key);
            const ty = this._tileTy(key);
            seen.add(key);
            out.push({ tx, ty, kind: 'carrier' });
        }

        for (let i = 0; i < this.momentumTiles.count; i++) {
            if (limit > 0 && out.length >= limit) break;
            const key = this.momentumTiles.active[i];
            if (seen.has(key)) continue;
            const tx = this._tileTx(key);
            const ty = this._tileTy(key);
            out.push({ tx, ty, kind: 'momentumOnly' });
        }

        return {
            tileSizeCells: this.activeTileSize,
            cells: out,
            truncated: limit > 0 && out.length >= limit
        };
    }

    addDensity(x, y, emitterR, emitterG, emitterB, emissionStrength) {
        const idx = this.IX(x, y);
        this.markCarrierCell(x, y);
        const normalizedEmissionEffect = (emissionStrength / 50.0) * config.DYE_PULL_RATE;
        this.densityR[idx] = Math.max(0, Math.min(255, this.densityR[idx] + (emitterR - this.densityR[idx]) * normalizedEmissionEffect));
        this.densityG[idx] = Math.max(0, Math.min(255, this.densityG[idx] + (emitterG - this.densityG[idx]) * normalizedEmissionEffect));
        this.densityB[idx] = Math.max(0, Math.min(255, this.densityB[idx] + (emitterB - this.densityB[idx]) * normalizedEmissionEffect));
    }

    addVelocity(x, y, amountX, amountY) {
        const idx = this.IX(x, y);
        this.markMomentumCell(x, y);
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

    _normalizeDomain(domainLike) {
        const N = this.size;
        const maxCell = N - 2;
        if (domainLike && domainLike.rowSpans instanceof Map) return domainLike;

        const xMin = domainLike ? Math.max(1, Math.min(maxCell, Math.floor(domainLike.xMin))) : 1;
        const xMax = domainLike ? Math.max(xMin, Math.min(maxCell, Math.floor(domainLike.xMax))) : maxCell;
        const yMin = domainLike ? Math.max(1, Math.min(maxCell, Math.floor(domainLike.yMin))) : 1;
        const yMax = domainLike ? Math.max(yMin, Math.min(maxCell, Math.floor(domainLike.yMax))) : maxCell;
        const rowSpans = new Map();
        for (let y = yMin; y <= yMax; y++) rowSpans.set(y, [[xMin, xMax]]);
        return { rowSpans, xMin, xMax, yMin, yMax, _compiledRows: null };
    }

    _compileDomain(domainLike) {
        const domain = this._normalizeDomain(domainLike);
        if (Array.isArray(domain._compiledRows)) return domain;
        const rows = [];
        for (const [y, spans] of domain.rowSpans.entries()) rows.push([y, spans]);
        domain._compiledRows = rows;
        return domain;
    }

    _forEachDomainSpan(domainLike, fn) {
        const domain = this._compileDomain(domainLike);
        for (let r = 0; r < domain._compiledRows.length; r++) {
            const [y, spans] = domain._compiledRows[r];
            for (let s = 0; s < spans.length; s++) {
                const span = spans[s];
                fn(y, span[0], span[1]);
            }
        }
    }

    lin_solve(b, x, x0, a_global_param, c_global_param, field_type, base_diff_rate, dt_param, bounds = null) {
        const N = this.size;
        const domain = this._compileDomain(bounds);
        const cRecipGlobal = 1.0 / c_global_param;
        const solverIterations = this._getSolverIterationsForField(field_type);

        const isVelocityField = (field_type === 'velX' || field_type === 'velY');
        const viscosityField = (isVelocityField && this.viscosityField) ? this.viscosityField : null;

        if (!viscosityField) {
            for (let k_iter = 0; k_iter < solverIterations; k_iter++) {
                for (let r = 0; r < domain._compiledRows.length; r++) {
                    const [j, spans] = domain._compiledRows[r];
                    for (let s = 0; s < spans.length; s++) {
                        const span = spans[s];
                        let idx = j * N + span[0];
                        const rowEnd = j * N + span[1] + 1;
                        while (idx < rowEnd) {
                            x[idx] = (x0[idx] + a_global_param * (x[idx + 1] + x[idx - 1] + x[idx + N] + x[idx - N])) * cRecipGlobal;
                            idx++;
                        }
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
        for (let r = 0; r < domain._compiledRows.length; r++) {
            const [j, spans] = domain._compiledRows[r];
            for (let s = 0; s < spans.length; s++) {
                const span = spans[s];
                let idx = j * N + span[0];
                const rowEnd = j * N + span[1] + 1;
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
        }

        for (let k_iter = 0; k_iter < solverIterations; k_iter++) {
            for (let r = 0; r < domain._compiledRows.length; r++) {
                const [j, spans] = domain._compiledRows[r];
                for (let s = 0; s < spans.length; s++) {
                    const span = spans[s];
                    let idx = j * N + span[0];
                    const rowEnd = j * N + span[1] + 1;
                    while (idx < rowEnd) {
                        const effectiveA = effectiveAByIdx[idx];
                        const effectiveCRecip = effectiveCRecipByIdx[idx];
                        x[idx] = (x0[idx] + effectiveA * (x[idx + 1] + x[idx - 1] + x[idx + N] + x[idx - N])) * effectiveCRecip;
                        idx++;
                    }
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
        const domain = this._compileDomain(bounds);

        for (let r = 0; r < domain._compiledRows.length; r++) {
            const [j, spans] = domain._compiledRows[r];
            for (let s = 0; s < spans.length; s++) {
                const span = spans[s];
                let idx = j * N + span[0];
                const rowEnd = j * N + span[1] + 1;
                while (idx < rowEnd) {
                    div_temp[idx] = -0.5 * (velocX_in_out[idx + 1] - velocX_in_out[idx - 1] + velocY_in_out[idx + N] - velocY_in_out[idx - N]) / N;
                    p_temp[idx] = 0;
                    idx++;
                }
            }
        }

        this.set_bnd(0, div_temp);
        this.set_bnd(0, p_temp);
        this.lin_solve(0, p_temp, div_temp, 1, 4, 'pressure', 0, 0, bounds);

        for (let r = 0; r < domain._compiledRows.length; r++) {
            const [j, spans] = domain._compiledRows[r];
            for (let s = 0; s < spans.length; s++) {
                const span = spans[s];
                let idx = j * N + span[0];
                const rowEnd = j * N + span[1] + 1;
                while (idx < rowEnd) {
                    velocX_in_out[idx] -= 0.5 * (p_temp[idx + 1] - p_temp[idx - 1]) * N;
                    velocY_in_out[idx] -= 0.5 * (p_temp[idx + N] - p_temp[idx - N]) * N;
                    idx++;
                }
            }
        }

        this.set_bnd(1, velocX_in_out);
        this.set_bnd(2, velocY_in_out);
    }

    advect(b, d_out, d_in, velocX_source, velocY_source, dt, bounds = null) {
        const N = this.size;
        const domain = this._compileDomain(bounds);
        const dtx_scaled = dt * N;
        const dty_scaled = dt * N;

        for (let r = 0; r < domain._compiledRows.length; r++) {
            const [j_cell, spans] = domain._compiledRows[r];
            for (let s = 0; s < spans.length; s++) {
                const span = spans[s];
                let current_idx = j_cell * N + span[0];
                const rowEnd = j_cell * N + span[1] + 1;

                for (let i_cell = span[0]; current_idx < rowEnd; i_cell++, current_idx++) {
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
     * Build a sparse cell-domain from active tiles (row -> merged [x0,x1] spans).
     */
    _buildSparseDomainFromTiles(tileMap) {
        if (!tileMap) return null;
        const hasTracker = Number.isFinite(tileMap.count) && tileMap.active;
        if (hasTracker && tileMap.count <= 0) return null;
        if (!hasTracker && tileMap.size === 0) return null;

        const maxCell = this.size - 2;
        const pad = Math.max(1, this.activeTileSize);
        const rows = new Map();

        if (hasTracker) {
            for (let i = 0; i < tileMap.count; i++) {
                const key = tileMap.active[i];
                const tx = this._tileTx(key);
                const ty = this._tileTy(key);

                const x0 = Math.max(1, (tx * this.activeTileSize) - pad);
                const x1 = Math.min(maxCell, ((tx + 1) * this.activeTileSize) + pad);
                const y0 = Math.max(1, (ty * this.activeTileSize) - pad);
                const y1 = Math.min(maxCell, ((ty + 1) * this.activeTileSize) + pad);

                for (let y = y0; y <= y1; y++) {
                    const list = rows.get(y) || [];
                    list.push([x0, x1]);
                    rows.set(y, list);
                }
            }
        } else {
            for (const key of tileMap.keys()) {
                let tx;
                let ty;
                if (typeof key === 'string') {
                    const parts = key.split(':');
                    if (parts.length !== 2) continue;
                    tx = Math.floor(Number(parts[0]));
                    ty = Math.floor(Number(parts[1]));
                    if (!Number.isFinite(tx) || !Number.isFinite(ty)) continue;
                } else {
                    tx = this._tileTx(Number(key));
                    ty = this._tileTy(Number(key));
                }

                const x0 = Math.max(1, (tx * this.activeTileSize) - pad);
                const x1 = Math.min(maxCell, ((tx + 1) * this.activeTileSize) + pad);
                const y0 = Math.max(1, (ty * this.activeTileSize) - pad);
                const y1 = Math.min(maxCell, ((ty + 1) * this.activeTileSize) + pad);

                for (let y = y0; y <= y1; y++) {
                    const list = rows.get(y) || [];
                    list.push([x0, x1]);
                    rows.set(y, list);
                }
            }
        }

        if (rows.size === 0) return null;

        let xMin = Infinity;
        let yMin = Infinity;
        let xMax = -Infinity;
        let yMax = -Infinity;
        const rowSpans = new Map();

        for (const [y, spans] of rows.entries()) {
            spans.sort((a, b) => a[0] - b[0]);
            const merged = [];
            for (const span of spans) {
                const last = merged[merged.length - 1];
                if (!last || span[0] > (last[1] + 1)) merged.push([span[0], span[1]]);
                else last[1] = Math.max(last[1], span[1]);
            }
            rowSpans.set(y, merged);
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
            for (const [sx, ex] of merged) {
                if (sx < xMin) xMin = sx;
                if (ex > xMax) xMax = ex;
            }
        }

        if (!Number.isFinite(xMin) || !Number.isFinite(yMin) || !Number.isFinite(xMax) || !Number.isFinite(yMax)) {
            return null;
        }

        return { rowSpans, xMin, yMin, xMax, yMax };
    }

    _mergeDomains(a, b) {
        if (!a) return b || null;
        if (!b) return a || null;
        const rows = new Map();

        const pushSpans = (domain) => {
            for (const [y, spans] of domain.rowSpans.entries()) {
                const list = rows.get(y) || [];
                for (const s of spans) list.push([s[0], s[1]]);
                rows.set(y, list);
            }
        };
        pushSpans(this._normalizeDomain(a));
        pushSpans(this._normalizeDomain(b));

        const mergedDomain = { rowSpans: new Map(), xMin: Infinity, yMin: Infinity, xMax: -Infinity, yMax: -Infinity };
        for (const [y, spans] of rows.entries()) {
            spans.sort((s1, s2) => s1[0] - s2[0]);
            const merged = [];
            for (const span of spans) {
                const last = merged[merged.length - 1];
                if (!last || span[0] > (last[1] + 1)) merged.push([span[0], span[1]]);
                else last[1] = Math.max(last[1], span[1]);
            }
            mergedDomain.rowSpans.set(y, merged);
            mergedDomain.yMin = Math.min(mergedDomain.yMin, y);
            mergedDomain.yMax = Math.max(mergedDomain.yMax, y);
            for (const [sx, ex] of merged) {
                mergedDomain.xMin = Math.min(mergedDomain.xMin, sx);
                mergedDomain.xMax = Math.max(mergedDomain.xMax, ex);
            }
        }
        return mergedDomain;
    }

    _expandTileMap(tileMap, haloTiles = 1) {
        const halo = Math.max(0, Math.floor(Number(haloTiles) || 0));
        if (!tileMap) return tileMap;
        if (halo <= 0) return tileMap;

        const active = [];
        const mark = new Uint8Array(this.totalActiveTiles);

        const pushExpanded = (tileId) => {
            const tx = this._tileTx(tileId);
            const ty = this._tileTy(tileId);
            for (let oy = -halo; oy <= halo; oy++) {
                for (let ox = -halo; ox <= halo; ox++) {
                    const nx = Math.max(0, Math.min(this.activeTileCols - 1, tx + ox));
                    const ny = Math.max(0, Math.min(this.activeTileRows - 1, ty + oy));
                    const id = this._tileKey(nx, ny);
                    if (mark[id] === 0) {
                        mark[id] = 1;
                        active.push(id);
                    }
                }
            }
        };

        if (Number.isFinite(tileMap.count) && tileMap.active) {
            for (let i = 0; i < tileMap.count; i++) pushExpanded(tileMap.active[i]);
        } else if (typeof tileMap.keys === 'function') {
            for (const key of tileMap.keys()) pushExpanded(Number(key));
        }

        return { count: active.length, active };
    }

    _subtractTileMaps(source, exclude) {
        const active = [];
        if (!source) return { count: 0, active };

        if (Number.isFinite(source.count) && source.active) {
            for (let i = 0; i < source.count; i++) {
                const key = source.active[i];
                if (exclude && Number.isFinite(exclude.count) && exclude.ttl && exclude.ttl[key] > 0) continue;
                active.push(key);
            }
        } else if (typeof source.keys === 'function') {
            for (const key of source.keys()) {
                const k = Number(key);
                if (exclude && Number.isFinite(exclude.count) && exclude.ttl && exclude.ttl[k] > 0) continue;
                active.push(k);
            }
        }
        return { count: active.length, active };
    }

    _buildDeepEmptyTileMap(tier1Tiles, tier2Tiles) {
        const active = [];
        const tier1Ttl = tier1Tiles?.ttl;
        const tier2Mark = new Uint8Array(this.totalActiveTiles);

        if (tier2Tiles && Number.isFinite(tier2Tiles.count) && tier2Tiles.active) {
            for (let i = 0; i < tier2Tiles.count; i++) tier2Mark[tier2Tiles.active[i]] = 1;
        }

        for (let tileId = 0; tileId < this.totalActiveTiles; tileId++) {
            if (tier1Ttl && tier1Ttl[tileId] > 0) continue;
            if (tier2Mark[tileId] > 0) continue;
            active.push(tileId);
        }
        return { count: active.length, active };
    }

    step(worldTick = 0) {
        const tStart = Date.now();
        let t0 = Date.now();
        this.seedMomentumTilesFromVelocityField();
        const seedMomentumMs = Date.now() - t0;

        const momentumEvery = Math.max(1, Math.floor(Number(config.FLUID_MOMENTUM_ONLY_STEP_EVERY_N_TICKS) || 10));
        const emptyEvery = Math.max(1, Math.floor(Number(config.FLUID_EMPTY_STEP_EVERY_N_TICKS) || 24));
        const tick = Math.max(0, Math.floor(Number(worldTick) || 0));
        const allowMomentumSolve = (tick % momentumEvery) === 0;
        const allowEmptySolve = (tick % emptyEvery) === 0;

        // Build one active tile domain per tick, then compile spans once.
        const activeMark = new Uint8Array(this.totalActiveTiles);
        const activeTiles = [];
        const addActiveTile = (tileId) => {
            if (!Number.isFinite(tileId)) return;
            const id = Math.floor(tileId);
            if (id < 0 || id >= this.totalActiveTiles) return;
            if (activeMark[id] !== 0) return;
            activeMark[id] = 1;
            activeTiles.push(id);
        };

        for (let i = 0; i < this.carrierTiles.count; i++) addActiveTile(this.carrierTiles.active[i]);

        let expandedMomentumTiles = null;
        if (allowMomentumSolve || allowEmptySolve) {
            const momentumNonCarrierTiles = this._subtractTileMaps(this.momentumTiles, this.carrierTiles);
            expandedMomentumTiles = this._expandTileMap(momentumNonCarrierTiles, 1);
            if (allowMomentumSolve && expandedMomentumTiles && expandedMomentumTiles.count > 0) {
                for (let i = 0; i < expandedMomentumTiles.count; i++) addActiveTile(expandedMomentumTiles.active[i]);
            }
        }

        if (allowEmptySolve) {
            const deepEmptyTiles = this._buildDeepEmptyTileMap(this.carrierTiles, expandedMomentumTiles || { count: 0, active: [] });
            for (let i = 0; i < deepEmptyTiles.count; i++) addActiveTile(deepEmptyTiles.active[i]);
        }

        let bounds = this._buildSparseDomainFromTiles({ count: activeTiles.length, active: activeTiles });

        if (!bounds) {
            this.lastStepPerf = {
                totalMs: Date.now() - tStart,
                seedMomentumMs,
                diffuseVelMs: 0,
                projectMs: 0,
                advectVelMs: 0,
                diffuseDensityMs: 0,
                advectDensityMs: 0,
                fadeMs: 0
            };
            this._finalizeActiveTileTelemetry();
            return;
        }

        bounds = this._compileDomain(bounds);

        t0 = Date.now();
        this.diffuse(1, this.Vx0, this.Vx, this.viscosity, this.dt, 'velX', bounds);
        this.diffuse(2, this.Vy0, this.Vy, this.viscosity, this.dt, 'velY', bounds);
        const diffuseVelMs = Date.now() - t0;

        this.clampVelocityComponents(this.Vx0);
        this.clampVelocityComponents(this.Vy0);

        t0 = Date.now();
        this.project(this.Vx0, this.Vy0, this.Vx, this.Vy, bounds);
        this.project(this.Vx, this.Vy, this.Vx0, this.Vy0, bounds);
        const projectMs = Date.now() - t0;

        t0 = Date.now();
        this.advect(1, this.Vx, this.Vx0, this.Vx0, this.Vy0, this.dt, bounds);
        this.advect(2, this.Vy, this.Vy0, this.Vx0, this.Vy0, this.dt, bounds);
        const advectVelMs = Date.now() - t0;

        this.clampVelocityComponents(this.Vx);
        this.clampVelocityComponents(this.Vy);

        t0 = Date.now();
        this.diffuse(0, this.densityR0, this.densityR, this.diffusion, this.dt, 'density', bounds);
        this.diffuse(0, this.densityG0, this.densityG, this.diffusion, this.dt, 'density', bounds);
        this.diffuse(0, this.densityB0, this.densityB, this.diffusion, this.dt, 'density', bounds);
        const diffuseDensityMs = Date.now() - t0;

        t0 = Date.now();
        this.advect(0, this.densityR, this.densityR0, this.Vx, this.Vy, this.dt, bounds);
        this.advect(0, this.densityG, this.densityG0, this.Vx, this.Vy, this.dt, bounds);
        this.advect(0, this.densityB, this.densityB0, this.Vx, this.Vy, this.dt, bounds);
        const advectDensityMs = Date.now() - t0;

        t0 = Date.now();
        if (bounds) {
            this._forEachDomainSpan(bounds, (y, xStart, xEnd) => {
                let idx = y * this.size + xStart;
                const rowEnd = y * this.size + xEnd + 1;
                while (idx < rowEnd) {
                    this.densityR[idx] = Math.max(0, this.densityR[idx] - config.FLUID_FADE_RATE * 255 * this.dt);
                    this.densityG[idx] = Math.max(0, this.densityG[idx] - config.FLUID_FADE_RATE * 255 * this.dt);
                    this.densityB[idx] = Math.max(0, this.densityB[idx] - config.FLUID_FADE_RATE * 255 * this.dt);
                    idx++;
                }
            });
        } else {
            for (let i = 0; i < this.densityR.length; i++) {
                this.densityR[i] = Math.max(0, this.densityR[i] - config.FLUID_FADE_RATE * 255 * this.dt);
                this.densityG[i] = Math.max(0, this.densityG[i] - config.FLUID_FADE_RATE * 255 * this.dt);
                this.densityB[i] = Math.max(0, this.densityB[i] - config.FLUID_FADE_RATE * 255 * this.dt);
            }
        }
        const fadeMs = Date.now() - t0;

        this.lastStepPerf = {
            totalMs: Date.now() - tStart,
            seedMomentumMs,
            diffuseVelMs,
            projectMs,
            advectVelMs,
            diffuseDensityMs,
            advectDensityMs,
            fadeMs
        };

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
