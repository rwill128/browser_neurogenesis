class FluidField {
    constructor(size, diffusion, viscosity, dt, scaleX, scaleY) {
        this.size = Math.round(size);
        this.dt = dt;
        this.diffusion = diffusion;
        this.viscosity = viscosity;
        this.scaleX = scaleX; 
        this.scaleY = scaleY; 
        this.useWrapping = false;
        this.maxVelComponent = MAX_FLUID_VELOCITY_COMPONENT;

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

        this.iterations = 4; 
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

    addDensity(x, y, emitterR, emitterG, emitterB, emissionStrength) {
        const idx = this.IX(x, y);
        const normalizedEmissionEffect = (emissionStrength / 50.0) * DYE_PULL_RATE;
        this.densityR[idx] = Math.max(0, Math.min(255, this.densityR[idx] + (emitterR - this.densityR[idx]) * normalizedEmissionEffect));
        this.densityG[idx] = Math.max(0, Math.min(255, this.densityG[idx] + (emitterG - this.densityG[idx]) * normalizedEmissionEffect));
        this.densityB[idx] = Math.max(0, Math.min(255, this.densityB[idx] + (emitterB - this.densityB[idx]) * normalizedEmissionEffect));
    }

    addVelocity(x, y, amountX, amountY) {
        const idx = this.IX(x, y);
        this.Vx[idx] = Math.max(-this.maxVelComponent, Math.min(this.Vx[idx] + amountX, this.maxVelComponent));
        this.Vy[idx] = Math.max(-this.maxVelComponent, Math.min(this.Vy[idx] + amountY, this.maxVelComponent));
    }

    clampVelocityComponents(arr) {
        for(let i=0; i < arr.length; i++) {
            arr[i] = Math.max(-this.maxVelComponent, Math.min(arr[i], this.maxVelComponent));
        }
    }

    lin_solve(b, x, x0, a_global_param, c_global_param, field_type, base_diff_rate, dt_param) {
        const cRecipGlobal = 1.0 / c_global_param;
        for (let k_iter = 0; k_iter < this.iterations; k_iter++) {
            for (let j = 1; j < this.size - 1; j++) {
                for (let i = 1; i < this.size - 1; i++) {
                    const idx = this.IX(i,j);
                    let effective_a = a_global_param;
                    let effective_cRecip = cRecipGlobal;

                    if ((field_type === 'velX' || field_type === 'velY')) {
                        if (viscosityField && viscosityField[idx] !== undefined) { 
                            const localViscosityMultiplier = Math.max(MIN_VISCOSITY_MULTIPLIER, Math.min(viscosityField[idx], MAX_VISCOSITY_MULTIPLIER));
                            const cell_specific_diff_rate = base_diff_rate * localViscosityMultiplier; 
                            const temp_effective_a = dt_param * cell_specific_diff_rate * (this.size - 2) * (this.size - 2);
                            const temp_denominator_c = 1 + 4 * temp_effective_a;
                            if (temp_denominator_c !== 0 && !isNaN(temp_effective_a) && isFinite(temp_effective_a)) {
                                effective_a = temp_effective_a;
                                effective_cRecip = 1.0 / temp_denominator_c;
                            } 
                        }
                    }
                    x[idx] = (x0[idx] + effective_a * (x[this.IX(i+1,j)] + x[this.IX(i-1,j)] + x[this.IX(i,j+1)] + x[this.IX(i,j-1)])) * effective_cRecip;
                }
            }
            this.set_bnd(b, x);
        }
    }

    diffuse(b, x_out, x_in, base_diff_rate, dt, field_type = 'density') {
        const a_global = dt * base_diff_rate * (this.size - 2) * (this.size - 2);
        this.lin_solve(b, x_out, x_in, a_global, 1 + 4 * a_global, field_type, base_diff_rate, dt);
    }

    project(velocX_in_out, velocY_in_out, p_temp, div_temp) {
        for (let j = 1; j < this.size - 1; j++) {
            for (let i = 1; i < this.size - 1; i++) {
                const idx = this.IX(i,j);
                div_temp[idx] = -0.5 * (velocX_in_out[this.IX(i+1,j)] - velocX_in_out[this.IX(i-1,j)] + velocY_in_out[this.IX(i,j+1)] - velocY_in_out[this.IX(i,j-1)]) / this.size;
                p_temp[idx] = 0;
            }
        }
        this.set_bnd(0, div_temp);
        this.set_bnd(0, p_temp);
        this.lin_solve(0, p_temp, div_temp, 1, 4, 'pressure', 0, 0);

        for (let j = 1; j < this.size - 1; j++) {
            for (let i = 1; i < this.size - 1; i++) {
                const idx = this.IX(i,j);
                velocX_in_out[idx] -= 0.5 * (p_temp[this.IX(i+1,j)] - p_temp[this.IX(i-1,j)]) * this.size;
                velocY_in_out[idx] -= 0.5 * (p_temp[this.IX(i,j+1)] - p_temp[this.IX(i,j-1)]) * this.size;
            }
        }
        this.set_bnd(1, velocX_in_out);
        this.set_bnd(2, velocY_in_out);
    }

    advect(b, d_out, d_in, velocX_source, velocY_source, dt) {
        let i0, i1, j0, j1;
        const N = this.size;
        const dtx_scaled = dt * N; 
        const dty_scaled = dt * N;

        let s0, s1, t0, t1;
        let x, y;

        for (let j_cell = 1; j_cell < N - 1; j_cell++) {
            for (let i_cell = 1; i_cell < N - 1; i_cell++) {
                const current_idx = this.IX(i_cell, j_cell);
                x = i_cell - (dtx_scaled * velocX_source[current_idx]); 
                y = j_cell - (dty_scaled * velocY_source[current_idx]);

                if (this.useWrapping) {
                    x = (x % N + N) % N;
                    y = (y % N + N) % N;
                    i0 = Math.floor(x);
                    j0 = Math.floor(y);
                    i1 = (i0 + 1) % N;
                    j1 = (j0 + 1) % N;
                } else {
                    if (x < 0.5) x = 0.5;
                    if (x > N - 1.5) x = N - 1.5; 
                    i0 = Math.floor(x);
                    i1 = i0 + 1;
                    if (y < 0.5) y = 0.5;
                    if (y > N - 1.5) y = N - 1.5;
                    j0 = Math.floor(y);
                    j1 = j0 + 1;
                }

                s1 = x - i0;
                s0 = 1.0 - s1;
                t1 = y - j0;
                t0 = 1.0 - t1;
                
                d_out[current_idx] = s0 * (t0 * d_in[this.IX(i0,j0)] + t1 * d_in[this.IX(i0,j1)]) +
                                     s1 * (t0 * d_in[this.IX(i1,j0)] + t1 * d_in[this.IX(i1,j1)]);
            }
        }
        this.set_bnd(b, d_out);
    }

    set_bnd(b, x_arr) {
        if (this.useWrapping) {
            for (let i = 1; i < this.size - 1; i++) {
                x_arr[this.IX(i, 0)] = x_arr[this.IX(i, this.size - 2)];
                x_arr[this.IX(i, this.size - 1)] = x_arr[this.IX(i, 1)];
            }
            for (let j = 1; j < this.size - 1; j++) {
                x_arr[this.IX(0, j)] = x_arr[this.IX(this.size - 2, j)];
                x_arr[this.IX(this.size - 1, j)] = x_arr[this.IX(1, j)];
            }
            x_arr[this.IX(0, 0)] = 0.5 * (x_arr[this.IX(1, 0)] + x_arr[this.IX(0, 1)]);
            x_arr[this.IX(0, this.size - 1)] = 0.5 * (x_arr[this.IX(1, this.size - 1)] + x_arr[this.IX(0, this.size - 2)]);
            x_arr[this.IX(this.size - 1, 0)] = 0.5 * (x_arr[this.IX(this.size - 2, 0)] + x_arr[this.IX(this.size - 1, 1)]);
            x_arr[this.IX(this.size - 1, this.size - 1)] = 0.5 * (x_arr[this.IX(this.size - 2, this.size - 1)] + x_arr[this.IX(this.size - 1, this.size - 2)]);
        } else {
            for (let i = 1; i < this.size - 1; i++) {
                x_arr[this.IX(i, 0)] = b === 2 ? -x_arr[this.IX(i, 1)] : x_arr[this.IX(i, 1)];
                x_arr[this.IX(i, this.size - 1)] = b === 2 ? -x_arr[this.IX(i, this.size - 2)] : x_arr[this.IX(i, this.size - 2)];
            }
            for (let j = 1; j < this.size - 1; j++) {
                x_arr[this.IX(0, j)] = b === 1 ? -x_arr[this.IX(1, j)] : x_arr[this.IX(1, j)];
                x_arr[this.IX(this.size - 1, j)] = b === 1 ? -x_arr[this.IX(this.size - 2, j)] : x_arr[this.IX(this.size - 2, j)];
            }
            x_arr[this.IX(0, 0)] = 0.5 * (x_arr[this.IX(1, 0)] + x_arr[this.IX(0, 1)]);
            x_arr[this.IX(0, this.size - 1)] = 0.5 * (x_arr[this.IX(1, this.size - 1)] + x_arr[this.IX(0, this.size - 2)]);
            x_arr[this.IX(this.size - 1, 0)] = 0.5 * (x_arr[this.IX(this.size - 2, 0)] + x_arr[this.IX(this.size - 1, 1)]);
            x_arr[this.IX(this.size - 1, this.size - 1)] = 0.5 * (x_arr[this.IX(this.size - 2, this.size - 1)] + x_arr[this.IX(this.size - 1, this.size - 2)]);
        }
    }

    step() {
        this.diffuse(1, this.Vx0, this.Vx, this.viscosity, this.dt, 'velX');
        this.diffuse(2, this.Vy0, this.Vy, this.viscosity, this.dt, 'velY');
        this.clampVelocityComponents(this.Vx0);
        this.clampVelocityComponents(this.Vy0);
        this.project(this.Vx0, this.Vy0, this.Vx, this.Vy);
        this.advect(1, this.Vx, this.Vx0, this.Vx0, this.Vy0, this.dt);
        this.advect(2, this.Vy, this.Vy0, this.Vx0, this.Vy0, this.dt);
        this.clampVelocityComponents(this.Vx);
        this.clampVelocityComponents(this.Vy);
        this.project(this.Vx, this.Vy, this.Vx0, this.Vy0);
        this.diffuse(0, this.densityR0, this.densityR, this.diffusion, this.dt, 'density');
        this.diffuse(0, this.densityG0, this.densityG, this.diffusion, this.dt, 'density');
        this.diffuse(0, this.densityB0, this.densityB, this.diffusion, this.dt, 'density');
        this.advect(0, this.densityR, this.densityR0, this.Vx, this.Vy, this.dt);
        this.advect(0, this.densityG, this.densityG0, this.Vx, this.Vy, this.dt);
        this.advect(0, this.densityB, this.densityB0, this.Vx, this.Vy, this.dt);
        for (let i = 0; i < this.densityR.length; i++) {
            this.densityR[i] = Math.max(0, this.densityR[i] - FLUID_FADE_RATE * 255 * this.dt);
            this.densityG[i] = Math.max(0, this.densityG[i] - FLUID_FADE_RATE * 255 * this.dt);
            this.densityB[i] = Math.max(0, this.densityB[i] - FLUID_FADE_RATE * 255 * this.dt);
        }
    }

    draw(ctxToDrawOn, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
        console.log("Drawing with CPU FluidField"); // Ensure this log is active
        const N = Math.round(this.size);
        if (N <= 0 || !Number.isFinite(N)) {
            console.error("FluidField.draw: Invalid N size:", N);
            return;
        }

        const worldCellWidth = WORLD_WIDTH / N;
        const worldCellHeight = WORLD_HEIGHT / N;

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
} 