class GPUFluidField {
    constructor(canvas, size, diffusion, viscosity, dt, scaleX, scaleY) {
        this.gl = initWebGL(canvas); // Use the utility from gpuUtils.js

        if (!this.gl) {
            console.error("Falling back to CPU fluid simulation due to WebGL initialization failure.");
            // Here, you might want a more robust fallback or to stop execution
            // For now, we'll just mark this instance as unusable for GPU tasks.
            this.gpuEnabled = false;
            return; // Early exit if WebGL isn't available
        }
        this.gpuEnabled = true;

        this.size = Math.round(size);
        this.dt = dt;
        this.diffusion = diffusion;
        this.viscosity = viscosity;
        this.scaleX = scaleX;
        this.scaleY = scaleY;
        this.useWrapping = false; // Will need to handle this in shaders or logic
        this.maxVelComponent = 10.0; // Default, sync with config (e.g., MAX_FLUID_VELOCITY_COMPONENT)

        this.iterations = 4; // Standard solver iterations

        // Placeholders for WebGL resources
        this.programs = {}; // To store shader programs (e.g., diffuse, advect, project_divergence, etc.)
        this.textures = {}; // To store textures (density, velocity - front and back for ping-pong)
        this.framebuffers = {}; // For rendering to textures
        this.quadVertexBuffer = null; // A simple quad to draw on for shader execution

        console.log("GPUFluidField initialized with WebGL.");

        // --- Define all shader sources as class properties ---
        this.basicVertexShaderSource = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            varying vec2 v_texCoord;
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }
        `;

        this.clearFragmentShaderSource = `
            precision mediump float;
            void main() {
                gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
            }
        `;
        
        this.copyFragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_inputTexture;
            void main() { gl_FragColor = texture2D(u_inputTexture, v_texCoord); }
        `;

        this.advectionFragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_velocityTexture;
            uniform sampler2D u_sourceTexture;
            uniform vec2 u_resolution;
            uniform float u_dt;
            uniform bool u_useWrapping;
            void main() {
                float dx = 1.0 / u_resolution.x;
                float dy = 1.0 / u_resolution.y;
                vec2 vel = texture2D(u_velocityTexture, v_texCoord).rg;
                vec2 prevCoords = v_texCoord - vel * u_dt * vec2(dx, dy);
                if (u_useWrapping) {
                    prevCoords = (mod(prevCoords, 1.0) + 1.0);
                    prevCoords = mod(prevCoords, 1.0);
                } else {
                    prevCoords = clamp(prevCoords, 0.0, 1.0);
                }
                gl_FragColor = texture2D(u_sourceTexture, prevCoords);
            }
        `;

        this.diffusionFragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_x_prev_iter_Texture;
            uniform sampler2D u_x0_Texture;
            uniform vec2 u_resolution;
            uniform float u_alpha;
            uniform float u_rBeta;
            void main() {
                float dx = 1.0 / u_resolution.x;
                float dy = 1.0 / u_resolution.y;
                vec4 x0 = texture2D(u_x0_Texture, v_texCoord);
                vec4 val_up    = texture2D(u_x_prev_iter_Texture, v_texCoord + vec2(0.0, dy));
                vec4 val_down  = texture2D(u_x_prev_iter_Texture, v_texCoord - vec2(0.0, dy));
                vec4 val_left  = texture2D(u_x_prev_iter_Texture, v_texCoord - vec2(dx, 0.0));
                vec4 val_right = texture2D(u_x_prev_iter_Texture, v_texCoord + vec2(dx, 0.0));
                gl_FragColor = (x0 + u_alpha * (val_up + val_down + val_left + val_right)) * u_rBeta;
            }
        `;

        this.divergenceFragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_velocityTexture;
            uniform vec2 u_resolution;
            void main() {
                float dx = 1.0 / u_resolution.x;
                float dy = 1.0 / u_resolution.y;
                float vx_left  = texture2D(u_velocityTexture, v_texCoord - vec2(dx, 0.0)).r;
                float vx_right = texture2D(u_velocityTexture, v_texCoord + vec2(dx, 0.0)).r;
                float vy_down  = texture2D(u_velocityTexture, v_texCoord - vec2(0.0, dy)).g;
                float vy_up    = texture2D(u_velocityTexture, v_texCoord + vec2(0.0, dy)).g;
                float divergence = (vx_right - vx_left) * (u_resolution.x * 0.5) + 
                                   (vy_up    - vy_down)  * (u_resolution.y * 0.5);
                gl_FragColor = vec4(divergence, 0.0, 0.0, 1.0);
            }
        `;

        this.pressureSolveFragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_p_prev_iter_Texture;
            uniform sampler2D u_divergenceTexture;
            uniform vec2 u_resolution;
            void main() {
                float dx = 1.0 / u_resolution.x;
                float divergence = texture2D(u_divergenceTexture, v_texCoord).r;
                float p_left  = texture2D(u_p_prev_iter_Texture, v_texCoord - vec2(dx, 0.0)).r;
                float p_right = texture2D(u_p_prev_iter_Texture, v_texCoord + vec2(dx, 0.0)).r;
                float p_up    = texture2D(u_p_prev_iter_Texture, v_texCoord + vec2(0.0, dx)).r;
                float p_down  = texture2D(u_p_prev_iter_Texture, v_texCoord - vec2(0.0, dx)).r;
                float cellWidthSq = dx * dx;
                float new_pressure = (p_left + p_right + p_up + p_down - divergence * cellWidthSq) * 0.25;
                gl_FragColor = vec4(new_pressure, 0.0, 0.0, 1.0);
            }
        `;

        this.gradientSubtractFragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_velocityTexture; 
            uniform sampler2D u_pressureTexture;   
            uniform vec2 u_resolution;            
            void main() {
                float dx_texel = 1.0 / u_resolution.x; 
                float dy_texel = 1.0 / u_resolution.y;
                vec2 current_velocity = texture2D(u_velocityTexture, v_texCoord).rg;
                float p_left  = texture2D(u_pressureTexture, v_texCoord - vec2(dx_texel, 0.0)).r;
                float p_right = texture2D(u_pressureTexture, v_texCoord + vec2(dx_texel, 0.0)).r;
                float p_up    = texture2D(u_pressureTexture, v_texCoord + vec2(0.0, dy_texel)).r;
                float p_down  = texture2D(u_pressureTexture, v_texCoord - vec2(0.0, dy_texel)).r;
                vec2 gradient = vec2(
                    (p_right - p_left) * u_resolution.x * 0.5,
                    (p_up    - p_down) * u_resolution.y * 0.5
                );
                vec2 new_velocity = current_velocity - gradient;
                gl_FragColor = vec4(new_velocity, 0.0, 1.0); 
            }
        `;

        this.drawTextureFragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_textureToDraw;
            void main() { gl_FragColor = texture2D(u_textureToDraw, v_texCoord); }
        `;

        this._initGeometry();
        this._initTextures();
        this._initFramebuffers();
        this._initShaders();
        console.log(`GPUFluidField constructor finished. gpuEnabled: ${this.gpuEnabled}`);
    }

    // --- Public API (matching CPU version where possible) ---
    step() {
        if (!this.gpuEnabled || !this.programs.advectProgram) {
            console.warn("GPUFluidField.step() called but GPU not enabled or advectProgram missing.");
            return; 
        }

        const gl = this.gl;

        // --- 1. Advect Density Field ---
        const advectUniforms = {
            u_resolution: [this.size, this.size],
            u_dt: this.dt,
            u_useWrapping: this.useWrapping ? 1 : 0, // GLSL bools are often handled as int/float
            // u_gridScale is derived inside shader from u_resolution
        };

        const advectDensityInputTextures = [
            { uniformName: "u_velocityTexture", texture: this.textures.velocityFront },
            { uniformName: "u_sourceTexture", texture: this.textures.densityFront }
        ];

        this._runShaderPass(
            this.programs.advectProgram, 
            this.framebuffers.densityBackFBO, // Render to densityBack
            advectDensityInputTextures, 
            advectUniforms
        );

        this.framebuffers.densityFrontFBO = this.framebuffers.densityBackFBO;
        this.framebuffers.densityBackFBO = tempFBO;

        // --- 1b. Diffuse Density Field (Dye Diffusion) ---
        if (this.diffusion > 0 && this.programs.diffusionProgram) {
            const N = this.size;
            let alpha = this.dt * this.diffusion * N * N; 
            let rBeta = 1.0 / (1.0 + 4.0 * alpha);

            const densityDiffusionUniforms = {
                u_resolution: [N, N],
                u_alpha: alpha,
                u_rBeta: rBeta
            };

            let x0ForDensityDiffusion = this.textures.densityFront; // Result of advection is the x0 for diffusion

            for (let k = 0; k < this.iterations; k++) {
                const diffusionInputTextures = [
                    { uniformName: "u_x_prev_iter_Texture", texture: this.textures.densityFront }, 
                    { uniformName: "u_x0_Texture", texture: x0ForDensityDiffusion } 
                ];

                this._runShaderPass(
                    this.programs.diffusionProgram,
                    this.framebuffers.densityBackFBO, // Write to densityBack
                    diffusionInputTextures,
                    densityDiffusionUniforms
                );

                // Ping-pong for next iteration
                tempTex = this.textures.densityFront;
                this.textures.densityFront = this.textures.densityBack;
                this.textures.densityBack = tempTex;

                tempFBO = this.framebuffers.densityFrontFBO;
                this.framebuffers.densityFrontFBO = this.framebuffers.densityBackFBO;
                this.framebuffers.densityBackFBO = tempFBO;
            }
            // Diffused density is now in this.textures.densityFront
        }

        // --- 2. Advect Velocity Field (Velocity Advects Itself) ---
        // The same advection shader and uniforms can be used.
        // The source texture is now the current velocity field, 
        // and the velocity texture driving the advection is also the current velocity field.
        const advectVelocityInputTextures = [
            { uniformName: "u_velocityTexture", texture: this.textures.velocityFront }, // Velocity field guiding the advection
            { uniformName: "u_sourceTexture", texture: this.textures.velocityFront }    // Velocity field being advected
        ];

        // advectUniforms is already defined from density advection and can be reused
        this._runShaderPass(
            this.programs.advectProgram,
            this.framebuffers.velocityBackFBO, // Render to velocityBack
            advectVelocityInputTextures,
            advectUniforms 
        );

        // Ping-pong velocity textures and associated FBOs
        tempTex = this.textures.velocityFront;
        this.textures.velocityFront = this.textures.velocityBack;
        this.textures.velocityBack = tempTex;

        tempFBO = this.framebuffers.velocityFrontFBO;
        this.framebuffers.velocityFrontFBO = this.framebuffers.velocityBackFBO;
        this.framebuffers.velocityBackFBO = tempFBO;

        // --- 3. Diffuse Velocity Field (Viscosity) ---
        if (this.viscosity > 0 && this.programs.diffusionProgram) {
            const N = this.size;
            // alpha in diffusion shader corresponds to (nu * dt / (h*h)) if h is cell size
            // Here, if dx = 1.0/N, then h*h = (1/N)^2. So N*N comes into numerator.
            // Or, if shader uses texture coords [0,1] and dx=1/N, then effectively h=1. alpha = nu*dt.
            // Let's match the CPU version's `a = dt * diff * (N-2)*(N-2)` logic more closely for alpha interpretation.
            // The shader uses dx = 1.0/N. The `a` in `lin_solve` is `dt * base_diff_rate * (grid_cells_dim - 2)^2`.
            // For the shader, alpha is part of `(x0 + alpha * sum_neighbors) * rBeta`.
            // If `sum_neighbors` is just sum of values, alpha needs to incorporate cell spacing effects.
            // Let alpha = dt * viscosity. The shader samples neighbors directly using dx, dy based on resolution.
            // The CPU lin_solve: x = (x0 + a * sum_neighbors_x_prev) / (1 + 4a).
            // Our shader: gl_FragColor = (x0 + u_alpha * sum_neighbors_x_prev) * u_rBeta;
            // So, u_alpha in shader is equivalent to `a` in CPU, and u_rBeta is `1/(1+4a)`.
            // CPU `a` = `dt * this.viscosity * (N-2)*(N-2)`. Let's use N*N for simplicity with full grid shaders.
            // The (N-2)^2 was specific to the CPU boundary handling loop starting from 1 to N-1.
            // Our shaders operate on 0 to N-1 implicitly via texture coords 0 to 1.
            
            let alpha = this.dt * this.viscosity * N * N; // N*N factor to scale for grid size
            let rBeta = 1.0 / (1.0 + 4.0 * alpha);

            const diffusionUniforms = {
                u_resolution: [N, N],
                u_alpha: alpha,
                u_rBeta: rBeta
            };

            let x0ForDiffusion = this.textures.velocityFront; // This is the result of advection

            for (let k = 0; k < this.iterations; k++) {
                const diffusionInputTextures = [
                    { uniformName: "u_x_prev_iter_Texture", texture: this.textures.velocityFront }, // x_k (current iteration's input)
                    { uniformName: "u_x0_Texture", texture: x0ForDiffusion } // x_original (post-advection)
                ];

                this._runShaderPass(
                    this.programs.diffusionProgram,
                    this.framebuffers.velocityBackFBO, // Write to velocityBack (x_k+1)
                    diffusionInputTextures,
                    diffusionUniforms
                );

                // Ping-pong for next iteration
                tempTex = this.textures.velocityFront;
                this.textures.velocityFront = this.textures.velocityBack;
                this.textures.velocityBack = tempTex;

                tempFBO = this.framebuffers.velocityFrontFBO;
                this.framebuffers.velocityFrontFBO = this.framebuffers.velocityBackFBO;
                this.framebuffers.velocityBackFBO = tempFBO;
            }
        }

        // --- 4. Calculate Divergence of Velocity Field ---
        if (this.programs.divergenceProgram) {
            const divergenceUniforms = {
                u_resolution: [this.size, this.size]
            };
            const divergenceInputTextures = [
                { uniformName: "u_velocityTexture", texture: this.textures.velocityFront } // Velocity after advection and diffusion
            ];

            this._runShaderPass(
                this.programs.divergenceProgram,
                this.framebuffers.divergenceFBO, // Render to the divergence texture
                divergenceInputTextures,
                divergenceUniforms
            );
            // Result is now in this.textures.divergenceTexture
        }

        // --- 5. Solve for Pressure (Poisson Equation using Jacobi Iteration) ---
        if (this.programs.pressureSolveProgram && this.programs.clearProgram) {
            // Clear pressure textures to zero before solving
            this._runShaderPass(this.programs.clearProgram, this.framebuffers.pressureFrontFBO, null, null);
            this._runShaderPass(this.programs.clearProgram, this.framebuffers.pressureBackFBO, null, null);

            const pressureUniforms = {
                u_resolution: [this.size, this.size]
                // The pressure shader calculates cellWidthSq from resolution internally
            };

            for (let k = 0; k < this.iterations; k++) { // Using same iterations as diffusion for now
                const pressureInputTextures = [
                    { uniformName: "u_p_prev_iter_Texture", texture: this.textures.pressureFront },
                    { uniformName: "u_divergenceTexture", texture: this.textures.divergenceTexture }
                ];

                this._runShaderPass(
                    this.programs.pressureSolveProgram,
                    this.framebuffers.pressureBackFBO, // Write to pressureBack
                    pressureInputTextures,
                    pressureUniforms
                );

                // Ping-pong pressure textures and FBOs for next iteration
                let tempTex = this.textures.pressureFront;
                this.textures.pressureFront = this.textures.pressureBack;
                this.textures.pressureBack = tempTex;

                let tempFBO = this.framebuffers.pressureFrontFBO;
                this.framebuffers.pressureFrontFBO = this.framebuffers.pressureBackFBO;
                this.framebuffers.pressureBackFBO = tempFBO;
            }
            // Solved pressure is now in this.textures.pressureFront
        }

        // --- 6. Subtract Pressure Gradient from Velocity Field ---
        if (this.programs.gradientSubtractProgram) {
            const gradSubUniforms = {
                u_resolution: [this.size, this.size]
            };
            const gradSubInputTextures = [
                // Important: Use the velocity field *before* pressure solving (i.e., after diffusion)
                // If diffusion modified velocityFront, and pressure also used velocityFront as an input for some reason
                // (which it doesn't directly, it uses divergence derived from it), ensure correct version of velocity is used.
                // Currently, after diffusion loop, velocityFront holds diffused velocity. This is what we want to correct.
                { uniformName: "u_velocityTexture", texture: this.textures.velocityFront }, 
                { uniformName: "u_pressureTexture", texture: this.textures.pressureFront }
            ];

            this._runShaderPass(
                this.programs.gradientSubtractProgram,
                this.framebuffers.velocityBackFBO, // Write corrected velocity to velocityBack
                gradSubInputTextures,
                gradSubUniforms
            );

            // Ping-pong velocity textures and FBOs one last time for this step
            // The result (divergence-free velocity) is now in velocityFront for the next simulation tick (or for drawing)
            let tempTex = this.textures.velocityFront;
            this.textures.velocityFront = this.textures.velocityBack;
            this.textures.velocityBack = tempTex;

            let tempFBO = this.framebuffers.velocityFrontFBO;
            this.framebuffers.velocityFrontFBO = this.framebuffers.velocityBackFBO;
            this.framebuffers.velocityBackFBO = tempFBO;
        }

        // console.log("GPUFluidField.step() - All steps completed for this tick.");
    }

    addDensity(x, y, emitterR, emitterG, emitterB, emissionStrength) {
        if (!this.gpuEnabled) return;
        const gl = this.gl;

        // For now, directly set the texel. This is a simplified approach.
        // A more robust method would use an additive blending shader pass or an impulse texture.
        // Ensure x, y are valid grid coordinates
        const gridX = Math.floor(x); 
        const gridY = Math.floor(y);

        if (gridX < 0 || gridX >= this.size || gridY < 0 || gridY >= this.size) {
            return; // Out of bounds
        }

        // Normalize emitterRGB to [0,1] if they are [0,255]
        // Assuming emitterR, G, B are already in a suitable range (e.g. 0-255 or 0-1 based on CPU version)
        // The CPU version does a kind of blend. Here we just set directly for simplicity.
        // Scale emissionStrength to a [0,1] factor for color intensity.
        const intensity = Math.min(1.0, Math.max(0.0, emissionStrength / 255.0)); // Example scaling
        
        // Data for a single texel (RGBA)
        // Let's assume emitterR,G,B are 0-255 like in CPU addDensity before normalization.
        const pixelData = new Float32Array([
            (emitterR / 255.0) * intensity, 
            (emitterG / 255.0) * intensity, 
            (emitterB / 255.0) * intensity, 
            intensity // Use intensity for Alpha, or 1.0 for opaque
        ]);

        gl.bindTexture(gl.TEXTURE_2D, this.textures.densityFront);
        // texSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels)
        gl.texSubImage2D(gl.TEXTURE_2D, 0, gridX, gridY, 1, 1, gl.RGBA, gl.FLOAT, pixelData);
        gl.bindTexture(gl.TEXTURE_2D, null); // Unbind

        // console.log(`GPUFluidField.addDensity at (${gridX},${gridY})`);
    }

    addVelocity(x, y, amountX, amountY) {
        if (!this.gpuEnabled) return;
        const gl = this.gl;

        // Ensure x, y are valid grid coordinates
        const gridX = Math.floor(x);
        const gridY = Math.floor(y);

        if (gridX < 0 || gridX >= this.size || gridY < 0 || gridY >= this.size) {
            return; // Out of bounds
        }

        // Clamp velocity components
        const clampedAmountX = Math.max(-this.maxVelComponent, Math.min(this.maxVelComponent, amountX));
        const clampedAmountY = Math.max(-this.maxVelComponent, Math.min(this.maxVelComponent, amountY));

        // Data for a single texel (Vx, Vy, 0, 0 for RGBA texture format)
        // Our _initTextures sets up velocity as RG or RGBA (with B,A unused for actual velocity)
        const isWebGL2 = gl.getParameter(gl.VERSION).includes("WebGL 2.0");
        const numVelocityComponents = isWebGL2 && this.textures.velocityFront ? (gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_INTERNAL_FORMAT) === gl.RG32F || gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_INTERNAL_FORMAT) === gl.RG16F ? 2 : 4) : 4; // A bit complex to get actual components, default to 4 for safety with texSubImage2D for RGBA format.
        // Simpler: Assume pixelData is always RGBA for texSubImage2D based on current _initTextures logic for WebGL1 fallback
        
        const pixelData = new Float32Array([
            clampedAmountX,
            clampedAmountY,
            0.0, // Blue channel (unused for 2D velocity)
            0.0  // Alpha channel (unused for 2D velocity)
        ]);

        gl.bindTexture(gl.TEXTURE_2D, this.textures.velocityFront);
        // texSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels)
        // The format argument to texSubImage2D must match the original internalFormat's base format (e.g. RGBA for RGBA32F)
        let uploadFormat = gl.RGBA;
        if (isWebGL2) {
             // Check the actual format of the texture if possible, or stick to what was used at creation
             // For simplicity, if velocityInternalFormat was RG32F, format should be RG.
             // However, _initTextures currently uses gl.RG for format with gl.RG32F internalFormat.
             // Sticking to gl.RGBA for pixelData array and texSubImage2D format for broader compatibility for now.
        }

        gl.texSubImage2D(gl.TEXTURE_2D, 0, gridX, gridY, 1, 1, gl.RGBA, gl.FLOAT, pixelData);
        gl.bindTexture(gl.TEXTURE_2D, null); // Unbind

        // console.log(`GPUFluidField.addVelocity at (${gridX},${gridY}) with (${clampedAmountX}, ${clampedAmountY})`);
    }

    draw(ctxToDrawOn, viewportCanvasWidth, viewportCanvasHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
        if (!this.gpuEnabled || !this.programs.drawTextureProgram || !this.textures.densityFront) {
            // console.warn("GPUFluidField.draw() called but GPU not enabled, shader missing, or texture missing.");
            return;
        }
        console.log("Drawing with GPUFluidField");

        const gl = this.gl;
        const programInfo = this.programs.drawTextureProgram;

        // Bind the default framebuffer (render to screen)
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        // Set the viewport to the size of the canvas we are drawing to
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height); // Use gl.canvas.width/height for the WebGL viewport

        gl.useProgram(programInfo.program);

        // Setup vertex attributes for the quad
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);

        const positionLoc = programInfo.attributes.position;
        if (positionLoc !== -1 && positionLoc !== undefined) {
            gl.enableVertexAttribArray(positionLoc);
            gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 4 * Float32Array.BYTES_PER_ELEMENT, 0);
        }
        
        const texCoordLoc = programInfo.attributes.texCoord;
        if (texCoordLoc !== -1 && texCoordLoc !== undefined) {
            gl.enableVertexAttribArray(texCoordLoc);
            gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 4 * Float32Array.BYTES_PER_ELEMENT, 2 * Float32Array.BYTES_PER_ELEMENT);
        }

        // Activate texture unit 0 and bind our density texture to it
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.densityFront);
        // Tell the shader to use texture unit 0 for the u_textureToDraw uniform
        if (programInfo.uniforms.textureToDraw) {
            gl.uniform1i(programInfo.uniforms.textureToDraw, 0);
        }

        // We don't need to clear the main canvas here, as the simulation's main draw loop does that.
        // We are effectively drawing the fluid layer on top.
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); // Draw the quad

        // Optional: Unbind buffer, disable attributes if not done by a higher-level state manager
        // gl.bindBuffer(gl.ARRAY_BUFFER, null);
        // if (positionLoc !== -1 && positionLoc !== undefined) gl.disableVertexAttribArray(positionLoc);
        // if (texCoordLoc !== -1 && texCoordLoc !== undefined) gl.disableVertexAttribArray(texCoordLoc);
    }

    clear() {
        if (!this.gpuEnabled || !this.programs.clearProgram) return;

        console.log("GPUFluidField: Clearing data textures...");

        // Clear densityFront texture
        this._runShaderPass(this.programs.clearProgram, this.framebuffers.densityFrontFBO, null, null);
        // Clear densityBack texture
        this._runShaderPass(this.programs.clearProgram, this.framebuffers.densityBackFBO, null, null);
        // Clear velocityFront texture
        this._runShaderPass(this.programs.clearProgram, this.framebuffers.velocityFrontFBO, null, null);
        // Clear velocityBack texture
        this._runShaderPass(this.programs.clearProgram, this.framebuffers.velocityBackFBO, null, null);

        console.log("GPUFluidField: Data textures cleared.");
    }

    // --- Internal WebGL Helper Methods (Private-like) ---
    _createTexture(gl, target, width, height, internalFormat, format, type, data, filter = gl.NEAREST, wrap = gl.CLAMP_TO_EDGE) {
        const texture = gl.createTexture();
        gl.bindTexture(target, texture);
        //                  target, level, internalFormat, width, height, border, format, type, pixels
        gl.texImage2D(target, 0,     internalFormat, width, height, 0,      format, type, data);
        
        gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, filter);
        gl.texParameteri(target, gl.TEXTURE_WRAP_S, wrap);
        gl.texParameteri(target, gl.TEXTURE_WRAP_T, wrap);
        
        gl.bindTexture(target, null); // Unbind
        return texture;
    }

    _initTextures() {
        const gl = this.gl;
        if (!gl) return;

        const width = this.size;
        const height = this.size;

        // Texture formats - prefer WebGL2 float formats if available
        // For WebGL1, OES_texture_float is needed for gl.FLOAT type with gl.RGBA.
        // EXT_color_buffer_float is needed to render to float textures.
        let densityInternalFormat = gl.RGBA32F; // WebGL2
        let densityFormat = gl.RGBA;
        let densityType = gl.FLOAT;
        let velocityInternalFormat = gl.RG32F; // WebGL2 for 2-component float texture (Vx, Vy)
        let velocityFormat = gl.RG;            // WebGL2
        let velocityType = gl.FLOAT;

        const isWebGL2 = gl.getParameter(gl.VERSION).includes("WebGL 2.0");

        if (!isWebGL2) {
            // Fallback for WebGL1 - requires OES_texture_float
            // and often data is packed into RGBA if RG formats aren't directly supported for float textures.
            if (!gl.getExtension('OES_texture_float')) {
                console.error("OES_texture_float extension not available. Cannot create float textures for WebGL1.");
                this.gpuEnabled = false; return;
            }
            // WebGL1 doesn't have RG32F, so use RGBA32F (or RGBA with FLOAT type)
            densityInternalFormat = gl.RGBA;
            velocityInternalFormat = gl.RGBA; // Store Vx, Vy in R,G channels; B,A unused or for other data
            velocityFormat = gl.RGBA;
        }
        
        // Initial data (all zeros)
        const numDensityComponents = 4; // RGBA
        const initialDensityData = new Float32Array(width * height * numDensityComponents).fill(0.0);
        
        const numVelocityComponents = isWebGL2 ? 2 : 4; // RG for WebGL2, RGBA for WebGL1 fallback
        const initialVelocityData = new Float32Array(width * height * numVelocityComponents).fill(0.0);

        // Create Density Textures (Ping-Pong)
        this.textures.densityFront = this._createTexture(gl, gl.TEXTURE_2D, width, height, densityInternalFormat, densityFormat, densityType, initialDensityData);
        this.textures.densityBack = this._createTexture(gl, gl.TEXTURE_2D, width, height, densityInternalFormat, densityFormat, densityType, initialDensityData);

        // Create Velocity Textures (Ping-Pong)
        this.textures.velocityFront = this._createTexture(gl, gl.TEXTURE_2D, width, height, velocityInternalFormat, velocityFormat, velocityType, initialVelocityData);
        this.textures.velocityBack = this._createTexture(gl, gl.TEXTURE_2D, width, height, velocityInternalFormat, velocityFormat, velocityType, initialVelocityData);

        // Create Divergence Texture (scalar field, use R channel of RGBA for now)
        this.textures.divergenceTexture = this._createTexture(gl, gl.TEXTURE_2D, width, height, densityInternalFormat, densityFormat, densityType, initialDensityData); // Reuse density format for simplicity

        // Create Pressure Textures (Ping-Pong, scalar field, use R channel of RGBA)
        this.textures.pressureFront = this._createTexture(gl, gl.TEXTURE_2D, width, height, densityInternalFormat, densityFormat, densityType, initialDensityData);
        this.textures.pressureBack = this._createTexture(gl, gl.TEXTURE_2D, width, height, densityInternalFormat, densityFormat, densityType, initialDensityData);

        console.log("GPUFluidField: Data textures initialized (Density, Velocity, Divergence, Pressure).");
    }

    _initFramebuffers() {
        const gl = this.gl;
        if (!gl) return;

        // Create FBO for densityBack texture
        this.framebuffers.densityBackFBO = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.densityBackFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.densityBack, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            console.error("GPUFluidField: Framebuffer for densityBack is not complete.");
        }

        // Create FBO for velocityBack texture
        this.framebuffers.velocityBackFBO = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.velocityBackFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.velocityBack, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            console.error("GPUFluidField: Framebuffer for velocityBack is not complete.");
        }

        // Create FBO for densityFront texture (can be useful for some operations like clear)
        this.framebuffers.densityFrontFBO = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.densityFrontFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.densityFront, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            console.error("GPUFluidField: Framebuffer for densityFront is not complete.");
        }

        // Create FBO for velocityFront texture
        this.framebuffers.velocityFrontFBO = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.velocityFrontFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.velocityFront, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            console.error("GPUFluidField: Framebuffer for velocityFront is not complete.");
        }

        // Create FBO for divergence texture
        this.framebuffers.divergenceFBO = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.divergenceFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.divergenceTexture, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            console.error("GPUFluidField: Framebuffer for divergenceTexture is not complete.");
        }

        // Create FBOs for pressure textures
        this.framebuffers.pressureFrontFBO = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.pressureFrontFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.pressureFront, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            console.error("GPUFluidField: Framebuffer for pressureFront is not complete.");
        }

        this.framebuffers.pressureBackFBO = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.pressureBackFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.pressureBack, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            console.error("GPUFluidField: Framebuffer for pressureBack is not complete.");
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Unbind FBO
        console.log("GPUFluidField: Framebuffers initialized.");
    }

    _initShaders() {
        const gl = this.gl;
        if (!gl || !this.gpuEnabled) {
            console.error("GPUFluidField: Cannot initialize shaders, WebGL not enabled or not available.");
            this.gpuEnabled = false;
            return;
        }

        // --- Compile Basic Vertex Shader --- 
        if (!this.basicVertexShaderSource) {
            console.error("GPUFluidField: basicVertexShaderSource is not defined.");
            this.gpuEnabled = false; return;
        }
        const basicVertexShader = createShader(gl, gl.VERTEX_SHADER, this.basicVertexShaderSource);
        if (!basicVertexShader) {
            console.error("GPUFluidField: Failed to compile basic vertex shader.");
            this.gpuEnabled = false; return;
        }

        // Helper to create a program
        const setupProgram = (programName, fragmentShaderSourceString) => {
            if (!this[fragmentShaderSourceString]) {
                console.error(`GPUFluidField: Shader source this.${fragmentShaderSourceString} is not defined.`);
                return null;
            }
            const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, this[fragmentShaderSourceString]);
            if (!fragmentShader) {
                console.error(`GPUFluidField: Failed to compile fragment shader for ${programName}.`);
                return null;
            }
            const program = createProgram(gl, basicVertexShader, fragmentShader);
            if (!program) {
                console.error(`GPUFluidField: Failed to link shader program for ${programName}.`);
                gl.deleteShader(fragmentShader); // Clean up fragment shader
                return null;
            }
            gl.deleteShader(fragmentShader); // Clean up fragment shader after successful link
            return program;
        };

        // --- Initialize All Shader Programs ---
        // Copy Program
        this.programs.copyProgram = {};
        this.programs.copyProgram.program = setupProgram("copyProgram", "copyFragmentShaderSource");
        if (this.programs.copyProgram.program) {
            this.programs.copyProgram.attributes = {
                position: gl.getAttribLocation(this.programs.copyProgram.program, "a_position"),
                texCoord: gl.getAttribLocation(this.programs.copyProgram.program, "a_texCoord"),
            };
            this.programs.copyProgram.uniforms = {
                inputTexture: gl.getUniformLocation(this.programs.copyProgram.program, "u_inputTexture"),
            };
            console.log("GPUFluidField: Copy shader program initialized.");
        } else { this.gpuEnabled = false; return; }

        // Clear Program
        this.programs.clearProgram = {};
        this.programs.clearProgram.program = setupProgram("clearProgram", "clearFragmentShaderSource");
        if (this.programs.clearProgram.program) {
            this.programs.clearProgram.attributes = {
                position: gl.getAttribLocation(this.programs.clearProgram.program, "a_position"),
                texCoord: gl.getAttribLocation(this.programs.clearProgram.program, "a_texCoord"),
            };
            this.programs.clearProgram.uniforms = {}; 
            console.log("GPUFluidField: Clear shader program initialized.");
        } else { this.gpuEnabled = false; return; }
        
        // Advection Program
        this.programs.advectProgram = {};
        this.programs.advectProgram.program = setupProgram("advectProgram", "advectionFragmentShaderSource");
        if (this.programs.advectProgram.program) {
            this.programs.advectProgram.attributes = {
                position: gl.getAttribLocation(this.programs.advectProgram.program, "a_position"),
                texCoord: gl.getAttribLocation(this.programs.advectProgram.program, "a_texCoord"),
            };
            this.programs.advectProgram.uniforms = {
                velocityTexture: gl.getUniformLocation(this.programs.advectProgram.program, "u_velocityTexture"),
                sourceTexture: gl.getUniformLocation(this.programs.advectProgram.program, "u_sourceTexture"),
                resolution: gl.getUniformLocation(this.programs.advectProgram.program, "u_resolution"),
                dt: gl.getUniformLocation(this.programs.advectProgram.program, "u_dt"),
                useWrapping: gl.getUniformLocation(this.programs.advectProgram.program, "u_useWrapping"),
            };
            console.log("GPUFluidField: Advection shader program initialized.");
        } else { this.gpuEnabled = false; return; }

        // Diffusion Program
        this.programs.diffusionProgram = {};
        this.programs.diffusionProgram.program = setupProgram("diffusionProgram", "diffusionFragmentShaderSource");
        if (this.programs.diffusionProgram.program) {
            this.programs.diffusionProgram.attributes = {
                position: gl.getAttribLocation(this.programs.diffusionProgram.program, "a_position"),
                texCoord: gl.getAttribLocation(this.programs.diffusionProgram.program, "a_texCoord"),
            };
            this.programs.diffusionProgram.uniforms = {
                x_prev_iter_Texture: gl.getUniformLocation(this.programs.diffusionProgram.program, "u_x_prev_iter_Texture"),
                x0_Texture: gl.getUniformLocation(this.programs.diffusionProgram.program, "u_x0_Texture"),
                resolution: gl.getUniformLocation(this.programs.diffusionProgram.program, "u_resolution"),
                alpha: gl.getUniformLocation(this.programs.diffusionProgram.program, "u_alpha"),
                rBeta: gl.getUniformLocation(this.programs.diffusionProgram.program, "u_rBeta"),
            };
            console.log("GPUFluidField: Diffusion shader program initialized.");
        } else { this.gpuEnabled = false; return; }

        // Divergence Program
        this.programs.divergenceProgram = {};
        this.programs.divergenceProgram.program = setupProgram("divergenceProgram", "divergenceFragmentShaderSource");
        if (this.programs.divergenceProgram.program) {
            this.programs.divergenceProgram.attributes = {
                position: gl.getAttribLocation(this.programs.divergenceProgram.program, "a_position"),
                texCoord: gl.getAttribLocation(this.programs.divergenceProgram.program, "a_texCoord"),
            };
            this.programs.divergenceProgram.uniforms = {
                velocityTexture: gl.getUniformLocation(this.programs.divergenceProgram.program, "u_velocityTexture"),
                resolution: gl.getUniformLocation(this.programs.divergenceProgram.program, "u_resolution"),
            };
            console.log("GPUFluidField: Divergence shader program initialized.");
        } else { this.gpuEnabled = false; return; }

        // Pressure Solver Program
        this.programs.pressureSolveProgram = {};
        this.programs.pressureSolveProgram.program = setupProgram("pressureSolveProgram", "pressureSolveFragmentShaderSource");
        if (this.programs.pressureSolveProgram.program) {
            this.programs.pressureSolveProgram.attributes = {
                position: gl.getAttribLocation(this.programs.pressureSolveProgram.program, "a_position"),
                texCoord: gl.getAttribLocation(this.programs.pressureSolveProgram.program, "a_texCoord"),
            };
            this.programs.pressureSolveProgram.uniforms = {
                p_prev_iter_Texture: gl.getUniformLocation(this.programs.pressureSolveProgram.program, "u_p_prev_iter_Texture"),
                divergenceTexture: gl.getUniformLocation(this.programs.pressureSolveProgram.program, "u_divergenceTexture"),
                resolution: gl.getUniformLocation(this.programs.pressureSolveProgram.program, "u_resolution"),
            };
            console.log("GPUFluidField: Pressure solver shader program initialized.");
        } else { this.gpuEnabled = false; return; }

        // Gradient Subtraction Program
        this.programs.gradientSubtractProgram = {};
        this.programs.gradientSubtractProgram.program = setupProgram("gradientSubtractProgram", "gradientSubtractFragmentShaderSource");
        if (this.programs.gradientSubtractProgram.program) {
            this.programs.gradientSubtractProgram.attributes = {
                position: gl.getAttribLocation(this.programs.gradientSubtractProgram.program, "a_position"),
                texCoord: gl.getAttribLocation(this.programs.gradientSubtractProgram.program, "a_texCoord"),
            };
            this.programs.gradientSubtractProgram.uniforms = {
                velocityTexture: gl.getUniformLocation(this.programs.gradientSubtractProgram.program, "u_velocityTexture"),
                pressureTexture: gl.getUniformLocation(this.programs.gradientSubtractProgram.program, "u_pressureTexture"),
                resolution: gl.getUniformLocation(this.programs.gradientSubtractProgram.program, "u_resolution"),
            };
            console.log("GPUFluidField: Gradient subtraction shader program initialized.");
        } else { this.gpuEnabled = false; return; }

        // Draw Texture Program (for rendering to screen)
        this.programs.drawTextureProgram = {};
        this.programs.drawTextureProgram.program = setupProgram("drawTextureProgram", "drawTextureFragmentShaderSource");
        if (this.programs.drawTextureProgram.program) {
            this.programs.drawTextureProgram.attributes = {
                position: gl.getAttribLocation(this.programs.drawTextureProgram.program, "a_position"),
                texCoord: gl.getAttribLocation(this.programs.drawTextureProgram.program, "a_texCoord"),
            };
            this.programs.drawTextureProgram.uniforms = {
                textureToDraw: gl.getUniformLocation(this.programs.drawTextureProgram.program, "u_textureToDraw"),
            };
            console.log("GPUFluidField: DrawTexture shader program initialized.");
        } else { this.gpuEnabled = false; return; }

        // Clean up the common vertex shader as it's now linked into all programs
        if (basicVertexShader) {
            gl.deleteShader(basicVertexShader);
        }
    }

    _runShaderPass(shaderProgramInfo, targetFBO, inputTexturesAnfInfo, uniforms) {
        const gl = this.gl;
        if (!gl || !this.gpuEnabled || !shaderProgramInfo || !shaderProgramInfo.program) {
            console.error("GPUFluidField: Cannot run shader pass, WebGL not enabled or shader program missing.");
            return;
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
        gl.viewport(0, 0, this.size, this.size); // Render to the full texture size

        gl.useProgram(shaderProgramInfo.program);

        // Setup vertex attributes
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
        
        const positionLoc = shaderProgramInfo.attributes.position;
        if (positionLoc !== -1 && positionLoc !== undefined) {
            gl.enableVertexAttribArray(positionLoc);
            // params: index, size (num components per iter), type, normalize, stride (bytes), offset (bytes)
            gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 4 * Float32Array.BYTES_PER_ELEMENT, 0); 
        }
        
        const texCoordLoc = shaderProgramInfo.attributes.texCoord;
        if (texCoordLoc !== -1 && texCoordLoc !== undefined) {
            gl.enableVertexAttribArray(texCoordLoc);
            gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 4 * Float32Array.BYTES_PER_ELEMENT, 2 * Float32Array.BYTES_PER_ELEMENT);
        }

        // Bind input textures and set texture uniforms
        if (inputTexturesAnfInfo) {
            inputTexturesAnfInfo.forEach((texInfo, index) => {
                if (shaderProgramInfo.uniforms[texInfo.uniformName]) {
                    gl.activeTexture(gl.TEXTURE0 + index);
                    gl.bindTexture(gl.TEXTURE_2D, texInfo.texture);
                    gl.uniform1i(shaderProgramInfo.uniforms[texInfo.uniformName], index);
                }
            });
        }

        // Set other uniforms (floats, ints, vectors, etc.)
        if (uniforms) {
            for (const uniformName in uniforms) {
                if (Object.hasOwnProperty.call(uniforms, uniformName) && shaderProgramInfo.uniforms[uniformName]) {
                    const value = uniforms[uniformName];
                    const location = shaderProgramInfo.uniforms[uniformName];
                    if (typeof value === 'number') {
                        gl.uniform1f(location, value);
                    } else if (Array.isArray(value)) {
                        if (value.length === 2) gl.uniform2fv(location, value);
                        else if (value.length === 3) gl.uniform3fv(location, value);
                        else if (value.length === 4) gl.uniform4fv(location, value);
                    } else if (typeof value === 'boolean'){
                        gl.uniform1i(location, value ? 1 : 0); // Pass booleans as int
                    } else {
                        console.warn(`GPUFluidField: Unsupported uniform type for ${uniformName}`);
                    }
                }
            }
        }

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); // Draw the quad

        // Unbind framebuffer to return to default (screen) or for next pass that might bind another FBO
        // gl.bindFramebuffer(gl.FRAMEBUFFER, null); 
        // Disabling attributes after draw can also be good practice if not managed by VAOs (which we aren't using yet for simplicity)
        // if (positionLoc !== -1 && positionLoc !== undefined) gl.disableVertexAttribArray(positionLoc);
        // if (texCoordLoc !== -1 && texCoordLoc !== undefined) gl.disableVertexAttribArray(texCoordLoc);
    }

    _initGeometry() {
        const gl = this.gl;
        if (!gl) return;

        // A simple quad covering the entire clip space. 
        // x, y for position; s, t for texture coordinates (optional but good practice)
        const quadVertices = new Float32Array([
            // Positions     // Texture Coords (optional, but useful for sampling)
            -1.0, -1.0,      0.0, 0.0, // bottom left
             1.0, -1.0,      1.0, 0.0, // bottom right
            -1.0,  1.0,      0.0, 1.0, // top left
             1.0,  1.0,      1.0, 1.0  // top right
        ]);

        this.quadVertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);

        console.log("GPUFluidField: Quad vertex buffer initialized.");
    }
}