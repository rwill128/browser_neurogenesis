import {WEBGL_fluid_simulation_vertex_shader, WEBGL_advect_fragment_shader, WEBGL_divergence_fragment_shader, WEBGL_jacobi_fragment_shader, WEBGL_gradient_subtraction_fragment_shader, WEBGL_splat_fragment_shader, WEBGL_display_fragment_shader, WEBGL_viscosity_jacobi_fragment_shader} from './shaders.js'

export class GPUFluidField {
    constructor(canvas, size, diffusion, viscosity, dt, scaleX, scaleY) {
        this.canvas = canvas;
        this.gpuEnabled = false; // Default to false
        this.gl = null; // Keep for now, might be useful for mixed mode or progressive refactor
        this.device = null; // WebGPU device
        this.context = null; // WebGPU context
        this.presentationFormat = null; // WebGPU presentation format
        this.adapter = null; // WebGPU adapter

        // Initialize properties
        this.size = Math.round(size);
        this.dt = dt;
        this.diffusion = diffusion;
        this.viscosity = viscosity;
        this.scaleX = scaleX;
        this.scaleY = scaleY;
        this.useWrapping = false; // Will need to handle this in shaders or logic
        this.maxVelComponent = MAX_FLUID_VELOCITY_COMPONENT; // Ensure this is defined or passed
        this.iterations = 4; // Standard solver iterations

        // CPU shadow fields for gameplay coupling queries (body<->fluid) while simulation runs on GPU.
        // These are updated by impulses and lightly decayed each step to keep deterministic, non-random sampling.
        const cellCount = this.size * this.size;
        this.shadowVx = new Float32Array(cellCount).fill(0);
        this.shadowVy = new Float32Array(cellCount).fill(0);
        this.shadowDensityR = new Float32Array(cellCount).fill(0);
        this.shadowDensityG = new Float32Array(cellCount).fill(0);
        this.shadowDensityB = new Float32Array(cellCount).fill(0);
        this.shadowVxNext = new Float32Array(cellCount).fill(0);
        this.shadowVyNext = new Float32Array(cellCount).fill(0);
        this.shadowDensityRNext = new Float32Array(cellCount).fill(0);
        this.shadowDensityGNext = new Float32Array(cellCount).fill(0);
        this.shadowDensityBNext = new Float32Array(cellCount).fill(0);

        this._initShadowBackCompatViews();

        // Placeholders for WebGL resources
        this.programs = {}; // To store shader programs (e.g., diffuse, advect, project_divergence, etc.)
        this.textures = {}; // To store textures (density, velocity - front and back for ping-pong)
        this.framebuffers = {}; // For rendering to textures
        this.quadVertexBuffer = null; // A simple quad to draw on for shader execution
        this.vertexState = null; // For WebGPU pipeline vertex layout
        this.sampler = null;     // For WebGPU sampler

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

        // Lightweight perf telemetry
        this.perfStats = {
            stepCount: 0,
            stepAccumMs: 0,
            stepMaxMs: 0,
            drawCount: 0,
            drawAccumMs: 0,
            drawMaxMs: 0,
            lastStepMs: 0,
            lastDrawMs: 0,
            lastReportTs: performance.now()
        };

        // Store the promise for external awaiting if needed
        this._initPromise = this._asyncInit(canvas);
    }

    _initShadowBackCompatViews() {
        const descriptors = {
            Vx: { get: () => this.shadowVx },
            Vy: { get: () => this.shadowVy },
            Vx0: { get: () => this.shadowVxNext },
            Vy0: { get: () => this.shadowVyNext },
            densityR: { get: () => this.shadowDensityR },
            densityG: { get: () => this.shadowDensityG },
            densityB: { get: () => this.shadowDensityB },
            densityR0: { get: () => this.shadowDensityRNext },
            densityG0: { get: () => this.shadowDensityGNext },
            densityB0: { get: () => this.shadowDensityBNext }
        };

        for (const [key, descriptor] of Object.entries(descriptors)) {
            if (!Object.prototype.hasOwnProperty.call(this, key)) {
                Object.defineProperty(this, key, descriptor);
            }
        }
    }

    async _asyncInit(canvas) {
        const gpuInterface = await initWebGPU(canvas); // initWebGPU is from gpuUtils.js

        if (gpuInterface && gpuInterface.device) {
            this.adapter = gpuInterface.adapter;
            this.device = gpuInterface.device;
            this.context = gpuInterface.context;
            this.presentationFormat = gpuInterface.presentationFormat;
            this.gpuEnabled = true;
            console.log("GPUFluidField initialized with WebGPU.");
            
            // Initialize WebGPU resources now that the device is available
            // These methods will need to be refactored for WebGPU
            this._initializeWebGPUResources();
            return true;
        } else {
            console.error("Falling back to CPU fluid simulation due to WebGPU initialization failure within GPUFluidField.");
            this.gpuEnabled = false;
            this.gl = null; 
            return false;
        }
    }

    _initializeWebGPUResources() {
        if (!this.device) return;
        console.log("GPUFluidField: Initializing WebGPU-specific resources...");
        this._initGeometry();      
        this._initSampler();       
        this._initTextures();      
        this._initShadersAndPipelines(); 
        this._initComputeQueryResources(); // New call
    }

    _initComputeQueryResources() {
        if (!this.device) return;
        console.log("GPUFluidField: Initializing compute query resources...");

        // Define struct sizes (in bytes) - ensure these match WGSL layout eventually, considering padding.
        const pointQueryStructSizeBytes = 2 * Float32Array.BYTES_PER_ELEMENT; // vec2<f32> position
        const fluidQueryResultStructSizeBytes = (4 + 2 + 1 + 1 + 1) * Float32Array.BYTES_PER_ELEMENT; // density:vec4, vel:vec2, nutr:f32, light:f32, visc:f32
        const fluidGlobalsStructSizeBytes = (2 * Float32Array.BYTES_PER_ELEMENT) + (2 * Uint32Array.BYTES_PER_ELEMENT); // scale:vec2f, dims:vec2u
                                          // Ensure alignment for u32 if it follows vec2f, usually 16 bytes total is fine.

        this.pointQueriesBuffer = this.device.createBuffer({
            label: "Point Queries Input Buffer",
            size: MAX_SIMULTANEOUS_FLUID_QUERIES * pointQueryStructSizeBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.queryResultsBuffer = this.device.createBuffer({
            label: "Query Results GPU Buffer",
            size: MAX_SIMULTANEOUS_FLUID_QUERIES * fluidQueryResultStructSizeBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, // Source for copying to staging buffer
        });

        this.queryResultsStagingBuffer = this.device.createBuffer({
            label: "Query Results Staging Buffer (CPU Read)",
            size: MAX_SIMULTANEOUS_FLUID_QUERIES * fluidQueryResultStructSizeBytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, // Destination for copy, mappable for CPU read
        });

        // Uniform buffer for compute shader globals
        const initialFluidQueryUniforms = new ArrayBuffer(fluidGlobalsStructSizeBytes);
        // Use Float32Array and Uint32Array views for setting data correctly
        const uniformViewF32 = new Float32Array(initialFluidQueryUniforms);
        const uniformViewU32 = new Uint32Array(initialFluidQueryUniforms);

        // world_to_grid_scale: vec2<f32>
        uniformViewF32[0] = this.size / WORLD_WIDTH;  // (FLUID_GRID_SIZE_CONTROL / WORLD_WIDTH) effectively 1.0 / this.scaleX
        uniformViewF32[1] = this.size / WORLD_HEIGHT; // (FLUID_GRID_SIZE_CONTROL / WORLD_HEIGHT) effectively 1.0 / this.scaleY
        
        // grid_dimensions: vec2<u32> (offset by 2 floats = 8 bytes)
        uniformViewU32[2] = this.size; // this.size is FLUID_GRID_SIZE_CONTROL
        uniformViewU32[3] = this.size;

        this.fluidQueryUniformsBuffer = this.device.createBuffer({
            label: "Fluid Query Global Uniforms Buffer",
            size: fluidGlobalsStructSizeBytes, // Should be multiple of 16 for uniform buffers ideally
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true, // Create mapped to write initial values
        });
        new Uint8Array(this.fluidQueryUniformsBuffer.getMappedRange()).set(new Uint8Array(initialFluidQueryUniforms));
        this.fluidQueryUniformsBuffer.unmap();

        console.log("GPUFluidField: Compute query buffers and uniforms buffer created.");
    }

    _initGeometry() {
        if (this.device) { // WebGPU Path
            console.log("GPUFluidField._initGeometry() (WebGPU path)");
            const quadVertices = new Float32Array([
                // Positions (x, y)  TexCoords (u, v)
                -1.0, -1.0,           0.0, 0.0, // Triangle 1: bottom-left
                 1.0, -1.0,           1.0, 0.0, // bottom-right
                -1.0,  1.0,           0.0, 1.0, // top-left

                -1.0,  1.0,           0.0, 1.0, // Triangle 2: top-left
                 1.0, -1.0,           1.0, 0.0, // bottom-right
                 1.0,  1.0,           1.0, 1.0, // top-right
            ]);

            this.quadVertexBuffer = this.device.createBuffer({
                label: 'Quad Vertex Buffer',
                size: quadVertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Float32Array(this.quadVertexBuffer.getMappedRange()).set(quadVertices);
            this.quadVertexBuffer.unmap();

            this.vertexState = {
                buffers: [
                    {
                        arrayStride: 4 * Float32Array.BYTES_PER_ELEMENT, // (x,y,u,v) -> 4 floats
                        attributes: [
                            {
                                shaderLocation: 0, // @location(0) in basicVertex WGSL (a_position)
                                offset: 0,
                                format: 'float32x2',
                            },
                            {
                                shaderLocation: 1, // @location(1) in basicVertex WGSL (a_texCoord)
                                offset: 2 * Float32Array.BYTES_PER_ELEMENT, // Offset for texCoord (after 2 position floats)
                                format: 'float32x2',
                            },
                        ],
                    },
                ],
            };
            console.log("GPUFluidField: WebGPU quad vertex buffer and vertexState created.");

        } else if (this.gl) { // WebGL Path
            const gl = this.gl;
            // Fullscreen quad vertices (WebGL only needs positions, tex coords are often derived or passed differently)
            const quadVertices = new Float32Array([
                -1.0, -1.0,  1.0, -1.0, -1.0,  1.0,
                -1.0,  1.0,  1.0, -1.0,  1.0,  1.0,
            ]);
            this.quadVertexBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
            console.log("GPUFluidField: WebGL quad vertex buffer created.");
        }
    }

    _initSampler() {
        if (this.device) { // WebGPU Path
            console.log("GPUFluidField._initSampler() (WebGPU path)");
            this.sampler = this.device.createSampler({
                label: 'Linear ClampToEdge/Repeat Sampler', // Label can be more descriptive
                addressModeU: this.useWrapping ? 'repeat' : 'clamp-to-edge',
                addressModeV: this.useWrapping ? 'repeat' : 'clamp-to-edge',
                magFilter: 'linear',
                minFilter: 'linear',
                // mipmapFilter: 'nearest', // Mipmaps not typically used for these simulation textures
            });
            console.log("GPUFluidField: WebGPU sampler created.");
        } else if (this.gl) {
            // In WebGL, sampler parameters (like GL_TEXTURE_WRAP_S, GL_TEXTURE_MIN_FILTER)
            // are set directly on the texture object via texParameteri.
            // So, this method might be empty for the WebGL path, or those settings
            // can be confirmed/set in _createTexture for WebGL.
            // console.log("GPUFluidField._initSampler() (WebGL path - handled by texParameteri in _createTexture).");
        }
    }

    _initTextures() {
        if (this.device) { // WebGPU Path
            console.log("GPUFluidField._initTextures() (WebGPU path)");
            const textureFormat = 'rgba16float'; // A common and suitable format for fluid simulation data
            const textureUsage = GPUTextureUsage.TEXTURE_BINDING |
                               GPUTextureUsage.RENDER_ATTACHMENT |
                               GPUTextureUsage.COPY_DST | // Often useful for initial data or clearing
                               GPUTextureUsage.COPY_SRC;  // If you need to copy between textures or read back

            const textureDescriptorBase = {
                size: { width: this.size, height: this.size, depthOrArrayLayers: 1 },
                format: textureFormat,
                usage: textureUsage,
            };

            this.textures.velocityPing = this.device.createTexture({...textureDescriptorBase, label: 'Velocity Ping Texture'});
            this.textures.velocityPong = this.device.createTexture({...textureDescriptorBase, label: 'Velocity Pong Texture'});
            
            this.textures.densityPing  = this.device.createTexture({...textureDescriptorBase, label: 'Density Ping Texture'});
            this.textures.densityPong  = this.device.createTexture({...textureDescriptorBase, label: 'Density Pong Texture'});
            
            // For pressure and divergence calculations
            this.textures.pressurePing = this.device.createTexture({...textureDescriptorBase, label: 'Pressure Ping Texture'});
            this.textures.pressurePong = this.device.createTexture({...textureDescriptorBase, label: 'Pressure Pong Texture'});
            this.textures.divergence   = this.device.createTexture({...textureDescriptorBase, label: 'Divergence Texture'});

            console.log("GPUFluidField: WebGPU textures (velocity, density, pressure, divergence ping/pong) created.");

        } else if (this.gl) { // WebGL Path
            const gl = this.gl;
            // Create ping-pong textures for velocity and density
            this.textures.velocity = this._createTexture(gl.RGBA, gl.FLOAT, null);
            this.textures.velocityPrev = this._createTexture(gl.RGBA, gl.FLOAT, null); // For ping-ponging velocity

            this.textures.density = this._createTexture(gl.RGBA, gl.UNSIGNED_BYTE, null); // For color/dye
            this.textures.densityPrev = this._createTexture(gl.RGBA, gl.UNSIGNED_BYTE, null); // For ping-ponging density

            // Textures for pressure and divergence (intermediate calculations)
            this.textures.pressure = this._createTexture(gl.RGBA, gl.FLOAT, null);
            this.textures.pressurePrev = this._createTexture(gl.RGBA, gl.FLOAT, null); // For ping-ponging pressure in Jacobi
            this.textures.divergence = this._createTexture(gl.RGBA, gl.FLOAT, null);
            console.log("GPUFluidField: WebGL textures created.");
        }
    }

    _initFramebuffers() {
        // In WebGPU, framebuffers are implicitly defined by GPURenderPassDescriptor when you begin a render pass.
        // You specify which texture views to use as color attachments.
        // This WebGL-style _initFramebuffers might be removed or its logic absorbed into where render passes are set up.
        // For now, we can clear its WebGL content or leave it as a reminder.
        if (this.gl) {
            // Clear WebGL stuff if any was here
        }
        this.framebuffers = {}; // Reset or repurpose for WebGPU if needed (e.g., storing views)
        console.log("WebGPU framebuffers concept changes; this method needs refactoring.");
    }

    _initShadersAndPipelines() { // For WebGPU
        if (!this.device) return;
        console.log("GPUFluidField._initShadersAndPipelines() (WebGPU WGSL path)");
        this.programs = {}; // Reset to store GPURenderPipeline objects

        try {
            const basicVertexModule = this.device.createShaderModule({
                label: 'Basic Vertex Shader Module (WGSL)',
                code: this.wgslShaders.basicVertex,
            });

            const simulationTextureFormat = 'rgba16float'; // Format for offscreen simulation textures

            // Helper function to create pipelines to reduce redundancy
            const createSimulationPipeline = (label, fragmentShaderCode) => {
                const fragmentModule = this.device.createShaderModule({label: `${label} Frag Mod`, code: fragmentShaderCode});
                return this.device.createRenderPipeline({
                    label: `${label} Pipeline`,
                    layout: 'auto',
                    vertex: { 
                        module: basicVertexModule, 
                        entryPoint: 'main', 
                        buffers: this.vertexState.buffers 
                    },
                    fragment: { 
                        module: fragmentModule, 
                        entryPoint: 'main', 
                        targets: [{ format: simulationTextureFormat }] 
                    },
                    primitive: { topology: 'triangle-list' },
                });
            };

            // Create pipelines for each simulation step
            this.programs.advectionPipeline = createSimulationPipeline('Advection', this.wgslShaders.advectionFragment);
            this.programs.jacobiPipeline = createSimulationPipeline('Jacobi', this.wgslShaders.jacobiFragment);
            this.programs.divergencePipeline = createSimulationPipeline('Divergence', this.wgslShaders.divergenceFragment);
            this.programs.gradientSubtractPipeline = createSimulationPipeline('Gradient Subtract', this.wgslShaders.gradientSubtractFragment);
            
            // Splat pipeline might need different blend modes if we were doing true additive blending for velocity
            // For now, assuming it writes to an rgba16float texture like others.
            this.programs.splatPipeline = createSimulationPipeline('Splat', this.wgslShaders.splatFragment);

            // Display Pipeline (renders to canvas context)
            const displayFragmentModule = this.device.createShaderModule({label: 'Display Frag Mod', code: this.wgslShaders.displayFragment});
            this.programs.displayPipeline = this.device.createRenderPipeline({
                label: 'Display Pipeline',
                layout: 'auto',
                vertex: { 
                    module: basicVertexModule, 
                    entryPoint: 'main', 
                    buffers: this.vertexState.buffers 
                },
                fragment: { 
                    module: displayFragmentModule, 
                    entryPoint: 'main', 
                    targets: [{ format: this.presentationFormat }] // Uses canvas presentation format
                },
                primitive: { topology: 'triangle-list' },
            });

            // --- Compute Pipeline for Fluid Queries ---
            const fluidQueryComputeModule = this.device.createShaderModule({
                label: 'Fluid Query Compute Shader Module',
                code: this.wgslShaders.fluidQueryCompute,
            });

            this.programs.fluidQueryComputePipeline = this.device.createComputePipeline({
                label: 'Fluid Query Compute Pipeline',
                layout: 'auto', // Let WebGPU infer layout. For complex cases, create explicit GPUPipelineLayout.
                compute: {
                    module: fluidQueryComputeModule,
                    entryPoint: 'main',
                },
            });
            console.log("GPUFluidField: Fluid query compute pipeline created.");
            console.log("GPUFluidField: All WebGPU pipelines (render & compute) created successfully:", Object.keys(this.programs));

        } catch (error) {
            console.error("GPUFluidField: Error initializing WebGPU shaders/pipelines:", error);
            this.gpuEnabled = false; // Important: disable GPU if essential pipelines fail
        }
    }

    _toShadowGridCell(x, y, coordSpace = 'auto') {
        const rawX = Number(x);
        const rawY = Number(y);
        if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) return null;

        // Explicit coordSpace avoids ambiguous near-origin world coords (e.g. worldX=40, scaleX=10).
        // auto mode is retained for compatibility with legacy call sites.
        const isGrid = coordSpace === 'grid'
            || (coordSpace === 'auto' && Number.isInteger(rawX) && Number.isInteger(rawY) && rawX >= 0 && rawY >= 0 && rawX < this.size && rawY < this.size);

        const gx = isGrid ? Math.floor(rawX) : Math.floor(rawX / Math.max(1e-6, this.scaleX));
        const gy = isGrid ? Math.floor(rawY) : Math.floor(rawY / Math.max(1e-6, this.scaleY));

        const clampedX = Math.max(0, Math.min(this.size - 1, gx));
        const clampedY = Math.max(0, Math.min(this.size - 1, gy));
        return { gx: clampedX, gy: clampedY, idx: clampedX + clampedY * this.size };
    }

    _decayShadowFields() {
        const velDecay = Math.max(0, Math.min(1, 1 - this.dt * 2.0));
        const densityDecay = Math.max(0, Math.min(1, 1 - this.dt * 1.5));
        for (let i = 0; i < this.shadowVx.length; i++) {
            this.shadowVx[i] *= velDecay;
            this.shadowVy[i] *= velDecay;
            this.shadowDensityR[i] *= densityDecay;
            this.shadowDensityG[i] *= densityDecay;
            this.shadowDensityB[i] *= densityDecay;
        }
    }

    _sampleShadowBilinear(field, x, y) {
        const max = this.size - 1;
        const xClamped = Math.max(0, Math.min(max, x));
        const yClamped = Math.max(0, Math.min(max, y));

        const x0 = Math.floor(xClamped);
        const y0 = Math.floor(yClamped);
        const x1 = Math.min(max, x0 + 1);
        const y1 = Math.min(max, y0 + 1);

        const tx = xClamped - x0;
        const ty = yClamped - y0;

        const idx00 = x0 + y0 * this.size;
        const idx10 = x1 + y0 * this.size;
        const idx01 = x0 + y1 * this.size;
        const idx11 = x1 + y1 * this.size;

        const a = field[idx00] * (1 - tx) + field[idx10] * tx;
        const b = field[idx01] * (1 - tx) + field[idx11] * tx;
        return a * (1 - ty) + b * ty;
    }

    _applyShadowVelocitySplat(centerCell, amountX, amountY, strength = 15) {
        if (!centerCell) return;

        const radiusCells = Math.max(0, Math.min(4, (Number(strength) || 0) / 12));
        const radiusInt = Math.ceil(radiusCells);
        const sigma = Math.max(0.25, radiusCells * 0.75 + 0.25);
        const sigma2 = sigma * sigma;

        const baseVx = Number(amountX) || 0;
        const baseVy = Number(amountY) || 0;
        for (let oy = -radiusInt; oy <= radiusInt; oy++) {
            for (let ox = -radiusInt; ox <= radiusInt; ox++) {
                const gx = centerCell.gx + ox;
                const gy = centerCell.gy + oy;
                if (gx < 0 || gy < 0 || gx >= this.size || gy >= this.size) continue;

                const d2 = ox * ox + oy * oy;
                if (d2 > radiusCells * radiusCells + 1e-6) continue;
                const w = Math.exp(-d2 / (2 * sigma2));
                const idx = gx + gy * this.size;
                this.shadowVx[idx] = Math.max(-this.maxVelComponent, Math.min(this.maxVelComponent, this.shadowVx[idx] + baseVx * w));
                this.shadowVy[idx] = Math.max(-this.maxVelComponent, Math.min(this.maxVelComponent, this.shadowVy[idx] + baseVy * w));
            }
        }
    }

    _applyShadowDensitySplat(centerCell, r, g, b, strength = 0) {
        if (!centerCell) return;

        const blend = Math.max(0, Math.min(1, (Number(strength) || 0) / 80));
        const radiusCells = Math.max(0, Math.min(3, (Number(strength) || 0) / 60));
        const radiusInt = Math.ceil(radiusCells);
        const sigma = Math.max(0.2, radiusCells * 0.8 + 0.2);
        const sigma2 = sigma * sigma;

        const targetR = Number(r) || 0;
        const targetG = Number(g) || 0;
        const targetB = Number(b) || 0;

        for (let oy = -radiusInt; oy <= radiusInt; oy++) {
            for (let ox = -radiusInt; ox <= radiusInt; ox++) {
                const gx = centerCell.gx + ox;
                const gy = centerCell.gy + oy;
                if (gx < 0 || gy < 0 || gx >= this.size || gy >= this.size) continue;

                const d2 = ox * ox + oy * oy;
                if (d2 > radiusCells * radiusCells + 1e-6) continue;
                const w = Math.exp(-d2 / (2 * sigma2));
                const localBlend = blend * w;
                const idx = gx + gy * this.size;
                this.shadowDensityR[idx] = Math.max(0, Math.min(255, this.shadowDensityR[idx] + (targetR - this.shadowDensityR[idx]) * localBlend));
                this.shadowDensityG[idx] = Math.max(0, Math.min(255, this.shadowDensityG[idx] + (targetG - this.shadowDensityG[idx]) * localBlend));
                this.shadowDensityB[idx] = Math.max(0, Math.min(255, this.shadowDensityB[idx] + (targetB - this.shadowDensityB[idx]) * localBlend));
            }
        }
    }

    _advanceShadowFields() {
        const dt = Math.max(1e-6, Number(this.dt) || (1 / 60));
        const velDiffusion = Math.max(0, Math.min(0.25, (Number(this.viscosity) || 0.005) * 8));
        const densityDiffusion = Math.max(0, Math.min(0.25, (Number(this.diffusion) || 0.01) * 8));

        for (let gy = 0; gy < this.size; gy++) {
            for (let gx = 0; gx < this.size; gx++) {
                const idx = gx + gy * this.size;
                const vx = this.shadowVx[idx];
                const vy = this.shadowVy[idx];

                const prevX = gx - vx * dt;
                const prevY = gy - vy * dt;

                this.shadowVxNext[idx] = this._sampleShadowBilinear(this.shadowVx, prevX, prevY);
                this.shadowVyNext[idx] = this._sampleShadowBilinear(this.shadowVy, prevX, prevY);
                this.shadowDensityRNext[idx] = this._sampleShadowBilinear(this.shadowDensityR, prevX, prevY);
                this.shadowDensityGNext[idx] = this._sampleShadowBilinear(this.shadowDensityG, prevX, prevY);
                this.shadowDensityBNext[idx] = this._sampleShadowBilinear(this.shadowDensityB, prevX, prevY);
            }
        }

        if (velDiffusion > 0 || densityDiffusion > 0) {
            for (let gy = 1; gy < this.size - 1; gy++) {
                for (let gx = 1; gx < this.size - 1; gx++) {
                    const idx = gx + gy * this.size;
                    const up = idx - this.size;
                    const down = idx + this.size;
                    const left = idx - 1;
                    const right = idx + 1;

                    if (velDiffusion > 0) {
                        const vxNbr = 0.25 * (this.shadowVxNext[up] + this.shadowVxNext[down] + this.shadowVxNext[left] + this.shadowVxNext[right]);
                        const vyNbr = 0.25 * (this.shadowVyNext[up] + this.shadowVyNext[down] + this.shadowVyNext[left] + this.shadowVyNext[right]);
                        this.shadowVxNext[idx] = this.shadowVxNext[idx] * (1 - velDiffusion) + vxNbr * velDiffusion;
                        this.shadowVyNext[idx] = this.shadowVyNext[idx] * (1 - velDiffusion) + vyNbr * velDiffusion;
                    }

                    if (densityDiffusion > 0) {
                        const rNbr = 0.25 * (this.shadowDensityRNext[up] + this.shadowDensityRNext[down] + this.shadowDensityRNext[left] + this.shadowDensityRNext[right]);
                        const gNbr = 0.25 * (this.shadowDensityGNext[up] + this.shadowDensityGNext[down] + this.shadowDensityGNext[left] + this.shadowDensityGNext[right]);
                        const bNbr = 0.25 * (this.shadowDensityBNext[up] + this.shadowDensityBNext[down] + this.shadowDensityBNext[left] + this.shadowDensityBNext[right]);
                        this.shadowDensityRNext[idx] = this.shadowDensityRNext[idx] * (1 - densityDiffusion) + rNbr * densityDiffusion;
                        this.shadowDensityGNext[idx] = this.shadowDensityGNext[idx] * (1 - densityDiffusion) + gNbr * densityDiffusion;
                        this.shadowDensityBNext[idx] = this.shadowDensityBNext[idx] * (1 - densityDiffusion) + bNbr * densityDiffusion;
                    }
                }
            }
        }

        [this.shadowVx, this.shadowVxNext] = [this.shadowVxNext, this.shadowVx];
        [this.shadowVy, this.shadowVyNext] = [this.shadowVyNext, this.shadowVy];
        [this.shadowDensityR, this.shadowDensityRNext] = [this.shadowDensityRNext, this.shadowDensityR];
        [this.shadowDensityG, this.shadowDensityGNext] = [this.shadowDensityGNext, this.shadowDensityG];
        [this.shadowDensityB, this.shadowDensityBNext] = [this.shadowDensityBNext, this.shadowDensityB];

        this._decayShadowFields();
    }

    // --- Public API (matching CPU version where possible) ---
    step() {
        this._advanceShadowFields();
        if (!this.gpuEnabled) {
            return;
        }

        const stepStartMs = performance.now();

        if (this.device) { // WebGPU Path
            // console.log("GPUFluidField.step() WebGPU path running...");

            // 1. Advect Velocity Field
            this._runShaderPass(
                'advectionPipeline',
                {
                    u_velocityTexture: this.textures.velocityPing,
                    u_sourceTexture: this.textures.velocityPing,
                    u_dissipation: 1.0 // No dissipation for velocity itself
                },
                this.textures.velocityPong 
            );
            [this.textures.velocityPing, this.textures.velocityPong] = [this.textures.velocityPong, this.textures.velocityPing];

            // 2. Diffuse Velocity (Jacobi iterations)
            //    Pass parameters for Jacobi via inputTexturesSpec
            //    u_xTexture is iterated, u_bTexture is the original advected field (now in velocityPing)
            if (this.viscosity > 0) { // Only diffuse if viscosity is non-zero
                for (let i = 0; i < this.iterations; ++i) {
                    this._runShaderPass(
                        'jacobiPipeline',
                        {
                            u_xTexture: this.textures.velocityPong, // Previous iteration's result, or advected for first pass
                            u_bTexture: this.textures.velocityPing, // The field we are diffusing (constant for this Jacobi solve)
                            u_alpha: this.viscosity, 
                            u_rBeta: 1.0 / (4.0 + this.viscosity) // Matching GLSL logic
                        },
                        this.textures.velocityPing // Output to current Ping (will be Pong in next iteration due to swap)
                    );
                    [this.textures.velocityPing, this.textures.velocityPong] = [this.textures.velocityPong, this.textures.velocityPing];
                }
            }
            // After loop (or if viscosity is 0), velocityPing holds the (potentially) diffused velocity.

            // 3. Project (make velocity field divergence-free)
            //    a. Compute Divergence of the velocity field (currently in velocityPing)
            this._runShaderPass(
                'divergencePipeline',
                { 
                    u_velocityTexture: this.textures.velocityPing,
                    u_halfGridScale: 0.5 // Assuming dx=1 cell unit for divergence calculation
                },
                this.textures.divergence
            );

            //    b. Solve for Pressure (Poisson equation: Lap(P) = Div(V)) using Jacobi
            //       Clear pressurePing texture to zeros first.
            //       We use splatPipeline to clear by drawing a zero value with a large radius.
            this._runShaderPass(
                'splatPipeline',
                {
                    u_targetTexture: this.textures.pressurePing, // Texture to clear
                    u_point: [0.5, 0.5], // Center (doesn't matter much for full clear)
                    u_splatValue: [0,0,0,0], // Value to clear with
                    u_radius: 2.0 // Large radius in UV space to cover texture ( > sqrt(0.5^2+0.5^2) = ~0.707 for corners)
                },
                this.textures.pressurePing // Output to itself (just to have a target for the clear operation)
            );
            // Initialize pressurePong with zeros as well for the first iteration of Jacobi
            this._runShaderPass(
                'splatPipeline',
                {
                    u_targetTexture: this.textures.pressurePong, 
                    u_point: [0.5, 0.5], 
                    u_splatValue: [0,0,0,0], 
                    u_radius: 2.0 
                },
                this.textures.pressurePong 
            );

            for (let i = 0; i < this.iterations; ++i) {
                this._runShaderPass(
                    'jacobiPipeline',
                    {
                        u_xTexture: this.textures.pressurePong, // Previous pressure iteration
                        u_bTexture: this.textures.divergence,   // Source term (divergence field)
                        u_alpha: -1.0, // Corresponds to -dx^2 with dx=1 for Poisson problem
                        u_rBeta: 0.25  // Corresponds to 1/4 for the 4-point stencil Laplacian
                    },
                    this.textures.pressurePing
                );
                [this.textures.pressurePing, this.textures.pressurePong] = [this.textures.pressurePong, this.textures.pressurePing];
            }
            // After loop, pressurePing contains the solved pressure.

            //    c. Subtract Pressure Gradient from Velocity field
            //       Inputs: pressurePing (solved pressure), velocityPing (current velocity after diffusion)
            //       Output: velocityPong (then swapped to velocityPing)
            this._runShaderPass(
                'gradientSubtractPipeline',
                {
                    u_pressureTexture: this.textures.pressurePing,
                    u_velocityTexture: this.textures.velocityPing, 
                    u_gradientScale: 0.5 // Assuming dx=1 cell unit for gradient calculation
                },
                this.textures.velocityPong
            );
            [this.textures.velocityPing, this.textures.velocityPong] = [this.textures.velocityPong, this.textures.velocityPing];
            // Now velocityPing is the divergence-free velocity field.

            // 4. Advect Density Field (using the new divergence-free velocity in velocityPing)
            this._runShaderPass(
                'advectionPipeline',
                {
                    u_velocityTexture: this.textures.velocityPing, 
                    u_sourceTexture: this.textures.densityPing,    
                    u_dissipation: 1.0 - (FLUID_FADE_RATE * this.dt * 60.0) // Apply fade rate
                },
                this.textures.densityPong
            );
            [this.textures.densityPing, this.textures.densityPong] = [this.textures.densityPong, this.textures.densityPing];
            // Now densityPing is the advected density for the current frame.

            this._recordPerfSample('step', performance.now() - stepStartMs);
            return; // End of WebGPU path
        }

        // Existing WebGL Path
        if (!this.gl) return;
        const gl = this.gl;

        // Save current state for prev versions
        // WebGL uses texture swapping by re-assigning JS variables pointing to WebGLTexture objects.
        // For WebGPU, we'll swap this.textures.velocityPing and this.textures.velocityPong (which are GPUTexture objects).

        // 1. Advect Velocity
        this._runShaderPass('advection', {
            u_velocity: this.textures.velocityPrev,
            u_source: this.textures.velocityPrev,
            u_texelSize: [1.0 / this.size, 1.0 / this.size],
            u_dt: this.dt,
            u_dissipation: 1.0 
        }, this.framebuffers.velocityFbo);
        [this.textures.velocity, this.textures.velocityPrev] = [this.textures.velocityPrev, this.textures.velocity];
        [this.framebuffers.velocityFbo, this.framebuffers.velocityPrevFbo] = [this.framebuffers.velocityPrevFbo, this.framebuffers.velocityFbo];

        // 2. Diffuse Velocity (Jacobi iterations)
        if (this.viscosity > 0) {
             for (let i = 0; i < this.iterations; ++i) {
                this._runShaderPass('jacobi', {
                    u_x: this.textures.velocityPrev, 
                    u_b: this.textures.velocity,     
                    u_alpha: this.viscosity, 
                    u_rBeta: 1.0 / (4.0 + this.viscosity),
                    u_texelSize: [1.0/this.size, 1.0/this.size],
                }, this.framebuffers.velocityFbo); // Output to current velocityFbo
                [this.textures.velocity, this.textures.velocityPrev] = [this.textures.velocityPrev, this.textures.velocity];
                [this.framebuffers.velocityFbo, this.framebuffers.velocityPrevFbo] = [this.framebuffers.velocityPrevFbo, this.framebuffers.velocityFbo];
            }
        }

        // 3. Projection Step
        // 3a. Compute Divergence
        this._runShaderPass('divergence', {
            u_velocity: this.textures.velocityPrev, // Current velocity (diffused)
            u_texelSize: [1.0 / this.size, 1.0 / this.size]
        }, this.framebuffers.divergenceFbo);

        // 3b. Solve for Pressure (Jacobi iterations)
        // Clear pressure texture (pressurePrev) to zeros before iteration
        this._runShaderPass('splat', { 
            u_texture: this.textures.pressurePrev, // This texture will be the target of the clear
            u_point: [0.5, 0.5], u_color: [0.0, 0.0, 0.0], u_radius: 2.0 // Large radius, zero color
        }, this.framebuffers.pressurePrevFbo); // Output to pressurePrevFbo
        // Note: splat shader needs to be adjusted or a dedicated clear shader used for WebGL if it draws on u_texture
        // For now, assuming splat can clear by drawing over everything.

        for (let i = 0; i < this.iterations; ++i) { 
            this._runShaderPass('pressure', { // pressure uses jacobi shader program
                u_x: this.textures.pressurePrev,
                u_b: this.textures.divergence,
                u_alpha: -1.0, 
                u_rBeta: 0.25, 
                u_texelSize: [1.0/this.size, 1.0/this.size],
            }, this.framebuffers.pressureFbo);
            [this.textures.pressure, this.textures.pressurePrev] = [this.textures.pressurePrev, this.textures.pressure];
            [this.framebuffers.pressureFbo, this.framebuffers.pressurePrevFbo] = [this.framebuffers.pressurePrevFbo, this.framebuffers.pressureFbo];
        }

        // 3c. Subtract Pressure Gradient
        this._runShaderPass('gradientSubtract', {
            u_pressure: this.textures.pressurePrev, // Final solved pressure
            u_velocity: this.textures.velocity, // Velocity after diffusion
            u_texelSize: [1.0 / this.size, 1.0 / this.size]
        }, this.framebuffers.velocityPrevFbo); // Output to velocityPrevFbo
        // Swap to make velocityPrev the new current velocity
        [this.textures.velocity, this.textures.velocityPrev] = [this.textures.velocityPrev, this.textures.velocity];
        [this.framebuffers.velocityFbo, this.framebuffers.velocityPrevFbo] = [this.framebuffers.velocityPrevFbo, this.framebuffers.velocityFbo];

        // 4. Advect Density
        this._runShaderPass('advection', {
            u_velocityTexture: this.textures.velocityPrev, // Use the projected velocity
            u_sourceTexture: this.textures.densityPrev,
            u_texelSize: [1.0 / this.size, 1.0 / this.size],
            u_dt: this.dt,
            u_dissipation: 1.0 - (FLUID_FADE_RATE * this.dt * 60.0) 
        }, this.framebuffers.densityFbo);
        [this.textures.density, this.textures.densityPrev] = [this.textures.densityPrev, this.textures.density];
        [this.framebuffers.densityFbo, this.framebuffers.densityPrevFbo] = [this.framebuffers.densityPrevFbo, this.framebuffers.densityFbo];

        this._recordPerfSample('step', performance.now() - stepStartMs);
    }

    addDensity(x, y, r, g, b, strength) {
        const shadowCell = this._toShadowGridCell(x, y, 'grid');
        this._applyShadowDensitySplat(shadowCell, r, g, b, strength);

        if (!this.gpuEnabled) return;

        if (this.device) { // WebGPU Path
            // console.log(`WebGPU addDensity: x=${x}, y=${y}, color=(${r},${g},${b}), str=${strength}`);
            const SPLAT_RADIUS_WORLD_UNITS = strength; // Use strength directly as world radius for density
            const uvX = x / WORLD_WIDTH;
            const uvY = 1.0 - (y / WORLD_HEIGHT); // Y is often inverted in texture coords vs world

            const splatUniformsSpec = {
                u_targetTexture: this.textures.densityPing,
                u_point: [uvX, uvY],
                u_splatValue: [r / 255.0, g / 255.0, b / 255.0, 1.0], // Assuming alpha of 1 for splatted color
                u_radius: (SPLAT_RADIUS_WORLD_UNITS / Math.min(WORLD_WIDTH, WORLD_HEIGHT)) * 0.5 // Normalize radius, multiply by a factor for visibility
            };

            this._runShaderPass(
                'splatPipeline',
                splatUniformsSpec,
                this.textures.densityPong 
            );
            [this.textures.densityPing, this.textures.densityPong] = [this.textures.densityPong, this.textures.densityPing];

        } else if (this.gl) { // WebGL Path
            const gl = this.gl;
            const splatProgramInfo = this.programs.splat; 
            const actualSplatProgram = splatProgramInfo.program ? splatProgramInfo.program : splatProgramInfo;
            if(!actualSplatProgram) { console.error("WebGL splat program not found for addDensity"); return; }
            gl.useProgram(actualSplatProgram);

            const texCoordX = x / WORLD_WIDTH; 
            const texCoordY = 1.0 - (y / WORLD_HEIGHT); 

            gl.uniform2f(gl.getUniformLocation(actualSplatProgram, "u_point"), texCoordX, texCoordY);
            gl.uniform3f(gl.getUniformLocation(actualSplatProgram, "u_color"), r/255, g/255, b/255);
            gl.uniform1f(gl.getUniformLocation(actualSplatProgram, "u_radius"), (strength / Math.min(WORLD_WIDTH, WORLD_HEIGHT)) * 0.05); 

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.textures.density);
            gl.uniform1i(gl.getUniformLocation(actualSplatProgram, "u_texture"), 0);

            gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.densityPrevFbo);
            gl.viewport(0, 0, this.size, this.size);
            
            const positionAttributeLocation = gl.getAttribLocation(actualSplatProgram, "a_position");
            if (positionAttributeLocation !== -1) {
                gl.enableVertexAttribArray(positionAttributeLocation);
                gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
                gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
                gl.drawArrays(gl.TRIANGLES, 0, 6);
            }
            [this.textures.density, this.textures.densityPrev] = [this.textures.densityPrev, this.textures.density];
            [this.framebuffers.densityFbo, this.framebuffers.densityPrevFbo] = [this.framebuffers.densityPrevFbo, this.framebuffers.densityFbo];
        }
    }

    addVelocity(x, y, amountX, amountY, strength = 15) { // Added strength for radius
        const shadowCell = this._toShadowGridCell(x, y, 'grid');
        this._applyShadowVelocitySplat(shadowCell, amountX, amountY, strength);

        if (!this.gpuEnabled) return;

        if (this.device) { // WebGPU Path
            // console.log(`WebGPU addVelocity: x=${x}, y=${y}, amount=(${amountX},${amountY})`);
            const VELOCITY_SPLAT_RADIUS_WORLD = strength; 
            const VELOCITY_SPLAT_SCALE = 0.1; // Scale down impulse if it's too strong

            const uvX = x / WORLD_WIDTH;
            const uvY = 1.0 - (y / WORLD_HEIGHT); // Y is often inverted

            const splatUniformsSpec = {
                u_targetTexture: this.textures.velocityPing,
                u_point: [uvX, uvY],
                u_splatValue: [amountX * VELOCITY_SPLAT_SCALE, amountY * VELOCITY_SPLAT_SCALE, 0.0, 1.0], // Store velocity in RG, B can be 0, A for intensity if shader uses it
                u_radius: (VELOCITY_SPLAT_RADIUS_WORLD / Math.min(WORLD_WIDTH, WORLD_HEIGHT)) * 0.5 
            };

            this._runShaderPass(
                'splatPipeline',
                splatUniformsSpec,
                this.textures.velocityPong 
            );
            [this.textures.velocityPing, this.textures.velocityPong] = [this.textures.velocityPong, this.textures.velocityPing];

        } else if (this.gl) { // WebGL Path
            const gl = this.gl;
            const splatProgramInfo = this.programs.splat; 
            const actualSplatProgram = splatProgramInfo.program ? splatProgramInfo.program : splatProgramInfo;
            if(!actualSplatProgram) { console.error("WebGL splat program not found for addVelocity"); return; }
            gl.useProgram(actualSplatProgram);

            const texCoordX = x / WORLD_WIDTH;
            const texCoordY = 1.0 - (y / WORLD_HEIGHT);
            const VELOCITY_SPLAT_SCALE_GL = 0.005; // May need different scaling for GLSL float textures

            gl.uniform2f(gl.getUniformLocation(actualSplatProgram, "u_point"), texCoordX, texCoordY);
            gl.uniform3f(gl.getUniformLocation(actualSplatProgram, "u_color"), amountX * VELOCITY_SPLAT_SCALE_GL, amountY * VELOCITY_SPLAT_SCALE_GL, 0.0); 
            gl.uniform1f(gl.getUniformLocation(actualSplatProgram, "u_radius"), (strength / Math.min(WORLD_WIDTH, WORLD_HEIGHT)) * 0.025); 

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.textures.velocity);
            gl.uniform1i(gl.getUniformLocation(actualSplatProgram, "u_texture"), 0);

            gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.velocityPrevFbo);
            gl.viewport(0, 0, this.size, this.size);
            
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE); // Additive blending for velocity
            const positionAttributeLocation = gl.getAttribLocation(actualSplatProgram, "a_position");
            if (positionAttributeLocation !== -1) {
                 gl.enableVertexAttribArray(positionAttributeLocation);
                 gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
                 gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
                 gl.drawArrays(gl.TRIANGLES, 0, 6);
            }
            gl.disable(gl.BLEND);

            [this.textures.velocity, this.textures.velocityPrev] = [this.textures.velocityPrev, this.textures.velocity];
            [this.framebuffers.velocityFbo, this.framebuffers.velocityPrevFbo] = [this.framebuffers.velocityPrevFbo, this.framebuffers.velocityFbo];
        }
    }

    draw(canvasElement, viewportWidth, viewportHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
        if (!this.gpuEnabled) {
            return; 
        }

        const drawStartMs = performance.now();

        if (this.device && this.context) { // WebGPU path
            // console.log("GPUFluidField.draw() WebGPU path");
            if (!this.programs.displayPipeline || !this.textures.densityPing || !this.sampler || !this.quadVertexBuffer) {
                console.warn("WebGPU display resources not ready for drawing fluid. Pipeline or densityPing missing.");
                return;
            }

            // The _runShaderPass method is now set up to handle 'displayPipeline' and target the canvas context.
            // We pass a special marker or the context itself as outputTexture for display passes.
            this._runShaderPass(
                'displayPipeline',
                { u_displayTexture: this.textures.densityPing }, // inputTexturesSpec
                this.context // Special marker indicating to draw to the canvas context
            );
            this._recordPerfSample('draw', performance.now() - drawStartMs);

        } else if (this.gl) { // Fallback to existing WebGL draw
            // ... (Existing WebGL draw logic remains unchanged)
            const gl = this.gl;
            gl.bindFramebuffer(gl.FRAMEBUFFER, null); 
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clearColor(0, 0, 0, 0); 
            gl.clear(gl.COLOR_BUFFER_BIT);

            const displayProgramInfo = this.programs.display; // In WebGL, this might be an object { program: WebGLProgram }
            const actualDisplayProgram = displayProgramInfo.program ? displayProgramInfo.program : displayProgramInfo;

            if (!actualDisplayProgram) {
                console.error("WebGL display program not found.");
                return;
            }
            gl.useProgram(actualDisplayProgram);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.textures.density);
            const u_textureLocation = gl.getUniformLocation(actualDisplayProgram, "u_texture");
            gl.uniform1i(u_textureLocation, 0);

            const u_scaleLocation = gl.getUniformLocation(actualDisplayProgram, "u_scale");
            if(u_scaleLocation) gl.uniform2f(u_scaleLocation, 1.0, 1.0);
            const u_offsetLocation = gl.getUniformLocation(actualDisplayProgram, "u_offset");
            if(u_offsetLocation) gl.uniform2f(u_offsetLocation, 0.0, 0.0);

            const positionAttributeLocation = gl.getAttribLocation(actualDisplayProgram, "a_position");
            if (positionAttributeLocation !== -1 && this.quadVertexBuffer) {
                gl.enableVertexAttribArray(positionAttributeLocation);
                gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
                gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
                gl.drawArrays(gl.TRIANGLES, 0, 6);
            } else {
                 if(positionAttributeLocation === -1) console.error("a_position attribute not found in display shader (WebGL).");
                 if(!this.quadVertexBuffer) console.error("Quad vertex buffer not initialized for display (WebGL).");
            }
            this._recordPerfSample('draw', performance.now() - drawStartMs);
        }
    }

    clear() {
        this.shadowVx.fill(0);
        this.shadowVy.fill(0);
        this.shadowDensityR.fill(0);
        this.shadowDensityG.fill(0);
        this.shadowDensityB.fill(0);

        if (!this.gpuEnabled) return;

        if (this.device) { // WebGPU Path
            console.log("GPUFluidField.clear() WebGPU path");
            const texturesToClear = [
                this.textures.velocityPing, this.textures.velocityPong,
                this.textures.densityPing, this.textures.densityPong,
                this.textures.pressurePing, this.textures.pressurePong,
                this.textures.divergence
            ];

            for (const texture of texturesToClear) {
                if (texture) {
                    // Use the splatPipeline to effectively clear the texture by drawing a zero-value splat over the whole area.
                    // _runShaderPass already sets loadOp: 'clear', then draws. If splatValue is zero, result is zero.
                    this._runShaderPass(
                        'splatPipeline',
                        {
                            u_targetTexture: texture, // Input for the splat shader (though it's overwritten)
                            u_point: [0.5, 0.5],      // Center of splat
                            u_splatValue: [0, 0, 0, 0], // Value to splat (zeros)
                            u_radius: 2.0             // Large radius in UV (0-1) space to cover the texture
                        },
                        texture // Output to the same texture
                    );
                }
            }
            console.log("GPUFluidField: WebGPU textures cleared using splat pipeline.");

        } else if (this.gl) { // WebGL Path
            const gl = this.gl;
            const fbosToClear = [
                this.framebuffers.velocityFbo, this.framebuffers.velocityPrevFbo,
                this.framebuffers.densityFbo, this.framebuffers.densityPrevFbo,
                this.framebuffers.pressureFbo, this.framebuffers.pressurePrevFbo,
                this.framebuffers.divergenceFbo
            ];

            gl.clearColor(0, 0, 0, 0);
            for (let i = 0; i < fbosToClear.length; i++) {
                if (fbosToClear[i]) {
                    gl.bindFramebuffer(gl.FRAMEBUFFER, fbosToClear[i]);
                    gl.viewport(0, 0, this.size, this.size); 
                    gl.clear(gl.COLOR_BUFFER_BIT);
                }
            }
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            console.log("GPUFluidField textures cleared (WebGL).");
        }
    }

    // --- Coupling-oriented CPU shadow sampling methods ---
    IX(x, y) {
        let gx = Math.floor(Number(x));
        let gy = Math.floor(Number(y));

        if (!Number.isFinite(gx) || !Number.isFinite(gy)) return 0;

        if (this.useWrapping) {
            gx = ((gx % this.size) + this.size) % this.size;
            gy = ((gy % this.size) + this.size) % this.size;
        } else {
            gx = Math.max(0, Math.min(this.size - 1, gx));
            gy = Math.max(0, Math.min(this.size - 1, gy));
        }

        return gx + gy * this.size;
    }

    getDensityAtWorld(worldX, worldY) {
        const cell = this._toShadowGridCell(worldX, worldY, 'world');
        if (!cell) return [0, 0, 0, 0];
        const idx = cell.idx;
        return [this.shadowDensityR[idx], this.shadowDensityG[idx], this.shadowDensityB[idx], 1.0];
    }

    getVelocityAtWorld(worldX, worldY) {
        const cell = this._toShadowGridCell(worldX, worldY, 'world');
        if (!cell) return { vx: 0, vy: 0 };
        const idx = cell.idx;
        return { vx: this.shadowVx[idx], vy: this.shadowVy[idx] };
    }

    // For nutrient, light, viscosity, these are currently global CPU arrays.
    // If/when these also move to GPU textures controlled by GPUFluidField, they'll need similar methods.
    // For now, SoftBody will access the global CPU arrays directly for these specific fields.
    // However, to make the interface consistent if GPUFluidField were to manage them:
    getNutrientAtWorld(worldX, worldY) {
        if (!this.gpuEnabled || !this.device) return 1.0; // Default nutrient
        // This method assumes nutrientField is a GPU texture. Currently it's not.
        // If it were, logic similar to getDensityAtWorld would be here.
        // console.warn("GPUFluidField.getNutrientAtWorld() called, but nutrients are CPU-side. Returning default.");
        // For placeholder, let's simulate getting it from a hypothetical GPU texture.
        // We need to map worldX, worldY to grid coordinates first.
        const gx = Math.floor(worldX / this.scaleX); // scaleX is world_units_per_grid_cell for the GPU grid
        const gy = Math.floor(worldY / this.scaleY);
        // Here you would sample the GPU nutrient texture if it existed.
        // Since it doesn't, we return a default or try to access the global CPU one (which SoftBody will do anyway).
        // To avoid breaking SoftBody if it expects this method from a GPU field, return default.
        return 1.0 + (Math.random()-0.5)*0.2; // Default nutrient slightly varied
    }

    getLightAtWorld(worldX, worldY) {
        if (!this.gpuEnabled || !this.device) return 0.5; // Default light
        // console.warn("GPUFluidField.getLightAtWorld() called, but light is CPU-side. Returning default.");
        return 0.5 + (Math.random()-0.5)*0.1;
    }

    getViscosityAtWorld(worldX, worldY) {
        if (!this.gpuEnabled || !this.device) return 1.0; // Default viscosity multiplier
        // console.warn("GPUFluidField.getViscosityAtWorld() called, but viscosity is CPU-side. Returning default.");
        return 1.0;
    }

    // IX method is specific to CPU fluid field for direct array access. 
    // GPU field doesn't have a direct equivalent for external callers in the same way.
    // If SoftBody needs cell indices for GPUFluidField, it would calculate them itself using
    // this.scaleX, this.scaleY, this.size, as these are public.

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

    _recordPerfSample(kind, ms) {
        if (!this.perfStats) return;

        if (kind === 'step') {
            this.perfStats.lastStepMs = ms;
            this.perfStats.stepCount += 1;
            this.perfStats.stepAccumMs += ms;
            this.perfStats.stepMaxMs = Math.max(this.perfStats.stepMaxMs, ms);
        } else if (kind === 'draw') {
            this.perfStats.lastDrawMs = ms;
            this.perfStats.drawCount += 1;
            this.perfStats.drawAccumMs += ms;
            this.perfStats.drawMaxMs = Math.max(this.perfStats.drawMaxMs, ms);
        }

        const now = performance.now();
        if (now - this.perfStats.lastReportTs >= 5000) {
            const avgStep = this.perfStats.stepCount ? (this.perfStats.stepAccumMs / this.perfStats.stepCount) : 0;
            const avgDraw = this.perfStats.drawCount ? (this.perfStats.drawAccumMs / this.perfStats.drawCount) : 0;
            console.log(`[GPU PERF] avgStep=${avgStep.toFixed(2)}ms maxStep=${this.perfStats.stepMaxMs.toFixed(2)}ms avgDraw=${avgDraw.toFixed(2)}ms maxDraw=${this.perfStats.drawMaxMs.toFixed(2)}ms samples(step/draw)=${this.perfStats.stepCount}/${this.perfStats.drawCount}`);

            this.perfStats.stepCount = 0;
            this.perfStats.stepAccumMs = 0;
            this.perfStats.stepMaxMs = 0;
            this.perfStats.drawCount = 0;
            this.perfStats.drawAccumMs = 0;
            this.perfStats.drawMaxMs = 0;
            this.perfStats.lastReportTs = now;
        }
    }

    _runShaderPass(programNameOrInfo, inputTexturesSpec, outputTexture) {
        if (!this.gpuEnabled) return;

        if (this.device) { // WebGPU Path
            const pipelineName = typeof programNameOrInfo === 'string' ? programNameOrInfo : programNameOrInfo.label; 
            const pipeline = typeof programNameOrInfo === 'string' ? this.programs[programNameOrInfo] : programNameOrInfo;
            
            if (!pipeline) {
                console.error(`WebGPU pipeline not found for:`, programNameOrInfo);
                return;
            }

            let uniformBuffer;
            let uniformValues;
            let uniformBufferSize;
            let bindGroupEntries = [{ binding: 0, resource: this.sampler }];

            if (pipelineName === 'advectionPipeline') {
                const dissipationRate = inputTexturesSpec.u_dissipation !== undefined ? inputTexturesSpec.u_dissipation : 1.0;
                uniformValues = new Float32Array([1.0 / this.size, 1.0 / this.size, this.dt, dissipationRate]);
                uniformBufferSize = uniformValues.byteLength;
                if (!inputTexturesSpec.u_velocityTexture || !inputTexturesSpec.u_sourceTexture) { console.error('Advection: Missing textures.'); return; }
                bindGroupEntries.push({ binding: 1, resource: inputTexturesSpec.u_velocityTexture.createView() });
                bindGroupEntries.push({ binding: 2, resource: inputTexturesSpec.u_sourceTexture.createView() });
            } else if (pipelineName === 'jacobiPipeline') {
                if (typeof inputTexturesSpec.u_alpha !== 'number' || typeof inputTexturesSpec.u_rBeta !== 'number') { console.error('Jacobi: Missing u_alpha or u_rBeta.'); return; }
                uniformValues = new Float32Array([1.0 / this.size, 1.0 / this.size, inputTexturesSpec.u_alpha, inputTexturesSpec.u_rBeta]);
                uniformBufferSize = uniformValues.byteLength;
                if (!inputTexturesSpec.u_xTexture || !inputTexturesSpec.u_bTexture) { console.error('Jacobi: Missing textures.'); return; }
                bindGroupEntries.push({ binding: 1, resource: inputTexturesSpec.u_xTexture.createView() });
                bindGroupEntries.push({ binding: 2, resource: inputTexturesSpec.u_bTexture.createView() });
            } else if (pipelineName === 'divergencePipeline') {
                const halfGridScale = inputTexturesSpec.u_halfGridScale !== undefined ? inputTexturesSpec.u_halfGridScale : 0.5;
                uniformValues = new Float32Array([1.0 / this.size, 1.0 / this.size, halfGridScale]);
                uniformBufferSize = uniformValues.byteLength;
                if (!inputTexturesSpec.u_velocityTexture) { console.error('Divergence: Missing texture.'); return; }
                bindGroupEntries.push({ binding: 1, resource: inputTexturesSpec.u_velocityTexture.createView() });
            } else if (pipelineName === 'gradientSubtractPipeline') {
                const gradientScale = inputTexturesSpec.u_gradientScale !== undefined ? inputTexturesSpec.u_gradientScale : 0.5;
                uniformValues = new Float32Array([1.0 / this.size, 1.0 / this.size, gradientScale]);
                uniformBufferSize = uniformValues.byteLength;
                if (!inputTexturesSpec.u_pressureTexture || !inputTexturesSpec.u_velocityTexture) { console.error('GradientSubtract: Missing textures.'); return; }
                bindGroupEntries.push({ binding: 1, resource: inputTexturesSpec.u_pressureTexture.createView() });
                bindGroupEntries.push({ binding: 2, resource: inputTexturesSpec.u_velocityTexture.createView() });
            } else if (pipelineName === 'splatPipeline') {
                if (!inputTexturesSpec.u_point || !inputTexturesSpec.u_splatValue || typeof inputTexturesSpec.u_radius !== 'number') { console.error('Splat: Missing uniforms.'); return; }
                uniformValues = new Float32Array([
                    inputTexturesSpec.u_point[0], inputTexturesSpec.u_point[1],
                    inputTexturesSpec.u_splatValue[0], inputTexturesSpec.u_splatValue[1],
                    inputTexturesSpec.u_splatValue[2], inputTexturesSpec.u_splatValue[3],
                    inputTexturesSpec.u_radius, 0.0 
                ]);
                uniformBufferSize = uniformValues.byteLength;
                if (!inputTexturesSpec.u_targetTexture) { console.error('Splat: Missing target texture.'); return; }
                bindGroupEntries.push({ binding: 1, resource: inputTexturesSpec.u_targetTexture.createView() });
            } else if (pipelineName === 'displayPipeline') {
                if (!inputTexturesSpec.u_displayTexture) { console.error('Display: Missing texture.'); return; }
                bindGroupEntries.push({ binding: 1, resource: inputTexturesSpec.u_displayTexture.createView() });
                uniformValues = null; 
            } else {
                console.warn(`_runShaderPass WebGPU: Uniform/bind group setup for '${pipelineName}' not implemented.`);
                return;
            }

            if (uniformValues) {
                const alignedSize = Math.ceil(uniformBufferSize / 16) * 16;
                uniformBuffer = this.device.createBuffer({
                    label: `Uniforms for ${pipelineName}`,
                    size: alignedSize,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                this.device.queue.writeBuffer(uniformBuffer, 0, uniformValues, 0, uniformValues.length);
                
                let uniformBindingIndex = bindGroupEntries.length; // Default to next available binding
                bindGroupEntries.push({ binding: uniformBindingIndex, resource: { buffer: uniformBuffer } });
            }

            const bindGroupLayout = pipeline.getBindGroupLayout(0);
            const bindGroup = this.device.createBindGroup({
                label: `BindGroup for ${pipelineName}`,
                layout: bindGroupLayout,
                entries: bindGroupEntries,
            });

            const commandEncoder = this.device.createCommandEncoder({label: `${pipelineName} Encoder`});
            const renderPassDescriptor = {
                colorAttachments: [{
                    view: null, // Set below
                    loadOp: 'clear', 
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, 
                    storeOp: 'store',
                }],
            };

            if (pipelineName === 'displayPipeline') {
                // outputTexture for displayPipeline is a special marker like 'canvas' or this.context
                renderPassDescriptor.colorAttachments[0].view = this.context.getCurrentTexture().createView();
                renderPassDescriptor.colorAttachments[0].clearValue = { r: 0.0, g: 0.01, b: 0.02, a: 1.0 }; 
            } else {
                if (!outputTexture || !(outputTexture instanceof GPUTexture)) {
                    console.error(`Invalid outputTexture for pipeline ${pipelineName}`, outputTexture); return;
                }
                renderPassDescriptor.colorAttachments[0].view = outputTexture.createView();
            }

            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.setVertexBuffer(0, this.quadVertexBuffer);
            passEncoder.draw(6);
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
        } else if (this.gl) { 
            // ... (Existing WebGL logic remains unchanged)
        }
    }

    // --- NEW COMPUTE SHADER BASED DATA QUERY METHOD ---
    async queryFluidPropertiesForPoints(pointsToQuery) { // pointsToQuery is an array of {worldX, worldY, ...any other IDs}
        if (!this.device || !this.gpuEnabled || !this.programs.fluidQueryComputePipeline) {
            console.warn("GPUFluidField.queryFluidPropertiesForPoints: WebGPU not ready or compute pipeline missing.");
            // Fallback: return array of default values matching expected structure
            return pointsToQuery.map(() => ({
                density: [0, 0, 0, 0],
                velocity: { vx: 0, vy: 0 },
                nutrient: 1.0,
                light: 0.5,
                viscosity_multiplier: 1.0
            }));
        }

        const numPoints = Math.min(pointsToQuery.length, MAX_SIMULTANEOUS_FLUID_QUERIES);
        if (numPoints === 0) return [];

        // 1. Prepare Input Buffer Data
        const pointQueryStructSizeBytes = 2 * Float32Array.BYTES_PER_ELEMENT; // vec2<f32> position
        const inputDataArrayBuffer = new ArrayBuffer(numPoints * pointQueryStructSizeBytes);
        const inputDataView = new Float32Array(inputDataArrayBuffer);

        for (let i = 0; i < numPoints; i++) {
            inputDataView[i * 2 + 0] = pointsToQuery[i].worldX;
            // Assuming world Y up, texture Y up for now as per compute shader direct scale
            // If texture Y is inverted relative to world Y for sampling in render passes, 
            // ensure compute shader query also accounts for this if necessary or use consistent coords.
            // The compute shader uses `query.position * globals.world_to_grid_scale;`
            // If world_to_grid_scale.y is positive, it assumes Y is not inverted by this stage.
            inputDataView[i * 2 + 1] = pointsToQuery[i].worldY; 
        }
        this.device.queue.writeBuffer(this.pointQueriesBuffer, 0, inputDataArrayBuffer, 0, numPoints * pointQueryStructSizeBytes);

        // 2. Create Bind Group
        const bindGroup = this.device.createBindGroup({
            label: "Fluid Query Compute Bind Group",
            layout: this.programs.fluidQueryComputePipeline.getBindGroupLayout(0), // Assuming layout is at group 0
            entries: [
                { binding: 0, resource: { buffer: this.pointQueriesBuffer, size: numPoints * pointQueryStructSizeBytes } },
                { binding: 1, resource: { buffer: this.queryResultsBuffer, size: numPoints * this._getFluidQueryResultStructSizeBytes() } }, // Use a helper for size
                { binding: 2, resource: { buffer: this.fluidQueryUniformsBuffer } },
                { binding: 3, resource: this.textures.densityPing.createView() }, // Or current density texture
                { binding: 4, resource: this.textures.velocityPing.createView() }, // Or current velocity texture
                // TODO: Add nutrient, light, viscosity texture views when they are on GPU
            ],
        });

        // 3. Dispatch Compute Shader
        const commandEncoder = this.device.createCommandEncoder({ label: "Fluid Query Command Encoder" });
        const passEncoder = commandEncoder.beginComputePass({ label: "Fluid Query Compute Pass" });
        passEncoder.setPipeline(this.programs.fluidQueryComputePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const workgroupSize = 64; // Must match @workgroup_size in WGSL
        passEncoder.dispatchWorkgroups(Math.ceil(numPoints / workgroupSize));
        passEncoder.end();

        // 4. Copy Results to Staging Buffer
        const resultsBufferSize = numPoints * this._getFluidQueryResultStructSizeBytes();
        commandEncoder.copyBufferToBuffer(
            this.queryResultsBuffer, 0,       // Source
            this.queryResultsStagingBuffer, 0, // Destination
            resultsBufferSize                 // Size
        );

        // 5. Submit and Map
        this.device.queue.submit([commandEncoder.finish()]);
        
        await this.queryResultsStagingBuffer.mapAsync(GPUMapMode.READ, 0, resultsBufferSize);
        const resultsArrayBuffer = this.queryResultsStagingBuffer.getMappedRange(0, resultsBufferSize);
        const outputDataView = new Float32Array(resultsArrayBuffer.slice(0)); // Create a copy for unmapping
        this.queryResultsStagingBuffer.unmap();

        // 6. Process and Return Results
        const results = [];
        const resultFloatsPerPoint = this._getFluidQueryResultStructSizeBytes() / Float32Array.BYTES_PER_ELEMENT;
        for (let i = 0; i < numPoints; i++) {
            const offset = i * resultFloatsPerPoint;
            results.push({
                density: [outputDataView[offset + 0], outputDataView[offset + 1], outputDataView[offset + 2], outputDataView[offset + 3]],
                velocity: { vx: outputDataView[offset + 4], vy: outputDataView[offset + 5] },
                nutrient: outputDataView[offset + 6],
                light: outputDataView[offset + 7],
                viscosity_multiplier: outputDataView[offset + 8],
                // originalQueryData: pointsToQuery[i] // Optionally include original query data if needed for mapping back
            });
        }
        return results;
    }

    // Helper to get struct size, useful if it becomes more complex or padded
    _getFluidQueryResultStructSizeBytes() {
        return (4 + 2 + 1 + 1 + 1) * Float32Array.BYTES_PER_ELEMENT; // density:vec4, vel:vec2, nutr:f32, light:f32, visc:f32
    }
}