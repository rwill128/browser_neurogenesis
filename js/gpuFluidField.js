class GPUFluidField {
    // WGSL Shader Strings (New)
    wgslShaders = {
        basicVertex: `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
};

@vertex
fn main(@location(0) a_position: vec2<f32>,
        @location(1) a_texCoord: vec2<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(a_position, 0.0, 1.0);
    output.texCoord = a_texCoord;
    return output;
}
        `,
        advectionFragment: `
struct AdvectionUniforms {
    u_texelSize: vec2<f32>,
    u_dt: f32,
    u_dissipation: f32,
};

@group(0) @binding(0) var u_sampler: sampler;
@group(0) @binding(1) var u_velocityTexture: texture_2d<f32>;
@group(0) @binding(2) var u_sourceTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> advectionUniforms: AdvectionUniforms;

@fragment
fn main(@location(0) fragTexCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let velocity = textureSample(u_velocityTexture, u_sampler, fragTexCoord).xy;
    let prevTexCoord = fragTexCoord - velocity * advectionUniforms.u_dt * advectionUniforms.u_texelSize;
    var advectedValue = textureSample(u_sourceTexture, u_sampler, prevTexCoord);
    advectedValue = advectedValue * advectionUniforms.u_dissipation;
    return advectedValue;
}
        `,
        jacobiFragment: `
struct JacobiUniforms {
    u_texelSize: vec2<f32>,
    u_alpha: f32,
    u_rBeta: f32,
};

@group(0) @binding(0) var u_sampler: sampler;
@group(0) @binding(1) var u_xTexture: texture_2d<f32>;
@group(0) @binding(2) var u_bTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> jacobiUniforms: JacobiUniforms;

@fragment
fn main(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let W = textureSample(u_xTexture, u_sampler, texCoord - vec2<f32>(jacobiUniforms.u_texelSize.x, 0.0));
    let E = textureSample(u_xTexture, u_sampler, texCoord + vec2<f32>(jacobiUniforms.u_texelSize.x, 0.0));
    let N = textureSample(u_xTexture, u_sampler, texCoord + vec2<f32>(0.0, jacobiUniforms.u_texelSize.y));
    let S = textureSample(u_xTexture, u_sampler, texCoord - vec2<f32>(0.0, jacobiUniforms.u_texelSize.y));
    let bC = textureSample(u_bTexture, u_sampler, texCoord);
    return (W + E + N + S + jacobiUniforms.u_alpha * bC) * jacobiUniforms.u_rBeta;
}
        `,
        divergenceFragment: `
struct DivergenceUniforms {
    u_texelSize: vec2<f32>,
    u_halfGridScale: f32, 
};

@group(0) @binding(0) var u_sampler: sampler;
@group(0) @binding(1) var u_velocityTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> divergenceUniforms: DivergenceUniforms;

@fragment
fn main(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let velW = textureSample(u_velocityTexture, u_sampler, texCoord - vec2<f32>(divergenceUniforms.u_texelSize.x, 0.0)).x;
    let velE = textureSample(u_velocityTexture, u_sampler, texCoord + vec2<f32>(divergenceUniforms.u_texelSize.x, 0.0)).x;
    let velS = textureSample(u_velocityTexture, u_sampler, texCoord - vec2<f32>(0.0, divergenceUniforms.u_texelSize.y)).y;
    let velN = textureSample(u_velocityTexture, u_sampler, texCoord + vec2<f32>(0.0, divergenceUniforms.u_texelSize.y)).y;
    let divergence = divergenceUniforms.u_halfGridScale * (velE - velW + velN - velS);
    return vec4<f32>(divergence, 0.0, 0.0, 1.0);
}
        `,
        gradientSubtractFragment: `
struct GradSubUniforms {
    u_texelSize: vec2<f32>,
    u_gradientScale: f32,
};

@group(0) @binding(0) var u_sampler: sampler;
@group(0) @binding(1) var u_pressureTexture: texture_2d<f32>;
@group(0) @binding(2) var u_velocityTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> gradSubUniforms: GradSubUniforms;

@fragment
fn main(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let pressureW = textureSample(u_pressureTexture, u_sampler, texCoord - vec2<f32>(gradSubUniforms.u_texelSize.x, 0.0)).x;
    let pressureE = textureSample(u_pressureTexture, u_sampler, texCoord + vec2<f32>(gradSubUniforms.u_texelSize.x, 0.0)).x;
    let pressureS = textureSample(u_pressureTexture, u_sampler, texCoord - vec2<f32>(0.0, gradSubUniforms.u_texelSize.y)).x;
    let pressureN = textureSample(u_pressureTexture, u_sampler, texCoord + vec2<f32>(0.0, gradSubUniforms.u_texelSize.y)).x;
    let currentVelocity = textureSample(u_velocityTexture, u_sampler, texCoord).xy;
    let gradP = vec2<f32>(
        (pressureE - pressureW) * gradSubUniforms.u_gradientScale,
        (pressureN - pressureS) * gradSubUniforms.u_gradientScale
    );
    return vec4<f32>(currentVelocity - gradP, 0.0, 1.0);
}
        `,
        splatFragment: `
struct SplatAddUniforms {
    u_point: vec2<f32>,
    u_splatValue: vec4<f32>,
    u_radius: f32,
};

@group(0) @binding(0) var u_sampler: sampler;
@group(0) @binding(1) var u_targetTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> splatUniforms: SplatAddUniforms;

@fragment
fn main(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let dist_sq = dot(texCoord - splatUniforms.u_point, texCoord - splatUniforms.u_point);
    let radius_sq = splatUniforms.u_radius * splatUniforms.u_radius;
    var outputColor = textureSample(u_targetTexture, u_sampler, texCoord);
    if (dist_sq < radius_sq) {
        let intensity_factor = 1.0 - smoothstep(0.0, radius_sq, dist_sq);
        outputColor += splatUniforms.u_splatValue * intensity_factor;
    }
    return outputColor;
}
        `,
        displayFragment: `
@group(0) @binding(0) var u_sampler: sampler;
@group(0) @binding(1) var u_displayTexture: texture_2d<f32>;

@fragment
fn main(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(u_displayTexture, u_sampler, texCoord);
}
        `,
    };

    constructor(canvas, size, diffusion, viscosity, dt, scaleX, scaleY) {
        this.canvas = canvas;
        this.size = Math.round(size);
        this.diffusion = diffusion;
        this.viscosity = viscosity;
        this.dt = dt;
        this.iterations = 4;
        this.gpuEnabled = false;
        
        // CPU-compatible properties for interface compatibility
        this.scaleX = scaleX;
        this.scaleY = scaleY;
        this.useWrapping = false;
        this.maxVelComponent = MAX_FLUID_VELOCITY_COMPONENT;
        
        // CPU-compatible arrays - these will be synced from GPU when needed
        this.Vx = new Float32Array(this.size * this.size).fill(0);
        this.Vy = new Float32Array(this.size * this.size).fill(0);
        this.densityR = new Float32Array(this.size * this.size).fill(0);
        this.densityG = new Float32Array(this.size * this.size).fill(0);
        this.densityB = new Float32Array(this.size * this.size).fill(0);
        
        // GPU-specific properties
        this.device = null;
        this.context = null;
        this.presentationFormat = null;
        this.gl = null;
        
        this.textures = {};
        this.framebuffers = {};
        this.programs = {};
        this.sampler = null;
        this.quadVertexBuffer = null;
        this.vertexState = null;
        
        // Compute query resources
        this.fluidQueryComputePipeline = null;
        this.queryUniformBuffer = null;
        this.queryResultBuffer = null;
        this.queryData = new Float32Array(MAX_SIMULTANEOUS_FLUID_QUERIES * 4); // x, y, vx, vy per query
        
        this._initPromise = this._asyncInit(canvas);
    }

    // CPU-compatible IX method for grid index calculation
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
        this._initGeometry();      // Sets up quadVertexBuffer and vertexState
        this._initSampler();       // Sets up a common sampler
        this._initTextures();      // Sets up WebGPU textures (density, velocity etc.)
        // Framebuffers are part of render pass descriptors, so _initFramebuffers might be refactored/removed
        this._initShadersAndPipelines(); // Compiles WGSL, creates pipelines
        
        // Initialize compute query resources
        console.log("GPUFluidField: Initializing compute query resources...");
        this._initComputeQueryResources();
        
        console.log("GPUFluidField: WebGPU initialization complete.");
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

            console.log("GPUFluidField: All WebGPU pipelines created successfully:", Object.keys(this.programs));

        } catch (error) {
            console.error("GPUFluidField: Error initializing WebGPU shaders/pipelines:", error);
            this.gpuEnabled = false; // Important: disable GPU if essential pipelines fail
        }
    }

    _initComputeQueryResources() {
        if (!this.device) return;
        
        try {
            // Create compute shader for fluid queries
            const computeShaderModule = this.device.createShaderModule({
                label: 'Fluid Query Compute Shader',
                code: `
                    struct QueryData {
                        x: f32,
                        y: f32,
                        vx: f32,
                        vy: f32,
                    }
                    
                    @group(0) @binding(0) var<storage, read_write> queries: array<QueryData>;
                    @group(0) @binding(1) var velocityTexture: texture_2d<f32>;
                    @group(0) @binding(2) var densityTexture: texture_2d<f32>;
                    @group(0) @binding(3) var linearSampler: sampler;
                    
                    @compute @workgroup_size(64)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let index = global_id.x;
                        if (index >= ${MAX_SIMULTANEOUS_FLUID_QUERIES}) {
                            return;
                        }
                        
                        let query = queries[index];
                        let uv = vec2<f32>(query.x, query.y);
                        
                        // Sample velocity at query position
                        let velocity = textureSampleLevel(velocityTexture, linearSampler, uv, 0.0);
                        
                        // Store results back
                        queries[index].vx = velocity.x;
                        queries[index].vy = velocity.y;
                    }
                `
            });
            
            // Create compute pipeline
            this.fluidQueryComputePipeline = this.device.createComputePipeline({
                label: 'Fluid Query Compute Pipeline',
                layout: 'auto',
                compute: {
                    module: computeShaderModule,
                    entryPoint: 'main',
                }
            });
            
            // Create buffers for query data
            this.queryUniformBuffer = this.device.createBuffer({
                label: 'Query Uniform Buffer',
                size: MAX_SIMULTANEOUS_FLUID_QUERIES * 4 * 4, // 4 floats per query * 4 bytes per float
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            
            this.queryResultBuffer = this.device.createBuffer({
                label: 'Query Result Buffer',
                size: MAX_SIMULTANEOUS_FLUID_QUERIES * 4 * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            
            console.log("GPUFluidField: Compute query resources initialized successfully.");
            
        } catch (error) {
            console.error("GPUFluidField: Error initializing compute query resources:", error);
            this.gpuEnabled = false;
        }
    }

    // Method to sync GPU data back to CPU arrays for compatibility
    async syncFromGPU() {
        if (!this.device || !this.gpuEnabled) return;
        
        try {
            // This would be used to read back velocity and density data from GPU textures
            // For now, we'll implement a simplified version
            console.log("GPUFluidField: GPU to CPU sync not fully implemented yet");
        } catch (error) {
            console.error("GPUFluidField: Error syncing from GPU:", error);
        }
    }
    
    // Method to perform batched fluid queries using compute shader
    async queryFluidData(queries) {
        if (!this.device || !this.fluidQueryComputePipeline || !queries.length) {
            return [];
        }
        
        try {
            // Prepare query data
            const queryData = new Float32Array(queries.length * 4);
            for (let i = 0; i < queries.length; i++) {
                const query = queries[i];
                queryData[i * 4] = query.x / WORLD_WIDTH; // Convert to UV coordinates
                queryData[i * 4 + 1] = 1.0 - (query.y / WORLD_HEIGHT); // Y-flip for UV
                queryData[i * 4 + 2] = 0; // vx (to be filled by compute shader)
                queryData[i * 4 + 3] = 0; // vy (to be filled by compute shader)
            }
            
            // Upload query data
            this.device.queue.writeBuffer(this.queryUniformBuffer, 0, queryData);
            
            // Create bind group for compute
            const bindGroup = this.device.createBindGroup({
                layout: this.fluidQueryComputePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.queryUniformBuffer } },
                    { binding: 1, resource: this.textures.velocityPing.createView() },
                    { binding: 2, resource: this.textures.densityPing.createView() },
                    { binding: 3, resource: this.sampler },
                ],
            });
            
            // Dispatch compute shader
            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.fluidQueryComputePipeline);
            computePass.setBindGroup(0, bindGroup);
            const workgroupCount = Math.ceil(queries.length / 64);
            computePass.dispatchWorkgroups(workgroupCount);
            computePass.end();
            
            // Copy results to readable buffer
            commandEncoder.copyBufferToBuffer(
                this.queryUniformBuffer, 0,
                this.queryResultBuffer, 0,
                queries.length * 4 * 4
            );
            
            this.device.queue.submit([commandEncoder.finish()]);
            
            // Read back results
            await this.queryResultBuffer.mapAsync(GPUMapMode.READ);
            const resultData = new Float32Array(this.queryResultBuffer.getMappedRange());
            
            const results = [];
            for (let i = 0; i < queries.length; i++) {
                results.push({
                    vx: resultData[i * 4 + 2],
                    vy: resultData[i * 4 + 3]
                });
            }
            
            this.queryResultBuffer.unmap();
            return results;
            
        } catch (error) {
            console.error("GPUFluidField: Error in queryFluidData:", error);
            return [];
        }
    }

    // --- Public API (matching CPU version where possible) ---
    step() {
        if (!this.gpuEnabled) {
            return;
        }

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
    }

    addDensity(x, y, r, g, b, strength) {
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
        }
    }

    clear() {
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
                // Fix uniform buffer size alignment - ensure minimum 48 bytes
                const minSize = 48; // WebGPU minimum binding size requirement
                const alignedSize = Math.max(minSize, Math.ceil(uniformBufferSize / 16) * 16);
                uniformBuffer = this.device.createBuffer({
                    label: `Uniforms for ${pipelineName}`,
                    size: alignedSize,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                
                // Create a properly sized array for the aligned buffer
                const paddedValues = new Float32Array(alignedSize / 4);
                paddedValues.set(uniformValues, 0); // Copy original values to the beginning
                
                this.device.queue.writeBuffer(uniformBuffer, 0, paddedValues, 0, paddedValues.length);
                
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
}