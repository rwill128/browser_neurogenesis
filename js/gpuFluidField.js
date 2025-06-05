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
};

@group(0) @binding(0) var u_sampler: sampler;
@group(0) @binding(1) var u_velocityTexture: texture_2d<f32>;
@group(0) @binding(2) var u_sourceTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> advectionUniforms: AdvectionUniforms;

@fragment
fn main(@location(0) fragTexCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let velocity = textureSample(u_velocityTexture, u_sampler, fragTexCoord).xy;
    let prevTexCoord = fragTexCoord - velocity * advectionUniforms.u_dt * advectionUniforms.u_texelSize;
    let advectedValue = textureSample(u_sourceTexture, u_sampler, prevTexCoord);
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

        // Store the promise for external awaiting if needed
        this._initPromise = this._asyncInit(canvas);
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

    // --- Public API (matching CPU version where possible) ---
    step() {
        if (!this.gpuEnabled) {
            if (this.gl) { /* console.warn("GPU step called but WebGL context found, check logic") */ } 
            // else console.warn("GPU step called but GPU not enabled and no GL context.");
            return; // Or call a CPU step if that's the fallback
        }
        // WebGPU step logic will replace/augment WebGL logic
        // This will involve multiple _runShaderPass calls with different pipelines (shaders)
        // and ping-ponging between textures.

        // Placeholder: an actual WebGPU step would look more like a sequence of dispatches
        // for compute shaders or render passes for fragment shaders.

        // Example sequence (conceptual, actual calls to _runShaderPass or similar)
        // 1. Advect velocity field (uses vel_ping, writes to vel_pong)
        //    this._runShaderPass(this.programs.advection, { u_velocityTexture: this.textures.velocityPing, u_sourceTexture: this.textures.velocityPing }, this.framebuffers.velocityPongFbo);
        //    [this.textures.velocityPing, this.textures.velocityPong] = [this.textures.velocityPong, this.textures.velocityPing]; // Swap

        // 2. Diffuse velocity field (Jacobi iterations)
        // 3. Add external forces (if any, not in original simplified sim)
        // 4. Project (remove divergence)
        //    - Calculate divergence
        //    - Solve pressure (Jacobi iterations)
        //    - Subtract pressure gradient from velocity
        // 5. Advect density field (uses new vel_ping (after projection), density_ping, writes to density_pong)
        //    this._runShaderPass(this.programs.advection, { u_velocityTexture: this.textures.velocityPing, u_sourceTexture: this.textures.densityPing }, this.framebuffers.densityPongFbo);
        //    [this.textures.densityPing, this.textures.densityPong] = [this.textures.densityPong, this.textures.densityPing]; // Swap
        
        // (Existing WebGL code remains for now)
        if (this.gl) {
            const gl = this.gl;
            const temp = this.textures.velocityPrev;
            this.textures.velocityPrev = this.textures.velocity;
            this.textures.velocity = temp;
            const tempFb = this.framebuffers.velocityPrevFbo;
            this.framebuffers.velocityPrevFbo = this.framebuffers.velocityFbo;
            this.framebuffers.velocityFbo = tempFb;

            // Advect velocity
            this._runShaderPass(this.programs.advection, 
                { u_texture: this.textures.velocityPrev, u_velocity: this.textures.velocityPrev, u_dt: this.dt, u_texelSize: [1/this.size, 1/this.size] },
                this.framebuffers.velocityFbo);

            // Diffuse velocity (Jacobi iterations)
            for (let i = 0; i < this.iterations; ++i) {
                const temp2 = this.textures.velocityPrev;
                this.textures.velocityPrev = this.textures.velocity;
                this.textures.velocity = temp2;
                const tempFb2 = this.framebuffers.velocityPrevFbo;
                this.framebuffers.velocityPrevFbo = this.framebuffers.velocityFbo;
                this.framebuffers.velocityFbo = tempFb2;
                this._runShaderPass(this.programs.jacobi, 
                    { u_x: this.textures.velocityPrev, u_b: this.textures.velocityPrev, u_alpha: this.diffusion, u_rBeta: 1.0 / (4.0 + this.diffusion) }, 
                    this.framebuffers.velocityFbo);
            }
            // ... (rest of WebGL step logic) ...
        }
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

    draw(canvasElement, viewportWidth, viewportHeight, viewOffsetXWorld, viewOffsetYWorld, currentZoom) {
        if (!this.gpuEnabled && !this.gl) {
            // If neither WebGPU nor WebGL is enabled, potentially call CPU draw or do nothing
            // console.log("GPUFluidField.draw: Neither WebGPU nor WebGL initialized.");
            return;
        }

        if (this.device && this.context) { // WebGPU path
            // 1. Get the current texture to display (e.g., this.textures.densityPing or a specific display texture).
            // 2. Create a command encoder.
            // 3. Begin a render pass targeting the canvas context's current texture view.
            //    (this.context.getCurrentTexture().createView())
            // 4. Use a display pipeline (vertex shader: basicVertex, fragment shader: simple texture lookup).
            // 5. Create a bind group for the display texture and sampler.
            // 6. Set pipeline, bind group, vertex buffer.
            // 7. Draw quad.
            // 8. End pass, submit command encoder.
            // console.log("GPUFluidField.draw using WebGPU (placeholder).");

            // This is a very simplified placeholder. Actual drawing involves render passes.
            const commandEncoder = this.device.createCommandEncoder();
            const passDescriptor = {
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    loadOp: 'clear',
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }, // Black background for the canvas
                    storeOp: 'store',
                }],
            };
            const passEncoder = commandEncoder.beginRenderPass(passDescriptor);
            
            // TODO: Set up a display pipeline (this.programs.displayPipeline)
            // if (this.programs.displayPipeline && this.textures.densityPing && this.sampler) {
            //     passEncoder.setPipeline(this.programs.displayPipeline);
            //     const displayBindGroup = this.device.createBindGroup({
            //         layout: this.programs.displayPipeline.getBindGroupLayout(0),
            //         entries: [
            //             { binding: 0, resource: this.sampler },
            //             { binding: 1, resource: this.textures.densityPing.createView() },
            //             // Add uniforms for viewport transform if needed
            //         ],
            //     });
            //     passEncoder.setBindGroup(0, displayBindGroup);
            //     passEncoder.setVertexBuffer(0, this.quadVertexBuffer);
            //     passEncoder.draw(6); // Draw the quad
            // } else {
            //     console.warn("Display pipeline or textures not ready for WebGPU draw.");
            // }

            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);

        } else if (this.gl) { // Fallback to existing WebGL draw
            const gl = this.gl;
            gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Draw to canvas
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clearColor(0,0,0,0); // Clear with transparent for fluid overlay
            gl.clear(gl.COLOR_BUFFER_BIT);

            this._runShaderPass(this.programs.display, 
                { u_texture: this.textures.density, u_scale: [1,1], u_offset: [0,0] }, 
                null); // null fbo means draw to canvas
        }
    }

    clear() {
        if (!this.gpuEnabled && !this.gl) return;
        // For WebGPU, this would involve clearing textures, perhaps with a clear render pass
        // or by writing zero data with device.queue.writeTexture.
        if (this.gl) {
            const gl = this.gl;
            // ... (existing WebGL clear code)
        }
         // Placeholder for WebGPU texture clearing
        if (this.device) {
            // Example: Clear densityPing texture
            // This could be done with a render pass that clears, or queue.writeTexture if small enough
            // For large textures, a clear pass is often better.
            if (this.textures.densityPing) {
                const commandEncoder = this.device.createCommandEncoder();
                const passDescriptor = {
                    colorAttachments: [{
                        view: this.textures.densityPing.createView(),
                        loadOp: 'clear',
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                        storeOp: 'store',
                    }],
                };
                const passEncoder = commandEncoder.beginRenderPass(passDescriptor);
                passEncoder.end();
                // Repeat for other textures (densityPong, velocityPing, velocityPong)
                // ...
                this.device.queue.submit([commandEncoder.finish()]);
            }
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
            const pipeline = typeof programNameOrInfo === 'string' ? this.programs[programNameOrInfo] : programNameOrInfo; // Assuming programNameOrInfo can be a key or direct pipeline
            if (!pipeline) {
                console.error(`WebGPU pipeline not found for:`, programNameOrInfo);
                return;
            }

            // Uniforms Buffer preparation - this is highly specific to each shader.
            // We need a more generic way or a way to know which uniforms are needed.
            // For now, let's focus on a specific case like advection.

            let uniformBuffer;
            let bindGroupEntries = [
                { binding: 0, resource: this.sampler },
            ];

            if (programNameOrInfo === 'advectionPipeline') {
                // 1. Create/Update Uniform Buffer for Advection
                const advectionUniforms = new Float32Array([
                    1.0 / this.size, 1.0 / this.size, // u_texelSize (vec2<f32>)
                    this.dt,                          // u_dt (f32)
                    0.0                               // Padding for vec2 alignment if u_dt was f32 followed by vec2, or for std140 rules if struct was larger.
                                                      // Current AdvectionUniforms is vec2, f32. WGSL struct layout might need padding for f32 if not careful.
                                                      // For simple vec2, f32, padding might not be strictly needed if they pack well, but being explicit is safer.
                                                      // Let's assume AdvectionUniforms struct { u_texelSize: vec2f, u_dt: f32 } packs tightly for now.
                                                      // If issues, check alignment and add padding to JS ArrayBuffer and WGSL struct.
                ]);
                // Create a new buffer each time for simplicity, or reuse and update for performance.
                uniformBuffer = this.device.createBuffer({
                    label: 'Advection Uniforms Buffer',
                    size: advectionUniforms.byteLength, // Ensure size is multiple of 16 if not careful with offsets
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                this.device.queue.writeBuffer(uniformBuffer, 0, advectionUniforms);

                // 2. Prepare Bind Group Entries for Advection
                // inputTexturesSpec should provide GPUTexture objects for 'u_velocityTexture' and 'u_sourceTexture'
                if (!inputTexturesSpec.u_velocityTexture || !inputTexturesSpec.u_sourceTexture) {
                    console.error('Missing velocity or source texture for advection pass.');
                    return;
                }
                bindGroupEntries.push({ binding: 1, resource: inputTexturesSpec.u_velocityTexture.createView() });
                bindGroupEntries.push({ binding: 2, resource: inputTexturesSpec.u_sourceTexture.createView() });
                bindGroupEntries.push({ binding: 3, resource: { buffer: uniformBuffer } });
            
            } else if (programNameOrInfo === 'displayPipeline') {
                if (!inputTexturesSpec.u_displayTexture) {
                    console.error('Missing display texture for display pass.');
                    return;
                }
                bindGroupEntries.push({ binding: 1, resource: inputTexturesSpec.u_displayTexture.createView() });
                // No specific uniform buffer for the simple display shader, but could have one for scale/offset.
            } else {
                console.warn(`_runShaderPass WebGPU path: Uniform and bind group setup not yet implemented for pipeline:`, programNameOrInfo);
                // For other pipelines, you'd create their specific uniform buffers and bind group entries here.
                // For now, just return to avoid errors.
                return; 
            }

            const bindGroupLayout = pipeline.getBindGroupLayout(0); // Assuming all resources are in group 0
            const bindGroup = this.device.createBindGroup({
                label: `BindGroup for \${typeof programNameOrInfo === 'string' ? programNameOrInfo : 'customPipeline'}`,
                layout: bindGroupLayout,
                entries: bindGroupEntries,
            });

            const commandEncoder = this.device.createCommandEncoder();
            const renderPassDescriptor = {
                colorAttachments: [
                    {
                        view: outputTexture.createView(), // outputTexture is expected to be a GPUTexture
                        loadOp: 'clear', // Or 'load' if you want to preserve previous content (e.g., for blending)
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, // Clear to black/transparent
                        storeOp: 'store',
                    },
                ],
            };
            // Special case for drawing to canvas
            if (outputTexture === this.context.getCurrentTexture()) { // This check needs a reliable way to identify canvas target
                 // If outputTexture is a string like 'canvas', or a direct reference to context.getCurrentTexture()
                 // For now, let's assume if programNameOrInfo is displayPipeline, it draws to canvas.
                if (programNameOrInfo === 'displayPipeline') {
                    renderPassDescriptor.colorAttachments[0].view = this.context.getCurrentTexture().createView();
                    renderPassDescriptor.colorAttachments[0].loadOp = 'clear'; // Clear canvas before drawing
                    renderPassDescriptor.colorAttachments[0].clearValue = { r: 0.0, g: 0.01, b: 0.02, a: 1.0 }; // Slightly off-black
                }
            }


            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.setVertexBuffer(0, this.quadVertexBuffer);
            passEncoder.draw(6); // 6 vertices for a quad
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);

        } else if (this.gl) { // WebGL Path (existing logic)
            const gl = this.gl;
            const programObject = (programNameOrInfo && programNameOrInfo.program) ? programNameOrInfo.program : programNameOrInfo;

            if (!programObject) {
                console.error("Invalid program passed to _runShaderPass (WebGL)", programNameOrInfo);
                return;
            }
            gl.useProgram(programObject);

            const resolutionLocation = gl.getUniformLocation(programObject, "u_resolution");
            if (resolutionLocation) {
                gl.uniform2f(resolutionLocation, this.size, this.size);
            }

            let textureUnit = 0;
            for (const uniformName in inputTexturesSpec) {
                const textureOrValue = inputTexturesSpec[uniformName];
                const location = gl.getUniformLocation(programObject, uniformName);
                if (location) {
                    if (textureOrValue instanceof WebGLTexture) {
                        gl.activeTexture(gl.TEXTURE0 + textureUnit);
                        gl.bindTexture(gl.TEXTURE_2D, textureOrValue);
                        gl.uniform1i(location, textureUnit);
                        textureUnit++;
                    } else if (Array.isArray(textureOrValue)) {
                        if (textureOrValue.length === 2) gl.uniform2fv(location, textureOrValue);
                        else if (textureOrValue.length === 3) gl.uniform3fv(location, textureOrValue);
                        else if (textureOrValue.length === 4) gl.uniform4fv(location, textureOrValue);
                    } else if (typeof textureOrValue === 'number') {
                        gl.uniform1f(location, textureOrValue);
                    } else {
                        console.warn(`Unsupported uniform type for \${uniformName} in _runShaderPass (WebGL)`);
                    }
                }
            }

            gl.bindFramebuffer(gl.FRAMEBUFFER, outputTexture); // outputTexture is an FBO in WebGL path
            if (outputTexture === null) { 
                gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            } else { 
                gl.viewport(0, 0, this.size, this.size);
            }
            
            const positionAttributeLocation = gl.getAttribLocation(programObject, "a_position");
            if (positionAttributeLocation !== -1 && this.quadVertexBuffer) {
                gl.enableVertexAttribArray(positionAttributeLocation);
                gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVertexBuffer);
                gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
                gl.drawArrays(gl.TRIANGLES, 0, 6);
            } else {
                if(positionAttributeLocation === -1) console.error("a_position attribute not found in shader (WebGL)");
                if(!this.quadVertexBuffer) console.error("Quad vertex buffer not initialized (WebGL)");
            }
        }
    }
}