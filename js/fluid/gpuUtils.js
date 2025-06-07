function initWebGL(canvas) {
    let gl = null;
    try {
        // Try to grab the standard context. If it fails, fallback to experimental.
        gl = canvas.getContext("webgl2");
        if (!gl) {
            console.log("WebGL 2 not available, trying WebGL 1.");
            gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
        }
    } catch (e) {
        // Ignore errors
    }

    if (!gl) {
        alert("Unable to initialize WebGL. Your browser may not support it.");
        console.error("WebGL initialization failed.");
        gl = null;
    } else {
        console.log("WebGL context initialized successfully.");
        // Useful extensions we might need later, especially for floating point textures
        // For WebGL1, these are crucial. For WebGL2, some (like float textures) are core.
        if (gl.getExtension('OES_texture_float')) {
            console.log("OES_texture_float extension enabled.");
        }
        if (gl.getExtension('OES_texture_float_linear')) {
            console.log("OES_texture_float_linear extension enabled.");
        }
        if (gl.getExtension('EXT_color_buffer_float')) { // For rendering to float textures
            console.log("EXT_color_buffer_float extension enabled.");
        }
    }
    return gl;
}

async function initWebGPU(canvas) {
    console.log("[gpuUtils.js] Attempting initWebGPU...");
    if (!navigator.gpu) {
        alert("WebGPU is not supported on your browser.");
        console.error("WebGPU not supported.");
        return null;
    }
    console.log("[gpuUtils.js] navigator.gpu exists.");

    let adapter = null;
    try {
        adapter = await navigator.gpu.requestAdapter();
    } catch (e) {
        console.error("[gpuUtils.js] Error requesting adapter:", e);
        alert("Error requesting WebGPU adapter. See console.");
        return null;
    }

    if (!adapter) {
        alert("WebGPU adapter is not available (returned null).");
        console.error("[gpuUtils.js] WebGPU adapter not available (requestAdapter returned null).");
        return null;
    }
    console.log("[gpuUtils.js] WebGPU Adapter obtained:", adapter);

    let device = null;
    try {
        device = await adapter.requestDevice();
    } catch (e) {
        console.error("[gpuUtils.js] Error requesting device:", e);
        alert("Error requesting WebGPU device. See console.");
        return null;
    }

    if (!device) {
        alert("WebGPU device is not available (returned null).");
        console.error("[gpuUtils.js] WebGPU device not available (requestDevice returned null).");
        return null;
    }
    console.log("[gpuUtils.js] WebGPU Device obtained:", device);

    const context = canvas.getContext('webgpu');
    if (!context) {
        alert("Failed to get WebGPU context from canvas.");
        console.error("[gpuUtils.js] Failed to get WebGPU context from canvas.");
        return null;
    }
    console.log("[gpuUtils.js] WebGPU Context obtained.");

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    console.log("[gpuUtils.js] Preferred canvas format:", presentationFormat);
    try {
        context.configure({
            device,
            format: presentationFormat,
            alphaMode: 'opaque',
        });
    } catch (e) {
        console.error("[gpuUtils.js] Error configuring WebGPU context:", e);
        alert("Error configuring WebGPU context. See console.");
        return null;
    }
    console.log("[gpuUtils.js] WebGPU Context configured successfully.");

    return { device, context, presentationFormat, adapter };
}

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(`An error occurred compiling the shaders: ${gl.getShaderInfoLog(shader)}`);
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error(`Unable to initialize the shader program: ${gl.getProgramInfoLog(program)}`);
        gl.deleteProgram(program); // Clean up program
        gl.deleteShader(vertexShader); // Clean up shaders
        gl.deleteShader(fragmentShader);
        return null;
    }
    return program;
}