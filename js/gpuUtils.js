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
    if (!navigator.gpu) {
        alert("WebGPU is not supported on your browser.");
        console.error("WebGPU not supported.");
        return null;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        alert("WebGPU adapter is not available.");
        console.error("WebGPU adapter not available.");
        return null;
    }

    const device = await adapter.requestDevice();
    if (!device) {
        alert("WebGPU device is not available."); // Should not happen if adapter is available, but good practice
        console.error("WebGPU device not available.");
        return null;
    }

    const context = canvas.getContext('webgpu');
    if (!context) {
        alert("Failed to get WebGPU context from canvas.");
        console.error("Failed to get WebGPU context.");
        return null;
    }

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: presentationFormat,
        alphaMode: 'opaque', // or 'premultiplied'
    });

    console.log("WebGPU context initialized successfully.");
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