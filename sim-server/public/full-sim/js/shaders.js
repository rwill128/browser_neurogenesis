export const WEBGL_fluid_simulation_vertex_shader = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_position * 0.5 + 0.5;
    }
`;

export const WEBGL_advect_fragment_shader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_velocityTexture;
    uniform sampler2D u_sourceTexture;
    uniform vec2 u_texelSize;
    uniform float u_dt;
    uniform float u_dissipation;

    void main() {
        vec2 velocity = texture2D(u_velocityTexture, v_texCoord).xy;
        vec2 prevTexCoord = v_texCoord - velocity * u_dt * u_texelSize;
        vec4 advectedValue = texture2D(u_sourceTexture, prevTexCoord);
        
        // Apply dissipation
        advectedValue *= u_dissipation;

        gl_FragColor = advectedValue;
    }
`;

export const WEBGL_divergence_fragment_shader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_velocityTexture;
    uniform vec2 u_texelSize;
    
    void main() {
        float velW = texture2D(u_velocityTexture, v_texCoord - vec2(u_texelSize.x, 0.0)).x;
        float velE = texture2D(u_velocityTexture, v_texCoord + vec2(u_texelSize.x, 0.0)).x;
        float velS = texture2D(u_velocityTexture, v_texCoord - vec2(0.0, u_texelSize.y)).y;
        float velN = texture2D(u_velocityTexture, v_texCoord + vec2(0.0, u_texelSize.y)).y;

        float divergence = 0.5 * (velE - velW + velN - velS);
        gl_FragColor = vec4(divergence, 0.0, 0.0, 1.0);
    }
`;

export const WEBGL_jacobi_fragment_shader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_xTexture; // pressure texture in this case
    uniform sampler2D u_bTexture; // divergence texture
    uniform float u_alpha;
    uniform float u_rBeta;

    void main() {
        vec4 xW = texture2D(u_xTexture, v_texCoord - vec2(u_texelSize.x, 0.0));
        vec4 xE = texture2D(u_xTexture, v_texCoord + vec2(u_texelSize.x, 0.0));
        vec4 xN = texture2D(u_xTexture, v_texCoord + vec2(0.0, u_texelSize.y));
        vec4 xS = texture2D(u_xTexture, v_texCoord - vec2(0.0, u_texelSize.y));
        vec4 bC = texture2D(u_bTexture, v_texCoord);
        
        gl_FragColor = (xW + xE + xN + xS + u_alpha * bC) * u_rBeta;
    }
`;

export const WEBGL_gradient_subtraction_fragment_shader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_pressureTexture;
    uniform sampler2D u_velocityTexture;
    uniform vec2 u_texelSize;
    
    void main() {
        float pW = texture2D(u_pressureTexture, v_texCoord - vec2(u_texelSize.x, 0.0)).x;
        float pE = texture2D(u_pressureTexture, v_texCoord + vec2(u_texelSize.x, 0.0)).x;
        float pS = texture2D(u_pressureTexture, v_texCoord - vec2(0.0, u_texelSize.y)).x;
        float pN = texture2D(u_pressureTexture, v_texCoord + vec2(0.0, u_texelSize.y)).x;
        
        vec2 currentVelocity = texture2D(u_velocityTexture, v_texCoord).xy;
        vec2 gradP = 0.5 * vec2(pE - pW, pN - pS);
        
        gl_FragColor = vec4(currentVelocity - gradP, 0.0, 1.0);
    }
`;

export const WEBGL_splat_fragment_shader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_targetTexture;
    uniform vec2 u_point;
    uniform vec3 u_color;
    uniform float u_radius;

    void main() {
        vec2 coord_diff = v_texCoord - u_point;
        float dist_sq = dot(coord_diff, coord_diff);
        float radius_sq = u_radius * u_radius;

        float intensity = exp(-dist_sq / radius_sq);
        vec4 color_to_add = vec4(u_color * intensity, 1.0);

        gl_FragColor = texture2D(u_targetTexture, v_texCoord) + color_to_add;
    }
`;

export const WEBGL_display_fragment_shader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_displayTexture;

    void main() {
        gl_FragColor = texture2D(u_displayTexture, v_texCoord);
    }
`;

export const WEBGL_viscosity_jacobi_fragment_shader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_velocityTexture;
    uniform sampler2D u_viscosityTexture;
    uniform float u_alpha;
    uniform float u_rBeta;
    uniform vec2 u_texelSize;

    void main() {
        vec4 velW = texture2D(u_velocityTexture, v_texCoord - vec2(u_texelSize.x, 0.0));
        vec4 velE = texture2D(u_velocityTexture, v_texCoord + vec2(u_texelSize.x, 0.0));
        vec4 velN = texture2D(u_velocityTexture, v_texCoord + vec2(0.0, u_texelSize.y));
        vec4 velS = texture2D(u_velocityTexture, v_texCoord - vec2(0.0, u_texelSize.y));
        vec4 velC = texture2D(u_velocityTexture, v_texCoord);
        
        float viscosity_multiplier = texture2D(u_viscosityTexture, v_texCoord).x;
        float effective_alpha = u_alpha * viscosity_multiplier;
        float effective_rBeta = 1.0 / (1.0 + 4.0 * effective_alpha);

        gl_FragColor = (velW + velE + velN + velS + effective_alpha * velC) * effective_rBeta;
    }
`; 