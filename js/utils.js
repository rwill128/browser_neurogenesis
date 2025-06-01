// --- Neural Network Math Helpers ---
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Box-Muller transform to get a normally distributed random number (mean 0, stdDev 1)
// Returns one sample. Call twice if you need two independent samples (or cache the second one).
let _z1_gauss = null; // Cache for the second sample from Box-Muller
function sampleGaussianBoxMuller() {
    if (_z1_gauss !== null) {
        const z1 = _z1_gauss;
        _z1_gauss = null;
        return z1;
    }
    let u1 = 0, u2 = 0;
    while (u1 === 0) u1 = Math.random(); //Working with [0,1)
    while (u2 === 0) u2 = Math.random();
    const R = Math.sqrt(-2.0 * Math.log(u1));
    const Theta = 2.0 * Math.PI * u2;
    const z0 = R * Math.cos(Theta);
    _z1_gauss = R * Math.sin(Theta); // Cache the second sample
    return z0;
}

function sampleGaussian(mean, stdDev) {
    const z0 = sampleGaussianBoxMuller();
    return z0 * stdDev + mean;
}

function logPdfGaussian(x, mean, stdDev) {
    if (stdDev <= 1e-6) { // Avoid division by zero or log of zero if stdDev is tiny
        // If stdDev is effectively zero, probability is extremely high if x is mean, else very low.
        // This is a simplification; proper handling might involve a dirac delta or just clamping stdDev.
        return (Math.abs(x - mean) < 1e-6) ? 0 : -Infinity; 
    }
    const M_PI = Math.PI;
    const variance = stdDev * stdDev;
    const logSqrt2PiVariance = 0.5 * Math.log(2 * M_PI * variance);
    const exponent = -((x - mean) * (x - mean)) / (2 * variance);
    return exponent - logSqrt2PiVariance;
}

function initializeMatrix(rows, cols, randomRange = 1) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
        matrix[i] = [];
        for (let j = 0; j < cols; j++) {
            matrix[i][j] = (Math.random() * 2 - 1) * randomRange;
        }
    }
    return matrix;
}

function initializeVector(size, randomRange = 1) {
    const vector = [];
    for (let i = 0; i < size; i++) {
        vector[i] = (Math.random() * 2 - 1) * randomRange;
    }
    return vector;
}

function multiplyMatrixVector(matrix, vector) {
    const result = [];
    if (!matrix || matrix.length === 0 || !vector || vector.length === 0 || matrix[0].length !== vector.length) {
        console.error("Matrix-vector multiplication dimension mismatch:", matrix, vector);
        // Return a zero vector of expected output size if possible, or an empty array
        const expectedOutputSize = matrix && matrix.length > 0 ? matrix.length : 0;
        for(let i = 0; i < expectedOutputSize; i++) result.push(0);
        return result;
    }
    for (let i = 0; i < matrix.length; i++) {
        let sum = 0;
        for (let j = 0; j < vector.length; j++) {
            sum += matrix[i][j] * vector[j];
        }
        result.push(sum);
    }
    return result;
}

function addVectors(vectorA, vectorB) {
    if (vectorA.length !== vectorB.length) {
        console.error("Vector addition dimension mismatch:", vectorA, vectorB);
        // Return a zero vector of the first vector's length, or an empty array
        const result = [];
        for(let i = 0; i < vectorA.length; i++) result.push(0);
        return result;
    }
    const result = [];
    for (let i = 0; i < vectorA.length; i++) {
        result.push(vectorA[i] + vectorB[i]);
    }
    return result;
}

// --- Vector2D Class ---
class Vec2 {
    constructor(x = 0, y = 0) { this.x = x; this.y = y; }
    add(other) { return new Vec2(this.x + other.x, this.y + other.y); }
    sub(other) { return new Vec2(this.x - other.x, this.y - other.y); }
    mul(scalar) { return new Vec2(this.x * scalar, this.y * scalar); }
    div(scalar) { return scalar !== 0 ? new Vec2(this.x / scalar, this.y / scalar) : new Vec2(); }
    mag() { return Math.sqrt(this.x * this.x + this.y * this.y); }
    magSq() { return this.x * this.x + this.y * this.y; }
    normalize() { const m = this.mag(); return m > 0 ? this.div(m) : new Vec2(); }
    static dot(v1, v2) { return v1.x * v2.x + v1.y * v2.y; }
    clone() { return new Vec2(this.x, this.y); }
}

// Simple Perlin Noise placeholder
const PerlinNoise = function() {
    this.p = new Uint8Array(512);
    for (let i=0; i < 256 ; i++) this.p[i] = Math.floor(Math.random()*256);
    for (let i=0; i < 256 ; i++) this.p[256+i] = this.p[i];

    this.fade = function(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    this.lerp = function(t, a, b) { return a + t * (b - a); }
    this.grad = function(hash, x, y_param) { // Renamed y to y_param
        const h = hash & 15;
        const u = h < 8 ? x : y_param;
        const v = h < 4 ? y_param : h === 12 || h === 14 ? x : 0;
        return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
    }
    this.noise = function(x, y_param) { // Renamed y to y_param
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y_param) & 255;
        x -= Math.floor(x);
        y_param -= Math.floor(y_param);
        const u = this.fade(x);
        const v = this.fade(y_param);
        const A = this.p[X  ] + Y;
        const B = this.p[X+1] + Y;
        return this.lerp(v, this.lerp(u, this.grad(this.p[A  ], x, y_param  ),
                                         this.grad(this.p[B  ], x-1,y_param  )),
                           this.lerp(u, this.grad(this.p[A+1], x, y_param-1),
                                         this.grad(this.p[B+1], x-1,y_param-1)));
    }
};
const perlin = new PerlinNoise(); // Instantiate it


function getNodeTypeString(nodeType) {
    switch(nodeType) {
        case NodeType.PREDATOR: return "Predator";
        case NodeType.EATER: return "Eater";
        case NodeType.PHOTOSYNTHETIC: return "Photosynthetic";
        case NodeType.NEURON: return "Neuron";
        case NodeType.EMITTER: return "Emitter (Dye)";
        case NodeType.SWIMMER: return "Swimmer (Propulsion)";
        case NodeType.EYE: return "Eye (Particle Detector)";
        default: return "Unknown_NodeType";
    }
}

function getMovementTypeString(movementType) {
    switch(movementType) {
        case MovementType.FIXED: return "Fixed";
        case MovementType.FLOATING: return "Floating";
        case MovementType.NEUTRAL: return "Neutral";
        default: return "Unknown_MovementType";
    }
}

function getSensedChannelString(channelId) {
    switch(channelId) {
        case DyeChannel.RED: return "Red";
        case DyeChannel.GREEN: return "Green";
        case DyeChannel.BLUE: return "Blue";
        case DyeChannel.AVERAGE: return "Average Intensity";
        default: return "Unknown";
    }
}

// Reflects point P across the line defined by L1 and L2
function reflectPointAcrossLine(P, L1, L2) {
    // Line L1L2 direction vector
    const dX = L2.x - L1.x;
    const dY = L2.y - L1.y;

    if (dX === 0 && dY === 0) { // L1 and L2 are the same point, reflection is just P itself or undefined based on interpretation
        return P.clone(); 
    }

    // t = [(P.x - L1.x) * dX + (P.y - L1.y) * dY] / (dX*dX + dY*dY)
    // This t is the parameter for the projection of P onto the line L1L2, 
    // where projected point M = L1 + t * (L2 - L1)
    const t = ((P.x - L1.x) * dX + (P.y - L1.y) * dY) / (dX * dX + dY * dY);

    // M: Projection of P onto the line L1L2
    const Mx = L1.x + t * dX;
    const My = L1.y + t * dY;

    // Reflected point P_reflected = P + 2 * (M - P)
    // P_reflected.x = P.x + 2 * (Mx - P.x) = 2*Mx - P.x
    // P_reflected.y = P.y + 2 * (My - P.y) = 2*My - P.y
    const reflectedX = 2 * Mx - P.x;
    const reflectedY = 2 * My - P.y;

    return new Vec2(reflectedX, reflectedY);
} 