const NodeType = {
    PREDATOR: 0,
    EATER: 1,
    PHOTOSYNTHETIC: 2,
    NEURON: 3,
    EMITTER: 4, // For dye
    SWIMMER: 5, // For propulsion
    EYE: 6,      // New: For particle detection
    JET: 7,       // New: For fluid propulsion
    ATTRACTOR: 8,
    REPULSOR: 9
};

const RLRewardStrategy = {
    ENERGY_CHANGE: 0,
    REPRODUCTION_EVENT: 1,
    PARTICLE_PROXIMITY: 2,
    ENERGY_SECOND_DERIVATIVE: 3, // New: Reward based on the change in energy change rate
    CREATURE_PROXIMITY: 4,
    CREATURE_DISTANCE: 5,
    // New rewards based on NN inputs
    SENSED_DYE_R: 6,
    SENSED_DYE_R_INV: 7,
    SENSED_DYE_G: 8,
    SENSED_DYE_G_INV: 9,
    SENSED_DYE_B: 10,
    SENSED_DYE_B_INV: 11,
    ENERGY_RATIO: 12,
    ENERGY_RATIO_INV: 13,
    REL_COM_POS_X_POS: 14,
    REL_COM_POS_X_NEG: 15,
    REL_COM_POS_Y_POS: 16,
    REL_COM_POS_Y_NEG: 17,
    REL_COM_VEL_X_POS: 18,
    REL_COM_VEL_X_NEG: 19,
    REL_COM_VEL_Y_POS: 20,
    REL_COM_VEL_Y_NEG: 21,
    SENSED_NUTRIENT: 22,
    SENSED_NUTRIENT_INV: 23,
    AVG_SPRING_COMPRESSION: 24,
    AVG_SPRING_EXTENSION: 25,
    AVG_FLUID_VEL_X_POS: 26,
    AVG_FLUID_VEL_X_NEG: 27,
    AVG_FLUID_VEL_Y_POS: 28,
    AVG_FLUID_VEL_Y_NEG: 29,
    EYE_SEES_TARGET: 30,
    EYE_TARGET_PROXIMITY: 31,
    EYE_TARGET_DISTANCE: 32,
};

const RLAlgorithmType = {
    REINFORCE: 0, // Your current policy gradient
    SAC: 1        // Soft Actor-Critic
};

const EyeTargetType = {
    PARTICLE: 0,
    FOREIGN_BODY_POINT: 1
};

const MovementType = {
    FIXED: 0,    // Fixed in place, does not interact with fluid velocity but can affect it (if Swimmer)
    FLOATING: 1, // Pushed by fluid, cannot be a Swimmer
    NEUTRAL: 2   // Standard soft body physics, only interacts with fluid if Swimmer (by pushing it)
};

const ActivationPatternType = {
    FLAT: 0,    // Constant exertion level
    SINE: 1,    // Sinusoidal exertion (0 to Max)
    PULSE: 2   // Periodic sharp pulse (0 or Max)
};

const DyeChannel = {
    RED: 0,
    GREEN: 1,
    BLUE: 2,
    AVERAGE: 3
};

function getNodeTypeString(nodeType) {
    switch(nodeType) {
        case NodeType.PREDATOR: return "Predator";
        case NodeType.EATER: return "Eater";
        case NodeType.PHOTOSYNTHETIC: return "Photosynthetic";
        case NodeType.NEURON: return "Neuron";
        case NodeType.EMITTER: return "Emitter (Dye)";
        case NodeType.SWIMMER: return "Swimmer (Propulsion)";
        case NodeType.EYE: return "Eye (Particle Detector)";
        case NodeType.JET: return "Jet (Fluid Propulsion)";
        case NodeType.ATTRACTOR: return "Attractor";
        case NodeType.REPULSOR: return "Repulsor";
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

function getRewardStrategyString(strategy) {
    switch(strategy) {
        case RLRewardStrategy.ENERGY_CHANGE: return "Energy Change";
        case RLRewardStrategy.REPRODUCTION_EVENT: return "Reproduction Event";
        case RLRewardStrategy.PARTICLE_PROXIMITY: return "Particle Proximity";
        case RLRewardStrategy.ENERGY_SECOND_DERIVATIVE: return "Energy Change Rate Change";
        case RLRewardStrategy.CREATURE_PROXIMITY: return "Creature Proximity";
        case RLRewardStrategy.CREATURE_DISTANCE: return "Creature Distance";
        case RLRewardStrategy.SENSED_DYE_R: return "Sensed Dye (Red)";
        case RLRewardStrategy.SENSED_DYE_R_INV: return "Avoid Dye (Red)";
        case RLRewardStrategy.SENSED_DYE_G: return "Sensed Dye (Green)";
        case RLRewardStrategy.SENSED_DYE_G_INV: return "Avoid Dye (Green)";
        case RLRewardStrategy.SENSED_DYE_B: return "Sensed Dye (Blue)";
        case RLRewardStrategy.SENSED_DYE_B_INV: return "Avoid Dye (Blue)";
        case RLRewardStrategy.ENERGY_RATIO: return "High Energy Ratio";
        case RLRewardStrategy.ENERGY_RATIO_INV: return "Low Energy Ratio";
        case RLRewardStrategy.REL_COM_POS_X_POS: return "CoM Right of Brain";
        case RLRewardStrategy.REL_COM_POS_X_NEG: return "CoM Left of Brain";
        case RLRewardStrategy.REL_COM_POS_Y_POS: return "CoM Below Brain";
        case RLRewardStrategy.REL_COM_POS_Y_NEG: return "CoM Above Brain";
        case RLRewardStrategy.REL_COM_VEL_X_POS: return "CoM Velocity Right";
        case RLRewardStrategy.REL_COM_VEL_X_NEG: return "CoM Velocity Left";
        case RLRewardStrategy.REL_COM_VEL_Y_POS: return "CoM Velocity Down";
        case RLRewardStrategy.REL_COM_VEL_Y_NEG: return "CoM Velocity Up";
        case RLRewardStrategy.SENSED_NUTRIENT: return "High Nutrients";
        case RLRewardStrategy.SENSED_NUTRIENT_INV: return "Low Nutrients";
        case RLRewardStrategy.AVG_SPRING_COMPRESSION: return "Springs Compressed";
        case RLRewardStrategy.AVG_SPRING_EXTENSION: return "Springs Extended";
        case RLRewardStrategy.AVG_FLUID_VEL_X_POS: return "In Rightward Fluid";
        case RLRewardStrategy.AVG_FLUID_VEL_X_NEG: return "In Leftward Fluid";
        case RLRewardStrategy.AVG_FLUID_VEL_Y_POS: return "In Downward Fluid";
        case RLRewardStrategy.AVG_FLUID_VEL_Y_NEG: return "In Upward Fluid";
        case RLRewardStrategy.EYE_SEES_TARGET: return "Eye Sees Target";
        case RLRewardStrategy.EYE_TARGET_PROXIMITY: return "Eye Near Target";
        case RLRewardStrategy.EYE_TARGET_DISTANCE: return "Eye Far From Target";
        default: return "Unknown_Strategy";
    }
}

function getEyeTargetTypeString(eyeTargetType) {
    switch(eyeTargetType) {
        case EyeTargetType.PARTICLE: return "Particle";
        case EyeTargetType.FOREIGN_BODY_POINT: return "Foreign Body Point";
        default: return "Unknown_EyeTargetType";
    }
} 