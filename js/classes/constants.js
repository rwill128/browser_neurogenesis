export const NodeType = {
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
    // Old types like NEUTRAL, FLOATING, FIXED_ROOT, EMITTER_SWIMMER are removed
};

export const RLRewardStrategy = {
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

export const RLAlgorithmType = {
    REINFORCE: 0, // Your current policy gradient
    SAC: 1        // Soft Actor-Critic
};

export const EyeTargetType = {
    PARTICLE: 0,
    FOREIGN_BODY_POINT: 1
};

export const MovementType = {
    FIXED: 0,    // Fixed in place, does not interact with fluid velocity but can affect it (if Swimmer)
    FLOATING: 1, // Pushed by fluid, cannot be a Swimmer
    NEUTRAL: 2   // Standard soft body physics, only interacts with fluid if Swimmer (by pushing it)
}; 