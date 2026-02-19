#pragma once

#include <cstdint>

namespace native_core {

struct RigidSoA {
  std::uint32_t count = 0;
  float* x = nullptr;
  float* y = nullptr;
  float* vx = nullptr;
  float* vy = nullptr;
  float* theta = nullptr;
  float* omega = nullptr;
  float* mass = nullptr;
  float* inertia = nullptr;
  float* radius = nullptr;
};

struct SoftNodeSoA {
  std::uint32_t count = 0;
  float* x = nullptr;
  float* y = nullptr;
  float* vx = nullptr;
  float* vy = nullptr;
  float* mass = nullptr;
  std::int32_t* cluster_id = nullptr;
};

struct StepConfig {
  float dt = 0.01f;
  float rigid_slop = 0.32f;
  float rigid_soft_node_slop = 0.18f;
  float rigid_soft_edge_slop = 0.16f;
};

struct StepStats {
  std::uint64_t collision_pairs_tested = 0;
  std::uint64_t collision_pairs_resolved = 0;
  std::uint64_t soft_nodes_touched = 0;
};

bool step_collision(RigidSoA& rigid, SoftNodeSoA& soft, const StepConfig& cfg, StepStats& stats);

bool step_body_fluid_coupling(RigidSoA& rigid, SoftNodeSoA& soft, const StepConfig& cfg, StepStats& stats);

bool step_integrate(RigidSoA& rigid, SoftNodeSoA& soft, const StepConfig& cfg, StepStats& stats);

} // namespace native_core
