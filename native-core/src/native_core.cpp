#include "native_core.h"

namespace native_core {

bool step_collision(RigidSoA& rigid, SoftNodeSoA& soft, const StepConfig& /*cfg*/, StepStats& stats) {
  // TODO(rick+assistant): first real vertical slice target.
  // Keep this function deterministic and in-place on SoA buffers.
  stats.collision_pairs_tested += static_cast<std::uint64_t>(rigid.count) * static_cast<std::uint64_t>(rigid.count > 0 ? rigid.count - 1 : 0) / 2;
  (void)soft;
  return true;
}

bool step_body_fluid_coupling(RigidSoA& rigid, SoftNodeSoA& soft, const StepConfig& /*cfg*/, StepStats& stats) {
  // TODO: migrate gather/reduction/apply from JS hot path.
  stats.soft_nodes_touched += soft.count;
  (void)rigid;
  return true;
}

bool step_integrate(RigidSoA& rigid, SoftNodeSoA& soft, const StepConfig& cfg, StepStats& /*stats*/) {
  // Temporary deterministic no-op integrator skeleton (for wiring only).
  for (std::uint32_t i = 0; i < rigid.count; ++i) {
    rigid.x[i] += rigid.vx[i] * cfg.dt;
    rigid.y[i] += rigid.vy[i] * cfg.dt;
    rigid.theta[i] += rigid.omega[i] * cfg.dt;
  }
  for (std::uint32_t i = 0; i < soft.count; ++i) {
    soft.x[i] += soft.vx[i] * cfg.dt;
    soft.y[i] += soft.vy[i] * cfg.dt;
  }
  return true;
}

} // namespace native_core
