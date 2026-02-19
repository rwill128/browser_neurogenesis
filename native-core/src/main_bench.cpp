#include "native_core.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  using namespace native_core;

  constexpr std::uint32_t rigid_count = 512;
  constexpr std::uint32_t soft_count = 2048;
  constexpr std::uint32_t steps = 256;

  std::vector<float> rx(rigid_count, 10.0f), ry(rigid_count, 10.0f), rvx(rigid_count, 0.1f), rvy(rigid_count, -0.1f);
  std::vector<float> rtheta(rigid_count, 0.0f), romega(rigid_count, 0.01f), rmass(rigid_count, 1.0f), rinertia(rigid_count, 1.0f), rradius(rigid_count, 1.0f);

  std::vector<float> sx(soft_count, 20.0f), sy(soft_count, 20.0f), svx(soft_count, 0.05f), svy(soft_count, -0.02f), smass(soft_count, 1.0f);
  std::vector<std::int32_t> scluster(soft_count, 0);

  RigidSoA rigid{rigid_count, rx.data(), ry.data(), rvx.data(), rvy.data(), rtheta.data(), romega.data(), rmass.data(), rinertia.data(), rradius.data()};
  SoftNodeSoA soft{soft_count, sx.data(), sy.data(), svx.data(), svy.data(), smass.data(), scluster.data()};

  StepConfig cfg{};
  StepStats stats{};

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (std::uint32_t i = 0; i < steps; ++i) {
    step_collision(rigid, soft, cfg, stats);
    step_body_fluid_coupling(rigid, soft, cfg, stats);
    step_integrate(rigid, soft, cfg, stats);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  std::cout << "{\n"
            << "  \"steps\": " << steps << ",\n"
            << "  \"elapsed_us\": " << us << ",\n"
            << "  \"us_per_step\": " << (double(us) / double(steps)) << ",\n"
            << "  \"collision_pairs_tested\": " << stats.collision_pairs_tested << ",\n"
            << "  \"soft_nodes_touched\": " << stats.soft_nodes_touched << "\n"
            << "}\n";

  return 0;
}
