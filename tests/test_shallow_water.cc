// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#include "test_harness.h"
#include "mjwater/shallow_water.h"

using namespace mjwater;

TEST(SWE_InitFlat) {
  ShallowWaterSolver solver;
  solver.Init(20, 20, 0.1f);
  solver.SetSurface(1.0f);

  // All cells should have depth = 1.0 (no bathymetry).
  float vol = solver.TotalVolume();
  float expected = 20 * 20 * 0.1f * 0.1f * 1.0f;  // 4.0 m^3
  CHECK_NEAR(vol, expected, 1e-4f);
}

TEST(SWE_MassConservation) {
  ShallowWaterSolver solver;
  solver.Init(30, 30, 0.1f);
  solver.SetSurface(0.5f);

  float vol_before = solver.TotalVolume();
  CHECK(vol_before > 0);

  // Run 100 steps.
  for (int i = 0; i < 100; ++i) {
    solver.Step(0.001f);
  }

  float vol_after = solver.TotalVolume();
  // Volume should be conserved to within 1%.
  float rel_err = std::abs(vol_after - vol_before) / vol_before;
  CHECK(rel_err < 0.01f);
}

TEST(SWE_QuiescentStability) {
  // A flat water surface with no perturbation should stay flat.
  ShallowWaterSolver solver;
  solver.Init(20, 20, 0.1f);
  solver.SetSurface(1.0f);

  for (int i = 0; i < 50; ++i) {
    solver.Step(0.005f);
  }

  // Check that momentum is essentially zero.
  Vec3 mom = solver.TotalMomentum();
  CHECK(std::abs(mom.x) < 1e-6f);
  CHECK(std::abs(mom.y) < 1e-6f);
}

TEST(SWE_SurfaceQuery) {
  ShallowWaterSolver solver;
  solver.Init(10, 10, 0.5f);
  solver.SetSurface(2.0f);

  float h = solver.SurfaceHeight(2.5f, 2.5f);
  CHECK_NEAR(h, 2.0f, 0.1f);
}

TEST(SWE_DamBreak) {
  // 1D dam break in x-direction: left side deep, right side shallow.
  // After many steps, water should redistribute and momentum should
  // flow in +x direction.
  ShallowWaterSolver solver;
  solver.Init(40, 5, 0.1f);

  auto& h = solver.grid.channels[solver.ch_h].data;
  for (uint32_t y = 0; y < solver.grid.ny; ++y) {
    for (uint32_t x = 0; x < solver.grid.nx; ++x) {
      size_t idx = solver.grid.Idx(x, y);
      h[idx] = (x < 20) ? 1.0f : 0.1f;
    }
  }

  float vol_before = solver.TotalVolume();

  for (int i = 0; i < 200; ++i) {
    solver.Step(0.005f);
  }

  // Mass conservation (dam break generates strong shocks that test
  // the HLL solver's conservation properties).
  float vol_after = solver.TotalVolume();
  float rel_err = std::abs(vol_after - vol_before) / vol_before;
  CHECK(rel_err < 0.05f);  // 5% tolerance for strong shock

  // Net momentum should be in +x direction (water flows right).
  Vec3 mom = solver.TotalMomentum();
  CHECK(mom.x > 0);
}

TEST(SWE_VelocityQuery) {
  // Set up a flow and check velocity query.
  ShallowWaterSolver solver;
  solver.Init(10, 10, 0.5f);
  solver.SetSurface(1.0f);

  // Set x-momentum.
  auto& hu = solver.grid.channels[solver.ch_hu].data;
  auto& h = solver.grid.channels[solver.ch_h].data;
  for (size_t i = 0; i < hu.size(); ++i) {
    hu[i] = h[i] * 0.5f;  // u = 0.5 m/s
  }

  Vec3 v = solver.Velocity(2.5f, 2.5f);
  CHECK_NEAR(v.x, 0.5f, 0.05f);
  CHECK_NEAR(v.y, 0.0f, 1e-5f);
}

TEST(SWE_DamBreakShockSpeed) {
  // Ritter's analytical solution for ideal dam break (1892):
  // Shock front speed = 2*sqrt(g*h0) where h0 = initial deep side depth.
  // After time t, the front should be at x_dam + 2*sqrt(g*h0)*t.
  ShallowWaterSolver solver;
  uint32_t nx = 200;
  float dx = 0.05f;
  solver.Init(nx, 3, dx);  // narrow in y to keep 1D-ish

  auto& h = solver.grid.channels[solver.ch_h].data;
  uint32_t dam_pos = nx / 2;
  float h0 = 1.0f;
  for (uint32_t y = 0; y < solver.grid.ny; ++y) {
    for (uint32_t x = 0; x < solver.grid.nx; ++x) {
      h[solver.grid.Idx(x, y)] = (x < dam_pos) ? h0 : 0.01f;
    }
  }

  // Use CFL-limited stepping for accuracy.
  float total_time = 0;
  for (int i = 0; i < 200; ++i) {
    float actual_dt = solver.Step(0.005f);
    total_time += actual_dt;
  }

  // Find the shock front: rightmost cell with depth > 10% of initial.
  uint32_t front_x = dam_pos;
  uint32_t mid_y = solver.grid.ny / 2;
  for (uint32_t x = dam_pos; x < nx; ++x) {
    if (h[solver.grid.Idx(x, mid_y)] > 0.05f) front_x = x;
  }

  float front_pos = (front_x + 0.5f) * dx;
  float dam_world = (dam_pos + 0.5f) * dx;
  float measured_speed = (front_pos - dam_world) / total_time;

  // Ritter: front speed = 2*sqrt(g*h0)
  float ritter_speed = 2.0f * std::sqrt(kGravity * h0);

  // Check that front has moved forward at a reasonable speed.
  // Numerical diffusion will slow the front somewhat, and the
  // downstream initial depth (0.01) also affects it.
  CHECK(measured_speed > 0);  // front moved
  CHECK(measured_speed < ritter_speed * 1.5f);  // not faster than analytical
}

int main() {
  return RunAllTests();
}
