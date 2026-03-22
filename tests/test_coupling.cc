// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#include "test_harness.h"
#include "mjwater/coupling.h"
#include "mjwater/shallow_water.h"

using namespace mjwater;

TEST(Coupling_Buoyancy) {
  // A submerged body should feel upward buoyancy force.
  FluidCoupling::FluidState fluid;
  fluid.velocity = {};
  fluid.density = kWaterDensity;
  fluid.depth = 1.0f;
  fluid.submerged = true;

  BodyCoupling body;
  body.volume = 0.001f;      // 1 liter
  body.cross_section = 0.01f;
  body.drag_coeff = 1.0f;

  Vec3 body_vel = {};
  auto force = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);

  // Buoyancy should point up (positive z).
  CHECK(force.force.z > 0);
  // Expected: rho * g * V = 998.2 * 9.81 * 0.001 = ~9.79 N
  CHECK_NEAR(force.force.z, kWaterDensity * kGravity * body.volume, 0.1f);
}

TEST(Coupling_Drag) {
  FluidCoupling::FluidState fluid;
  fluid.velocity = {};
  fluid.density = kWaterDensity;
  fluid.depth = 1.0f;
  fluid.submerged = true;

  BodyCoupling body;
  body.volume = 0.001f;
  body.cross_section = 0.01f;
  body.drag_coeff = 1.0f;

  // Body moving in +x direction -> drag should oppose (negative x).
  Vec3 body_vel = {1.0f, 0, 0};
  auto force = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);
  CHECK(force.force.x < 0);  // drag opposes motion
}

TEST(Coupling_NoForceWhenDry) {
  FluidCoupling::FluidState fluid;
  fluid.submerged = false;

  BodyCoupling body;
  body.volume = 0.001f;

  Vec3 body_vel = {1.0f, 0, 0};
  auto force = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);

  CHECK_NEAR(force.force.x, 0.0f, 1e-10f);
  CHECK_NEAR(force.force.z, 0.0f, 1e-10f);
}

TEST(Coupling_SWEQuery) {
  ShallowWaterSolver swe;
  swe.Init(10, 10, 0.5f);
  swe.SetSurface(2.0f);

  auto state = FluidCoupling::QuerySWE(swe, {2.5f, 2.5f, 1.0f});
  CHECK(state.submerged);
  CHECK(state.depth > 0);
}

TEST(Coupling_TwoWayWaveGeneration) {
  // A moving body should inject wave sources into SWE cells.
  ShallowWaterSolver swe;
  swe.Init(20, 20, 0.5f);
  swe.SetSurface(2.0f);

  float vol_before = swe.TotalVolume();

  // Body at grid center, moving downward (displacing water).
  BodyCoupling body;
  body.volume = 1.0f;
  body.cross_section = 0.5f;
  body.two_way_enabled = true;
  body.body_radius = 0.5f;

  Vec3 body_pos = {5.0f, 5.0f, 1.5f};
  Vec3 body_vel = {0, 0, -1.0f};  // moving down into water

  auto sources = FluidCoupling::ComputeWaveSources(swe, body, body_pos, body_vel);

  // Should have produced source terms.
  CHECK(sources.count > 0);

  // Apply sources and check volume changed.
  swe.ApplySourceTerms(sources.cell_x.data(), sources.cell_y.data(),
                        sources.dh.data(), sources.dhu.data(), sources.dhv.data(),
                        sources.count, 0.01f);
  float vol_after = swe.TotalVolume();

  // Body moving down displaces water, volume should increase.
  CHECK(vol_after > vol_before);
}

TEST(Coupling_ReynoldsDragCoeff) {
  // Verify drag coefficient correlations return reasonable values.
  // Stokes regime: Cd = 24/Re
  float cd_stokes = FluidCoupling::SphereDragCoeff(0.5f);
  CHECK_NEAR(cd_stokes, 48.0f, 1.0f);  // 24/0.5 = 48

  // Newton regime: Cd ~ 0.44
  float cd_newton = FluidCoupling::SphereDragCoeff(5000.0f);
  CHECK_NEAR(cd_newton, 0.44f, 0.01f);

  // Post-critical: Cd ~ 0.1
  float cd_post = FluidCoupling::SphereDragCoeff(1e6f);
  CHECK_NEAR(cd_post, 0.1f, 0.01f);
}

int main() {
  return RunAllTests();
}
