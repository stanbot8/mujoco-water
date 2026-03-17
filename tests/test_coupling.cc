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

int main() {
  return RunAllTests();
}
