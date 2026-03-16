// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#include "test_harness.h"
#include "mjwater/sph.h"

using namespace mjwater;

TEST(SPH_ParticleCreation) {
  SPHSolver solver;
  SPHParams params;
  params.smoothing_length = 0.02f;
  AABB domain = {{0, 0, 0}, {1, 1, 1}};
  solver.Init(params, domain);

  AABB box = {{0.1f, 0.1f, 0.1f}, {0.3f, 0.3f, 0.3f}};
  solver.AddParticlesInBox(box, 0.02f);

  CHECK(solver.Count() > 0);
}

TEST(SPH_MassConservation) {
  SPHSolver solver;
  SPHParams params;
  params.smoothing_length = 0.03f;
  AABB domain = {{0, 0, 0}, {0.5f, 0.5f, 0.5f}};
  solver.Init(params, domain);

  AABB box = {{0.05f, 0.05f, 0.05f}, {0.45f, 0.45f, 0.45f}};
  solver.AddParticlesInBox(box, 0.03f);

  float mass_before = solver.TotalMass();
  CHECK(mass_before > 0);

  // Run 50 steps.
  for (int i = 0; i < 50; ++i) {
    solver.Step(0.0005f);
  }

  float mass_after = solver.TotalMass();
  // Mass must be exactly conserved (particles don't disappear).
  CHECK_NEAR(mass_after, mass_before, 1e-6f);
}

TEST(SPH_KernelNormalization) {
  WendlandC2 kernel;
  kernel.Init(0.02f);

  // W(0) should be the maximum value.
  float w0 = kernel.W(0.0f);
  CHECK(w0 > 0);

  // W at support radius should be zero.
  float w_support = kernel.W(kernel.support);
  CHECK_NEAR(w_support, 0.0f, 1e-6f);

  // W should decrease monotonically.
  float w_mid = kernel.W(kernel.support * 0.5f);
  CHECK(w_mid > 0);
  CHECK(w_mid < w0);
}

TEST(SPH_GradientDirection) {
  WendlandC2 kernel;
  kernel.Init(0.02f);

  // Gradient should be negative (kernel decreases with distance).
  float dw = kernel.DW(0.01f);
  CHECK(dw < 0);

  // Zero beyond support.
  float dw_far = kernel.DW(0.05f);
  CHECK_NEAR(dw_far, 0.0f, 1e-10f);
}

int main() {
  return RunAllTests();
}
