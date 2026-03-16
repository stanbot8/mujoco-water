// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#include "test_harness.h"
#include "mjwater/lbm.h"

using namespace mjwater;

TEST(LBM_InitAtRest) {
  LBMSolver solver;
  LBMParams params;
  params.dx_phys = 0.001f;
  solver.Init(10, 10, 10, params);
  solver.InitAtRest(1.0f);

  // All cells should have density ~1.0 and velocity ~0.
  for (uint32_t z = 0; z < 10; ++z) {
    for (uint32_t y = 0; y < 10; ++y) {
      for (uint32_t x = 0; x < 10; ++x) {
        size_t c = solver.grid.Idx(x, y, z);
        if (solver.grid.solid[c]) continue;
        float rho = solver.grid.Density(c);
        CHECK_NEAR(rho, 1.0f, 1e-5f);
        Vec3 u = solver.grid.Velocity(c);
        CHECK(std::abs(u.x) < 1e-5f);
        CHECK(std::abs(u.y) < 1e-5f);
        CHECK(std::abs(u.z) < 1e-5f);
      }
    }
  }
}

TEST(LBM_MassConservation) {
  LBMSolver solver;
  LBMParams params;
  params.dx_phys = 0.001f;
  solver.Init(10, 10, 10, params);
  solver.InitAtRest(1.0f);

  // Zero out gravity to test pure conservation (no open-boundary leakage).
  for (auto& f : solver.body_force) f = {};

  float mass_before = solver.TotalMass();
  CHECK(mass_before > 0);

  // Run 20 steps.
  for (int i = 0; i < 20; ++i) {
    solver.Step();
  }

  float mass_after = solver.TotalMass();
  float rel_err = std::abs(mass_after - mass_before) / mass_before;
  CHECK(rel_err < 0.01f);
}

TEST(LBM_EquilibriumConsistency) {
  // Equilibrium distribution should recover input density and velocity.
  float rho = 1.5f;
  Vec3 u = {0.01f, -0.02f, 0.005f};
  float feq[kQ];
  LBMSolver::Equilibrium(rho, u, feq);

  // Sum feq = rho.
  float sum_rho = 0;
  Vec3 sum_u{};
  for (int i = 0; i < kQ; ++i) {
    sum_rho += feq[i];
    sum_u.x += feq[i] * kEx[i];
    sum_u.y += feq[i] * kEy[i];
    sum_u.z += feq[i] * kEz[i];
  }

  CHECK_NEAR(sum_rho, rho, 1e-5f);
  CHECK_NEAR(sum_u.x / rho, u.x, 1e-5f);
  CHECK_NEAR(sum_u.y / rho, u.y, 1e-5f);
  CHECK_NEAR(sum_u.z / rho, u.z, 1e-5f);
}

TEST(LBM_PhysicalTimestep) {
  LBMParams params;
  params.tau = 0.8f;
  params.dx_phys = 0.001f;
  params.target_viscosity = kWaterViscosity;
  params.ComputeTimestep();

  // dt should be positive and very small.
  CHECK(params.dt_phys > 0);
  CHECK(params.dt_phys < 1.0f);  // much less than 1 second
}

TEST(LBM_BounceBack) {
  // Body force drives flow; solid walls should produce no-slip.
  // Run short to avoid open-boundary divergence.
  LBMSolver solver;
  LBMParams params;
  params.dx_phys = 0.001f;
  params.tau = 0.8f;
  params.target_viscosity = 0.001f;
  uint32_t nx = 5, ny = 21, nz = 5;
  solver.Init(nx, ny, nz, params);

  float G_lattice = 1e-6f;
  for (auto& f : solver.body_force) f = {G_lattice, 0, 0};

  // Solid walls at y=0 and y=ny-1 (Poiseuille plates).
  for (uint32_t z = 0; z < nz; ++z) {
    for (uint32_t x = 0; x < nx; ++x) {
      solver.SetSolid(x, 0, z, true);
      solver.SetSolid(x, ny - 1, z, true);
      solver.body_force[solver.grid.Idx(x, 0, z)] = {};
      solver.body_force[solver.grid.Idx(x, ny - 1, z)] = {};
    }
  }

  // Run 100 steps (before open-boundary divergence).
  for (int i = 0; i < 100; ++i) solver.Step();

  size_t c_mid = solver.grid.Idx(nx / 2, ny / 2, nz / 2);
  Vec3 u_mid = solver.grid.Velocity(c_mid);

  // Center should have positive x-velocity (body force drives flow).
  CHECK(u_mid.x > 0);

  // Velocity near wall should be smaller (no-slip effect).
  size_t c_wall = solver.grid.Idx(nx / 2, 1, nz / 2);
  Vec3 u_wall = solver.grid.Velocity(c_wall);
  CHECK(u_wall.x < u_mid.x);

  // Density should be stable (~1.0).
  float rho = solver.grid.Density(c_mid);
  CHECK(std::abs(rho - 1.0f) < 0.01f);
}

TEST(LBM_MomentumConservation) {
  // No body force: momentum should be conserved.
  LBMSolver solver;
  LBMParams params;
  params.dx_phys = 0.001f;
  solver.Init(10, 10, 10, params);

  // Clear forces.
  for (auto& f : solver.body_force) f = {};

  // Set initial momentum in x.
  for (uint32_t z = 1; z < 9; ++z) {
    for (uint32_t y = 1; y < 9; ++y) {
      for (uint32_t x = 1; x < 9; ++x) {
        size_t c = solver.grid.Idx(x, y, z);
        if (solver.grid.solid[c]) continue;
        Vec3 u = {0.01f, 0, 0};
        float feq[kQ];
        LBMSolver::Equilibrium(1.0f, u, feq);
        float* fi = solver.grid.f_src.data() + c * kQ;
        for (int i = 0; i < kQ; ++i) fi[i] = feq[i];
      }
    }
  }

  // Measure momentum.
  float mom_x_before = 0;
  for (size_t c = 0; c < solver.grid.CellCount(); ++c) {
    if (solver.grid.solid[c]) continue;
    Vec3 u = solver.grid.Velocity(c);
    float rho = solver.grid.Density(c);
    mom_x_before += rho * u.x;
  }

  for (int i = 0; i < 50; ++i) solver.Step();

  float mom_x_after = 0;
  for (size_t c = 0; c < solver.grid.CellCount(); ++c) {
    if (solver.grid.solid[c]) continue;
    Vec3 u = solver.grid.Velocity(c);
    float rho = solver.grid.Density(c);
    mom_x_after += rho * u.x;
  }

  // Momentum should be approximately conserved (open boundary leakage OK).
  float rel_err = std::abs(mom_x_after - mom_x_before) /
                  (std::abs(mom_x_before) + 1e-10f);
  CHECK(rel_err < 0.1f);
}

int main() {
  return RunAllTests();
}
