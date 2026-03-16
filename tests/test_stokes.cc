// Tests for Stokes creeping flow solver.
#include "test_harness.h"
#include "mjwater/stokes.h"
#include <cmath>

using namespace mjwater;

// --- Basic initialization ---

TEST(StokesInit) {
  StokesSolver solver;
  StokesParams p;
  p.dx = 10e-6f;
  solver.Init(10, 10, 10, p);

  CHECK(solver.grid.CellCount() == 1000);
  CHECK(solver.MaxSpeed() == 0.0f);
  CHECK(solver.TotalKineticEnergy() == 0.0f);
}

// --- Stable timestep calculation ---

TEST(StableDt) {
  StokesParams p;
  p.dx = 10e-6f;        // 10 um
  p.viscosity = 1e-3f;  // water
  p.density = 998.2f;

  float dt = p.StableDt();
  // nu = 1e-3/998.2 ~ 1.002e-6 m^2/s
  // dt < dx^2/(6*nu) = (1e-10)/(6*1e-6) ~ 1.67e-5
  // with safety 0.15: dt ~ 2.5e-6
  CHECK(dt > 0);
  CHECK(dt < 1e-4f);  // sanity: not too large
}

// --- Reynolds number ---

TEST(ReynoldsNumber) {
  StokesParams p;
  p.viscosity = 1e-3f;
  p.density = 1000.0f;

  // E. coli: L ~ 2um, U ~ 30um/s (Berg 1993)
  float Re = p.Reynolds(2e-6f, 30e-6f);
  // Re = 1000 * 30e-6 * 2e-6 / 1e-3 = 6e-8
  CHECK(Re < 1e-4f);  // definitely creeping flow
  CHECK(Re > 0);
}

// --- Pressure-driven channel flow (Poiseuille) ---
// Stokes flow in a channel with pressure gradient should develop
// a parabolic velocity profile. This is the fundamental validation.

TEST(PoiseuilleFlow) {
  // Small channel: 5x20x5 grid, flow in y-direction.
  uint32_t nx = 5, ny = 20, nz = 5;
  StokesSolver solver;
  StokesParams p;
  p.dx = 50e-6f;  // 50um spacing (250um x 1mm x 250um channel)
  p.viscosity = 1e-3f;
  p.density = 1000.0f;
  p.pressure_iters = 100;
  solver.Init(nx, ny, nz, p);

  // Set solid walls on x and z boundaries (channel walls).
  for (uint32_t z = 0; z < nz; ++z) {
    for (uint32_t y = 0; y < ny; ++y) {
      solver.SetSolid(0, y, z, true);
      solver.SetSolid(nx-1, y, z, true);
    }
  }
  for (uint32_t x = 0; x < nx; ++x) {
    for (uint32_t y = 0; y < ny; ++y) {
      solver.SetSolid(x, y, 0, true);
      solver.SetSolid(x, y, nz-1, true);
    }
  }

  // Apply body force in y-direction (mimics pressure gradient).
  solver.SetBodyForce({0, 1.0f, 0});  // 1 N/m^3

  // Step many times to reach steady state.
  float dt = p.StableDt();
  for (int i = 0; i < 200; ++i) {
    solver.Step(dt);
  }

  // Check: velocity should be nonzero in interior.
  float max_v = solver.MaxSpeed();
  CHECK(max_v > 0);

  // Check: center velocity > edge velocity (parabolic profile).
  // Center of channel: x=2, z=2
  float v_center = solver.grid.v[solver.grid.Idx(2, ny/2, 2)];
  // Near wall: x=1, z=2
  float v_edge = solver.grid.v[solver.grid.Idx(1, ny/2, 2)];

  CHECK(v_center > v_edge);
}

// --- Divergence-free after projection ---

TEST(DivergenceFree) {
  StokesSolver solver;
  StokesParams p;
  p.dx = 50e-6f;
  p.pressure_iters = 100;
  solver.Init(10, 10, 10, p);

  // Apply force and step.
  solver.SetBodyForce({0, 0, -0.01f});
  float dt = p.StableDt();
  solver.Step(dt);

  // Max divergence should be very small.
  float max_div = solver.MaxDivergence();
  CHECK(max_div < 1.0f);  // should be much smaller in practice
}

// --- Concentration diffusion ---

TEST(ConcentrationDiffusion) {
  // Point source of concentration should spread over time.
  StokesSolver solver;
  StokesParams p;
  p.dx = 10e-6f;
  p.diffusivity = 1e-9f;  // typical small molecule
  solver.Init(20, 20, 20, p);

  // Set initial concentration at center.
  uint32_t cx = 10, cy = 10, cz = 10;
  solver.grid.concentration[solver.grid.Idx(cx, cy, cz)] = 1000.0f;

  // Step with no flow (pure diffusion).
  float dt = p.StableDt() * 0.5f;  // conservative
  for (int i = 0; i < 50; ++i) {
    solver.Step(dt);
  }

  // Center concentration should have decreased (spread out).
  float center_c = solver.grid.concentration[solver.grid.Idx(cx, cy, cz)];
  CHECK(center_c < 1000.0f);
  CHECK(center_c > 0.0f);

  // Neighbors should have nonzero concentration.
  float neighbor_c = solver.grid.concentration[solver.grid.Idx(cx+1, cy, cz)];
  CHECK(neighbor_c > 0.0f);
}

// --- Concentration source ---

TEST(ConcentrationSource) {
  StokesSolver solver;
  StokesParams p;
  p.dx = 10e-6f;
  solver.Init(10, 10, 10, p);

  // Add continuous source at center.
  solver.SetSource(5, 5, 5, 100.0f);

  float dt = p.StableDt() * 0.5f;
  for (int i = 0; i < 20; ++i) {
    solver.Step(dt);
  }

  // Concentration at source should have grown.
  float c = solver.grid.concentration[solver.grid.Idx(5, 5, 5)];
  CHECK(c > 0.0f);
}

// --- Quiescent state ---

TEST(QuiescentStable) {
  StokesSolver solver;
  StokesParams p;
  solver.Init(10, 10, 10, p);

  // No forces, no initial velocity -> should stay at rest.
  float dt = p.StableDt();
  for (int i = 0; i < 10; ++i) {
    solver.Step(dt);
  }

  CHECK(solver.MaxSpeed() < 1e-20f);
}

// --- Solid boundary no-slip ---

TEST(SolidNoSlip) {
  StokesSolver solver;
  StokesParams p;
  p.dx = 50e-6f;
  solver.Init(10, 10, 10, p);

  // Mark center cell as solid.
  solver.SetSolid(5, 5, 5, true);
  solver.SetBodyForce({0, 0, -0.1f});

  float dt = p.StableDt();
  for (int i = 0; i < 50; ++i) {
    solver.Step(dt);
  }

  // Solid cell should have zero velocity.
  size_t solid_idx = solver.grid.Idx(5, 5, 5);
  CHECK(solver.grid.u[solid_idx] == 0.0f);
  CHECK(solver.grid.v[solid_idx] == 0.0f);
  CHECK(solver.grid.w[solid_idx] == 0.0f);
}

// --- Velocity query ---

TEST(VelocityQuery) {
  StokesSolver solver;
  StokesParams p;
  p.dx = 50e-6f;
  solver.Init(10, 10, 10, p);

  // Set known velocity at a cell.
  solver.grid.u[solver.grid.Idx(5, 5, 5)] = 1e-4f;

  Vec3 pos;
  solver.grid.GridToWorld(5, 5, 5, pos.x, pos.y, pos.z);
  Vec3 v = solver.Velocity(pos);

  CHECK(std::abs(v.x - 1e-4f) < 1e-5f);
}

int main() { return RunAllTests(); }
