// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// LOD 4 (finest): Stokes creeping flow solver for cellular scales.
//
// At cellular scales (capillaries, cells, microorganisms), the Reynolds
// number is << 1: viscous forces dominate inertia completely. The full
// Navier-Stokes equations reduce to the Stokes equations:
//
//   -grad(p) + mu * laplacian(u) + f = 0    (momentum)
//   div(u) = 0                                (incompressibility)
//
// Solved on a regular 3D grid via pressure projection with Jacobi
// relaxation. Each timestep:
//   1. Compute tentative velocity from body forces + viscous diffusion
//   2. Solve pressure Poisson equation (Jacobi iteration)
//   3. Project velocity to divergence-free
//
// Also supports a scalar concentration field for chemotaxis/diffusion:
//   dc/dt = D * laplacian(c) - u . grad(c) + S
//
// References:
//   Happel, J. & Brenner, H. (1983) "Low Reynolds Number Hydrodynamics"
//     (Martinus Nijhoff). Canonical Stokes flow reference.
//   Purcell, E.M. (1977) "Life at Low Reynolds Number"
//     Am. J. Phys. 45(1), 3-11. Biology motivation.
//   Kim, S. & Karrila, S.J. (2005) "Microhydrodynamics: Principles
//     and Selected Applications" (Dover). Numerical methods.
//   Berg, H.C. (1993) "Random Walks in Biology" (Princeton)
//     Biological diffusion scales.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mjwater/types.h"
#include "mjwater/grid_ops.h"

namespace mjwater {

// --- Stokes flow parameters ---
// All physical constants grounded in literature.

struct StokesParams {
  float dx = 10e-6f;           // grid spacing (m); 10 um default (cell scale)
  float viscosity = 1.0e-3f;   // dynamic viscosity (Pa*s); water at 20C = 1.002e-3
  float density = 998.2f;      // fluid density (kg/m^3)
  float diffusivity = 1.0e-9f; // scalar diffusion coefficient (m^2/s)
                                // typical small molecule in water ~ 1e-9 m^2/s
                                // (Berg 1993, Table 1.2)

  uint32_t pressure_iters = 50; // Jacobi iterations for pressure solve
  float pressure_omega = 0.8f;  // relaxation parameter (under-relax for stability)

  // Derived: kinematic viscosity.
  float KinematicViscosity() const { return viscosity / density; }

  // Compute stable timestep for explicit viscous diffusion.
  // CFL: dt < dx^2 / (6 * nu) for 3D explicit diffusion.
  float StableDt() const {
    float nu = KinematicViscosity();
    return 0.15f * dx * dx / nu;  // safety factor 0.9 * 1/6
  }

  // Reynolds number for a body of size L moving at speed U.
  float Reynolds(float L, float U) const {
    return density * U * L / viscosity;
  }
};

// --- 3D Stokes grid ---

struct StokesGrid {
  uint32_t nx = 0, ny = 0, nz = 0;
  float dx = 10e-6f;
  float origin_x = 0, origin_y = 0, origin_z = 0;

  // Velocity components (staggered MAC grid conceptually, but stored collocated
  // for simplicity at this scale; pressure projection enforces div-free).
  std::vector<float> u, v, w;       // velocity (m/s)
  std::vector<float> u_tmp, v_tmp, w_tmp;  // scratch
  std::vector<float> p;              // pressure (Pa)
  std::vector<float> div;            // divergence scratch
  std::vector<float> concentration;  // scalar field (mol/m^3 or arbitrary)
  std::vector<float> conc_tmp;       // scratch for advection-diffusion
  std::vector<float> source;         // concentration source term (mol/(m^3*s))
  std::vector<uint8_t> solid;        // boundary mask (1 = solid)

  // Body force (per unit volume, N/m^3).
  Vec3 body_force = {0, 0, 0};

  size_t CellCount() const {
    return static_cast<size_t>(nx) * ny * nz;
  }

  bool Contains(Vec3 pos) const {
    return Contains3D(pos, origin_x, origin_y, origin_z, dx, nx, ny, nz);
  }

  size_t Idx(uint32_t x, uint32_t y, uint32_t z) const {
    return Idx3D(x, y, z, nx, ny);
  }

  void Init(uint32_t nx_, uint32_t ny_, uint32_t nz_, float dx_,
            float ox = 0, float oy = 0, float oz = 0) {
    nx = nx_; ny = ny_; nz = nz_; dx = dx_;
    origin_x = ox; origin_y = oy; origin_z = oz;
    size_t n = CellCount();
    u.assign(n, 0); v.assign(n, 0); w.assign(n, 0);
    u_tmp.assign(n, 0); v_tmp.assign(n, 0); w_tmp.assign(n, 0);
    p.assign(n, 0);
    div.assign(n, 0);
    concentration.assign(n, 0);
    conc_tmp.assign(n, 0);
    source.assign(n, 0);
    solid.assign(n, 0);
  }

  void GridToWorld(uint32_t x, uint32_t y, uint32_t z,
                    float& wx, float& wy, float& wz) const {
    GridToWorld3D(x, y, z, origin_x, origin_y, origin_z, dx, wx, wy, wz);
  }

  bool WorldToGridCell(float wx, float wy, float wz,
                        uint32_t& cx, uint32_t& cy, uint32_t& cz) const {
    return WorldToGridCell3D(wx, wy, wz, origin_x, origin_y, origin_z, dx,
                              nx, ny, nz, cx, cy, cz);
  }

  void Clear() {
    ClearVelocityFields(u, v, w);
    std::fill(p.begin(), p.end(), 0.0f);
  }
};

// --- Stokes solver ---

struct StokesSolver {
  StokesGrid grid;
  StokesParams params;

  void Init(uint32_t nx, uint32_t ny, uint32_t nz,
            const StokesParams& p,
            float ox = 0, float oy = 0, float oz = 0) {
    params = p;
    grid.Init(nx, ny, nz, p.dx, ox, oy, oz);
  }

  // Set a cell as solid boundary (no-slip).
  void SetSolid(uint32_t x, uint32_t y, uint32_t z, bool is_solid) {
    grid.solid[grid.Idx(x, y, z)] = is_solid ? 1 : 0;
  }

  // Set body force (e.g. gravity, pressure gradient driving flow).
  void SetBodyForce(Vec3 f) { grid.body_force = f; }

  // Set concentration source at a cell.
  void SetSource(uint32_t x, uint32_t y, uint32_t z, float rate) {
    grid.source[grid.Idx(x, y, z)] = rate;
  }

  // One Stokes timestep.
  // dt should be <= params.StableDt() for stability.
  void Step(float dt) {
    // 1. Apply body forces and viscous diffusion to get tentative velocity.
    DiffuseAndForce(dt);

    // 2. Pressure projection: solve for pressure that makes velocity div-free.
    ComputeDivergence();
    SolvePressure(dt);
    ProjectVelocity(dt);

    // 3. Enforce boundary conditions.
    EnforceBoundaries();

    // 4. Advect-diffuse concentration field.
    AdvectDiffuseConcentration(dt);
  }

  // Query velocity at world position (trilinear interpolation).
  Vec3 Velocity(Vec3 pos) const {
    float gx, gy, gz;
    WorldToGrid3D(pos.x, pos.y, pos.z,
                   grid.origin_x, grid.origin_y, grid.origin_z, grid.dx,
                   gx, gy, gz);
    gx -= 0.5f; gy -= 0.5f; gz -= 0.5f;
    return mjwater::TrilinearVec3(gx, gy, gz, grid.u, grid.v, grid.w,
                                   grid.nx, grid.ny, grid.nz);
  }

  // Query concentration at world position.
  float Concentration(Vec3 pos) const {
    float gx, gy, gz;
    WorldToGrid3D(pos.x, pos.y, pos.z,
                   grid.origin_x, grid.origin_y, grid.origin_z, grid.dx,
                   gx, gy, gz);
    gx -= 0.5f; gy -= 0.5f; gz -= 0.5f;
    return mjwater::TrilinearScalar(gx, gy, gz, grid.concentration,
                                     grid.nx, grid.ny, grid.nz);
  }

  // Query pressure at world position.
  float Pressure(Vec3 pos) const {
    float gx, gy, gz;
    WorldToGrid3D(pos.x, pos.y, pos.z,
                   grid.origin_x, grid.origin_y, grid.origin_z, grid.dx,
                   gx, gy, gz);
    gx -= 0.5f; gy -= 0.5f; gz -= 0.5f;
    return mjwater::TrilinearScalar(gx, gy, gz, grid.p,
                                     grid.nx, grid.ny, grid.nz);
  }

  // Total kinetic energy (for conservation/convergence checks).
  float TotalKineticEnergy() const {
    float e = 0;
    float cell_vol = grid.dx * grid.dx * grid.dx;
    for (size_t i = 0; i < grid.CellCount(); ++i) {
      if (grid.solid[i]) continue;
      float speed2 = grid.u[i] * grid.u[i] + grid.v[i] * grid.v[i] +
                      grid.w[i] * grid.w[i];
      e += speed2;
    }
    return 0.5f * params.density * cell_vol * e;
  }

  // Maximum velocity magnitude (for diagnostics).
  float MaxSpeed() const {
    float max_v2 = 0;
    for (size_t i = 0; i < grid.CellCount(); ++i) {
      float v2 = grid.u[i] * grid.u[i] + grid.v[i] * grid.v[i] +
                  grid.w[i] * grid.w[i];
      max_v2 = std::max(max_v2, v2);
    }
    return std::sqrt(max_v2);
  }

  // Total concentration (for conservation checks).
  float TotalConcentration() const {
    float total = 0;
    float cell_vol = grid.dx * grid.dx * grid.dx;
    for (size_t i = 0; i < grid.CellCount(); ++i) {
      if (!grid.solid[i]) total += grid.concentration[i];
    }
    return total * cell_vol;
  }

  // Maximum divergence (should be ~0 after projection).
  float MaxDivergence() const {
    float max_div = 0;
    uint32_t nx = grid.nx, ny = grid.ny, nz = grid.nz;
    float inv_dx = 1.0f / grid.dx;
    for (uint32_t z = 1; z + 1 < nz; ++z) {
      for (uint32_t y = 1; y + 1 < ny; ++y) {
        for (uint32_t x = 1; x + 1 < nx; ++x) {
          size_t c = grid.Idx(x, y, z);
          if (grid.solid[c]) continue;
          float d = (grid.u[grid.Idx(x+1,y,z)] - grid.u[grid.Idx(x-1,y,z)] +
                     grid.v[grid.Idx(x,y+1,z)] - grid.v[grid.Idx(x,y-1,z)] +
                     grid.w[grid.Idx(x,y,z+1)] - grid.w[grid.Idx(x,y,z-1)]) *
                    (0.5f * inv_dx);
          max_div = std::max(max_div, std::abs(d));
        }
      }
    }
    return max_div;
  }

 private:
  // --- Step 1: Viscous diffusion + body force ---
  // Stokes: no advection (Re << 1). Only diffusion and forcing.
  //   u* = u + dt * (nu * laplacian(u) + f/rho)
  void DiffuseAndForce(float dt) {
    uint32_t nx = grid.nx, ny = grid.ny, nz = grid.nz;
    float nu = params.KinematicViscosity();
    float alpha = nu * dt / (grid.dx * grid.dx);
    Vec3 f_per_rho = grid.body_force / params.density;

    for (uint32_t z = 1; z + 1 < nz; ++z) {
      for (uint32_t y = 1; y + 1 < ny; ++y) {
        for (uint32_t x = 1; x + 1 < nx; ++x) {
          size_t c = grid.Idx(x, y, z);
          if (grid.solid[c]) continue;

          grid.u_tmp[c] = grid.u[c] + alpha * Laplacian6(grid.u, x, y, z, nx, ny) + dt * f_per_rho.x;
          grid.v_tmp[c] = grid.v[c] + alpha * Laplacian6(grid.v, x, y, z, nx, ny) + dt * f_per_rho.y;
          grid.w_tmp[c] = grid.w[c] + alpha * Laplacian6(grid.w, x, y, z, nx, ny) + dt * f_per_rho.z;
        }
      }
    }

    std::swap(grid.u, grid.u_tmp);
    std::swap(grid.v, grid.v_tmp);
    std::swap(grid.w, grid.w_tmp);
  }

  // --- Step 2a: Compute divergence of tentative velocity ---
  void ComputeDivergence() {
    uint32_t nx = grid.nx, ny = grid.ny, nz = grid.nz;
    float inv_2dx = 0.5f / grid.dx;

    for (uint32_t z = 1; z + 1 < nz; ++z) {
      for (uint32_t y = 1; y + 1 < ny; ++y) {
        for (uint32_t x = 1; x + 1 < nx; ++x) {
          size_t c = grid.Idx(x, y, z);
          if (grid.solid[c]) { grid.div[c] = 0; continue; }

          Vec3 gu = Gradient6(grid.u, x, y, z, nx, ny);
          Vec3 gv = Gradient6(grid.v, x, y, z, nx, ny);
          Vec3 gw = Gradient6(grid.w, x, y, z, nx, ny);
          grid.div[c] = (gu.x + gv.y + gw.z) * inv_2dx;
        }
      }
    }
  }

  // --- Step 2b: Solve pressure Poisson equation ---
  // laplacian(p) = (rho / dt) * div(u*)
  // Red-Black Gauss-Seidel with SOR (2x faster convergence than Jacobi).
  void SolvePressure(float dt) {
    uint32_t nx = grid.nx, ny = grid.ny, nz = grid.nz;
    float rho_over_dt = params.density / dt;
    float dx2 = grid.dx * grid.dx;
    float omega = params.pressure_omega;

    for (uint32_t iter = 0; iter < params.pressure_iters; ++iter) {
      // Red-Black ordering: first pass (x+y+z) even, second pass odd.
      // This allows using updated values immediately (Gauss-Seidel)
      // while maintaining parallelism within each color.
      for (int color = 0; color < 2; ++color) {
        for (uint32_t z = 1; z + 1 < nz; ++z) {
          for (uint32_t y = 1; y + 1 < ny; ++y) {
            for (uint32_t x = 1; x + 1 < nx; ++x) {
              if (((x + y + z) & 1) != static_cast<uint32_t>(color)) continue;
              size_t c = grid.Idx(x, y, z);
              if (grid.solid[c]) continue;

              float p_sum =
                grid.p[grid.Idx(x+1,y,z)] + grid.p[grid.Idx(x-1,y,z)] +
                grid.p[grid.Idx(x,y+1,z)] + grid.p[grid.Idx(x,y-1,z)] +
                grid.p[grid.Idx(x,y,z+1)] + grid.p[grid.Idx(x,y,z-1)];

              float p_new = (p_sum - rho_over_dt * grid.div[c] * dx2) / 6.0f;
              grid.p[c] += omega * (p_new - grid.p[c]);
            }
          }
        }
      }
    }
  }

  // --- Step 2c: Project velocity ---
  // u = u* - (dt/rho) * grad(p)
  void ProjectVelocity(float dt) {
    uint32_t nx = grid.nx, ny = grid.ny, nz = grid.nz;
    float dt_over_rho = dt / params.density;
    float inv_2dx = 0.5f / grid.dx;

    for (uint32_t z = 1; z + 1 < nz; ++z) {
      for (uint32_t y = 1; y + 1 < ny; ++y) {
        for (uint32_t x = 1; x + 1 < nx; ++x) {
          size_t c = grid.Idx(x, y, z);
          if (grid.solid[c]) continue;

          Vec3 grad_p = Gradient6(grid.p, x, y, z, nx, ny) * inv_2dx;
          grid.u[c] -= dt_over_rho * grad_p.x;
          grid.v[c] -= dt_over_rho * grad_p.y;
          grid.w[c] -= dt_over_rho * grad_p.z;
        }
      }
    }
  }

  // --- Step 3: No-slip boundary conditions ---
  void EnforceBoundaries() {
    // Solid cells: zero velocity.
    for (size_t i = 0; i < grid.CellCount(); ++i) {
      if (grid.solid[i]) {
        grid.u[i] = 0; grid.v[i] = 0; grid.w[i] = 0;
      }
    }
    // Domain boundaries: zero velocity (no-slip walls).
    ZeroFaceVelocities(grid.u, grid.v, grid.w, grid.nx, grid.ny, grid.nz);
  }

  // --- Step 4: Advection-diffusion of concentration field ---
  // dc/dt = D * laplacian(c) - u . grad(c) + S
  void AdvectDiffuseConcentration(float dt) {
    uint32_t nx = grid.nx, ny = grid.ny, nz = grid.nz;
    float D = params.diffusivity;
    float alpha = D * dt / (grid.dx * grid.dx);
    float inv_2dx = 0.5f / grid.dx;

    for (uint32_t z = 1; z + 1 < nz; ++z) {
      for (uint32_t y = 1; y + 1 < ny; ++y) {
        for (uint32_t x = 1; x + 1 < nx; ++x) {
          size_t c = grid.Idx(x, y, z);
          if (grid.solid[c]) { grid.conc_tmp[c] = 0; continue; }

          float lap_c = Laplacian6(grid.concentration, x, y, z, nx, ny);

          Vec3 grad_c = Gradient6(grid.concentration, x, y, z, nx, ny);
          float advection = (grid.u[c] * grad_c.x +
                             grid.v[c] * grad_c.y +
                             grid.w[c] * grad_c.z) * inv_2dx;

          grid.conc_tmp[c] = grid.concentration[c] +
                              alpha * lap_c -
                              dt * advection +
                              dt * grid.source[c];

          // Clamp to non-negative.
          grid.conc_tmp[c] = std::max(0.0f, grid.conc_tmp[c]);
        }
      }
    }

    std::swap(grid.concentration, grid.conc_tmp);
  }

};

}  // namespace mjwater
