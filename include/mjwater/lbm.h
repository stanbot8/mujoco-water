// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// LOD 2: Lattice Boltzmann Method (D3Q19 BGK) solver.
//
// Solves the incompressible Navier-Stokes equations at cellular
// scales via the Boltzmann transport equation on a regular lattice.
//
// References:
//   Kruger et al. (2017) "The Lattice Boltzmann Method: Principles
//     and Practice" (Springer)
//   Succi, S. (2001) "The Lattice Boltzmann Equation for Fluid
//     Dynamics and Beyond" (Oxford)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mjwater/types.h"
#include "mjwater/water_grid.h"

namespace mjwater {

struct LBMParams {
  float tau = 0.8f;            // BGK relaxation time (> 0.5 for stability)
  float dx_phys = 0.001f;     // physical lattice spacing (m)
  float dt_phys = 0;          // physical timestep (computed from tau, dx, viscosity)
  float target_viscosity = kWaterViscosity;  // target kinematic viscosity (m^2/s)
  bool use_trt = true;         // use Two-Relaxation-Time (TRT) collision
  float magic_param = 0.25f;   // TRT magic parameter (1/4 = exact bounce-back)

  // Compute physical timestep from lattice parameters.
  // nu = c_s^2 * (tau - 0.5) * dt_lattice
  // In lattice units: dt_lattice = 1, dx_lattice = 1, c_s^2 = 1/3
  // Physical: nu_phys = (dx_phys^2 / dt_phys) * (tau - 0.5) / 3
  // => dt_phys = dx_phys^2 * (tau - 0.5) / (3 * nu_phys)
  void ComputeTimestep() {
    dt_phys = dx_phys * dx_phys * (tau - 0.5f) /
              (3.0f * target_viscosity);
  }

  // Lattice-to-physical conversion factors.
  float VelocityScale() const { return dx_phys / dt_phys; }
  float ForceScale() const { return dx_phys / (dt_phys * dt_phys); }
  float DensityScale() const { return kWaterDensity; }
};

struct LBMSolver {
  LatticeGrid grid;
  LBMParams params;

  // External body force per cell (in lattice units).
  // Applied via Guo forcing scheme.
  std::vector<Vec3> body_force;

  void Init(uint32_t nx, uint32_t ny, uint32_t nz, const LBMParams& p,
            float ox = 0, float oy = 0, float oz = 0) {
    params = p;
    params.ComputeTimestep();
    grid.Init(nx, ny, nz, p.dx_phys, ox, oy, oz);
    body_force.assign(grid.CellCount(), Vec3{});

    // Set default gravity force (convert physical to lattice units).
    // f_lattice = f_physical * dt^2 / dx (in lattice Boltzmann units)
    float gz_lattice = -kGravity * params.dt_phys * params.dt_phys /
                       params.dx_phys;
    for (auto& f : body_force) f.z = gz_lattice;
  }

  // Compute equilibrium distribution for given density and velocity.
  static void Equilibrium(float rho, Vec3 u, float feq[kQ]) {
    float u_dot_u = u.Dot(u);
    for (int i = 0; i < kQ; ++i) {
      float e_dot_u = kEx[i] * u.x + kEy[i] * u.y + kEz[i] * u.z;
      feq[i] = kW[i] * rho * (1.0f + e_dot_u / kCsSq +
               e_dot_u * e_dot_u / (2.0f * kCsSq * kCsSq) -
               u_dot_u / (2.0f * kCsSq));
    }
  }

  // One LBM timestep: collide, apply forcing, stream, bounce-back.
  void Step() {
    const uint32_t nx = grid.nx, ny = grid.ny, nz = grid.nz;
    float omega_plus = 1.0f / params.tau;  // symmetric relaxation rate
    // TRT: antisymmetric relaxation rate from magic parameter.
    // Lambda = (tau_s - 0.5) * (tau_a - 0.5) = magic_param (typically 1/4).
    // tau_a = 0.5 + magic_param / (tau_s - 0.5)
    float tau_anti = 0.5f + params.magic_param / (params.tau - 0.5f);
    float omega_minus = 1.0f / tau_anti;
    bool use_trt = params.use_trt;

    // --- Collision + Guo forcing (fluid) / bounce-back (solid) ---
    for (uint32_t z = 0; z < nz; ++z) {
      for (uint32_t y = 0; y < ny; ++y) {
        for (uint32_t x = 0; x < nx; ++x) {
          size_t c = grid.Idx(x, y, z);
          float* fi = grid.f_src.data() + c * kQ;
          float* fo = grid.f_dst.data() + c * kQ;

          // Solid cells: bounce-back (reflect pre-collision distributions).
          // Must happen here so streaming picks up the reflected values.
          if (grid.solid[c]) {
            for (int i = 0; i < kQ; ++i) fo[i] = fi[kOpp[i]];
            continue;
          }

          // Compute macroscopic fields.
          float rho = 0;
          Vec3 u{};
          for (int i = 0; i < kQ; ++i) {
            rho += fi[i];
            u.x += fi[i] * kEx[i];
            u.y += fi[i] * kEy[i];
            u.z += fi[i] * kEz[i];
          }
          if (rho > 1e-10f) { u.x /= rho; u.y /= rho; u.z /= rho; }

          // Add half-force correction to velocity (Guo scheme).
          Vec3 F = body_force[c];
          Vec3 u_eq = u + F * (0.5f / rho);

          // Compute equilibrium.
          float feq[kQ];
          Equilibrium(rho, u_eq, feq);

          if (use_trt) {
            // TRT collision: separate symmetric and antisymmetric parts.
            // f_i^+ = (f_i + f_opp_i) / 2, f_i^- = (f_i - f_opp_i) / 2
            // Same for feq. Relax each with its own rate.
            for (int i = 0; i < kQ; ++i) {
              int opp = kOpp[i];
              float fi_plus  = 0.5f * (fi[i] + fi[opp]);
              float fi_minus = 0.5f * (fi[i] - fi[opp]);
              float feq_plus  = 0.5f * (feq[i] + feq[opp]);
              float feq_minus = 0.5f * (feq[i] - feq[opp]);

              // Guo source term.
              float e_dot_u = kEx[i] * u.x + kEy[i] * u.y + kEz[i] * u.z;
              float si_x = (kEx[i] - u.x) / kCsSq +
                           e_dot_u * kEx[i] / (kCsSq * kCsSq);
              float si_y = (kEy[i] - u.y) / kCsSq +
                           e_dot_u * kEy[i] / (kCsSq * kCsSq);
              float si_z = (kEz[i] - u.z) / kCsSq +
                           e_dot_u * kEz[i] / (kCsSq * kCsSq);
              float Si = (1.0f - 0.5f * omega_plus) * kW[i] *
                         (si_x * F.x + si_y * F.y + si_z * F.z);

              fo[i] = fi[i]
                      - omega_plus  * (fi_plus  - feq_plus)
                      - omega_minus * (fi_minus - feq_minus)
                      + Si;
            }
          } else {
            // BGK collision with Guo forcing.
            for (int i = 0; i < kQ; ++i) {
              float e_dot_u = kEx[i] * u.x + kEy[i] * u.y + kEz[i] * u.z;
              float si_x = (kEx[i] - u.x) / kCsSq +
                           e_dot_u * kEx[i] / (kCsSq * kCsSq);
              float si_y = (kEy[i] - u.y) / kCsSq +
                           e_dot_u * kEy[i] / (kCsSq * kCsSq);
              float si_z = (kEz[i] - u.z) / kCsSq +
                           e_dot_u * kEz[i] / (kCsSq * kCsSq);
              float Si = (1.0f - 0.5f * omega_plus) * kW[i] *
                         (si_x * F.x + si_y * F.y + si_z * F.z);

              fo[i] = fi[i] - omega_plus * (fi[i] - feq[i]) + Si;
            }
          }
        }
      }
    }

    // --- Streaming ---
    // Move distributions from f_dst (post-collision) to f_src (post-stream)
    // by pulling: f_src[x,y,z,i] = f_dst[x-ex, y-ey, z-ez, i]
    for (uint32_t z = 0; z < nz; ++z) {
      for (uint32_t y = 0; y < ny; ++y) {
        for (uint32_t x = 0; x < nx; ++x) {
          size_t c = grid.Idx(x, y, z);
          float* fi = grid.f_src.data() + c * kQ;

          for (int i = 0; i < kQ; ++i) {
            int sx = static_cast<int>(x) - kEx[i];
            int sy = static_cast<int>(y) - kEy[i];
            int sz = static_cast<int>(z) - kEz[i];

            // Periodic or clamp boundary.
            if (sx < 0 || sx >= static_cast<int>(nx) ||
                sy < 0 || sy >= static_cast<int>(ny) ||
                sz < 0 || sz >= static_cast<int>(nz)) {
              // Open boundary: keep post-collision value.
              fi[i] = grid.f_dst[c * kQ + i];
              continue;
            }

            size_t src_c = grid.Idx(static_cast<uint32_t>(sx),
                                     static_cast<uint32_t>(sy),
                                     static_cast<uint32_t>(sz));
            fi[i] = grid.f_dst[src_c * kQ + i];
          }
        }
      }
    }

    // Bounce-back is now handled during collision (above), so streaming
    // naturally picks up reflected distributions from solid cells.
  }

  // Physical timestep (seconds).
  float PhysicalDt() const { return params.dt_phys; }

  // Query macroscopic density at grid coordinates (in physical units).
  float DensityPhysical(uint32_t x, uint32_t y, uint32_t z) const {
    size_t c = grid.Idx(x, y, z);
    return grid.Density(c) * params.DensityScale();
  }

  // Query macroscopic velocity at grid coordinates (in physical units).
  Vec3 VelocityPhysical(uint32_t x, uint32_t y, uint32_t z) const {
    size_t c = grid.Idx(x, y, z);
    Vec3 u = grid.Velocity(c);
    float scale = params.VelocityScale();
    return u * scale;
  }

  // Set a cell as solid (bounce-back boundary).
  void SetSolid(uint32_t x, uint32_t y, uint32_t z, bool solid) {
    grid.solid[grid.Idx(x, y, z)] = solid ? 1 : 0;
  }

  // Initialize the lattice at rest with uniform density.
  void InitAtRest(float rho_lattice = 1.0f) {
    for (size_t c = 0; c < grid.CellCount(); ++c) {
      float feq[kQ];
      Equilibrium(rho_lattice, Vec3{}, feq);
      float* fi = grid.f_src.data() + c * kQ;
      for (int i = 0; i < kQ; ++i) fi[i] = feq[i];
    }
  }

  // Total mass in lattice units (sum of all densities).
  float TotalMass() const {
    float total = 0;
    for (size_t c = 0; c < grid.CellCount(); ++c) {
      if (!grid.solid[c]) total += grid.Density(c);
    }
    return total;
  }
};

}  // namespace mjwater
