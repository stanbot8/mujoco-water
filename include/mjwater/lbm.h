// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// LOD 3: Lattice Boltzmann Method (D3Q19 TRT) solver.
//
// The Lattice Boltzmann Method (LBM) solves fluid dynamics by tracking
// particle distribution functions f_i on a regular lattice, rather than
// solving the Navier-Stokes equations directly. It recovers the correct
// macroscopic behavior (mass and momentum conservation) through the
// Chapman-Enskog expansion.
//
// D3Q19 lattice:
//   3D lattice with 19 velocity directions per cell: 1 rest + 6 face
//   neighbors + 12 edge neighbors. Each direction i has:
//     - Lattice velocity (ex, ey, ez): integer direction vector
//     - Weight w_i: 1/3 for rest, 1/18 for faces, 1/36 for edges
//   The lattice sound speed is c_s^2 = 1/3 (in lattice units).
//
// Algorithm (each timestep):
//   1. Collision: relax distributions toward equilibrium
//      f_eq = w_i * rho * (1 + e.u/cs^2 + (e.u)^2/(2*cs^4) - u.u/(2*cs^2))
//   2. Forcing: add body forces via Guo scheme (accounts for discrete lattice)
//   3. Bounce-back: solid cells reflect distributions to opposite direction
//   4. Streaming: propagate distributions to neighbor cells (pull scheme)
//
// TRT (Two-Relaxation-Time) collision:
//   Standard BGK uses a single relaxation rate omega = 1/tau. This makes
//   the effective boundary position depend on viscosity (tau), which is
//   unphysical. TRT decomposes distributions into symmetric and
//   antisymmetric parts, relaxing each with its own rate:
//     omega_plus  = 1/tau         (controls viscosity)
//     omega_minus = 1/tau_anti    (controls wall position)
//   The "magic parameter" Lambda = (tau - 0.5)(tau_anti - 0.5) = 1/4
//   gives exact bounce-back wall positioning independent of viscosity.
//
// Guo forcing scheme (Guo et al. 2002):
//   Body forces (gravity, external) are added as a source term in the
//   collision operator. The velocity used for equilibrium is corrected
//   by half the force: u_eq = u + F/(2*rho). The source term Si includes
//   both first-order (e_i - u) and second-order (e_i*e_i*u) contributions
//   to correctly recover the forced Navier-Stokes equations.
//
// Pull streaming:
//   Each cell pulls distributions FROM its neighbors (as opposed to push
//   streaming where each cell pushes TO neighbors). Pull is cache-friendly
//   for reading. Bounce-back on solid cells is handled during collision:
//   solid cells write f[opp_i] = f[i], then streaming naturally picks up
//   the reflected values when pulling from solid neighbors.
//
// Unit conversion:
//   LBM operates in lattice units (dx=1, dt=1). Physical quantities are
//   recovered via scaling:
//     nu_phys = (dx_phys^2 / dt_phys) * (tau - 0.5) / 3
//     u_phys  = u_lattice * dx_phys / dt_phys
//     F_phys  = F_lattice * dx_phys / dt_phys^2
//
// References:
//   Kruger et al. (2017) "The Lattice Boltzmann Method: Principles
//     and Practice" (Springer). Comprehensive modern reference.
//   Guo, Z. et al. (2002) "Discrete lattice effects on the forcing
//     term in the LBM" Phys. Rev. E 65(4). Guo forcing scheme.
//   Ginzburg, I. (2005) "Equilibrium-type and link-type lattice
//     Boltzmann models" Adv. Water Resources. TRT magic parameter.
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

    // Convert gravitational acceleration to lattice force units.
    // In Guo forcing, F is acceleration in lattice units:
    //   F_lattice = a_physical * dt_phys^2 / dx_phys
    // This comes from: a_phys [m/s^2] * (dt_phys/dx_phys) * dt_phys
    //                 = a_phys * dt^2/dx [lattice acceleration]
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

          // Guo half-force velocity correction: the physical velocity is
          // u_phys = (sum f_i * e_i)/rho + F*dt/2. Since body_force stores
          // acceleration (force per unit mass, not per unit volume), the
          // correction is simply F * 0.5 (half timestep in lattice units).
          Vec3 F = body_force[c];
          Vec3 u_eq = u + F * 0.5f;

          // Compute equilibrium.
          float feq[kQ];
          Equilibrium(rho, u_eq, feq);

          // Compute Guo source terms for all directions.
          float Si[kQ];
          for (int i = 0; i < kQ; ++i) {
            float e_dot_u = kEx[i] * u.x + kEy[i] * u.y + kEz[i] * u.z;
            float si_x = (kEx[i] - u.x) / kCsSq +
                         e_dot_u * kEx[i] / (kCsSq * kCsSq);
            float si_y = (kEy[i] - u.y) / kCsSq +
                         e_dot_u * kEy[i] / (kCsSq * kCsSq);
            float si_z = (kEz[i] - u.z) / kCsSq +
                         e_dot_u * kEz[i] / (kCsSq * kCsSq);
            Si[i] = (1.0f - 0.5f * omega_plus) * kW[i] *
                    (si_x * F.x + si_y * F.y + si_z * F.z);
          }

          if (use_trt) {
            // TRT collision: separate symmetric and antisymmetric parts.
            for (int i = 0; i < kQ; ++i) {
              int opp = kOpp[i];
              float fi_plus   = 0.5f * (fi[i] + fi[opp]);
              float fi_minus  = 0.5f * (fi[i] - fi[opp]);
              float feq_plus  = 0.5f * (feq[i] + feq[opp]);
              float feq_minus = 0.5f * (feq[i] - feq[opp]);
              fo[i] = fi[i]
                      - omega_plus  * (fi_plus  - feq_plus)
                      - omega_minus * (fi_minus - feq_minus)
                      + Si[i];
            }
          } else {
            // BGK collision.
            for (int i = 0; i < kQ; ++i) {
              fo[i] = fi[i] - omega_plus * (fi[i] - feq[i]) + Si[i];
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
