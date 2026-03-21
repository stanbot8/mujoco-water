// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Bidirectional state transfer between LOD levels.
//
// Handles escalation (coarse -> fine) and de-escalation (fine -> coarse)
// with mass/momentum conservation corrections.
//
// Transitions:
//   Ocean <-> SWE: spectral surface <-> height field boundary conditions
//   SWE <-> SPH: height columns <-> particle distributions
//   SPH <-> LBM: particle fields <-> equilibrium distributions
//   LBM <-> Stokes: lattice fields <-> Stokes grid velocity/pressure

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mjwater/types.h"
#include "mjwater/water_grid.h"
#include "mjwater/spectral_ocean.h"
#include "mjwater/shallow_water.h"
#include "mjwater/sph.h"
#include "mjwater/lbm.h"
#include "mjwater/stokes.h"
#include "mjwater/lod_manager.h"
#include "mjwater/flux_register.h"

namespace mjwater {

// Ghost zone blend factor: linearly ramps from 1.0 at the boundary of the
// fine solver to 0.0 at ghost_cells depth into the coarse solver. Used by
// all coupling functions to smoothly blend coarse data into fine boundaries.
inline float GhostBlend(int dist_from_boundary, int ghost_cells) {
  return 1.0f - static_cast<float>(dist_from_boundary) / ghost_cells;
}

struct LODTransition {

  // --- SWE -> SPH: Spawn particles from height field columns ---
  //
  // For each SWE cell in the target region, create a column of particles
  // matching the local depth and velocity. Total mass is conserved.
  static void SWEToSPH(const ShallowWaterSolver& swe, SPHSolver& sph,
                        const AABB& region, float particle_spacing) {
    float mass = sph.params.ParticleMass(particle_spacing);
    const auto& h_data  = swe.grid.channels[swe.ch_h].data;
    const auto& hu_data = swe.grid.channels[swe.ch_hu].data;
    const auto& hv_data = swe.grid.channels[swe.ch_hv].data;
    const auto& b_data  = swe.grid.channels[swe.ch_bathy].data;

    for (uint32_t j = 0; j < swe.grid.ny; ++j) {
      for (uint32_t i = 0; i < swe.grid.nx; ++i) {
        float wx, wy;
        swe.grid.GridToWorld(i, j, wx, wy);

        // Check if this cell center falls within the target region.
        if (wx < region.min.x || wx > region.max.x ||
            wy < region.min.y || wy > region.max.y) continue;

        size_t idx = swe.grid.Idx(i, j);
        float depth = h_data[idx];
        if (depth < swe.params.min_depth) continue;

        float bathy = b_data[idx];
        float inv_h = 1.0f / depth;
        float ux = hu_data[idx] * inv_h;
        float uy = hv_data[idx] * inv_h;

        // Spawn column of particles.
        for (float z = bathy + particle_spacing * 0.5f;
             z < bathy + depth;
             z += particle_spacing) {
          if (sph.particles.size() >= sph.params.max_particles)
            return;
          SPHParticle p;
          p.pos = {wx, wy, z};
          p.vel = {ux, uy, 0.0f};
          p.mass = mass;
          p.density = sph.params.rest_density;
          sph.particles.push_back(p);
        }
      }
    }
  }

  // --- SPH -> SWE: Collapse particles to depth-averaged fields ---
  //
  // Accumulate particle contributions into SWE cells. Each particle adds
  // mass (-> depth) and momentum to its nearest cell.
  static void SPHToSWE(const SPHSolver& sph, ShallowWaterSolver& swe,
                        const AABB& region) {
    float dx = swe.grid.dx;
    float cell_area = dx * dx;
    auto& h_data  = swe.grid.channels[swe.ch_h].data;
    auto& hu_data = swe.grid.channels[swe.ch_hu].data;
    auto& hv_data = swe.grid.channels[swe.ch_hv].data;

    // Zero the region first.
    for (uint32_t j = 0; j < swe.grid.ny; ++j) {
      for (uint32_t i = 0; i < swe.grid.nx; ++i) {
        float wx, wy;
        swe.grid.GridToWorld(i, j, wx, wy);
        if (wx < region.min.x || wx > region.max.x ||
            wy < region.min.y || wy > region.max.y) continue;
        size_t idx = swe.grid.Idx(i, j);
        h_data[idx] = 0;
        hu_data[idx] = 0;
        hv_data[idx] = 0;
      }
    }

    // Accumulate particles.
    float rho = sph.params.rest_density;
    for (const auto& p : sph.particles) {
      if (!region.Contains(p.pos)) continue;

      uint32_t gx, gy;
      if (!swe.grid.WorldToGridCell(p.pos.x, p.pos.y, gx, gy)) continue;

      size_t idx = swe.grid.Idx(gx, gy);
      float dh = p.mass / (rho * cell_area);  // mass -> depth contribution
      h_data[idx] += dh;
      hu_data[idx] += dh * p.vel.x;
      hv_data[idx] += dh * p.vel.y;
    }
  }

  // --- SPH -> LBM: Interpolate particle fields to lattice ---
  //
  // For each LBM cell, interpolate density and velocity from nearby SPH
  // particles, then initialize equilibrium distributions.
  static void SPHToLBM(const SPHSolver& sph, LBMSolver& lbm) {
    for (uint32_t z = 0; z < lbm.grid.nz; ++z) {
      for (uint32_t y = 0; y < lbm.grid.ny; ++y) {
        for (uint32_t x = 0; x < lbm.grid.nx; ++x) {
          size_t c = lbm.grid.Idx(x, y, z);
          if (lbm.grid.solid[c]) continue;

          // World position of this LBM cell.
          Vec3 pos = {
            lbm.grid.origin_x + (x + 0.5f) * lbm.grid.dx,
            lbm.grid.origin_y + (y + 0.5f) * lbm.grid.dx,
            lbm.grid.origin_z + (z + 0.5f) * lbm.grid.dx
          };

          // Interpolate from SPH particles.
          float rho_phys = sph.InterpolateDensity(pos);
          Vec3 vel_phys = sph.InterpolateVelocity(pos);

          // Convert to lattice units.
          float rho_lattice = (rho_phys > 0)
            ? rho_phys / lbm.params.DensityScale()
            : 1.0f;
          float vel_scale = 1.0f / lbm.params.VelocityScale();
          Vec3 u_lattice = vel_phys * vel_scale;

          // Initialize equilibrium.
          float feq[kQ];
          LBMSolver::Equilibrium(rho_lattice, u_lattice, feq);
          float* fi = lbm.grid.f_src.data() + c * kQ;
          for (int i = 0; i < kQ; ++i) fi[i] = feq[i];
        }
      }
    }
  }

  // --- LBM -> SPH: Spawn particles from lattice cells ---
  //
  // For each fluid LBM cell, create an SPH particle carrying the local
  // density and velocity.
  static void LBMToSPH(const LBMSolver& lbm, SPHSolver& sph,
                         float particle_spacing) {
    float mass = sph.params.ParticleMass(particle_spacing);

    for (uint32_t z = 0; z < lbm.grid.nz; ++z) {
      for (uint32_t y = 0; y < lbm.grid.ny; ++y) {
        for (uint32_t x = 0; x < lbm.grid.nx; ++x) {
          size_t c = lbm.grid.Idx(x, y, z);
          if (lbm.grid.solid[c]) continue;

          float rho_lattice = lbm.grid.Density(c);
          if (rho_lattice < 0.01f) continue;  // skip near-empty cells

          if (sph.particles.size() >= sph.params.max_particles)
            return;

          Vec3 pos = {
            lbm.grid.origin_x + (x + 0.5f) * lbm.grid.dx,
            lbm.grid.origin_y + (y + 0.5f) * lbm.grid.dx,
            lbm.grid.origin_z + (z + 0.5f) * lbm.grid.dx
          };

          Vec3 vel_lattice = lbm.grid.Velocity(c);
          float vel_scale = lbm.params.VelocityScale();

          SPHParticle p;
          p.pos = pos;
          p.vel = vel_lattice * vel_scale;
          p.mass = mass;
          p.density = rho_lattice * lbm.params.DensityScale();
          sph.particles.push_back(p);
        }
      }
    }
  }

  // --- Boundary coupling: SWE provides boundary conditions to SPH ---
  //
  // SPH particles near the edge of the SPH zone get velocity/pressure
  // nudged toward the SWE solution (ghost zone).
  static void CoupleSWEToSPH(const ShallowWaterSolver& swe, SPHSolver& sph,
                               const AABB& sph_zone, float ghost_width) {
    for (auto& p : sph.particles) {
      // Distance from SPH zone boundary (negative = inside).
      float dx_lo = p.pos.x - sph_zone.min.x;
      float dx_hi = sph_zone.max.x - p.pos.x;
      float dy_lo = p.pos.y - sph_zone.min.y;
      float dy_hi = sph_zone.max.y - p.pos.y;
      float dist_to_edge = std::min({dx_lo, dx_hi, dy_lo, dy_hi});

      if (dist_to_edge > ghost_width) continue;  // not in ghost zone

      // Blend factor: 1 at edge, 0 at ghost_width inside.
      float blend = 1.0f - std::clamp(dist_to_edge / ghost_width, 0.0f, 1.0f);

      // Get SWE velocity at this position.
      Vec3 swe_vel = swe.Velocity(p.pos.x, p.pos.y);

      // Nudge particle velocity toward SWE solution.
      p.vel.x += blend * (swe_vel.x - p.vel.x);
      p.vel.y += blend * (swe_vel.y - p.vel.y);
    }
  }

  // --- Boundary coupling: SPH provides boundary conditions to LBM ---
  //
  // LBM cells near the edge of the LBM zone get distributions nudged
  // toward f_eq + f_neq matching the SPH density, velocity, and stress.
  // Second-order lifting (Kruger et al. 2017): the non-equilibrium part
  // encodes the viscous stress tensor from SPH, preserving stress
  // continuity across the interface.
  static void CoupleSPHToLBM(const SPHSolver& sph, LBMSolver& lbm,
                               int ghost_cells) {
    float vel_scale = 1.0f / lbm.params.VelocityScale();
    float rho_scale = 1.0f / lbm.params.DensityScale();
    // Stress scale: physical stress -> lattice stress.
    // sigma_lattice = sigma_phys * dt^2 / (rho_phys * dx^2)
    float stress_scale = lbm.params.dt_phys * lbm.params.dt_phys /
                         (lbm.params.DensityScale() * lbm.params.dx_phys * lbm.params.dx_phys);

    for (uint32_t z = 0; z < lbm.grid.nz; ++z) {
      for (uint32_t y = 0; y < lbm.grid.ny; ++y) {
        for (uint32_t x = 0; x < lbm.grid.nx; ++x) {
          int dist = std::min({
            static_cast<int>(x), static_cast<int>(lbm.grid.nx - 1 - x),
            static_cast<int>(y), static_cast<int>(lbm.grid.ny - 1 - y),
            static_cast<int>(z), static_cast<int>(lbm.grid.nz - 1 - z)
          });
          if (dist >= ghost_cells) continue;

          size_t c = lbm.grid.Idx(x, y, z);
          if (lbm.grid.solid[c]) continue;

          Vec3 pos = {
            lbm.grid.origin_x + (x + 0.5f) * lbm.grid.dx,
            lbm.grid.origin_y + (y + 0.5f) * lbm.grid.dx,
            lbm.grid.origin_z + (z + 0.5f) * lbm.grid.dx
          };

          float rho_phys = sph.InterpolateDensity(pos);
          Vec3 vel_phys = sph.InterpolateVelocity(pos);

          float rho_lattice = (rho_phys > 0) ? rho_phys * rho_scale : 1.0f;
          Vec3 u_lattice = vel_phys * vel_scale;

          float blend = GhostBlend(dist, ghost_cells);

          // Equilibrium component.
          float feq[kQ];
          LBMSolver::Equilibrium(rho_lattice, u_lattice, feq);

          // Non-equilibrium component from SPH stress tensor.
          // f_neq_i = -w_i / (2 * cs^4) * Q_iab * Pi_neq_ab
          // where Q_iab = e_ia * e_ib - cs^2 * delta_ab
          SPHSolver::StressTensor stress;
          bool has_stress = sph.InterpolateStressTensor(pos, vel_phys, stress);

          float* fi = lbm.grid.f_src.data() + c * kQ;
          float inv_2cs4 = 1.0f / (2.0f * kCsSq * kCsSq);

          for (int i = 0; i < kQ; ++i) {
            float f_target = feq[i];

            if (has_stress) {
              // Convert physical stress to lattice units.
              float sxx = stress.xx * stress_scale;
              float syy = stress.yy * stress_scale;
              float szz = stress.zz * stress_scale;
              float sxy = stress.xy * stress_scale;
              float sxz = stress.xz * stress_scale;
              float syz = stress.yz * stress_scale;

              // Q tensor components for direction i.
              float ex = static_cast<float>(kEx[i]);
              float ey = static_cast<float>(kEy[i]);
              float ez = static_cast<float>(kEz[i]);
              float qxx = ex * ex - kCsSq;
              float qyy = ey * ey - kCsSq;
              float qzz = ez * ez - kCsSq;
              float qxy = ex * ey;
              float qxz = ex * ez;
              float qyz = ey * ez;

              float f_neq = -kW[i] * inv_2cs4 *
                (qxx * sxx + qyy * syy + qzz * szz +
                 2.0f * (qxy * sxy + qxz * sxz + qyz * syz));

              f_target += f_neq;
            }

            // Clamp to non-negative to prevent instability.
            f_target = std::max(f_target, 0.0f);

            fi[i] += blend * (f_target - fi[i]);
          }
        }
      }
    }
  }

  // --- Boundary coupling: Ocean provides BCs to SWE edge cells ---
  //
  // SWE cells at the domain boundary get their depth and velocity nudged
  // toward the ocean solution. This feeds swell energy into the SWE domain.
  static void CoupleOceanToSWE(const SpectralOceanSolver& ocean,
                                 ShallowWaterSolver& swe,
                                 int ghost_cells, float mean_depth) {
    auto& h  = swe.grid.channels[swe.ch_h].data;
    auto& hu = swe.grid.channels[swe.ch_hu].data;
    auto& hv = swe.grid.channels[swe.ch_hv].data;
    uint32_t nx = swe.grid.nx, ny = swe.grid.ny;

    for (uint32_t j = 0; j < ny; ++j) {
      for (uint32_t i = 0; i < nx; ++i) {
        // Distance from nearest domain edge (in cells).
        int dist = std::min({
          static_cast<int>(i), static_cast<int>(nx - 1 - i),
          static_cast<int>(j), static_cast<int>(ny - 1 - j)
        });
        if (dist >= ghost_cells) continue;

        float wx, wy;
        swe.grid.GridToWorld(i, j, wx, wy);

        // Ocean surface perturbation at this position.
        float eta = ocean.SurfaceHeight(wx, wy);
        Vec3 vel = ocean.OrbitalVelocity(wx, wy, 0.0f);

        // Target SWE state from ocean.
        float target_h = mean_depth + eta;
        float target_hu = target_h * vel.x;
        float target_hv = target_h * vel.y;

        // Blend: strongest at edge, weakest at ghost_cells inside.
        float blend = GhostBlend(dist, ghost_cells);
        blend *= blend;  // quadratic falloff for smoother transition

        size_t idx = swe.grid.Idx(i, j);
        h[idx]  += blend * (target_h  - h[idx]);
        hu[idx] += blend * (target_hu - hu[idx]);
        hv[idx] += blend * (target_hv - hv[idx]);
      }
    }
  }

  // --- LBM -> Stokes: Initialize Stokes grid from LBM fields ---
  //
  // Copies velocity and pressure from LBM to a Stokes grid that covers
  // a subregion of the LBM domain at finer resolution.
  static void LBMToStokes(const LBMSolver& lbm, StokesSolver& stokes) {
    for (uint32_t z = 0; z < stokes.grid.nz; ++z) {
      for (uint32_t y = 0; y < stokes.grid.ny; ++y) {
        for (uint32_t x = 0; x < stokes.grid.nx; ++x) {
          size_t sc = stokes.grid.Idx(x, y, z);
          if (stokes.grid.solid[sc]) continue;

          // World position of this Stokes cell.
          float wx, wy, wz;
          stokes.grid.GridToWorld(x, y, z, wx, wy, wz);

          // Find nearest LBM cell.
          float inv_dx = 1.0f / lbm.grid.dx;
          int lx = static_cast<int>((wx - lbm.grid.origin_x) * inv_dx);
          int ly = static_cast<int>((wy - lbm.grid.origin_y) * inv_dx);
          int lz = static_cast<int>((wz - lbm.grid.origin_z) * inv_dx);

          if (lx < 0 || lx >= static_cast<int>(lbm.grid.nx) ||
              ly < 0 || ly >= static_cast<int>(lbm.grid.ny) ||
              lz < 0 || lz >= static_cast<int>(lbm.grid.nz)) continue;

          size_t lc = lbm.grid.Idx(static_cast<uint32_t>(lx),
                                     static_cast<uint32_t>(ly),
                                     static_cast<uint32_t>(lz));
          if (lbm.grid.solid[lc]) continue;

          Vec3 vel = lbm.grid.Velocity(lc);
          float vel_scale = lbm.params.VelocityScale();
          float rho_lattice = lbm.grid.Density(lc);

          stokes.grid.u[sc] = vel.x * vel_scale;
          stokes.grid.v[sc] = vel.y * vel_scale;
          stokes.grid.w[sc] = vel.z * vel_scale;
          // Transfer pressure: p = (rho - rho0) * cs^2 in physical units.
          stokes.grid.p[sc] = (rho_lattice - 1.0f) * kCsSq
                              * lbm.params.DensityScale()
                              * vel_scale * vel_scale;
        }
      }
    }
  }

  // --- Stokes -> LBM: Non-equilibrium extrapolation ---
  //
  // For LBM cells that overlap with the Stokes domain, set distributions
  // using the Stokes velocity/pressure for equilibrium, but preserve the
  // non-equilibrium part from the nearest interior LBM cell. This maintains
  // viscous stress continuity across the interface.
  //
  // f_boundary = feq(rho_stokes, u_stokes) + fneq_interior
  // where fneq = f - feq at the interior cell.
  static void StokesToLBM(const StokesSolver& stokes, LBMSolver& lbm) {
    float vel_scale_inv = 1.0f / lbm.params.VelocityScale();

    for (uint32_t z = 0; z < lbm.grid.nz; ++z) {
      for (uint32_t y = 0; y < lbm.grid.ny; ++y) {
        for (uint32_t x = 0; x < lbm.grid.nx; ++x) {
          size_t lc = lbm.grid.Idx(x, y, z);
          if (lbm.grid.solid[lc]) continue;

          float wx, wy, wz;
          lbm.grid.GridToWorld(x, y, z, wx, wy, wz);

          uint32_t sx, sy, sz;
          if (!stokes.grid.WorldToGridCell(wx, wy, wz, sx, sy, sz)) continue;

          size_t sc = stokes.grid.Idx(sx, sy, sz);
          if (stokes.grid.solid[sc]) continue;

          // Target state from Stokes.
          Vec3 vel_phys = {stokes.grid.u[sc], stokes.grid.v[sc],
                           stokes.grid.w[sc]};
          Vec3 u_target = vel_phys * vel_scale_inv;

          // Use Stokes pressure for target density (p = rho * cs^2).
          float p_stokes = stokes.grid.p[sc];
          float p_scale = lbm.params.DensityScale() *
                          lbm.params.VelocityScale() * lbm.params.VelocityScale();
          float rho_target = 1.0f + p_stokes / (kCsSq * p_scale);
          rho_target = std::max(rho_target, 0.5f);

          // Compute target equilibrium.
          float feq_target[kQ];
          LBMSolver::Equilibrium(rho_target, u_target, feq_target);

          // Extract non-equilibrium part from current LBM cell (interior
          // extrapolation: use this cell's own fneq as best estimate).
          float* fi = lbm.grid.f_src.data() + lc * kQ;
          float rho_current = lbm.grid.Density(lc);
          Vec3 u_current = lbm.grid.Velocity(lc);
          float feq_current[kQ];
          LBMSolver::Equilibrium(rho_current, u_current, feq_current);

          constexpr float blend = 0.8f;
          for (int i = 0; i < kQ; ++i) {
            float fneq = fi[i] - feq_current[i];
            float f_target = std::max(feq_target[i] + fneq, 0.0f);
            fi[i] += blend * (f_target - fi[i]);
          }
        }
      }
    }
  }

  // --- Boundary coupling: LBM provides BCs to Stokes ghost cells ---
  //
  // Stokes cells near the boundary get velocity nudged toward LBM values.
  static void CoupleLBMToStokes(const LBMSolver& lbm, StokesSolver& stokes,
                                  int ghost_cells) {
    uint32_t nx = stokes.grid.nx, ny = stokes.grid.ny, nz = stokes.grid.nz;
    for (uint32_t z = 0; z < nz; ++z) {
      for (uint32_t y = 0; y < ny; ++y) {
        for (uint32_t x = 0; x < nx; ++x) {
          int dist = std::min({
            static_cast<int>(x), static_cast<int>(nx - 1 - x),
            static_cast<int>(y), static_cast<int>(ny - 1 - y),
            static_cast<int>(z), static_cast<int>(nz - 1 - z)
          });
          if (dist >= ghost_cells) continue;

          size_t sc = stokes.grid.Idx(x, y, z);
          if (stokes.grid.solid[sc]) continue;

          float wx, wy, wz;
          stokes.grid.GridToWorld(x, y, z, wx, wy, wz);

          // Look up LBM velocity at this position.
          float inv_dx = 1.0f / lbm.grid.dx;
          int lx = static_cast<int>((wx - lbm.grid.origin_x) * inv_dx);
          int ly = static_cast<int>((wy - lbm.grid.origin_y) * inv_dx);
          int lz = static_cast<int>((wz - lbm.grid.origin_z) * inv_dx);

          if (lx < 0 || lx >= static_cast<int>(lbm.grid.nx) ||
              ly < 0 || ly >= static_cast<int>(lbm.grid.ny) ||
              lz < 0 || lz >= static_cast<int>(lbm.grid.nz)) continue;

          size_t lc = lbm.grid.Idx(static_cast<uint32_t>(lx),
                                     static_cast<uint32_t>(ly),
                                     static_cast<uint32_t>(lz));
          if (lbm.grid.solid[lc]) continue;

          Vec3 vel = lbm.grid.Velocity(lc);
          float vel_scale = lbm.params.VelocityScale();

          float blend = GhostBlend(dist, ghost_cells);

          stokes.grid.u[sc] += blend * (vel.x * vel_scale - stokes.grid.u[sc]);
          stokes.grid.v[sc] += blend * (vel.y * vel_scale - stokes.grid.v[sc]);
          stokes.grid.w[sc] += blend * (vel.z * vel_scale - stokes.grid.w[sc]);
        }
      }
    }
  }

  // --- Mass conservation correction ---
  //
  // After a transition, adjust the target representation so total mass
  // matches the source. Returns the correction factor applied.
  static float CorrectMass_SPH(SPHSolver& sph, float target_mass) {
    float current = sph.TotalMass();
    if (current < 1e-10f) return 1.0f;
    float factor = target_mass / current;
    for (auto& p : sph.particles) p.mass *= factor;
    return factor;
  }

  // --- Momentum conservation correction for SPH ---
  //
  // After a transition, adjust particle velocities so total momentum
  // matches the target. Uniform velocity offset preserves relative motion.
  static Vec3 CorrectMomentum_SPH(SPHSolver& sph, Vec3 target_momentum) {
    Vec3 current{};
    float total_mass = 0;
    for (const auto& p : sph.particles) {
      current += p.vel * p.mass;
      total_mass += p.mass;
    }
    if (total_mass < 1e-10f) return {};
    // Apply uniform velocity offset to match target momentum.
    Vec3 correction = (target_momentum - current) / total_mass;
    for (auto& p : sph.particles) {
      p.vel += correction;
    }
    return correction;
  }

  static float CorrectMass_SWE(ShallowWaterSolver& swe, const AABB& region,
                                 float target_volume) {
    float current = 0;
    auto& h = swe.grid.channels[swe.ch_h].data;
    float cell_area = swe.grid.dx * swe.grid.dx;

    std::vector<size_t> cells_in_region;
    for (uint32_t j = 0; j < swe.grid.ny; ++j) {
      for (uint32_t i = 0; i < swe.grid.nx; ++i) {
        float wx, wy;
        swe.grid.GridToWorld(i, j, wx, wy);
        if (wx < region.min.x || wx > region.max.x ||
            wy < region.min.y || wy > region.max.y) continue;
        size_t idx = swe.grid.Idx(i, j);
        current += h[idx] * cell_area;
        cells_in_region.push_back(idx);
      }
    }

    if (current < 1e-10f || cells_in_region.empty()) return 1.0f;
    float factor = target_volume / current;
    for (size_t idx : cells_in_region) h[idx] *= factor;
    return factor;
  }

  // --- Berger-Colella flux correction at SWE-SPH interface ---
  //
  // After all SPH substeps complete, compare:
  //   - SWE interface flux: what the SWE grid thinks crossed the SPH boundary
  //   - SPH particle flux: what particles actually crossed the boundary
  //
  // The difference is the conservation error. Distribute it as a uniform
  // correction to SPH particles near the boundary and to the SWE cells
  // at the interface.
  //
  // swe_flux: from ShallowWaterSolver::ComputeInterfaceFluxes(sph_zone)
  // sph_tracker: accumulated particle crossings from SPHFluxTracker
  // master_dt: the total time over which fluxes were accumulated

  static void CorrectSWESPHInterface(
      ShallowWaterSolver& swe,
      SPHSolver& sph,
      const SWEInterfaceFlux& swe_flux,
      const SPHFluxTracker& sph_tracker,
      const AABB& sph_zone,
      float master_dt) {
    if (master_dt < 1e-12f) return;

    // Mass flux mismatch: what SWE transported in minus what SPH received.
    // swe_flux.net_mass is in [m^2/s] (depth flux), scale to mass:
    float cell_area = swe.grid.dx * swe.grid.dx;
    float swe_mass_flux = swe_flux.net_mass * master_dt * cell_area * kWaterDensity;
    float sph_mass_flux = sph_tracker.accumulated_mass;

    float mass_err = swe_mass_flux - sph_mass_flux;

    // Skip correction if error is negligible.
    if (std::abs(mass_err) < 1e-8f) return;

    // Distribute correction to SPH particles near the boundary.
    // Apply a uniform mass adjustment to particles within one smoothing
    // length of the boundary (these are the particles most affected by
    // the ghost zone blending).
    float h_sph = sph.params.smoothing_length;
    AABB inner = {
      {sph_zone.min.x + h_sph, sph_zone.min.y + h_sph, sph_zone.min.z},
      {sph_zone.max.x - h_sph, sph_zone.max.y - h_sph, sph_zone.max.z}
    };

    uint32_t n_boundary_particles = 0;
    float total_boundary_mass = 0;
    for (uint32_t i = 0; i < sph.Count(); ++i) {
      Vec3 p = sph.particles[i].pos;
      bool in_zone = SPHFluxTracker::IsInside(p, sph_zone);
      bool in_inner = SPHFluxTracker::IsInside(p, inner);
      if (in_zone && !in_inner) {
        ++n_boundary_particles;
        total_boundary_mass += sph.particles[i].mass;
      }
    }

    if (n_boundary_particles == 0 || total_boundary_mass < 1e-10f) return;

    // Scale factor: adjust boundary particle masses to absorb the error.
    float correction_factor = 1.0f + mass_err / total_boundary_mass;
    // Clamp to prevent extreme corrections (safety).
    correction_factor = std::clamp(correction_factor, 0.95f, 1.05f);

    for (uint32_t i = 0; i < sph.Count(); ++i) {
      Vec3 p = sph.particles[i].pos;
      bool in_zone = SPHFluxTracker::IsInside(p, sph_zone);
      bool in_inner = SPHFluxTracker::IsInside(p, inner);
      if (in_zone && !in_inner) {
        sph.particles[i].mass *= correction_factor;
      }
    }

    // Momentum correction: apply velocity offset to boundary particles.
    Vec3 swe_mom_flux = {
      swe_flux.net_momentum_x * master_dt * cell_area * kWaterDensity,
      swe_flux.net_momentum_y * master_dt * cell_area * kWaterDensity,
      0
    };
    Vec3 mom_err = swe_mom_flux - sph_tracker.accumulated_momentum;
    if (mom_err.Length() > 1e-8f && total_boundary_mass > 1e-10f) {
      Vec3 vel_offset = mom_err / total_boundary_mass;
      // Clamp velocity correction to 10% of max particle speed.
      float max_correction = 0.1f * sph.ComputeMaxSpeed();
      float correction_mag = vel_offset.Length();
      if (correction_mag > max_correction && correction_mag > 0) {
        vel_offset = vel_offset * (max_correction / correction_mag);
      }
      for (uint32_t i = 0; i < sph.Count(); ++i) {
        Vec3 p = sph.particles[i].pos;
        bool in_zone = SPHFluxTracker::IsInside(p, sph_zone);
        bool in_inner = SPHFluxTracker::IsInside(p, inner);
        if (in_zone && !in_inner) {
          sph.particles[i].vel += vel_offset;
        }
      }
    }
  }
};

}  // namespace mjwater
