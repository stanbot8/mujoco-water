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

namespace mjwater {

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
  // toward equilibrium matching the SPH velocity/density field.
  static void CoupleSPHToLBM(const SPHSolver& sph, LBMSolver& lbm,
                               int ghost_cells) {
    for (uint32_t z = 0; z < lbm.grid.nz; ++z) {
      for (uint32_t y = 0; y < lbm.grid.ny; ++y) {
        for (uint32_t x = 0; x < lbm.grid.nx; ++x) {
          // Check if this cell is within ghost_cells of any boundary.
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

          float rho_lattice = (rho_phys > 0)
            ? rho_phys / lbm.params.DensityScale()
            : 1.0f;
          float vel_scale = 1.0f / lbm.params.VelocityScale();
          Vec3 u_lattice = vel_phys * vel_scale;

          // Blend: stronger at boundary, weaker deeper inside.
          float blend = 1.0f - static_cast<float>(dist) / ghost_cells;

          float feq[kQ];
          LBMSolver::Equilibrium(rho_lattice, u_lattice, feq);
          float* fi = lbm.grid.f_src.data() + c * kQ;
          for (int i = 0; i < kQ; ++i) {
            fi[i] += blend * (feq[i] - fi[i]);
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
        float blend = 1.0f - static_cast<float>(dist) / ghost_cells;
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

  // --- Stokes -> LBM: Transfer Stokes velocity to LBM distributions ---
  //
  // For LBM cells that overlap with the Stokes domain, nudge distributions
  // toward equilibrium matching the Stokes velocity field.
  static void StokesToLBM(const StokesSolver& stokes, LBMSolver& lbm) {
    for (uint32_t z = 0; z < lbm.grid.nz; ++z) {
      for (uint32_t y = 0; y < lbm.grid.ny; ++y) {
        for (uint32_t x = 0; x < lbm.grid.nx; ++x) {
          size_t lc = lbm.grid.Idx(x, y, z);
          if (lbm.grid.solid[lc]) continue;

          float wx, wy, wz;
          lbm.grid.GridToWorld(x, y, z, wx, wy, wz);

          // Check if this LBM cell falls within the Stokes domain.
          uint32_t sx, sy, sz;
          if (!stokes.grid.WorldToGridCell(wx, wy, wz, sx, sy, sz)) continue;

          size_t sc = stokes.grid.Idx(sx, sy, sz);
          if (stokes.grid.solid[sc]) continue;

          // Get Stokes velocity.
          Vec3 vel_phys = {stokes.grid.u[sc], stokes.grid.v[sc],
                           stokes.grid.w[sc]};

          float rho_lattice = lbm.grid.Density(lc);
          float vel_scale = 1.0f / lbm.params.VelocityScale();
          Vec3 u_lattice = vel_phys * vel_scale;

          // Blend toward equilibrium matching Stokes velocity (avoids
          // velocity discontinuity that hard-set would produce).
          float feq[kQ];
          LBMSolver::Equilibrium(rho_lattice, u_lattice, feq);
          float* fi = lbm.grid.f_src.data() + lc * kQ;
          constexpr float blend = 0.8f;  // strong but not instant
          for (int i = 0; i < kQ; ++i) fi[i] += blend * (feq[i] - fi[i]);
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

          float blend = 1.0f - static_cast<float>(dist) / ghost_cells;

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
};

}  // namespace mjwater
