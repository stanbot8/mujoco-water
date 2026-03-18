// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Conservative flux tracking for multi-scale solver coupling.
//
// Flux registers accumulate fine-grid fluxes at coarse-fine interfaces
// during subcycled integration. After all fine substeps complete, the
// coarse cell adjacent to the interface is corrected by the difference
// between the coarse flux it used and the area-weighted sum of fine
// fluxes that actually crossed the same interface.
//
// This ensures mass and momentum conservation to machine precision
// across LOD boundaries during stepping, not just on activation.
//
// Based on the Berger-Colella (1989) flux correction methodology:
//   Berger, M.J. & Colella, P. (1989) "Local Adaptive Mesh Refinement
//     for Shock Hydrodynamics" J. Comput. Phys. 82, 64-84
//
// Data structures:
//   FluxRegister2D: for 2D SWE grid interfaces (Ocean-SWE, SWE-SPH)
//   FluxRegister3D: for 3D lattice interfaces (LBM-Stokes)
//   SPHFluxTracker: for Lagrangian particle flux at SPH zone boundary
//   SPHStressBuffer: non-equilibrium stress tensor for LBM lifting

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mjwater/types.h"

namespace mjwater {

// --- 2D flux register for SWE grid interfaces ---

struct FluxRegister2D {
  struct EdgeFlux {
    float mass = 0;        // accumulated h-flux * dt
    float momentum_x = 0;  // accumulated hu-flux * dt
    float momentum_y = 0;  // accumulated hv-flux * dt
  };

  // Per-edge accumulated fine fluxes and coarse fluxes used.
  std::vector<EdgeFlux> fine_accumulated;
  std::vector<EdgeFlux> coarse_used;
  uint32_t n_edges = 0;

  void Init(uint32_t count) {
    n_edges = count;
    fine_accumulated.assign(count, EdgeFlux{});
    coarse_used.assign(count, EdgeFlux{});
  }

  void Reset() {
    std::fill(fine_accumulated.begin(), fine_accumulated.end(), EdgeFlux{});
    std::fill(coarse_used.begin(), coarse_used.end(), EdgeFlux{});
  }

  // Called after each fine substep: accumulate the fine-grid flux
  // at this interface cell, weighted by the fine dt.
  void AccumulateFine(uint32_t edge_idx, float fh, float fhu, float fhv, float dt) {
    if (edge_idx >= n_edges) return;
    fine_accumulated[edge_idx].mass += fh * dt;
    fine_accumulated[edge_idx].momentum_x += fhu * dt;
    fine_accumulated[edge_idx].momentum_y += fhv * dt;
  }

  // Called once after the coarse step: record the coarse flux used.
  void SetCoarse(uint32_t edge_idx, float fh, float fhu, float fhv, float dt) {
    if (edge_idx >= n_edges) return;
    coarse_used[edge_idx].mass = fh * dt;
    coarse_used[edge_idx].momentum_x = fhu * dt;
    coarse_used[edge_idx].momentum_y = fhv * dt;
  }

  // Compute correction = (fine_accumulated - coarse_used) for one edge cell.
  // Positive means fine level transported more mass/momentum than coarse
  // accounted for, so coarse needs to lose that amount.
  EdgeFlux Correction(uint32_t edge_idx) const {
    if (edge_idx >= n_edges) return {};
    return {
      fine_accumulated[edge_idx].mass - coarse_used[edge_idx].mass,
      fine_accumulated[edge_idx].momentum_x - coarse_used[edge_idx].momentum_x,
      fine_accumulated[edge_idx].momentum_y - coarse_used[edge_idx].momentum_y
    };
  }

  // Total correction summed over all edges.
  EdgeFlux TotalCorrection() const {
    EdgeFlux total;
    for (uint32_t i = 0; i < n_edges; ++i) {
      auto c = Correction(i);
      total.mass += c.mass;
      total.momentum_x += c.momentum_x;
      total.momentum_y += c.momentum_y;
    }
    return total;
  }
};

// --- SPH flux tracker for Lagrangian particle boundary crossings ---
//
// Tracks particles crossing the SPH zone boundary during substeps.
// Before each substep, snapshots positions of particles near the boundary.
// After each substep, detects which particles crossed and accumulates
// the mass/momentum flux.

struct SPHFluxTracker {
  struct ParticleSnapshot {
    uint32_t index;
    Vec3 pos;
    Vec3 vel;
    float mass;
    bool was_inside;  // was inside SPH zone before step
  };

  std::vector<ParticleSnapshot> snapshots;
  float accumulated_mass = 0;
  Vec3 accumulated_momentum;
  float dead_zone = 0;  // half-smoothing-length buffer to avoid oscillation noise

  void Reset() {
    snapshots.clear();
    accumulated_mass = 0;
    accumulated_momentum = {};
  }

  void SetDeadZone(float smoothing_length) {
    dead_zone = smoothing_length * 0.5f;
  }

  // Check if a position is inside the SPH zone (with optional dead zone shrink).
  static bool IsInside(Vec3 pos, const AABB& zone, float margin = 0) {
    return pos.x >= zone.min.x + margin && pos.x <= zone.max.x - margin &&
           pos.y >= zone.min.y + margin && pos.y <= zone.max.y - margin &&
           pos.z >= zone.min.z + margin && pos.z <= zone.max.z - margin;
  }

  // Snapshot boundary particles before an SPH substep.
  // Only tracks particles within dead_zone of the boundary.
  template <typename ParticleArray>
  void BeforeStep(const ParticleArray& particles, uint32_t count,
                  const AABB& zone) {
    snapshots.clear();
    float margin = dead_zone;
    // Expanded zone to capture particles near boundary.
    AABB expanded = {
      {zone.min.x - margin, zone.min.y - margin, zone.min.z - margin},
      {zone.max.x + margin, zone.max.y + margin, zone.max.z + margin}
    };

    for (uint32_t i = 0; i < count; ++i) {
      Vec3 p = particles[i].pos;
      if (IsInside(p, expanded) && !IsInside(p, zone, margin)) {
        // Particle is near the boundary.
        snapshots.push_back({
          i, p, particles[i].vel, particles[i].mass,
          IsInside(p, zone)
        });
      }
    }
  }

  // After an SPH substep, detect crossings and accumulate flux.
  template <typename ParticleArray>
  void AfterStep(const ParticleArray& particles, uint32_t count,
                 const AABB& zone, float dt) {
    (void)count; (void)dt;
    for (auto& snap : snapshots) {
      if (snap.index >= count) continue;
      bool now_inside = IsInside(particles[snap.index].pos, zone);
      if (snap.was_inside && !now_inside) {
        // Particle left the SPH zone: outgoing flux.
        accumulated_mass -= snap.mass;
        accumulated_momentum -= particles[snap.index].vel * snap.mass;
      } else if (!snap.was_inside && now_inside) {
        // Particle entered the SPH zone: incoming flux.
        accumulated_mass += snap.mass;
        accumulated_momentum += particles[snap.index].vel * snap.mass;
      }
    }
  }
};

// --- 3D flux register for LBM-Stokes interface ---

struct FluxRegister3D {
  struct FaceFlux {
    float mass = 0;
    Vec3 momentum;
  };

  std::vector<FaceFlux> fine_accumulated;
  std::vector<FaceFlux> coarse_used;
  uint32_t ny_face = 0, nz_face = 0;

  void Init(uint32_t ny, uint32_t nz) {
    ny_face = ny;
    nz_face = nz;
    uint32_t count = ny * nz * 6;  // 6 faces
    fine_accumulated.assign(count, FaceFlux{});
    coarse_used.assign(count, FaceFlux{});
  }

  void Reset() {
    std::fill(fine_accumulated.begin(), fine_accumulated.end(), FaceFlux{});
    std::fill(coarse_used.begin(), coarse_used.end(), FaceFlux{});
  }

  uint32_t FaceIdx(uint32_t face, uint32_t fy, uint32_t fz) const {
    return face * ny_face * nz_face + fy * nz_face + fz;
  }

  void AccumulateFine(uint32_t face, uint32_t fy, uint32_t fz,
                      float mass_flux, Vec3 mom_flux, float dt) {
    uint32_t idx = FaceIdx(face, fy, fz);
    if (idx >= fine_accumulated.size()) return;
    fine_accumulated[idx].mass += mass_flux * dt;
    fine_accumulated[idx].momentum += mom_flux * dt;
  }

  void SetCoarse(uint32_t face, uint32_t fy, uint32_t fz,
                 float mass_flux, Vec3 mom_flux, float dt) {
    uint32_t idx = FaceIdx(face, fy, fz);
    if (idx >= coarse_used.size()) return;
    coarse_used[idx].mass = mass_flux * dt;
    coarse_used[idx].momentum = mom_flux * dt;
  }

  FaceFlux Correction(uint32_t face, uint32_t fy, uint32_t fz) const {
    uint32_t idx = FaceIdx(face, fy, fz);
    if (idx >= fine_accumulated.size()) return {};
    return {
      fine_accumulated[idx].mass - coarse_used[idx].mass,
      fine_accumulated[idx].momentum - coarse_used[idx].momentum
    };
  }
};

// --- Non-equilibrium stress buffer for SPH-LBM lifting ---
//
// Stores the viscous stress tensor computed from SPH at LBM boundary
// cells. The stress tensor enables second-order accurate distribution
// function reconstruction: f_i = feq(rho, u) + fneq(stress).
//
// Without this, coupling only transfers density and velocity (first-order),
// losing the viscous stress information and creating artificial boundary
// layers at the SPH-LBM interface.

struct SPHStressBuffer {
  // Symmetric 3x3 stress tensor (6 independent components).
  struct Stress {
    float xx = 0, yy = 0, zz = 0;
    float xy = 0, xz = 0, yz = 0;
  };

  std::vector<Stress> data;
  uint32_t count = 0;

  void Init(uint32_t n_boundary_cells) {
    count = n_boundary_cells;
    data.assign(count, Stress{});
  }

  void Reset() {
    std::fill(data.begin(), data.end(), Stress{});
  }
};

// --- Conservation diagnostics ---
//
// Tracks mass and momentum across all active solvers and interfaces.
// The total_mass calculation avoids double-counting in overlap zones:
// each spatial point belongs to the finest active solver covering it.

struct ConservationDiag {
  float swe_mass = 0;           // SWE total volume * rho
  float sph_mass = 0;           // SPH total particle mass
  float lbm_mass = 0;           // LBM total mass
  float stokes_mass = 0;        // Stokes mass (should be constant from div-free)
  float ocean_boundary_flux = 0; // net mass entering through ocean boundary
  float swe_sph_flux_err = 0;   // flux mismatch at SWE-SPH interface
  float sph_lbm_flux_err = 0;   // flux mismatch at SPH-LBM interface
  float lbm_stokes_flux_err = 0; // flux mismatch at LBM-Stokes interface
  float total_mass = 0;          // sum of non-overlapping regions
};

}  // namespace mjwater
