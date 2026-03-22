// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Top-level water simulation orchestrator.
//
// Manages a 5-level multi-resolution, multi-timestep simulation loop that
// spans roughly 10 orders of magnitude in spatial scale:
//
//   Level 0: Ocean (spectral)     km scale,  analytic (no timestep)
//   Level 1: SWE (shallow water)  m scale,   master_dt (~10 ms)
//   Level 2: SPH (particles)      cm scale,  sph_dt = master_dt / n_sph
//   Level 3: LBM (lattice)        mm scale,  lbm_dt = sph_dt / n_lbm
//   Level 4: Stokes (creeping)    um scale,  stokes_dt = lbm_dt / n_stokes
//
// Each finer level substeps within its parent. The total number of fine
// steps per master step can be large: n_sph * n_lbm * n_stokes. A runtime
// guardrail caps total fine steps at kMaxFineSteps to prevent runaway.
//
// LOD activation is driven by camera distance (auto-LOD mode) or manual
// checkboxes. When a solver activates, it is seeded from the next coarser
// solver via LODTransition coupling functions. When it deactivates, its
// state is merged back into the parent.
//
// Absorbing sponge boundaries at the SWE domain edges prevent wave
// reflections. When ocean is active, the sponge relaxes toward the
// analytic ocean state (radiation boundary). Without ocean, it damps
// momentum to zero (absorbing boundary).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mjwater/types.h"
#include "mjwater/water_grid.h"
#include "mjwater/water_sdf.h"
#include "mjwater/spectral_ocean.h"
#include "mjwater/shallow_water.h"
#include "mjwater/sph.h"
#include "mjwater/lbm.h"
#include "mjwater/stokes.h"
#include "mjwater/lod_manager.h"
#include "mjwater/lod_transition.h"
#include "mjwater/coupling.h"

namespace mjwater {

struct WaterEngineConfig {
  // --- Ocean (LOD 0): spectral, no grid ---
  OceanParams ocean;
  bool ocean_enabled = false;      // enable spectral ocean background
  int ocean_ghost_cells = 3;       // SWE boundary cells fed by ocean
  float ocean_mean_depth = 10.0f;  // mean water depth for ocean->SWE coupling

  // --- SWE (LOD 1): 2D height field ---
  uint32_t swe_nx = 100, swe_ny = 100;
  float swe_dx = 0.1f;             // 10 cm cells for SWE

  // --- SPH (LOD 2): 3D particles ---
  float sph_spacing = 0.02f;       // 2 cm particle spacing
  float sph_smoothing = 0.03f;     // 3 cm smoothing length

  // --- LBM (LOD 3): 3D lattice ---
  uint32_t lbm_nx = 50, lbm_ny = 50, lbm_nz = 50;
  float lbm_dx = 0.001f;           // 1 mm lattice spacing

  // --- Stokes (LOD 4): 3D grid, cellular scale ---
  uint32_t stokes_nx = 30, stokes_ny = 30, stokes_nz = 30;
  StokesParams stokes;
  int stokes_substeps = 10;        // Stokes substeps per LBM step
  int stokes_ghost_cells = 3;      // boundary coupling depth

  // --- Coupling ---
  int sph_to_lbm_ghost_cells = 5;   // ghost zone depth for SPH->LBM coupling
  float lbm_reposition_threshold = 0.25f;    // reposition LBM grid when focus drifts this fraction of extent
  float stokes_reposition_threshold = 0.25f; // same for Stokes grid

  // --- Timestep hierarchy ---
  float master_dt = 0.01f;         // 10 ms master step
  int sph_substeps = 10;           // SPH runs at ~1 ms
  int lbm_substeps = 100;          // LBM runs at ~0.01 ms per SPH step

  // --- LOD configuration ---
  LODConfig lod;

  // Initial water surface height.
  float initial_surface_z = 0.5f;

  // Domain bounds.
  AABB domain = {{0, 0, 0}, {10, 10, 2}};

  // --- Absorbing boundaries ---
  int sponge_width = 0;           // sponge zone width in cells (0 = disabled)
  float sponge_strength = 4.0f;   // max damping coefficient (1/s)

  // --- Background current ---
  Vec3 current_velocity;           // uniform background current (m/s), default zero
};

struct WaterEngine {
  SpectralOceanSolver ocean;
  ShallowWaterSolver swe;
  SPHSolver sph;
  LBMSolver lbm;
  StokesSolver stokes;
  LODManager lod_manager;
  WaterEngineConfig config;

  bool ocean_active = false;
  bool sph_active = false;
  bool lbm_active = false;
  bool stokes_active = false;

  float sim_time = 0;

  // Body coupling state.
  std::vector<BodyCoupling> bodies;

  // Per-body state for two-way coupling.
  struct BodyState {
    Vec3 pos;
    Vec3 vel;
    Vec3 wave_reaction;  // reaction force from wave generation (previous step)
  };
  std::vector<BodyState> body_states;

  // Sponge target buffers (filled from ocean each frame).
  std::vector<float> sponge_target_h;
  std::vector<float> sponge_target_hu;
  std::vector<float> sponge_target_hv;

  // Flux tracking for conservative coupling.
  SPHFluxTracker sph_flux_tracker;
  FluxRegister2D::EdgeFlux last_ocean_flux;  // mass/momentum injected by ocean coupling

  void Init(const WaterEngineConfig& cfg) {
    config = cfg;

    // Initialize ocean (spectral, always ready if enabled).
    if (cfg.ocean_enabled) {
      // Sync current velocity to ocean for wave-current Doppler interaction.
      OceanParams ocean_params = cfg.ocean;
      ocean_params.current_x = cfg.current_velocity.x;
      ocean_params.current_y = cfg.current_velocity.y;
      ocean.Init(ocean_params);
      ocean_active = true;
    }

    // Initialize SWE (always on).
    swe.Init(cfg.swe_nx, cfg.swe_ny, cfg.swe_dx,
             cfg.domain.min.x, cfg.domain.min.y);
    swe.SetSurface(cfg.initial_surface_z);

    // Apply background current as initial momentum.
    if (cfg.current_velocity.x != 0 || cfg.current_velocity.y != 0) {
      auto& h  = swe.grid.channels[swe.ch_h].data;
      auto& hu = swe.grid.channels[swe.ch_hu].data;
      auto& hv = swe.grid.channels[swe.ch_hv].data;
      for (size_t i = 0; i < h.size(); ++i) {
        hu[i] = h[i] * cfg.current_velocity.x;
        hv[i] = h[i] * cfg.current_velocity.y;
      }
    }

    // Initialize SPH (starts inactive, activated by LOD).
    SPHParams sph_params;
    sph_params.smoothing_length = cfg.sph_smoothing;
    sph.Init(sph_params, cfg.domain);

    // Initialize LBM (starts inactive, activated by LOD).
    LBMParams lbm_params;
    lbm_params.dx_phys = cfg.lbm_dx;
    lbm.Init(cfg.lbm_nx, cfg.lbm_ny, cfg.lbm_nz, lbm_params);

    // Initialize Stokes (starts inactive, activated by LOD).
    stokes.Init(cfg.stokes_nx, cfg.stokes_ny, cfg.stokes_nz, cfg.stokes);

    // Initialize LOD manager.
    lod_manager.Init(cfg.lod);
  }

  void Init(const WaterEngineConfig& cfg, const WaterSDF& sdf) {
    Init(cfg);
    // Bake SDF to height field bathymetry.
    sdf.BakeToHeightField(swe.grid, swe.ch_bathy);
    swe.SetSurface(cfg.initial_surface_z);
  }

  // Set focus point (typically camera or agent position).
  // Repositions fine grids when focus moves beyond their coverage.
  void SetFocus(Vec3 pos) {
    lod_manager.SetFocus(pos);
    RepositionGrids(pos);
  }

  // Register a body for fluid coupling. Returns body index.
  size_t AddBody(const BodyCoupling& body) {
    bodies.push_back(body);
    body_states.push_back({});
    return bodies.size() - 1;
  }

  // Main simulation step.
  //
  // Multi-timestep hierarchy (coarsest to finest):
  //   Ocean (analytic, no substep) -> SWE (master dt) -> SPH (substeps)
  //     -> LBM (sub-substeps) -> Stokes (sub-sub-substeps)
  void Step() {
    float master_dt = config.master_dt;

    // 1. Update LOD zones.
    UpdateLOD();

    // 2. Ocean step (advance time for spectral waves).
    if (ocean_active) {
      ocean.Step(master_dt);
      // Feed ocean boundary conditions into SWE edge cells.
      // Conservative wrapper tracks mass/momentum injected.
      last_ocean_flux = LODTransition::CoupleOceanToSWE_Conservative(
        ocean, swe, config.ocean_ghost_cells, config.ocean_mean_depth);
    }

    // 3. Two-way body coupling: inject wave sources into SWE before stepping.
    ApplyTwoWayCoupling(master_dt);

    // 4. SWE step (full master timestep).
    swe.Step(master_dt);

    // 5. Absorbing sponge boundaries.
    if (config.sponge_width > 0) {
      bool has_target = ocean_active ||
                        config.current_velocity.x != 0 ||
                        config.current_velocity.y != 0;
      if (has_target) {
        PrepareSpongeTarget();
        swe.ApplySponge(master_dt, config.sponge_width, config.sponge_strength,
                         sponge_target_h.data(), sponge_target_hu.data(),
                         sponge_target_hv.data());
      } else {
        swe.ApplySponge(master_dt, config.sponge_width, config.sponge_strength);
      }
    }

    // 4. SPH substeps.
    // Each finer solver substeps within its parent to satisfy its own CFL
    // condition. SPH CFL depends on max particle velocity and smoothing
    // length. LBM CFL is fixed by the relaxation time tau. Stokes CFL
    // depends on kinematic viscosity and grid spacing.
    //
    // Total fine steps per master step = n_sph * n_lbm * n_stokes.
    // With defaults (10 * 100 * 10) this can reach 10,000. The guardrail
    // below caps total fine steps to prevent runaway in edge cases.
    constexpr int kMaxFineSteps = 2000;
    int total_fine_steps = 0;

    if (sph_active) {
      AABB sph_zone = lod_manager.GetZoneBounds(LODLevel::kSPH);
      float ghost_width = config.sph_smoothing * 3.0f;

      // Compute SWE interface fluxes at SPH boundary for flux correction.
      auto swe_flux = ComputeSWEInterfaceFluxes(swe, sph_zone);

      // Initialize flux tracker for this master step.
      sph_flux_tracker.Reset();
      sph_flux_tracker.SetDeadZone(config.sph_smoothing);

      // Determine SPH substep count from CFL condition.
      float sph_cfl_dt = sph.Count() > 0 ? sph.ComputeMaxDt() : master_dt;
      int n_sph = std::max(1, static_cast<int>(std::ceil(master_dt / sph_cfl_dt)));
      n_sph = std::min(n_sph, config.sph_substeps * 4);
      float sph_dt = master_dt / n_sph;

      for (int s = 0; s < n_sph && total_fine_steps < kMaxFineSteps; ++s) {
        LODTransition::CoupleSWEToSPH(swe, sph, sph_zone, ghost_width);

        // Track particles near boundary before step.
        sph_flux_tracker.BeforeStep(sph.particles, sph.Count(), sph_zone);
        sph.Step(sph_dt);
        // Detect crossings and accumulate flux.
        sph_flux_tracker.AfterStep(sph.particles, sph.Count(), sph_zone, sph_dt);
        ++total_fine_steps;

        // 5. LBM sub-substeps (fixed dt from lattice relaxation time).
        if (lbm_active) {
          int ghost_cells = config.sph_to_lbm_ghost_cells;

          for (int l = 0; l < config.lbm_substeps && total_fine_steps < kMaxFineSteps; ++l) {
            LODTransition::CoupleSPHToLBM(sph, lbm, ghost_cells);
            lbm.Step();
            ++total_fine_steps;

            // 6. Stokes sub-sub-substeps (CFL from viscous diffusion).
            if (stokes_active) {
              float stokes_cfl_dt = stokes.params.StableDt();
              float lbm_dt = lbm.PhysicalDt();
              int n_stokes = std::max(1,
                static_cast<int>(std::ceil(lbm_dt / stokes_cfl_dt)));
              n_stokes = std::min(n_stokes, config.stokes_substeps * 4);
              float stokes_dt = lbm_dt / n_stokes;

              for (int st = 0; st < n_stokes && total_fine_steps < kMaxFineSteps; ++st) {
                LODTransition::CoupleLBMToStokes(lbm, stokes,
                                                   config.stokes_ghost_cells);
                stokes.Step(stokes_dt);
                ++total_fine_steps;
              }
            }
          }
        }
      }

      // Berger-Colella flux correction: reconcile SWE and SPH fluxes.
      LODTransition::CorrectSWESPHInterface(
        swe, sph, swe_flux, sph_flux_tracker, sph_zone, master_dt);
    }

    // Warn if fine step budget was exhausted (simulation is under-resolved).
    if (total_fine_steps >= kMaxFineSteps) {
      fprintf(stderr, "mujoco-water: fine step budget exhausted (%d steps). "
              "Consider reducing LOD substep counts.\n", total_fine_steps);
    }

    sim_time += master_dt;
  }

  // Query fluid state at a world position (uses highest active LOD).
  // Falls through to coarser solvers if the finer one's grid doesn't
  // cover the query point.
  FluidCoupling::FluidState Query(Vec3 pos) const {
    LODLevel lod = lod_manager.GetLOD(pos);

    if (lod == LODLevel::kStokes && stokes_active) {
      if (stokes.grid.Contains(pos))
        return FluidCoupling::QueryStokes(stokes, pos);
    }
    if (lod >= LODLevel::kLBM && lbm_active) {
      if (lbm.grid.Contains(pos))
        return FluidCoupling::QueryLBM(lbm, pos);
    }
    if (lod >= LODLevel::kSPH && sph_active) {
      auto s = FluidCoupling::QuerySPH(sph, pos);
      if (s.submerged) return s;
    }
    if (ocean_active && lod == LODLevel::kOcean) {
      return FluidCoupling::QueryOcean(ocean, pos);
    }
    // Fall back to SWE (always available).
    return FluidCoupling::QuerySWE(swe, pos);
  }

  // Compute coupling force on a body.
  CouplingForce ComputeBodyForce(size_t body_idx, Vec3 body_pos,
                                   Vec3 body_vel, float dt) {
    if (body_idx >= bodies.size()) return {};
    auto& body = bodies[body_idx];

    // Store body state for two-way coupling (used next Step()).
    if (body_idx < body_states.size()) {
      body_states[body_idx].pos = body_pos;
      body_states[body_idx].vel = body_vel;
    }

    auto fluid = Query(body_pos);
    auto force = FluidCoupling::ComputeForces(fluid, body, body_vel, dt);
    body.prev_fluid_vel = fluid.velocity;

    // Add wave radiation reaction from previous step.
    if (body.two_way_enabled && body_idx < body_states.size()) {
      force.force = force.force + body_states[body_idx].wave_reaction;
    }

    return force;
  }

  // Conservation queries.
  float TotalVolume() const { return swe.TotalVolume(); }
  float TotalSPHMass() const { return sph.TotalMass(); }
  float TotalLBMMass() const { return lbm.TotalMass(); }

  // Per-layer statistics for HUD display.
  struct LayerStats {
    const char* name;
    bool active;
    uint32_t cells;
    float memory_kb;
    float step_us;
  };

  std::array<LayerStats, 5> GetLayerStats() const {
    std::array<LayerStats, 5> s;
    s[0] = {"Ocean", ocean_active, static_cast<uint32_t>(ocean.waves.size()),
             static_cast<float>(ocean.waves.size() * sizeof(float) * 8) / 1024.0f,
             config.master_dt * 1e6f};
    s[1] = {"SWE", true, swe.grid.nx * swe.grid.ny,
             static_cast<float>(swe.grid.nx * swe.grid.ny * 3 * sizeof(float)) / 1024.0f,
             config.master_dt * 1e6f};
    s[2] = {"SPH", sph_active, static_cast<uint32_t>(sph.particles.size()),
             static_cast<float>(sph.particles.size() * sizeof(SPHParticle)) / 1024.0f,
             config.sph_substeps > 0 ? (config.master_dt / config.sph_substeps) * 1e6f : 0};
    s[3] = {"LBM", lbm_active, lbm.grid.nx * lbm.grid.ny * lbm.grid.nz,
             static_cast<float>(lbm.grid.nx * lbm.grid.ny * lbm.grid.nz * 19 * sizeof(float)) / 1024.0f,
             (config.sph_substeps > 0 && config.lbm_substeps > 0)
               ? (config.master_dt / config.sph_substeps / config.lbm_substeps) * 1e6f : 0};
    s[4] = {"Stokes", stokes_active,
             stokes.grid.nx * stokes.grid.ny * stokes.grid.nz,
             static_cast<float>(stokes.grid.nx * stokes.grid.ny * stokes.grid.nz * 6 * sizeof(float)) / 1024.0f,
             stokes.params.StableDt() * 1e6f};
    return s;
  }

  // --- Conservation diagnostics ---
  //
  // Computes total mass across all active solvers, avoiding double-counting
  // in overlap zones. Each spatial point belongs to the finest active solver.
  ConservationDiag ComputeConservation() const {
    ConservationDiag diag;
    float cell_area = swe.grid.dx * swe.grid.dx;

    const auto& h = swe.grid.channels[swe.ch_h].data;
    float swe_volume = 0;
    for (size_t i = 0; i < h.size(); ++i) swe_volume += h[i];
    diag.swe_mass = swe_volume * cell_area * kWaterDensity;

    if (sph_active) {
      for (uint32_t i = 0; i < sph.Count(); ++i)
        diag.sph_mass += sph.particles[i].mass;
    }

    if (lbm_active) {
      float lbm_cell_vol = lbm.params.dx_phys * lbm.params.dx_phys * lbm.params.dx_phys;
      for (size_t c = 0; c < lbm.grid.CellCount(); ++c) {
        if (lbm.grid.solid[c]) continue;
        diag.lbm_mass += lbm.grid.Density(c) * lbm.params.DensityScale() * lbm_cell_vol;
      }
    }

    if (stokes_active) {
      float stokes_vol = stokes.params.dx * stokes.params.dx * stokes.params.dx;
      for (size_t c = 0; c < stokes.grid.CellCount(); ++c) {
        if (stokes.grid.solid[c]) continue;
        diag.stokes_mass += kWaterDensity * stokes_vol;
      }
    }

    diag.total_mass = diag.swe_mass;
    if (sph_active) {
      AABB sph_zone = lod_manager.GetZoneBounds(LODLevel::kSPH);
      float swe_in_sph = 0;
      for (uint32_t j = 0; j < swe.grid.ny; ++j) {
        for (uint32_t i = 0; i < swe.grid.nx; ++i) {
          float wx, wy;
          swe.grid.GridToWorld(i, j, wx, wy);
          if (wx >= sph_zone.min.x && wx <= sph_zone.max.x &&
              wy >= sph_zone.min.y && wy <= sph_zone.max.y) {
            swe_in_sph += h[swe.grid.Idx(i, j)];
          }
        }
      }
      diag.total_mass -= swe_in_sph * cell_area * kWaterDensity;
      diag.total_mass += diag.sph_mass;
    }

    return diag;
  }

 private:
  // Inject body wave sources into SWE before stepping.
  void ApplyTwoWayCoupling(float dt) {
    for (size_t i = 0; i < bodies.size(); ++i) {
      if (!bodies[i].two_way_enabled) continue;
      if (i >= body_states.size()) continue;
      auto& bs = body_states[i];

      auto sources = FluidCoupling::ComputeWaveSources(
          swe, bodies[i], bs.pos, bs.vel);
      if (sources.count > 0) {
        swe.ApplySourceTerms(sources.cell_x.data(), sources.cell_y.data(),
                              sources.dh.data(), sources.dhu.data(),
                              sources.dhv.data(), sources.count, dt);
        // Reaction force: negative of momentum injected into water.
        bs.wave_reaction = FluidCoupling::WaveReactionForce(
            sources, swe.grid.dx, dt);
      } else {
        bs.wave_reaction = {};
      }
    }
  }

  // Fill sponge target buffers from ocean + current state.
  void PrepareSpongeTarget() {
    size_t n = swe.grid.nx * swe.grid.ny;
    sponge_target_h.resize(n);
    sponge_target_hu.resize(n);
    sponge_target_hv.resize(n);

    for (uint32_t j = 0; j < swe.grid.ny; ++j) {
      for (uint32_t i = 0; i < swe.grid.nx; ++i) {
        float wx, wy;
        swe.grid.GridToWorld(i, j, wx, wy);
        size_t idx = swe.grid.Idx(i, j);

        float bathy = swe.grid.channels[swe.ch_bathy].data[idx];

        if (ocean_active) {
          float ocean_z = config.ocean_mean_depth + ocean.SurfaceHeight(wx, wy);
          sponge_target_h[idx] = std::max(0.0f, ocean_z - bathy);
          Vec3 vel = ocean.OrbitalVelocity(wx, wy, 0);
          sponge_target_hu[idx] = sponge_target_h[idx] * (vel.x + config.current_velocity.x);
          sponge_target_hv[idx] = sponge_target_h[idx] * (vel.y + config.current_velocity.y);
        } else {
          // Current only: target is initial surface with current momentum.
          sponge_target_h[idx] = std::max(0.0f, config.initial_surface_z - bathy);
          sponge_target_hu[idx] = sponge_target_h[idx] * config.current_velocity.x;
          sponge_target_hv[idx] = sponge_target_h[idx] * config.current_velocity.y;
        }
      }
    }
  }

  // Reposition fine grids when the focus point has moved far enough
  // that the grid no longer covers the focus zone adequately.
  // We re-center the grid when the focus moves more than 25% of
  // the grid extent from the grid center.
  void RepositionGrids(Vec3 focus) {
    if (lbm_active) {
      Vec3 lbm_center = {
        lbm.grid.origin_x + config.lbm_dx * config.lbm_nx * 0.5f,
        lbm.grid.origin_y + config.lbm_dx * config.lbm_ny * 0.5f,
        lbm.grid.origin_z + config.lbm_dx * config.lbm_nz * 0.5f
      };
      float lbm_extent = config.lbm_dx * std::min({config.lbm_nx,
                                                     config.lbm_ny,
                                                     config.lbm_nz});
      float drift = (focus - lbm_center).Length();
      if (drift > lbm_extent * config.lbm_reposition_threshold) {
        // Save LBM state to SPH, reposition, reinitialize from SPH.
        if (stokes_active) {
          LODTransition::StokesToLBM(stokes, lbm);
        }
        LODTransition::LBMToSPH(lbm, sph, config.sph_spacing);
        lbm.grid.origin_x = focus.x - config.lbm_dx * config.lbm_nx * 0.5f;
        lbm.grid.origin_y = focus.y - config.lbm_dx * config.lbm_ny * 0.5f;
        lbm.grid.origin_z = focus.z - config.lbm_dx * config.lbm_nz * 0.5f;
        lbm.InitAtRest();
        LODTransition::SPHToLBM(sph, lbm);
      }
    }

    if (stokes_active) {
      float sdx = config.stokes.dx;
      Vec3 stokes_center = {
        stokes.grid.origin_x + sdx * config.stokes_nx * 0.5f,
        stokes.grid.origin_y + sdx * config.stokes_ny * 0.5f,
        stokes.grid.origin_z + sdx * config.stokes_nz * 0.5f
      };
      float stokes_extent = sdx * std::min({config.stokes_nx,
                                              config.stokes_ny,
                                              config.stokes_nz});
      float drift = (focus - stokes_center).Length();
      if (drift > stokes_extent * config.stokes_reposition_threshold) {
        // Reset and reinitialize Stokes at new position.
        stokes.grid.origin_x = focus.x - sdx * config.stokes_nx * 0.5f;
        stokes.grid.origin_y = focus.y - sdx * config.stokes_ny * 0.5f;
        stokes.grid.origin_z = focus.z - sdx * config.stokes_nz * 0.5f;
        stokes.grid.Clear();
        if (lbm_active) {
          LODTransition::LBMToStokes(lbm, stokes);
        }
      }
    }
  }

  void UpdateLOD() {
    lod_manager.UpdateAll();

    // Check if any region or focus point requires finer LOD.
    bool need_sph = false, need_lbm = false, need_stokes = false;
    for (const auto& r : lod_manager.regions) {
      if (!r.active) continue;
      if (r.current >= LODLevel::kSPH) need_sph = true;
      if (r.current >= LODLevel::kLBM) need_lbm = true;
      if (r.current >= LODLevel::kStokes) need_stokes = true;
    }

    // Also check based on focus distance.
    LODLevel focus_lod = lod_manager.GetLOD(lod_manager.focus);
    if (focus_lod >= LODLevel::kSPH) need_sph = true;
    if (focus_lod >= LODLevel::kLBM) need_lbm = true;
    if (focus_lod >= LODLevel::kStokes) need_stokes = true;

    // Skip if current state already matches what's needed.
    if (need_sph == sph_active && need_lbm == lbm_active &&
        need_stokes == stokes_active) {
      return;
    }

    // Activate/deactivate from coarsest to finest.
    if (need_sph && !sph_active) {
      ActivateSPH();
    } else if (!need_sph && sph_active) {
      DeactivateSPH();
    }

    // LBM requires SPH to be active with particles for initialization.
    if (need_lbm && !lbm_active && sph_active && sph.Count() > 0) {
      ActivateLBM();
    } else if (!need_lbm && lbm_active) {
      DeactivateLBM();
    }

    // Stokes requires LBM to be active for initialization.
    if (need_stokes && !stokes_active && lbm_active) {
      ActivateStokes();
    } else if (!need_stokes && stokes_active) {
      DeactivateStokes();
    }
  }

  void ActivateSPH() {
    AABB zone = lod_manager.GetZoneBounds(LODLevel::kSPH);
    float mass_before = swe.TotalVolume() * kWaterDensity;
    Vec3 mom_before = swe.TotalMomentum() * kWaterDensity;

    sph.particles.clear();
    LODTransition::SWEToSPH(swe, sph, zone, config.sph_spacing);

    // Build spatial hash so interpolation queries work immediately
    // (needed by ActivateLBM which may run before the first SPH step).
    if (sph.Count() > 0) {
      sph.hash.Init(sph.kernel.support, sph.Count());
      sph.hash.Build(sph.particles);
    }

    // Conserve mass and momentum.
    float sph_mass = sph.TotalMass();
    float vol_fraction = zone.Volume() / config.domain.Volume();
    float target_mass = mass_before * vol_fraction;
    if (sph_mass > 0 && target_mass > 0) {
      LODTransition::CorrectMass_SPH(sph, target_mass);
      Vec3 target_mom = mom_before * vol_fraction;
      LODTransition::CorrectMomentum_SPH(sph, target_mom);
    }

    sph_active = true;
  }

  void DeactivateSPH() {
    AABB zone = lod_manager.GetZoneBounds(LODLevel::kSPH);
    // Save SPH mass/momentum before de-escalation.
    float sph_mass = sph.TotalMass();
    float sph_vol = sph_mass / kWaterDensity;
    LODTransition::SPHToSWE(sph, swe, zone);
    // Correct SWE volume to match SPH mass.
    if (sph_vol > 0) {
      LODTransition::CorrectMass_SWE(swe, zone, sph_vol);
    }
    sph.particles.clear();
    sph_active = false;
    lbm_active = false;  // LBM requires SPH
  }

  void ActivateLBM() {
    AABB zone = lod_manager.GetZoneBounds(LODLevel::kLBM);
    // Position LBM grid at zone center.
    Vec3 center = zone.Center();
    lbm.grid.origin_x = center.x - config.lbm_dx * config.lbm_nx * 0.5f;
    lbm.grid.origin_y = center.y - config.lbm_dx * config.lbm_ny * 0.5f;
    lbm.grid.origin_z = center.z - config.lbm_dx * config.lbm_nz * 0.5f;

    LODTransition::SPHToLBM(sph, lbm);
    lbm_active = true;
  }

  void DeactivateLBM() {
    if (stokes_active) DeactivateStokes();
    if (sph_active) {
      LODTransition::LBMToSPH(lbm, sph, config.sph_spacing);
    }
    lbm.InitAtRest();
    lbm_active = false;
  }

  void ActivateStokes() {
    AABB zone = lod_manager.GetZoneBounds(LODLevel::kStokes);
    Vec3 center = zone.Center();
    float dx = config.stokes.dx;

    // Position Stokes grid at zone center.
    stokes.grid.origin_x = center.x - dx * config.stokes_nx * 0.5f;
    stokes.grid.origin_y = center.y - dx * config.stokes_ny * 0.5f;
    stokes.grid.origin_z = center.z - dx * config.stokes_nz * 0.5f;

    // Initialize from LBM if active.
    if (lbm_active) {
      LODTransition::LBMToStokes(lbm, stokes);
    }

    stokes_active = true;
  }

  void DeactivateStokes() {
    if (lbm_active) {
      LODTransition::StokesToLBM(stokes, lbm);
    }
    // Reset Stokes fields.
    std::fill(stokes.grid.u.begin(), stokes.grid.u.end(), 0.0f);
    std::fill(stokes.grid.v.begin(), stokes.grid.v.end(), 0.0f);
    std::fill(stokes.grid.w.begin(), stokes.grid.w.end(), 0.0f);
    std::fill(stokes.grid.p.begin(), stokes.grid.p.end(), 0.0f);
    stokes_active = false;
  }
};

}  // namespace mjwater
