// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#include "test_harness.h"
#include "mjwater/lod_manager.h"
#include "mjwater/lod_transition.h"
#include "mjwater/water_engine.h"
#include "mjwater/water_sdf.h"

using namespace mjwater;

TEST(LOD_DistanceAssignment) {
  LODManager mgr;
  LODConfig cfg;
  cfg.lbm_radius = 0.05f;
  cfg.sph_radius = 0.5f;
  mgr.Init(cfg);
  mgr.SetFocus(0, 0, 0);

  // Point at origin -> Stokes (finest, within stokes_radius=0.005).
  CHECK(mgr.GetLOD({0, 0, 0}) == LODLevel::kStokes);
  // Point at 0.01m -> LBM (outside stokes 0.005, inside lbm 0.05).
  CHECK(mgr.GetLOD({0.01f, 0, 0}) == LODLevel::kLBM);
  // Point at 0.1m -> SPH.
  CHECK(mgr.GetLOD({0.1f, 0, 0}) == LODLevel::kSPH);
  // Point at 1.0m -> SWE.
  CHECK(mgr.GetLOD({1.0f, 0, 0}) == LODLevel::kShallowWater);
  // Point at 100m -> Ocean.
  CHECK(mgr.GetLOD({100.0f, 0, 0}) == LODLevel::kOcean);
}

TEST(LOD_Hysteresis) {
  LODManager mgr;
  LODConfig cfg;
  cfg.lbm_radius = 1.0f;
  cfg.sph_radius = 5.0f;
  cfg.hysteresis = 0.2f;
  mgr.Init(cfg);
  mgr.SetFocus(0, 0, 0);

  // At 1.1m with current=LBM -> should stay LBM (within outer threshold 1.2).
  LODLevel lod = mgr.GetLODWithHysteresis({1.1f, 0, 0}, LODLevel::kLBM);
  CHECK(lod == LODLevel::kLBM);

  // At 1.3m with current=LBM -> should downgrade (past outer 1.2).
  lod = mgr.GetLODWithHysteresis({1.3f, 0, 0}, LODLevel::kLBM);
  CHECK(lod == LODLevel::kSPH);

  // At 0.9m with current=SPH -> should stay SPH (inner threshold is 0.8).
  lod = mgr.GetLODWithHysteresis({0.9f, 0, 0}, LODLevel::kSPH);
  CHECK(lod == LODLevel::kSPH);

  // At 0.7m with current=SPH -> should upgrade to LBM.
  lod = mgr.GetLODWithHysteresis({0.7f, 0, 0}, LODLevel::kSPH);
  CHECK(lod == LODLevel::kLBM);
}

TEST(LOD_ZoneBounds) {
  LODManager mgr;
  LODConfig cfg;
  cfg.lbm_radius = 0.05f;
  cfg.sph_radius = 0.5f;
  mgr.Init(cfg);
  mgr.SetFocus({1, 2, 3});

  AABB lbm_zone = mgr.GetZoneBounds(LODLevel::kLBM);
  CHECK_NEAR(lbm_zone.Center().x, 1.0f, 1e-5f);
  CHECK_NEAR(lbm_zone.Size().x, 0.1f, 1e-5f);  // 2 * 0.05

  AABB sph_zone = mgr.GetZoneBounds(LODLevel::kSPH);
  CHECK_NEAR(sph_zone.Size().x, 1.0f, 1e-5f);  // 2 * 0.5
}

TEST(LOD_SWEToSPHTransition) {
  ShallowWaterSolver swe;
  swe.Init(10, 10, 0.1f);
  swe.SetSurface(0.5f);

  CHECK(swe.TotalVolume() > 0);

  SPHSolver sph;
  SPHParams sp;
  sp.smoothing_length = 0.02f;
  AABB domain = {{0, 0, 0}, {1, 1, 1}};
  sph.Init(sp, domain);

  AABB region = {{0.2f, 0.2f, 0}, {0.8f, 0.8f, 1}};
  LODTransition::SWEToSPH(swe, sph, region, 0.05f);

  CHECK(sph.Count() > 0);
  CHECK(sph.TotalMass() > 0);
}

TEST(LOD_SWEToSPHMassConservation) {
  // Set up a small SWE grid with known water volume.
  ShallowWaterSolver swe;
  swe.Init(10, 10, 0.1f);
  swe.SetSurface(0.3f);

  float swe_vol = swe.TotalVolume();
  CHECK(swe_vol > 0);

  // Transfer to SPH.
  SPHSolver sph;
  SPHParams sp;
  sp.smoothing_length = 0.02f;
  AABB domain = {{0, 0, 0}, {1, 1, 1}};
  sph.Init(sp, domain);

  AABB region = {{0, 0, 0}, {1, 1, 1}};
  float spacing = 0.05f;
  LODTransition::SWEToSPH(swe, sph, region, spacing);
  CHECK(sph.Count() > 0);

  // Apply mass correction and check conservation.
  float target_mass = swe_vol * kWaterDensity;
  LODTransition::CorrectMass_SPH(sph, target_mass);

  float sph_mass = sph.TotalMass();
  float rel_err = std::abs(sph_mass - target_mass) / target_mass;
  CHECK(rel_err < 1e-5f);
}

TEST(LOD_SPHToSWERoundTrip) {
  // Create SWE grid, transfer to SPH, then back. Check mass is preserved.
  ShallowWaterSolver swe;
  swe.Init(10, 10, 0.1f);
  swe.SetSurface(0.4f);

  float original_vol = swe.TotalVolume();

  SPHSolver sph;
  SPHParams sp;
  sp.smoothing_length = 0.02f;
  AABB domain = {{0, 0, 0}, {1, 1, 1}};
  sph.Init(sp, domain);

  AABB region = {{0, 0, 0}, {1, 1, 1}};
  LODTransition::SWEToSPH(swe, sph, region, 0.05f);

  // Correct mass to match original.
  float target_mass = original_vol * kWaterDensity;
  LODTransition::CorrectMass_SPH(sph, target_mass);

  // Transfer back to SWE.
  LODTransition::SPHToSWE(sph, swe, region);

  // Correct volume to match original.
  LODTransition::CorrectMass_SWE(swe, region, original_vol);

  float restored_vol = swe.TotalVolume();
  float rel_err = std::abs(restored_vol - original_vol) / original_vol;
  CHECK(rel_err < 0.01f);
}

TEST(LOD_SPHToLBMRoundTrip) {
  // Create SPH particles, transfer to LBM, transfer back. Check particle count.
  SPHSolver sph;
  SPHParams sp;
  sp.smoothing_length = 0.02f;
  AABB domain = {{0, 0, 0}, {0.1f, 0.1f, 0.1f}};
  sph.Init(sp, domain);

  // Manually add some particles.
  float spacing = 0.02f;
  for (float z = 0.01f; z < 0.09f; z += spacing) {
    for (float y = 0.01f; y < 0.09f; y += spacing) {
      for (float x = 0.01f; x < 0.09f; x += spacing) {
        SPHParticle p;
        p.pos = {x, y, z};
        p.vel = {0.1f, 0, 0};
        p.mass = sp.ParticleMass(spacing);
        p.density = sp.rest_density;
        sph.particles.push_back(p);
      }
    }
  }

  float original_mass = sph.TotalMass();
  CHECK(sph.Count() > 0);

  // Build neighbor hash (required for SPH interpolation queries).
  sph.hash.Init(sph.kernel.support, sph.Count());
  sph.hash.Build(sph.particles);

  // Transfer to LBM.
  LBMSolver lbm;
  LBMParams lp;
  lp.dx_phys = 0.005f;
  // Domain is 0.1m, spacing 0.005m -> 20 cells per axis.
  uint32_t ncells = static_cast<uint32_t>(0.1f / lp.dx_phys);
  lbm.Init(ncells, ncells, ncells, lp);

  LODTransition::SPHToLBM(sph, lbm);

  // Transfer back to a fresh SPH.
  SPHSolver sph2;
  sph2.Init(sp, domain);
  LODTransition::LBMToSPH(lbm, sph2, spacing);

  CHECK(sph2.Count() > 0);

  // Mass correction should recover original mass.
  // Note: LBM->SPH spawns particles at cell centers, so pre-correction
  // mass differs. CorrectMass_SPH scales all particle masses to match target.
  LODTransition::CorrectMass_SPH(sph2, original_mass);
  float restored_mass = sph2.TotalMass();
  float rel_err = std::abs(restored_mass - original_mass) / original_mass;
  CHECK(rel_err < 0.01f);  // 1% tolerance (exact after correction)
}

TEST(LOD_MomentumPreservation) {
  // Check that SWE->SPH preserves momentum direction.
  ShallowWaterSolver swe;
  swe.Init(10, 10, 0.1f);
  swe.SetSurface(0.3f);

  // Set uniform x-momentum in SWE.
  auto& hu = swe.grid.channels[swe.ch_hu].data;
  auto& h = swe.grid.channels[swe.ch_h].data;
  for (size_t i = 0; i < hu.size(); ++i) {
    hu[i] = h[i] * 0.5f;  // u = 0.5 m/s in x
  }

  SPHSolver sph;
  SPHParams sp;
  sp.smoothing_length = 0.02f;
  AABB domain = {{0, 0, 0}, {1, 1, 1}};
  sph.Init(sp, domain);

  AABB region = {{0, 0, 0}, {1, 1, 1}};
  LODTransition::SWEToSPH(swe, sph, region, 0.05f);

  // All particles should have positive x-velocity.
  float total_vx = 0, total_vy = 0;
  for (const auto& p : sph.particles) {
    total_vx += p.vel.x * p.mass;
    total_vy += p.vel.y * p.mass;
  }
  CHECK(total_vx > 0);
  CHECK(std::abs(total_vy) < 1e-6f * std::abs(total_vx));
}

TEST(WaterEngine_Init) {
  WaterEngine engine;
  WaterEngineConfig cfg;
  cfg.swe_nx = 20;
  cfg.swe_ny = 20;
  cfg.swe_dx = 0.1f;
  cfg.initial_surface_z = 0.5f;
  cfg.domain = {{0, 0, 0}, {2, 2, 1}};
  // Disable deeper LOD levels for this SWE-only test.
  cfg.lod.sph_radius = -1.0f;
  cfg.lod.lbm_radius = -1.0f;
  cfg.lod.stokes_radius = -1.0f;
  engine.Init(cfg);

  CHECK(engine.TotalVolume() > 0);
  CHECK(!engine.sph_active);
  CHECK(!engine.lbm_active);
}

TEST(WaterEngine_Step) {
  WaterEngine engine;
  WaterEngineConfig cfg;
  cfg.swe_nx = 10;
  cfg.swe_ny = 10;
  cfg.swe_dx = 0.2f;
  cfg.initial_surface_z = 0.3f;
  cfg.domain = {{0, 0, 0}, {2, 2, 1}};
  cfg.master_dt = 0.005f;
  // Disable deeper LOD levels for this SWE-only test.
  cfg.lod.sph_radius = -1.0f;
  cfg.lod.lbm_radius = -1.0f;
  cfg.lod.stokes_radius = -1.0f;
  engine.Init(cfg);

  float vol_before = engine.TotalVolume();
  engine.Step();
  float vol_after = engine.TotalVolume();

  // Volume should be conserved (flat quiescent pool, no sources).
  float rel_err = std::abs(vol_after - vol_before) / (vol_before + 1e-10f);
  CHECK(rel_err < 1e-4f);
}

TEST(WaterEngine_FiveLevels) {
  // Multi-level engine: SWE + SPH + LBM + Stokes.
  WaterEngine engine;
  WaterEngineConfig cfg;
  cfg.swe_nx = 20;
  cfg.swe_ny = 20;
  cfg.swe_dx = 0.1f;
  cfg.initial_surface_z = 0.5f;
  cfg.domain = {{0, 0, 0}, {2, 2, 1}};
  cfg.master_dt = 0.001f;
  cfg.sph_substeps = 2;
  cfg.sph_spacing = 0.05f;
  cfg.sph_smoothing = 0.08f;
  cfg.lbm_nx = 5; cfg.lbm_ny = 5; cfg.lbm_nz = 5;
  cfg.lbm_dx = 0.01f;
  cfg.lbm_substeps = 2;
  cfg.stokes_nx = 5; cfg.stokes_ny = 5; cfg.stokes_nz = 5;
  cfg.stokes.dx = 10e-6f;
  cfg.stokes.diffusivity = 1e-9f;
  cfg.stokes_substeps = 2;
  cfg.lod.sph_radius = 1.0f;
  cfg.lod.lbm_radius = 0.5f;
  cfg.lod.stokes_radius = 0.1f;
  cfg.lod.swe_radius = 50.0f;
  engine.Init(cfg);

  // Focus at center activates LOD levels.
  engine.SetFocus({1, 1, 0.5f});

  // Step to trigger LOD transitions.
  for (int i = 0; i < 3; ++i) engine.Step();

  // SPH should have been activated by LOD manager.
  CHECK(engine.sph_active);

  // Query fluid below surface should return submerged.
  auto state = engine.Query({1, 1, 0.2f});
  CHECK(state.submerged);

  // GetLayerStats should report active layers.
  auto stats = engine.GetLayerStats();
  CHECK(stats[1].active);  // SWE (always on)
  CHECK(stats[2].active);  // SPH
}

TEST(WaterEngine_Query) {
  WaterEngine engine;
  WaterEngineConfig cfg;
  cfg.swe_nx = 10;
  cfg.swe_ny = 10;
  cfg.swe_dx = 0.2f;
  cfg.initial_surface_z = 0.5f;
  cfg.domain = {{0, 0, 0}, {2, 2, 1}};
  cfg.lod.sph_radius = -1.0f;
  cfg.lod.lbm_radius = -1.0f;
  cfg.lod.stokes_radius = -1.0f;
  engine.Init(cfg);

  // Query below water surface should be submerged.
  auto s1 = engine.Query({1.0f, 1.0f, 0.2f});
  CHECK(s1.submerged);
  CHECK(s1.depth > 0);

  // Query above water surface should not be submerged.
  auto s2 = engine.Query({1.0f, 1.0f, 0.8f});
  CHECK(!s2.submerged);
}

TEST(SDF_PoolBathymetry) {
  WaterSDF sdf;
  sdf.InitPool(2.0f, 1.0f, 0.5f);  // 2m x 1m, 0.5m deep

  // Pool floor should be at z = -0.5 (box centered at cz = -0.25).
  // Inside the pool: Evaluate should be <= 0 at floor level.
  CHECK(sdf.Evaluate(1.0f, 0.5f, -0.1f) <= 0);   // inside pool
  CHECK(sdf.Evaluate(1.0f, 0.5f, 0.5f) > 0);     // above pool

  // Bake to height field.
  ShallowWaterSolver swe;
  swe.Init(20, 10, 0.1f);
  sdf.BakeToHeightField(swe.grid, swe.ch_bathy);

  // Center of pool: bed should be at bottom (z = -0.5).
  float bed_center = swe.grid.Sample(swe.ch_bathy, 1.0f, 0.5f);
  CHECK(bed_center < 0);  // pool floor is below z=0

  // Set surface and verify water depth.
  swe.SetSurface(0.0f);
  float h_center = swe.grid.Sample(swe.ch_h, 1.0f, 0.5f);
  CHECK(h_center > 0.4f);  // ~0.5m of water
}

TEST(WaterEngine_InitWithSDF) {
  WaterSDF sdf;
  sdf.InitPool(2.0f, 1.0f, 0.5f);

  WaterEngineConfig cfg;
  cfg.swe_nx = 20;
  cfg.swe_ny = 10;
  cfg.swe_dx = 0.1f;
  cfg.initial_surface_z = 0.0f;
  cfg.domain = {{0, 0, -1}, {2, 1, 1}};
  cfg.lod.sph_radius = -1.0f;
  cfg.lod.lbm_radius = -1.0f;
  cfg.lod.stokes_radius = -1.0f;

  WaterEngine engine;
  engine.Init(cfg, sdf);

  CHECK(engine.TotalVolume() > 0);

  // Query below surface should be submerged.
  auto s = engine.Query({1.0f, 0.5f, -0.2f});
  CHECK(s.submerged);
}

TEST(WaterEngine_LODRoundTrip) {
  // Activate all LOD levels, step, deactivate, verify mass is conserved.
  WaterEngine engine;
  WaterEngineConfig cfg;
  cfg.swe_nx = 20;
  cfg.swe_ny = 20;
  cfg.swe_dx = 0.1f;
  cfg.initial_surface_z = 0.5f;
  cfg.domain = {{0, 0, 0}, {2, 2, 1}};
  cfg.master_dt = 0.001f;
  cfg.sph_substeps = 1;
  cfg.sph_spacing = 0.05f;
  cfg.sph_smoothing = 0.08f;
  cfg.lbm_nx = 5; cfg.lbm_ny = 5; cfg.lbm_nz = 5;
  cfg.lbm_dx = 0.01f;
  cfg.lbm_substeps = 1;
  cfg.stokes_nx = 5; cfg.stokes_ny = 5; cfg.stokes_nz = 5;
  cfg.stokes.dx = 10e-6f;
  cfg.stokes_substeps = 1;
  cfg.lod.sph_radius = 1.0f;
  cfg.lod.lbm_radius = 0.5f;
  cfg.lod.stokes_radius = 0.1f;
  cfg.lod.swe_radius = 50.0f;
  engine.Init(cfg);

  float vol_before = engine.TotalVolume();

  // Activate: focus at center triggers all levels.
  engine.SetFocus({1, 1, 0.5f});
  for (int i = 0; i < 5; ++i) engine.Step();
  CHECK(engine.sph_active);

  // Deactivate: move focus far away so LOD drops to SWE only.
  engine.SetFocus({1000, 1000, 0.5f});
  engine.Step();  // triggers UpdateLOD -> deactivation

  float vol_after = engine.TotalVolume();
  // With flux-matching corrections, conservation should be within 5%.
  float rel_err = std::abs(vol_after - vol_before) / (vol_before + 1e-10f);
  CHECK(rel_err < 0.05f);
}

TEST(WaterEngine_GridReposition) {
  // Test that LBM grid follows focus movement.
  WaterEngine engine;
  WaterEngineConfig cfg;
  cfg.swe_nx = 40;
  cfg.swe_ny = 40;
  cfg.swe_dx = 0.1f;
  cfg.initial_surface_z = 0.5f;
  cfg.domain = {{0, 0, 0}, {4, 4, 1}};
  cfg.master_dt = 0.001f;
  cfg.sph_substeps = 1;
  cfg.sph_spacing = 0.05f;
  cfg.sph_smoothing = 0.08f;
  cfg.lbm_nx = 5; cfg.lbm_ny = 5; cfg.lbm_nz = 5;
  cfg.lbm_dx = 0.01f;
  cfg.lbm_substeps = 1;
  cfg.stokes_nx = 5; cfg.stokes_ny = 5; cfg.stokes_nz = 5;
  cfg.stokes.dx = 10e-6f;
  cfg.stokes_substeps = 1;
  cfg.lod.sph_radius = 2.0f;
  cfg.lod.lbm_radius = 1.0f;
  cfg.lod.stokes_radius = 0.5f;
  cfg.lod.swe_radius = 50.0f;
  engine.Init(cfg);

  // Focus at (1,1) - activates all LOD levels.
  engine.SetFocus({1, 1, 0.5f});
  engine.Step();
  CHECK(engine.lbm_active);

  // Record LBM grid center.
  float orig_ox = engine.lbm.grid.origin_x;

  // Move focus significantly (more than 25% of grid extent).
  engine.SetFocus({3, 3, 0.5f});

  // LBM grid should have repositioned.
  CHECK(engine.lbm.grid.origin_x != orig_ox);

  // New grid center should be near new focus.
  float new_cx = engine.lbm.grid.origin_x + cfg.lbm_dx * cfg.lbm_nx * 0.5f;
  CHECK_NEAR(new_cx, 3.0f, 0.1f);
}

TEST(LOD_SWEOnlyMassConservation) {
  // SWE-only: run 10 steps with no sources, verify total volume drift < 1e-4.
  // Uses a standalone SWE solver (no WaterEngine) to avoid initializing
  // the full multi-level stack which is slow.
  ShallowWaterSolver swe;
  swe.Init(10, 10, 0.1f);
  swe.SetSurface(0.5f);

  float vol0 = swe.TotalVolume();
  CHECK(vol0 > 0);

  for (int i = 0; i < 10; ++i) swe.Step(0.005f);

  float vol1 = swe.TotalVolume();
  float drift = std::abs(vol1 - vol0) / vol0;
  CHECK(drift < 1e-4f);
}

int main() {
  return RunAllTests();
}
