// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Distance-based LOD zone assignment with hysteresis.
//
// Assigns each spatial region to a solver fidelity level (SWE/SPH/LBM)
// based on distance from a movable focus point. Hysteresis prevents
// rapid flickering at zone boundaries.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mjwater/types.h"

namespace mjwater {

struct LODZone {
  float radius;         // outer radius of this zone (m)
  LODLevel level;       // solver level within this zone
};

struct LODConfig {
  // Zone radii from focus point (innermost to outermost).
  // Stokes < LBM < SPH < SWE; Ocean covers everything beyond SWE.
  float stokes_radius = 0.005f; // 5 mm, cellular detail zone
  float lbm_radius    = 0.05f;  // 5 cm, microfluidics zone
  float sph_radius    = 0.5f;   // 50 cm, splash/particle zone
  float swe_radius    = 50.0f;  // 50 m, shallow water zone
  // Beyond swe_radius: ocean (spectral)
  float hysteresis = 0.1f;      // relative hysteresis band (fraction of radius)
};

// Per-region tracking for LOD assignment.
struct LODRegion {
  AABB bounds;
  LODLevel current = LODLevel::kShallowWater;
  bool active = true;
};

struct LODManager {
  Vec3 focus;                       // current focus point (world coords)
  LODConfig config;
  std::vector<LODRegion> regions;

  void Init(const LODConfig& cfg) {
    config = cfg;
    focus = {};
    regions.clear();
  }

  void SetFocus(Vec3 pos) { focus = pos; }
  void SetFocus(float x, float y, float z) { focus = {x, y, z}; }

  // Determine LOD level for a point based on distance from focus.
  LODLevel GetLOD(Vec3 pos) const {
    float dist = (pos - focus).Length();
    if (dist <= config.stokes_radius) return LODLevel::kStokes;
    if (dist <= config.lbm_radius)    return LODLevel::kLBM;
    if (dist <= config.sph_radius)    return LODLevel::kSPH;
    if (dist <= config.swe_radius)    return LODLevel::kShallowWater;
    return LODLevel::kOcean;
  }

  // LOD with hysteresis: upgrading requires crossing inner threshold,
  // downgrading requires crossing outer threshold.
  LODLevel GetLODWithHysteresis(Vec3 pos, LODLevel current_level) const {
    float dist = (pos - focus).Length();
    float hyst = config.hysteresis;

    // Inner (upgrade) and outer (downgrade) thresholds for each zone.
    float stokes_in  = config.stokes_radius * (1.0f - hyst);
    float stokes_out = config.stokes_radius * (1.0f + hyst);
    float lbm_in     = config.lbm_radius * (1.0f - hyst);
    float lbm_out    = config.lbm_radius * (1.0f + hyst);
    float sph_in     = config.sph_radius * (1.0f - hyst);
    float sph_out    = config.sph_radius * (1.0f + hyst);
    float swe_in     = config.swe_radius * (1.0f - hyst);
    float swe_out    = config.swe_radius * (1.0f + hyst);

    // Determine level based on current level (hysteresis: sticky to current).
    switch (current_level) {
      case LODLevel::kStokes:
        if (dist > stokes_out) {
          if (dist <= lbm_out) return LODLevel::kLBM;
          if (dist <= sph_out) return LODLevel::kSPH;
          if (dist <= swe_out) return LODLevel::kShallowWater;
          return LODLevel::kOcean;
        }
        return LODLevel::kStokes;

      case LODLevel::kLBM:
        if (dist <= stokes_in) return LODLevel::kStokes;
        if (dist > lbm_out) {
          if (dist <= sph_out) return LODLevel::kSPH;
          if (dist <= swe_out) return LODLevel::kShallowWater;
          return LODLevel::kOcean;
        }
        return LODLevel::kLBM;

      case LODLevel::kSPH:
        if (dist <= lbm_in) {
          if (dist <= stokes_in) return LODLevel::kStokes;
          return LODLevel::kLBM;
        }
        if (dist > sph_out) {
          if (dist <= swe_out) return LODLevel::kShallowWater;
          return LODLevel::kOcean;
        }
        return LODLevel::kSPH;

      case LODLevel::kShallowWater:
        if (dist <= sph_in) {
          if (dist <= lbm_in) {
            if (dist <= stokes_in) return LODLevel::kStokes;
            return LODLevel::kLBM;
          }
          return LODLevel::kSPH;
        }
        if (dist > swe_out) return LODLevel::kOcean;
        return LODLevel::kShallowWater;

      case LODLevel::kOcean:
      default:
        if (dist <= stokes_in) return LODLevel::kStokes;
        if (dist <= lbm_in) return LODLevel::kLBM;
        if (dist <= sph_in) return LODLevel::kSPH;
        if (dist <= swe_in) return LODLevel::kShallowWater;
        return LODLevel::kOcean;
    }
  }

  // Add a region to track.
  size_t AddRegion(const AABB& bounds) {
    LODRegion r;
    r.bounds = bounds;
    r.current = GetLOD(bounds.Center());
    regions.push_back(r);
    return regions.size() - 1;
  }

  // Update all regions, return number of transitions.
  int UpdateAll() {
    int transitions = 0;
    for (auto& r : regions) {
      if (!r.active) continue;
      LODLevel new_lod = GetLODWithHysteresis(r.bounds.Center(), r.current);
      if (new_lod != r.current) {
        r.current = new_lod;
        ++transitions;
      }
    }
    return transitions;
  }

  // Query the AABB that should be covered by a given LOD level.
  AABB GetZoneBounds(LODLevel level) const {
    float r;
    switch (level) {
      case LODLevel::kStokes:       r = config.stokes_radius; break;
      case LODLevel::kLBM:          r = config.lbm_radius; break;
      case LODLevel::kSPH:          r = config.sph_radius; break;
      case LODLevel::kShallowWater: r = config.swe_radius; break;
      case LODLevel::kOcean:
      default:                      r = config.swe_radius * 100.0f; break;
    }
    return AABB{focus - Vec3{r, r, r}, focus + Vec3{r, r, r}};
  }
};

}  // namespace mjwater
