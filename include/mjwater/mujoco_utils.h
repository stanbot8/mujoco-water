// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// MuJoCo integration utilities.
//
// Helper functions for connecting mujoco-water solvers to MuJoCo's rendering
// and physics systems. These require MuJoCo headers; guard with
// MJGAME_VIEWER or similar if needed.
//
// Usage:
//   #include <mujoco/mujoco.h>
//   #include "mjwater/mujoco_utils.h"
//
//   // Copy SWE surface to a MuJoCo hfield for rendering.
//   mjwater::UpdateMuJoCoHField(m, &con, engine, hfield_id, max_z);
//
//   // Apply fluid forces to a MuJoCo body.
//   mjwater::ApplyFluidForces(d, body_id, force);

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "mjwater/water_engine.h"

namespace mjwater {

// Disable MuJoCo's built-in fluid forces on a specific body.
// Zeroes geom_fluid for every geom attached to body_id, preventing
// MuJoCo from applying its own drag/buoyancy on that body while
// leaving all other bodies on the built-in fluid model.
// Call once per body after mjModel is loaded, before the simulation loop.
inline void DisableBuiltinFluid(mjModel* m, int body_id) {
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_bodyid[g] == body_id) {
      for (int k = 0; k < mjNFLUID; ++k) {
        m->geom_fluid[g * mjNFLUID + k] = 0;
      }
    }
  }
}

// Copy SWE + ocean surface heights to a MuJoCo hfield asset.
//
// The hfield data is normalized to [0, 1] where 1.0 maps to z_max.
// Call mjr_uploadHField after this to push to GPU.
//
// Parameters:
//   m:         MuJoCo model (reads hfield dimensions, writes hfield_data)
//   con:       MuJoCo render context (for mjr_uploadHField)
//   engine:    WaterEngine with active SWE (and optionally ocean)
//   hfield_id: index from mj_name2id(m, mjOBJ_HFIELD, "name")
//   z_max:     hfield z scale (the "size[2]" from the hfield asset)
//
// Note: hfield rows map to -Y..+Y in MuJoCo, so we flip the Y axis.
inline void UpdateMuJoCoHField(mjModel* m, const mjrContext* con,
                                const WaterEngine& engine, int hfield_id,
                                float z_max) {
  const auto& swe = engine.swe;
  const auto& h_data = swe.grid.channels[swe.ch_h].data;
  const auto& b_data = swe.grid.channels[swe.ch_bathy].data;

  int nrow = m->hfield_nrow[hfield_id];
  int ncol = m->hfield_ncol[hfield_id];
  float* hdata = m->hfield_data + m->hfield_adr[hfield_id];
  float inv_zmax = 1.0f / z_max;

  for (int row = 0; row < nrow; ++row) {
    for (int col = 0; col < ncol; ++col) {
      uint32_t gx = static_cast<uint32_t>(col);
      uint32_t gy = static_cast<uint32_t>(nrow - 1 - row);  // flip Y
      if (gx >= swe.grid.nx) gx = swe.grid.nx - 1;
      if (gy >= swe.grid.ny) gy = swe.grid.ny - 1;

      size_t idx = swe.grid.Idx(gx, gy);
      float surface = b_data[idx] + h_data[idx];

      // Add ocean wave height if ocean is active.
      if (engine.ocean_active) {
        float wx, wy;
        swe.grid.GridToWorld(gx, gy, wx, wy);
        surface += engine.ocean.SurfaceHeight(wx, wy);
      }

      float normalized = std::clamp(surface * inv_zmax, 0.0f, 1.0f);
      hdata[row * ncol + col] = normalized;
    }
  }
  mjr_uploadHField(m, con, hfield_id);
}

// Apply a CouplingForce to a MuJoCo body via xfrc_applied.
//
// MuJoCo layout: xfrc_applied[6*body_id + 0..2] = force (fx, fy, fz)
//                xfrc_applied[6*body_id + 3..5] = torque (tx, ty, tz)
inline void ApplyFluidForces(mjData* d, int body_id,
                              const CouplingForce& force) {
  d->xfrc_applied[6 * body_id + 0] = force.force.x;
  d->xfrc_applied[6 * body_id + 1] = force.force.y;
  d->xfrc_applied[6 * body_id + 2] = force.force.z;
  d->xfrc_applied[6 * body_id + 3] = force.torque.x;
  d->xfrc_applied[6 * body_id + 4] = force.torque.y;
  d->xfrc_applied[6 * body_id + 5] = force.torque.z;
}

}  // namespace mjwater
