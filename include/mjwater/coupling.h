// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// MuJoCo body-fluid coupling.
//
// Queries local flow velocity and pressure from the active LOD solver,
// computes buoyancy, drag, and added mass forces, and writes them to
// MuJoCo's xfrc_applied array.
//
// References:
//   Batchelor, G.K. (1967) "An Introduction to Fluid Dynamics" (Cambridge)
//   Newman, J.N. (1977) "Marine Hydrodynamics" (MIT Press)

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "mjwater/types.h"
#include "mjwater/spectral_ocean.h"
#include "mjwater/shallow_water.h"
#include "mjwater/sph.h"
#include "mjwater/lbm.h"
#include "mjwater/stokes.h"
#include "mjwater/lod_manager.h"

namespace mjwater {

struct BodyCoupling {
  float drag_coeff = 1.0f;      // Cd (dimensionless)
  float added_mass_coeff = 0.5f; // Ca (0.5 for sphere)
  float cross_section = 0.01f;  // reference area (m^2)
  float volume = 0.001f;        // displaced volume (m^3)
  Vec3 prev_fluid_vel;          // for added mass computation
};

struct CouplingForce {
  Vec3 force;    // total force (N)
  Vec3 torque;   // total torque (N*m), reserved for future use
};

struct FluidCoupling {

  // Query fluid state at a point from whichever solver is active.
  struct FluidState {
    Vec3 velocity;
    float density = kWaterDensity;
    float depth = 0;     // water depth at this location (from SWE)
    bool submerged = false;
  };

  static FluidState QueryOcean(const SpectralOceanSolver& ocean,
                                Vec3 pos) {
    FluidState s;
    float surface = ocean.SurfaceHeight(pos.x, pos.y);
    s.depth = surface - pos.z;
    s.submerged = (s.depth > 0);
    Vec3 vel = ocean.OrbitalVelocity(pos.x, pos.y, pos.z);
    s.velocity = vel;
    s.density = kWaterDensity;
    return s;
  }

  static FluidState QuerySWE(const ShallowWaterSolver& swe,
                               Vec3 pos) {
    FluidState s;
    float surface = swe.SurfaceHeight(pos.x, pos.y);
    s.depth = surface - pos.z;  // positive if underwater
    s.submerged = (s.depth > 0);
    s.velocity = swe.Velocity(pos.x, pos.y);
    s.density = kWaterDensity;
    return s;
  }

  static FluidState QuerySPH(const SPHSolver& sph, Vec3 pos) {
    FluidState s;
    s.density = sph.InterpolateDensity(pos);
    s.velocity = sph.InterpolateVelocity(pos);
    // Kernel deficiency-based free surface detection:
    // interior points have kernel integral ~ 1, surface points < 0.6.
    s.submerged = sph.IsSubmerged(pos);
    s.depth = s.submerged ? 1.0f : 0.0f;  // approximate
    return s;
  }

  static FluidState QueryLBM(const LBMSolver& lbm, Vec3 pos) {
    FluidState s;
    // Find nearest grid cell.
    float inv_dx = 1.0f / lbm.grid.dx;
    int gx = static_cast<int>((pos.x - lbm.grid.origin_x) * inv_dx);
    int gy = static_cast<int>((pos.y - lbm.grid.origin_y) * inv_dx);
    int gz = static_cast<int>((pos.z - lbm.grid.origin_z) * inv_dx);

    if (gx < 0 || gx >= static_cast<int>(lbm.grid.nx) ||
        gy < 0 || gy >= static_cast<int>(lbm.grid.ny) ||
        gz < 0 || gz >= static_cast<int>(lbm.grid.nz)) {
      return s;  // outside LBM domain
    }

    size_t c = lbm.grid.Idx(static_cast<uint32_t>(gx),
                              static_cast<uint32_t>(gy),
                              static_cast<uint32_t>(gz));
    if (lbm.grid.solid[c]) return s;

    float rho_lattice = lbm.grid.Density(c);
    Vec3 vel_lattice = lbm.grid.Velocity(c);

    s.density = rho_lattice * lbm.params.DensityScale();
    s.velocity = vel_lattice * lbm.params.VelocityScale();
    s.submerged = (rho_lattice > 0.5f);
    s.depth = s.submerged ? 1.0f : 0.0f;
    return s;
  }

  static FluidState QueryStokes(const StokesSolver& stokes, Vec3 pos) {
    FluidState s;
    s.velocity = stokes.Velocity(pos);
    s.density = stokes.params.density;
    // At cellular scales, everything is submerged.
    s.submerged = true;
    s.depth = 1.0f;
    return s;
  }

  // Compute forces on a body given fluid state and body coupling params.
  // char_length: characteristic body length (e.g. sphere diameter) for
  //   computing submersion fraction from depth.
  static CouplingForce ComputeForces(const FluidState& fluid,
                                      const BodyCoupling& body,
                                      Vec3 body_vel,
                                      float dt,
                                      float char_length = 0.0f) {
    CouplingForce result;
    if (!fluid.submerged) return result;

    // Submersion fraction: ramp from 0 to 1 over the body's characteristic
    // length. Prevents full-force impulse when barely touching water.
    float submersion = 1.0f;
    if (char_length > 0 && fluid.depth > 0 && fluid.depth < char_length) {
      submersion = fluid.depth / char_length;
      submersion = submersion * submersion;  // quadratic ramp: smoother onset
    }

    // 1. Buoyancy: F_b = rho * g * V * z_hat (scaled by submersion)
    float buoyancy = fluid.density * kGravity * body.volume * submersion;
    result.force.z += buoyancy;

    // 2. Drag: F_d = -0.5 * rho * Cd * A * |u_rel| * u_rel
    Vec3 u_rel = body_vel - fluid.velocity;
    float speed = u_rel.Length();
    if (speed > 1e-10f) {
      float drag_mag = 0.5f * fluid.density * body.drag_coeff *
                       body.cross_section * speed * submersion;
      result.force -= u_rel * drag_mag;
    }

    // 3. Added mass: F_am = Ca * rho * V * du_fluid/dt
    //    Clamp fluid acceleration to avoid spikes at wave fronts.
    if (dt > 0) {
      Vec3 fluid_acc = (fluid.velocity - body.prev_fluid_vel) / dt;
      // Clamp to 2g because wave fronts can produce unrealistic dv/dt
      // in shallow water due to grid-scale discontinuities.
      constexpr float kMaxFluidAcc = 2.0f * kGravity;
      float acc_mag = fluid_acc.Length();
      if (acc_mag > kMaxFluidAcc) {
        fluid_acc = fluid_acc * (kMaxFluidAcc / acc_mag);
      }
      float am = body.added_mass_coeff * fluid.density * body.volume * submersion;
      result.force += fluid_acc * am;
    }

    return result;
  }
};

}  // namespace mjwater
