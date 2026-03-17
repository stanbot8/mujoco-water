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
#include <vector>

#include "mjwater/types.h"
#include "mjwater/spectral_ocean.h"
#include "mjwater/shallow_water.h"
#include "mjwater/sph.h"
#include "mjwater/lbm.h"
#include "mjwater/stokes.h"
#include "mjwater/lod_manager.h"

namespace mjwater {

// Body shape for automatic Re-dependent Cd/Cm lookup.
enum class BodyShape : uint8_t {
  kSphere = 0,
  kCylinder = 1,
  kCustom = 2,  // use user-supplied Cd/Ca directly
};

struct BodyCoupling {
  float drag_coeff = 1.0f;      // Cd (used when shape == kCustom)
  float added_mass_coeff = 0.5f; // Ca (used when shape == kCustom)
  float cross_section = 0.01f;  // reference area (m^2)
  float volume = 0.001f;        // displaced volume (m^3)
  Vec3 prev_fluid_vel;          // for added mass computation

  // Body shape for automatic Re-dependent Cd/Cm lookup.
  // Default kCustom uses user-supplied drag_coeff/added_mass_coeff directly.
  BodyShape shape = BodyShape::kCustom;

  // Two-way coupling (body creates waves).
  bool two_way_enabled = false;  // opt-in (default off for backward compat)
  float body_radius = 0.1f;     // effective radius for SWE cell coverage
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
    uint32_t cx, cy, cz;
    if (!lbm.grid.WorldToGridCell(pos.x, pos.y, pos.z, cx, cy, cz))
      return s;

    size_t c = lbm.grid.Idx(cx, cy, cz);
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

  // Reynolds-dependent drag coefficient for a smooth sphere.
  // Piecewise fit to experimental data (Schlichting, Clift et al.).
  //   Re < 1:      Stokes regime, Cd = 24/Re
  //   1 < Re < 1e3: transitional, Schiller-Naumann correlation
  //   1e3 < Re < 2e5: Newton regime, Cd ~ 0.44
  //   Re > 2e5:    post-critical (drag crisis), Cd ~ 0.1
  //
  // Reference: DNV-RP-C205 (2021) Table 6-4, Clift et al. (1978).
  static float SphereDragCoeff(float Re) {
    if (Re < 0.1f) return 240.0f;  // cap at very low Re
    if (Re < 1.0f) return 24.0f / Re;
    if (Re < 1000.0f) return 24.0f / Re * (1.0f + 0.15f * std::pow(Re, 0.687f));
    if (Re < 2e5f) return 0.44f;
    return 0.1f;  // post drag-crisis
  }

  // Reynolds-dependent drag coefficient for a smooth cylinder (2D cross-flow).
  // Reference: DNV-RP-C205 (2021) Table 6-3, Zdravkovich (1997).
  static float CylinderDragCoeff(float Re) {
    if (Re < 0.1f) return 100.0f;
    if (Re < 1.0f) return 10.0f / std::sqrt(Re);
    if (Re < 1e3f) return 1.0f + 10.0f / std::pow(Re, 2.0f / 3.0f);
    if (Re < 2e5f) return 1.2f;
    if (Re < 5e5f) return 0.3f;  // drag crisis
    return 0.6f;  // post-critical
  }

  // Inertia coefficient Cm = 1 + Ca for oscillatory flow.
  // Keulegan-Carpenter number KC = u_max * T / D governs the ratio of
  // drag to inertia dominance. At low KC, inertia dominates (Cm -> 2.0
  // for sphere). At high KC, drag dominates and Cm decreases.
  //
  // Reference: Sarpkaya (1976), DNV-RP-C205 Section 6.7.
  static float SphereInertiaCoeff(float KC) {
    // Sphere: Ca_potential = 0.5 (irrotational), Cm = 1 + Ca = 1.5.
    // At low KC, Cm increases toward potential flow value.
    // At high KC, Cm decreases due to separation.
    if (KC < 3.0f) return 1.5f;
    if (KC < 15.0f) return 1.5f - 0.033f * (KC - 3.0f);
    return 1.1f;
  }

  // Compute forces on a body given fluid state and body coupling params.
  // char_length: characteristic body length (e.g. sphere diameter) for
  //   computing submersion fraction from depth.
  //
  // Implements the Morison equation (Morison et al. 1950):
  //   F = F_buoyancy + F_drag + F_inertia
  //   F_drag   = 0.5 * rho * Cd(Re) * A * |u_rel| * u_rel
  //   F_inertia = Cm(KC) * rho * V * du_fluid/dt
  //
  // When body.shape == kCustom, uses body.drag_coeff and body.added_mass_coeff
  // directly (backward compatible).
  static CouplingForce ComputeForces(const FluidState& fluid,
                                      const BodyCoupling& body,
                                      Vec3 body_vel,
                                      float dt,
                                      float char_length = 0.0f) {
    CouplingForce result;
    if (!fluid.submerged) return result;

    // Hydrostatic submersion: fraction of body volume below the waterline.
    // Uses smoothstep (hermite) interpolation of depth/char_length.
    // At t=0 (surface): submersion=0, at t=1 (fully submerged): submersion=1.
    // Smoothstep gives correct behavior for convex bodies:
    //   - Linear region near t=0.5 (matches cylinder exactly)
    //   - S-curve onset/offset (close to sphere volume integral)
    //   - Continuous first derivative (no force discontinuity at waterline)
    float submersion = 1.0f;
    float submerged_volume_fraction = 1.0f;
    if (char_length > 0 && fluid.depth > 0) {
      float t = std::clamp(fluid.depth / char_length, 0.0f, 1.0f);
      // Smoothstep: 3t^2 - 2t^3 (C1 continuous, matches sphere within 5%)
      submersion = t * t * (3.0f - 2.0f * t);
      submerged_volume_fraction = submersion;
    }

    // 1. Buoyancy: F_b = rho * g * V_submerged * z_hat
    // Uses submerged volume fraction for accurate partial submersion.
    float buoyancy = fluid.density * kGravity * body.volume * submerged_volume_fraction;
    result.force.z += buoyancy;

    // Froude-Krylov force (lateral hydrostatic pressure gradient).
    // F_FK = rho * V_sub * a_fluid, where a_fluid is the undisturbed
    // fluid acceleration. This is already captured by the inertia term
    // (Cm includes 1.0 for the Froude-Krylov component). No separate
    // lateral force needed here; the Morison inertia term handles it.

    // Relative velocity for drag.
    Vec3 u_rel = body_vel - fluid.velocity;
    float speed = u_rel.Length();

    // Compute Cd and Cm (either from correlations or user-supplied).
    float Cd = body.drag_coeff;
    float Cm = 1.0f + body.added_mass_coeff;  // Cm = 1 + Ca

    if (body.shape != BodyShape::kCustom && char_length > 0) {
      // Reynolds number: Re = |u_rel| * D / nu
      float Re = speed * char_length / kWaterViscosity;

      if (body.shape == BodyShape::kSphere) {
        Cd = SphereDragCoeff(Re);
      } else if (body.shape == BodyShape::kCylinder) {
        Cd = CylinderDragCoeff(Re);
      }

      // Keulegan-Carpenter number for inertia coefficient.
      // KC = u_max * T / D. Approximate T from wave period if available,
      // otherwise use speed * dt as a proxy.
      float u_fluid = fluid.velocity.Length();
      float T_approx = (u_fluid > 1e-6f) ? char_length / u_fluid : 1.0f;
      float KC = u_fluid * T_approx / char_length;
      Cm = SphereInertiaCoeff(KC);
    }

    // 2. Drag: Morison drag term.
    if (speed > 1e-10f) {
      float drag_mag = 0.5f * fluid.density * Cd *
                       body.cross_section * speed * submersion;
      result.force -= u_rel * drag_mag;
    }

    // 3. Inertia: Morison inertia term (Cm * rho * V * du_fluid/dt).
    if (dt > 0) {
      Vec3 fluid_acc = (fluid.velocity - body.prev_fluid_vel) / dt;
      constexpr float kMaxFluidAcc = 2.0f * kGravity;
      float acc_mag = fluid_acc.Length();
      if (acc_mag > kMaxFluidAcc) {
        fluid_acc = fluid_acc * (kMaxFluidAcc / acc_mag);
      }
      float inertia = Cm * fluid.density * body.volume * submersion;
      result.force += fluid_acc * inertia;
    }

    // 4. Waterline damping: extra vertical drag near the surface.
    // When partially submerged (0 < submersion < 1), the body pierces
    // the free surface. Real physics: wave-making resistance, spray
    // generation, and surface tension all resist vertical motion through
    // the waterline. Modeled as quadratic drag on vertical velocity,
    // scaled by the waterline proximity factor (peaks at half submersion).
    if (submersion > 0.01f && submersion < 0.99f) {
      float waterline_factor = 4.0f * submersion * (1.0f - submersion);
      float vz = body_vel.z;
      float waterline_drag = 0.5f * fluid.density * body.cross_section *
                              waterline_factor * std::abs(vz) * vz;
      result.force.z -= waterline_drag;
    }

    return result;
  }

  // --- Two-way coupling: body creates waves in SWE ---

  struct WaveSourceTerms {
    std::vector<uint32_t> cell_x;
    std::vector<uint32_t> cell_y;
    std::vector<float> dh;   // depth rate (m/s) per cell
    std::vector<float> dhu;  // x-momentum rate per cell
    std::vector<float> dhv;  // y-momentum rate per cell
    size_t count = 0;
  };

  // Compute SWE source terms from a body displacing water.
  // Body at (pos) with velocity (vel) and radius pushes water aside,
  // creating waves. Returns per-cell source rates.
  static WaveSourceTerms ComputeWaveSources(
      const ShallowWaterSolver& swe,
      const BodyCoupling& body,
      Vec3 pos, Vec3 vel) {
    WaveSourceTerms src;

    float speed = vel.Length();
    if (speed < 1e-6f) return src;

    float dx = swe.grid.dx;
    float cell_area = dx * dx;
    float r = body.body_radius;

    // Grid coordinates of body center.
    uint32_t cx, cy;
    if (!swe.grid.WorldToGridCell(pos.x, pos.y, cx, cy)) return src;
    int cell_r = static_cast<int>(std::ceil(r / dx));

    // Check submersion: body must be at least partially in water.
    float surface = swe.SurfaceHeight(pos.x, pos.y);
    if (pos.z - r > surface) return src;  // above water

    // Submersion fraction for scaling.
    float submersion = std::clamp((surface - (pos.z - r)) / (2.0f * r), 0.0f, 1.0f);

    const auto& h = swe.grid.channels[swe.ch_h].data;

    for (int dy = -cell_r; dy <= cell_r; ++dy) {
      for (int ddx = -cell_r; ddx <= cell_r; ++ddx) {
        int gx = static_cast<int>(cx) + ddx;
        int gy = static_cast<int>(cy) + dy;
        if (gx < 0 || gx >= static_cast<int>(swe.grid.nx)) continue;
        if (gy < 0 || gy >= static_cast<int>(swe.grid.ny)) continue;

        // Distance from cell center to body center (in world coords).
        float cell_wx, cell_wy;
        swe.grid.GridToWorld(static_cast<uint32_t>(gx),
                              static_cast<uint32_t>(gy), cell_wx, cell_wy);
        float dist_x = cell_wx - pos.x;
        float dist_y = cell_wy - pos.y;
        float dist = std::sqrt(dist_x * dist_x + dist_y * dist_y);
        if (dist > r) continue;

        // Submerged cross-sectional area of body column in this cell.
        // Cylindrical projection: fraction of cell covered by body circle.
        float coverage = std::max(0.0f, 1.0f - dist / r);
        coverage *= coverage;  // smoother falloff
        float area = cell_area * coverage * submersion;

        // Vertical displacement: body moving down pushes water out.
        float dh_vert = -vel.z * area / cell_area;

        // Horizontal displacement: leading face pushes water ahead.
        float face_dot = 0.0f;
        if (dist > 1e-6f) {
          float nx = dist_x / dist;
          float ny = dist_y / dist;
          face_dot = vel.x * nx + vel.y * ny;
        }
        float dh_horiz = face_dot * area / cell_area;

        float total_dh = dh_vert + dh_horiz;

        // Clamp: source cannot drain more than 50% of cell depth per second.
        size_t idx = swe.grid.Idx(static_cast<uint32_t>(gx),
                                    static_cast<uint32_t>(gy));
        float max_drain = 0.5f * h[idx];  // per second (dt applied later)
        total_dh = std::max(total_dh, -max_drain);

        src.cell_x.push_back(static_cast<uint32_t>(gx));
        src.cell_y.push_back(static_cast<uint32_t>(gy));
        src.dh.push_back(total_dh);
        src.dhu.push_back(vel.x * coverage * submersion);
        src.dhv.push_back(vel.y * coverage * submersion);
      }
    }
    src.count = src.cell_x.size();
    return src;
  }

  // Compute reaction force on the body from wave generation.
  // Newton's third law: momentum injected into water = negative force on body.
  static Vec3 WaveReactionForce(const WaveSourceTerms& src,
                                  float dx, float dt) {
    if (src.count == 0 || dt <= 0) return {};
    float cell_area = dx * dx;
    Vec3 reaction{};
    for (size_t i = 0; i < src.count; ++i) {
      // Momentum rate injected: rho * dhu * cell_area (kg*m/s^2 = N)
      reaction.x -= kWaterDensity * src.dhu[i] * cell_area;
      reaction.y -= kWaterDensity * src.dhv[i] * cell_area;
      reaction.z -= kWaterDensity * src.dh[i] * cell_area * kGravity * dt;
    }
    return reaction;
  }
};

}  // namespace mjwater
