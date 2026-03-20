// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// MuJoCo body-fluid coupling via the Morison equation.
//
// The Morison equation decomposes hydrodynamic force on a submerged body into
// three components:
//
//   F = F_drag + F_inertia + F_buoyancy
//
//   F_drag    = 0.5 * rho * Cd * A * |u_rel| * u_rel   (quadratic drag)
//   F_inertia = rho * Cm * V * du/dt                    (fluid acceleration)
//   F_buoyancy = rho * g * V_submerged                  (Archimedes)
//
// Where:
//   Cd = drag coefficient (depends on Reynolds number Re = |u|*D/nu)
//   Cm = inertia coefficient = 1 + Ca  (Ca = added mass coefficient)
//   A  = projected frontal area
//   V  = displaced volume
//
// The Keulegan-Carpenter number KC = U*T/D determines which term dominates:
//   KC < 5:  inertia-dominated (standing waves, slow oscillation)
//   KC > 20: drag-dominated (steady current, fast flow)
//   Between: both terms matter (most engineering wave conditions)
//
// Drag coefficient regimes for a smooth sphere (Clift et al. 1978):
//   Re < 1:      Stokes regime, Cd = 24/Re (viscous drag dominates)
//   1 < Re < 1e3: transitional, Cd = 24/Re * (1 + 0.15*Re^0.687)
//   1e3 < Re < 2e5: Newton regime, Cd ~ 0.44 (turbulent wake)
//   Re > 2e5:    drag crisis, Cd drops to ~0.1 (turbulent boundary layer)
//
// Buoyancy uses a smoothstep submersion model: the submerged volume fraction
// is a smooth S-curve from 0 (fully above water) to 1 (fully below), avoiding
// discontinuous force jumps as bodies cross the free surface.
//
// References:
//   Morison, J.R. et al. (1950) "The Force Exerted by Surface Waves on Piles"
//   Clift, R., Grace, J.R., Weber, M.E. (1978) "Bubbles, Drops, and Particles"
//   Sarpkaya, T. (2010) "Wave Forces on Offshore Structures" (Cambridge)
//   DNV-RP-C205 (2021) "Environmental Conditions and Environmental Loads"
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

// Directional added mass coefficients along body-local principal axes.
// Real bodies have different inertial resistance along different axes.
// A long cylinder has Ca_axial ~ 0 but Ca_transverse = 1.0 (Lamb 1932).
// A sphere has Ca = 0.5 in all directions (potential flow exact result).
//
// Reference: Newman, J.N. (1977) "Marine Hydrodynamics" Chapter 4.
//            Lamb, H. (1932) "Hydrodynamics" Section 339.
struct AddedMass3 {
  // Added mass coefficients along body-local principal axes.
  // x = forward/axial, y = lateral, z = vertical.
  float ca_x = 0.5f;
  float ca_y = 0.5f;
  float ca_z = 0.5f;

  // Body-local axes in world frame. Caller updates each frame from
  // the body's orientation quaternion.
  Vec3 axis_x = {1, 0, 0};
  Vec3 axis_y = {0, 1, 0};
  Vec3 axis_z = {0, 0, 1};

  // Convenience: prolate ellipsoid from fineness ratio (length / diameter).
  // Uses Lamb (1932) exact solution for the axial added mass coefficient.
  // Transverse coefficient is approximated as Ca_transverse ~ 1 - Ca_axial.
  static AddedMass3 Ellipsoid(float fineness) {
    AddedMass3 am;
    if (fineness <= 1.0f) return am;  // sphere or oblate: all 0.5
    float k = fineness;
    float e = std::sqrt(1.0f - 1.0f / (k * k));
    // Lamb's alpha_0 for prolate ellipsoid (axial direction).
    float alpha_0 = (2.0f * (1.0f - e * e) / (e * e * e)) *
                    (0.5f * std::log((1.0f + e) / (1.0f - e)) - e);
    am.ca_x = alpha_0 / (2.0f - alpha_0);
    // Beta_0 for transverse: related by constraint alpha_0 + 2*beta_0 = 2.
    float beta_0 = (2.0f - alpha_0) * 0.5f;
    am.ca_y = beta_0 / (2.0f - beta_0);
    am.ca_z = am.ca_y;
    return am;
  }

  // Convenience: long cylinder (Ca_axial ~ 0, Ca_transverse = 1.0).
  static AddedMass3 Cylinder() {
    return {0.0f, 1.0f, 1.0f, {1,0,0}, {0,1,0}, {0,0,1}};
  }
};

struct BodyCoupling {
  float drag_coeff = 1.0f;      // Cd (used when shape == kCustom)
  float added_mass_coeff = 0.5f; // Ca scalar (used when shape == kCustom and not directional)
  float cross_section = 0.01f;  // reference area (m^2)
  float volume = 0.001f;        // displaced volume (m^3)
  Vec3 prev_fluid_vel;          // for added mass computation

  // Body shape for automatic Re-dependent Cd/Cm lookup.
  // Default kCustom uses user-supplied drag_coeff/added_mass_coeff directly.
  BodyShape shape = BodyShape::kCustom;

  // Directional added mass. When enabled, replaces the scalar
  // added_mass_coeff with axis-dependent coefficients.
  bool use_directional_added_mass = false;
  AddedMass3 added_mass_3;

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
  // Regime boundaries (Clift et al. 1978, Table 5.2).
  static constexpr float kReStokes = 0.1f;       // below: creeping flow
  static constexpr float kReTransition = 1.0f;    // above: Schiller-Naumann
  static constexpr float kReNewton = 1000.0f;     // above: constant Cd plateau
  static constexpr float kReCritical = 2e5f;      // above: drag crisis
  static constexpr float kCdNewtonSphere = 0.44f; // Cd in Newton regime
  static constexpr float kCdPostCritical = 0.1f;  // Cd after drag crisis

  static float SphereDragCoeff(float Re) {
    if (Re < kReStokes) return 24.0f / kReStokes;
    if (Re < kReTransition) return 24.0f / Re;
    if (Re < kReNewton) return 24.0f / Re * (1.0f + 0.15f * std::pow(Re, 0.687f));
    if (Re < kReCritical) return kCdNewtonSphere;
    return kCdPostCritical;
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

      // Keulegan-Carpenter number: KC = U_max * T / D.
      // Estimate oscillation period T from fluid acceleration:
      //   T ~ 2*pi*|u| / |du/dt|  (period of the dominant oscillation).
      // When prev_fluid_vel is available and dt > 0, this gives a local
      // estimate of the wave period driving the flow past the body.
      // Falls back to potential flow value (Cm = 1.5 for sphere) when
      // acceleration is too small to estimate T reliably.
      float u_fluid = fluid.velocity.Length();
      if (dt > 0 && u_fluid > 1e-6f) {
        Vec3 fluid_acc = (fluid.velocity - body.prev_fluid_vel) / dt;
        float acc_mag = fluid_acc.Length();
        if (acc_mag > 1e-6f) {
          float T_approx = kTwoPi * u_fluid / acc_mag;
          float KC = u_fluid * T_approx / char_length;
          Cm = SphereInertiaCoeff(KC);
        }
        // else: steady flow (no acceleration), keep potential flow Cm
      }
    }

    // 2. Drag: Morison drag term.
    if (speed > 1e-10f) {
      float drag_mag = 0.5f * fluid.density * Cd *
                       body.cross_section * speed * submersion;
      result.force -= u_rel * drag_mag;
    }

    // 3. Inertia: Morison inertia term.
    // Scalar: F = Cm * rho * V * du_fluid/dt  (Cm = 1 + Ca).
    // Directional: decompose acceleration into body-frame axes and apply
    // per-axis Cm_i = 1 + Ca_i. This correctly models elongated bodies
    // where axial and transverse added mass differ substantially.
    if (dt > 0) {
      Vec3 fluid_acc = (fluid.velocity - body.prev_fluid_vel) / dt;
      // In steep/breaking waves, fluid acceleration can reach 3-5g at the free
      // surface.  Cap at 10g to reject numerical noise from LOD transitions
      // while preserving extreme wave physics.
      constexpr float kMaxFluidAcc = 10.0f * kGravity;
      float acc_mag = fluid_acc.Length();
      if (acc_mag > kMaxFluidAcc) {
        fluid_acc = fluid_acc * (kMaxFluidAcc / acc_mag);
      }

      float rhoVsub = fluid.density * body.volume * submersion;
      if (body.use_directional_added_mass) {
        const auto& am = body.added_mass_3;
        float ax = fluid_acc.Dot(am.axis_x);
        float ay = fluid_acc.Dot(am.axis_y);
        float az = fluid_acc.Dot(am.axis_z);
        result.force += am.axis_x * ((1.0f + am.ca_x) * ax * rhoVsub);
        result.force += am.axis_y * ((1.0f + am.ca_y) * ay * rhoVsub);
        result.force += am.axis_z * ((1.0f + am.ca_z) * az * rhoVsub);
      } else {
        float inertia = Cm * rhoVsub;
        result.force += fluid_acc * inertia;
      }
    }

    // 4. Waterline damping: extra vertical drag near the surface.
    // When partially submerged (0 < submersion < 1), the body pierces
    // the free surface. Real physics: wave-making resistance, spray
    // generation, and surface tension all resist vertical motion through
    // the waterline. Modeled as quadratic drag on vertical velocity,
    // scaled by the waterline proximity factor (peaks at half submersion).
    //
    // The 4*s*(1-s) profile is a beta(2,2) distribution normalized to
    // peak at s=0.5 with value 1.0. This is an empirical approximation
    // to the wave-making resistance coefficient of Wehausen (1973)
    // for bodies piercing a free surface at low Froude numbers.
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

// --- Strip sampling for distributed Morison loading ---
//
// The Morison equation (1950) was originally formulated per unit length
// of a slender cylinder. For a finite body in a non-uniform flow (waves),
// the correct application is to integrate along the body axis, sampling
// the fluid field at discrete stations. This captures:
//   - Different submersion along the body length
//   - Phase variation of orbital velocities across the body
//   - Torques from non-uniform loading (pitch/yaw moments)
//
// Reference: DNV-RP-C205 (2021) Section 6.2 "Strip Theory"

struct StripConfig {
  uint8_t num_samples = 5;  // stations along the body axis (3 to 10)
  float length = 0.0f;      // body length along its axis (m); 0 = disabled
  // Body-local forward axis in world frame. Caller updates each frame
  // from the body's orientation quaternion.
  Vec3 axis = {1, 0, 0};
};

// Integrate Morison forces along a body axis by sampling the fluid field
// at evenly spaced strip stations. Each station contributes force and
// torque (moment about body_center). The per-strip body parameters are
// the parent body's values divided by num_samples.
//
// query_fn: callable(Vec3 pos) -> FluidCoupling::FluidState. Pass a lambda
// wrapping WaterEngine::Query or a direct solver query.
template <typename QueryFn>
CouplingForce ComputeStripForces(
    QueryFn query_fn,
    const BodyCoupling& body,
    const StripConfig& strip,
    Vec3 body_center,
    Vec3 body_vel,
    Vec3 body_angular_vel,
    float dt,
    float char_length = 0.0f) {
  CouplingForce total;
  int N = std::max(2, static_cast<int>(strip.num_samples));
  float ds = strip.length / (N - 1);

  // Per-strip body: divide area and volume uniformly.
  BodyCoupling strip_body = body;
  strip_body.cross_section = body.cross_section / N;
  strip_body.volume = body.volume / N;

  for (int i = 0; i < N; ++i) {
    float t = (i - (N - 1) * 0.5f) * ds;
    Vec3 offset = strip.axis * t;
    Vec3 pos = body_center + offset;

    // Local velocity includes rotational contribution.
    Vec3 local_vel = body_vel + body_angular_vel.Cross(offset);

    auto fluid = query_fn(pos);

    // Store previous fluid vel on the strip body for inertia computation.
    // For strip sampling, we use the parent body's prev_fluid_vel as an
    // approximation (the alternative would require per-station history).
    strip_body.prev_fluid_vel = body.prev_fluid_vel;

    auto strip_force = FluidCoupling::ComputeForces(
        fluid, strip_body, local_vel, dt, char_length);

    total.force += strip_force.force;
    total.torque += offset.Cross(strip_force.force);
  }
  return total;
}

// --- Lifting surface model ---
//
// Thin airfoil theory with stall: any flat or curved surface moving
// through water at an angle of attack generates lift perpendicular to
// the flow. At small angles, CL = 2*pi*alpha (Kutta-Joukowski, exact
// for thin airfoils in potential flow). Beyond the stall angle, flow
// separates and CL drops.
//
// The Kirchhoff/Helmholtz separation function gives a smooth stall
// transition without a discontinuity at the stall angle.
//
// References:
//   Abbott, I.H. & Von Doenhoff, A.E. (1959) "Theory of Wing Sections"
//   Leishman, J.G. & Beddoes, T.S. (1989) "A Semi-Empirical Model for
//     Dynamic Stall" J. Am. Heli. Soc. 34(3)
//   Hoerner, S.F. (1965) "Fluid Dynamic Drag" (flat plate data)

struct LiftingSurface {
  float area = 0.01f;          // planform area (m^2)
  float aspect_ratio = 4.0f;   // span^2 / area
  float stall_angle = 0.15f;   // radians (~9 deg, flat plate in water)
  float cl_slope = 6.283f;     // dCL/dalpha = 2*pi (thin airfoil theory)
  float cd0 = 0.01f;           // zero-lift drag coefficient (skin friction)
  float oswald = 0.85f;        // Oswald span efficiency factor

  // Surface normal in world frame (perpendicular to the chord plane).
  // Caller updates each frame from body orientation.
  Vec3 normal = {0, 0, 1};

  // Position relative to body center (world frame, caller updates).
  Vec3 offset = {};

  // Kirchhoff separation function for smooth stall transition.
  // Returns 1.0 (fully attached) at alpha=0, decays past stall_angle.
  static float Separation(float alpha, float alpha_s) {
    float x = (std::abs(alpha) - alpha_s) / 0.1f;
    float f = 1.0f / (1.0f + std::exp(x));
    return f * f;
  }
};

// Compute lift and drag on a lifting surface immersed in a fluid.
// Returns force in world frame. Does not include buoyancy (that comes
// from ComputeForces on the parent body).
inline CouplingForce ComputeLiftDrag(
    const FluidCoupling::FluidState& fluid,
    const LiftingSurface& surface,
    Vec3 body_vel,
    float submersion = 1.0f) {
  CouplingForce result;
  if (!fluid.submerged || submersion < 0.01f) return result;

  // Flow velocity relative to the surface (approaching the surface).
  Vec3 u_rel = fluid.velocity - body_vel;
  float speed = u_rel.Length();
  if (speed < 1e-8f) return result;

  Vec3 u_hat = u_rel / speed;

  // Angle of attack: sine of the angle between flow direction and
  // the surface plane. Positive alpha = flow hits the "top" of the surface.
  float sin_alpha = u_hat.Dot(surface.normal);
  float alpha = std::asin(std::clamp(sin_alpha, -1.0f, 1.0f));

  // Lift coefficient with Kirchhoff stall model.
  float f = LiftingSurface::Separation(alpha, surface.stall_angle);
  float CL = surface.cl_slope * alpha * 0.5f * (1.0f + f);

  // Drag coefficient: zero-lift + induced drag.
  float CD = surface.cd0 +
             CL * CL / (kPi * surface.aspect_ratio * surface.oswald);

  // Dynamic pressure * area * submersion.
  float qA = 0.5f * fluid.density * speed * speed * surface.area * submersion;

  // Lift direction: perpendicular to flow, in the plane containing
  // the surface normal and the flow vector.
  Vec3 lift_dir = u_hat.Cross(surface.normal).Cross(u_hat);
  float lift_len = lift_dir.Length();
  if (lift_len > 1e-10f) {
    lift_dir = lift_dir / lift_len;
  } else {
    return result;  // flow parallel to surface, no lift
  }

  result.force = lift_dir * (CL * qA) - u_hat * (CD * qA);
  // Torque about body center from offset.
  result.torque = surface.offset.Cross(result.force);
  return result;
}

// --- Thruster / actuator disc model ---
//
// Models a propulsive element as an actuator disc that adds momentum to
// the flow passing through it. Thrust depends on the advance ratio:
// at bollard pull (stationary), losses are highest. At design speed,
// efficiency peaks. Beyond design speed, the propeller windmills.
//
// The efficiency curve is a simplified fit to Wageningen B-series
// open-water characteristics for a typical 4-blade propeller.
//
// References:
//   Carlton, J. (2012) "Marine Propellers and Propulsion" (Butterworth)
//   Bernitsas, M. et al. (1981) "KT, KQ and Efficiency Curves for the
//     Wageningen B-Series Propellers" Report 237, Dept of Naval Arch.

struct ThrusterModel {
  float max_thrust = 10.0f;    // maximum thrust at bollard pull (N)
  float diameter = 0.1f;       // propeller disc diameter (m)
  float command = 0.0f;        // current thrust command, [-1, 1]

  // Thrust direction in world frame (caller updates from body orientation).
  Vec3 direction = {1, 0, 0};
  // Position relative to body center (world frame, caller updates).
  Vec3 offset = {};

  // Open-water efficiency as a function of advance ratio J = V_adv / V_jet.
  // V_jet = sqrt(2 * max_thrust / (rho * A)) is the ideal induced velocity
  // at bollard pull (actuator disc theory).
  //
  // Simplified piecewise fit to B4-70 series:
  //   J=0:   eta=0.50  (bollard pull, ~50% momentum loss)
  //   J=0.7: eta=0.65  (peak efficiency)
  //   J>1.2: eta=0     (windmilling, no net thrust)
  static float Efficiency(float J) {
    J = std::abs(J);
    if (J < 0.7f)  return 0.50f + 0.214f * J;
    if (J < 1.2f)  return 0.65f - 1.3f * (J - 0.7f);
    return 0.0f;
  }
};

// Compute thrust force from a thruster immersed in a fluid.
// Returns force (along thrust direction) and torque (from offset).
inline CouplingForce ComputeThrustForce(
    const FluidCoupling::FluidState& fluid,
    const ThrusterModel& thruster,
    Vec3 body_vel) {
  CouplingForce result;
  if (!fluid.submerged) return result;

  // Advance velocity: component of flow along thrust axis approaching the disc.
  float V_adv = (body_vel - fluid.velocity).Dot(thruster.direction);

  // Ideal jet velocity at bollard pull (actuator disc theory).
  float disc_area = 0.25f * kPi * thruster.diameter * thruster.diameter;
  float V_jet = std::sqrt(std::max(0.0f,
      2.0f * std::abs(thruster.max_thrust) / (fluid.density * disc_area)));

  // Advance ratio.
  float J = (V_jet > 1e-10f) ? V_adv / V_jet : 0.0f;

  float eta = ThrusterModel::Efficiency(J);
  float T = thruster.command * thruster.max_thrust * eta;

  result.force = thruster.direction * T;
  result.torque = thruster.offset.Cross(result.force);
  return result;
}

}  // namespace mjwater
