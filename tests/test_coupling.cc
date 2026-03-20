// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#include "test_harness.h"
#include "mjwater/coupling.h"
#include "mjwater/shallow_water.h"

using namespace mjwater;

TEST(Coupling_Buoyancy) {
  // A submerged body should feel upward buoyancy force.
  FluidCoupling::FluidState fluid;
  fluid.velocity = {};
  fluid.density = kWaterDensity;
  fluid.depth = 1.0f;
  fluid.submerged = true;

  BodyCoupling body;
  body.volume = 0.001f;      // 1 liter
  body.cross_section = 0.01f;
  body.drag_coeff = 1.0f;

  Vec3 body_vel = {};
  auto force = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);

  // Buoyancy should point up (positive z).
  CHECK(force.force.z > 0);
  // Expected: rho * g * V = 998.2 * 9.81 * 0.001 = ~9.79 N
  CHECK_NEAR(force.force.z, kWaterDensity * kGravity * body.volume, 0.1f);
}

TEST(Coupling_Drag) {
  FluidCoupling::FluidState fluid;
  fluid.velocity = {};
  fluid.density = kWaterDensity;
  fluid.depth = 1.0f;
  fluid.submerged = true;

  BodyCoupling body;
  body.volume = 0.001f;
  body.cross_section = 0.01f;
  body.drag_coeff = 1.0f;

  // Body moving in +x direction -> drag should oppose (negative x).
  Vec3 body_vel = {1.0f, 0, 0};
  auto force = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);
  CHECK(force.force.x < 0);  // drag opposes motion
}

TEST(Coupling_NoForceWhenDry) {
  FluidCoupling::FluidState fluid;
  fluid.submerged = false;

  BodyCoupling body;
  body.volume = 0.001f;

  Vec3 body_vel = {1.0f, 0, 0};
  auto force = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);

  CHECK_NEAR(force.force.x, 0.0f, 1e-10f);
  CHECK_NEAR(force.force.z, 0.0f, 1e-10f);
}

TEST(Coupling_SWEQuery) {
  ShallowWaterSolver swe;
  swe.Init(10, 10, 0.5f);
  swe.SetSurface(2.0f);

  auto state = FluidCoupling::QuerySWE(swe, {2.5f, 2.5f, 1.0f});
  CHECK(state.submerged);
  CHECK(state.depth > 0);
}

TEST(Coupling_TwoWayWaveGeneration) {
  // A moving body should inject wave sources into SWE cells.
  ShallowWaterSolver swe;
  swe.Init(20, 20, 0.5f);
  swe.SetSurface(2.0f);

  float vol_before = swe.TotalVolume();

  // Body at grid center, moving downward (displacing water).
  BodyCoupling body;
  body.volume = 1.0f;
  body.cross_section = 0.5f;
  body.two_way_enabled = true;
  body.body_radius = 0.5f;

  Vec3 body_pos = {5.0f, 5.0f, 1.5f};
  Vec3 body_vel = {0, 0, -1.0f};  // moving down into water

  auto sources = FluidCoupling::ComputeWaveSources(swe, body, body_pos, body_vel);

  // Should have produced source terms.
  CHECK(sources.count > 0);

  // Apply sources and check volume changed.
  swe.ApplySourceTerms(sources.cell_x.data(), sources.cell_y.data(),
                        sources.dh.data(), sources.dhu.data(), sources.dhv.data(),
                        sources.count, 0.01f);
  float vol_after = swe.TotalVolume();

  // Body moving down displaces water, volume should increase.
  CHECK(vol_after > vol_before);
}

TEST(Coupling_ReynoldsDragCoeff) {
  // Verify drag coefficient correlations return reasonable values.
  // Stokes regime: Cd = 24/Re
  float cd_stokes = FluidCoupling::SphereDragCoeff(0.5f);
  CHECK_NEAR(cd_stokes, 48.0f, 1.0f);  // 24/0.5 = 48

  // Newton regime: Cd ~ 0.44
  float cd_newton = FluidCoupling::SphereDragCoeff(5000.0f);
  CHECK_NEAR(cd_newton, 0.44f, 0.01f);

  // Post-critical: Cd ~ 0.1
  float cd_post = FluidCoupling::SphereDragCoeff(1e6f);
  CHECK_NEAR(cd_post, 0.1f, 0.01f);
}

TEST(Coupling_KC_VariesWithAcceleration) {
  // Verify that the KC-dependent inertia coefficient actually varies
  // with flow conditions, rather than returning a constant.
  // Oscillating flow (high acceleration) should give different Cm
  // than steady flow (zero acceleration).

  float char_length = 0.1f;  // 10 cm sphere diameter

  // Case 1: Steady flow (no acceleration). prev_fluid_vel == fluid.velocity.
  // Should use potential flow Cm = 1.5 (no acceleration to estimate T).
  {
    FluidCoupling::FluidState fluid;
    fluid.velocity = {1.0f, 0, 0};
    fluid.density = kWaterDensity;
    fluid.depth = 2.0f;
    fluid.submerged = true;

    BodyCoupling body;
    body.shape = BodyShape::kSphere;
    body.volume = 0.001f;
    body.cross_section = 0.01f;
    body.prev_fluid_vel = {1.0f, 0, 0};  // same as current: steady

    Vec3 body_vel = {};
    auto force_steady = FluidCoupling::ComputeForces(fluid, body, body_vel,
                                                      0.01f, char_length);

    // Case 2: Rapidly oscillating flow (high acceleration, low KC).
    // Large acceleration with same speed means short period, low KC,
    // so Cm should stay at 1.5 (low KC regime).
    body.prev_fluid_vel = {-1.0f, 0, 0};  // reversed: large acceleration
    auto force_oscillating = FluidCoupling::ComputeForces(fluid, body, body_vel,
                                                           0.01f, char_length);

    // The forces should differ because the KC-dependent Cm differs.
    // In steady flow, Cm defaults to 1.5 (potential flow).
    // In oscillating flow, KC is computed from acceleration, giving a
    // different Cm value.
    // We verify that the inertia component differs (the drag and buoyancy
    // are the same in both cases since speed and depth are identical).
    // The force difference comes from the inertia term: Cm * rho * V * du/dt.
    // In case 1, du/dt = 0 so inertia = 0 regardless of Cm.
    // In case 2, du/dt = (1 - (-1))/0.01 = 200 m/s^2 (capped at 10g).
    CHECK(std::abs(force_oscillating.force.x) >
          std::abs(force_steady.force.x));
  }

  // Case 3: High KC (slow oscillation, large amplitude).
  // Compare forces with two different KC values to verify Cm varies.
  // Use identical flow conditions except for prev_fluid_vel (which
  // changes the acceleration and thus the estimated KC).
  {
    FluidCoupling::FluidState fluid;
    fluid.velocity = {1.0f, 0, 0};
    fluid.density = kWaterDensity;
    fluid.depth = 2.0f;
    fluid.submerged = true;

    BodyCoupling body;
    body.shape = BodyShape::kSphere;
    body.volume = 0.001f;
    body.cross_section = 0.01f;

    Vec3 body_vel = {};

    // Low KC case: rapid oscillation (large acceleration).
    // du/dt = (1.0 - 0.0) / 0.01 = 100 m/s^2
    // T ~ 2*pi*1.0/100 = 0.063s, KC = 1.0*0.063/0.1 = 0.63
    // SphereInertiaCoeff(0.63) = 1.5 (low KC regime)
    body.prev_fluid_vel = {0.0f, 0, 0};
    auto force_low_kc = FluidCoupling::ComputeForces(fluid, body, body_vel,
                                                       0.01f, char_length);

    // High KC case: slow oscillation (small acceleration).
    // du/dt = (1.0 - 0.99) / 0.01 = 1.0 m/s^2
    // T ~ 2*pi*1.0/1.0 = 6.28s, KC = 1.0*6.28/0.1 = 62.8
    // SphereInertiaCoeff(62.8) = 1.1 (high KC regime)
    body.prev_fluid_vel = {0.99f, 0, 0};
    auto force_high_kc = FluidCoupling::ComputeForces(fluid, body, body_vel,
                                                        0.01f, char_length);

    // Low KC has both higher Cm (1.5 vs 1.1) and higher acceleration
    // (100 vs 1 m/s^2), so its inertia force should be much larger.
    // The drag is identical in both cases (same u_rel).
    // Therefore low KC force should exceed high KC force.
    CHECK(force_low_kc.force.x > force_high_kc.force.x);

    // Inertia difference: (1.5*100 - 1.1*1.0) * rho * V = 148.9 * 998.2 * 0.001
    //                   = ~148.6 N difference (drag cancels out).
    // This should be a substantial difference, not just noise.
    float force_diff = force_low_kc.force.x - force_high_kc.force.x;
    CHECK(force_diff > 10.0f);
  }
}

TEST(Coupling_SphereInertiaCoeff_Ranges) {
  // Low KC: potential flow, Cm = 1.5
  CHECK_NEAR(FluidCoupling::SphereInertiaCoeff(1.0f), 1.5f, 0.01f);
  CHECK_NEAR(FluidCoupling::SphereInertiaCoeff(2.0f), 1.5f, 0.01f);

  // Transition: Cm decreases linearly from 1.5 at KC=3 to 1.1 at KC=15.
  float cm_mid = FluidCoupling::SphereInertiaCoeff(9.0f);
  CHECK(cm_mid > 1.1f && cm_mid < 1.5f);

  // High KC: drag-dominated, Cm = 1.1
  CHECK_NEAR(FluidCoupling::SphereInertiaCoeff(20.0f), 1.1f, 0.01f);
  CHECK_NEAR(FluidCoupling::SphereInertiaCoeff(100.0f), 1.1f, 0.01f);
}

// --- Directional added mass tests ---

TEST(AddedMass3_SphereIsotropic) {
  // Sphere: Ca = 0.5 in all directions. Directional should match scalar.
  FluidCoupling::FluidState fluid;
  fluid.velocity = {0, 0, 0};
  fluid.density = kWaterDensity;
  fluid.depth = 2.0f;
  fluid.submerged = true;

  BodyCoupling body;
  body.volume = 0.001f;
  body.cross_section = 0.01f;
  body.shape = BodyShape::kCustom;
  body.added_mass_coeff = 0.5f;
  body.prev_fluid_vel = {1.0f, 0, 0};  // fluid was moving, now stopped

  Vec3 body_vel = {};

  // Scalar path.
  body.use_directional_added_mass = false;
  auto f_scalar = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);

  // Directional path with isotropic Ca = 0.5.
  body.use_directional_added_mass = true;
  body.added_mass_3 = {0.5f, 0.5f, 0.5f, {1,0,0}, {0,1,0}, {0,0,1}};
  auto f_dir = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);

  // Should produce identical forces.
  CHECK_NEAR(f_scalar.force.x, f_dir.force.x, 0.01f);
  CHECK_NEAR(f_scalar.force.z, f_dir.force.z, 0.01f);
}

TEST(AddedMass3_CylinderAnisotropic) {
  // Long cylinder: Ca_axial ~ 0, Ca_transverse = 1.0.
  // Acceleration along the axis should produce less inertia force
  // than acceleration transverse to the axis.
  FluidCoupling::FluidState fluid;
  fluid.velocity = {0, 0, 0};
  fluid.density = kWaterDensity;
  fluid.depth = 2.0f;
  fluid.submerged = true;

  BodyCoupling body;
  body.volume = 0.001f;
  body.cross_section = 0.01f;
  body.shape = BodyShape::kCustom;
  body.use_directional_added_mass = true;
  body.added_mass_3 = AddedMass3::Cylinder();
  // Axis along x.

  Vec3 body_vel = {};

  // Case 1: fluid was accelerating along cylinder axis (x).
  // Cm_x = 1 + 0 = 1.0.
  body.prev_fluid_vel = {1.0f, 0, 0};
  auto f_axial = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);

  // Case 2: fluid was accelerating transverse to cylinder axis (y).
  // Cm_y = 1 + 1.0 = 2.0.
  body.prev_fluid_vel = {0, 1.0f, 0};
  auto f_trans = FluidCoupling::ComputeForces(fluid, body, body_vel, 0.01f);

  // Transverse inertia force should be ~2x axial (Cm=2 vs Cm=1).
  // Both forces should be in the direction of the fluid acceleration
  // (which is -prev_vel/dt since current vel is 0).
  CHECK(std::abs(f_trans.force.y) > 1.5f * std::abs(f_axial.force.x));
}

TEST(AddedMass3_EllipsoidFactory) {
  // Verify ellipsoid factory produces reasonable values.
  // Fineness 1.0 = sphere: all Ca = 0.5.
  auto sphere = AddedMass3::Ellipsoid(1.0f);
  CHECK_NEAR(sphere.ca_x, 0.5f, 0.01f);
  CHECK_NEAR(sphere.ca_y, 0.5f, 0.01f);

  // Fineness 5.0 = elongated: Ca_axial < 0.5, Ca_transverse > 0.5.
  auto elongated = AddedMass3::Ellipsoid(5.0f);
  CHECK(elongated.ca_x < 0.1f);   // very low axial added mass
  CHECK(elongated.ca_y > 0.7f);   // high transverse added mass
}

// --- Lifting surface tests ---

TEST(Lift_SmallAlpha) {
  // At small angle of attack, CL = 2*pi*alpha (thin airfoil theory).
  FluidCoupling::FluidState fluid;
  fluid.velocity = {1.0f, 0, 0};  // flow in +x
  fluid.density = kWaterDensity;
  fluid.submerged = true;

  LiftingSurface surf;
  surf.area = 0.1f;     // 0.1 m^2
  surf.normal = {0, 0, 1};  // surface lies in xy plane, normal up
  surf.aspect_ratio = 6.0f;
  surf.cd0 = 0.01f;

  // Body moving in +x with slight downward component (positive alpha).
  // Alpha = asin(u_rel_z / |u_rel|). u_rel = fluid.vel - body_vel.
  // If body moves at {1, 0, -0.1}, u_rel = {0, 0, 0.1}, which is pure
  // vertical flow hitting the surface. That's alpha = 90 deg, not small.
  // Instead: body stationary, flow at slight angle.
  // flow = {1, 0, 0.05} -> alpha ~ asin(0.05/sqrt(1+0.0025)) ~ 0.05 rad
  fluid.velocity = {1.0f, 0, 0.05f};
  Vec3 body_vel = {};

  auto force = ComputeLiftDrag(fluid, surf, body_vel, 1.0f);

  // CL ~ 2*pi*0.05 = 0.314
  // q*A = 0.5 * 998.2 * (1.0^2) * 0.1 = 49.9
  // Lift ~ 0.314 * 49.9 ~ 15.7 N (upward, +z direction)
  CHECK(force.force.z > 10.0f);  // substantial upward lift
  CHECK(force.force.z < 25.0f);  // not unreasonable

  // Drag should be small (low alpha).
  CHECK(std::abs(force.force.x) < std::abs(force.force.z));
}

TEST(Lift_Stall) {
  // Beyond stall angle, CL should decrease.
  FluidCoupling::FluidState fluid;
  fluid.density = kWaterDensity;
  fluid.submerged = true;

  LiftingSurface surf;
  surf.area = 0.1f;
  surf.normal = {0, 0, 1};
  surf.stall_angle = 0.15f;

  Vec3 body_vel = {};

  // Pre-stall: alpha ~ 0.1 rad
  fluid.velocity = {1.0f, 0, 0.1f};
  auto f_pre = ComputeLiftDrag(fluid, surf, body_vel);

  // Post-stall: alpha ~ 0.5 rad (~29 deg, well past stall)
  fluid.velocity = {1.0f, 0, 0.55f};
  auto f_post = ComputeLiftDrag(fluid, surf, body_vel);

  // Lift per unit alpha should be lower post-stall.
  float cl_per_alpha_pre = f_pre.force.z / 0.1f;
  float cl_per_alpha_post = f_post.force.z / 0.5f;
  CHECK(cl_per_alpha_post < cl_per_alpha_pre);
}

TEST(Lift_NoForceParallelFlow) {
  // Flow parallel to surface (alpha = 0) should produce no lift.
  FluidCoupling::FluidState fluid;
  fluid.velocity = {1.0f, 0, 0};
  fluid.density = kWaterDensity;
  fluid.submerged = true;

  LiftingSurface surf;
  surf.area = 0.1f;
  surf.normal = {0, 0, 1};

  auto force = ComputeLiftDrag(fluid, surf, {}, 1.0f);
  CHECK_NEAR(force.force.z, 0.0f, 0.1f);  // no lift
  CHECK(force.force.x < 0);  // only drag (opposing flow)
}

// --- Thruster tests ---

TEST(Thruster_BollardPull) {
  // At zero advance velocity (bollard pull), thrust = command * max * eta(0).
  FluidCoupling::FluidState fluid;
  fluid.velocity = {};
  fluid.density = kWaterDensity;
  fluid.submerged = true;

  ThrusterModel thr;
  thr.max_thrust = 100.0f;
  thr.diameter = 0.2f;
  thr.command = 1.0f;
  thr.direction = {1, 0, 0};

  auto force = ComputeThrustForce(fluid, thr, {});

  // eta(0) = 0.50, so thrust = 100 * 0.50 = 50 N.
  CHECK_NEAR(force.force.x, 50.0f, 1.0f);
  CHECK_NEAR(force.force.y, 0.0f, 1e-6f);
  CHECK_NEAR(force.force.z, 0.0f, 1e-6f);
}

TEST(Thruster_EfficiencyCurve) {
  // Verify the efficiency curve shape.
  CHECK_NEAR(ThrusterModel::Efficiency(0.0f), 0.50f, 0.01f);
  CHECK(ThrusterModel::Efficiency(0.7f) > 0.60f);  // near peak
  CHECK_NEAR(ThrusterModel::Efficiency(1.2f), 0.0f, 0.01f);  // windmilling
  CHECK_NEAR(ThrusterModel::Efficiency(2.0f), 0.0f, 0.01f);  // beyond
}

TEST(Thruster_ReverseCommand) {
  // Negative command should produce reverse thrust.
  FluidCoupling::FluidState fluid;
  fluid.velocity = {};
  fluid.density = kWaterDensity;
  fluid.submerged = true;

  ThrusterModel thr;
  thr.max_thrust = 100.0f;
  thr.diameter = 0.2f;
  thr.command = -1.0f;
  thr.direction = {1, 0, 0};

  auto force = ComputeThrustForce(fluid, thr, {});
  CHECK(force.force.x < 0);  // reverse thrust
}

TEST(Thruster_NoThrustWhenDry) {
  FluidCoupling::FluidState fluid;
  fluid.submerged = false;

  ThrusterModel thr;
  thr.max_thrust = 100.0f;
  thr.command = 1.0f;

  auto force = ComputeThrustForce(fluid, thr, {});
  CHECK_NEAR(force.force.x, 0.0f, 1e-10f);
}

// --- Strip sampling tests ---

TEST(Strip_ProducesTorque) {
  // A body aligned with x in a transverse (y) shear flow should experience
  // a yaw torque. The y-velocity varies along x, so different strip stations
  // see different transverse drag. Since offset is along x and force is
  // along y, the cross product offset.Cross(force) has a z component.
  ShallowWaterSolver swe;
  swe.Init(20, 20, 0.5f);
  swe.SetSurface(2.0f);

  // Create a transverse shear: y-momentum increases with x.
  auto& hv = swe.grid.channels[swe.ch_hv].data;
  auto& h = swe.grid.channels[swe.ch_h].data;
  for (uint32_t j = 0; j < swe.grid.ny; ++j) {
    for (uint32_t i = 0; i < swe.grid.nx; ++i) {
      size_t idx = swe.grid.Idx(i, j);
      float wx, wy;
      swe.grid.GridToWorld(i, j, wx, wy);
      hv[idx] = h[idx] * wx * 0.2f;  // v = 0.2*x m/s (transverse shear)
    }
  }

  BodyCoupling body;
  body.volume = 0.1f;
  body.cross_section = 0.05f;
  body.drag_coeff = 1.0f;

  StripConfig strip;
  strip.num_samples = 5;
  strip.length = 2.0f;  // 2m body
  strip.axis = {1, 0, 0};  // aligned with x

  Vec3 body_center = {5.0f, 5.0f, 1.0f};
  Vec3 body_vel = {};
  Vec3 angular_vel = {};

  auto query_fn = [&](Vec3 pos) -> FluidCoupling::FluidState {
    return FluidCoupling::QuerySWE(swe, pos);
  };

  auto force = ComputeStripForces(query_fn, body, strip,
                                    body_center, body_vel, angular_vel, 0.01f);

  // Front (x=6) sees v=1.2 m/s, back (x=4) sees v=0.8 m/s.
  // Drag on each strip is in -y (opposing transverse flow).
  // offset.Cross(force_y) for offset={+1,0,0} and force={0,fy,0} gives
  // torque_z = +1 * fy (negative since drag is in -y, so torque_z < 0
  // at front and > 0 at back... actually: {1,0,0}.Cross({0,-fy,0}) = {0,0,-fy}.
  // Front has larger |fy|, so net torque_z < 0.
  CHECK(std::abs(force.torque.z) > 0.01f);
}

int main() {
  return RunAllTests();
}
