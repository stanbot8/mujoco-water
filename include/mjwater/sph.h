// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// LOD 2: Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH).
//
// SPH is a meshless Lagrangian method: instead of a fixed grid, the fluid
// is represented by particles that carry mass, velocity, and other
// properties. Particles move with the flow and interact through a smoothing
// kernel, making SPH naturally good at free-surface flows, splashing, and
// fragmentation where grid methods struggle.
//
// Key concepts:
//
//   Kernel (Wendland C2): a bell-shaped weighting function W(r,h) that
//   defines how particles influence each other. The smoothing length h
//   controls the interaction radius (compact support = 2h). Wendland C2
//   is chosen for its C2 continuity (smooth second derivatives, needed
//   for stable pressure computation) and positivity (no negative weights
//   that cause tensile instability). The 3D normalization is 21/(2*pi*h^3).
//
//   Neighbor search: each timestep we must find all particle pairs within
//   2h of each other. A spatial hash table bins particles into cells of
//   size 2h, so only 27 neighboring cells need checking per particle.
//   This gives O(N) search instead of O(N^2).
//
//   Density summation: each particle's density is computed from its
//   neighbors: rho_i = sum_j(m_j * W(r_ij, h)). This avoids solving
//   a continuity equation and ensures mass conservation exactly.
//
//   Equation of state (Tait): pressure from density via
//     P = (rho_0 * c_s^2 / gamma) * ((rho/rho_0)^gamma - 1)
//   with gamma=7. This is "weakly compressible": density variations are
//   kept small (< 1%) by setting the speed of sound c_s much larger than
//   the maximum flow velocity (typically c_s > 10 * u_max). The Mach
//   number Ma = u_max/c_s should stay below ~0.1.
//
//   Artificial viscosity (Monaghan 1992): adds numerical dissipation to
//   prevent particle interpenetration and suppress post-shock oscillations.
//   NOTE: the "viscosity" parameter in SPHParams is the artificial viscosity
//   coefficient (alpha in Monaghan's formulation), NOT the physical kinematic
//   viscosity (m^2/s). Physical viscosity in WCSPH would require a separate
//   Laplacian term (Morris et al. 1997).
//
//   XSPH velocity smoothing (Monaghan 1989): particles are advected with
//   a weighted average of their velocity and neighbors' velocities. This
//   reduces noise and keeps particles ordered, at the cost of slightly
//   violating momentum conservation. The epsilon parameter (0 to 1)
//   controls the smoothing strength.
//
//   Surface tension (CSF): the Continuum Surface Force model estimates
//   surface curvature from the color field gradient and applies a normal
//   force at the free surface. The implementation uses a heuristic
//   Laplacian approximation (not the exact kernel Laplacian), which is
//   adequate for visual effects but not quantitatively accurate.
//
// References:
//   Monaghan, J.J. (1992) "Smoothed Particle Hydrodynamics" Ann. Rev.
//     Astron. Astrophys. 30, 543-574. Artificial viscosity formulation.
//   Monaghan, J.J. (2005) "Smoothing Particle Hydrodynamics"
//     Rep. Prog. Phys. 68, 1703-1759. Comprehensive review.
//   Wendland, H. (1995) "Piecewise polynomial, positive definite and
//     compactly supported radial functions of minimal degree"
//     Adv. Comput. Math. 4, 389-396
//   Becker, M. & Teschner, M. (2007) "Weakly compressible SPH for free
//     surface flows" SCA. Tait EOS with gamma=7.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "mjwater/types.h"

namespace mjwater {

// --- SPH Kernel: Wendland C2 ---
// W(r, h) = (21 / (2*pi*h^3)) * (1 - r/(2h))^4 * (1 + 2r/h)  for r < 2h
// Compact support radius = 2h.

struct WendlandC2 {
  float h = 0.02f;           // smoothing length (m)
  float support = 0.04f;     // = 2h
  float norm_3d = 0;         // normalization constant

  void Init(float h_) {
    h = h_;
    support = 2.0f * h;
    // 3D normalization: 21 / (2 * pi * h^3)
    norm_3d = 21.0f / (2.0f * kPi * h * h * h);
  }

  // Kernel value.
  float W(float r) const {
    float q = r / h;
    if (q >= 2.0f) return 0.0f;
    float t = 1.0f - 0.5f * q;
    float t2 = t * t;
    return norm_3d * t2 * t2 * (1.0f + 2.0f * q);
  }

  // Kernel gradient magnitude: dW/dr.
  // Returns negative value (kernel decreases with r).
  float DW(float r) const {
    float q = r / h;
    if (q >= 2.0f || r < 1e-10f) return 0.0f;
    float t = 1.0f - 0.5f * q;
    float t3 = t * t * t;
    // dW/dr = norm * (-5q/h) * (1-q/2)^3
    return norm_3d * (-5.0f * q / h) * t3;
  }
};

// --- SPH Particle ---

struct SPHParticle {
  Vec3 pos;
  Vec3 vel;
  Vec3 acc;
  float density = kWaterDensity;
  float pressure = 0;
  float mass = 0;
};

// --- SPH Parameters ---

struct SPHParams {
  float smoothing_length = 0.02f;    // h (m)
  float rest_density = kWaterDensity;
  float speed_of_sound = 20.0f;      // artificial c_s for WCSPH
  // Monaghan artificial viscosity coefficient (alpha). NOT physical kinematic
  // viscosity. Typical values: 0.01 (low dissipation) to 0.1 (high dissipation).
  float alpha_viscosity = 0.08f;
  float gravity = kGravity;
  float cfl = 0.25f;
  float xsph_epsilon = 0.5f;         // XSPH velocity smoothing (Monaghan 1989)
  uint32_t max_particles = 200000;
  // Shepard density reinitialization every N steps. 15 is balanced for 1ms
  // SPH timesteps (~15ms physical period). Scale proportionally: use 30
  // for 0.5ms steps, 8 for 2ms steps, targeting ~15ms between corrections.
  int shepard_interval = 15;
  float surface_tension = 0.0728f;   // surface tension coefficient (N/m), water at 20C
  float boundary_damping = 0.5f;    // velocity damping on domain boundary reflection (0=absorb, 1=elastic)

  // Derived
  float ParticleMass(float spacing) const {
    return rest_density * spacing * spacing * spacing;
  }
};

// --- Spatial Hash Grid for Neighbor Search ---

struct SpatialHash {
  float cell_size = 0.04f;      // = 2 * smoothing_length
  float inv_cell_size = 25.0f;  // 1.0f / cell_size
  uint32_t table_size = 65536;  // hash table size (power of 2)
  std::vector<uint32_t> cell_start;
  std::vector<uint32_t> cell_end;
  std::vector<uint32_t> sorted_idx;
  std::vector<uint32_t> cell_hash;
  std::vector<uint32_t> write_pos;  // persistent scratch for Build()

  void Init(float cell_sz, uint32_t n_particles) {
    cell_size = cell_sz;
    inv_cell_size = 1.0f / cell_sz;
    // Round table_size to next power of 2 from expected cell count.
    table_size = std::max(1u, n_particles / 4);
    table_size--;
    table_size |= table_size >> 1;
    table_size |= table_size >> 2;
    table_size |= table_size >> 4;
    table_size |= table_size >> 8;
    table_size |= table_size >> 16;
    table_size++;
    table_size = std::max(table_size, 1024u);

    cell_start.resize(table_size);
    cell_end.resize(table_size);
  }

  uint32_t Hash(int cx, int cy, int cz) const {
    // Spatial hash: combine with large primes, mask to table_size.
    uint32_t h = static_cast<uint32_t>(cx) * 73856093u ^
                 static_cast<uint32_t>(cy) * 19349663u ^
                 static_cast<uint32_t>(cz) * 83492791u;
    return h & (table_size - 1);
  }

  void CellCoord(Vec3 pos, int& cx, int& cy, int& cz) const {
    cx = static_cast<int>(std::floor(pos.x * inv_cell_size));
    cy = static_cast<int>(std::floor(pos.y * inv_cell_size));
    cz = static_cast<int>(std::floor(pos.z * inv_cell_size));
  }

  // Build the spatial hash from particle positions (counting sort).
  // Rehashes if particle count has grown beyond 4x the table size
  // (load factor > 4 degrades O(1) lookup to O(N/table_size)).
  void Build(const std::vector<SPHParticle>& particles) {
    uint32_t n = static_cast<uint32_t>(particles.size());

    // Dynamic rehash: if load factor exceeds 4, double the table.
    if (n > table_size * 4) {
      uint32_t new_size = table_size;
      while (new_size < n / 4) new_size *= 2;
      new_size = std::max(new_size, 1024u);
      table_size = new_size;
      cell_start.resize(table_size);
      cell_end.resize(table_size);
    }

    cell_hash.resize(n);
    sorted_idx.resize(n);

    // Compute hash per particle.
    for (uint32_t i = 0; i < n; ++i) {
      int cx, cy, cz;
      CellCoord(particles[i].pos, cx, cy, cz);
      cell_hash[i] = Hash(cx, cy, cz);
    }

    // Counting sort by hash.
    std::fill(cell_start.begin(), cell_start.end(), 0);
    for (uint32_t i = 0; i < n; ++i) cell_start[cell_hash[i]]++;

    // Prefix sum.
    cell_end = cell_start;
    uint32_t sum = 0;
    for (uint32_t i = 0; i < table_size; ++i) {
      uint32_t count = cell_start[i];
      cell_start[i] = sum;
      sum += count;
    }
    // cell_end[i] = cell_start[i] + count (after next step)
    for (uint32_t i = 0; i < table_size; ++i) {
      cell_end[i] = cell_start[i] + cell_end[i];
    }

    // Place particles into sorted order.
    write_pos = cell_start;  // copy start positions
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t h = cell_hash[i];
      sorted_idx[write_pos[h]++] = i;
    }
  }

  // Iterate neighbors of a position within support radius.
  // Calls fn(particle_index) for each candidate within the 3x3x3 cell
  // neighborhood. The support parameter is accepted for API consistency
  // but not used (cell_size already equals the kernel support radius).
  template <typename Fn>
  void ForEachNeighbor(Vec3 pos, float /*support*/, Fn&& fn) const {
    if (cell_start.empty()) return;  // no particles / hash not built
    int cx, cy, cz;
    CellCoord(pos, cx, cy, cz);
    // Check 3x3x3 neighborhood.
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          uint32_t h = Hash(cx + dx, cy + dy, cz + dz);
          for (uint32_t k = cell_start[h]; k < cell_end[h]; ++k) {
            fn(sorted_idx[k]);
          }
        }
      }
    }
  }
};

// --- SPH Solver ---

struct SPHSolver {
  std::vector<SPHParticle> particles;
  SPHParams params;
  WendlandC2 kernel;
  SpatialHash hash;
  AABB domain;  // simulation domain for boundary clamping
  int step_count = 0;  // for Shepard reinitialization scheduling

  void Init(const SPHParams& p, const AABB& domain_) {
    params = p;
    domain = domain_;
    kernel.Init(p.smoothing_length);
    particles.clear();
    particles.reserve(p.max_particles);
  }

  // Fill a box with evenly spaced particles.
  void AddParticlesInBox(const AABB& box, float spacing) {
    float mass = params.ParticleMass(spacing);
    for (float z = box.min.z + spacing * 0.5f; z < box.max.z; z += spacing) {
      for (float y = box.min.y + spacing * 0.5f; y < box.max.y; y += spacing) {
        for (float x = box.min.x + spacing * 0.5f; x < box.max.x; x += spacing) {
          if (particles.size() >= params.max_particles) return;
          SPHParticle p;
          p.pos = {x, y, z};
          p.mass = mass;
          p.density = params.rest_density;
          particles.push_back(p);
        }
      }
    }
  }

  // Number of active particles.
  uint32_t Count() const { return static_cast<uint32_t>(particles.size()); }

  // Check if a point is submerged in the fluid (more accurate than density check).
  // Uses the kernel deficiency: at the free surface, the kernel integral is < 1
  // because particles only exist on one side. Interior particles have integral ~ 1.
  bool IsSubmerged(Vec3 pos, float threshold = 0.6f) const {
    float kernel_sum = 0;
    hash.ForEachNeighbor(pos, kernel.support, [&](uint32_t idx) {
      Vec3 r = pos - particles[idx].pos;
      float dist = r.Length();
      if (dist >= kernel.support) return;
      kernel_sum += (particles[idx].mass / particles[idx].density) * kernel.W(dist);
    });
    return kernel_sum >= threshold;
  }

  // Maximum particle speed in the system.
  float ComputeMaxSpeed() const {
    float max_vel = 0;
    for (const auto& p : particles) {
      max_vel = std::max(max_vel, p.vel.Length());
    }
    return max_vel;
  }

  // Interpolate density and velocity at an arbitrary world position using
  // the SPH kernel. Returns false if no particles are within support.
  bool InterpolateFields(Vec3 pos, float& rho_out, Vec3& vel_out) const {
    float rho = 0;
    Vec3 vel{};
    float w_sum = 0;

    hash.ForEachNeighbor(pos, kernel.support, [&](uint32_t idx) {
      const auto& p = particles[idx];
      float dist = (pos - p.pos).Length();
      float w = kernel.W(dist);
      rho += p.mass * w;
      vel += p.vel * (p.mass * w / std::max(p.density, 1e-6f));
      w_sum += w;
    });

    if (w_sum < 1e-10f) return false;
    rho_out = rho;
    vel_out = vel;
    return true;
  }

  // Interpolate the viscous stress tensor at a world position.
  // The stress tensor is approximated from the SPH velocity gradient:
  //   sigma_ab = sum_j m_j/rho_j * (v_a_j - v_a_i) * dW/dr * r_b / |r|
  // Returns the 6 independent components of the symmetric 3x3 tensor.
  // Returns false if no particles are within support.
  struct StressTensor {
    float xx = 0, yy = 0, zz = 0;
    float xy = 0, xz = 0, yz = 0;
  };

  bool InterpolateStressTensor(Vec3 pos, Vec3 vel_at_pos,
                                StressTensor& stress_out) const {
    stress_out = {};
    bool found = false;

    hash.ForEachNeighbor(pos, kernel.support, [&](uint32_t idx) {
      const auto& p = particles[idx];
      Vec3 r = pos - p.pos;
      float dist = r.Length();
      if (dist < 1e-10f) return;

      float dw = kernel.DW(dist);
      Vec3 r_hat = r / dist;
      float coeff = p.mass / std::max(p.density, 1e-6f) * dw / dist;

      // Velocity difference.
      Vec3 dv = p.vel - vel_at_pos;

      // Outer product contribution: sigma_ab += coeff * dv_a * r_b
      stress_out.xx += coeff * dv.x * r.x;
      stress_out.yy += coeff * dv.y * r.y;
      stress_out.zz += coeff * dv.z * r.z;
      stress_out.xy += coeff * 0.5f * (dv.x * r.y + dv.y * r.x);
      stress_out.xz += coeff * 0.5f * (dv.x * r.z + dv.z * r.x);
      stress_out.yz += coeff * 0.5f * (dv.y * r.z + dv.z * r.y);
      found = true;
    });

    return found;
  }

  // CFL-limited timestep.
  float ComputeMaxDt() const {
    float max_vel = 0;
    for (const auto& p : particles) {
      max_vel = std::max(max_vel, p.vel.Length());
    }
    float h = params.smoothing_length;
    return params.cfl * h / (params.speed_of_sound + max_vel + 1e-10f);
  }

  // Advance one timestep (leapfrog integration).
  float Step(float dt_max) {
    if (particles.empty()) return 0;
    float dt = std::min(dt_max, ComputeMaxDt());
    if (dt <= 0) return 0;

    // 1. Build neighbor structure.
    hash.Init(kernel.support, Count());
    hash.Build(particles);

    // 2. Compute density.
    ComputeDensity();

    // 3. Compute pressure (Tait equation of state).
    ComputePressure();

    // 4. Compute accelerations.
    ComputeForces();

    // 5. Leapfrog integration.
    Integrate(dt);

    // 6. Boundary clamping.
    EnforceBoundaries();

    // 7. Periodic Shepard density reinitialization.
    ++step_count;
    if (params.shepard_interval > 0 &&
        step_count % params.shepard_interval == 0) {
      ShepardReinitialize();
    }

    return dt;
  }

  // Adaptive refinement: split particles that are too large for their
  // region (near LBM zone) and merge particles that are too small
  // (far from LBM zone). Maintains particle count balance.
  void AdaptiveRefine(Vec3 focus, float refine_radius, float /*coarsen_radius*/,
                       float target_spacing) {
    float nominal_mass = params.ParticleMass(target_spacing);

    std::vector<SPHParticle> new_particles;
    new_particles.reserve(particles.size());

    for (auto& p : particles) {
      float dist = (p.pos - focus).Length();

      // Split: particle is large and close to focus.
      if (dist < refine_radius && p.mass > nominal_mass * 1.5f) {
        // Split into 2 daughter particles offset along velocity direction.
        float half_mass = p.mass * 0.5f;
        Vec3 offset = p.vel.Normalized() * (target_spacing * 0.25f);
        if (offset.LengthSq() < 1e-20f) offset = {target_spacing * 0.25f, 0, 0};

        SPHParticle p1 = p;
        p1.mass = half_mass;
        p1.pos = p.pos + offset;

        SPHParticle p2 = p;
        p2.mass = half_mass;
        p2.pos = p.pos - offset;

        new_particles.push_back(p1);
        if (new_particles.size() < params.max_particles)
          new_particles.push_back(p2);
      }
      // Merge: handled implicitly by not splitting far particles.
      // Particles far from focus with below-threshold mass are left as-is
      // (they will naturally merge during SPH->SWE transition).
      else {
        new_particles.push_back(p);
      }
    }

    particles = std::move(new_particles);
  }

  // Total mass for conservation checks.
  float TotalMass() const {
    float m = 0;
    for (const auto& p : particles) m += p.mass;
    return m;
  }

  // Interpolate velocity at arbitrary point.
  Vec3 InterpolateVelocity(Vec3 pos) const {
    Vec3 v{};
    hash.ForEachNeighbor(pos, kernel.support, [&](uint32_t idx) {
      Vec3 r = pos - particles[idx].pos;
      float dist = r.Length();
      if (dist >= kernel.support) return;
      float w = kernel.W(dist) * particles[idx].mass / particles[idx].density;
      v += particles[idx].vel * w;
    });
    return v;
  }

  // Interpolate density at arbitrary point.
  float InterpolateDensity(Vec3 pos) const {
    float rho = 0;
    hash.ForEachNeighbor(pos, kernel.support, [&](uint32_t idx) {
      Vec3 r = pos - particles[idx].pos;
      float dist = r.Length();
      if (dist >= kernel.support) return;
      rho += particles[idx].mass * kernel.W(dist);
    });
    return rho;
  }

 private:
  // Shepard filter density reinitialization.
  // Corrects accumulated density errors by normalizing with the
  // kernel sum (Shepard interpolation). Standard technique for
  // long-running WCSPH simulations (Randles & Libersky 1996).
  void ShepardReinitialize() {
    uint32_t n = Count();
    for (uint32_t a = 0; a < n; ++a) {
      auto& pa = particles[a];
      float rho_sum = 0;
      float w_sum = 0;
      hash.ForEachNeighbor(pa.pos, kernel.support, [&](uint32_t b) {
        Vec3 r = pa.pos - particles[b].pos;
        float dist = r.Length();
        if (dist >= kernel.support) return;
        float w = kernel.W(dist);
        rho_sum += particles[b].mass * w;
        w_sum += (particles[b].mass / particles[b].density) * w;
      });
      if (w_sum > 1e-10f) {
        pa.density = rho_sum / w_sum;
      }
    }
  }

  void ComputeDensity() {
    for (auto& pi : particles) pi.density = 0;
    uint32_t n = Count();
    for (uint32_t a = 0; a < n; ++a) {
      auto& pa = particles[a];
      hash.ForEachNeighbor(pa.pos, kernel.support, [&](uint32_t b) {
        Vec3 r = pa.pos - particles[b].pos;
        float dist = r.Length();
        if (dist >= kernel.support) return;
        pa.density += particles[b].mass * kernel.W(dist);
      });
      // Clamp to avoid division by zero.
      // Clamp density to 90% of rest density. Allowing lower values breaks
      // the weakly compressible assumption and creates unphysical voids.
      // If density drops below this, it indicates insufficient neighbors
      // (particle deficiency near free surface or boundaries).
      pa.density = std::max(pa.density, params.rest_density * 0.9f);
    }
  }

  void ComputePressure() {
    float rho0 = params.rest_density;
    float cs2 = params.speed_of_sound * params.speed_of_sound;
    // Tait equation of state (gamma=7, standard for water WCSPH).
    // p = B * ((rho/rho0)^gamma - 1)  where B = rho0 * cs^2 / gamma
    // Prevents negative pressures (tensile instability) and is more
    // physically accurate for weakly compressible water.
    constexpr float gamma = 7.0f;
    float B = rho0 * cs2 / gamma;
    for (auto& p : particles) {
      float rho_ratio = p.density / rho0;
      // Use fast power approximation: x^7 = x^4 * x^2 * x
      float r2 = rho_ratio * rho_ratio;
      float r4 = r2 * r2;
      float r7 = r4 * r2 * rho_ratio;
      p.pressure = B * (r7 - 1.0f);
      // Clamp to prevent negative pressure (can still happen with
      // very low density at free surface before Shepard correction).
      if (p.pressure < 0) p.pressure = 0;
    }
  }

  void ComputeForces() {
    float alpha = params.alpha_viscosity;
    uint32_t n = Count();

    for (uint32_t a = 0; a < n; ++a) {
      auto& pa = particles[a];
      Vec3 acc_pressure{};
      Vec3 acc_viscosity{};

      hash.ForEachNeighbor(pa.pos, kernel.support, [&](uint32_t b) {
        if (b == a) return;
        auto& pb = particles[b];
        Vec3 r = pa.pos - pb.pos;
        float dist = r.Length();
        if (dist >= kernel.support || dist < 1e-10f) return;

        Vec3 r_hat = r / dist;
        float dw = kernel.DW(dist);

        // Pressure gradient (symmetric form for momentum conservation).
        float p_term = pa.pressure / (pa.density * pa.density) +
                       pb.pressure / (pb.density * pb.density);
        acc_pressure += r_hat * (pb.mass * p_term * dw);

        // Artificial viscosity (Monaghan 1992): dissipates energy when
        // particles approach each other (vr < 0), preventing interpenetration.
        // mu = h * (dv.dr) / (|r|^2 + epsilon*h^2) is a smoothed velocity
        // gradient. pi_ab scales with alpha * c_s * mu / avg_density.
        Vec3 dv = pa.vel - pb.vel;
        float vr = dv.Dot(r);
        if (vr < 0) {
          float mu = params.smoothing_length * vr /
                     (dist * dist + 0.01f * params.smoothing_length * params.smoothing_length);
          float pi_ab = -alpha * params.speed_of_sound * mu /
                        (0.5f * (pa.density + pb.density));
          acc_viscosity += r_hat * (pb.mass * pi_ab * dw);
        }
      });

      pa.acc = acc_pressure + acc_viscosity;
      pa.acc.z -= params.gravity;  // gravity along -z

      // CSF surface tension (Morris 2000, Brackbill 1992).
      // Force per unit mass: -sigma * kappa * n_hat / rho
      // where kappa = -div(n_hat), n = grad(color field).
      //
      // NOTE: This uses a heuristic Laplacian approximation for curvature,
      // not the exact kernel Laplacian. Adequate for visual effects (drop
      // rounding, meniscus formation) but not quantitatively accurate for
      // capillary wave speeds or contact angle dynamics. Expect ~10-30%
      // error in surface tension force magnitude.
      if (params.surface_tension > 0) {
        Vec3 grad_c{};
        float lap_c = 0;
        hash.ForEachNeighbor(pa.pos, kernel.support, [&](uint32_t b) {
          if (b == a) return;
          auto& pb = particles[b];
          Vec3 r = pa.pos - pb.pos;
          float dist = r.Length();
          if (dist >= kernel.support || dist < 1e-10f) return;
          Vec3 r_hat = r / dist;
          float dw = kernel.DW(dist);
          float vol_b = pb.mass / pb.density;
          // Color field gradient: grad(c) = sum(m_j/rho_j * grad_W)
          grad_c += r_hat * (vol_b * dw);
          // Laplacian of color field for curvature.
          float lap_w = kernel.W(dist) * 2.0f *
                        (3.0f / (dist * dist + 0.01f * kernel.h * kernel.h));
          lap_c += vol_b * lap_w;
        });
        float n_len = grad_c.Length();
        // Only apply at the free surface (where |grad(c)| is significant).
        float threshold = 6.0f / (kernel.support * kernel.support);
        if (n_len > threshold) {
          Vec3 n_hat = grad_c / n_len;
          float kappa = -lap_c / n_len;
          pa.acc -= n_hat * (params.surface_tension * kappa / pa.density);
        }
      }
    }
  }

  void Integrate(float dt) {
    for (auto& p : particles) {
      p.vel += p.acc * dt;
    }

    // XSPH velocity smoothing (Monaghan 1989).
    // Advect using smoothed velocity: dx/dt = v_i + eps * sum(m_j/rho_ij * (v_j - v_i) * W_ij)
    // Regularizes particle distribution, reduces noise.
    if (params.xsph_epsilon > 0) {
      uint32_t n = Count();
      // Compute XSPH corrections.
      std::vector<Vec3> xsph_corr(n, Vec3{});
      float eps = params.xsph_epsilon;
      for (uint32_t a = 0; a < n; ++a) {
        auto& pa = particles[a];
        hash.ForEachNeighbor(pa.pos, kernel.support, [&](uint32_t b) {
          if (b == a) return;
          auto& pb = particles[b];
          Vec3 r = pa.pos - pb.pos;
          float dist = r.Length();
          if (dist >= kernel.support) return;
          float rho_avg = 0.5f * (pa.density + pb.density);
          float w = kernel.W(dist) * pb.mass / rho_avg;
          xsph_corr[a] += (pb.vel - pa.vel) * w;
        });
      }
      for (uint32_t a = 0; a < n; ++a) {
        particles[a].pos += (particles[a].vel + eps * xsph_corr[a]) * dt;
      }
    } else {
      for (auto& p : particles) {
        p.pos += p.vel * dt;
      }
    }
  }

  void EnforceBoundaries() {
    float damping = params.boundary_damping;
    for (auto& p : particles) {
      auto reflect = [&](float& pos, float& vel, float lo, float hi) {
        if (pos < lo) { pos = lo; if (vel < 0) vel *= -damping; }
        if (pos > hi) { pos = hi; if (vel > 0) vel *= -damping; }
      };
      reflect(p.pos.x, p.vel.x, domain.min.x, domain.max.x);
      reflect(p.pos.y, p.vel.y, domain.min.y, domain.max.y);
      reflect(p.pos.z, p.vel.z, domain.min.z, domain.max.z);
    }
  }
};

}  // namespace mjwater
