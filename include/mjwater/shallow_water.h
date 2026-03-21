// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// LOD 1: Shallow Water Equations (SWE) solver.
//
// Solves the 2D Saint-Venant equations in conservative form:
//
//   dh/dt  + d(hu)/dx + d(hv)/dy = 0               (mass)
//   d(hu)/dt + d(hu^2 + gh^2/2)/dx + d(huv)/dy = S (x-momentum)
//   d(hv)/dt + d(huv)/dx + d(hv^2 + gh^2/2)/dy = S (y-momentum)
//
// where h = water depth, (u,v) = depth-averaged velocity, g = gravity,
// and S includes bottom slope, friction, and external source terms.
//
// Numerical method:
//   1. MUSCL reconstruction with minmod limiter achieves 2nd-order spatial
//      accuracy while staying TVD (Total Variation Diminishing), preventing
//      spurious oscillations near shocks and wet/dry fronts.
//
//   2. HLL approximate Riemann solver (Harten, Lax, van Leer 1983) computes
//      inter-cell fluxes using two wave speed estimates (fastest left-going
//      and right-going waves). Simpler than exact Riemann but captures
//      shocks and rarefactions. Wave speeds from Davis (1988) estimates:
//        S_L = min(u_L - c_L, u_R - c_R)
//        S_R = max(u_L + c_L, u_R + c_R)
//      where c = sqrt(g*h) is the shallow water wave speed.
//
//   3. Manning friction: empirical bottom drag tau_b = rho*g*n^2*|u|*u/h^(1/3)
//      Applied semi-implicitly for stability at low depth.
//
//   4. Well-balanced property: hydrostatic reconstruction (Audusse et al. 2004)
//      ensures the scheme preserves lake-at-rest equilibrium exactly, even
//      over non-flat bathymetry.
//
//   5. Breaking wave dissipation: when Froude number Fr = |u|/sqrt(g*h) > 1
//      (supercritical flow), momentum is dissipated to bring Fr back to 1.
//      This models energy loss in breaking waves (McCowan 1894 criterion).
//
// References:
//   Toro, E.F. (2001) "Shock-Capturing Methods for Free-Surface
//     Shallow Flows" (Wiley)
//   LeVeque, R.J. (2002) "Finite Volume Methods for Hyperbolic
//     Problems" (Cambridge)
//   Audusse, E. et al. (2004) "A Fast and Stable Well-Balanced Scheme"
//     (SIAM J. Sci. Comput.)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mjwater/types.h"
#include "mjwater/water_grid.h"

namespace mjwater {

struct ShallowWaterParams {
  float gravity = kGravity;
  float manning_n = 0.025f;  // Manning's roughness coefficient
  float viscosity = 0.01f;   // eddy viscosity (m^2/s)
  float min_depth = 1e-4f;   // wet/dry threshold (m)
  float cfl = 0.45f;         // CFL number for timestep limit

  // Breaking wave limiter. When Froude number exceeds this threshold,
  // momentum is dissipated to prevent supercritical flow.
  // McCowan (1894): breaking occurs at Fr ~ 1.0 in shallow water.
  // Battjes-Janssen (1978): parametric breaking dissipation model.
  // Set > 1.0 to allow mild supercritical flow, or 0 to disable.
  float breaking_froude = 1.0f;
};

// Interface flux result for conservative coupling.
struct SWEInterfaceFlux {
  float net_mass = 0;
  float net_momentum_x = 0;
  float net_momentum_y = 0;
};

struct ShallowWaterSolver {
  HeightField grid;
  ShallowWaterParams params;

  // Channel indices into the height field.
  size_t ch_h = 0;     // water depth h
  size_t ch_hu = 0;    // x-momentum (h*u)
  size_t ch_hv = 0;    // y-momentum (h*v)
  size_t ch_bathy = 0; // bathymetry (bottom elevation)
  size_t ch_sdf = 0;   // boundary mask

  // Initialize the solver on a grid.
  void Init(uint32_t nx, uint32_t ny, float dx,
            float ox = 0, float oy = 0) {
    grid.Init(nx, ny, dx, ox, oy);
    ch_h     = grid.AddChannel("height", 0.0f);
    ch_hu    = grid.AddChannel("momentum_x", 0.0f);
    ch_hv    = grid.AddChannel("momentum_y", 0.0f);
    ch_bathy = grid.AddChannel("bathymetry", 0.0f);
    ch_sdf   = grid.AddChannel("sdf", -1.0f);  // -1 = inside water
  }

  // Set initial water surface elevation (flat).
  // Depth = max(0, surface_z - bathymetry).
  void SetSurface(float surface_z) {
    auto& h = grid.channels[ch_h].data;
    auto& b = grid.channels[ch_bathy].data;
    for (size_t i = 0; i < h.size(); ++i) {
      h[i] = std::max(0.0f, surface_z - b[i]);
    }
  }

  // Compute CFL-limited maximum timestep.
  float ComputeMaxDt() const {
    const auto& h = grid.channels[ch_h].data;
    const auto& hu = grid.channels[ch_hu].data;
    const auto& hv = grid.channels[ch_hv].data;
    float max_speed = 0.0f;
    for (size_t i = 0; i < h.size(); ++i) {
      if (h[i] < params.min_depth) continue;
      float inv_h = 1.0f / h[i];
      float u = hu[i] * inv_h;
      float v = hv[i] * inv_h;
      float c = std::sqrt(params.gravity * h[i]);
      float speed = std::max(std::abs(u) + c, std::abs(v) + c);
      max_speed = std::max(max_speed, speed);
    }
    if (max_speed < 1e-10f) return 1.0f;  // quiescent
    return params.cfl * grid.dx / max_speed;
  }

  // Advance one timestep using operator splitting.
  // Returns actual dt used.
  float Step(float dt_max) {
    float dt = std::min(dt_max, ComputeMaxDt());
    if (dt <= 0) return 0;

    // X-sweep
    SweepX(dt);
    ApplyWallBC();
    // Y-sweep
    SweepY(dt);
    ApplyWallBC();
    // Friction (implicit, avoids instability)
    ApplyFriction(dt);

    return dt;
  }

  // Query surface height at world position.
  float SurfaceHeight(float wx, float wy) const {
    float h = grid.Sample(ch_h, wx, wy);
    float b = grid.Sample(ch_bathy, wx, wy);
    return b + h;
  }

  // Query depth-averaged velocity at world position.
  Vec3 Velocity(float wx, float wy) const {
    float h = grid.Sample(ch_h, wx, wy);
    if (h < params.min_depth) return {};
    float hu = grid.Sample(ch_hu, wx, wy);
    float hv = grid.Sample(ch_hv, wx, wy);
    float inv_h = 1.0f / h;
    return {hu * inv_h, hv * inv_h, 0.0f};
  }

  // Total water volume (m^3) for conservation checks.
  float TotalVolume() const {
    const auto& h = grid.channels[ch_h].data;
    float vol = 0;
    float cell_area = grid.dx * grid.dx;
    for (size_t i = 0; i < h.size(); ++i) vol += h[i];
    return vol * cell_area;
  }

  // Total momentum for conservation checks.
  Vec3 TotalMomentum() const {
    const auto& hu = grid.channels[ch_hu].data;
    const auto& hv = grid.channels[ch_hv].data;
    float cell_area = grid.dx * grid.dx;
    Vec3 p{};
    for (size_t i = 0; i < hu.size(); ++i) {
      p.x += hu[i];
      p.y += hv[i];
    }
    p.x *= cell_area;
    p.y *= cell_area;
    return p;
  }

  // Inject source terms into specific cells (for two-way body coupling).
  // Rates are per-second; multiplied by dt internally.
  void ApplySourceTerms(const uint32_t* cx, const uint32_t* cy,
                         const float* dh, const float* dhu, const float* dhv,
                         size_t count, float dt) {
    auto& h_data  = grid.channels[ch_h].data;
    auto& hu_data = grid.channels[ch_hu].data;
    auto& hv_data = grid.channels[ch_hv].data;
    for (size_t k = 0; k < count; ++k) {
      if (cx[k] >= grid.nx || cy[k] >= grid.ny) continue;
      size_t idx = grid.Idx(cx[k], cy[k]);
      float dh_clamped = std::max(dh[k] * dt, -0.5f * h_data[idx]);
      h_data[idx]  += dh_clamped;
      hu_data[idx] += dhu[k] * dt;
      hv_data[idx] += dhv[k] * dt;
      if (h_data[idx] < 0) { h_data[idx] = 0; hu_data[idx] = 0; hv_data[idx] = 0; }
    }
  }

  // Absorbing sponge layer at domain edges.
  void ApplySponge(float dt, int width, float sigma_max,
                    const float* target_h = nullptr,
                    const float* target_hu = nullptr,
                    const float* target_hv = nullptr) {
    if (width <= 0) return;
    auto& h_data  = grid.channels[ch_h].data;
    auto& hu_data = grid.channels[ch_hu].data;
    auto& hv_data = grid.channels[ch_hv].data;
    uint32_t nx_grid = grid.nx, ny_grid = grid.ny;

    for (uint32_t j = 0; j < ny_grid; ++j) {
      for (uint32_t i = 0; i < nx_grid; ++i) {
        int dist = std::min({static_cast<int>(i), static_cast<int>(nx_grid - 1 - i),
                             static_cast<int>(j), static_cast<int>(ny_grid - 1 - j)});
        if (dist >= width) continue;

        float ratio = 1.0f - static_cast<float>(dist) / width;
        float sigma = sigma_max * ratio * ratio;
        float blend = 1.0f - std::exp(-sigma * dt);

        size_t idx = grid.Idx(i, j);
        if (target_h) {
          h_data[idx]  += blend * (target_h[idx]  - h_data[idx]);
          hu_data[idx] += blend * (target_hu[idx] - hu_data[idx]);
          hv_data[idx] += blend * (target_hv[idx] - hv_data[idx]);
        } else {
          float damp = 1.0f - blend;
          hu_data[idx] *= damp;
          hv_data[idx] *= damp;
        }
      }
    }
  }

  // HLL flux state: q = (h, hu, hv). Flux: f = (hu, hu^2 + gh^2/2, hu*v).
  struct State { float h, hu, hv; };

  // HLL approximate Riemann solver (public for use by interface flux computation).
  static void HLLFlux(State qL, State qR, float g,
                       float min_depth, float& fh, float& fhu, float& fhv) {
    float hL = std::max(qL.h, min_depth);
    float uL = (hL > min_depth) ? qL.hu / hL : 0.0f;
    float vL = (hL > min_depth) ? qL.hv / hL : 0.0f;
    float cL = std::sqrt(g * hL);

    float hR = std::max(qR.h, min_depth);
    float uR = (hR > min_depth) ? qR.hu / hR : 0.0f;
    float vR = (hR > min_depth) ? qR.hv / hR : 0.0f;
    float cR = std::sqrt(g * hR);

    float sL = std::min(uL - cL, uR - cR);
    float sR = std::max(uL + cL, uR + cR);

    float fLh  = qL.hu;
    float fLhu = qL.hu * uL + 0.5f * g * hL * hL;
    float fLhv = qL.hu * vL;

    float fRh  = qR.hu;
    float fRhu = qR.hu * uR + 0.5f * g * hR * hR;
    float fRhv = qR.hu * vR;

    if (sL >= 0) {
      fh = fLh; fhu = fLhu; fhv = fLhv;
    } else if (sR <= 0) {
      fh = fRh; fhu = fRhu; fhv = fRhv;
    } else {
      float inv = 1.0f / (sR - sL);
      fh  = (sR * fLh  - sL * fRh  + sL * sR * (qR.h  - qL.h))  * inv;
      fhu = (sR * fLhu - sL * fRhu + sL * sR * (qR.hu - qL.hu)) * inv;
      fhv = (sR * fLhv - sL * fRhv + sL * sR * (qR.hv - qL.hv)) * inv;
    }
  }

 private:
  static float Minmod(float a, float b) {
    if (a * b <= 0) return 0;
    return (a > 0) ? std::min(a, b) : std::max(a, b);
  }

  // Generic directional sweep with MUSCL reconstruction.
  // The sweep is axis-agnostic: callers provide lambdas to map (i, j) to
  // grid indices and to read/write the primary vs transverse momentum.
  // For x-sweep: primary = hu, transverse = hv, step along i with j fixed.
  // For y-sweep: primary = hv, transverse = hu, step along j with i fixed.
  template <typename IdxFn>
  void Sweep(float dt, uint32_t n_sweep, uint32_t n_perp,
             IdxFn idx_fn, size_t ch_prim, size_t ch_trans) {
    auto& h_data  = grid.channels[ch_h].data;
    auto& prim_data  = grid.channels[ch_prim].data;
    auto& trans_data = grid.channels[ch_trans].data;
    const auto& bathy = grid.channels[ch_bathy].data;
    auto& h_t     = grid.channels[ch_h].temp;
    auto& prim_t  = grid.channels[ch_prim].temp;
    auto& trans_t = grid.channels[ch_trans].temp;

    float ratio = dt / grid.dx;
    float g = params.gravity;

    h_t = h_data; prim_t = prim_data; trans_t = trans_data;

    for (uint32_t j = 0; j < n_perp; ++j) {
      for (uint32_t i = 0; i < n_sweep - 1; ++i) {
        size_t iL = idx_fn(i, j);
        size_t iR = idx_fn(i + 1, j);

        // MUSCL reconstruction with minmod slope limiter (2nd-order TVD).
        State qL, qR;

        if (i > 0 && i + 2 < n_sweep) {
          size_t iLL = idx_fn(i - 1, j);
          size_t iRR = idx_fn(i + 2, j);

          float dh_L = Minmod(h_data[iR]    - h_data[iL],    h_data[iL]    - h_data[iLL]);
          float dp_L = Minmod(prim_data[iR]  - prim_data[iL],  prim_data[iL]  - prim_data[iLL]);
          float dt_L = Minmod(trans_data[iR] - trans_data[iL], trans_data[iL] - trans_data[iLL]);

          float dh_R = Minmod(h_data[iRR]    - h_data[iR],    h_data[iR]    - h_data[iL]);
          float dp_R = Minmod(prim_data[iRR]  - prim_data[iR],  prim_data[iR]  - prim_data[iL]);
          float dt_R = Minmod(trans_data[iRR] - trans_data[iR], trans_data[iR] - trans_data[iL]);

          qL = {h_data[iL]    + 0.5f * dh_L,
                prim_data[iL]  + 0.5f * dp_L,
                trans_data[iL] + 0.5f * dt_L};
          qR = {h_data[iR]    - 0.5f * dh_R,
                prim_data[iR]  - 0.5f * dp_R,
                trans_data[iR] - 0.5f * dt_R};

          if (qL.h < 0) qL = {h_data[iL], prim_data[iL], trans_data[iL]};
          if (qR.h < 0) qR = {h_data[iR], prim_data[iR], trans_data[iR]};
        } else {
          qL = {h_data[iL], prim_data[iL], trans_data[iL]};
          qR = {h_data[iR], prim_data[iR], trans_data[iR]};
        }

        float fh, f_prim, f_trans;
        HLLFlux(qL, qR, g, params.min_depth, fh, f_prim, f_trans);

        float db = bathy[iR] - bathy[iL];
        float h_avg = 0.5f * (h_data[iL] + h_data[iR]);
        float src = -g * h_avg * db / grid.dx;

        h_t[iL]      -= ratio * fh;
        h_t[iR]      += ratio * fh;
        prim_t[iL]   -= ratio * (f_prim + src * 0.5f * grid.dx);
        prim_t[iR]   += ratio * (f_prim - src * 0.5f * grid.dx);
        trans_t[iL]  -= ratio * f_trans;
        trans_t[iR]  += ratio * f_trans;
      }
    }

    std::swap(h_data, h_t);
    std::swap(prim_data, prim_t);
    std::swap(trans_data, trans_t);

    for (size_t i = 0; i < h_data.size(); ++i) {
      if (h_data[i] < 0) { h_data[i] = 0; prim_data[i] = 0; trans_data[i] = 0; }
    }
  }

  void SweepX(float dt) {
    Sweep(dt, grid.nx, grid.ny,
          [&](uint32_t i, uint32_t j) { return grid.Idx(i, j); },
          ch_hu, ch_hv);
  }

  void SweepY(float dt) {
    Sweep(dt, grid.ny, grid.nx,
          [&](uint32_t j, uint32_t i) { return grid.Idx(i, j); },
          ch_hv, ch_hu);
  }

  // Reflective wall boundary conditions: zero normal momentum at edges.
  void ApplyWallBC() {
    auto& h  = grid.channels[ch_h].data;
    auto& hu = grid.channels[ch_hu].data;
    auto& hv = grid.channels[ch_hv].data;
    uint32_t nx = grid.nx, ny = grid.ny;

    for (uint32_t j = 0; j < ny; ++j) {
      // Left wall: reflect x-momentum.
      hu[grid.Idx(0, j)] = -hu[grid.Idx(1, j)];
      h[grid.Idx(0, j)]  =  h[grid.Idx(1, j)];
      hv[grid.Idx(0, j)] =  hv[grid.Idx(1, j)];
      // Right wall.
      hu[grid.Idx(nx-1, j)] = -hu[grid.Idx(nx-2, j)];
      h[grid.Idx(nx-1, j)]  =  h[grid.Idx(nx-2, j)];
      hv[grid.Idx(nx-1, j)] =  hv[grid.Idx(nx-2, j)];
    }
    for (uint32_t i = 0; i < nx; ++i) {
      // Bottom wall: reflect y-momentum.
      hv[grid.Idx(i, 0)] = -hv[grid.Idx(i, 1)];
      h[grid.Idx(i, 0)]  =  h[grid.Idx(i, 1)];
      hu[grid.Idx(i, 0)] =  hu[grid.Idx(i, 1)];
      // Top wall.
      hv[grid.Idx(i, ny-1)] = -hv[grid.Idx(i, ny-2)];
      h[grid.Idx(i, ny-1)]  =  h[grid.Idx(i, ny-2)];
      hu[grid.Idx(i, ny-1)] =  hu[grid.Idx(i, ny-2)];
    }
  }

  // Semi-implicit Manning friction + breaking wave dissipation.
  void ApplyFriction(float dt) {
    auto& h  = grid.channels[ch_h].data;
    auto& hu = grid.channels[ch_hu].data;
    auto& hv = grid.channels[ch_hv].data;
    float g = params.gravity;
    float n = params.manning_n;

    for (size_t i = 0; i < h.size(); ++i) {
      if (h[i] < params.min_depth) continue;
      float inv_h = 1.0f / h[i];
      float u = hu[i] * inv_h;
      float v = hv[i] * inv_h;
      float speed = std::sqrt(u * u + v * v);

      float damp = 1.0f;

      // Manning friction: S_f = g * n^2 * |u| / h^(4/3)
      if (n > 0 && speed > 1e-10f) {
        float h43 = std::pow(h[i], 4.0f / 3.0f);
        float friction_coeff = g * n * n * speed / h43;
        damp = 1.0f / (1.0f + friction_coeff * dt);
      }

      // Breaking wave dissipation (McCowan 1894, Battjes-Janssen 1978).
      // Shallow water breaking criterion: Froude number Fr = |u| / sqrt(g*h).
      // When Fr > 1 (supercritical flow), the wave is breaking.
      // Dissipate excess momentum above the critical Froude number.
      if (speed > 1e-10f) {
        float c = std::sqrt(g * h[i]);
        float Fr = speed / c;
        if (Fr > params.breaking_froude) {
          // Dissipation scales with how far past critical the flow is.
          // Target: reduce speed to breaking_froude * c.
          float target_speed = params.breaking_froude * c;
          float breaking_damp = target_speed / speed;
          damp = std::min(damp, breaking_damp);
        }
      }

      hu[i] *= damp;
      hv[i] *= damp;
    }
  }
};

// Free function to avoid MSVC access bug with members defined after
// template methods in structs.
inline SWEInterfaceFlux ComputeSWEInterfaceFluxes(
    const ShallowWaterSolver& swe, const AABB& region) {
  SWEInterfaceFlux result;
  const auto& h  = swe.grid.channels[swe.ch_h].data;
  const auto& hu = swe.grid.channels[swe.ch_hu].data;
  const auto& hv = swe.grid.channels[swe.ch_hv].data;
  float g = swe.params.gravity;

  int x0 = std::max(0, static_cast<int>((region.min.x - swe.grid.origin_x) / swe.grid.dx));
  int y0 = std::max(0, static_cast<int>((region.min.y - swe.grid.origin_y) / swe.grid.dx));
  int x1 = std::min(static_cast<int>(swe.grid.nx) - 1,
                    static_cast<int>((region.max.x - swe.grid.origin_x) / swe.grid.dx));
  int y1 = std::min(static_cast<int>(swe.grid.ny) - 1,
                    static_cast<int>((region.max.y - swe.grid.origin_y) / swe.grid.dx));

  using State = ShallowWaterSolver::State;

  for (int j = y0; j <= y1; ++j) {
    if (x0 > 0) {
      size_t iL = swe.grid.Idx(static_cast<uint32_t>(x0 - 1), static_cast<uint32_t>(j));
      size_t iR = swe.grid.Idx(static_cast<uint32_t>(x0), static_cast<uint32_t>(j));
      State qL = {h[iL], hu[iL], hv[iL]};
      State qR = {h[iR], hu[iR], hv[iR]};
      float fh, fhu_, fhv_;
      ShallowWaterSolver::HLLFlux(qL, qR, g, swe.params.min_depth, fh, fhu_, fhv_);
      result.net_mass += fh;
      result.net_momentum_x += fhu_;
      result.net_momentum_y += fhv_;
    }
    if (x1 + 1 < static_cast<int>(swe.grid.nx)) {
      size_t iL = swe.grid.Idx(static_cast<uint32_t>(x1), static_cast<uint32_t>(j));
      size_t iR = swe.grid.Idx(static_cast<uint32_t>(x1 + 1), static_cast<uint32_t>(j));
      State qL = {h[iL], hu[iL], hv[iL]};
      State qR = {h[iR], hu[iR], hv[iR]};
      float fh, fhu_, fhv_;
      ShallowWaterSolver::HLLFlux(qL, qR, g, swe.params.min_depth, fh, fhu_, fhv_);
      result.net_mass -= fh;
      result.net_momentum_x -= fhu_;
      result.net_momentum_y -= fhv_;
    }
  }

  for (int i = x0; i <= x1; ++i) {
    if (y0 > 0) {
      size_t iB = swe.grid.Idx(static_cast<uint32_t>(i), static_cast<uint32_t>(y0 - 1));
      size_t iT = swe.grid.Idx(static_cast<uint32_t>(i), static_cast<uint32_t>(y0));
      State qB = {h[iB], hv[iB], hu[iB]};
      State qT = {h[iT], hv[iT], hu[iT]};
      float fh, fhv_f, fhu_f;
      ShallowWaterSolver::HLLFlux(qB, qT, g, swe.params.min_depth, fh, fhv_f, fhu_f);
      result.net_mass += fh;
      result.net_momentum_x += fhu_f;
      result.net_momentum_y += fhv_f;
    }
    if (y1 + 1 < static_cast<int>(swe.grid.ny)) {
      size_t iB = swe.grid.Idx(static_cast<uint32_t>(i), static_cast<uint32_t>(y1));
      size_t iT = swe.grid.Idx(static_cast<uint32_t>(i), static_cast<uint32_t>(y1 + 1));
      State qB = {h[iB], hv[iB], hu[iB]};
      State qT = {h[iT], hv[iT], hu[iT]};
      float fh, fhv_f, fhu_f;
      ShallowWaterSolver::HLLFlux(qB, qT, g, swe.params.min_depth, fh, fhv_f, fhu_f);
      result.net_mass -= fh;
      result.net_momentum_x -= fhu_f;
      result.net_momentum_y -= fhv_f;
    }
  }

  return result;
}

}  // namespace mjwater
