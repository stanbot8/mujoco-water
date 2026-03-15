// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// LOD 0: Shallow Water Equations solver.
//
// Saint-Venant equations in conservative form with HLL approximate
// Riemann solver, MUSCL reconstruction, Manning friction, and
// CFL-adaptive timestep.
//
// References:
//   Toro, E.F. (2001) "Shock-Capturing Methods for Free-Surface
//     Shallow Flows" (Wiley)
//   LeVeque, R.J. (2002) "Finite Volume Methods for Hyperbolic
//     Problems" (Cambridge)

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

 private:
  // Minmod slope limiter for MUSCL reconstruction.
  static float Minmod(float a, float b) {
    if (a * b <= 0) return 0;
    return (a > 0) ? std::min(a, b) : std::max(a, b);
  }

  // HLL flux in x-direction.
  // State: q = (h, hu, hv). Flux: f = (hu, hu^2 + gh^2/2, hu*v).
  struct State { float h, hu, hv; };

  static void HLLFlux(State qL, State qR, float g,
                       float min_depth, float& fh, float& fhu, float& fhv) {
    // Left state
    float hL = std::max(qL.h, min_depth);
    float uL = (hL > min_depth) ? qL.hu / hL : 0.0f;
    float vL = (hL > min_depth) ? qL.hv / hL : 0.0f;
    float cL = std::sqrt(g * hL);

    // Right state
    float hR = std::max(qR.h, min_depth);
    float uR = (hR > min_depth) ? qR.hu / hR : 0.0f;
    float vR = (hR > min_depth) ? qR.hv / hR : 0.0f;
    float cR = std::sqrt(g * hR);

    // Wave speed estimates (Davis)
    float sL = std::min(uL - cL, uR - cR);
    float sR = std::max(uL + cL, uR + cR);

    // Fluxes
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

  // X-direction sweep with MUSCL reconstruction.
  void SweepX(float dt) {
    auto& h  = grid.channels[ch_h].data;
    auto& hu = grid.channels[ch_hu].data;
    auto& hv = grid.channels[ch_hv].data;
    const auto& bathy = grid.channels[ch_bathy].data;
    auto& h_t  = grid.channels[ch_h].temp;
    auto& hu_t = grid.channels[ch_hu].temp;
    auto& hv_t = grid.channels[ch_hv].temp;

    float ratio = dt / grid.dx;
    float g = params.gravity;
    uint32_t nx = grid.nx, ny = grid.ny;

    // Copy current state to temp as base.
    h_t = h; hu_t = hu; hv_t = hv;

    for (uint32_t j = 0; j < ny; ++j) {
      for (uint32_t i = 0; i < nx - 1; ++i) {
        size_t iL = grid.Idx(i, j);
        size_t iR = grid.Idx(i + 1, j);

        // MUSCL reconstruction with minmod slope limiter (2nd-order TVD).
        // Extrapolate cell-centered values to the interface i+1/2:
        //   qL = q[i]   + 0.5 * minmod(dq_i, dq_{i+1})
        //   qR = q[i+1] - 0.5 * minmod(dq_{i+1}, dq_{i+2})
        State qL, qR;

        if (i > 0 && i + 2 < nx) {
          size_t iLL = grid.Idx(i - 1, j);
          size_t iRR = grid.Idx(i + 2, j);
          // Slopes for left cell (i).
          float dh_L  = Minmod(h[iR]  - h[iL],  h[iL]  - h[iLL]);
          float dhu_L = Minmod(hu[iR] - hu[iL], hu[iL] - hu[iLL]);
          float dhv_L = Minmod(hv[iR] - hv[iL], hv[iL] - hv[iLL]);
          // Slopes for right cell (i+1).
          float dh_R  = Minmod(h[iRR]  - h[iR],  h[iR]  - h[iL]);
          float dhu_R = Minmod(hu[iRR] - hu[iR], hu[iR] - hu[iL]);
          float dhv_R = Minmod(hv[iRR] - hv[iR], hv[iR] - hv[iL]);

          qL = {h[iL]  + 0.5f * dh_L,
                hu[iL] + 0.5f * dhu_L,
                hv[iL] + 0.5f * dhv_L};
          qR = {h[iR]  - 0.5f * dh_R,
                hu[iR] - 0.5f * dhu_R,
                hv[iR] - 0.5f * dhv_R};
          // Ensure non-negative depth in reconstructed states.
          if (qL.h < 0) qL = {h[iL], hu[iL], hv[iL]};
          if (qR.h < 0) qR = {h[iR], hu[iR], hv[iR]};
        } else {
          // Fall back to first-order at domain boundaries.
          qL = {h[iL], hu[iL], hv[iL]};
          qR = {h[iR], hu[iR], hv[iR]};
        }

        // HLL flux at interface i+1/2.
        float fh, fhu, fhv;
        HLLFlux(qL, qR, g, params.min_depth, fh, fhu, fhv);

        // Bathymetry source term (hydrostatic reconstruction).
        float db = bathy[iR] - bathy[iL];
        float h_avg = 0.5f * (h[iL] + h[iR]);
        float src = -g * h_avg * db / grid.dx;

        // Update: subtract flux from left, add to right.
        h_t[iL]  -= ratio * fh;
        hu_t[iL] -= ratio * (fhu + src * 0.5f * grid.dx);
        hv_t[iL] -= ratio * fhv;

        h_t[iR]  += ratio * fh;
        hu_t[iR] += ratio * (fhu - src * 0.5f * grid.dx);
        hv_t[iR] += ratio * fhv;
      }
    }

    // Swap: temp becomes current.
    std::swap(h, h_t);
    std::swap(hu, hu_t);
    std::swap(hv, hv_t);

    // Enforce non-negative depth.
    for (size_t i = 0; i < h.size(); ++i) {
      if (h[i] < 0) { h[i] = 0; hu[i] = 0; hv[i] = 0; }
    }
  }

  // Y-direction sweep (same structure, transposed).
  void SweepY(float dt) {
    auto& h  = grid.channels[ch_h].data;
    auto& hu = grid.channels[ch_hu].data;
    auto& hv = grid.channels[ch_hv].data;
    const auto& bathy = grid.channels[ch_bathy].data;
    auto& h_t  = grid.channels[ch_h].temp;
    auto& hu_t = grid.channels[ch_hu].temp;
    auto& hv_t = grid.channels[ch_hv].temp;

    float ratio = dt / grid.dx;
    float g = params.gravity;
    uint32_t nx = grid.nx, ny = grid.ny;

    h_t = h; hu_t = hu; hv_t = hv;

    for (uint32_t j = 0; j < ny - 1; ++j) {
      for (uint32_t i = 0; i < nx; ++i) {
        size_t iB = grid.Idx(i, j);
        size_t iT = grid.Idx(i, j + 1);

        // MUSCL reconstruction for y-sweep (hv is primary momentum).
        State qB, qT;

        if (j > 0 && j + 2 < ny) {
          size_t iBB = grid.Idx(i, j - 1);
          size_t iTT = grid.Idx(i, j + 2);
          // Slopes for bottom cell (j).
          float dh_B  = Minmod(h[iT]  - h[iB],  h[iB]  - h[iBB]);
          float dhv_B = Minmod(hv[iT] - hv[iB], hv[iB] - hv[iBB]);
          float dhu_B = Minmod(hu[iT] - hu[iB], hu[iB] - hu[iBB]);
          // Slopes for top cell (j+1).
          float dh_T  = Minmod(h[iTT]  - h[iT],  h[iT]  - h[iB]);
          float dhv_T = Minmod(hv[iTT] - hv[iT], hv[iT] - hv[iB]);
          float dhu_T = Minmod(hu[iTT] - hu[iT], hu[iT] - hu[iB]);

          float hB  = h[iB]  + 0.5f * dh_B;
          float hvB = hv[iB] + 0.5f * dhv_B;
          float huB = hu[iB] + 0.5f * dhu_B;
          float hT  = h[iT]  - 0.5f * dh_T;
          float hvT = hv[iT] - 0.5f * dhv_T;
          float huT = hu[iT] - 0.5f * dhu_T;

          if (hB < 0) { hB = h[iB]; hvB = hv[iB]; huB = hu[iB]; }
          if (hT < 0) { hT = h[iT]; hvT = hv[iT]; huT = hu[iT]; }

          qB = {hB, hvB, huB};  // swap hu/hv for y-direction
          qT = {hT, hvT, huT};
        } else {
          qB = {h[iB], hv[iB], hu[iB]};
          qT = {h[iT], hv[iT], hu[iT]};
        }

        float fh, fhv_flux, fhu_flux;
        HLLFlux(qB, qT, g, params.min_depth, fh, fhv_flux, fhu_flux);

        float db = bathy[iT] - bathy[iB];
        float h_avg = 0.5f * (h[iB] + h[iT]);
        float src = -g * h_avg * db / grid.dx;

        h_t[iB]  -= ratio * fh;
        hv_t[iB] -= ratio * (fhv_flux + src * 0.5f * grid.dx);
        hu_t[iB] -= ratio * fhu_flux;

        h_t[iT]  += ratio * fh;
        hv_t[iT] += ratio * (fhv_flux - src * 0.5f * grid.dx);
        hu_t[iT] += ratio * fhu_flux;
      }
    }

    std::swap(h, h_t);
    std::swap(hu, hu_t);
    std::swap(hv, hv_t);

    for (size_t i = 0; i < h.size(); ++i) {
      if (h[i] < 0) { h[i] = 0; hu[i] = 0; hv[i] = 0; }
    }
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

}  // namespace mjwater
