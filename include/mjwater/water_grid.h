// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// 2D and 3D spatial grids for water simulation.
//
// HeightField: 2D grid for shallow water equations (LOD 0).
//   Stores water depth h(x,y), momentum hu, hv, and bathymetry.
//
// LatticeGrid: 3D grid for lattice Boltzmann method (LOD 2).
//   Stores D3Q19 distribution functions per cell.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mjwater/types.h"

namespace mjwater {

// --- 2D Height Field Grid ---

struct HeightField {
  uint32_t nx = 0, ny = 0;
  float dx = 0.1f;          // cell size (m)
  float origin_x = 0, origin_y = 0;  // world-space origin

  // Named channels (flat arrays of nx*ny floats).
  struct Channel {
    std::string name;
    std::vector<float> data;
    std::vector<float> temp;  // scratch buffer for updates
  };
  std::vector<Channel> channels;

  void Init(uint32_t nx_, uint32_t ny_, float dx_,
            float ox = 0, float oy = 0) {
    nx = nx_; ny = ny_; dx = dx_;
    origin_x = ox; origin_y = oy;
    channels.clear();
  }

  size_t Idx(uint32_t x, uint32_t y) const {
    return static_cast<size_t>(y) * nx + x;
  }

  size_t AddChannel(const std::string& name, float init_val = 0.0f) {
    size_t idx = channels.size();
    Channel ch;
    ch.name = name;
    ch.data.assign(static_cast<size_t>(nx) * ny, init_val);
    ch.temp.resize(ch.data.size());
    channels.push_back(std::move(ch));
    return idx;
  }

  int FindChannel(const std::string& name) const {
    for (size_t i = 0; i < channels.size(); ++i)
      if (channels[i].name == name) return static_cast<int>(i);
    return -1;
  }

  size_t CellCount() const { return static_cast<size_t>(nx) * ny; }

  // World -> grid coordinate (continuous)
  void WorldToGrid(float wx, float wy, float& gx, float& gy) const {
    gx = (wx - origin_x) / dx;
    gy = (wy - origin_y) / dx;
  }

  // World -> grid cell index (returns false if outside).
  bool WorldToGridCell(float wx, float wy, uint32_t& cx, uint32_t& cy) const {
    float gx = (wx - origin_x) / dx;
    float gy = (wy - origin_y) / dx;
    if (gx < 0 || gy < 0) return false;
    cx = static_cast<uint32_t>(gx);
    cy = static_cast<uint32_t>(gy);
    return cx < nx && cy < ny;
  }

  // Grid -> world coordinate (cell center)
  void GridToWorld(uint32_t x, uint32_t y, float& wx, float& wy) const {
    wx = origin_x + (x + 0.5f) * dx;
    wy = origin_y + (y + 0.5f) * dx;
  }

  // Bilinear interpolation of a channel at world position.
  float Sample(size_t ch, float wx, float wy) const {
    float gx, gy;
    WorldToGrid(wx, wy, gx, gy);
    gx -= 0.5f; gy -= 0.5f;  // shift to cell corners

    int x0 = static_cast<int>(std::floor(gx));
    int y0 = static_cast<int>(std::floor(gy));
    float fx = gx - x0;
    float fy = gy - y0;

    auto clamp_x = [&](int x) -> uint32_t {
      return static_cast<uint32_t>(std::clamp(x, 0, static_cast<int>(nx) - 1));
    };
    auto clamp_y = [&](int y) -> uint32_t {
      return static_cast<uint32_t>(std::clamp(y, 0, static_cast<int>(ny) - 1));
    };

    const auto& d = channels[ch].data;
    float v00 = d[Idx(clamp_x(x0),     clamp_y(y0))];
    float v10 = d[Idx(clamp_x(x0 + 1), clamp_y(y0))];
    float v01 = d[Idx(clamp_x(x0),     clamp_y(y0 + 1))];
    float v11 = d[Idx(clamp_x(x0 + 1), clamp_y(y0 + 1))];

    return v00 * (1 - fx) * (1 - fy) +
           v10 * fx * (1 - fy) +
           v01 * (1 - fx) * fy +
           v11 * fx * fy;
  }
};

// --- 3D Lattice Grid (for LBM) ---

// D3Q19 lattice velocity vectors.
// Index 0 is rest; indices 1-6 are face neighbors; 7-18 are edge neighbors.
constexpr int kQ = 19;

// Lattice velocity components: e[i] = (ex[i], ey[i], ez[i])
constexpr int kEx[kQ] = { 0,  1,-1, 0, 0, 0, 0,  1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
constexpr int kEy[kQ] = { 0,  0, 0, 1,-1, 0, 0,  1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
constexpr int kEz[kQ] = { 0,  0, 0, 0, 0, 1,-1,  0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};

// Lattice weights: w[0]=1/3, w[1..6]=1/18, w[7..18]=1/36
constexpr float kW[kQ] = {
  1.0f/3.0f,
  1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
  1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
  1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
};

// Opposite direction index for bounce-back: opp[i] swaps +/- direction.
constexpr int kOpp[kQ] = {0,  2,1, 4,3, 6,5,  8,7, 10,9, 12,11, 14,13, 16,15, 18,17};

// Speed of sound squared in lattice units: c_s^2 = 1/3.
constexpr float kCsSq = 1.0f / 3.0f;

struct LatticeGrid {
  uint32_t nx = 0, ny = 0, nz = 0;
  float dx = 0.001f;         // lattice spacing (m)
  float origin_x = 0, origin_y = 0, origin_z = 0;

  // Distribution functions: f[cell * kQ + i] for direction i.
  // Two copies for ping-pong streaming.
  std::vector<float> f_src;
  std::vector<float> f_dst;

  // Solid mask: 1 = solid (bounce-back), 0 = fluid.
  std::vector<uint8_t> solid;

  void Init(uint32_t nx_, uint32_t ny_, uint32_t nz_, float dx_,
            float ox = 0, float oy = 0, float oz = 0) {
    nx = nx_; ny = ny_; nz = nz_; dx = dx_;
    origin_x = ox; origin_y = oy; origin_z = oz;
    size_t n = CellCount();
    f_src.assign(n * kQ, 0.0f);
    f_dst.assign(n * kQ, 0.0f);
    solid.assign(n, 0);

    // Initialize to equilibrium at rest density, zero velocity.
    for (size_t c = 0; c < n; ++c) {
      for (int i = 0; i < kQ; ++i) {
        f_src[c * kQ + i] = kW[i];  // rho=1 in lattice units
      }
    }
  }

  size_t CellCount() const {
    return static_cast<size_t>(nx) * ny * nz;
  }

  size_t Idx(uint32_t x, uint32_t y, uint32_t z) const {
    return (static_cast<size_t>(z) * ny + y) * nx + x;
  }

  // Check if a world position is inside the grid domain.
  bool Contains(Vec3 pos) const {
    float gx = (pos.x - origin_x) / dx;
    float gy = (pos.y - origin_y) / dx;
    float gz = (pos.z - origin_z) / dx;
    return gx >= 0 && gx < nx && gy >= 0 && gy < ny && gz >= 0 && gz < nz;
  }

  void WorldToGrid(float wx, float wy, float wz,
                    float& gx, float& gy, float& gz) const {
    gx = (wx - origin_x) / dx;
    gy = (wy - origin_y) / dx;
    gz = (wz - origin_z) / dx;
  }

  void GridToWorld(uint32_t x, uint32_t y, uint32_t z,
                    float& wx, float& wy, float& wz) const {
    wx = origin_x + (x + 0.5f) * dx;
    wy = origin_y + (y + 0.5f) * dx;
    wz = origin_z + (z + 0.5f) * dx;
  }

  // Compute macroscopic density at a cell.
  float Density(size_t cell_idx) const {
    float rho = 0;
    const float* fi = f_src.data() + cell_idx * kQ;
    for (int i = 0; i < kQ; ++i) rho += fi[i];
    return rho;
  }

  // Compute macroscopic velocity at a cell.
  Vec3 Velocity(size_t cell_idx) const {
    const float* fi = f_src.data() + cell_idx * kQ;
    float rho = 0;
    Vec3 u{};
    for (int i = 0; i < kQ; ++i) {
      rho += fi[i];
      u.x += fi[i] * kEx[i];
      u.y += fi[i] * kEy[i];
      u.z += fi[i] * kEz[i];
    }
    if (rho > 1e-10f) {
      u.x /= rho; u.y /= rho; u.z /= rho;
    }
    return u;
  }
};

}  // namespace mjwater
