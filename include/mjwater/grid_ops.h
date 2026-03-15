// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Shared grid operations: coordinate transforms, interpolation, stencils.
//
// Eliminates duplication across HeightField, LatticeGrid, StokesGrid,
// and coupling code. All grid types use these free functions.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mjwater/types.h"

namespace mjwater {

// --- Coordinate transforms (2D) ---

inline void WorldToGrid2D(float wx, float wy,
                           float origin_x, float origin_y, float dx,
                           float& gx, float& gy) {
  gx = (wx - origin_x) / dx;
  gy = (wy - origin_y) / dx;
}

inline bool WorldToGridCell2D(float wx, float wy,
                               float origin_x, float origin_y, float dx,
                               uint32_t nx, uint32_t ny,
                               uint32_t& cx, uint32_t& cy) {
  float gx = (wx - origin_x) / dx;
  float gy = (wy - origin_y) / dx;
  if (gx < 0 || gy < 0) return false;
  cx = static_cast<uint32_t>(gx);
  cy = static_cast<uint32_t>(gy);
  return cx < nx && cy < ny;
}

inline void GridToWorld2D(uint32_t x, uint32_t y,
                           float origin_x, float origin_y, float dx,
                           float& wx, float& wy) {
  wx = origin_x + (x + 0.5f) * dx;
  wy = origin_y + (y + 0.5f) * dx;
}

// --- Coordinate transforms (3D) ---

inline void WorldToGrid3D(float wx, float wy, float wz,
                           float origin_x, float origin_y, float origin_z,
                           float dx,
                           float& gx, float& gy, float& gz) {
  gx = (wx - origin_x) / dx;
  gy = (wy - origin_y) / dx;
  gz = (wz - origin_z) / dx;
}

inline bool WorldToGridCell3D(float wx, float wy, float wz,
                               float origin_x, float origin_y, float origin_z,
                               float dx,
                               uint32_t nx, uint32_t ny, uint32_t nz,
                               uint32_t& cx, uint32_t& cy, uint32_t& cz) {
  float gx = (wx - origin_x) / dx;
  float gy = (wy - origin_y) / dx;
  float gz = (wz - origin_z) / dx;
  if (gx < 0 || gy < 0 || gz < 0) return false;
  cx = static_cast<uint32_t>(gx);
  cy = static_cast<uint32_t>(gy);
  cz = static_cast<uint32_t>(gz);
  return cx < nx && cy < ny && cz < nz;
}

inline void GridToWorld3D(uint32_t x, uint32_t y, uint32_t z,
                           float origin_x, float origin_y, float origin_z,
                           float dx,
                           float& wx, float& wy, float& wz) {
  wx = origin_x + (x + 0.5f) * dx;
  wy = origin_y + (y + 0.5f) * dx;
  wz = origin_z + (z + 0.5f) * dx;
}

inline bool Contains3D(Vec3 pos,
                        float origin_x, float origin_y, float origin_z,
                        float dx, uint32_t nx, uint32_t ny, uint32_t nz) {
  float gx = (pos.x - origin_x) / dx;
  float gy = (pos.y - origin_y) / dx;
  float gz = (pos.z - origin_z) / dx;
  return gx >= 0 && gx < nx && gy >= 0 && gy < ny && gz >= 0 && gz < nz;
}

// --- 3D index ---

inline size_t Idx3D(uint32_t x, uint32_t y, uint32_t z,
                     uint32_t nx, uint32_t ny) {
  return (static_cast<size_t>(z) * ny + y) * nx + x;
}

// --- Trilinear interpolation (3D scalar field) ---

inline float TrilinearScalar(float gx, float gy, float gz,
                              const std::vector<float>& field,
                              uint32_t nx, uint32_t ny, uint32_t nz) {
  int x0 = static_cast<int>(std::floor(gx));
  int y0 = static_cast<int>(std::floor(gy));
  int z0 = static_cast<int>(std::floor(gz));
  float fx = gx - x0; float fy = gy - y0; float fz = gz - z0;

  auto cl = [](int v, int mx) -> uint32_t {
    return static_cast<uint32_t>(std::clamp(v, 0, mx - 1));
  };
  int NX = static_cast<int>(nx);
  int NY = static_cast<int>(ny);
  int NZ = static_cast<int>(nz);

  auto idx = [nx, ny](uint32_t x, uint32_t y, uint32_t z) -> size_t {
    return Idx3D(x, y, z, nx, ny);
  };

  float c000 = field[idx(cl(x0,NX),   cl(y0,NY),   cl(z0,NZ))];
  float c100 = field[idx(cl(x0+1,NX), cl(y0,NY),   cl(z0,NZ))];
  float c010 = field[idx(cl(x0,NX),   cl(y0+1,NY), cl(z0,NZ))];
  float c110 = field[idx(cl(x0+1,NX), cl(y0+1,NY), cl(z0,NZ))];
  float c001 = field[idx(cl(x0,NX),   cl(y0,NY),   cl(z0+1,NZ))];
  float c101 = field[idx(cl(x0+1,NX), cl(y0,NY),   cl(z0+1,NZ))];
  float c011 = field[idx(cl(x0,NX),   cl(y0+1,NY), cl(z0+1,NZ))];
  float c111 = field[idx(cl(x0+1,NX), cl(y0+1,NY), cl(z0+1,NZ))];

  float c00 = c000 * (1-fx) + c100 * fx;
  float c10 = c010 * (1-fx) + c110 * fx;
  float c01 = c001 * (1-fx) + c101 * fx;
  float c11 = c011 * (1-fx) + c111 * fx;
  float c0  = c00 * (1-fy)  + c10 * fy;
  float c1  = c01 * (1-fy)  + c11 * fy;
  return c0 * (1-fz) + c1 * fz;
}

// Trilinear interpolation of a 3D vector field (3 separate arrays).
inline Vec3 TrilinearVec3(float gx, float gy, float gz,
                           const std::vector<float>& fu,
                           const std::vector<float>& fv,
                           const std::vector<float>& fw,
                           uint32_t nx, uint32_t ny, uint32_t nz) {
  return {
    TrilinearScalar(gx, gy, gz, fu, nx, ny, nz),
    TrilinearScalar(gx, gy, gz, fv, nx, ny, nz),
    TrilinearScalar(gx, gy, gz, fw, nx, ny, nz)
  };
}

// --- 3D Laplacian (6-point stencil) ---
// Returns the discrete Laplacian at cell (x,y,z) without the 1/dx^2 factor.
// Caller multiplies by alpha = nu*dt/dx^2 or similar.

inline float Laplacian6(const std::vector<float>& field,
                         uint32_t x, uint32_t y, uint32_t z,
                         uint32_t nx, uint32_t ny) {
  size_t c = Idx3D(x, y, z, nx, ny);
  return field[Idx3D(x+1,y,z, nx,ny)] + field[Idx3D(x-1,y,z, nx,ny)] +
         field[Idx3D(x,y+1,z, nx,ny)] + field[Idx3D(x,y-1,z, nx,ny)] +
         field[Idx3D(x,y,z+1, nx,ny)] + field[Idx3D(x,y,z-1, nx,ny)] -
         6.0f * field[c];
}

// --- 3D gradient (central difference) ---
// Returns gradient without the 1/(2*dx) factor.

inline Vec3 Gradient6(const std::vector<float>& field,
                       uint32_t x, uint32_t y, uint32_t z,
                       uint32_t nx, uint32_t ny) {
  return {
    field[Idx3D(x+1,y,z, nx,ny)] - field[Idx3D(x-1,y,z, nx,ny)],
    field[Idx3D(x,y+1,z, nx,ny)] - field[Idx3D(x,y-1,z, nx,ny)],
    field[Idx3D(x,y,z+1, nx,ny)] - field[Idx3D(x,y,z-1, nx,ny)]
  };
}

// --- Clear 3D velocity fields ---

inline void ClearVelocityFields(std::vector<float>& u,
                                 std::vector<float>& v,
                                 std::vector<float>& w) {
  std::fill(u.begin(), u.end(), 0.0f);
  std::fill(v.begin(), v.end(), 0.0f);
  std::fill(w.begin(), w.end(), 0.0f);
}

// --- Zero velocity at all 6 domain faces ---

inline void ZeroFaceVelocities(std::vector<float>& u,
                                std::vector<float>& v,
                                std::vector<float>& w,
                                uint32_t nx, uint32_t ny, uint32_t nz) {
  auto zero = [&](uint32_t x, uint32_t y, uint32_t z) {
    size_t c = Idx3D(x, y, z, nx, ny);
    u[c] = 0; v[c] = 0; w[c] = 0;
  };
  // X faces
  for (uint32_t z = 0; z < nz; ++z)
    for (uint32_t y = 0; y < ny; ++y) {
      zero(0, y, z);
      zero(nx - 1, y, z);
    }
  // Y faces
  for (uint32_t z = 0; z < nz; ++z)
    for (uint32_t x = 0; x < nx; ++x) {
      zero(x, 0, z);
      zero(x, ny - 1, z);
    }
  // Z faces
  for (uint32_t y = 0; y < ny; ++y)
    for (uint32_t x = 0; x < nx; ++x) {
      zero(x, y, 0);
      zero(x, y, nz - 1);
    }
}

}  // namespace mjwater
