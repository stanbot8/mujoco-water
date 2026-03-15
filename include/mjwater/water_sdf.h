// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Water volume definition via signed distance field primitives.
//
// Negative = inside water, positive = outside.
// Provides preset configurations (pool, lake, microchannel) and
// baking to HeightField or LatticeGrid.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "mjwater/types.h"
#include "mjwater/water_grid.h"

namespace mjwater {

struct WaterPrimitive {
  std::string name;
  enum class Shape { kBox, kSphere, kCylinder };
  Shape shape = Shape::kBox;
  float cx = 0, cy = 0, cz = 0;  // center (m)
  float rx = 1, ry = 1, rz = 1;  // half-extents (m)

  // SDF evaluation for this primitive.
  float Evaluate(float x, float y, float z) const {
    float px = x - cx, py = y - cy, pz = z - cz;
    switch (shape) {
      case Shape::kBox: {
        float dx = std::abs(px) - rx;
        float dy = std::abs(py) - ry;
        float dz = std::abs(pz) - rz;
        float outside = std::sqrt(
            std::max(0.0f, dx) * std::max(0.0f, dx) +
            std::max(0.0f, dy) * std::max(0.0f, dy) +
            std::max(0.0f, dz) * std::max(0.0f, dz));
        float inside = std::min(0.0f, std::max({dx, dy, dz}));
        return outside + inside;
      }
      case Shape::kSphere: {
        float r = std::sqrt(px * px + py * py + pz * pz);
        return r - rx;  // rx = radius
      }
      case Shape::kCylinder: {
        // Cylinder along z-axis: radius rx, half-height rz
        float r_xy = std::sqrt(px * px + py * py) - rx;
        float h_z = std::abs(pz) - rz;
        float outside = std::sqrt(
            std::max(0.0f, r_xy) * std::max(0.0f, r_xy) +
            std::max(0.0f, h_z) * std::max(0.0f, h_z));
        float inside = std::min(0.0f, std::max(r_xy, h_z));
        return outside + inside;
      }
    }
    return 1e10f;
  }
};

struct WaterSDF {
  std::vector<WaterPrimitive> primitives;
  float smooth_k = 0.1f;  // smooth union blend radius (m)

  // Smooth minimum for blending primitives (polynomial smooth min).
  static float SmoothMin(float a, float b, float k) {
    if (k <= 0) return std::min(a, b);
    float h = std::max(0.0f, k - std::abs(a - b)) / k;
    return std::min(a, b) - h * h * k * 0.25f;
  }

  // Evaluate the combined SDF at a world position.
  float Evaluate(float x, float y, float z) const {
    if (primitives.empty()) return 1e10f;
    float d = primitives[0].Evaluate(x, y, z);
    for (size_t i = 1; i < primitives.size(); ++i) {
      d = SmoothMin(d, primitives[i].Evaluate(x, y, z), smooth_k);
    }
    return d;
  }

  // Get the bounding box of all primitives (with margin).
  AABB Bounds(float margin = 0.5f) const {
    if (primitives.empty()) return {};
    AABB bb;
    bb.min = {1e10f, 1e10f, 1e10f};
    bb.max = {-1e10f, -1e10f, -1e10f};
    for (const auto& p : primitives) {
      float r = std::max({p.rx, p.ry, p.rz});
      bb.min.x = std::min(bb.min.x, p.cx - r);
      bb.min.y = std::min(bb.min.y, p.cy - r);
      bb.min.z = std::min(bb.min.z, p.cz - r);
      bb.max.x = std::max(bb.max.x, p.cx + r);
      bb.max.y = std::max(bb.max.y, p.cy + r);
      bb.max.z = std::max(bb.max.z, p.cz + r);
    }
    bb.min.x -= margin; bb.min.y -= margin; bb.min.z -= margin;
    bb.max.x += margin; bb.max.y += margin; bb.max.z += margin;
    return bb;
  }

  // --- Preset configurations ---

  // Rectangular pool.
  void InitPool(float length, float width, float depth) {
    primitives.clear();
    WaterPrimitive p;
    p.name = "pool";
    p.shape = WaterPrimitive::Shape::kBox;
    p.cx = length / 2; p.cy = width / 2; p.cz = -depth / 2;
    p.rx = length / 2; p.ry = width / 2; p.rz = depth / 2;
    primitives.push_back(p);
  }

  // Circular lake.
  void InitLake(float radius, float depth) {
    primitives.clear();
    WaterPrimitive p;
    p.name = "lake";
    p.shape = WaterPrimitive::Shape::kCylinder;
    p.cx = 0; p.cy = 0; p.cz = -depth / 2;
    p.rx = radius; p.rz = depth / 2;
    primitives.push_back(p);
  }

  // Microchannel for cellular-scale flows.
  void InitMicrochannel(float length, float width, float height) {
    primitives.clear();
    WaterPrimitive p;
    p.name = "microchannel";
    p.shape = WaterPrimitive::Shape::kBox;
    p.cx = length / 2; p.cy = width / 2; p.cz = height / 2;
    p.rx = length / 2; p.ry = width / 2; p.rz = height / 2;
    primitives.push_back(p);
  }

  // Bake SDF into a HeightField bathymetry channel.
  // For each (x,y) cell, finds the bed elevation: the lowest z where the SDF
  // is inside water (d <= 0). Points outside the SDF horizontally get a high
  // bed elevation (effectively no water).
  void BakeToHeightField(HeightField& grid, size_t sdf_channel) const {
    AABB bb = Bounds(0.0f);
    auto& data = grid.channels[sdf_channel].data;
    float z_top = bb.max.z;
    float z_bot = bb.min.z;
    // Search resolution: use grid dx or finer.
    float dz = grid.dx * 0.5f;
    int nz = std::max(1, static_cast<int>((z_top - z_bot) / dz));

    for (uint32_t y = 0; y < grid.ny; ++y) {
      for (uint32_t x = 0; x < grid.nx; ++x) {
        float wx, wy;
        grid.GridToWorld(x, y, wx, wy);

        // Search downward from top to find the water floor.
        float bed = z_top;  // default: no water (bed at top)
        for (int iz = 0; iz <= nz; ++iz) {
          float z = z_bot + iz * dz;
          if (Evaluate(wx, wy, z) <= 0) {
            bed = z;
            break;
          }
        }
        data[grid.Idx(x, y)] = bed;
      }
    }
  }

  // Bake SDF into LatticeGrid solid mask.
  void BakeToLattice(LatticeGrid& grid) const {
    for (uint32_t z = 0; z < grid.nz; ++z) {
      for (uint32_t y = 0; y < grid.ny; ++y) {
        for (uint32_t x = 0; x < grid.nx; ++x) {
          float wx, wy, wz;
          grid.GridToWorld(x, y, z, wx, wy, wz);
          float d = Evaluate(wx, wy, wz);
          // Outside water (d > 0) is solid for LBM.
          grid.solid[grid.Idx(x, y, z)] = (d > 0) ? 1 : 0;
        }
      }
    }
  }
};

}  // namespace mjwater
