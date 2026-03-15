// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Core types, constants, and enums for the mujoco-water fluid engine.

#include <cmath>
#include <cstdint>

namespace mjwater {

// --- Physical constants ---

constexpr float kGravity = 9.80665f;       // m/s^2 (standard gravity)
constexpr float kWaterDensity = 998.2f;    // kg/m^3 at 20C, 1 atm
constexpr float kWaterViscosity = 1.004e-6f; // m^2/s kinematic viscosity at 20C
constexpr float kPi = 3.14159265358979f;
constexpr float kTwoPi = 2.0f * kPi;

// --- LOD levels ---
// Ordered coarsest-to-finest. Higher values = finer resolution.
// 5 levels spanning ~10 orders of magnitude: ocean (km) to cellular (um).

enum class LODLevel : uint8_t {
  kOcean        = 0,  // Spectral ocean (Gerstner waves, Phillips spectrum)
  kShallowWater = 1,  // 2D height field (Saint-Venant equations)
  kSPH          = 2,  // 3D smoothed particle hydrodynamics
  kLBM          = 3,  // 3D lattice Boltzmann (D3Q19)
  kStokes       = 4,  // Stokes creeping flow (Re << 1, cellular scale)
};

constexpr int kNumLODLevels = 5;

// --- Geometry ---

struct Vec3 {
  float x = 0, y = 0, z = 0;

  Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
  Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  Vec3 operator/(float s) const { float inv = 1.0f / s; return {x * inv, y * inv, z * inv}; }
  Vec3& operator+=(Vec3 b) { x += b.x; y += b.y; z += b.z; return *this; }
  Vec3& operator-=(Vec3 b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
  Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

  float Dot(Vec3 b) const { return x * b.x + y * b.y + z * b.z; }
  Vec3 Cross(Vec3 b) const { return {y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x}; }
  float LengthSq() const { return x * x + y * y + z * z; }
  float Length() const { return std::sqrt(LengthSq()); }
  Vec3 Normalized() const { float len = Length(); return len > 0 ? *this / len : Vec3{}; }
};

inline Vec3 operator*(float s, Vec3 v) { return v * s; }

struct AABB {
  Vec3 min, max;

  Vec3 Center() const { return (min + max) * 0.5f; }
  Vec3 Size() const { return max - min; }
  float Volume() const { auto s = Size(); return s.x * s.y * s.z; }

  bool Contains(Vec3 p) const {
    return p.x >= min.x && p.x <= max.x &&
           p.y >= min.y && p.y <= max.y &&
           p.z >= min.z && p.z <= max.z;
  }
};

}  // namespace mjwater
