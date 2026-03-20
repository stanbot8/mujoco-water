// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Convenience header: includes the entire mujoco-water library.
//
// For finer-grained control, include individual headers instead:
//   mjwater/types.h           - Vec3, AABB, LODLevel, physical constants
//   mjwater/spectral_ocean.h  - Gerstner waves, Phillips/JONSWAP spectrum
//   mjwater/shallow_water.h   - HLL Riemann solver, MUSCL reconstruction
//   mjwater/sph.h             - WCSPH with Wendland C2 kernel
//   mjwater/lbm.h             - D3Q19 BGK lattice Boltzmann
//   mjwater/stokes.h          - Stokes creeping flow, concentration diffusion
//   mjwater/water_grid.h      - HeightField (2D) + LatticeGrid (3D)
//   mjwater/water_sdf.h       - Volume definition via SDF primitives
//   mjwater/lod_manager.h     - Distance-based 5-level LOD zones
//   mjwater/lod_transition.h  - Bidirectional state transfer between LODs
//   mjwater/coupling.h        - MuJoCo body-fluid coupling
//   mjwater/water_engine.h    - Top-level orchestrator

#include "mjwater/water_engine.h"
