# mujoco-water

Header-only C++20 multiscale water simulation for [MuJoCo](https://mujoco.org/). Five concurrent LOD levels span roughly 10 orders of magnitude: spectral ocean (km) down to cellular-scale Stokes flow (um). The water surface renders as a single solid hfield volume. Forces apply via `data->xfrc_applied`; disable MuJoCo's built-in fluid model (`opt.density = 0; opt.viscosity = 0;`) to avoid double-counting.

![Multiscale demo](docs/Screenshot.png)

MIT licensed.

---

## LOD Levels

| LOD | Method | Scale | Timestep | Physics |
|-----|--------|-------|----------|---------|
| 0 | Spectral Ocean | km, wavelength 1-300 m | master dt | Gerstner waves, Phillips/JONSWAP spectrum |
| 1 | Shallow Water (SWE) | pools to 1 km, dx 10 cm to 10 m | ~10 ms | HLL Riemann solver, MUSCL reconstruction, Manning friction |
| 2 | Smoothed Particle Hydrodynamics (SPH) | splashes, waves, ~cm | ~1 ms | WCSPH, Wendland C2 kernel, Tait EOS, XSPH, CSF surface tension |
| 3 | Lattice Boltzmann (LBM) | microfluidics, dx ~1 mm | ~0.01 ms | D3Q19 TRT collision, Guo forcing, bounce-back BCs |
| 4 | Stokes Flow | cellular/capillary, dx ~10 um | ~1 us | Creeping flow (Re << 1), concentration diffusion with source terms |

Ocean provides the far-field background. SWE covers the near field. SPH, LBM, and Stokes activate in progressively smaller focus zones and substep within their parent.

### Default zone radii (from focus point)

| LOD | Zone radius |
|-----|------------|
| 4 Stokes | 5 mm |
| 3 LBM | 5 cm |
| 2 SPH | 50 cm |
| 1 SWE | 50 m |
| 0 Ocean | beyond SWE |

---

## Architecture

The `Step()` pipeline runs the full hierarchy each frame:

```
Step(master_dt):
  0. RepositionGrids()    move LBM/Stokes grids if focus has drifted
  1. UpdateLOD()          5-level zone check; fire escalation/de-escalation
  2. Ocean step           advance spectral waves; couple to SWE boundary edges
  3. SWE step (MUSCL)     2nd-order TVD step at master_dt
  4. SPH substeps         N_sph steps per master step
     a. SWE->SPH coupling   ghost zone velocity nudging
     b. SPH step             neighbor search, density, Tait pressure,
                             forces + surface tension, XSPH leapfrog
     c. LBM substeps         N_lbm steps per SPH step
        i.  SPH->LBM coupling  ghost cell equilibrium nudging
        ii. LBM step           collide (+ bounce-back) + stream
        iii. Stokes substeps   N_stokes steps per LBM step
             - LBM->Stokes coupling    boundary velocity nudging
             - Stokes step             diffuse, pressure project,
                                       advect-diffuse concentration
  5. MuJoCo coupling      buoyancy + drag + added mass -> xfrc_applied
```

The water surface is rendered as a single solid hfield volume (`z_bottom = 50`; no separate box). Wireframe overlay uses chunk-based LOD: coarse everywhere, fine near the focus point.

---

## Library Headers

All headers live in `include/mjwater/`. Include via `#include "mjwater/<header>.h"`.

| Header | Description |
|--------|-------------|
| `mjwater.h` | Convenience umbrella: includes the entire library |
| `types.h` | Vec3, AABB, LODLevel enum (5 levels), physical constants |
| `spectral_ocean.h` | Gerstner waves, Phillips/JONSWAP spectrum, orbital velocity |
| `shallow_water.h` | HLL Riemann solver, MUSCL reconstruction, Manning friction |
| `sph.h` | WCSPH: Tait EOS, XSPH smoothing, CSF surface tension, Shepard density reinitialization, adaptive splitting |
| `lbm.h` | D3Q19 TRT collision, Guo body forcing, bounce-back boundary conditions |
| `stokes.h` | Stokes creeping flow, Red-Black Gauss-Seidel pressure projection, scalar concentration field |
| `water_grid.h` | HeightField (2D) and LatticeGrid (3D D3Q19) |
| `water_sdf.h` | Volume definition via SDF primitives |
| `lod_manager.h` | Distance-based 5-level zone assignment with hysteresis |
| `lod_transition.h` | Bidirectional state transfer across all LOD pairs with mass and momentum conservation |
| `coupling.h` | MuJoCo body-fluid coupling: buoyancy, drag, added mass |
| `water_engine.h` | Top-level orchestrator |
| `mujoco_utils.h` | MuJoCo helpers: hfield update, force application |

---

## Quick Start

### Tests only (no MuJoCo required)

```bash
cmake -B build -DMJWATER_TESTS_ONLY=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

### Full build with MuJoCo

[mujoco-game](https://github.com/stanbot8/mujoco-game) is fetched automatically via CMake FetchContent (or used from a sibling directory if present):

```bash
cmake -B build -DMJWATER_TESTS=ON
cmake --build build --config Release
```

### Multiscale demo

Interactive 1 km x 1 km ocean with all 5 LOD levels running simultaneously. Scroll from ocean scale (km) down to Stokes scale (um). Dear ImGui is fetched automatically via CMake FetchContent and is linked only to demo targets.

```bash
cmake -B build -DMJWATER_DEMO=ON -DMUJOCO_DIR=<path-to-mujoco>
cmake --build build --config Release --target multiscale_demo
./build/Release/multiscale_demo
```

Camera controls:

- Left drag: orbit
- Right drag: pan (XY)
- Shift + right drag: adjust height
- Scroll: zoom

The ImGui panel provides:

- Simulation: play/pause, sim time, reset, wireframe overlay with edge sensitivity
- Camera: distance, look-at position, elevation, azimuth
- LOD Layers: per-layer enable/disable, cell count, compute cost bars
- Ocean: wind speed, choppiness, JONSWAP gamma sliders
- Zone Radii: live adjustment of LOD activation distances
- Export To Project: 5 presets (swimming robot, pool/tank, ocean vehicle, microfluidics, full multiscale), feature toggles (surface tension, chemotaxis, body coupling, Python bindings), and export to a ready-to-compile folder

---

## Integration

The library is header-only with no rendering or GUI code. Only the headers in `include/mjwater/` are needed.

### Option A: add_subdirectory

```cmake
add_subdirectory(path/to/mujoco-water)
target_link_libraries(my_app PRIVATE mujoco-water)
```

### Option B: install and find_package

```bash
cmake -B build && cmake --install build --prefix /usr/local
```

```cmake
find_package(mujoco-water REQUIRED)
target_link_libraries(my_app PRIVATE mjwater::mujoco-water)
```

### Option C: copy headers

Copy `include/mjwater/` into your project. No build system integration needed.

### configure.py

An interactive CLI wizard that generates a ready-to-compile project folder. Run `python configure.py` and follow the prompts: select a backend (MuJoCo, Chrono, Standalone), choose a preset, and toggle features. The output folder contains only the headers and CMake config you selected.

---

## Usage

### MuJoCo integration

```cpp
#include <mujoco/mujoco.h>
#include "mjwater/mjwater.h"
#include "mjwater/mujoco_utils.h"

// Disable MuJoCo's built-in fluid model to avoid double-counting.
// m->opt.density = 0; m->opt.viscosity = 0;

mjwater::WaterEngineConfig cfg;
cfg.swe_nx = 40;  cfg.swe_ny = 40;  cfg.swe_dx = 0.1f;
cfg.initial_surface_z = 0.5f;
cfg.domain = {{-2, -2, -1}, {2, 2, 2}};

mjwater::WaterEngine water;
water.Init(cfg);

// Register each MuJoCo body to couple with fluid.
int hull_id = mj_name2id(m, mjOBJ_BODY, "hull");
mjwater::BodyCoupling hull;
hull.volume          = 0.05f;  // displaced volume (m^3)
hull.cross_section   = 0.1f;   // reference area (m^2)
hull.drag_coeff      = 0.8f;   // Cd
hull.added_mass_coeff = 0.5f;  // Ca (0.5 for sphere)
size_t hull_idx = water.AddBody(hull);

// In the simulation loop:
water.SetFocus({d->xpos[3*hull_id], d->xpos[3*hull_id+1],
                d->xpos[3*hull_id+2]});
water.Step();

mjwater::Vec3 body_pos = {d->xpos[3*hull_id],   d->xpos[3*hull_id+1],
                           d->xpos[3*hull_id+2]};
mjwater::Vec3 body_vel = {d->cvel[6*hull_id+3], d->cvel[6*hull_id+4],
                           d->cvel[6*hull_id+5]};
auto force = water.ComputeBodyForce(hull_idx, body_pos, body_vel,
                                    m->opt.timestep);
mjwater::ApplyFluidForces(d, hull_id, force);

// Update hfield for rendering (requires render context).
int hfield_id = mj_name2id(m, mjOBJ_HFIELD, "water_surface");
mjwater::UpdateMuJoCoHField(m, &con, water, hfield_id, max_water_z);

mj_step(m, d);
```

### Adaptive LOD (ocean + zoom)

```cpp
mjwater::WaterEngineConfig cfg;
cfg.ocean_enabled       = true;
cfg.ocean.wind_speed    = 15.0f;   // m/s
cfg.ocean.jonswap_gamma = 3.3f;
cfg.ocean_mean_depth    = 50.0f;   // m

mjwater::WaterEngine water;
water.Init(cfg);

// LOD zones activate around the focus point.
// Moving the focus inward activates finer layers (SPH, LBM, Stokes).
water.SetFocus(camera_position);
water.Step();

// Query fluid state at any point (highest active LOD used).
auto state = water.Query({1.0f, 0.0f, 0.0f});
// state.velocity, state.density, state.depth, state.submerged
```

### Standalone Stokes flow (no MuJoCo)

```cpp
#include "mjwater/stokes.h"

mjwater::StokesSolver solver;
mjwater::StokesParams p;
p.dx         = 5e-6f;    // 5 um grid spacing
p.viscosity  = 3.5e-3f;  // blood plasma (Pa*s)
p.diffusivity = 1e-10f;  // protein diffusion coefficient
solver.Init(40, 40, 40, p);

solver.SetSource({20, 20, 20}, 1e3f);
solver.SetBodyForce({100.0f, 0.0f, 0.0f});

float dt = p.StableDt();
for (int i = 0; i < 1000; ++i) solver.Step(dt);
float c = solver.Concentration({100e-6f, 100e-6f, 100e-6f});
```

---

## Test Suites

7 suites, 63 tests. All pass without MuJoCo.

| Suite | File | Coverage |
|-------|------|----------|
| Shallow water | `test_shallow_water.cc` | Mass conservation, dam break shock, Ritter analytical validation, quiescent stability, velocity query |
| SPH | `test_sph.cc` | Kernel normalization, mass conservation, density estimation |
| LBM | `test_lbm.cc` | Equilibrium distributions, mass/momentum conservation, bounce-back validation |
| LOD transitions | `test_lod.cc` | 5-level zone assignment, hysteresis, SWE/SPH/LBM state transfer, mass/momentum conservation round-trips, dynamic grid repositioning |
| Integration | `test_lod.cc` | Multi-level engine init/step/query, 5-level LOD activation from focus, submerged/dry query, LOD round-trip mass conservation |
| Coupling | `test_coupling.cc` | Buoyancy force, drag force, submerged body detection |
| Ocean | `test_ocean.cc` | Phillips spectrum, JONSWAP enhancement, dispersion relation (deep and shallow), Gerstner displacement, orbital velocity decay, circular orbits, energy, significant wave height |
| Stokes | `test_stokes.cc` | Poiseuille flow validation, divergence-free projection, concentration diffusion/source, Reynolds number, no-slip boundaries, velocity queries |

---

## Project Structure

```
mujoco-water/
  include/mjwater/     Library headers (header-only; ship these)
  tests/               Standalone test suites (no MuJoCo needed)
  demo/                MuJoCo demos with Dear ImGui (separate consumers)
  configure.py         Interactive project generator (5 presets, 3 backends)
  CMakeLists.txt       Build config: MJWATER_TESTS_ONLY, MJWATER_TESTS, MJWATER_DEMO
  build.sh             Linux/macOS helper: build, test, demo, clean
  launch.bat           Windows helper: kill, rebuild, launch demo
  docs/Screenshot.png  Demo screenshot
  LICENSE              MIT
```

---

## Requirements

- C++20 compiler: MSVC 19.30+, GCC 11+, or Clang 14+
- CMake 3.20+
- MuJoCo 3.0+ (required only for coupling and demos, not for tests or standalone use)

---

## References

- Tessendorf, J. (2001). "Simulating Ocean Water." SIGGRAPH Course Notes.
- Phillips, O. M. (1957). "On the generation of waves by turbulent wind." Journal of Fluid Mechanics 2(5).
- Hasselmann, K. et al. (1973). "Measurements of wind-wave growth and swell decay during the JONSWAP experiment." Deutsche Hydrographische Zeitschrift.
- Toro, E. F. (2001). "Shock-Capturing Methods for Free-Surface Shallow Flows." Wiley.
- LeVeque, R. J. (2002). "Finite Volume Methods for Hyperbolic Problems." Cambridge University Press.
- Monaghan, J. J. (2005). "Smoothed Particle Hydrodynamics." Reports on Progress in Physics 68.
- Monaghan, J. J. (1989). "On the Problem of Penetration in Particle Methods." Journal of Computational Physics 82.
- Morris, J. P. (2000). "Simulating Surface Tension with Smoothed Particle Hydrodynamics." International Journal for Numerical Methods in Fluids 33.
- Wendland, H. (1995). "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree." Advances in Computational Mathematics 4.
- Randles, P. W. and Libersky, L. D. (1996). "Smoothed Particle Hydrodynamics: Some recent improvements and applications." Computer Methods in Applied Mechanics and Engineering 139.
- Kruger, T. et al. (2017). "The Lattice Boltzmann Method: Principles and Practice." Springer.
- Ginzburg, I., Verhaeghe, F., and d'Humieres, D. (2008). "Two-Relaxation-Time Lattice Boltzmann Scheme." Communications in Computational Physics 3(2).
- Succi, S. (2001). "The Lattice Boltzmann Equation for Fluid Dynamics and Beyond." Oxford University Press.
- Happel, J. and Brenner, H. (1983). "Low Reynolds Number Hydrodynamics." Martinus Nijhoff.
- Purcell, E. M. (1977). "Life at Low Reynolds Number." American Journal of Physics 45.
- Berg, H. C. (1993). "Random Walks in Biology." Princeton University Press.
