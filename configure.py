#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# mujoco-water project configurator
#
# Asks 5 questions, generates a ready-to-compile project folder with
# only the headers, config, and CMakeLists you need.
#
# Usage: python configure.py [output_dir]

import os
import sys
import textwrap

PRESETS = {
    "1": {
        "name": "Swimming robot",
        "desc": "Pool/tank with floating or swimming body. SWE + SPH.",
        "levels": ["swe", "sph"],
        "headers": [
            "types.h", "water_grid.h", "shallow_water.h", "sph.h",
            "lod_manager.h", "lod_transition.h", "coupling.h",
            "water_engine.h", "water_sdf.h", "mujoco_utils.h",
        ],
        "config": {
            "ocean_enabled": False,
            "swe_nx": 40, "swe_ny": 40, "swe_dx": 0.1,
            "sph_spacing": 0.02, "sph_smoothing": 0.03,
            "master_dt": 0.01,
            "sph_substeps": 10,
            "lod": {"sph_radius": 0.5, "lbm_radius": -1, "stokes_radius": -1},
            "initial_surface_z": 0.5,
            "domain": {"min": [0, 0, 0], "max": [4, 4, 2]},
        },
    },
    "2": {
        "name": "Pool / tank",
        "desc": "Static water body with waves. SWE only, minimal.",
        "levels": ["swe"],
        "headers": [
            "types.h", "water_grid.h", "shallow_water.h",
            "coupling.h", "water_engine.h", "water_sdf.h", "mujoco_utils.h",
            "lod_manager.h", "lod_transition.h",
        ],
        "config": {
            "ocean_enabled": False,
            "swe_nx": 30, "swe_ny": 30, "swe_dx": 0.1,
            "master_dt": 0.01,
            "lod": {"sph_radius": -1, "lbm_radius": -1, "stokes_radius": -1},
            "initial_surface_z": 0.5,
            "domain": {"min": [0, 0, 0], "max": [3, 3, 1]},
        },
    },
    "3": {
        "name": "Ocean vehicle",
        "desc": "Open-water vehicle with wave forces. Ocean + SWE + body coupling.",
        "levels": ["ocean", "swe"],
        "headers": [
            "types.h", "water_grid.h", "spectral_ocean.h", "shallow_water.h",
            "lod_manager.h", "lod_transition.h", "coupling.h",
            "water_engine.h", "mujoco_utils.h",
        ],
        "config": {
            "ocean_enabled": True,
            "ocean": {"wind_speed": 10.0, "jonswap_gamma": 3.3},
            "ocean_mean_depth": 50.0,
            "swe_nx": 60, "swe_ny": 60, "swe_dx": 1.0,
            "master_dt": 0.01,
            "lod": {"sph_radius": -1, "lbm_radius": -1, "stokes_radius": -1},
            "initial_surface_z": 0.0,
            "domain": {"min": [-30, -30, -5], "max": [30, 30, 10]},
        },
    },
    "4": {
        "name": "Microfluidics",
        "desc": "Lab-on-chip scale. LBM + Stokes, no ocean or SWE waves.",
        "levels": ["lbm", "stokes"],
        "headers": [
            "types.h", "water_grid.h", "lbm.h", "stokes.h",
            "lod_manager.h", "lod_transition.h", "coupling.h",
            "water_engine.h",
        ],
        "config": {
            "ocean_enabled": False,
            "swe_nx": 10, "swe_ny": 10, "swe_dx": 0.001,
            "lbm_nx": 40, "lbm_ny": 40, "lbm_nz": 20,
            "lbm_dx": 0.0005,
            "stokes_nx": 30, "stokes_ny": 30, "stokes_nz": 30,
            "stokes": {"dx": 10e-6, "viscosity": 1e-3, "diffusivity": 1e-9},
            "stokes_substeps": 10,
            "master_dt": 0.001,
            "lbm_substeps": 10,
            "lod": {
                "sph_radius": -1, "lbm_radius": 0.05,
                "stokes_radius": 0.005, "swe_radius": 1.0,
            },
            "initial_surface_z": 0.001,
            "domain": {"min": [0, 0, 0], "max": [0.01, 0.01, 0.005]},
        },
    },
    "5": {
        "name": "Full multiscale",
        "desc": "All 5 LOD levels. Ocean down to cellular. Maximum capability.",
        "levels": ["ocean", "swe", "sph", "lbm", "stokes"],
        "headers": [
            "types.h", "water_grid.h", "spectral_ocean.h", "shallow_water.h",
            "sph.h", "lbm.h", "stokes.h",
            "lod_manager.h", "lod_transition.h", "coupling.h",
            "water_engine.h", "water_sdf.h", "mujoco_utils.h",
        ],
        "config": {
            "ocean_enabled": True,
            "swe_nx": 40, "swe_ny": 40, "swe_dx": 0.1,
            "sph_spacing": 0.02, "sph_smoothing": 0.03,
            "lbm_nx": 30, "lbm_ny": 30, "lbm_nz": 30,
            "lbm_dx": 0.001,
            "stokes_nx": 20, "stokes_ny": 20, "stokes_nz": 20,
            "stokes": {"dx": 10e-6, "diffusivity": 1e-9},
            "master_dt": 0.01,
            "sph_substeps": 10,
            "lbm_substeps": 100,
            "stokes_substeps": 10,
            "lod": {
                "sph_radius": 0.5, "lbm_radius": 0.05,
                "stokes_radius": 0.005, "swe_radius": 50.0,
            },
            "initial_surface_z": 0.5,
            "domain": {"min": [0, 0, 0], "max": [10, 10, 2]},
        },
    },
}

FEATURE_QUESTIONS = [
    ("surface_tension", "Surface tension (droplets, meniscus)?", True,
     "sph" , "params.surface_tension = 0.0728f;  // water at 20C"),
    ("chemotaxis", "Concentration field / chemotaxis?", False,
     "stokes", "// Use stokes.SetSource(x,y,z, rate) for concentration sources"),
    ("bodies", "Couple fluid forces to MuJoCo bodies?", True,
     None, None),
]


def ask(prompt, options=None, default=None):
    """Ask user a question, return answer."""
    suffix = ""
    if default is not None:
        suffix = f" [{default}]"
    if options:
        for k, v in options.items():
            print(f"  {k}) {v}")
    while True:
        ans = input(f"{prompt}{suffix}: ").strip()
        if not ans and default is not None:
            return default
        if options and ans in options:
            return ans
        if not options and ans:
            return ans
        if not options and default is not None:
            return default


def ask_yn(prompt, default=True):
    d = "Y/n" if default else "y/N"
    ans = input(f"{prompt} [{d}]: ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def fmt_float(v):
    """Format a number as a C++ float literal."""
    s = f"{v}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s + "f"


def gen_config_code(preset, features):
    """Generate C++ WaterEngineConfig initialization code."""
    cfg = preset["config"]
    lines = ["mjwater::WaterEngineConfig cfg;"]

    if cfg.get("ocean_enabled"):
        lines.append("cfg.ocean_enabled = true;")
        ocean = cfg.get("ocean", {})
        if "wind_speed" in ocean:
            lines.append(f"cfg.ocean.wind_speed = {fmt_float(ocean['wind_speed'])};")
        if "jonswap_gamma" in ocean:
            lines.append(f"cfg.ocean.jonswap_gamma = {fmt_float(ocean['jonswap_gamma'])};")
        if "ocean_mean_depth" in cfg:
            lines.append(f"cfg.ocean_mean_depth = {fmt_float(cfg['ocean_mean_depth'])};")

    lines.append(f"cfg.swe_nx = {cfg['swe_nx']};")
    lines.append(f"cfg.swe_ny = {cfg['swe_ny']};")
    lines.append(f"cfg.swe_dx = {fmt_float(cfg['swe_dx'])};")

    if "sph_spacing" in cfg:
        lines.append(f"cfg.sph_spacing = {fmt_float(cfg['sph_spacing'])};")
        lines.append(f"cfg.sph_smoothing = {fmt_float(cfg['sph_smoothing'])};")

    if "lbm_nx" in cfg:
        lines.append(f"cfg.lbm_nx = {cfg['lbm_nx']};")
        lines.append(f"cfg.lbm_ny = {cfg['lbm_ny']};")
        lines.append(f"cfg.lbm_nz = {cfg['lbm_nz']};")
        lines.append(f"cfg.lbm_dx = {fmt_float(cfg['lbm_dx'])};")

    if "stokes_nx" in cfg:
        lines.append(f"cfg.stokes_nx = {cfg['stokes_nx']};")
        lines.append(f"cfg.stokes_ny = {cfg['stokes_ny']};")
        lines.append(f"cfg.stokes_nz = {cfg['stokes_nz']};")
        stokes = cfg.get("stokes", {})
        if "dx" in stokes:
            lines.append(f"cfg.stokes.dx = {fmt_float(stokes['dx'])};")
        if "viscosity" in stokes:
            lines.append(f"cfg.stokes.viscosity = {fmt_float(stokes['viscosity'])};")
        if "diffusivity" in stokes:
            lines.append(f"cfg.stokes.diffusivity = {fmt_float(stokes['diffusivity'])};")

    lines.append(f"cfg.master_dt = {fmt_float(cfg['master_dt'])};")

    if "sph_substeps" in cfg:
        lines.append(f"cfg.sph_substeps = {cfg['sph_substeps']};")
    if "lbm_substeps" in cfg:
        lines.append(f"cfg.lbm_substeps = {cfg['lbm_substeps']};")
    if "stokes_substeps" in cfg:
        lines.append(f"cfg.stokes_substeps = {cfg['stokes_substeps']};")

    lod = cfg.get("lod", {})
    for key in ["sph_radius", "lbm_radius", "stokes_radius", "swe_radius"]:
        if key in lod:
            lines.append(f"cfg.lod.{key} = {fmt_float(lod[key])};")

    lines.append(f"cfg.initial_surface_z = {fmt_float(cfg['initial_surface_z'])};")

    domain = cfg.get("domain", {})
    mn = domain.get("min", [0, 0, 0])
    mx = domain.get("max", [10, 10, 2])
    lines.append(f"cfg.domain = {{{{{{{fmt_float(mn[0])}, {fmt_float(mn[1])}, {fmt_float(mn[2])}}}, {{{fmt_float(mx[0])}, {fmt_float(mx[1])}, {fmt_float(mx[2])}}}}}}};")

    return lines


def gen_main_cpp(preset, features, project_name):
    """Generate main.cpp with MuJoCo + water integration."""
    has_bodies = features.get("bodies", True)
    has_chemotaxis = features.get("chemotaxis", False)
    levels = preset["levels"]

    includes = ['#include <cstdio>']
    includes.append('#include <mujoco/mujoco.h>')
    includes.append('#include "mjwater/water_engine.h"')
    if has_bodies:
        includes.append('#include "mjwater/mujoco_utils.h"')

    config_lines_raw = gen_config_code(preset, features)
    config_lines = "\n  ".join(config_lines_raw)

    body_setup = ""
    body_loop = ""
    if has_bodies:
        body_setup = textwrap.dedent("""\

            // --- Register bodies for fluid coupling ---
            // Replace "my_body" with your MuJoCo body name.
            int body_id = mj_name2id(m, mjOBJ_BODY, "my_body");
            if (body_id >= 0) {
              mjwater::BodyCoupling bc;
              bc.volume = 0.01f;           // displaced volume (m^3)
              bc.cross_section = 0.05f;    // reference area (m^2)
              bc.drag_coeff = 0.8f;        // Cd
              bc.added_mass_coeff = 0.5f;  // Ca (0.5 for sphere)
              size_t bi = water.AddBody(bc);

              // Store bi for use in the simulation loop.
              (void)bi;
            }
        """)

        body_loop = textwrap.dedent("""\

            // --- Apply fluid forces to bodies ---
            if (body_id >= 0) {
              mjwater::Vec3 pos = {(float)d->xpos[3*body_id],
                                   (float)d->xpos[3*body_id+1],
                                   (float)d->xpos[3*body_id+2]};
              mjwater::Vec3 vel = {(float)d->cvel[6*body_id+3],
                                   (float)d->cvel[6*body_id+4],
                                   (float)d->cvel[6*body_id+5]};
              auto force = water.ComputeBodyForce(0, pos, vel, m->opt.timestep);
              mjwater::ApplyFluidForces(d, body_id, force);
            }
        """)

    chemotaxis_comment = ""
    if has_chemotaxis and "stokes" in levels:
        chemotaxis_comment = textwrap.dedent("""\

            // --- Chemotaxis ---
            // Set concentration sources on the Stokes grid:
            //   water.stokes.SetSource(x, y, z, rate);
            // Query concentration at a point:
            //   float c = water.stokes.Concentration({wx, wy, wz});
        """)

    surface_tension_line = ""
    if features.get("surface_tension", False) and "sph" in levels:
        surface_tension_line = "\n  // Surface tension is enabled by default (0.0728 N/m for water at 20C)."
        surface_tension_line += "\n  // Set cfg.sph_spacing smaller for visible capillary effects."

    src = f"""\
// {project_name}, generated by mujoco-water configure.py
// Preset: {preset['name']}
// LOD levels: {', '.join(levels)}
//
// Build:
//   cmake -B build -DMUJOCO_DIR=<path>
//   cmake --build build --config Release

{chr(10).join(includes)}

int main() {{
  // --- Water engine setup ---
  {config_lines}{surface_tension_line}

  mjwater::WaterEngine water;
  water.Init(cfg);
{chemotaxis_comment}
  // --- MuJoCo setup ---
  // Load your model here. Example:
  char error[1000] = "";
  mjModel* m = mj_loadXML("model.xml", nullptr, error, sizeof(error));
  if (!m) {{
    printf("MuJoCo load error: %s\\n", error);
    return 1;
  }}
  mjData* d = mj_makeData(m);

  // Disable MuJoCo built-in fluid (we provide our own).
  m->opt.density = 0;
  m->opt.viscosity = 0;
{body_setup}
  // --- Simulation loop ---
  for (int step = 0; step < 10000; ++step) {{
    // Set focus to camera or agent position.
    water.SetFocus({{(float)d->qpos[0], (float)d->qpos[1], (float)d->qpos[2]}});

    // Step water simulation.
    water.Step();
{body_loop}
    // Step MuJoCo.
    mj_step(m, d);

    // Print status every 1000 steps.
    if (step % 1000 == 0) {{
      auto stats = water.GetLayerStats();
      printf("Step %d: ", step);
      for (auto& s : stats) {{
        if (s.active) printf("%s(%u) ", s.name, s.cells);
      }}
      printf("\\n");
    }}
  }}

  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}}
"""
    return src


def gen_cmakelists(project_name, mjwater_path):
    return f"""\
cmake_minimum_required(VERSION 3.20)
project({project_name} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# mujoco-water (header-only)
add_subdirectory({mjwater_path} ${{CMAKE_BINARY_DIR}}/mujoco-water)

add_executable(${{PROJECT_NAME}} main.cpp)
target_link_libraries(${{PROJECT_NAME}} PRIVATE mujoco-water)
target_compile_features(${{PROJECT_NAME}} PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(${{PROJECT_NAME}} PRIVATE /W4)
endif()
"""


def main():
    print("=" * 60)
    print("  mujoco-water configurator")
    print("  Generates a ready-to-compile project for your use case.")
    print("=" * 60)
    print()

    # 1. Use case
    print("What are you simulating?")
    options = {k: f"{v['name']}: {v['desc']}" for k, v in PRESETS.items()}
    choice = ask("Pick a preset", options, "1")
    preset = PRESETS[choice]
    print(f"\n  -> {preset['name']}: {', '.join(preset['levels'])}")
    print()

    # 2. Backend
    # Only MuJoCo is implemented. Other backends (Chrono, standalone)
    # would need their own gen_main_cpp paths before being offered here.
    backend = "mujoco"
    print(f"  Backend: MuJoCo")
    print()

    # 3. Project name
    project_name = ask("Project name", default="my_water_sim")
    print()

    # 4. Features
    features = {}
    for key, question, default, required_level, _ in FEATURE_QUESTIONS:
        if required_level and required_level not in preset["levels"]:
            features[key] = False
            continue
        if key == "bodies" and backend == "standalone":
            features[key] = False
            continue
        features[key] = ask_yn(question, default)
    print()

    # 5. Output directory
    default_out = os.path.join(os.getcwd(), project_name)
    out_dir = sys.argv[1] if len(sys.argv) > 1 else ask("Output directory", default=default_out)
    print()

    # 6. Path to mujoco-water
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_mjwater = os.path.relpath(script_dir, out_dir).replace("\\", "/")
    mjwater_path = ask("Path to mujoco-water (from output dir)", default=default_mjwater)

    # --- Generate ---
    os.makedirs(out_dir, exist_ok=True)

    # main.cpp
    main_cpp = gen_main_cpp(preset, features, project_name)
    with open(os.path.join(out_dir, "main.cpp"), "w") as f:
        f.write(main_cpp)

    # CMakeLists.txt
    cmake = gen_cmakelists(project_name, mjwater_path)
    with open(os.path.join(out_dir, "CMakeLists.txt"), "w") as f:
        f.write(cmake)

    # Summary
    print()
    print("=" * 60)
    print(f"  Generated: {out_dir}/")
    print(f"    main.cpp        {preset['name']} with {', '.join(preset['levels'])}")
    print(f"    CMakeLists.txt  build config")
    print()
    print(f"  Headers used ({len(preset['headers'])}/{14} available):")
    for h in preset["headers"]:
        print(f"    mjwater/{h}")
    print()
    print("  Build:")
    print(f"    cd {out_dir}")
    if backend == "mujoco":
        print(f"    cmake -B build -DMUJOCO_DIR=<path-to-mujoco>")
    elif backend == "chrono":
        print(f"    cmake -B build -DChrono_DIR=<path-to-chrono>")
    else:
        print(f"    cmake -B build")
    print(f"    cmake --build build --config Release")
    print("=" * 60)


if __name__ == "__main__":
    main()
