// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
//
// Dam break demo with floating sphere.
//
// Water surface rendered as a MuJoCo height field (smooth continuous mesh)
// updated each frame from the shallow water solver. A deep-ocean bulk
// volume box sits underneath for the volumetric look.
//
// Controls: mouse drag to orbit, scroll to zoom, ESC to quit.
//
// Build:
//   cmake -B build-demo -DMJWATER_DEMO=ON
//   cmake --build build-demo --config Release
//   build-demo/Release/water_demo.exe

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include <mujoco/mujoco.h>

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include "mjwater/shallow_water.h"
#include "mjwater/sph.h"
#include "mjwater/lbm.h"
#include "mjwater/stokes.h"
#include "mjwater/coupling.h"
#include "mjwater/types.h"

using namespace mjwater;

// ---- Constants ----

static constexpr int kGridN = 80;               // SWE grid resolution
static constexpr float kGridDx = 1.0f / kGridN; // cell size (m)
static constexpr float kPoolSize = 1.0f;        // pool is 1m x 1m
static constexpr float kWallHeight = 0.3f;
static constexpr float kMaxWaterZ = 0.25f;      // hfield max elevation
static constexpr float kSphereRadius = 0.05f;
static constexpr float kSphereDiam = 2.0f * kSphereRadius;

// Scale slider state (0=Ocean .. 4=Stokes).
static int g_scale_level = 1;  // start at SWE
static bool g_reset_requested = false;

static const char* kScaleNames[] = {
  "Ocean (km)",
  "Shallow Water (m)",
  "SPH Particles (cm)",
  "Lattice Boltzmann (mm)",
  "Stokes Flow (um)"
};

static const char* kScaleGrid[] = {
  "analytic (no grid)",
  "2D height field, dx~10cm",
  "3D particles, h~2cm",
  "3D D3Q19 lattice, dx~1mm",
  "3D grid, dx~10um"
};

static const float kScaleZoom[] = {
  20.0f, 1.8f, 0.3f, 0.05f, 0.005f
};

// ---- Build MJCF with hfield asset ----

static std::string BuildMJCF() {
  // Height field: nrow x ncol, size = (x_half, y_half, z_max, z_base)
  // z_base = 0: the hfield bottom sits at the geom position.
  // z_max = kMaxWaterZ: data value 1.0 maps to this height.
  char buf[4096];
  snprintf(buf, sizeof(buf), R"(
<mujoco model="water_demo">
  <option timestep="0.001" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>

  <visual>
    <rgba fog="0.9 0.95 1.0 1"/>
    <quality shadowsize="2048"/>
    <map znear="0.001"/>
  </visual>

  <default>
    <geom contype="1" conaffinity="1" friction="0.8 0.02 0.01"
          rgba="0.6 0.6 0.6 1"/>
  </default>

  <asset>
    <hfield name="water_surface" nrow="%d" ncol="%d"
            size="%f %f %f 0.001"/>
  </asset>

  <worldbody>
    <light pos="0.5 0.5 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="0.5 -0.5 1.5" dir="0 0.5 -1" diffuse="0.3 0.3 0.3"/>

    <!-- Ground plane doubles as pool floor -->
    <geom type="plane" size="2 2 0.01" rgba="0.12 0.16 0.22 1"/>
    <geom name="wall_x0" type="box" pos="-0.01 0.5 0.15" size="0.01 0.52 0.15"
          rgba="0.4 0.45 0.5 0.7"/>
    <geom name="wall_x1" type="box" pos="1.01 0.5 0.15" size="0.01 0.52 0.15"
          rgba="0.4 0.45 0.5 0.7"/>
    <geom name="wall_y0" type="box" pos="0.5 -0.01 0.15" size="0.52 0.01 0.15"
          rgba="0.4 0.45 0.5 0.7"/>
    <geom name="wall_y1" type="box" pos="0.5 1.01 0.15" size="0.52 0.01 0.15"
          rgba="0.4 0.45 0.5 0.7"/>

    <!-- Water surface height field (raised 5mm above floor) -->
    <geom name="water" type="hfield" hfield="water_surface"
          pos="0.5 0.5 0.005" contype="0" conaffinity="0"
          rgba="0.05 0.32 0.48 1.0"/>

    <!-- Floating sphere -->
    <body name="sphere" pos="0.7 0.5 0.4">
      <freejoint name="sphere_joint"/>
      <geom type="sphere" size="0.05" rgba="0.9 0.2 0.15 1"
            mass="0.3" solref="-1000 -100"/>
    </body>
  </worldbody>
</mujoco>
  )", kGridN, kGridN,
      kPoolSize * 0.5f, kPoolSize * 0.5f, kMaxWaterZ);
  return std::string(buf);
}

// ---- Viewer state ----

struct Viewer {
  mjvCamera  cam  = {};
  mjvOption  opt  = {};
  mjvScene   scn  = {};
  mjrContext con  = {};
  GLFWwindow* window = nullptr;

  mjModel* model = nullptr;
  bool mouse_left = false, mouse_right = false, mouse_mid = false;
  double mouse_x = 0, mouse_y = 0;

  bool Init(mjModel* m) {
    model = m;
    if (!glfwInit()) return false;
    window = glfwCreateWindow(1280, 720, "mujoco-water: Dam Break", nullptr, nullptr);
    if (!window) { glfwTerminate(); return false; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(window, this);

    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int btn, int act, int) {
      auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
      bool p = (act == GLFW_PRESS);
      if (btn == GLFW_MOUSE_BUTTON_LEFT)   v->mouse_left  = p;
      if (btn == GLFW_MOUSE_BUTTON_RIGHT)  v->mouse_right = p;
      if (btn == GLFW_MOUSE_BUTTON_MIDDLE) v->mouse_mid   = p;
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
      auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
      double dx = x - v->mouse_x, dy = y - v->mouse_y;
      v->mouse_x = x; v->mouse_y = y;
      int ww, wh; glfwGetWindowSize(w, &ww, &wh);
      if (ww == 0 || wh == 0) return;
      if (v->mouse_left)
        mjv_moveCamera(v->model, mjMOUSE_ROTATE_V, dx/ww, dy/wh, &v->scn, &v->cam);
      else if (v->mouse_right)
        mjv_moveCamera(v->model, mjMOUSE_MOVE_V, dx/ww, dy/wh, &v->scn, &v->cam);
      else if (v->mouse_mid)
        mjv_moveCamera(v->model, mjMOUSE_ZOOM, 0, dy/wh, &v->scn, &v->cam);
    });

    glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy) {
      auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
      mjv_moveCamera(v->model, mjMOUSE_ZOOM, 0, -0.05*dy, &v->scn, &v->cam);
    });

    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int act, int) {
      if (key == GLFW_KEY_ESCAPE && act == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
      if (act == GLFW_PRESS || act == GLFW_REPEAT) {
        if (key == GLFW_KEY_LEFT || key == GLFW_KEY_DOWN)
          g_scale_level = std::max(0, g_scale_level - 1);
        if (key == GLFW_KEY_RIGHT || key == GLFW_KEY_UP)
          g_scale_level = std::min(4, g_scale_level + 1);
        if (key >= GLFW_KEY_1 && key <= GLFW_KEY_5)
          g_scale_level = key - GLFW_KEY_1;
        if (key == GLFW_KEY_R)
          g_reset_requested = true;
      }
    });

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    cam.type = mjCAMERA_FREE;
    cam.lookat[0] = 0.5; cam.lookat[1] = 0.5; cam.lookat[2] = 0.08;
    cam.distance = 1.8;
    cam.elevation = -30.0;
    cam.azimuth = 135.0;

    mjv_makeScene(m, &scn, 5000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);
    return true;
  }

  bool Running() const {
    return window && !glfwWindowShouldClose(window);
  }

  void Shutdown() {
    if (!window) return;
    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    glfwDestroyWindow(window);
    glfwTerminate();
    window = nullptr;
  }
};

// ---- Update hfield data from SWE solver ----

static void UpdateHField(mjModel* m, const mjrContext& con,
                          const ShallowWaterSolver& swe, int hfield_id) {
  const auto& h_data = swe.grid.channels[swe.ch_h].data;
  const auto& b_data = swe.grid.channels[swe.ch_bathy].data;

  int nrow = m->hfield_nrow[hfield_id];
  int ncol = m->hfield_ncol[hfield_id];
  float* hdata = m->hfield_data + m->hfield_adr[hfield_id];

  float inv_zmax = 1.0f / kMaxWaterZ;

  // MuJoCo hfield: row 0 = +Y, row nrow-1 = -Y (top to bottom in Y).
  // Column 0 = -X, column ncol-1 = +X.
  // We map SWE grid directly: gx -> col, gy -> row (inverted).
  for (int row = 0; row < nrow; ++row) {
    for (int col = 0; col < ncol; ++col) {
      // Map hfield (row, col) to SWE grid (gx, gy).
      uint32_t gx = static_cast<uint32_t>(col);
      uint32_t gy = static_cast<uint32_t>(nrow - 1 - row);

      // Clamp to grid bounds.
      if (gx >= swe.grid.nx) gx = swe.grid.nx - 1;
      if (gy >= swe.grid.ny) gy = swe.grid.ny - 1;

      size_t idx = swe.grid.Idx(gx, gy);
      float surface = b_data[idx] + h_data[idx];
      // Clamp min to 0.04 so hfield mesh never dips below the pool floor.
      // (0.04 * kMaxWaterZ = 0.01m, just above floor at z=0)
      float normalized = std::clamp(surface * inv_zmax, 0.04f, 1.0f);
      hdata[row * ncol + col] = normalized;
    }
  }

  // Upload updated hfield to GPU.
  mjr_uploadHField(m, &con, hfield_id);
}

// No bulk volume box needed; the hfield surface + dark pool floor is enough.

// ---- Main ----

int main() {
  // Build and load MuJoCo model with hfield.
  std::string mjcf = BuildMJCF();
  char error[1024] = {};
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", mjcf.c_str(),
                  static_cast<int>(mjcf.size()));
  mjModel* m = mj_loadXML("model.xml", &vfs, error, sizeof(error));
  mj_deleteVFS(&vfs);

  if (!m) {
    fprintf(stderr, "MuJoCo load error: %s\n", error);
    return 1;
  }
  mjData* d = mj_makeData(m);

  // Find hfield ID.
  int hfield_id = mj_name2id(m, mjOBJ_HFIELD, "water_surface");
  if (hfield_id < 0) {
    fprintf(stderr, "Height field not found\n");
    mj_deleteData(d); mj_deleteModel(m);
    return 1;
  }

  // Initialize shallow water solver.
  ShallowWaterSolver swe;
  swe.Init(kGridN, kGridN, kGridDx, 0.0f, 0.0f);

  // Dam break initial conditions (also used for reset).
  auto ResetDamBreak = [&]() {
    auto& h = swe.grid.channels[swe.ch_h].data;
    auto& hu = swe.grid.channels[swe.ch_hu].data;
    auto& hv = swe.grid.channels[swe.ch_hv].data;
    for (uint32_t y = 0; y < swe.grid.ny; ++y) {
      for (uint32_t x = 0; x < swe.grid.nx; ++x) {
        float wx, wy;
        swe.grid.GridToWorld(x, y, wx, wy);
        size_t idx = swe.grid.Idx(x, y);
        hu[idx] = 0.0f;
        hv[idx] = 0.0f;
        if (wx < 0.0f || wx > 1.0f || wy < 0.0f || wy > 1.0f) {
          h[idx] = 0.0f;
        } else if (wx < 0.4f) {
          h[idx] = 0.15f;
        } else {
          h[idx] = 0.02f;
        }
      }
    }
  };
  ResetDamBreak();

  // Body coupling for the sphere.
  BodyCoupling sphere_body;
  sphere_body.drag_coeff = 0.47f;
  sphere_body.added_mass_coeff = 0.5f;
  sphere_body.cross_section = kPi * kSphereRadius * kSphereRadius;
  sphere_body.volume = (4.0f/3.0f) * kPi * kSphereRadius * kSphereRadius * kSphereRadius;

  int sphere_id = mj_name2id(m, mjOBJ_BODY, "sphere");
  if (sphere_id < 0) {
    fprintf(stderr, "Sphere body not found\n");
    mj_deleteData(d); mj_deleteModel(m);
    return 1;
  }

  // Open viewer.
  Viewer viewer;
  if (!viewer.Init(m)) {
    fprintf(stderr, "Could not open viewer\n");
    mj_deleteData(d); mj_deleteModel(m);
    return 1;
  }

  printf("[water_demo] Dam break with floating sphere (hfield rendering)\n");
  printf("[water_demo] Mouse: drag=orbit, right=pan, scroll=zoom\n");
  printf("[water_demo] R=restart, arrows/1-5=scale slider, ESC=quit\n");

  float water_dt = 0.005f;
  float water_acc = 0.0f;
  double wall_start = glfwGetTime();

  while (viewer.Running()) {
    // Handle reset.
    if (g_reset_requested) {
      g_reset_requested = false;
      ResetDamBreak();
      sphere_body.prev_fluid_vel = {};
      mj_resetData(m, d);
      water_acc = 0.0f;
      wall_start = glfwGetTime();
    }
    double wall_now = glfwGetTime() - wall_start;
    double sim_behind = wall_now - d->time;

    int steps_this_frame = 0;
    while (d->time < wall_now && steps_this_frame < 20) {
      mj_step(m, d);
      steps_this_frame++;

      water_acc += static_cast<float>(m->opt.timestep);
      while (water_acc >= water_dt) {
        swe.Step(water_dt);
        water_acc -= water_dt;
      }

      // Fluid forces on sphere.
      Vec3 sphere_pos = {
        static_cast<float>(d->xpos[sphere_id * 3 + 0]),
        static_cast<float>(d->xpos[sphere_id * 3 + 1]),
        static_cast<float>(d->xpos[sphere_id * 3 + 2])
      };
      Vec3 sphere_vel = {
        static_cast<float>(d->cvel[sphere_id * 6 + 3]),
        static_cast<float>(d->cvel[sphere_id * 6 + 4]),
        static_cast<float>(d->cvel[sphere_id * 6 + 5])
      };

      auto fluid = FluidCoupling::QuerySWE(swe, sphere_pos);
      auto force = FluidCoupling::ComputeForces(fluid, sphere_body, sphere_vel,
                                                 static_cast<float>(m->opt.timestep),
                                                 kSphereDiam);
      sphere_body.prev_fluid_vel = fluid.velocity;

      // xfrc_applied layout: [fx, fy, fz, tx, ty, tz]
      d->xfrc_applied[sphere_id * 6 + 0] = force.force.x;
      d->xfrc_applied[sphere_id * 6 + 1] = force.force.y;
      d->xfrc_applied[sphere_id * 6 + 2] = force.force.z;
      d->xfrc_applied[sphere_id * 6 + 3] = 0;
      d->xfrc_applied[sphere_id * 6 + 4] = 0;
      d->xfrc_applied[sphere_id * 6 + 5] = 0;
    }

    if (sim_behind > 0.1) {
      wall_start = glfwGetTime() - d->time;
    }

    // Update hfield from SWE data.
    UpdateHField(m, viewer.con, swe, hfield_id);

    // Render.
    {
      int w, h;
      glfwGetFramebufferSize(viewer.window, &w, &h);
      mjrRect viewport = {0, 0, w, h};

      mjv_updateScene(m, d, &viewer.opt, nullptr, &viewer.cam,
                      mjCAT_ALL, &viewer.scn);

      Vec3 sp = {
        static_cast<float>(d->xpos[sphere_id * 3 + 0]),
        static_cast<float>(d->xpos[sphere_id * 3 + 1]),
        static_cast<float>(d->xpos[sphere_id * 3 + 2])
      };
      auto fl = FluidCoupling::QuerySWE(swe, sp);

      // Top-left: simulation info.
      char hud[256];
      snprintf(hud, sizeof(hud),
               "t=%.2fs  vol=%.4f m3  sphere_z=%.3f  depth=%.3f  %s",
               d->time, swe.TotalVolume(), sp.z,
               fl.depth, fl.submerged ? "SUBMERGED" : "dry");

      // Bottom-left: scale slider.
      char scale_hud[512];
      char scale_bar[128];
      // Build visual slider: [O]--[SWE]--[SPH]--[LBM]--[S]
      //                        ^cursor
      const char* pips[] = {"Ocean", "SWE", "SPH", "LBM", "Stokes"};
      int pos = 0;
      for (int i = 0; i < 5; ++i) {
        if (i == g_scale_level)
          pos += snprintf(scale_bar + pos, sizeof(scale_bar) - pos,
                          "[>%s<]", pips[i]);
        else
          pos += snprintf(scale_bar + pos, sizeof(scale_bar) - pos,
                          " %s ", pips[i]);
        if (i < 4)
          pos += snprintf(scale_bar + pos, sizeof(scale_bar) - pos, "--");
      }
      snprintf(scale_hud, sizeof(scale_hud),
               "LOD Scale (arrow keys / 1-5):\n"
               "%s\n"
               "Level %d: %s\n"
               "Grid: %s",
               scale_bar,
               g_scale_level, kScaleNames[g_scale_level],
               kScaleGrid[g_scale_level]);

      mjr_render(viewport, &viewer.scn, &viewer.con);
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport,
                  hud, nullptr, &viewer.con);
      mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport,
                  scale_hud, nullptr, &viewer.con);

      glfwSwapBuffers(viewer.window);
      glfwPollEvents();
    }
  }

  viewer.Shutdown();
  mj_deleteData(d);
  mj_deleteModel(m);
  printf("[water_demo] Done.\n");
  return 0;
}
