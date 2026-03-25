// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
//
// Multi-scale zoom demo: Ocean to Amoeba.
//
// Demonstrates additive-layer LOD architecture. All 5 solver levels
// run simultaneously in nested zones. Scroll to zoom through scales.
// ImGui side panel shows per-layer stats and provides LOD toggles.
//
// Controls: scroll=zoom, drag=orbit, R=restart, ESC=quit
//
// Build:
//   cmake -B build-demo -DMJWATER_DEMO=ON -DMUJOCO_DIR=<path>
//   cmake --build build-demo --config Release
//   build-demo/Release/multiscale_demo.exe

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include <mujoco/mujoco.h>

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "mjwater/water_engine.h"
#include "mjwater/coupling.h"

using namespace mjwater;

// ---- Constants ----

static constexpr int kSweN = 50;
static constexpr float kSweDx = 20.0f;            // 20m cells -> 1km x 1km domain
static constexpr float kOceanFloorZ = -100.0f;     // visual ocean floor
static constexpr float kPanelWidth = 300.0f;      // side panel width (mujoco-water convention)
// Water surface height. The single authoritative value.
// Everything else (hfield range, domain bounds, ball spawn) derives from this.
static float g_water_z = 100.0f;
// Domain extent in meters. Controls the physical size of the simulation.
static float g_extent = 1000.0f;

// Hfield rendering range, derived from g_water_z.
static float g_z_top() { return g_water_z * 1.2f; }
static float g_z_bottom() { return g_water_z * 0.3f; }

// ---- Helpers ----

struct CoupledBody {
  int body_id;
  float radius;
  BodyCoupling coupling;
};

// Find all free-joint sphere bodies and create CoupledBody entries.
static void InitCoupledBodies(mjModel* m, std::vector<CoupledBody>& bodies) {
  bodies.clear();
  for (int i = 1; i < m->nbody; ++i) {
    int jnt = m->body_jntadr[i];
    if (jnt < 0 || m->jnt_type[jnt] != 0) continue;  // 0 = mjJNT_FREE
    int g = m->body_geomadr[i];
    if (m->body_geomnum[i] <= 0) continue;
    float r = static_cast<float>(m->geom_size[3*g]);
    CoupledBody cb;
    cb.body_id = i;
    cb.radius = r;
    cb.coupling.volume = (4.0f / 3.0f) * kPi * r * r * r;
    cb.coupling.cross_section = kPi * r * r;
    cb.coupling.shape = BodyShape::kSphere;
    cb.coupling.two_way_enabled = true;
    cb.coupling.body_radius = r;
    bodies.push_back(cb);
  }
}

// ---- Global state ----

static bool g_lod_active[5] = {true, true, true, true, true};
static bool g_reset_requested = false;
static bool g_reload_model = false;
static bool g_simulate = true;
static bool g_auto_lod = true;  // auto-activate LOD levels based on camera distance
static float g_fps = 0;

// ---- MJCF ----

static std::string BuildMJCF(float ball_z) {
  float half = g_extent * 0.5f;

  char buf[4096];
  snprintf(buf, sizeof(buf), R"(
<mujoco model="multiscale_demo">
  <option timestep="0.01" gravity="0 0 %.5f"/>

  <visual>
    <rgba fog="0.6 0.62 0.65 1"/>
    <quality shadowsize="2048"/>
    <map znear="0.0001" zfar="5000" fogstart="500" fogend="3000"/>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" name="sky" builtin="gradient"
             rgb1="1 1 1" rgb2="0.7 0.8 0.9"
             width="256" height="256"/>
    <hfield name="water_surface" nrow="%d" ncol="%d"
            size="%f %f %f %f"/>
    <material name="water_mat" rgba="0.08 0.35 0.55 0.82"
              specular="0.5" shininess="0.6"/>
  </asset>

  <worldbody>
    <light pos="0 0 500" dir="0 0 -1" diffuse="0.9 0.9 0.85" castshadow="false"/>
    <light pos="300 -300 400" dir="-0.5 0.5 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <light pos="-200 200 300" dir="0.3 -0.3 -1" diffuse="0.3 0.35 0.4" castshadow="false"/>

    <!-- Ocean floor -->
    <geom type="plane" size="2000 2000 0.1" pos="0 0 %f"
          rgba="0.12 0.18 0.25 1" contype="0" conaffinity="0"/>

    <!-- Water surface height field -->
    <geom name="water" type="hfield" hfield="water_surface"
          pos="0 0 0" material="water_mat"
          contype="0" conaffinity="0"/>

    <!-- Floating platform -->
    <body name="ball" pos="0 0 %f">
      <freejoint/>
      <geom type="sphere" size="10" density="500"
            rgba="0.9 0.3 0.1 1"/>
    </body>

  </worldbody>
</mujoco>
  )", -kGravity,
      kSweN, kSweN,
      half, half, g_z_top(), g_z_bottom(),
      -g_water_z,
      ball_z);
  return std::string(buf);
}

// ---- Viewer ----

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
    window = glfwCreateWindow(1280, 720,
      "mujoco-water: Multi-scale Water Simulation", nullptr, nullptr);
    if (!window) { glfwTerminate(); return false; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // uncapped FPS
    glfwSetWindowUserPointer(window, this);

    // Mouse button: only orbit/pan when imgui doesn't want input.
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int btn, int act, int) {
      if (ImGui::GetIO().WantCaptureMouse) return;
      auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
      bool p = (act == GLFW_PRESS);
      if (btn == GLFW_MOUSE_BUTTON_LEFT)   v->mouse_left  = p;
      if (btn == GLFW_MOUSE_BUTTON_RIGHT)  v->mouse_right = p;
      if (btn == GLFW_MOUSE_BUTTON_MIDDLE) v->mouse_mid   = p;
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
      if (ImGui::GetIO().WantCaptureMouse) return;
      auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
      double dx = x - v->mouse_x, dy = y - v->mouse_y;
      v->mouse_x = x; v->mouse_y = y;
      int ww, wh; glfwGetWindowSize(w, &ww, &wh);
      if (ww == 0 || wh == 0) return;
      if (v->mouse_left) {
        // Update azimuth/elevation directly to avoid mjv_moveCamera
        // resetting cam.distance from scene bounds.
        v->cam.azimuth -= dx / ww * 180.0;
        v->cam.elevation -= dy / wh * 180.0;
        v->cam.elevation = std::clamp(v->cam.elevation, -89.0, 89.0);
      } else if (v->mouse_right) {
        bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        if (shift) {
          // Shift+right drag: boom arm height.
          v->cam.lookat[2] += dy / wh * v->cam.distance * 0.5;
        } else {
          // Right drag: pan lookat on XY plane.
          // Save distance before mjv_moveCamera (which can modify it),
          // then restore both distance and Z.
          double dist = v->cam.distance;
          double z = v->cam.lookat[2];
          mjv_moveCamera(v->model, mjMOUSE_MOVE_H, -dx/ww, -dy/wh, &v->scn, &v->cam);
          v->cam.lookat[2] = z;
          v->cam.distance = dist;
        }
      }
    });

    // Scroll = smooth zoom.
    glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy) {
      if (ImGui::GetIO().WantCaptureMouse) return;
      auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
      double factor = 1.0 - dy * 0.1;
      if (factor < 0.05) factor = 0.05;
      if (factor > 3.0) factor = 3.0;
      v->cam.distance *= factor;
      if (v->cam.distance < 0.001) v->cam.distance = 0.001;
      if (v->cam.distance > 5000.0) v->cam.distance = 5000.0;
    });

    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int act, int) {
      if (ImGui::GetIO().WantCaptureKeyboard) return;
      if (key == GLFW_KEY_ESCAPE && act == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
      if (key == GLFW_KEY_R && act == GLFW_PRESS)
        g_reset_requested = true;
    });

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    cam.type = mjCAMERA_FREE;
    cam.lookat[0] = 0; cam.lookat[1] = 0; cam.lookat[2] = 10.0;
    cam.distance = 800.0;
    cam.elevation = -30.0;
    cam.azimuth = 135.0;

    mjv_makeScene(m, &scn, 3000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Initialize Dear ImGui.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 2.0f;
    style.GrabRounding = 2.0f;
    style.ItemSpacing = ImVec2(6, 4);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    return true;
  }

  bool Running() const { return window && !glfwWindowShouldClose(window); }

  void Shutdown() {
    if (!window) return;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    glfwDestroyWindow(window);
    glfwTerminate();
    window = nullptr;
  }
};

// ---- Update hfield (one unified surface from all solvers) ----
//
static void UpdateHField(mjModel* m, const mjrContext& con,
                          const WaterEngine& engine, int hfield_id) {
  int nrow = m->hfield_nrow[hfield_id];
  int ncol = m->hfield_ncol[hfield_id];
  float* hdata = m->hfield_data + m->hfield_adr[hfield_id];
  float half_x = kSweDx * kSweN * 0.5f;
  float half_y = kSweDx * kSweN * 0.5f;
  float base_z = g_water_z;

  for (int r = 0; r < nrow; ++r) {
    float wy = -half_y + r * 2.0f * half_y / (nrow - 1);
    for (int c = 0; c < ncol; ++c) {
      float wx = -half_x + c * 2.0f * half_x / (ncol - 1);
      float surface = base_z;
      if (engine.ocean_active) surface += engine.ocean.SurfaceHeight(wx, wy);
      hdata[r * ncol + c] = std::clamp(surface / g_z_top(), 0.02f, 1.0f);
    }
  }

  mjr_uploadHField(m, &con, hfield_id);
}

static bool g_show_grid = true;
static int g_fine_chunks = 3;      // SWE fine wireframe
static int g_sph_chunks = 5;       // SPH sub-cell wireframe
static int g_lbm_chunks = 3;       // LBM wireframe
static int g_stokes_chunks = 3;    // Stokes wireframe

static void RenderDynamicGeoms(mjvScene* scn, const WaterEngine& engine) {
  if (engine.stokes_active && scn->ngeom < scn->maxgeom) {
    const auto& stokes = engine.stokes;
    float cx = stokes.grid.origin_x + stokes.grid.nx * stokes.params.dx * 0.5f;
    float cy = stokes.grid.origin_y + stokes.grid.ny * stokes.params.dx * 0.5f;
    float cz = stokes.grid.origin_z + stokes.grid.nz * stokes.params.dx * 0.5f;
    mjvGeom* g = &scn->geoms[scn->ngeom++];
    mjtNum size[3] = {50e-6, 50e-6, 50e-6};
    mjtNum pos[3] = {cx, cy, cz};
    mjtNum mat[9] = {1,0,0, 0,1,0, 0,0,1};
    float rgba[4] = {0.9f, 0.15f, 0.1f, 0.9f};
    mjv_initGeom(g, mjGEOM_SPHERE, size, pos, mat, rgba);
  }
}

static void RenderWireframe(mjvScene* scn, const WaterEngine& engine,
                             const mjvCamera& cam) {
  if (!g_show_grid) return;

  mjv_resetLines(scn);

  uint32_t nx = engine.swe.grid.nx, ny = engine.swe.grid.ny;
  float half_x = kSweDx * kSweN * 0.5f;
  float half_y = kSweDx * kSweN * 0.5f;
  float inv_nx = 2.0f * half_x / (nx - 1);
  float inv_ny = 2.0f * half_y / (ny - 1);
  float base_z = g_water_z;

  std::vector<float> wx(nx), wy(ny);
  for (uint32_t c = 0; c < nx; ++c) wx[c] = -half_x + c * inv_nx;
  for (uint32_t r = 0; r < ny; ++r) wy[r] = -half_y + r * inv_ny;

  std::vector<float> wz(nx * ny);
  for (uint32_t r = 0; r < ny; ++r)
    for (uint32_t c = 0; c < nx; ++c) {
      float z = base_z;
      if (engine.ocean_active) z += engine.ocean.SurfaceHeight(wx[c], wy[r]);
      wz[r * nx + c] = z;
    }

  float yellow[4] = {1.0f, 1.0f, 0.0f, 0.8f};

  auto add_line = [&](float x0, float y0, float z0,
                      float x1, float y1, float z1, const float* rgba) {
    float from[3] = {x0, y0, z0}, to[3] = {x1, y1, z1};
    mjv_addLine(scn, from, to, rgba);
  };

  constexpr uint32_t kCoarse = 5;

  float fx = engine.lod_manager.focus.x;
  float fy = engine.lod_manager.focus.y;
  int fc = std::clamp(static_cast<int>((fx + half_x) / (2.0f * half_x) * (nx - 1)),
                      0, static_cast<int>(nx - 1));
  int fr = std::clamp(static_cast<int>((fy + half_y) / (2.0f * half_y) * (ny - 1)),
                      0, static_cast<int>(ny - 1));

  // Center the fine area on the focus.
  // Even N: focus at a coarse intersection (boundary between chunks).
  // Odd N: focus at the center of a coarse chunk.
  int ck = static_cast<int>(kCoarse);
  int N = g_fine_chunks;
  int max_chunks = static_cast<int>(nx / kCoarse);

  int c_lo, r_lo;
  if (N % 2 == 0) {
    // Even: center on nearest coarse boundary. N/2 chunks each side.
    int boundary_c = static_cast<int>(std::round(static_cast<float>(fc) / ck));
    int boundary_r = static_cast<int>(std::round(static_cast<float>(fr) / ck));
    c_lo = std::clamp(boundary_c - N / 2, 0, std::max(max_chunks - N, 0));
    r_lo = std::clamp(boundary_r - N / 2, 0, std::max(max_chunks - N, 0));
  } else {
    // Odd: center on the chunk containing focus. (N-1)/2 chunks each side.
    int chunk_c = fc / ck;
    int chunk_r = fr / ck;
    c_lo = std::clamp(chunk_c - (N - 1) / 2, 0, std::max(max_chunks - N, 0));
    r_lo = std::clamp(chunk_r - (N - 1) / 2, 0, std::max(max_chunks - N, 0));
  }
  uint32_t cc0 = static_cast<uint32_t>(c_lo) * kCoarse;
  uint32_t cr0 = static_cast<uint32_t>(r_lo) * kCoarse;
  uint32_t cc1 = std::min(static_cast<uint32_t>(c_lo + N) * kCoarse, nx - 1);
  uint32_t cr1 = std::min(static_cast<uint32_t>(r_lo + N) * kCoarse, ny - 1);

  bool show_fine = cam.distance <= 500.0;
  auto in_fine = [&](uint32_t r, uint32_t c) {
    return show_fine && r >= cr0 && r <= cr1 && c >= cc0 && c <= cc1;
  };

  // Pass 1: Coarse grid.
  auto coarse_row = [&](uint32_t r) {
    for (uint32_t c = 0; c + 1 < nx; c += kCoarse) {
      uint32_t c2 = std::min(c + kCoarse, nx - 1);
      if (in_fine(r, c) && in_fine(r, c2)) continue;
      add_line(wx[c], wy[r], wz[r*nx+c], wx[c2], wy[r], wz[r*nx+c2], yellow);
    }
  };
  for (uint32_t r = 0; r < ny; r += kCoarse) coarse_row(r);
  if ((ny - 1) % kCoarse != 0) coarse_row(ny - 1);

  auto coarse_col = [&](uint32_t c) {
    for (uint32_t r = 0; r + 1 < ny; r += kCoarse) {
      uint32_t r2 = std::min(r + kCoarse, ny - 1);
      if (in_fine(r, c) && in_fine(r2, c)) continue;
      add_line(wx[c], wy[r], wz[r*nx+c], wx[c], wy[r2], wz[r2*nx+c], yellow);
    }
  };
  for (uint32_t c = 0; c < nx; c += kCoarse) coarse_col(c);
  if ((nx - 1) % kCoarse != 0) coarse_col(nx - 1);

  // Pass 2: Fine yellow grid.
  if (show_fine) {
    for (uint32_t r = cr0; r <= cr1; ++r)
      for (uint32_t c = cc0; c < cc1; ++c)
        add_line(wx[c], wy[r], wz[r*nx+c], wx[c+1], wy[r], wz[r*nx+c+1], yellow);
    for (uint32_t c = cc0; c <= cc1; ++c)
      for (uint32_t r = cr0; r < cr1; ++r)
        add_line(wx[c], wy[r], wz[r*nx+c], wx[c], wy[r+1], wz[(r+1)*nx+c], yellow);
  }

  // --- Unified sub-cell wireframe system ---
  // All layers work in fractional SWE cell indices, using the same
  // pre-computed wx/wy/wz arrays as the yellow coarse/fine grid.

  // Interpolate world position and z at fractional SWE cell index.
  auto interp_x = [&](float fi) -> float {
    int i = std::clamp(static_cast<int>(fi), 0, static_cast<int>(nx - 2));
    return wx[i] + (fi - i) * (wx[i + 1] - wx[i]);
  };
  auto interp_y = [&](float fj) -> float {
    int j = std::clamp(static_cast<int>(fj), 0, static_cast<int>(ny - 2));
    return wy[j] + (fj - j) * (wy[j + 1] - wy[j]);
  };
  auto interp_z = [&](float fi, float fj) -> float {
    int i = std::clamp(static_cast<int>(fi), 0, static_cast<int>(nx - 2));
    int j = std::clamp(static_cast<int>(fj), 0, static_cast<int>(ny - 2));
    float fx = fi - i, fy = fj - j;
    float z00 = wz[j * nx + i], z10 = wz[j * nx + i + 1];
    float z01 = wz[(j + 1) * nx + i], z11 = wz[(j + 1) * nx + i + 1];
    return (1 - fx) * (1 - fy) * z00 + fx * (1 - fy) * z10
         + (1 - fx) * fy * z01 + fx * fy * z11;
  };

  // Draw a line between two fractional SWE cell indices.
  // Steps at integer SWE cell boundaries so z follows the wave surface.
  // Draw a line between two fractional SWE cell indices.
  // Breaks at integer SWE cell boundaries so z follows the wave surface.
  auto add_line_fi = [&](float fi0, float fj0, float fi1, float fj1,
                          const float* rgba) {
    if (std::abs(fj0 - fj1) < 0.001f) {
      // Horizontal line: constant j, varying i.
      float lo = std::min(fi0, fi1), hi = std::max(fi0, fi1);
      float j = fj0;
      float cur = lo;
      while (cur < hi - 0.001f) {
        float next = std::min(std::floor(cur) + 1.0f, hi);
        if (next <= cur + 0.001f) next = std::min(cur + 1.0f, hi);
        add_line(interp_x(cur), interp_y(j), interp_z(cur, j),
                 interp_x(next), interp_y(j), interp_z(next, j), rgba);
        cur = next;
      }
    } else {
      // Vertical line: constant i, varying j.
      float lo = std::min(fj0, fj1), hi = std::max(fj0, fj1);
      float i = fi0;
      float cur = lo;
      while (cur < hi - 0.001f) {
        float next = std::min(std::floor(cur) + 1.0f, hi);
        if (next <= cur + 0.001f) next = std::min(cur + 1.0f, hi);
        add_line(interp_x(i), interp_y(cur), interp_z(i, cur),
                 interp_x(i), interp_y(next), interp_z(i, next), rgba);
        cur = next;
      }
    }
  };

  // Draw a grid layer. All coordinates in fractional SWE cell indices.
  struct LayerBounds { float c0, r0, c1, r1; };

  auto draw_layer = [&](LayerBounds b, float stride, float parent_stride,
                         const float* rgba) {
    // Skip lines on parent cell boundaries (absolute position).
    auto on_parent = [&](float pos) -> bool {
      float rel = pos / parent_stride;
      return std::abs(rel - std::round(rel)) < 0.01f;
    };
    // Horizontal lines. Skip parent boundaries.
    for (float r = b.r0; r <= b.r1 + stride * 0.01f; r += stride) {
      if (on_parent(r)) continue;
      add_line_fi(b.c0, r, b.c1, r, rgba);
    }
    // Vertical lines. Skip parent boundaries.
    for (float c = b.c0; c <= b.c1 + stride * 0.01f; c += stride) {
      if (on_parent(c)) continue;
      add_line_fi(c, b.r0, c, b.r1, rgba);
    }
  };

  // Focus in fractional SWE cell indices.
  float focus_ci = (fx + half_x) / inv_nx;
  float focus_ri = (fy + half_y) / inv_ny;

  // Center N parent-cells on focus with even/odd logic.
  auto center_bounds = [&](float parent_stride, int chunks) -> LayerBounds {
    float extent = chunks * parent_stride;
    float half = extent * 0.5f;
    if (chunks % 2 == 0) {
      float sc = std::round(focus_ci / parent_stride) * parent_stride;
      float sr = std::round(focus_ri / parent_stride) * parent_stride;
      return {sc - half, sr - half, sc + half, sr + half};
    } else {
      float sc = (std::floor(focus_ci / parent_stride) + 0.5f) * parent_stride;
      float sr = (std::floor(focus_ri / parent_stride) + 0.5f) * parent_stride;
      return {sc - half, sr - half, sc + half, sr + half};
    }
  };

  // --- Layer definitions ---
  // stride = cell spacing in SWE cell units.
  // parent_stride = parent's stride (for boundary skipping and centering).
  // chunks = how many parent cells this layer spans.

  // SPH: 10 lines per SWE cell, covers g_sph_chunks SWE cells.
  float sph_color[4] = {0.0f, 1.0f, 0.3f, 0.8f};
  float sph_stride = 1.0f / 10.0f;
  if (g_lod_active[2]) {
    LayerBounds sb = center_bounds(1.0f, g_sph_chunks);
    draw_layer(sb, sph_stride, 1.0f, sph_color);
  }

  // LBM: 2 lines per SPH cell, covers g_lbm_chunks SWE cells.
  float lbm_color[4] = {1.0f, 0.6f, 0.1f, 0.8f};
  float lbm_stride = sph_stride / 2.0f;
  if (g_lod_active[3]) {
    LayerBounds lb = center_bounds(1.0f, g_lbm_chunks);
    draw_layer(lb, lbm_stride, sph_stride, lbm_color);
  }

  // Stokes: 4 lines per LBM cell, covers g_stokes_chunks SWE cells.
  float stokes_color[4] = {0.9f, 0.2f, 0.1f, 0.8f};
  float stokes_stride = lbm_stride / 4.0f;
  if (g_lod_active[4]) {
    LayerBounds stb = center_bounds(1.0f, g_stokes_chunks);
    draw_layer(stb, stokes_stride, lbm_stride, stokes_color);
  }
}

// ---- ImGui side panel (mujoco-water convention: 300px, top-left, collapsing headers) ----

static void DrawPanel(WaterEngine& engine, const WaterEngineConfig& cfg,
                       float cam_distance, mjvCamera& cam,
                       const mjModel* m = nullptr, const mjData* d = nullptr,
                       const std::vector<CoupledBody>* bodies = nullptr) {
  int ww, wh;
  glfwGetWindowSize(glfwGetCurrentContext(), &ww, &wh);
  if (ww < 500 || wh < 400) return;  // skip panel on tiny windows

  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(kPanelWidth, static_cast<float>(wh - 20)),
                           ImGuiCond_Always);

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.12f, 0.95f));
  char title[64];
  snprintf(title, sizeof(title), "mujoco-water | %.0f FPS###main", g_fps);
  ImGui::Begin(title, nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
  ImGui::PopStyleColor();

  // ---- Simulation section ----
  if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Simulate", &g_simulate);
    ImGui::Text("Sim time: %.3f s", engine.sim_time);
    ImGui::Text("Master dt: %.4f s", cfg.master_dt);
    if (ImGui::Button("Reset (R)")) g_reset_requested = true;
    if (ImGui::SliderFloat("Water height", &g_water_z, 1.0f, 5000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic)) {
      g_reload_model = true;
    }
    if (ImGui::SliderFloat("Extent", &g_extent, 0.01f, 10000.0f, "%.2f m", ImGuiSliderFlags_Logarithmic)) {
      g_reload_model = true;
    }
  }

  // ---- Camera section ----
  if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Distance: %.2f m", cam_distance);
    float lookat[3] = {static_cast<float>(cam.lookat[0]),
                       static_cast<float>(cam.lookat[1]),
                       static_cast<float>(cam.lookat[2])};
    if (ImGui::DragFloat3("Look at", lookat, cam_distance * 0.01f)) {
      cam.lookat[0] = lookat[0];
      cam.lookat[1] = lookat[1];
      cam.lookat[2] = lookat[2];
    }
    float elev = static_cast<float>(cam.elevation);
    float azim = static_cast<float>(cam.azimuth);
    float dist = static_cast<float>(cam.distance);
    if (ImGui::SliderFloat("Elevation", &elev, -90.0f, 90.0f)) cam.elevation = elev;
    if (ImGui::SliderFloat("Azimuth", &azim, -180.0f, 180.0f)) cam.azimuth = azim;
    if (ImGui::DragFloat("Distance", &dist, dist * 0.01f, 0.001f, 5000.0f, "%.3f",
                          ImGuiSliderFlags_Logarithmic)) cam.distance = dist;
  }

  // ---- Solver Layers section ----
  if (ImGui::CollapsingHeader("Solver Layers", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Auto LOD (camera distance)", &g_auto_lod);
    ImGui::Separator();
    auto stats = engine.GetLayerStats();

    static const char* names[] = {
      "Ocean (spectral)",
      "Shallow Water",
      "Particles",
      "Lattice Boltzmann",
      "Stokes Flow",
    };
    static const ImVec4 colors[] = {
      {0.3f, 0.5f, 0.9f, 1.0f},  // ocean blue
      {0.0f, 0.8f, 0.8f, 1.0f},  // cyan
      {0.1f, 0.8f, 0.2f, 1.0f},  // green
      {1.0f, 0.6f, 0.1f, 1.0f},  // orange (LBM)
      {0.9f, 0.2f, 0.1f, 1.0f},  // red
    };

    uint32_t total_cells = 0;
    float total_mem = 0;

    for (int i = 0; i < 5; ++i) {
      ImGui::PushID(i);

      const auto& s = stats[i];
      bool active = s.active;

      // All layers get checkboxes. Ocean/SWE always on (disabled).
      ImGui::PushStyleColor(ImGuiCol_Text, colors[i]);
      if (i < 2) {
        bool always_on = true;
        ImGui::BeginDisabled();
        ImGui::Checkbox(names[i], &always_on);
        ImGui::EndDisabled();
      } else {
        if (g_auto_lod) ImGui::BeginDisabled();
        ImGui::Checkbox(names[i], &g_lod_active[i]);
        if (g_auto_lod) ImGui::EndDisabled();
      }
      ImGui::PopStyleColor();

      if (active && s.cells > 0) {
        total_cells += s.cells;
        total_mem += s.memory_kb;
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.1f, 1.0f), "%u cells", s.cells);
      }

      // Per-layer wireframe chunks.
      if (g_show_grid) {
        if (i == 1) ImGui::SliderInt("Fine chunks", &g_fine_chunks, 2, 7);
        if (i == 2 && active) ImGui::SliderInt("SPH chunks", &g_sph_chunks, 2, 10);
        if (i == 3 && active) ImGui::SliderInt("LBM chunks", &g_lbm_chunks, 2, 7);
        if (i == 4 && active) ImGui::SliderInt("Stokes chunks", &g_stokes_chunks, 2, 7);
      }

      ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Checkbox("Wireframe", &g_show_grid);
    ImGui::Text("Total: %u cells  %.1f KB", total_cells, total_mem);
  }

  // ---- Ocean parameters ----
  if (ImGui::CollapsingHeader("Ocean", ImGuiTreeNodeFlags_DefaultOpen)) {
    bool changed = false;
    changed |= ImGui::SliderFloat("Wind (m/s)", &engine.ocean.params.wind_speed,
                                   1.0f, 30.0f);
    changed |= ImGui::SliderFloat("Choppiness", &engine.ocean.params.choppiness,
                                   0.0f, 1.5f);
    changed |= ImGui::SliderFloat("JONSWAP gamma",
                                   &engine.ocean.params.jonswap_gamma,
                                   1.0f, 7.0f);
    if (changed) {
      engine.ocean.Init(engine.ocean.params);
    }
    ImGui::Separator();
    float Hs = 0, peak_omega = 0, peak_amp = 0;
    for (auto& w : engine.ocean.waves) {
      Hs += w.amplitude * w.amplitude;
      if (w.amplitude > peak_amp) { peak_amp = w.amplitude; peak_omega = w.omega; }
    }
    Hs = 4.0f * std::sqrt(Hs);
    float Tp = (peak_omega > 0) ? kTwoPi / peak_omega : 0;
    float Lp = (Tp > 0) ? 9.81f * Tp * Tp / (kTwoPi) : 0;
    ImGui::Text("Hs: %.2f m  Tp: %.1f s", Hs, Tp);
    ImGui::Text("Peak wavelength: %.0f m", Lp);
    ImGui::Text("Waves: %zu  Depth: %.0f m",
                engine.ocean.waves.size(), engine.ocean.params.depth);
  }

  // ---- Coupled Bodies ----
  if (d && bodies && !bodies->empty() &&
      ImGui::CollapsingHeader("Coupled Bodies", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (size_t bi = 0; bi < bodies->size(); ++bi) {
      auto& cb = (*bodies)[bi];
      int bid = cb.body_id;
      const char* name = (m && bid < m->nbody) ? mj_id2name(m, mjOBJ_BODY, bid) : nullptr;
      if (name)
        ImGui::Text("%s (r=%.1fm, %.0f kg/m3)", name, cb.radius,
                    cb.coupling.volume > 0 ? (0.3f / cb.coupling.volume) : 0);  // approximate
      else
        ImGui::Text("Body %d", bid);

      float px = static_cast<float>(d->xpos[3*bid]);
      float py = static_cast<float>(d->xpos[3*bid+1]);
      float pz = static_cast<float>(d->xpos[3*bid+2]);
      float vx = static_cast<float>(d->cvel[6*bid+3]);
      float vy = static_cast<float>(d->cvel[6*bid+4]);
      float vz = static_cast<float>(d->cvel[6*bid+5]);
      float speed = std::sqrt(vx*vx + vy*vy + vz*vz);

      float surface_z = cfg.initial_surface_z;
      if (engine.ocean_active)
        surface_z += engine.ocean.SurfaceHeight(px, py);
      float submersion = (surface_z - (pz - cb.radius)) / (2.0f * cb.radius);
      submersion = std::clamp(submersion, 0.0f, 1.0f);

      float buoyancy = kWaterDensity * kGravity * cb.coupling.volume * submersion;
      float fz = static_cast<float>(d->xfrc_applied[6*bid+2]);

      ImGui::Text("  pos: (%.1f, %.1f, %.1f) m", px, py, pz);
      ImGui::Text("  speed: %.2f m/s", speed);
      ImGui::Text("  submersion: %.0f%%", submersion * 100.0f);
      ImGui::Text("  buoyancy: %.0f N  Fz: %.0f N", buoyancy, fz);
      if (bi + 1 < bodies->size()) ImGui::Separator();
    }
  }

  // ---- Export To Project ----
  if (ImGui::CollapsingHeader("Export To Project")) {
    static int preset = 0;
    static int backend = 0;
    static bool feat_surface_tension = true;
    static bool feat_chemotaxis = false;
    static bool feat_bodies = true;
    static char project_name[128] = "my_water_sim";
    static char export_path[256] = "";
    static char export_status[256] = "";

    const char* backends[] = {"MuJoCo", "Chrono", "Standalone"};
    ImGui::Combo("Backend", &backend, backends, 3);

    const char* presets[] = {
      "Swimming robot (SWE+SPH)",
      "Pool / tank (SWE only)",
      "Ocean vehicle (Ocean+SWE)",
      "Microfluidics (LBM+Stokes)",
      "Full multiscale (all 5)",
    };
    ImGui::Combo("Preset", &preset, presets, 5);
    ImGui::InputText("Name", project_name, sizeof(project_name));

    ImGui::Separator();
    ImGui::Text("Features:");
    if (preset == 0 || preset == 4)
      ImGui::Checkbox("Surface tension", &feat_surface_tension);
    if (preset == 3 || preset == 4)
      ImGui::Checkbox("Chemotaxis", &feat_chemotaxis);
    ImGui::Checkbox("Body coupling", &feat_bodies);
    static bool feat_python = false;
    ImGui::Checkbox("Python bindings (pybind11)", &feat_python);

    ImGui::Separator();
    if (export_path[0] == '\0') {
      snprintf(export_path, sizeof(export_path), "./%s", project_name);
    }
    ImGui::InputText("Output dir", export_path, sizeof(export_path));

    if (ImGui::Button("Export", ImVec2(-1, 0))) {
      // Build config string based on preset.
      struct PresetCfg {
        const char* levels;
        bool has_ocean, has_sph, has_lbm, has_stokes;
        int swe_nx, swe_ny;
        float swe_dx, master_dt;
        float sph_radius, lbm_radius, stokes_radius;
      };
      PresetCfg pcfg[] = {
        {"swe, sph",           false, true,  false, false, 40, 40, 0.1f,  0.01f,  0.5f,  -1.0f, -1.0f},
        {"swe",                false, false, false, false, 30, 30, 0.1f,  0.01f, -1.0f,  -1.0f, -1.0f},
        {"ocean, swe",         true,  false, false, false, 60, 60, 1.0f,  0.01f, -1.0f,  -1.0f, -1.0f},
        {"lbm, stokes",        false, false, true,  true,  10, 10, 0.001f,0.001f,-1.0f,  0.05f, 0.005f},
        {"ocean,swe,sph,lbm,stokes", true, true, true, true, 40,40,0.1f, 0.01f, 0.5f,  0.05f, 0.005f},
      };
      auto& pc = pcfg[preset];

      // Generate main.cpp (varies by backend).
      std::string src;
      src += "// " + std::string(project_name) + ", generated by mujoco-water demo\n";
      src += "// Backend: " + std::string(backends[backend]) + "\n";
      src += "// Preset: " + std::string(presets[preset]) + "\n";
      src += "// LOD levels: " + std::string(pc.levels) + "\n\n";

      // Common water config setup.
      auto gen_water_config = [&](std::string& s) {
        char buf[128];
        s += "  mjwater::WaterEngineConfig cfg;\n";
        if (pc.has_ocean) {
          s += "  cfg.ocean_enabled = true;\n";
          s += "  cfg.ocean.wind_speed = 10.0f;\n";
          s += "  cfg.ocean.jonswap_gamma = 3.3f;\n";
          s += "  cfg.ocean_mean_depth = 50.0f;\n";
        }
        snprintf(buf, sizeof(buf), "  cfg.swe_nx = %d;  cfg.swe_ny = %d;\n", pc.swe_nx, pc.swe_ny);
        s += buf;
        snprintf(buf, sizeof(buf), "  cfg.swe_dx = %.4ff;\n", pc.swe_dx);
        s += buf;
        if (pc.has_sph) {
          s += "  cfg.sph_spacing = 0.02f;\n";
          s += "  cfg.sph_smoothing = 0.03f;\n";
          s += "  cfg.sph_substeps = 10;\n";
        }
        if (pc.has_lbm) {
          s += "  cfg.lbm_nx = 40;  cfg.lbm_ny = 40;  cfg.lbm_nz = 20;\n";
          s += "  cfg.lbm_dx = 0.0005f;\n";
          s += "  cfg.lbm_substeps = 10;\n";
        }
        if (pc.has_stokes) {
          s += "  cfg.stokes_nx = 30;  cfg.stokes_ny = 30;  cfg.stokes_nz = 30;\n";
          s += "  cfg.stokes.dx = 10e-6f;\n";
          s += "  cfg.stokes.diffusivity = 1e-9f;\n";
          s += "  cfg.stokes_substeps = 10;\n";
        }
        snprintf(buf, sizeof(buf), "  cfg.master_dt = %.4ff;\n", pc.master_dt);
        s += buf;
        snprintf(buf, sizeof(buf), "  cfg.lod.sph_radius = %.3ff;\n", pc.sph_radius);
        s += buf;
        snprintf(buf, sizeof(buf), "  cfg.lod.lbm_radius = %.3ff;\n", pc.lbm_radius);
        s += buf;
        snprintf(buf, sizeof(buf), "  cfg.lod.stokes_radius = %.4ff;\n", pc.stokes_radius);
        s += buf;
        s += "  cfg.initial_surface_z = 0.5f;\n";
        s += "  cfg.domain = {{{0.0f, 0.0f, 0.0f}, {4.0f, 4.0f, 2.0f}}};\n\n";
        s += "  mjwater::WaterEngine water;\n  water.Init(cfg);\n\n";
      };

      if (backend == 0) {
        // MuJoCo backend.
        src += "#include <cstdio>\n#include <mujoco/mujoco.h>\n";
        src += "#include \"mjwater/water_engine.h\"\n";
        if (feat_bodies) src += "#include \"mjwater/mujoco_utils.h\"\n";
        src += "\nint main() {\n";
        gen_water_config(src);
        src += "  char error[1000] = \"\";\n";
        src += "  mjModel* m = mj_loadXML(\"model.xml\", nullptr, error, sizeof(error));\n";
        src += "  if (!m) { printf(\"Load error: %s\\n\", error); return 1; }\n";
        src += "  mjData* d = mj_makeData(m);\n";
        src += "  m->opt.density = 0;\n  m->opt.viscosity = 0;\n\n";
        src += "  for (int step = 0; step < 10000; ++step) {\n";
        src += "    water.SetFocus({(float)d->qpos[0], (float)d->qpos[1], (float)d->qpos[2]});\n";
        src += "    water.Step();\n    mj_step(m, d);\n";
        src += "    if (step % 1000 == 0) printf(\"Step %d\\n\", step);\n";
        src += "  }\n\n";
        src += "  mj_deleteData(d);\n  mj_deleteModel(m);\n  return 0;\n}\n";
      } else if (backend == 1) {
        // Chrono backend.
        src += "// Requires: Project Chrono (https://projectchrono.org)\n";
        src += "#include <cstdio>\n";
        src += "#include \"chrono/physics/ChSystemNSC.h\"\n";
        src += "#include \"chrono/physics/ChBodyEasyBox.h\"\n";
        src += "#include \"mjwater/water_engine.h\"\n";
        if (feat_bodies) src += "// #include \"mjwater/chrono_utils.h\"  // TODO: implement\n";
        src += "\nint main() {\n";
        gen_water_config(src);
        src += "  chrono::ChSystemNSC sys;\n";
        src += "  sys.SetGravitationalAcceleration({0, 0, -9.81});\n\n";
        src += "  // Add a floating body.\n";
        src += "  auto body = chrono_types::make_shared<chrono::ChBodyEasyBox>(";
        src += "0.5, 0.5, 0.2, 500, true);\n";
        src += "  body->SetPos({0, 0, 0.5});\n";
        src += "  sys.AddBody(body);\n\n";
        src += "  for (int step = 0; step < 10000; ++step) {\n";
        src += "    auto p = body->GetPos();\n";
        src += "    water.SetFocus({(float)p.x(), (float)p.y(), (float)p.z()});\n";
        src += "    water.Step();\n";
        src += "    // TODO: apply water forces via chrono_utils.h\n";
        src += "    sys.DoStepDynamics(0.001);\n";
        src += "    if (step % 1000 == 0) printf(\"Step %d\\n\", step);\n";
        src += "  }\n  return 0;\n}\n";
      } else {
        // Standalone (no physics engine).
        src += "#include <cstdio>\n";
        src += "#include \"mjwater/water_engine.h\"\n";
        src += "\nint main() {\n";
        gen_water_config(src);
        src += "  for (int step = 0; step < 10000; ++step) {\n";
        src += "    water.SetFocus({0, 0, 0});\n";
        src += "    water.Step();\n";
        src += "    if (step % 1000 == 0) {\n";
        src += "      printf(\"Step %d, volume=%.4f\\n\", step, water.TotalVolume());\n";
        src += "    }\n";
        src += "  }\n  return 0;\n}\n";
      }

      // Generate CMakeLists.txt (varies by backend).
      std::string cmake;
      cmake += "cmake_minimum_required(VERSION 3.20)\n";
      cmake += "project(" + std::string(project_name) + " LANGUAGES CXX)\n";
      cmake += "set(CMAKE_CXX_STANDARD 20)\n\n";
      cmake += "add_subdirectory(../mujoco-water ${CMAKE_BINARY_DIR}/mujoco-water)\n\n";
      cmake += "add_executable(${PROJECT_NAME} main.cpp)\n";
      cmake += "target_link_libraries(${PROJECT_NAME} PRIVATE mujoco-water)\n";
      if (backend == 1) {
        cmake += "\n# Chrono (set CHRONO_DIR or install via package manager)\n";
        cmake += "find_package(Chrono REQUIRED)\n";
        cmake += "target_link_libraries(${PROJECT_NAME} PRIVATE Chrono::core)\n";
      }
      if (feat_python) {
        cmake += "\n# Python bindings (pybind11)\n";
        cmake += "find_package(pybind11 REQUIRED)\n";
        cmake += "pybind11_add_module(py${PROJECT_NAME} bindings.cpp)\n";
        cmake += "target_link_libraries(py${PROJECT_NAME} PRIVATE mujoco-water)\n";
      }

      // Write files
      std::string dir(export_path);
      std::filesystem::create_directories(dir);

      std::string main_path = dir + "/main.cpp";
      std::string cmake_path = dir + "/CMakeLists.txt";

      FILE* f1 = nullptr;
      FILE* f2 = nullptr;
#ifdef _WIN32
      fopen_s(&f1, main_path.c_str(), "w");
      fopen_s(&f2, cmake_path.c_str(), "w");
#else
      f1 = fopen(main_path.c_str(), "w");
      f2 = fopen(cmake_path.c_str(), "w");
#endif
      if (f1 && f2) {
        fwrite(src.c_str(), 1, src.size(), f1);
        fwrite(cmake.c_str(), 1, cmake.size(), f2);
        fclose(f1); fclose(f2);
        snprintf(export_status, sizeof(export_status),
                 "Exported to %s/", export_path);
      } else {
        if (f1) fclose(f1);
        if (f2) fclose(f2);
        snprintf(export_status, sizeof(export_status),
                 "ERROR: could not write to %s/", export_path);
      }
    }

    if (export_status[0]) {
      ImGui::TextWrapped("%s", export_status);
    }
  }

  // ---- Controls help ----
  if (ImGui::CollapsingHeader("Controls")) {
    ImGui::BulletText("Scroll: zoom in/out");
    ImGui::BulletText("Left drag: orbit");
    ImGui::BulletText("Right drag: pan");
    ImGui::BulletText("R: reset simulation");
    ImGui::BulletText("ESC: quit");
  }

  ImGui::End();
}

// ---- Main ----

int main() {
  // Build and load MuJoCo model.
  std::string mjcf = BuildMJCF(g_water_z + 10.0f);
  char error[1024] = {};
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", mjcf.c_str(),
                  static_cast<int>(mjcf.size()));
  mjModel* m = mj_loadXML("model.xml", &vfs, error, sizeof(error));
  mj_deleteVFS(&vfs);
  if (!m) { fprintf(stderr, "MuJoCo: %s\n", error); return 1; }
  mjData* d = mj_makeData(m);
  mj_forward(m, d);

  int hfield_id = mj_name2id(m, mjOBJ_HFIELD, "water_surface");
  if (hfield_id < 0) { fprintf(stderr, "No hfield\n"); return 1; }

  std::vector<CoupledBody> coupled_bodies;
  InitCoupledBodies(m, coupled_bodies);

  // Initialize WaterEngine with all 5 levels.
  WaterEngine engine;
  WaterEngineConfig cfg;
  cfg.ocean_enabled = true;
  cfg.ocean.wind_speed = 12.0f;
  cfg.ocean.jonswap_gamma = 3.3f;
  cfg.ocean.depth = 100.0f;
  cfg.ocean.num_waves = 64;
  cfg.swe_nx = kSweN;
  cfg.swe_ny = kSweN;
  cfg.swe_dx = kSweDx;
  cfg.initial_surface_z = g_water_z;
  float half_domain = kSweDx * kSweN * 0.5f;
  cfg.domain = {{-half_domain, -half_domain, -100},
                {half_domain, half_domain, cfg.initial_surface_z + 50}};
  cfg.master_dt = 0.01f;
  cfg.sph_substeps = 1;
  cfg.sph_smoothing = 0.06f;
  cfg.sph_spacing = 0.05f;
  cfg.lbm_nx = 8; cfg.lbm_ny = 8; cfg.lbm_nz = 8;
  cfg.lbm_dx = 1.0f;
  cfg.lbm_substeps = 1;
  cfg.stokes_nx = 10; cfg.stokes_ny = 10; cfg.stokes_nz = 10;
  cfg.stokes.dx = 10e-6f;
  cfg.stokes.diffusivity = 1e-9f;
  cfg.stokes_substeps = 1;
  cfg.sponge_width = 5;  // 5 cells absorbing boundary on each edge
  cfg.lod.lbm_radius = 0.05f;
  cfg.lod.sph_radius = 0.3f;
  cfg.lod.swe_radius = 1000.0f;
  cfg.lod.stokes_radius = 0.005f;
  engine.Init(cfg);

  // Register coupled bodies with engine for two-way coupling.
  for (auto& cb : coupled_bodies) {
    engine.AddBody(cb.coupling);
  }

  // Place amoeba (concentration source) at Stokes grid center.
  auto PlaceAmoeba = [&]() {
    uint32_t cx = cfg.stokes_nx / 2;
    uint32_t cy = cfg.stokes_ny / 2;
    uint32_t cz = cfg.stokes_nz / 2;
    engine.stokes.SetSolid(cx, cy, cz, true);
    engine.stokes.SetSource(cx, cy, cz, 1000.0f);
  };
  PlaceAmoeba();

  // Open viewer.
  Viewer viewer;
  if (!viewer.Init(m)) { fprintf(stderr, "No viewer\n"); return 1; }

  printf("[multiscale] Multi-scale water simulation with ImGui controls\n");

  // Frame-rate-independent physics with time budget.
  auto prev_time = std::chrono::steady_clock::now();
  double physics_accumulator = 0.0;
  constexpr double kMaxPhysicsPerFrame = 0.012;

  while (viewer.Running()) {
    auto now = std::chrono::steady_clock::now();
    double frame_dt = std::chrono::duration<double>(now - prev_time).count();
    prev_time = now;
    if (frame_dt > 0.1) frame_dt = 0.1;
    g_fps = (frame_dt > 0) ? static_cast<float>(1.0 / frame_dt) : 0;

    if (g_simulate) physics_accumulator += frame_dt;

    // Handle model reload (depth changed).
    if (g_reload_model) {
      g_reload_model = false;
      // Save camera and body state.
      mjvCamera saved_cam = viewer.cam;
      std::vector<double> saved_qpos(d->qpos, d->qpos + m->nq);
      std::vector<double> saved_qvel(d->qvel, d->qvel + m->nv);
      // Rebuild model.
      std::string new_mjcf = BuildMJCF(g_water_z + 10.0f);
      mjVFS new_vfs;
      mj_defaultVFS(&new_vfs);
      mj_addBufferVFS(&new_vfs, "model.xml", new_mjcf.c_str(),
                       static_cast<int>(new_mjcf.size()));
      char reload_err[1024] = {};
      mjModel* new_m = mj_loadXML("model.xml", &new_vfs, reload_err, sizeof(reload_err));
      mj_deleteVFS(&new_vfs);
      if (new_m) {
        mjData* new_d = mj_makeData(new_m);
        mj_forward(new_m, new_d);
        mjr_freeContext(&viewer.con);
        mjr_makeContext(new_m, &viewer.con, mjFONTSCALE_150);
        mj_deleteData(d);
        mj_deleteModel(m);
        m = new_m;
        d = new_d;
        hfield_id = mj_name2id(m, mjOBJ_HFIELD, "water_surface");
        // Rebuild coupled bodies.
        InitCoupledBodies(m, coupled_bodies);
      }
      // Restore body state (preserve ball position/velocity across reload).
      if (new_m && static_cast<int>(saved_qpos.size()) == m->nq) {
        std::copy(saved_qpos.begin(), saved_qpos.end(), d->qpos);
        std::copy(saved_qvel.begin(), saved_qvel.end(), d->qvel);
        mj_forward(m, d);
      }
      // Rebuild config from extent, then override water height.
      viewer.cam = saved_cam;
      cfg = mjwater::WaterEngineConfig::ForExtent(g_extent, kSweN);
      cfg.initial_surface_z = g_water_z;
      cfg.domain.min.z = -g_water_z;
      cfg.domain.max.z = g_water_z + g_extent * 0.05f;
      cfg.ocean.wind_speed = 15.0f;
      cfg.ocean.jonswap_gamma = 3.3f;
      // Recompute fetch-limited amplitude for the actual wind speed.
      float U = cfg.ocean.wind_speed, g = 9.80665f;
      float fetch_full = 77000.0f * (U * U) / (g * g);
      cfg.ocean.amplitude_scale = std::sqrt(std::min(1.0f, g_extent / fetch_full));
      engine.Init(cfg);
      PlaceAmoeba();
      physics_accumulator = 0;
    }

    // Handle reset.
    if (g_reset_requested) {
      g_reset_requested = false;
      engine.Init(cfg);
      PlaceAmoeba();
      physics_accumulator = 0;
    }

    // Auto-LOD: activate finer levels as camera zooms in.
    if (g_auto_lod) {
      float dist = static_cast<float>(viewer.cam.distance);
      g_lod_active[2] = dist < 30.0f;    // SPH at < 30m
      g_lod_active[3] = dist < 5.0f;    // LBM at < 5m
      g_lod_active[4] = dist < 0.1f;    // Stokes at < 10cm
    }

    // LOD activation from toggles (auto or manual).
    engine.lod_manager.config.sph_radius = g_lod_active[2]
        ? cfg.lod.sph_radius : -1.0f;
    engine.lod_manager.config.lbm_radius = g_lod_active[3]
        ? cfg.lod.lbm_radius : -1.0f;
    engine.lod_manager.config.stokes_radius = g_lod_active[4]
        ? cfg.lod.stokes_radius : -1.0f;
    engine.SetFocus({static_cast<float>(viewer.cam.lookat[0]),
                     static_cast<float>(viewer.cam.lookat[1]),
                     cfg.initial_surface_z});

    // Step physics with time budget.
    auto physics_start = std::chrono::steady_clock::now();
    while (physics_accumulator >= cfg.master_dt) {
      // Update engine body states for two-way coupling.
      for (size_t bi = 0; bi < coupled_bodies.size(); ++bi) {
        int bid = coupled_bodies[bi].body_id;
        if (bi < engine.body_states.size()) {
          engine.body_states[bi].pos = {
            static_cast<float>(d->xpos[3*bid]),
            static_cast<float>(d->xpos[3*bid+1]),
            static_cast<float>(d->xpos[3*bid+2])};
          engine.body_states[bi].vel = {
            static_cast<float>(d->cvel[6*bid+3]),
            static_cast<float>(d->cvel[6*bid+4]),
            static_cast<float>(d->cvel[6*bid+5])};
        }
      }

      engine.Step();

      // Apply water forces to all coupled bodies.
      for (auto& cb : coupled_bodies) {
        int bid = cb.body_id;
        for (int i = 0; i < 6; ++i) d->xfrc_applied[6*bid+i] = 0;
        float bx = static_cast<float>(d->xpos[3*bid]);
        float by = static_cast<float>(d->xpos[3*bid+1]);
        float bz = static_cast<float>(d->xpos[3*bid+2]);
        Vec3 query_pos = {bx, by, bz - cb.radius};
        Vec3 vel = {static_cast<float>(d->cvel[6*bid+3]),
                    static_cast<float>(d->cvel[6*bid+4]),
                    static_cast<float>(d->cvel[6*bid+5])};
        auto fluid = FluidCoupling::QueryOcean(engine.ocean, query_pos);
        fluid.depth += cfg.initial_surface_z;
        fluid.submerged = (fluid.depth > 0);
        auto force = FluidCoupling::ComputeForces(fluid, cb.coupling, vel,
                                                   cfg.master_dt, 2.0f * cb.radius);
        d->xfrc_applied[6*bid+0] += force.force.x;
        d->xfrc_applied[6*bid+1] += force.force.y;
        d->xfrc_applied[6*bid+2] += force.force.z;
        cb.coupling.prev_fluid_vel = fluid.velocity;
      }
      mj_step(m, d);
      physics_accumulator -= cfg.master_dt;

      double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - physics_start).count();
      if (elapsed > kMaxPhysicsPerFrame) {
        physics_accumulator = 0;
        break;
      }
    }

    // Update hfield.
    UpdateHField(m, viewer.con, engine, hfield_id);

    // MuJoCo render.
    int w, h;
    glfwGetFramebufferSize(viewer.window, &w, &h);
    mjrRect viewport = {0, 0, w, h};

    mjv_updateScene(m, d, &viewer.opt, nullptr, &viewer.cam,
                    mjCAT_ALL, &viewer.scn);

    RenderDynamicGeoms(&viewer.scn, engine);
    RenderWireframe(&viewer.scn, engine, viewer.cam);

    mjr_render(viewport, &viewer.scn, &viewer.con);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    DrawPanel(engine, cfg, static_cast<float>(viewer.cam.distance), viewer.cam,
              m, d, &coupled_bodies);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(viewer.window);
    glfwPollEvents();
  }

  viewer.Shutdown();
  mj_deleteData(d);
  mj_deleteModel(m);
  printf("[multiscale] Done.\n");
  return 0;
}
