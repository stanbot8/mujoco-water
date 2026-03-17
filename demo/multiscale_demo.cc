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

using namespace mjwater;

// ---- Constants ----

static constexpr int kSweN = 50;
static constexpr float kSweDx = 20.0f;            // 20m cells -> 1km x 1km domain
static constexpr float kMaxWaterZ = 20.0f;        // hfield z range (ocean waves at km scale)
static constexpr float kOceanFloorZ = -100.0f;    // visual ocean floor
static constexpr float kPanelWidth = 300.0f;      // side panel width (fwmc convention)

// ---- Global state ----

static bool g_lod_active[5] = {true, true, true, true, true};
static bool g_reset_requested = false;
static bool g_simulate = true;
static bool g_auto_lod = true;  // auto-activate LOD levels based on camera distance
static float g_fps = 0;

// ---- MJCF ----

static std::string BuildMJCF() {
  char buf[4096];
  snprintf(buf, sizeof(buf), R"(
<mujoco model="multiscale_demo">
  <option timestep="0.002" gravity="0 0 -9.81"/>

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
            size="%f %f %f 50"/>
    <material name="water_mat" rgba="0.08 0.35 0.55 0.82"
              specular="0.5" shininess="0.6"/>
  </asset>

  <worldbody>
    <light pos="0 0 500" dir="0 0 -1" diffuse="0.9 0.9 0.85" castshadow="false"/>
    <light pos="300 -300 400" dir="-0.5 0.5 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <light pos="-200 200 300" dir="0.3 -0.3 -1" diffuse="0.3 0.35 0.4" castshadow="false"/>

    <!-- Deep ocean floor -->
    <geom type="plane" size="2000 2000 0.1" pos="0 0 %f"
          rgba="0.12 0.18 0.25 1" contype="0" conaffinity="0"/>

    <!-- Water surface height field -->
    <geom name="water" type="hfield" hfield="water_surface"
          pos="0 0 0" material="water_mat"
          contype="0" conaffinity="0"/>

  </worldbody>
</mujoco>
  )", kSweN, kSweN,
      kSweDx * kSweN * 0.5f, kSweDx * kSweN * 0.5f, kMaxWaterZ,
      kOceanFloorZ);
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
        mjv_moveCamera(v->model, mjMOUSE_ROTATE_V, dx/ww, dy/wh, &v->scn, &v->cam);
      } else if (v->mouse_right) {
        bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        // Pan speed scales with distance.
        double speed = v->cam.distance * 0.002;
        if (shift) {
          // Shift+right drag: boom arm height.
          v->cam.lookat[2] += dy / wh * v->cam.distance * 0.5;
        } else {
          // Right drag: move pawn on XY plane.
          // Use MuJoCo's own pan math (knows its own azimuth convention),
          // then restore Z to lock movement to the XY plane.
          double z = v->cam.lookat[2];
          mjv_moveCamera(v->model, mjMOUSE_MOVE_H, -dx/ww, -dy/wh, &v->scn, &v->cam);
          v->cam.lookat[2] = z;
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
    cam.lookat[0] = 0; cam.lookat[1] = 0; cam.lookat[2] = 20.0;
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
// The height field is the single rendered water mesh. Each cell samples
// the SWE surface height (which is the canonical height field), with
// ocean waves added on top. SPH/LBM/Stokes run underneath and affect
// forces, but the visible surface comes from SWE + ocean.

static void UpdateHField(mjModel* m, const mjrContext& con,
                          const WaterEngine& engine, int hfield_id) {
  const auto& swe = engine.swe;
  const auto& h_data = swe.grid.channels[swe.ch_h].data;
  const auto& b_data = swe.grid.channels[swe.ch_bathy].data;

  int nrow = m->hfield_nrow[hfield_id];
  int ncol = m->hfield_ncol[hfield_id];
  float* hdata = m->hfield_data + m->hfield_adr[hfield_id];
  float inv_zmax = 1.0f / kMaxWaterZ;

  for (int row = 0; row < nrow; ++row) {
    for (int col = 0; col < ncol; ++col) {
      uint32_t gx = static_cast<uint32_t>(col);
      uint32_t gy = static_cast<uint32_t>(nrow - 1 - row);
      if (gx >= swe.grid.nx) gx = swe.grid.nx - 1;
      if (gy >= swe.grid.ny) gy = swe.grid.ny - 1;

      size_t idx = swe.grid.Idx(gx, gy);
      float surface = b_data[idx] + h_data[idx];

      if (engine.ocean_active) {
        float wx, wy;
        swe.grid.GridToWorld(gx, gy, wx, wy);
        // Scale ocean waves to be visible but not overwhelming.
        // Raw ocean height can be meters; we want ~10cm perturbations.
        surface += engine.ocean.SurfaceHeight(wx, wy);
      }

      float normalized = std::clamp(surface * inv_zmax, 0.02f, 1.0f);
      hdata[row * ncol + col] = normalized;
    }
  }
  mjr_uploadHField(m, &con, hfield_id);
}

// ---- Dynamic geom rendering ----

static bool g_show_grid = true;
static int g_fine_chunks = 3;  // NxN chunks of fine wireframe around focus (2-10)
// Sticky fine area: only recalculate when focus leaves current bounds.
static uint32_t g_fine_cc0 = 0, g_fine_cr0 = 0, g_fine_cc1 = 0, g_fine_cr1 = 0;
static bool g_fine_valid = false;

static void RenderDynamicGeoms(mjvScene* scn, const WaterEngine& engine,
                                const WaterEngineConfig& cfg,
                                const mjModel* m, int hfield_id,
                                const mjvCamera& cam) {
  // Amoeba at Stokes grid center.
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

  // Wireframe is now drawn in screen space after mjr_render. See RenderWireframe().
}

// Screen-space wireframe: projects hfield vertices via GL matrices, draws ImGui lines.
// Vertex z computation is identical to the old 3D geom approach.
static void RenderWireframe(const WaterEngine& engine, const mjModel* m,
                             int hfield_id, const mjvCamera& cam) {
  if (!g_show_grid) return;

  // Read GL matrices left by mjr_render.
  float proj[16], modelview[16];
  glGetFloatv(GL_PROJECTION_MATRIX, proj);
  glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
  int vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
  float vpw = static_cast<float>(vp[2]), vph = static_cast<float>(vp[3]);

  uint32_t nx = engine.swe.grid.nx, ny = engine.swe.grid.ny;

  // LOD wireframe: dense near focus, coarse far away.
  // Pre-compute vertex world positions and z once (O(N) not O(N^2)).
  float half_x = kSweDx * kSweN * 0.5f;
  float half_y = kSweDx * kSweN * 0.5f;
  constexpr float kZDown = 0.01f;
  constexpr float kZRange = kMaxWaterZ + kZDown;
  float z_offset = static_cast<float>(cam.distance) * 0.0002f;

  int ncol = m->hfield_ncol[hfield_id];
  const float* hdata = m->hfield_data + m->hfield_adr[hfield_id];
  float inv_nx = 2.0f * half_x / (nx - 1);
  float inv_ny = 2.0f * half_y / (ny - 1);

  // Pre-compute world x/y per vertex (avoid repeated math in inner loop).
  std::vector<float> wx(nx), wy(ny);
  for (uint32_t c = 0; c < nx; ++c) wx[c] = -half_x + c * inv_nx;
  for (uint32_t r = 0; r < ny; ++r) wy[r] = -half_y + r * inv_ny;

  // Pre-compute z per vertex from hfield data.
  std::vector<float> wz(nx * ny);
  for (uint32_t r = 0; r < ny; ++r)
    for (uint32_t c = 0; c < nx; ++c)
      wz[r * nx + c] = -kZDown + hdata[r * ncol + c] * kZRange + z_offset;

  // ---- LOD Wireframe ----
  // Two layers: coarse grid everywhere, fine grid in chunks near focus.
  // Fine area is sticky: only recalculates when focus leaves current bounds.
  // Even N: focus at center intersection. Odd N: focus chunk is center chunk.

  constexpr uint32_t kCoarse = 5;

  // Project 3D point to screen coords using MuJoCo's GL matrices.
  auto project = [&](float x, float y, float z, float& sx, float& sy) -> bool {
    float ex = modelview[0]*x + modelview[4]*y + modelview[8]*z + modelview[12];
    float ey = modelview[1]*x + modelview[5]*y + modelview[9]*z + modelview[13];
    float ez = modelview[2]*x + modelview[6]*y + modelview[10]*z + modelview[14];
    float cx = proj[0]*ex + proj[4]*ey + proj[8]*ez + proj[12];
    float cy = proj[1]*ex + proj[5]*ey + proj[9]*ez + proj[13];
    float cw = proj[3]*ex + proj[7]*ey + proj[11]*ez + proj[15];
    if (cw <= 0.001f) return false;
    sx = (cx/cw * 0.5f + 0.5f) * vpw;
    sy = (1.0f - (cy/cw * 0.5f + 0.5f)) * vph;
    return true;
  };

  // Pre-compute screen positions for all vertices.
  struct SV { float sx, sy; bool vis; };
  std::vector<SV> sv(nx * ny);
  for (uint32_t r = 0; r < ny; ++r)
    for (uint32_t c = 0; c < nx; ++c) {
      auto& v = sv[r * nx + c];
      v.vis = project(wx[c], wy[r], wz[r*nx+c], v.sx, v.sy);
    }

  auto* dl = ImGui::GetForegroundDrawList();
  ImU32 yellow = IM_COL32(255, 255, 0, 200);

  auto draw_edge = [&](uint32_t r0, uint32_t c0, uint32_t r1, uint32_t c1) {
    auto& a = sv[r0*nx+c0]; auto& b = sv[r1*nx+c1];
    if (a.vis && b.vis) dl->AddLine({a.sx, a.sy}, {b.sx, b.sy}, yellow, 1.0f);
  };

  // Map focus world position to cell index.
  float fx = static_cast<float>(cam.lookat[0]);
  float fy = static_cast<float>(cam.lookat[1]);
  int fc = std::clamp(static_cast<int>((fx + half_x) / (2.0f * half_x) * (nx - 1)),
                      0, static_cast<int>(nx - 1));
  int fr = std::clamp(static_cast<int>((fy + half_y) / (2.0f * half_y) * (ny - 1)),
                      0, static_cast<int>(ny - 1));

  // Compute fine area bounds every frame (snaps to chunk boundaries).
  int ck = static_cast<int>(kCoarse);
  int N = g_fine_chunks;
  int chunk_c = fc / ck, chunk_r = fr / ck;
  int max_chunks = static_cast<int>(nx / kCoarse);
  // Even/odd centering:
  // Odd: center chunk = focus chunk, expand (N-1)/2 each side.
  // Even: pick side based on position within chunk.
  int offset_c = (N - 1) / 2, offset_r = (N - 1) / 2;
  if (N % 2 == 0) {
    if ((fc % ck) < ck / 2) offset_c = N / 2;
    if ((fr % ck) < ck / 2) offset_r = N / 2;
  }
  int c_lo = std::clamp(chunk_c - offset_c, 0, std::max(max_chunks - N, 0));
  int r_lo = std::clamp(chunk_r - offset_r, 0, std::max(max_chunks - N, 0));
  uint32_t cc0 = static_cast<uint32_t>(c_lo) * kCoarse;
  uint32_t cr0 = static_cast<uint32_t>(r_lo) * kCoarse;
  uint32_t cc1 = std::min(static_cast<uint32_t>(c_lo + N) * kCoarse, nx - 1);
  uint32_t cr1 = std::min(static_cast<uint32_t>(r_lo + N) * kCoarse, ny - 1);

  // Is cell (r,c) inside the fine area? Only applies when fine grid is visible.
  bool show_fine = cam.distance <= 500.0;
  auto in_fine = [&](uint32_t r, uint32_t c) {
    return show_fine && r >= cr0 && r <= cr1 && c >= cc0 && c <= cc1;
  };

  // Pass 1: Coarse grid. Skip edges fully inside fine area.
  // Horizontal edges (including boundary row ny-1).
  auto coarse_row = [&](uint32_t r) {
    for (uint32_t c = 0; c + 1 < nx; c += kCoarse) {
      uint32_t c2 = std::min(c + kCoarse, nx - 1);
      if (in_fine(r, c) && in_fine(r, c2)) continue;
      draw_edge(r, c, r, c2);
    }
  };
  for (uint32_t r = 0; r < ny; r += kCoarse) coarse_row(r);
  if ((ny - 1) % kCoarse != 0) coarse_row(ny - 1);

  // Vertical edges (including boundary col nx-1).
  auto coarse_col = [&](uint32_t c) {
    for (uint32_t r = 0; r + 1 < ny; r += kCoarse) {
      uint32_t r2 = std::min(r + kCoarse, ny - 1);
      if (in_fine(r, c) && in_fine(r2, c)) continue;
      draw_edge(r, c, r2, c);
    }
  };
  for (uint32_t c = 0; c < nx; c += kCoarse) coarse_col(c);
  if ((nx - 1) % kCoarse != 0) coarse_col(nx - 1);

  // Green zone: kCoarse x kCoarse fine cells centered on the focus cell.
  int ghalf = static_cast<int>(kCoarse) / 2;
  uint32_t gc0 = static_cast<uint32_t>(std::max(fc - ghalf, 0));
  uint32_t gr0 = static_cast<uint32_t>(std::max(fr - ghalf, 0));
  uint32_t gc1 = std::min(gc0 + kCoarse, nx - 1);
  uint32_t gr1 = std::min(gr0 + kCoarse, ny - 1);
  bool show_green = engine.sph_active && cam.distance <= 50.0;

  auto in_green = [&](uint32_t r, uint32_t c) {
    return show_green && r >= gr0 && r <= gr1 && c >= gc0 && c <= gc1;
  };

  // Pass 2: Fine yellow grid (every cell). Skip edges inside green zone.
  if (show_fine) {
    for (uint32_t r = cr0; r <= cr1; ++r)
      for (uint32_t c = cc0; c < cc1; ++c)
        draw_edge(r, c, r, c+1);
    for (uint32_t c = cc0; c <= cc1; ++c)
      for (uint32_t r = cr0; r < cr1; ++r)
        draw_edge(r, c, r+1, c);
  }

  // Pass 3: Green 2m grid fills the focus zone when SPH is active.
  // Interpolates z from hfield vertices to follow the wave surface.
  if (show_green) {
    float green[4] = {0.0f, 1.0f, 0.3f, 0.9f};
    float cell = (2.0f * half_x / (nx - 1)) / 10.0f;  // 1/10th of actual hfield cell
    float gx0 = wx[gc0], gx1 = wx[gc1];
    float gy0 = wy[gr0], gy1 = wy[gr1];
    float inv_dx = (nx - 1) / (2.0f * half_x);
    float inv_dy = (ny - 1) / (2.0f * half_y);

    // Sample z from hfield with bilinear interpolation.
    auto sample_z = [&](float x, float y) -> float {
      float gx = (x + half_x) * inv_dx;
      float gy = (y + half_y) * inv_dy;
      int ix = std::clamp(static_cast<int>(gx), 0, static_cast<int>(nx - 2));
      int iy = std::clamp(static_cast<int>(gy), 0, static_cast<int>(ny - 2));
      float fx = gx - ix, fy = gy - iy;
      float z00 = wz[iy * nx + ix], z10 = wz[iy * nx + ix + 1];
      float z01 = wz[(iy+1) * nx + ix], z11 = wz[(iy+1) * nx + ix + 1];
      return (1-fx)*(1-fy)*z00 + fx*(1-fy)*z10 + (1-fx)*fy*z01 + fx*fy*z11;
    };

    // Draw green lines as segments between hfield vertices for z accuracy.
    float seg = 2.0f * half_x / (nx - 1);  // SWE cell spacing in meters
    // Skip green lines that fall on yellow cell boundaries (every 10th green line).
    auto on_yellow = [&](float pos, float origin) -> bool {
      float rel = (pos - origin) / cell;
      int nearest = static_cast<int>(rel + 0.5f);
      return nearest % 10 == 0 && std::abs(rel - nearest) < 0.01f;
    };
    ImU32 green_col = IM_COL32(0, 255, 80, 200);
    auto draw_green = [&](float x0, float y0, float z0, float x1, float y1, float z1) {
      float sx0, sy0, sx1, sy1;
      if (project(x0, y0, z0, sx0, sy0) && project(x1, y1, z1, sx1, sy1))
        dl->AddLine({sx0, sy0}, {sx1, sy1}, green_col, 1.0f);
    };
    // Horizontal lines.
    for (float y = gy0; y <= gy1 + 0.01f; y += cell) {
      if (on_yellow(y, gy0)) continue;
      for (float x = gx0; x < gx1 - 0.01f; x += seg) {
        float x2 = std::min(x + seg, gx1);
        draw_green(x, y, sample_z(x, y) + z_offset, x2, y, sample_z(x2, y) + z_offset);
      }
    }
    // Vertical lines.
    for (float x = gx0; x <= gx1 + 0.01f; x += cell) {
      if (on_yellow(x, gx0)) continue;
      for (float y = gy0; y < gy1 - 0.01f; y += seg) {
        float y2 = std::min(y + seg, gy1);
        draw_green(x, y, sample_z(x, y) + z_offset, x, y2, sample_z(x, y2) + z_offset);
      }
    }
  }
}

// ---- ImGui side panel (fwmc convention: 300px, top-left, collapsing headers) ----

static void DrawPanel(WaterEngine& engine, const WaterEngineConfig& cfg,
                       float cam_distance, mjvCamera& cam) {
  int ww, wh;
  glfwGetWindowSize(glfwGetCurrentContext(), &ww, &wh);
  if (ww < 500 || wh < 400) return;  // skip panel on tiny windows

  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(kPanelWidth, static_cast<float>(wh - 20)),
                           ImGuiCond_Always);

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.12f, 0.85f));
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
    ImGui::Separator();
    ImGui::Checkbox("Wireframe", &g_show_grid);
    if (g_show_grid) {
      if (ImGui::SliderInt("Fine chunks", &g_fine_chunks, 2, 7))
        g_fine_valid = false;
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

  // ---- LOD Layers section ----
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
    static const char* scales[] = {"km", "meters", "cm", "mm", "um"};
    static const ImVec4 colors[] = {
      {0.3f, 0.5f, 0.9f, 1.0f},  // ocean blue
      {0.0f, 0.8f, 0.8f, 1.0f},  // cyan
      {0.1f, 0.8f, 0.2f, 1.0f},  // green
      {0.9f, 0.8f, 0.1f, 1.0f},  // yellow
      {0.9f, 0.2f, 0.1f, 1.0f},  // red
    };

    uint32_t total_cells = 0;
    float total_mem = 0;

    for (int i = 0; i < 5; ++i) {
      ImGui::PushID(i);
      ImGui::PushStyleColor(ImGuiCol_Text, colors[i]);

      // Ocean and SWE are always structurally on; checkbox controls finer levels.
      if (i >= 2) {
        if (g_auto_lod) ImGui::BeginDisabled();
        ImGui::Checkbox(names[i], &g_lod_active[i]);
        if (g_auto_lod) ImGui::EndDisabled();
      } else {
        ImGui::TextColored(colors[i], "%s", names[i]);
      }

      ImGui::PopStyleColor();
      ImGui::SameLine();
      ImGui::TextDisabled("(%s)", scales[i]);

      const auto& s = stats[i];
      if (s.active) {
        ImGui::Text("  %u cells", s.cells);
        total_cells += s.cells;
        total_mem += s.memory_kb;

        // Compute cost bar.
        float cost_norm = std::min(s.step_us / 10000.0f, 1.0f);
        char label[32];
        if (s.step_us < 1000.0f)
          snprintf(label, sizeof(label), "%.0f us", s.step_us);
        else
          snprintf(label, sizeof(label), "%.1f ms", s.step_us / 1000.0f);
        ImGui::ProgressBar(cost_norm, ImVec2(-1, 0), label);
      } else {
        ImGui::SameLine(170);
        ImGui::TextDisabled("inactive");
      }

      ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("Total: %u cells  %.1f KB", total_cells, total_mem);
  }

  // ---- Ocean parameters ----
  if (ImGui::CollapsingHeader("Ocean")) {
    bool changed = false;
    changed |= ImGui::SliderFloat("Wind (m/s)", &engine.ocean.params.wind_speed,
                                   1.0f, 30.0f);
    changed |= ImGui::SliderFloat("Choppiness", &engine.ocean.params.choppiness,
                                   0.0f, 1.5f);
    changed |= ImGui::SliderFloat("JONSWAP gamma",
                                   &engine.ocean.params.jonswap_gamma,
                                   1.0f, 7.0f);
    if (changed) {
      // Reinitialize ocean spectrum with new parameters.
      engine.ocean.Init(engine.ocean.params);
    }
  }

  // ---- Zone Radii ----
  if (ImGui::CollapsingHeader("Zone Radii")) {
    ImGui::SliderFloat("SPH (m)", &engine.lod_manager.config.sph_radius,
                        0.05f, 2.0f);
    ImGui::SliderFloat("LBM (m)", &engine.lod_manager.config.lbm_radius,
                        0.01f, 0.5f);
    ImGui::SliderFloat("Stokes (m)", &engine.lod_manager.config.stokes_radius,
                        0.001f, 0.05f);
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
  std::string mjcf = BuildMJCF();
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
  cfg.initial_surface_z = 5.0f;
  cfg.domain = {{-500, -500, -100}, {500, 500, 50}};
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
  cfg.lod.lbm_radius = 0.05f;
  cfg.lod.sph_radius = 0.3f;
  cfg.lod.swe_radius = 1000.0f;
  cfg.lod.stokes_radius = 0.005f;
  engine.Init(cfg);

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
      g_lod_active[3] = dist < 1.0f;    // LBM at < 1m
      g_lod_active[4] = dist < 0.1f;    // Stokes at < 10cm
    }

    // LOD activation from toggles (auto or manual).
    engine.lod_manager.config.sph_radius = g_lod_active[2]
        ? cfg.lod.sph_radius : -1.0f;
    engine.lod_manager.config.lbm_radius = g_lod_active[3]
        ? cfg.lod.lbm_radius : -1.0f;
    engine.lod_manager.config.stokes_radius = g_lod_active[4]
        ? cfg.lod.stokes_radius : -1.0f;
    engine.SetFocus({0, 0, 0});

    // Step physics with time budget.
    auto physics_start = std::chrono::steady_clock::now();
    while (physics_accumulator >= cfg.master_dt) {
      engine.Step();
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
    RenderDynamicGeoms(&viewer.scn, engine, cfg, m, hfield_id, viewer.cam);

    mjr_render(viewport, &viewer.scn, &viewer.con);

    // ImGui frame (rendered on top of MuJoCo).
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    DrawPanel(engine, cfg, static_cast<float>(viewer.cam.distance), viewer.cam);
    RenderWireframe(engine, m, hfield_id, viewer.cam);

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
