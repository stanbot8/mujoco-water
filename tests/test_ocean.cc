// Tests for spectral ocean solver.
#include "test_harness.h"
#include "mjwater/spectral_ocean.h"
#include <cmath>

using namespace mjwater;

// --- Phillips spectrum tests ---

TEST(PhillipsSpectrumBasic) {
  float k = 0.1f;  // 63m wavelength
  float P = SpectralOceanSolver::PhillipsSpectrum(k, 10.0f, kGravity, 0.001f);
  CHECK(P > 0);

  float P_high = SpectralOceanSolver::PhillipsSpectrum(100.0f, 10.0f, kGravity, 0.01f);
  CHECK(P_high < P);
}

TEST(PhillipsSpectrumZeroK) {
  float P = SpectralOceanSolver::PhillipsSpectrum(0.0f, 10.0f, kGravity, 0.001f);
  CHECK(P == 0.0f);
}

// --- Dispersion relation tests ---

TEST(DispersionDeepWater) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.depth = 1000.0f;
  ocean.Init(p);

  float k = 0.1f;
  float omega = ocean.Dispersion(k);
  float expected = std::sqrt(kGravity * k);
  CHECK(std::abs(omega - expected) / expected < 0.01f);
}

TEST(DispersionShallowWater) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.depth = 0.5f;
  ocean.Init(p);

  float k = 0.01f;
  float omega = ocean.Dispersion(k);
  float expected = k * std::sqrt(kGravity * p.depth);
  CHECK(std::abs(omega - expected) / expected < 0.05f);
}

// --- Gerstner surface tests ---

TEST(SurfaceHeightQuiescent) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 0.0f;
  p.num_waves = 16;
  ocean.Init(p);

  float h = ocean.SurfaceHeight(50.0f, 50.0f);
  CHECK(std::abs(h) < 1e-6f);
}

TEST(SurfaceHeightNonzeroWind) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 10.0f;
  p.num_waves = 32;
  ocean.Init(p);

  float max_h = 0;
  for (float x = 0; x < 100.0f; x += 1.0f) {
    float h = std::abs(ocean.SurfaceHeight(x, 0.0f));
    max_h = std::max(max_h, h);
  }
  CHECK(max_h > 0.01f);
}

TEST(SignificantWaveHeightBeaufort) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 10.0f;
  p.num_waves = 64;
  ocean.Init(p);

  float Hs = ocean.SignificantWaveHeight();
  CHECK(Hs > 0.01f);
  CHECK(Hs < 20.0f);
}

// --- Orbital velocity tests ---

TEST(OrbitalVelocityDecaysWithDepth) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 10.0f;
  p.num_waves = 32;
  ocean.Init(p);

  Vec3 v_surface = ocean.OrbitalVelocity(10.0f, 0.0f, 0.0f);
  Vec3 v_deep = ocean.OrbitalVelocity(10.0f, 0.0f, -50.0f);

  float speed_surface = v_surface.Length();
  float speed_deep = v_deep.Length();

  CHECK(speed_surface > speed_deep);
  CHECK(speed_deep < speed_surface * 0.5f);
}

TEST(OrbitalVelocityCircularMotion) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 10.0f;
  p.num_waves = 1;
  p.min_wavelength = 50.0f;
  p.max_wavelength = 50.0f;
  ocean.Init(p);

  float sum_h = 0, sum_v = 0;
  int n = 100;
  for (int i = 0; i < n; ++i) {
    float x = static_cast<float>(i) * 50.0f / n;
    Vec3 v = ocean.OrbitalVelocity(x, 0.0f, 0.0f);
    float hor = std::sqrt(v.x * v.x + v.y * v.y);
    sum_h += hor * hor;
    sum_v += v.z * v.z;
  }
  float rms_h = std::sqrt(sum_h / n);
  float rms_v = std::sqrt(sum_v / n);

  if (rms_h > 1e-6f) {
    float ratio = rms_v / rms_h;
    CHECK(ratio > 0.5f && ratio < 2.0f);
  }
}

// --- Energy ---

TEST(EnergyPositive) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 15.0f;
  p.num_waves = 64;
  ocean.Init(p);

  CHECK(ocean.TotalEnergy() > 0);
}

// --- Time stepping ---

TEST(TimeStepping) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 10.0f;
  p.num_waves = 16;
  ocean.Init(p);

  float h0 = ocean.SurfaceHeight(10.0f, 0.0f);
  ocean.Step(1.0f);
  float h1 = ocean.SurfaceHeight(10.0f, 0.0f);

  CHECK(std::abs(h1 - h0) > 1e-8f);
}

// --- JONSWAP factor ---

TEST(JONSWAPFactorAtPeak) {
  float f = SpectralOceanSolver::JONSWAPFactor(1.0f, 1.0f, 3.3f);
  CHECK(std::abs(f - 3.3f) < 0.01f);
}

TEST(JONSWAPFactorFarFromPeak) {
  float f = SpectralOceanSolver::JONSWAPFactor(5.0f, 1.0f, 3.3f);
  CHECK(std::abs(f - 1.0f) < 0.01f);
}

TEST(JONSWAPDisabled) {
  float f = SpectralOceanSolver::JONSWAPFactor(1.0f, 1.0f, 1.0f);
  CHECK(std::abs(f - 1.0f) < 0.01f);
}

// --- Displacement tests ---

TEST(GerstnerDisplacementHorizontal) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 10.0f;
  p.num_waves = 32;
  p.choppiness = 1.0f;
  ocean.Init(p);

  Vec3 d = ocean.SurfaceDisplacement(10.0f, 0.0f);
  CHECK(std::abs(d.x) + std::abs(d.y) > 0);
}

TEST(GerstnerNoChoppiness) {
  SpectralOceanSolver ocean;
  OceanParams p;
  p.wind_speed = 10.0f;
  p.num_waves = 32;
  p.choppiness = 0.0f;
  ocean.Init(p);

  Vec3 d = ocean.SurfaceDisplacement(10.0f, 0.0f);
  CHECK(std::abs(d.x) < 1e-10f);
  CHECK(std::abs(d.y) < 1e-10f);
  CHECK(std::abs(d.z) > 0.0f);
}

int main() { return RunAllTests(); }
