// SPDX-License-Identifier: MIT
// Copyright (c) 2026 stanbot8
#pragma once
// Shared test infrastructure for mujoco-water unit tests.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

#define CHECK(expr) \
  do { if (!(expr)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
    ": CHECK failed: " #expr); } while(0)

#define CHECK_NEAR(a, b, tol) \
  do { if (std::abs((a) - (b)) > (tol)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
    ": CHECK_NEAR failed: " + std::to_string(a) + " vs " + std::to_string(b) + \
    " (tol=" + std::to_string(tol) + ")"); } while(0)

struct TestEntry { const char* name; void (*fn)(); };
inline std::vector<TestEntry>& GetTests() {
  static std::vector<TestEntry> tests;
  return tests;
}

#define TEST(name) \
  static void test_##name(); \
  struct Register_##name { \
    Register_##name() { GetTests().push_back({#name, test_##name}); } \
  } reg_##name; \
  static void test_##name()

// Define test but don't register it (skipped unless called explicitly).
#define TEST_DISABLED(name) \
  static void test_disabled_##name()

inline int RunAllTests() {
  int passed = 0, failed = 0;
  for (auto& t : GetTests()) {
    try {
      t.fn();
      printf("  PASS  %s\n", t.name);
      passed++;
    } catch (const std::exception& e) {
      printf("  FAIL  %s  (%s)\n", t.name, e.what());
      failed++;
    } catch (...) {
      printf("  FAIL  %s  (unknown exception)\n", t.name);
      failed++;
    }
  }
  printf("\n%d passed, %d failed\n", passed, failed);
  return failed > 0 ? 1 : 0;
}
