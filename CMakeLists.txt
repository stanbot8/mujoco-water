# SPDX-License-Identifier: MIT
# Copyright (c) 2026 stanbot8
cmake_minimum_required(VERSION 3.20)
project(mujoco-water LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Header-only interface library ---
add_library(mujoco-water INTERFACE)
target_include_directories(mujoco-water INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# --- Install (header-only) ---
include(GNUInstallDirs)

install(TARGETS mujoco-water EXPORT mujoco-water-targets)
install(DIRECTORY include/mjwater
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT mujoco-water-targets
        FILE mujoco-water-config.cmake
        NAMESPACE mjwater::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mujoco-water)
