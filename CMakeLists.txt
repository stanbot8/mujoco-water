# SPDX-License-Identifier: MIT
# Copyright (c) 2026 stanbot8
cmake_minimum_required(VERSION 3.20)
project(mujoco-water LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MJWATER_TESTS "Build tests (requires mujoco-game)" OFF)
option(MJWATER_TESTS_ONLY "Build tests without MuJoCo" OFF)
option(MJWATER_DEMO "Build dam break demo (requires MuJoCo + GLFW)" OFF)

# --- Header-only interface library ---
add_library(mujoco-water INTERFACE)
target_include_directories(mujoco-water INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# --- MuJoCo dependency (via mujoco-game) ---
if(NOT MJWATER_TESTS_ONLY)
  if(NOT TARGET mujoco-game)
    # Prefer local sibling directory if available, otherwise fetch from GitHub.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../mujoco-game/CMakeLists.txt)
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../mujoco-game
                       ${CMAKE_BINARY_DIR}/mujoco-game)
    else()
      include(FetchContent)
      FetchContent_Declare(mujoco-game
          GIT_REPOSITORY https://github.com/stanbot8/mujoco-game.git
          GIT_TAG main
      )
      FetchContent_MakeAvailable(mujoco-game)
    endif()
  endif()
  target_link_libraries(mujoco-water INTERFACE mujoco-game)
endif()

# --- Tests ---
if(MJWATER_TESTS OR MJWATER_TESTS_ONLY)
  enable_testing()

  # Test sources (all independent of MuJoCo)
  set(TEST_SOURCES
    tests/test_shallow_water.cc
    tests/test_sph.cc
    tests/test_lbm.cc
    tests/test_lod.cc
    tests/test_coupling.cc
    tests/test_ocean.cc
    tests/test_stokes.cc
  )

  foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    set(target_name "mw_${test_name}")
    add_executable(${target_name} ${test_src})
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    if(MSVC)
      target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
      target_compile_options(${target_name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${target_name} COMMAND ${target_name})
  endforeach()
endif()

# --- Demo (requires MuJoCo + GLFW) ---
if(MJWATER_DEMO AND NOT MJWATER_TESTS_ONLY)
  add_executable(water_demo demo/water_demo.cc)
  target_include_directories(water_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(water_demo PRIVATE mujoco-game)
  target_compile_features(water_demo PRIVATE cxx_std_20)
  target_compile_definitions(water_demo PRIVATE MJGAME_VIEWER=1)
  if(MSVC)
    target_compile_options(water_demo PRIVATE /W4 /permissive-)
  else()
    target_compile_options(water_demo PRIVATE -Wall -Wextra)
  endif()

  # Copy MuJoCo DLL next to executable.
  if(WIN32 AND MUJOCO_DLL)
    add_custom_command(TARGET water_demo POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MUJOCO_DLL}" $<TARGET_FILE_DIR:water_demo>)
  endif()

  # --- Dear ImGui (demo-only dependency) ---
  include(FetchContent)
  FetchContent_Declare(imgui
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG v1.91.8
  )
  FetchContent_MakeAvailable(imgui)

  add_library(imgui_demo STATIC
      ${imgui_SOURCE_DIR}/imgui.cpp
      ${imgui_SOURCE_DIR}/imgui_draw.cpp
      ${imgui_SOURCE_DIR}/imgui_widgets.cpp
      ${imgui_SOURCE_DIR}/imgui_tables.cpp
      ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
      ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(imgui_demo PUBLIC
      ${imgui_SOURCE_DIR}
      ${imgui_SOURCE_DIR}/backends
  )
  target_link_libraries(imgui_demo PUBLIC glfw)
  if(WIN32)
    target_link_libraries(imgui_demo PUBLIC opengl32)
  else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(imgui_demo PUBLIC OpenGL::GL)
  endif()
  # Suppress warnings in third-party code.
  if(MSVC)
    target_compile_options(imgui_demo PRIVATE /W0)
  else()
    target_compile_options(imgui_demo PRIVATE -w)
  endif()

  add_executable(multiscale_demo demo/multiscale_demo.cc)
  target_include_directories(multiscale_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(multiscale_demo PRIVATE mujoco-game imgui_demo)
  target_compile_features(multiscale_demo PRIVATE cxx_std_20)
  target_compile_definitions(multiscale_demo PRIVATE MJGAME_VIEWER=1)
  if(MSVC)
    target_compile_options(multiscale_demo PRIVATE /W4 /permissive-)
  else()
    target_compile_options(multiscale_demo PRIVATE -Wall -Wextra)
  endif()

  if(WIN32 AND MUJOCO_DLL)
    add_custom_command(TARGET multiscale_demo POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MUJOCO_DLL}" $<TARGET_FILE_DIR:multiscale_demo>)
  endif()
endif()

# --- Install (header-only, standalone mode only) ---
if(MJWATER_TESTS_ONLY OR NOT TARGET mujoco-game)
  include(GNUInstallDirs)

  install(TARGETS mujoco-water EXPORT mujoco-water-targets)
  install(DIRECTORY include/mjwater
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT mujoco-water-targets
          FILE mujoco-water-config.cmake
          NAMESPACE mjwater::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mujoco-water)
endif()
