#!/usr/bin/env bash
# mujoco-water build script
# Usage:
#   ./build.sh              Build tests only (no MuJoCo required)
#   ./build.sh test         Build and run tests
#   ./build.sh demo         Build demo (requires MuJoCo + mujoco-game)
#   ./build.sh clean        Remove build directory
#   ./build.sh install      Install headers to prefix (default: /usr/local)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${2:-Release}"
CMAKE="cmake"

# Find cmake on Windows if not in PATH.
if ! command -v cmake &>/dev/null; then
  for candidate in \
    "/c/Program Files/CMake/bin/cmake.exe" \
    "C:/Program Files/CMake/bin/cmake.exe" \
    "/c/Program Files (x86)/CMake/bin/cmake.exe"; do
    if [[ -x "$candidate" ]]; then
      CMAKE="$candidate"
      break
    fi
  done
fi

cmd="${1:-build}"

case "$cmd" in
  build)
    echo "=== Configuring (tests only, no MuJoCo) ==="
    "$CMAKE" -B "$BUILD_DIR" -DMJWATER_TESTS_ONLY=ON -S "$SCRIPT_DIR"
    echo "=== Building ($BUILD_TYPE) ==="
    "$CMAKE" --build "$BUILD_DIR" --config "$BUILD_TYPE"
    echo "=== Done. Run './build.sh test' to execute tests. ==="
    ;;

  test)
    if [[ ! -d "$BUILD_DIR" ]]; then
      bash "$SCRIPT_DIR/build.sh" build
    fi
    echo "=== Building ($BUILD_TYPE) ==="
    "$CMAKE" --build "$BUILD_DIR" --config "$BUILD_TYPE"
    echo "=== Running tests ==="
    "$CMAKE" --build "$BUILD_DIR" --config "$BUILD_TYPE" --target RUN_TESTS 2>&1 || \
      (cd "$BUILD_DIR" && "$CMAKE" -E env ctest -C "$BUILD_TYPE" --output-on-failure)
    ;;

  demo)
    echo "=== Configuring (demo + tests, requires MuJoCo) ==="
    "$CMAKE" -B "$BUILD_DIR" -DMJWATER_DEMO=ON -DMJWATER_TESTS=ON -S "$SCRIPT_DIR"
    echo "=== Building ($BUILD_TYPE) ==="
    "$CMAKE" --build "$BUILD_DIR" --config "$BUILD_TYPE"
    echo "=== Run: $BUILD_DIR/$BUILD_TYPE/water_demo ==="
    ;;

  clean)
    echo "=== Removing $BUILD_DIR ==="
    rm -rf "$BUILD_DIR"
    echo "=== Clean. ==="
    ;;

  install)
    PREFIX="${3:-/usr/local}"
    echo "=== Installing headers to $PREFIX/include/mjwater ==="
    mkdir -p "$PREFIX/include/mjwater"
    cp "$SCRIPT_DIR"/include/mjwater/*.h "$PREFIX/include/mjwater/"
    echo "=== Installed. Add $PREFIX/include to your include path. ==="
    ;;

  *)
    echo "Usage: $0 {build|test|demo|clean|install} [Release|Debug] [install-prefix]"
    exit 1
    ;;
esac
