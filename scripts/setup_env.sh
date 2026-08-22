#!/usr/bin/env bash
set -e

exec > >(stdbuf -oL -eL cat) 2>&1

echo "=== STAGE: apt build dependencies ==="
sudo add-apt-repository -y ppa:kicad/kicad-9.0-releases || true

# Enable deb-src for Ubuntu DEB822 (Ubuntu 24.04+) and traditional sources.list
sudo sed -i 's/Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
sudo sed -i -E 's/^#\s*(deb-src .*)/\1/' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true

# Mirror deb to deb-src as fallback if deb-src still missing
if ! grep -rq '^deb-src ' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
  if [ -f /etc/apt/sources.list ]; then
    grep '^deb ' /etc/apt/sources.list | sed 's/^deb /deb-src /' | sudo tee -a /etc/apt/sources.list >/dev/null || true
  fi
fi

sudo apt-get update -qq || true
sudo apt-get build-dep -y kicad || true
sudo apt-get install -y kicad cmake ninja-build build-essential libwxgtk3.2-dev libboost-all-dev libglm-dev libglew-dev libcurl4-openssl-dev libssl-dev libgl1-mesa-dev libglu1-mesa-dev libglvnd-dev mesa-common-dev python3-dev
pip install -q pybind11

echo "=== STAGE: restore cached KiCad build from Drive (if available) ==="
cd "$WORKDIR"
RESTORED_FROM_CACHE=0
if [ ! -d kicad-src ] && [ -f "$DRIVE_CACHE_TARBALL" ]; then
  echo "found $DRIVE_CACHE_TARBALL, restoring (skips clone + most of configure/build)"
  tar xzf "$DRIVE_CACHE_TARBALL"
  RESTORED_FROM_CACHE=1
else
  echo "no cache to restore (first run, or already have a local kicad-src)"
fi

echo "=== STAGE: clone KiCad source (tag $KICAD_TAG) ==="
if [ ! -d kicad-src ]; then
  git clone --depth 1 --branch "$KICAD_TAG" https://gitlab.com/kicad/code/kicad.git kicad-src
fi

echo "=== STAGE: wire pcbworld bridge into KiCad's CMake tree ==="
rm -rf kicad-src/pcbworld_bridge
cp -r Routerv3/pcbworld/engine/cpp kicad-src/pcbworld_bridge
MARKER='add_subdirectory( ${CMAKE_SOURCE_DIR}/pcbworld_bridge'
if ! grep -qF "$MARKER" kicad-src/pcbnew/CMakeLists.txt; then
  printf '\nadd_subdirectory( ${CMAKE_SOURCE_DIR}/pcbworld_bridge ${CMAKE_BINARY_DIR}/pcbworld_bridge )\n' >> kicad-src/pcbnew/CMakeLists.txt
fi

echo "=== STAGE: cmake configure (Release) ==="
cd "$WORKDIR/kicad-src"
PYBIND11_CMAKE_DIR="$(python3 -m pybind11 --cmakedir)"
echo "pybind11 cmake dir: $PYBIND11_CMAKE_DIR"

# A restored Drive cache carries the PAST session's build/CMakeCache.txt,
# whose recorded toolchain paths (notably build.ninja's own "regenerate"
# rule, which shells out to whatever CMAKE_COMMAND path was resolved at
# generation time) are only valid if Colab's underlying base image hasn't
# changed since. It does change -- hit for real: a cache built under a
# Python 3.12 Colab image recorded a pip-installed cmake at
# /usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake; a later
# session's image had moved to Python 3.13, that exact path no longer
# existed, and ninja's build.ninja regeneration failed outright before
# compiling anything. Mere existence of CMakeCache.txt (the ORIGINAL
# check) can't distinguish that from the legitimate same-session-retry
# case (re-running this script after an earlier failure, same runtime,
# same toolchain, safe to skip reconfigure) -- RESTORED_FROM_CACHE does,
# since it's only set true when THIS run pulled a tarball from Drive, a
# genuinely different environment than whichever session produced it.
if [ ! -f build/CMakeCache.txt ] || [ "$RESTORED_FROM_CACHE" = "1" ]; then
  if [ "$RESTORED_FROM_CACHE" = "1" ] && [ -f build/CMakeCache.txt ]; then
    echo "cache was restored from Drive this run -- its recorded toolchain paths may be stale in this runtime (already happened once: a Python-version change in Colab's base image broke a cached cmake path). Forcing a reconfigure rather than trusting it."
  fi
  cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DKICAD_BUILD_QA_TESTS=OFF \
    -DKICAD_SCRIPTING_WXPYTHON=OFF \
    -DKICAD_BUILD_I18N=OFF \
    -DKICAD_USE_CMAKE_FINDPROTOBUF=ON \
    -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"
else
  echo "build/CMakeCache.txt already exists from this same session -- skipping configure (rm -rf build to force a clean reconfigure)"
fi

echo "=== STAGE: build pcbworld_pns_bridge ==="
cmake --build build --target pcbworld_pns_bridge -j"$(nproc)"

echo "=== STAGE: save build to Drive cache ==="
cd "$WORKDIR"
tar czf "$DRIVE_CACHE_TARBALL.tmp" kicad-src
mv "$DRIVE_CACHE_TARBALL.tmp" "$DRIVE_CACHE_TARBALL"
echo "cached to $DRIVE_CACHE_TARBALL ($(du -h "$DRIVE_CACHE_TARBALL" | cut -f1))"

echo "=== DONE: bridge built successfully ==="
find kicad-src/build -iname 'pcbworld_pns_bridge*.so'
