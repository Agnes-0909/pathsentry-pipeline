#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD_DIR:-$ROOT/build_aarch64}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-$ROOT/../3rdlibrary}
if ! command -v cmake >/dev/null 2>&1; then
  IMAGE=${BUILD_IMAGE:-openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8}
  exec docker run --rm --platform linux/amd64 -v "$ROOT":/workspace/06_deploy -v "$THIRD_PARTY_ROOT":/workspace/3rdlibrary:ro -w /workspace/06_deploy "$IMAGE" bash -lc \
    'export PATH=/cmake-3.14.5-Linux-x86_64/bin:/opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin:$PATH; CXX=aarch64-none-linux-gnu-g++ CC=aarch64-none-linux-gnu-gcc ./build.sh'
fi
: "${CXX:=aarch64-linux-gnu-g++}"
: "${CC:=aarch64-linux-gnu-gcc}"
rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -DTHIRD_PARTY_ROOT="$THIRD_PARTY_ROOT" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$BUILD" -j"${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
