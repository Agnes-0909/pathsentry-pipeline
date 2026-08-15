#!/usr/bin/env bash
# 交叉编译 ps_collector（宿主机需安装 aarch64-linux-gnu-gcc/g++）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RDK_ROOT="${RDK_ROOT:-${REPO_ROOT}}"   # 三方库随仓库自带：REPO_ROOT/3rdlibrary
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_aarch64}"
TOOLCHAIN_FILE="${BUILD_DIR}/aarch64-linux-gnu.toolchain.cmake"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

mkdir -p "${BUILD_DIR}"

cat >"${TOOLCHAIN_FILE}" <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_AR aarch64-linux-gnu-ar)
set(CMAKE_RANLIB aarch64-linux-gnu-ranlib)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
EOF

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DRDK_ROOT="${RDK_ROOT}"

cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS}"

echo "产物: ${BUILD_DIR}/ps_collector"
echo "部署: scp ${BUILD_DIR}/ps_collector root@<板子IP>:/root/ps_collector"
