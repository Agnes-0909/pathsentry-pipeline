cmake_minimum_required(VERSION 3.14)
project(ps_collector LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 三方库随仓库自带（独立工程）：默认取仓库根的 3rdlibrary，可用 -DRDK_ROOT 覆盖
set(RDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
    "Directory containing 3rdlibrary/")
set(RDK_CAMERA_ROOT "${RDK_ROOT}/3rdlibrary/RDK_CAMERA" CACHE PATH "RDK Camera SDK")
set(OpenCV_DIR "${RDK_ROOT}/3rdlibrary/opencv_aarch64/lib/cmake/opencv4" CACHE PATH "OpenCV aarch64")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ---------------------------------------------------------------------------
# 模块划分（自底向上）：
#   core     : 数据结构与工具（无外部依赖）
#   capture  : 双目 VIO 采集管线（依赖 RDK_CAMERA SDK）
#   storage  : 落盘编码（依赖 OpenCV）
#   app      : 可执行入口（组装以上模块）
# ---------------------------------------------------------------------------
add_library(ps_collect_core INTERFACE)
target_include_directories(ps_collect_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(ps_collect_capture STATIC
  src/capture/vio_dual.cpp
  src/capture/sensor/sc132gs_linear_1088x1280_raw10_30fps_1lane.c)
target_include_directories(ps_collect_capture PUBLIC ${RDK_CAMERA_ROOT}/include)
target_include_directories(ps_collect_capture PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/capture/sensor)
target_link_directories(ps_collect_capture PUBLIC ${RDK_CAMERA_ROOT}/lib)
target_link_options(ps_collect_capture PRIVATE
  "-Wl,-rpath-link,${RDK_CAMERA_ROOT}/lib")
target_link_libraries(ps_collect_capture PUBLIC ps_collect_core vpf cam hbmem pthread)

add_library(ps_collect_storage STATIC src/storage/disk_writer.cpp)
target_link_libraries(ps_collect_storage PUBLIC ps_collect_core pthread ${OpenCV_LIBS})

add_library(ps_collect_cli STATIC src/cli/options.cpp)
target_link_libraries(ps_collect_cli PUBLIC ps_collect_core)

add_executable(ps_collector src/app/main.cpp)
target_link_libraries(ps_collector PRIVATE
  ps_collect_core ps_collect_capture ps_collect_storage ps_collect_cli)
