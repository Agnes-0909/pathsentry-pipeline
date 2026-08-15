cmake_minimum_required(VERSION 3.14)
project(ps_collector LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# RDK_resource 工程根目录（含 3rdlibrary），默认取仓库同级路径，可用 -DRDK_ROOT 覆盖
set(RDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../RDK_resource/RDK_proj" CACHE PATH
    "RDK_proj root with 3rdlibrary/")
set(RDK_CAMERA_ROOT "${RDK_ROOT}/3rdlibrary/RDK_CAMERA" CACHE PATH "RDK Camera SDK")
set(OpenCV_DIR "${RDK_ROOT}/3rdlibrary/opencv_aarch64/lib/cmake/opencv4" CACHE PATH "OpenCV aarch64")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(ps_collector
  src/main.cpp
  src/options.cpp
  src/disk_writer.cpp
  src/vio_dual.cpp
  src/sensor/sc132gs_linear_1088x1280_raw10_30fps_1lane.c
)

target_include_directories(ps_collector PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor
  ${RDK_CAMERA_ROOT}/include
  ${OpenCV_INCLUDE_DIRS})

target_link_directories(ps_collector PRIVATE ${RDK_CAMERA_ROOT}/lib)

target_link_options(ps_collector PRIVATE
  "-Wl,-rpath-link,${RDK_CAMERA_ROOT}/lib"
  "-Wl,-rpath-link,${OpenCV_LIB_PATH}")

target_link_libraries(ps_collector PRIVATE
  vpf cam hbmem pthread ${OpenCV_LIBS})
