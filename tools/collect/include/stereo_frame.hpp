#ifndef PS_COLLECT_STEREO_FRAME_HPP_
#define PS_COLLECT_STEREO_FRAME_HPP_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ps {

// 深拷贝到普通内存的一帧 NV12 图像（硬件 buffer 已归还，可安全跨线程使用）
struct Nv12Image {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t stride = 0;  // Y 与 UV plane 共用同一 stride（VSE 输出特性）
  std::vector<std::uint8_t> data;  // Y plane + UV plane 连续存放

  std::size_t ySize() const {
    return static_cast<std::size_t>(stride) * height;
  }
  std::size_t uvSize() const {
    return static_cast<std::size_t>(stride) * ((height + 1) / 2);
  }
};

// 一对完成硬件时间戳配对的左右目帧
struct StereoFrame {
  Nv12Image left;
  Nv12Image right;
  std::uint64_t left_frame_id = 0;
  std::uint64_t right_frame_id = 0;
  std::uint64_t left_timestamp_ns = 0;   // 硬件时间戳
  std::uint64_t right_timestamp_ns = 0;
  std::uint64_t skew_ns = 0;             // 左右时间戳差
};

inline void copyNv12(Nv12Image& dst, std::uint32_t width, std::uint32_t height,
                     std::uint32_t stride, const void* y, const void* uv) {
  dst.width = width;
  dst.height = height;
  dst.stride = stride;
  dst.data.resize(static_cast<std::size_t>(stride) * height * 3 / 2);
  std::memcpy(dst.data.data(), y, dst.ySize());
  std::memcpy(dst.data.data() + dst.ySize(), uv, dst.uvSize());
}

}  // namespace ps

#endif  // PS_COLLECT_STEREO_FRAME_HPP_
