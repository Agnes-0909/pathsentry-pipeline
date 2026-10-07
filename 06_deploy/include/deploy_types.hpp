#pragma once
#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>
#include <array>
#include <utility>

namespace deploy {

enum class PixelFormat : std::uint32_t { kUnknown = 0, kNv12 = 1 };
struct BufferLayout {
  int width = 0, height = 0, stride = 0, stride_y = 0, stride_uv = 0, plane_count = 0;
  std::uint32_t y_bytes = 0, uv_bytes = 0;
  std::uint64_t y_phy = 0, uv_phy = 0;
  void* y_virt = nullptr; void* uv_virt = nullptr;
  int y_fd = -1, uv_fd = -1; int y_share_id = 0, uv_share_id = 0;
  PixelFormat format = PixelFormat::kUnknown;
};
struct Frame {
  std::uint64_t id = 0, timestamp = 0;
  BufferLayout layout;
  std::function<int()> release_fn;
  bool valid = false;
  Frame() = default;
  Frame(std::uint64_t fid, BufferLayout l, std::function<int()> fn, std::uint64_t ts=0) : id(fid), timestamp(ts), layout(l), release_fn(std::move(fn)), valid(static_cast<bool>(release_fn)) {}
  bool validFrame() const noexcept { return valid; }
  std::uint64_t frameId() const noexcept { return id; }
  std::uint64_t hardwareTimestamp() const noexcept { return timestamp; }
  const BufferLayout& getLayout() const noexcept { return layout; }
  Frame(const Frame&) = delete;
  Frame& operator=(const Frame&) = delete;
  Frame(Frame&& other) noexcept { *this = std::move(other); }
  Frame& operator=(Frame&& other) noexcept {
    if (this != &other) { release(); id=other.id; timestamp=other.timestamp; layout=other.layout; release_fn=std::move(other.release_fn); valid=other.valid; other.valid=false; }
    return *this;
  }
  ~Frame() { release(); }
  void release() noexcept { if (valid) { valid=false; if (release_fn) (void)release_fn(); } }
};
struct Detection { float x1=0, y1=0, x2=0, y2=0, score=0; std::int32_t cls=0; };
constexpr std::uint32_t kMaskW = 240, kMaskH = 208;
struct Result { std::uint64_t id=0, timestamp=0; std::vector<Detection> detections; std::array<std::uint8_t,kMaskW*kMaskH> mask{}; float import_ms=0, bpu_ms=0, infer_ms=0, post_ms=0; float submit_ms=0, wait_ms=0, task_release_ms=0, cache_ms=0; };
}
