#pragma once
#include "deploy_types.hpp"
#include <cstdint>
#include <string>
namespace deploy {
constexpr std::uint32_t kFrameMagic = 0x44504631; // DPF1
constexpr std::uint32_t kResultMagic = 0x44505231; // DPR1
struct FrameHeader {
  std::uint32_t magic=kFrameMagic, version=1, fd_count=0, reserved=0;
  std::uint64_t frame_id=0, timestamp=0;
  std::uint32_t width=0, height=0, stride=0, y_bytes=0, uv_bytes=0, plane_count=0, format=0; std::int32_t y_share_id=0, uv_share_id=0; std::uint64_t y_phy=0, uv_phy=0;
};
struct ResultHeader {
  std::uint32_t magic=kResultMagic, version=1, detection_count=0, mask_bytes=kMaskW*kMaskH;
  std::uint64_t frame_id=0, timestamp=0;
  float import_ms=0, bpu_ms=0, infer_ms=0, post_ms=0;
};
bool sendFrame(int sock, const Frame& frame);
bool recvFrame(int sock, FrameHeader& header, int fds[2]);
bool sendResult(int sock, const Result& result);
bool recvResult(int sock, Result& result);
bool sendAll(int fd, const void* data, std::size_t bytes);
bool recvAll(int fd, void* data, std::size_t bytes);
int makeServer(const std::string& path);
int connectUnix(const std::string& path);
}
