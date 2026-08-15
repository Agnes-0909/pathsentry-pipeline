#ifndef PS_COLLECT_VIO_DUAL_HPP_
#define PS_COLLECT_VIO_DUAL_HPP_

#include "stereo_frame.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace ps::vio {

enum class StereoSyncAction {
  kPair,              // 时间戳对齐，可配对
  kRefreshLeft,       // 左目偏旧，丢弃重取
  kRefreshRight,      // 右目偏旧，丢弃重取
  kTimestampMissing,  // 时间戳缺失（异常）
};

StereoSyncAction decideStereoSync(std::uint64_t left_ts,
                                  std::uint64_t right_ts,
                                  std::uint64_t max_skew_ns) noexcept;

// 双路 MIPI camera -> VIN -> ISP -> VSE 全幅直通采集管线。
// 取帧时深拷贝 NV12 到普通内存后立即归还硬件 buffer。
class DualVioPipeline {
 public:
  DualVioPipeline(int left_host, int right_host);
  ~DualVioPipeline();

  DualVioPipeline(const DualVioPipeline&) = delete;
  DualVioPipeline& operator=(const DualVioPipeline&) = delete;

  bool initialize();
  bool acquireLeft(int timeout_ms, Nv12Image& out, std::uint64_t& frame_id,
                   std::uint64_t& timestamp_ns);
  bool acquireRight(int timeout_ms, Nv12Image& out, std::uint64_t& frame_id,
                    std::uint64_t& timestamp_ns);
  void stop() noexcept;
  const std::string& lastError() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ps::vio

#endif  // PS_COLLECT_VIO_DUAL_HPP_
