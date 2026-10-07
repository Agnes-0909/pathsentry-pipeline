#pragma once
#include "vse_frame.hpp"
#include "async_logger.hpp"
#include <memory>
#include <string>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>
namespace deploy {
struct HardwareSyncTiming { int fps=0; std::uint32_t period_us=0, offset_us=0, duty_us=0, trigger_source=0, trigger_mode=0; };
struct SensorRouting { int camera_phy=0, vin_rx=0, timestamp_source=0; };
HardwareSyncTiming sc132gsHardwareSyncTiming() noexcept;
bool shouldSetMclkAttribute(bool) noexcept;
SensorRouting makeSensorRouting(int,int,int) noexcept;
std::array<int,3> sensorPowerSequence(bool) noexcept;
std::optional<int> detectSensorAddress(const std::vector<int>&, const std::function<bool(int)>&);
class CleanupStack { public: using Cleanup=std::function<void()>; ~CleanupStack(); void push(Cleanup); void run() noexcept; private: std::vector<Cleanup> cleanups_; };
class LeftVioPipeline {
 public:
  explicit LeftVioPipeline(int host, AsyncLogger* logger=nullptr);
  ~LeftVioPipeline();
  LeftVioPipeline(const LeftVioPipeline&)=delete;
  LeftVioPipeline& operator=(const LeftVioPipeline&)=delete;
  bool initialize(); bool acquire(int timeout_ms, VseFrame& frame); void stop() noexcept;
  const std::string& lastError() const;
 private: class Impl; std::unique_ptr<Impl> impl_;
};
}
