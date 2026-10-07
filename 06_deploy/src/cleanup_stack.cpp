#include "vio_pipeline.hpp"

#include <utility>

namespace deploy {

HardwareSyncTiming sc132gsHardwareSyncTiming() noexcept {
  return {30, 33333, 10, 100, 0, 0};
}

bool shouldSetMclkAttribute(bool device_tree_has_pinctrl_names) noexcept {
  return device_tree_has_pinctrl_names;
}

SensorRouting makeSensorRouting(int camera_phy, int vcon_rx_phy,
                                int lpwm_channel) noexcept {
  return SensorRouting{camera_phy, vcon_rx_phy, lpwm_channel + 1};
}

std::array<int, 3> sensorPowerSequence(bool active_high) noexcept {
  const int active = active_high ? 1 : 0;
  return {active, 1 - active, active};
}

std::optional<int> detectSensorAddress(
    const std::vector<int>& candidates,
    const std::function<bool(int)>& probe) {
  if (!probe) return std::nullopt;
  for (const int candidate : candidates) {
    if (candidate > 0 && probe(candidate)) return candidate;
  }
  return std::nullopt;
}

CleanupStack::~CleanupStack() { run(); }

void CleanupStack::push(Cleanup cleanup) {
  if (cleanup) cleanups_.push_back(std::move(cleanup));
}

void CleanupStack::run() noexcept {
  while (!cleanups_.empty()) {
    Cleanup cleanup = std::move(cleanups_.back());
    cleanups_.pop_back();
    try {
      cleanup();
    } catch (...) {
    }
  }
}

}  // namespace deploy
