#pragma once
#include "deploy_types.hpp"
#include <memory>
#include <string>
namespace deploy {
class RawHeadRunner {
 public:
  struct Config { std::string model; float score_threshold=.25f, nms_threshold=.45f; int bpu_core=1; };
  explicit RawHeadRunner(Config cfg); ~RawHeadRunner();
  bool initialize();
  bool inferFromFd(int fd, std::uint32_t expected_size, std::int32_t share_id, std::uint64_t y_phy, std::uint64_t uv_phy, Result& result);
  const std::string& lastError() const;
 private: class Impl; std::unique_ptr<Impl> impl_;
};
}
