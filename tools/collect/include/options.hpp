#ifndef PS_COLLECT_OPTIONS_HPP_
#define PS_COLLECT_OPTIONS_HPP_

#include <cstdint>
#include <string>

namespace ps {

struct Options {
  std::string output_root = "data/raw";
  std::string scene = "untitled";   // 场景标签：grass/indoor/backlight/...
  std::string format = "jpg";       // jpg | png
  int jpg_quality = 92;
  int left_host = 0;
  int right_host = 1;
  int frame_timeout_ms = 1000;
  int max_consecutive_timeouts = 30;
  std::uint64_t max_pairs = 0;      // 0 = 不限，直到 Ctrl-C
  std::uint64_t duration_sec = 0;   // 0 = 不限
  std::uint32_t stride = 1;         // 每 stride 对保存一对（30fps 原始流抽帧）
  std::uint64_t max_pair_skew_ns = 1'000'000ULL;  // 左右硬同步容差
};

// 解析命令行；--help 抛出 HelpRequested
struct HelpRequested {};
Options parseOptions(int argc, char** argv);
std::string usage(const char* program);

}  // namespace ps

#endif  // PS_COLLECT_OPTIONS_HPP_
