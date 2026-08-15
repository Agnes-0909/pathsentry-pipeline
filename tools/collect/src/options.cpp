#include "options.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ps {

namespace {

std::uint64_t parseUint64(const char* value, const std::string& option) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == nullptr || *end != '\0') {
    throw std::invalid_argument(std::string("invalid integer for ") + option +
                                ": " + value);
  }
  return static_cast<std::uint64_t>(parsed);
}

}  // namespace

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + argument);
      }
      return argv[++i];
    };
    if (argument == "-h" || argument == "--help") {
      throw HelpRequested{};
    }
    if (argument == "-o" || argument == "--output") {
      options.output_root = value();
    } else if (argument == "--scene") {
      options.scene = value();
    } else if (argument == "--format") {
      options.format = value();
      if (options.format != "jpg" && options.format != "png") {
        throw std::invalid_argument("--format must be jpg or png");
      }
    } else if (argument == "--jpg-quality") {
      options.jpg_quality = static_cast<int>(parseUint64(value(), argument));
    } else if (argument == "--left-host") {
      options.left_host = static_cast<int>(parseUint64(value(), argument));
    } else if (argument == "--right-host") {
      options.right_host = static_cast<int>(parseUint64(value(), argument));
    } else if (argument == "--timeout-ms") {
      options.frame_timeout_ms = static_cast<int>(parseUint64(value(), argument));
    } else if (argument == "--max-pairs") {
      options.max_pairs = parseUint64(value(), argument);
    } else if (argument == "--duration-sec") {
      options.duration_sec = parseUint64(value(), argument);
    } else if (argument == "--stride") {
      options.stride = static_cast<std::uint32_t>(parseUint64(value(), argument));
      if (options.stride == 0) {
        throw std::invalid_argument("--stride must be >= 1");
      }
    } else if (argument == "--max-skew-ns") {
      options.max_pair_skew_ns = parseUint64(value(), argument);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  return options;
}

std::string usage(const char* program) {
  return std::string("用法: ") + program + " [选项]\n" +
         "PathSentry 双目数据采集工具（X5 板端运行）\n" +
         "  -o, --output DIR      输出根目录（默认 data/raw）\n"
         "  --scene TAG           场景标签，如 grass/indoor/backlight（默认 untitled）\n"
         "  --format FMT          jpg | png（默认 jpg）\n"
         "  --jpg-quality N       JPEG 质量 1-100（默认 92）\n"
         "  --left-host N         左目 MIPI host（默认 0）\n"
         "  --right-host N        右目 MIPI host（默认 1）\n"
         "  --timeout-ms N        单帧获取超时（默认 1000）\n"
         "  --max-pairs N         最多保存对数，0=不限（默认 0，Ctrl-C 停止）\n"
         "  --duration-sec N      最长采集秒数，0=不限（默认 0）\n"
         "  --stride N            每 N 对保存一对，用于原始流抽帧（默认 1）\n"
         "  --max-skew-ns N       左右硬件时间戳配对容差（默认 1000000）\n"
         "  -h, --help            显示帮助\n"
         "示例: " + program +
         " -o data/raw --scene grass_backlight --stride 3 --duration-sec 300\n"
         "产物: <output>/<scene>_<starttime>/{left,right}/*.jpg + index.csv + meta.json";
}

}  // namespace ps
