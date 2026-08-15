// ps_collector：PathSentry 双目数据采集主程序（X5 板端）
// 主循环：取左右帧 -> 硬件时间戳配对 -> 抽帧 -> 深拷贝入落盘队列
#include "disk_writer.hpp"
#include "options.hpp"
#include "vio_dual.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "hb_mem_mgr.h"

namespace {

std::atomic_bool g_stop_requested{false};

void handleSignal(int) { g_stop_requested.store(true); }

std::string makeSessionName(const std::string& scene) {
  char buffer[64];
  const auto now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local);
  return scene + "_" + buffer;
}

void writeMetaJson(const std::string& path, const ps::Options& options,
                   const std::string& session_name) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) return;
  output << "{\n"
         << "  \"tool\": \"ps_collector\",\n"
         << "  \"session\": \"" << session_name << "\",\n"
         << "  \"scene\": \"" << options.scene << "\",\n"
         << "  \"format\": \"" << options.format << "\",\n"
         << "  \"jpg_quality\": " << options.jpg_quality << ",\n"
         << "  \"stride\": " << options.stride << ",\n"
         << "  \"sensor\": \"sc132gs_linear_1088x1280_raw10_30fps_1lane\",\n"
         << "  \"resolution\": \"1088x1280 NV12\",\n"
         << "  \"stereo_sync\": \"LPWM hardware trigger\",\n"
         << "  \"max_pair_skew_ns\": " << options.max_pair_skew_ns << ",\n"
         << "  \"left_host\": " << options.left_host << ",\n"
         << "  \"right_host\": " << options.right_host << "\n"
         << "}\n";
}

constexpr int kMaxStereoRealignmentSteps = 16;

}  // namespace

int main(int argc, char** argv) {
  ps::Options options;
  try {
    options = ps::parseOptions(argc, argv);
  } catch (const ps::HelpRequested&) {
    std::cout << ps::usage(argv[0]) << '\n';
    return 0;
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << '\n' << ps::usage(argv[0]) << '\n';
    return 2;
  }

  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  const std::string session_name = makeSessionName(options.scene);
  const std::string session_dir = options.output_root + "/" + session_name;

  int exit_code = 0;
  std::string fatal_error;

  const int memory_result = hb_mem_module_open();
  if (memory_result != 0) {
    std::fprintf(stderr,
                 "[ps_collector] hb_mem_module_open failed with code %d\n",
                 memory_result);
    return 1;
  }

  ps::DiskWriter writer(session_dir, options.format, options.jpg_quality);
  ps::vio::DualVioPipeline pipelines(options.left_host, options.right_host);

  std::uint64_t acquired_pairs = 0;
  std::uint64_t saved_pairs = 0;
  std::uint64_t left_timeouts = 0;
  std::uint64_t right_timeouts = 0;
  std::uint64_t sync_drops = 0;
  std::uint64_t max_skew = 0;

  const auto start_time = std::chrono::steady_clock::now();

  do {
    if (!writer.start()) {
      std::fprintf(stderr, "[ps_collector] cannot create session at %s\n",
                   session_dir.c_str());
      exit_code = 1;
      break;
    }
    writeMetaJson(session_dir + "/meta.json", options, session_name);

    if (!pipelines.initialize()) {
      std::fprintf(stderr, "[ps_collector] VIO init failed: %s\n",
                   pipelines.lastError().c_str());
      fatal_error = pipelines.lastError();
      exit_code = 1;
      break;
    }
    std::printf("[ps_collector] session=%s 双目管线已启动，Ctrl-C 停止\n",
                session_dir.c_str());

    int consecutive_timeouts = 0;
    while (!g_stop_requested.load()) {
      if (options.max_pairs != 0 && saved_pairs >= options.max_pairs) break;
      if (options.duration_sec != 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now() - start_time)
                  .count() >= static_cast<long long>(options.duration_sec)) {
        break;
      }

      ps::Nv12Image left_image;
      ps::Nv12Image right_image;
      std::uint64_t left_id = 0;
      std::uint64_t right_id = 0;
      std::uint64_t left_ts = 0;
      std::uint64_t right_ts = 0;
      bool left_ok = pipelines.acquireLeft(options.frame_timeout_ms, left_image,
                                           left_id, left_ts);
      bool right_ok = pipelines.acquireRight(options.frame_timeout_ms,
                                             right_image, right_id, right_ts);

      int realignment_steps = 0;
      ps::vio::StereoSyncAction action;
      while (left_ok && right_ok) {
        action = ps::vio::decideStereoSync(left_ts, right_ts,
                                           options.max_pair_skew_ns);
        if (action == ps::vio::StereoSyncAction::kPair) break;
        if (action == ps::vio::StereoSyncAction::kTimestampMissing) {
          std::fprintf(stderr,
                       "[ps_collector] hardware timestamp missing, abort\n");
          exit_code = 1;
          break;
        }
        if (++realignment_steps > kMaxStereoRealignmentSteps) {
          std::fprintf(stderr,
                       "[ps_collector] cannot align stereo within %d steps\n",
                       kMaxStereoRealignmentSteps);
          exit_code = 1;
          break;
        }
        ++sync_drops;
        if (action == ps::vio::StereoSyncAction::kRefreshLeft) {
          left_ok = pipelines.acquireLeft(options.frame_timeout_ms, left_image,
                                          left_id, left_ts);
        } else {
          right_ok = pipelines.acquireRight(options.frame_timeout_ms,
                                            right_image, right_id, right_ts);
        }
      }
      if (exit_code != 0) break;

      if (!left_ok) ++left_timeouts;
      if (!right_ok) ++right_timeouts;
      if (!left_ok || !right_ok) {
        ++consecutive_timeouts;
        if (consecutive_timeouts >= options.max_consecutive_timeouts) {
          std::fprintf(stderr, "[ps_collector] too many consecutive timeouts\n");
          fatal_error = "too many consecutive frame timeouts";
          exit_code = 1;
          break;
        }
        continue;
      }
      consecutive_timeouts = 0;
      ++acquired_pairs;

      const std::uint64_t skew =
          left_ts > right_ts ? left_ts - right_ts : right_ts - left_ts;
      if (skew > max_skew) max_skew = skew;

      if (acquired_pairs % options.stride != 0) continue;  // 抽帧跳过
      if (options.max_pairs != 0 && saved_pairs >= options.max_pairs) break;

      ps::StereoFrame pair;
      pair.left = std::move(left_image);
      pair.right = std::move(right_image);
      pair.left_frame_id = left_id;
      pair.right_frame_id = right_id;
      pair.left_timestamp_ns = left_ts;
      pair.right_timestamp_ns = right_ts;
      pair.skew_ns = skew;
      writer.enqueue(std::move(pair));
      ++saved_pairs;

      if (saved_pairs % 100 == 0) {
        std::printf("[ps_collector] saved=%llu acquired=%llu max_skew=%lluns\n",
                    static_cast<unsigned long long>(saved_pairs),
                    static_cast<unsigned long long>(acquired_pairs),
                    static_cast<unsigned long long>(max_skew));
      }
    }
    pipelines.stop();
  } while (false);

  writer.close();
  const auto& stats = writer.stats();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start_time)
                           .count();

  std::printf(
      "[ps_collector] 采集结束: acquired=%llu saved=%llu dropped=%llu "
      "write_failures=%llu sync_drops=%llu timeouts=(%llu/%llu) "
      "max_skew=%lluns elapsed=%lldms\n",
      static_cast<unsigned long long>(acquired_pairs),
      static_cast<unsigned long long>(stats.written_pairs),
      static_cast<unsigned long long>(stats.dropped_pairs),
      static_cast<unsigned long long>(stats.write_failures),
      static_cast<unsigned long long>(sync_drops),
      static_cast<unsigned long long>(left_timeouts),
      static_cast<unsigned long long>(right_timeouts),
      static_cast<unsigned long long>(max_skew),
      static_cast<long long>(elapsed));
  if (!fatal_error.empty()) {
    std::fprintf(stderr, "[ps_collector] fatal: %s\n", fatal_error.c_str());
  }

  hb_mem_module_close();
  return exit_code;
}
