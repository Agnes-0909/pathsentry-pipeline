#ifndef PS_COLLECT_DISK_WRITER_HPP_
#define PS_COLLECT_DISK_WRITER_HPP_

#include "stereo_frame.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace ps {

// 后台落盘线程：有界队列接收配对帧，NV12->BGR->JPEG/PNG 写盘并记录 index.csv。
// 队列满时丢新帧，保证取帧线程不被磁盘速度拖垮。
class DiskWriter {
 public:
  struct Stats {
    std::uint64_t written_pairs = 0;
    std::uint64_t dropped_pairs = 0;
    std::uint64_t write_failures = 0;
  };

  static constexpr std::size_t kMaxQueueDepth = 16;

  DiskWriter(std::string session_dir, std::string format, int jpg_quality);
  ~DiskWriter();

  bool start();
  bool enqueue(StereoFrame&& frame);
  void close();
  const Stats& stats() const;

 private:
  void run();
  void writePair(const StereoFrame& frame);
  bool writeImage(const std::string& path, const Nv12Image& nv12,
                  const std::vector<int>& encode_params);

  std::string session_dir_;
  std::string format_;
  int jpg_quality_;
  std::thread worker_;
  std::mutex mutex_;
  std::mutex index_mutex_;
  std::condition_variable condition_;
  std::queue<StereoFrame> queue_;
  std::ofstream index_;
  std::uint64_t next_index_ = 0;
  bool closed_ = false;
  Stats stats_;
};

}  // namespace ps

#endif  // PS_COLLECT_DISK_WRITER_HPP_
