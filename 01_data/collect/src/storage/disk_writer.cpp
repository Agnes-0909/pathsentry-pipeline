#include "ps_collect/disk_writer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <utility>

namespace ps {

namespace {

// stride 对齐的 NV12 双 plane -> BGR（裁掉 stride padding）
cv::Mat nv12ToBgr(const Nv12Image& nv12) {
  // 只读包装：cvtColorTwoPlane 输出到独立 Mat，不会改动源数据
  auto* y_ptr = const_cast<std::uint8_t*>(nv12.data.data());
  const cv::Mat y_plane(static_cast<int>(nv12.height),
                        static_cast<int>(nv12.width), CV_8UC1, y_ptr,
                        nv12.stride);
  // NV12 的 UV 交织行 = width/2 个 uint16 元素（共 width 字节），step 仍为 stride
  const cv::Mat uv_plane(static_cast<int>((nv12.height + 1) / 2),
                         static_cast<int>(nv12.width / 2), CV_8UC2,
                         y_ptr + nv12.ySize(), nv12.stride);
  cv::Mat bgr;
  cv::cvtColorTwoPlane(y_plane, uv_plane, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

}  // namespace

DiskWriter::DiskWriter(std::string session_dir, std::string format,
                       int jpg_quality)
    : session_dir_(std::move(session_dir)),
      format_(std::move(format)),
      jpg_quality_(jpg_quality) {}

DiskWriter::~DiskWriter() { close(); }

bool DiskWriter::start() {
  const std::string left_dir = session_dir_ + "/left";
  const std::string right_dir = session_dir_ + "/right";
  const std::string command = "mkdir -p '" + left_dir + "' '" + right_dir + "'";
  if (std::system(command.c_str()) != 0) return false;
  index_.open(session_dir_ + "/index.csv", std::ios::trunc);
  if (!index_) return false;
  index_ << "frame_index,left_frame_id,right_frame_id,left_ts_ns,right_ts_ns,"
            "skew_ns,left_file,right_file\n";
  worker_ = std::thread([this] { run(); });
  return true;
}

bool DiskWriter::enqueue(StereoFrame&& frame) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_) return false;
    const bool queue_full = queue_.size() >= kMaxQueueDepth;
    if (queue_full) {
      ++stats_.dropped_pairs;  // 丢新帧保采集，不阻塞取帧线程
      return true;
    }
    queue_.push(std::move(frame));
  }
  condition_.notify_one();
  return true;
}

void DiskWriter::close() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_ || !worker_.joinable()) return;
    closed_ = true;
  }
  condition_.notify_all();
  worker_.join();
  if (index_.is_open()) index_.close();
}

const DiskWriter::Stats& DiskWriter::stats() const { return stats_; }

void DiskWriter::run() {
  while (true) {
    StereoFrame frame;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
      if (queue_.empty()) {
        if (closed_) break;
        continue;
      }
      frame = std::move(queue_.front());
      queue_.pop();
    }
    writePair(frame);
    condition_.notify_all();
  }
}

void DiskWriter::writePair(const StereoFrame& frame) {
  const std::uint64_t index = [this] {
    std::lock_guard<std::mutex> lock(index_mutex_);
    return next_index_++;
  }();

  const std::string ext = format_ == "png" ? "png" : "jpg";
  char name[64];
  std::snprintf(name, sizeof(name), "%06llu.%s",
                static_cast<unsigned long long>(index), ext.c_str());
  const std::string left_path = session_dir_ + "/left/" + name;
  const std::string right_path = session_dir_ + "/right/" + name;

  std::vector<int> encode_params;
  if (ext == "jpg") encode_params = {cv::IMWRITE_JPEG_QUALITY, jpg_quality_};

  bool ok = writeImage(left_path, frame.left, encode_params) &&
            writeImage(right_path, frame.right, encode_params);
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    index_ << index << ',' << frame.left_frame_id << ',' << frame.right_frame_id
           << ',' << frame.left_timestamp_ns << ',' << frame.right_timestamp_ns
           << ',' << frame.skew_ns << ',' << left_path << ',' << right_path
           << '\n';
  }
  if (ok) {
    ++stats_.written_pairs;
  } else {
    ++stats_.write_failures;
    std::fprintf(stderr, "[ps_collector] write failed for pair %llu\n",
                 static_cast<unsigned long long>(index));
  }
}

bool DiskWriter::writeImage(const std::string& path, const Nv12Image& nv12,
                            const std::vector<int>& encode_params) {
  const cv::Mat bgr = nv12ToBgr(nv12);
  if (bgr.empty()) return false;
  return cv::imwrite(path, bgr, encode_params);
}

}  // namespace ps
