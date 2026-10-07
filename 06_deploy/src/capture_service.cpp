#include "vio_pipeline.hpp"
#include "socket_protocol.hpp"
#include "logger.hpp"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

using namespace deploy;
namespace { std::atomic<bool> stop{false}; void onSignal(int){stop.store(true);} }

struct PendingFrame {
  VseFrame frame;
  float acquire_ms = 0;
  float send_ms = 0;
  float queue_wait_ms = 0;
  std::chrono::steady_clock::time_point acquired_at;
  std::chrono::steady_clock::time_point sent_at;
};

int main(int argc, char** argv) {
  std::string sock = "/tmp/cv_pipeline_left.sock", log = "/tmp/capture_service.log";
  int host = 0, max_frames = 0, warmup = 5, queue_capacity = 2;
  double hz = 15.;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--socket") sock = val(); else if (a == "--log") log = val();
    else if (a == "--host") host = std::atoi(val().c_str());
    else if (a == "--hz") hz = std::atof(val().c_str());
    else if (a == "--frames") max_frames = std::atoi(val().c_str());
    else if (a == "--warmup") warmup = std::atoi(val().c_str());
    else if (a == "--queue-capacity") queue_capacity = std::max(1, std::atoi(val().c_str()));
  }

  if (!std::isfinite(hz) || hz < 1. || hz > 120.) return 2;
  queue_capacity = std::min(queue_capacity, 2); // VSE owns only three buffers.
  Logger logger(log); signal(SIGINT, onSignal); signal(SIGTERM, onSignal);
  int server = makeServer(sock);
  if (server < 0) { logger.error("cannot create unix socket " + sock); return 2; }
  logger.info("capture service listening socket=" + sock + " host=" + std::to_string(host));
  int client = accept(server, nullptr, nullptr);
  if (client < 0) { logger.error("accept failed"); close(server); return 2; }

  AsyncLogger sdk_logger; LeftVioPipeline vio(host, &sdk_logger);
  if (!vio.initialize()) { logger.error("VIO initialize failed: " + vio.lastError()); close(client); close(server); return 3; }
  logger.info("left camera VIO/VSE started at " + std::to_string(hz) + " Hz warmup=" + std::to_string(warmup) + " queue=" + std::to_string(queue_capacity));

  for (int i = 0; i < warmup && !stop.load(); ++i) {
    VseFrame warm;
    if (vio.acquire(200, warm)) logger.debug("warmup frame=" + std::to_string(warm.frameId()));
  }

  std::mutex mutex;
  std::condition_variable queue_changed;
  std::deque<PendingFrame> pending;
  int produced = 0, completed = 0;

  std::thread receiver([&] {
    Result result; result.detections.reserve(100);
    auto previous_result = std::chrono::steady_clock::time_point{};
    while (!stop.load()) {
      auto receive_begin = std::chrono::steady_clock::now();
      if (!recvResult(client, result)) { if(!stop.load()) logger.warn("receive result ended"); break; }
      const auto received_at = std::chrono::steady_clock::now();
      PendingFrame completed_frame;
      bool found = false;
      {
        std::unique_lock<std::mutex> lock(mutex);
        for (auto it = pending.begin(); it != pending.end(); ++it) {
          if (it->frame.frameId() == result.id) {
            completed_frame = std::move(*it);
            pending.erase(it);
            found = true;
            ++completed;
            break;
          }
        }
      }
      if (!found) { logger.warn("result for unknown frame=" + std::to_string(result.id)); continue; }
      const auto release_begin = std::chrono::steady_clock::now();
      completed_frame.frame.release();
      const auto released_at = std::chrono::steady_clock::now();
      queue_changed.notify_all();
      const float interval_ms = previous_result == std::chrono::steady_clock::time_point{} ? 0.f :
          std::chrono::duration<float, std::milli>(received_at - previous_result).count();
      previous_result = received_at;
      float result_wait_ms = std::chrono::duration<float, std::milli>(
          received_at - completed_frame.sent_at).count();
      float receiver_wait_ms = std::chrono::duration<float, std::milli>(
          received_at - receive_begin).count();
      logger.info("frame=" + std::to_string(result.id) + " det=" + std::to_string(result.detections.size()) +
                  " acquire_ms=" + std::to_string(completed_frame.acquire_ms) +
                  " send_ms=" + std::to_string(completed_frame.send_ms) +
                  " result_wait_ms=" + std::to_string(result_wait_ms) +
                  " import_ms=" + std::to_string(result.import_ms) +
                  " bpu_ms=" + std::to_string(result.bpu_ms) +
                  " post_ms=" + std::to_string(result.post_ms) +
                  " e2e_ms=" + std::to_string(std::chrono::duration<float, std::milli>(received_at - completed_frame.acquired_at).count()) +
                  " receiver_wait_ms=" + std::to_string(receiver_wait_ms) +
                  " queue_wait_ms=" + std::to_string(completed_frame.queue_wait_ms) +
                  " state_lock_ms=" + std::to_string(std::chrono::duration<float, std::milli>(release_begin - received_at).count()) +
                  " frame_release_ms=" + std::to_string(std::chrono::duration<float, std::milli>(released_at - release_begin).count()) +
                  " result_interval_ms=" + std::to_string(interval_ms));
    }
    stop.store(true); queue_changed.notify_all();
  });

  const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / hz));
  auto next = std::chrono::steady_clock::now();
  while (!stop.load() && (max_frames <= 0 || produced < max_frames)) {
    std::this_thread::sleep_until(next);
    const auto slot_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex);
    while (!stop.load() && pending.size() >= static_cast<std::size_t>(queue_capacity))
      queue_changed.wait_for(lock, std::chrono::milliseconds(50));
    if (stop.load()) break;
    lock.unlock();

    auto acquire_begin = std::chrono::steady_clock::now();
    // Skip missed slots; never burst to catch up after a stalled frame.
    next = std::max(next + period, acquire_begin + period);
    VseFrame frame;
    if (!vio.acquire(100, frame)) {
      logger.warn("frame acquire failed: " + vio.lastError());
      continue;
    }
    float acquire_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - acquire_begin).count();
    logger.debug("frame=" + std::to_string(frame.id) + " fd=" + std::to_string(frame.layout.y_fd) +
                 " share=" + std::to_string(frame.layout.y_share_id) + " phy=" + std::to_string(frame.layout.y_phy) +
                 " bytes=" + std::to_string(frame.layout.y_bytes + frame.layout.uv_bytes));

    PendingFrame item;
    item.acquire_ms = acquire_ms;
    item.acquired_at = acquire_begin;
    item.queue_wait_ms = std::chrono::duration<float, std::milli>(acquire_begin-slot_begin).count();
    item.sent_at = std::chrono::steady_clock::now();
    item.frame = std::move(frame);
    lock.lock();
    pending.emplace_back(std::move(item));
    PendingFrame& queued = pending.back();
    auto send_begin = std::chrono::steady_clock::now();
    bool sent = sendFrame(client, queued.frame);
    queued.send_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - send_begin).count();
    queued.sent_at = std::chrono::steady_clock::now();
    lock.unlock();
    if (!sent) { logger.error("send frame failed"); stop.store(true); queue_changed.notify_all(); break; }
    ++produced;
  }

  {
    std::unique_lock<std::mutex> lock(mutex);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!stop.load() && !pending.empty()) {
      if (queue_changed.wait_until(lock, deadline) == std::cv_status::timeout) {
        logger.error("result drain timeout"); break;
      }
    }
  }
  stop.store(true); shutdown(client, SHUT_RDWR); if (receiver.joinable()) receiver.join();
  pending.clear(); // Return every frame before destroying the VIO flow.
  vio.stop(); close(client); close(server); unlink(sock.c_str());
  logger.info("capture service stopped produced=" + std::to_string(produced) + " completed=" + std::to_string(completed));
  return 0;
}
