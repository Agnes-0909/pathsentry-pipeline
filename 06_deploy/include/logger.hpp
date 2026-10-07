#pragma once
#include <array>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
namespace deploy {
// Bounded diagnostics: slow storage must not stall the inference path.
class Logger {
 public:
  explicit Logger(std::string path); ~Logger();
  void info(const std::string& msg); void warn(const std::string& msg);
  void error(const std::string& msg); void debug(const std::string& msg);
  void stop();
 private:
  void write(const char* level, const std::string& msg);
  void consume();
  std::mutex mutex_; std::condition_variable changed_;
  std::ofstream file_; std::array<std::string,1024> queue_;
  std::size_t head_=0, size_=0; std::uint64_t dropped_=0;
  bool stopped_=false, debug_=false;
  std::thread worker_;
};
}
