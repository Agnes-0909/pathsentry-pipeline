#pragma once
#include <cstdint>
#include <map>
#include <string>
enum class LogLevel { kDebug, kInfo, kError };
using LogFields = std::map<std::string, std::uint64_t>;
struct LogEvent { LogLevel level; std::string component,event,thread,channel; std::int64_t frame_id; std::string message; LogFields fields; };
class AsyncLogger { public: void log(const LogEvent&) noexcept {} };
