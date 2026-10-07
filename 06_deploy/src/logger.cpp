#include "logger.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
namespace deploy {
Logger::Logger(std::string path) : file_(std::move(path), std::ios::app) {
  const char* level=std::getenv("CV_LOG_LEVEL");
  debug_=level && std::string(level)=="debug";
  worker_=std::thread(&Logger::consume,this);
}
Logger::~Logger(){stop();}
void Logger::stop(){
  {std::lock_guard<std::mutex> lock(mutex_); stopped_=true;}
  changed_.notify_one();
  if(worker_.joinable()) worker_.join();
}
void Logger::write(const char* level,const std::string& msg){
  auto now=std::chrono::system_clock::now();
  auto t=std::chrono::system_clock::to_time_t(now); std::tm tm{}; localtime_r(&t,&tm);
  char ts[32]; std::strftime(ts,sizeof(ts),"%Y-%m-%dT%H:%M:%S",&tm);
  std::string line=std::string(ts)+" ["+level+"] "+msg+"\n";
  {std::lock_guard<std::mutex> lock(mutex_);
   if(stopped_) return;
   if(size_==queue_.size()){++dropped_; return;}
   queue_[(head_+size_)%queue_.size()]=std::move(line); ++size_;}
  changed_.notify_one();
}
void Logger::consume(){
  std::vector<std::string> batch; batch.reserve(queue_.size());
  for(;;){
    std::uint64_t lost=0; bool done=false;
    {std::unique_lock<std::mutex> lock(mutex_);
     changed_.wait_for(lock,std::chrono::milliseconds(100),[&]{return stopped_ || size_>=64;});
     while(size_){batch.emplace_back(std::move(queue_[head_])); head_=(head_+1)%queue_.size(); --size_;}
     lost=dropped_; dropped_=0; done=stopped_;}
    for(const auto& line:batch){std::cerr<<line; if(file_) file_<<line;}
    if(lost){std::string msg="[WARN] logger dropped="+std::to_string(lost)+"\n"; std::cerr<<msg; if(file_) file_<<msg;}
    if(file_) file_.flush();
    batch.clear();
    if(done) break;
  }
}
void Logger::info(const std::string&s){write("INFO",s);}
void Logger::warn(const std::string&s){write("WARN",s);}
void Logger::error(const std::string&s){write("ERROR",s);}
void Logger::debug(const std::string&s){if(debug_)write("DEBUG",s);}
}
