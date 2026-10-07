#pragma once
#include "deploy_types.hpp"
#include "logger.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
namespace deploy {
class ResultWriter {
 public:
  ResultWriter(std::string directory, std::string mode, Logger& logger)
      : directory_(std::move(directory)), mode_(std::move(mode)), logger_(logger) {
    if(mode_!="off") std::filesystem::create_directories(directory_);
    for(auto& result:queue_) result.detections.reserve(100);
    if(mode_=="async") worker_=std::thread(&ResultWriter::consume,this);
  }
  ~ResultWriter(){
    {std::lock_guard<std::mutex> lock(mutex_); stopped_=true;}
    changed_.notify_one(); if(worker_.joinable()) worker_.join();
    logger_.info("result writer stopped written="+std::to_string(written_)+
      " dropped="+std::to_string(dropped_)+" errors="+std::to_string(errors_));
  }
  void submit(const Result& result){
    if(mode_=="off") return;
    if(mode_=="sync"){save(result); return;}
    {std::lock_guard<std::mutex> lock(mutex_);
     if(size_==queue_.size()){++dropped_; return;}
     queue_[(head_+size_)%queue_.size()]=result; ++size_;}
    changed_.notify_one();
  }
 private:
  void consume(){
    Result result; result.detections.reserve(100);
    for(;;){
      {std::unique_lock<std::mutex> lock(mutex_);
       changed_.wait(lock,[&]{return stopped_ || size_;});
       if(!size_ && stopped_) break;
       std::swap(result,queue_[head_]); head_=(head_+1)%queue_.size(); --size_;}
      save(result);
    }
  }
  void save(const Result& r){
    const auto begin=std::chrono::steady_clock::now();
    std::ofstream jf(directory_+"/frame_"+std::to_string(r.id)+".json");
    jf<<"{\"frame_id\":"<<r.id<<",\"infer_ms\":"<<r.infer_ms<<",\"post_ms\":"<<r.post_ms<<",\"detections\":[";
    for(std::size_t i=0;i<r.detections.size();++i){const auto& d=r.detections[i];if(i)jf<<",";
      jf<<"{\"class\":"<<d.cls<<",\"score\":"<<d.score<<",\"xyxy\":["<<d.x1<<","<<d.y1<<","<<d.x2<<","<<d.y2<<"]}";}
    jf<<"]}\n"; jf.close();
    std::ofstream mf(directory_+"/frame_"+std::to_string(r.id)+".pgm",std::ios::binary);
    mf<<"P5\n"<<kMaskW<<" "<<kMaskH<<"\n255\n";
    mf.write(reinterpret_cast<const char*>(r.mask.data()),r.mask.size()); mf.close();
    if(!jf || !mf){++errors_;logger_.error("result file write failed frame="+std::to_string(r.id));}
    else ++written_;
    const float elapsed=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-begin).count();
    logger_.info("saved frame="+std::to_string(r.id)+" file_write_ms="+std::to_string(elapsed));
  }
  std::string directory_,mode_; Logger& logger_;
  std::array<Result,8> queue_; std::size_t head_=0,size_=0;
  std::mutex mutex_; std::condition_variable changed_; bool stopped_=false;
  std::uint64_t written_=0,dropped_=0,errors_=0;
  std::thread worker_;
};
}
