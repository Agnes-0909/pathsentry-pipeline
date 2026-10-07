#include "raw_head_runner.hpp"
#include "socket_protocol.hpp"
#include "logger.hpp"
#include "result_writer.hpp"
#include <atomic>
#include <csignal>
#include <chrono>
#include <cstdlib>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
using namespace deploy;
namespace {
std::atomic<bool> stop{false};
void onSignal(int){stop.store(true);}
using Clock=std::chrono::steady_clock;
float ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<float,std::milli>(b-a).count();}
}
int main(int argc,char**argv){
  std::string sock="/tmp/cv_pipeline_left.sock",model="/opt/cv_pipeline/models/h2_raw_head.bin";
  std::string log="/tmp/inference_service.log",out_dir="/opt/cv_pipeline/logs/results",save_mode="async";
  float score=.25f,nms=.45f;
  for(int i=1;i<argc;++i){
    std::string a=argv[i]; auto val=[&](){return i+1<argc?std::string(argv[++i]):std::string();};
    if(a=="--socket")sock=val(); else if(a=="--model")model=val();
    else if(a=="--log")log=val(); else if(a=="--output-dir")out_dir=val();
    else if(a=="--score")score=std::atof(val().c_str()); else if(a=="--nms")nms=std::atof(val().c_str());
    else if(a=="--save-results")save_mode=val();
  }
  Logger logger(log); signal(SIGINT,onSignal); signal(SIGTERM,onSignal);
  if(save_mode!="async" && save_mode!="sync" && save_mode!="off"){
    logger.error("--save-results must be async, sync or off");return 2;}
  int client=connectUnix(sock);
  if(client<0){logger.error("connect failed "+sock);return 2;}
  RawHeadRunner runner({model,score,nms,1});
  if(!runner.initialize()){logger.error("model initialize failed: "+runner.lastError());close(client);return 3;}
  ResultWriter writer(out_dir,save_mode,logger);
  logger.info("inference service ready model="+model+" save_results="+save_mode);
  Result r; r.detections.reserve(100);
  while(!stop.load()){
    const auto recv_begin=Clock::now(); FrameHeader h{};int fds[2];
    if(!recvFrame(client,h,fds))break;
    const auto received=Clock::now();
    logger.debug("received frame="+std::to_string(h.frame_id)+" phy="+std::to_string(h.y_phy));
    r.id=h.frame_id;r.timestamp=h.timestamp;
    bool ok=runner.inferFromFd(fds[0],h.y_bytes+h.uv_bytes,h.y_share_id,h.y_phy,h.uv_phy,r);
    for(int fd:fds)if(fd>=0)close(fd);
    if(!ok){logger.error("inference failed frame="+std::to_string(h.frame_id)+" "+runner.lastError());break;}
    const auto persist_begin=Clock::now();
    // Diagnostic sync mode preserves the old ordering for controlled comparisons.
    if(save_mode=="sync")writer.submit(r);
    const auto send_begin=Clock::now();
    if(!sendResult(client,r)){logger.error("send result failed");break;}
    const auto sent=Clock::now();
    if(save_mode=="async")writer.submit(r);
    const auto submitted=Clock::now();
    logger.info("frame="+std::to_string(r.id)+" det="+std::to_string(r.detections.size())+
      " recv_wait_ms="+std::to_string(ms(recv_begin,received))+
      " import_ms="+std::to_string(r.import_ms)+" bpu_ms="+std::to_string(r.bpu_ms)+
      " submit_ms="+std::to_string(r.submit_ms)+" task_wait_ms="+std::to_string(r.wait_ms)+
      " task_release_ms="+std::to_string(r.task_release_ms)+" cache_ms="+std::to_string(r.cache_ms)+
      " post_ms="+std::to_string(r.post_ms)+" result_send_ms="+std::to_string(ms(send_begin,sent))+
      " service_ms="+std::to_string(ms(received,sent))+
      " persist_submit_ms="+std::to_string(ms(persist_begin,send_begin)+ms(sent,submitted)));
  }
  close(client);logger.info("inference service stopped");return 0;
}
