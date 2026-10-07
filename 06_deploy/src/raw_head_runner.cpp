#include "raw_head_runner.hpp"
#include <dnn/hb_dnn.h>
#include <hb_mem_mgr.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unistd.h>
namespace deploy {
namespace { float sigmoid(float x){return 1.f/(1.f+std::exp(-std::max(-80.f,std::min(80.f,x))));}
struct Box{Detection d;}; float iou(const Detection&a,const Detection&b){float x1=std::max(a.x1,b.x1),y1=std::max(a.y1,b.y1),x2=std::min(a.x2,b.x2),y2=std::min(a.y2,b.y2);float w=std::max(0.f,x2-x1),h=std::max(0.f,y2-y1), inter=w*h;float aa=std::max(0.f,a.x2-a.x1)*std::max(0.f,a.y2-a.y1),bb=std::max(0.f,b.x2-b.x1)*std::max(0.f,b.y2-b.y1);return inter/(aa+bb-inter+1e-6f);}}
class RawHeadRunner::Impl {
 public:
  explicit Impl(Config c):cfg_(std::move(c)){}
  ~Impl(){for(auto&kv:registered_){hbSysUnregisterMem(&kv.second.mem); if(kv.second.common.fd>=0) hb_mem_free_buf(kv.second.common.fd);}for(auto&t:outs_)hbSysFreeMem(&t.sysMem[0]);if(handle_)hbDNNRelease(packed_);}
  bool initialize(){const char* f=cfg_.model.c_str();int r=hbDNNInitializeFromFiles(&packed_,&f,1);if(r){err("hbDNNInitializeFromFiles",r);return false;}const char** names=nullptr;int n=0;if((r=hbDNNGetModelNameList(&names,&n,packed_))||n<1){err("hbDNNGetModelNameList",r);return false;}if((r=hbDNNGetModelHandle(&handle_,packed_,names[0]))){err("hbDNNGetModelHandle",r);return false;}hbDNNGetInputTensorProperties(&input_prop_,handle_,0);int oc=0;hbDNNGetOutputCount(&oc,handle_);if(oc!=7){last_error_="raw-head model must have 7 outputs, got "+std::to_string(oc);return false;}outs_.resize(oc);out_ptrs_.resize(oc);for(int i=0;i<oc;i++){hbDNNGetOutputTensorProperties(&outs_[i].properties,handle_,i);if(hbSysAllocCachedMem(&outs_[i].sysMem[0],outs_[i].properties.alignedByteSize)){err("hbSysAllocCachedMem output",i);return false;}out_ptrs_[i]=&outs_[i];}initialized_=true;return true;}
  bool inferFromFd(int fd, std::uint32_t expected, std::int32_t share_id,
                   std::uint64_t y_phy, std::uint64_t uv_phy, Result& r) {
    (void)fd; (void)share_id; (void)uv_phy;
    if (!initialized_) { last_error_ = "runner not initialized"; return false; }
    auto cached = registered_.find(y_phy);
    if (cached != registered_.end()) {
      r.import_ms = 0.0f;
      return runBpu(cached->second.mem, r);
    }
    const auto import_begin = std::chrono::steady_clock::now();
    hb_mem_common_buf_t common{}; common.fd = -1;
    constexpr std::uint32_t kBytes = 960u * 832u + 960u * 416u;
    if (hb_mem_import_com_buf_with_paddr(y_phy, kBytes, 3, &common) != 0) {
      last_error_ = "hb_mem_import_com_buf_with_paddr failed"; return false;
    }
    if (!common.virt_addr || !common.phys_addr) {
      if (common.fd >= 0) hb_mem_free_buf(common.fd);
      last_error_ = "invalid imported common buffer"; return false;
    }
    if (expected && common.size < expected) {
      if (common.fd >= 0) hb_mem_free_buf(common.fd);
      last_error_ = "graphic buffer smaller than frame metadata"; return false;
    }
    hbSysMem mem{}; mem.phyAddr = common.phys_addr; mem.virAddr = common.virt_addr;
    mem.memSize = static_cast<std::uint32_t>(common.size);
    const int rr = hbSysRegisterMem(&mem);
    if (rr) {
      if (common.fd >= 0) hb_mem_free_buf(common.fd);
      err("hbSysRegisterMem", rr); return false;
    }
    Registered reg; reg.mem = mem; reg.common = common;
    auto inserted = registered_.emplace(y_phy, std::move(reg));
    if (!inserted.second) {
      hbSysUnregisterMem(&mem);
      if (common.fd >= 0) hb_mem_free_buf(common.fd);
      r.import_ms = 0.0f;
      return runBpu(inserted.first->second.mem, r);
    }
    r.import_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - import_begin).count();
    return runBpu(inserted.first->second.mem, r);
  }
  const std::string& error()const{return last_error_;}
 private:
  void err(const char*op,int code){last_error_=std::string(op)+" failed: "+std::to_string(code);}
  static int dim(const hbDNNTensorProperties&p,int i){return p.validShape.dimensionSize[i];}
  bool runBpu(const hbSysMem& mem, Result& r) {
    hbDNNTensor input{}; input.properties = input_prop_; input.sysMem[0] = mem;
    const auto t0 = std::chrono::steady_clock::now();
    hbDNNInferCtrlParam ctrl; HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
    ctrl.bpuCoreId = cfg_.bpu_core; hbDNNTaskHandle_t task = nullptr;
    int rr = hbDNNInfer(&task, out_ptrs_.data(), &input, handle_, &ctrl);
    if (rr) { err("hbDNNInfer", rr); return false; }
    const auto submitted = std::chrono::steady_clock::now();
    rr = hbDNNWaitTaskDone(task, 1000);
    const auto waited = std::chrono::steady_clock::now();
    hbDNNReleaseTask(task);
    if (rr) { err("hbDNNWaitTaskDone", rr); return false; }
    const auto t1 = std::chrono::steady_clock::now();
    for (auto& o : outs_) hbSysFlushMem(&o.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    r.submit_ms = std::chrono::duration<float, std::milli>(submitted - t0).count();
    r.wait_ms = std::chrono::duration<float, std::milli>(waited - submitted).count();
    r.task_release_ms = std::chrono::duration<float, std::milli>(t1 - waited).count();
    r.cache_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t1).count();
    r.bpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    r.infer_ms = r.import_ms + r.bpu_ms;
    const auto p0 = std::chrono::steady_clock::now(); decode(r);
    const auto p1 = std::chrono::steady_clock::now();
    r.post_ms = std::chrono::duration<float, std::milli>(p1 - p0).count();
    return true;
  }
  void decode(Result&r){r.detections.clear();constexpr int strides[3]={8,16,32};constexpr int reg=16;for(int s=0;s<3;s++){const float*cls=(const float*)outs_[2*s].sysMem[0].virAddr;const float*b=(const float*)outs_[2*s+1].sysMem[0].virAddr;int h=dim(outs_[2*s].properties,1),w=dim(outs_[2*s].properties,2),nc=dim(outs_[2*s].properties,3);int n=h*w;for(int pos=0;pos<n;pos++){int gy=pos/w,gx=pos%w;for(int c=0;c<nc;c++){float score=sigmoid(cls[pos*nc+c]);if(score<cfg_.score_threshold)continue;float dist[4]{};for(int k=0;k<4;k++){float mx=-1e30f;for(int j=0;j<reg;j++)mx=std::max(mx,b[(pos*4*reg)+k*reg+j]);float sum=0,val=0;for(int j=0;j<reg;j++){float e=std::exp(b[(pos*4*reg)+k*reg+j]-mx);sum+=e;val+=e*j;}dist[k]=val/(sum+1e-9f)*strides[s];}Detection d;float ax=(gx+.5f)*strides[s],ay=(gy+.5f)*strides[s];d.x1=std::max(0.f,ax-dist[0]);d.y1=std::max(0.f,ay-dist[1]);d.x2=std::min(960.f,ax+dist[2]);d.y2=std::min(832.f,ay+dist[3]);d.score=score;d.cls=c;r.detections.push_back(d);}}}std::sort(r.detections.begin(),r.detections.end(),[](const Detection&a,const Detection&b){return a.score>b.score;});std::vector<Detection> keep;keep.reserve(r.detections.size());for(const auto&d:r.detections){bool drop=false;for(const auto&k:keep)if(d.cls==k.cls&&iou(d,k)>cfg_.nms_threshold){drop=true;break;}if(!drop)keep.push_back(d);if(keep.size()>=100)break;}r.detections.swap(keep);const float*seg=(const float*)outs_[6].sysMem[0].virAddr;for(std::size_t i=0;i<r.mask.size();i++)r.mask[i]=sigmoid(seg[i])>.5f?255:0;}
  Config cfg_;hbPackedDNNHandle_t packed_=nullptr;hbDNNHandle_t handle_=nullptr;hbDNNTensorProperties input_prop_{};std::vector<hbDNNTensor> outs_;std::vector<hbDNNTensor*>out_ptrs_;struct Registered { hbSysMem mem{}; hb_mem_graphic_buf_t imported{}; hb_mem_common_buf_t common{}; }; std::unordered_map<std::uint64_t,Registered> registered_;bool initialized_=false;bool input_alloc_=false;std::string last_error_;
};
RawHeadRunner::RawHeadRunner(Config c):impl_(std::make_unique<Impl>(std::move(c))){ } RawHeadRunner::~RawHeadRunner()=default;bool RawHeadRunner::initialize(){return impl_->initialize();}bool RawHeadRunner::inferFromFd(int fd,std::uint32_t n,std::int32_t share_id,std::uint64_t y_phy,std::uint64_t uv_phy,Result&r){return impl_->inferFromFd(fd,n,share_id,y_phy,uv_phy,r);}const std::string&RawHeadRunner::lastError()const{return impl_->error();}
}
