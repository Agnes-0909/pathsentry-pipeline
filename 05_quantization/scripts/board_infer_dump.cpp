#include <dnn/hb_dnn.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
int main(int argc,char**argv){
 if(argc<4){std::cerr<<"usage model image prefix\n";return 2;}
 const char* mf=argv[1]; const char* imf=argv[2]; const char* pref=argv[3];
 hbPackedDNNHandle_t packed=nullptr; int r=hbDNNInitializeFromFiles(&packed,&mf,1); if(r){std::cerr<<"init "<<r<<"\n";return 1;}
 const char** names=nullptr; int nc=0; hbDNNGetModelNameList(&names,&nc,packed); if(nc<1){return 1;} hbDNNHandle_t h=nullptr; r=hbDNNGetModelHandle(&h,packed,names[0]); if(r){std::cerr<<"handle "<<r<<"\n";return 1;}
 int ic=0,oc=0; hbDNNGetInputCount(&ic,h); hbDNNGetOutputCount(&oc,h); std::cerr<<"ic="<<ic<<" oc="<<oc<<" name="<<names[0]<<"\n";
 hbDNNTensorProperties ip{}; hbDNNGetInputTensorProperties(&ip,h,0);
 std::cerr<<"input dims"; for(int i=0;i<ip.validShape.numDimensions;i++)std::cerr<<" "<<ip.validShape.dimensionSize[i]; std::cerr<<" type="<<ip.tensorType<<" layout="<<ip.tensorLayout<<" bytes="<<ip.alignedByteSize<<"\n";
 cv::Mat bgr=cv::imread(imf,cv::IMREAD_COLOR); if(bgr.empty()){std::cerr<<"bad image\n";return 1;} cv::Mat rgb,res; cv::cvtColor(bgr,rgb,cv::COLOR_BGR2RGB); cv::resize(rgb,res,cv::Size(ip.validShape.dimensionSize[3],ip.validShape.dimensionSize[2]));
 hbDNNTensor input{}; input.properties=ip; r=hbSysAllocCachedMem(&input.sysMem[0],ip.alignedByteSize); if(r){std::cerr<<"alloc in "<<r<<"\n";return 1;}
 int H=res.rows,W=res.cols; cv::Mat yuv; cv::cvtColor(res,yuv,cv::COLOR_RGB2YUV_I420); unsigned char *p=(unsigned char*)input.sysMem[0].virAddr; std::memcpy(p,yuv.data,H*W); unsigned char *u=yuv.data+H*W, *v=u+(H/2)*(W/2); for(int y=0;y<H/2;y++) for(int x=0;x<W/2;x++){p[H*W + y*W + 2*x]=u[y*(W/2)+x]; p[H*W + y*W + 2*x+1]=v[y*(W/2)+x];} hbSysFlushMem(&input.sysMem[0],HB_SYS_MEM_CACHE_CLEAN);
 std::vector<hbDNNTensor> outs(oc); std::vector<hbDNNTensor*> outp(oc); for(int i=0;i<oc;i++){hbDNNGetOutputTensorProperties(&outs[i].properties,h,i); r=hbSysAllocCachedMem(&outs[i].sysMem[0],outs[i].properties.alignedByteSize); if(r){std::cerr<<"alloc out "<<i<<" "<<r<<"\n";return 1;} outp[i]=&outs[i]; std::cerr<<"out"<<i<<" dims";for(int j=0;j<outs[i].properties.validShape.numDimensions;j++)std::cerr<<" "<<outs[i].properties.validShape.dimensionSize[j];std::cerr<<" type="<<outs[i].properties.tensorType<<" bytes="<<outs[i].properties.alignedByteSize<<"\n";}
 hbDNNInferCtrlParam ctrl; HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl); hbDNNTaskHandle_t task=nullptr; r=hbDNNInfer(&task,outp.data(),&input,h,&ctrl); std::cerr<<"infer="<<r<<"\n"; if(!r){r=hbDNNWaitTaskDone(task,10000);std::cerr<<"wait="<<r<<"\n";}
 for(int i=0;i<oc;i++){hbSysFlushMem(&outs[i].sysMem[0],HB_SYS_MEM_CACHE_INVALIDATE); std::string f=std::string(pref)+"_"+std::to_string(i)+".bin"; std::ofstream o(f,std::ios::binary); o.write((char*)outs[i].sysMem[0].virAddr,outs[i].properties.alignedByteSize); std::cerr<<"wrote "<<f<<"\n"; hbSysFreeMem(&outs[i].sysMem[0]);} if(task)hbDNNReleaseTask(task); hbSysFreeMem(&input.sysMem[0]); hbDNNRelease(packed); }
