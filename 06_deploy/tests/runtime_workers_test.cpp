#include "logger.hpp"
#include "result_writer.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <unistd.h>
using namespace deploy;
static std::string read(const std::filesystem::path& path){
  std::ifstream stream(path,std::ios::binary);
  return {std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
}
int main(){
  const auto dir=std::filesystem::temp_directory_path()/("cv-worker-test-"+std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  {
    Logger logger((dir/"test.log").string());
    std::thread a([&]{for(int i=0;i<100;++i)logger.info("thread_a="+std::to_string(i));});
    std::thread b([&]{for(int i=0;i<100;++i)logger.info("thread_b="+std::to_string(i));});
    a.join(); b.join();
    {
      ResultWriter writer((dir/"results").string(),"async",logger);
      Result r;
      for(int i=0;i<5;++i){
        r.id=i;r.mask.fill(static_cast<unsigned char>(i+1));
        r.detections.assign(1,Detection{1,2,3,4,.9f,i});
        writer.submit(r);
      }
      // Reuse the source immediately: background writes must own a snapshot.
      r.mask.fill(255);r.detections.clear();
    }
    logger.stop(); // Drains messages, and destructor must remain safe afterward.
  }
  for(int i=0;i<5;++i){
    const auto mask=read(dir/"results"/("frame_"+std::to_string(i)+".pgm"));
    const std::string header="P5\n240 208\n255\n";
    assert(mask.size()==header.size()+kMaskW*kMaskH);
    assert(mask.substr(0,header.size())==header);
    for(std::size_t j=header.size();j<mask.size();++j)assert(mask[j]==i+1);
    const auto json=read(dir/"results"/("frame_"+std::to_string(i)+".json"));
    assert(json.find("\"class\":"+std::to_string(i))!=std::string::npos);
  }
  const auto log=read(dir/"test.log");
  for(int i=0;i<100;++i){assert(log.find("thread_a="+std::to_string(i)+"\n")!=std::string::npos);assert(log.find("thread_b="+std::to_string(i)+"\n")!=std::string::npos);}
  assert(log.find("written=5 dropped=0 errors=0")!=std::string::npos);
  std::filesystem::remove_all(dir);
}
