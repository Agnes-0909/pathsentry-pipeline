// 双目 MIPI 采集管线：camera -> VIN -> ISP -> VSE（LPWM 硬件同步触发）
// 改编自 RDK_proj/mipi_vse_bpu_zerocopy/src/vio_pipeline.cpp，去掉推理与日志依赖。
#include "ps_collect/vio_dual.hpp"

#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <memory>
#include <sstream>
#include <sys/ioctl.h>
#include <unistd.h>

#include "hb_camera_interface.h"
#include "hbn_api.h"
#include "vse_cfg.h"
#include "vp_sensors.h"

// tros 版 vp_sensors.h 未逐一声明各 sensor 配置，这里手动引入本工程用到的符号
extern "C" {
extern vp_sensor_config_t sc132gs_linear_1088x1280_raw10_30fps_1lane;
}

namespace ps::vio {
namespace {

constexpr int kSensorWidth = 1088;
constexpr int kSensorHeight = 1280;

struct Pipeline {
  std::string channel;
  int host = -1;
  int i2c_bus = -1;
  camera_config_t camera{};
  mipi_config_t mipi{};
  vin_node_attr_t vin_node_attr{};
  vin_ichn_attr_t vin_ichn_attr{};
  vin_ochn_attr_t vin_ochn_attr{};
  vin_attr_ex_t vin_attr_ex{};
  isp_attr_t isp_attr{};
  isp_ichn_attr_t isp_ichn_attr{};
  isp_ochn_attr_t isp_ochn_attr{};
  camera_handle_t camera_handle = -1;
  hbn_vnode_handle_t vin = 0;
  hbn_vnode_handle_t isp = 0;
  hbn_vnode_handle_t vse = 0;
  hbn_vflow_handle_t flow = 0;
  bool camera_created = false;
  bool vin_created = false;
  bool isp_created = false;
  bool vse_created = false;
  bool flow_created = false;
  bool flow_started = false;
};

struct HardwareSyncTiming {
  int fps = 0;
  std::uint32_t period_us = 0;
  std::uint32_t offset_us = 0;
  std::uint32_t duty_us = 0;
  std::uint32_t trigger_source = 0;
  std::uint32_t trigger_mode = 0;
};

struct SensorRouting {
  int camera_phy = 0;
  int vin_rx = 0;
  int timestamp_source = 0;
};

HardwareSyncTiming sc132gsHardwareSyncTiming() noexcept {
  return {30, 33333, 10, 100, 0, 0};
}

SensorRouting makeSensorRouting(int camera_phy, int vcon_rx_phy,
                                int lpwm_channel) noexcept {
  // ts_src 不覆盖：沿用 sensor 配置默认值（与 tros 在本模组上验证的行为一致）
  return SensorRouting{camera_phy, vcon_rx_phy, 0};
}

std::array<int, 3> sensorPowerSequence(bool active_high) noexcept {
  const int active = active_high ? 1 : 0;
  return {active, 1 - active, active};
}

void cloneSensorConfig(Pipeline& pipeline, int host) {
  const vp_sensor_config_t& source =
      sc132gs_linear_1088x1280_raw10_30fps_1lane;
  pipeline.host = host;
  pipeline.camera = *source.camera_config;
  pipeline.mipi = *source.camera_config->mipi_cfg;
  pipeline.camera.mipi_cfg = &pipeline.mipi;
  pipeline.vin_node_attr = *source.vin_node_attr;
  pipeline.vin_ichn_attr = *source.vin_ichn_attr;
  pipeline.vin_ochn_attr = *source.vin_ochn_attr;
  pipeline.vin_attr_ex = *source.vin_attr_ex;
  pipeline.isp_attr = *source.isp_attr;
  pipeline.isp_ichn_attr = *source.isp_ichn_attr;
  pipeline.isp_ochn_attr = *source.isp_ochn_attr;
  const auto sync = sc132gsHardwareSyncTiming();
  // LPWM 触发模式（对齐 tros mipi_cam 在本模组上的行为：sensor_mode=6，fps 统一 30）
  pipeline.camera.sensor_mode = 6;
  pipeline.camera.fps = sync.fps;
  pipeline.mipi.rx_attr.fps = sync.fps;
  pipeline.vin_node_attr.lpwm_attr.enable = 1;  // LPWM 硬件触发，保证左右目同曝
  for (auto& channel : pipeline.vin_node_attr.lpwm_attr.lpwm_chn_attr) {
    channel.trigger_source = sync.trigger_source;
    channel.trigger_mode = sync.trigger_mode;
    channel.period = sync.period_us;
    channel.offset = sync.offset_us;
    channel.duty_time = sync.duty_us;
    channel.threshold = 0;
    channel.adjust_step = 0;
  }
  pipeline.isp_attr.input_mode = 2;
}

bool hostHasMclkConfig(int host) {
  static const char* const node_suffixes[] = {
      "3d060000", "3d070000", "3d080000", "3d090000"};
  if (host < 0 || host >= 4) return false;
  const std::string path =
      std::string("/proc/device-tree/soc/cam/mipi_host@") +
      node_suffixes[host] + "/pinctrl-names";
  return access(path.c_str(), F_OK) == 0;
}

bool readBigEndianWord(const std::string& path, std::size_t index, int& value) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) return false;
  if (std::fseek(file, static_cast<long>(index * sizeof(std::uint32_t)),
                 SEEK_SET) != 0) {
    std::fclose(file);
    return false;
  }
  std::uint32_t encoded = 0;
  const bool ok = std::fread(&encoded, sizeof(encoded), 1, file) == 1;
  std::fclose(file);
  if (!ok) return false;
  value = static_cast<int>(ntohl(encoded));
  return true;
}

bool writeSysfs(const std::string& path, const std::string& value) {
  FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) return false;
  const bool ok = std::fputs(value.c_str(), file) >= 0;
  const bool closed = std::fclose(file) == 0;
  return ok && closed;
}

bool pulseSensorPower(int gpio, bool active_high) {
  if (gpio <= 0) return false;
  const std::string gpio_path = "/sys/class/gpio/gpio" + std::to_string(gpio);
  const bool already_exported = access(gpio_path.c_str(), F_OK) == 0;
  if (!already_exported &&
      !writeSysfs("/sys/class/gpio/export", std::to_string(gpio))) {
    return false;
  }
  usleep(30 * 1000);
  if (!writeSysfs(gpio_path + "/direction", "out")) return false;
  usleep(30 * 1000);
  for (const int value : sensorPowerSequence(active_high)) {
    if (!writeSysfs(gpio_path + "/value", std::to_string(value))) return false;
    usleep(30 * 1000);
  }
  if (!already_exported) {
    return writeSysfs("/sys/class/gpio/unexport", std::to_string(gpio));
  }
  return true;
}

bool readSc132gsChipId(int bus, int address) {
  const std::string device = "/dev/i2c-" + std::to_string(bus);
  const int file = open(device.c_str(), O_RDWR);
  if (file < 0) return false;
  std::uint8_t register_address[2] = {0x31, 0x07};
  std::uint8_t response[2] = {};
  i2c_msg messages[2]{};
  messages[0].addr = static_cast<__u16>(address);
  messages[0].flags = 0;
  messages[0].len = 2;
  messages[0].buf = register_address;
  messages[1].addr = static_cast<__u16>(address);
  messages[1].flags = I2C_M_RD;
  messages[1].len = 2;
  messages[1].buf = response;
  i2c_rdwr_ioctl_data transfer{};
  transfer.msgs = messages;
  transfer.nmsgs = 2;
  const int ret = ioctl(file, I2C_RDWR, &transfer);
  close(file);
  if (ret < 0) return false;
  const int chip_id = (static_cast<int>(response[0]) << 8) | response[1];
  return chip_id == 0x0132;
}

bool applyDeviceTreeRouting(Pipeline& pipeline) {
  const std::string base = "/proc/device-tree/soc/cam/vcon@" +
                           std::to_string(pipeline.host) + "/";
  int rx_phy = 0;
  int lpwm_channel = 0;
  int reset_gpio = 0;
  int i2c_bus = 0;
  if (!readBigEndianWord(base + "rx_phy", 1, rx_phy) ||
      !readBigEndianWord(base + "lpwm_chn", 0, lpwm_channel) ||
      !readBigEndianWord(base + "gpio_oth", 0, reset_gpio) ||
      !readBigEndianWord(base + "bus", 0, i2c_bus)) {
    return false;
  }
  const SensorRouting routing =
      makeSensorRouting(pipeline.mipi.rx_attr.phy, rx_phy, lpwm_channel);
  pipeline.mipi.rx_attr.phy = routing.camera_phy;
  pipeline.vin_node_attr.cim_attr.mipi_rx = routing.vin_rx;
  pipeline.vin_node_attr.cim_attr.func.ts_src = routing.timestamp_source;
  pipeline.i2c_bus = i2c_bus;
  const bool active_high = pipeline.camera.gpio_level_bit == 0;
  return pulseSensorPower(reset_gpio, active_high);
}

}  // namespace

StereoSyncAction decideStereoSync(std::uint64_t left_ts,
                                  std::uint64_t right_ts,
                                  std::uint64_t max_skew_ns) noexcept {
  if (left_ts == 0 || right_ts == 0) return StereoSyncAction::kTimestampMissing;
  const std::uint64_t skew =
      left_ts > right_ts ? left_ts - right_ts : right_ts - left_ts;
  if (skew <= max_skew_ns) return StereoSyncAction::kPair;
  return left_ts < right_ts ? StereoSyncAction::kRefreshLeft
                            : StereoSyncAction::kRefreshRight;
}

class DualVioPipeline::Impl {
 public:
  Impl(int left_host, int right_host) {
    left_.channel = "left";
    right_.channel = "right";
    left_host_ = left_host;
    right_host_ = right_host;
  }

  ~Impl() { stop(); }

  bool initialize() {
    if (initialized_) return true;
    cloneSensorConfig(right_, right_host_);
    cloneSensorConfig(left_, left_host_);
    if (!prepareOne(right_) || !prepareOne(left_) || !startOne(right_) ||
        !startOne(left_)) {
      stop();
      return false;
    }
    initialized_ = true;
    return true;
  }

  bool acquire(Pipeline& pipeline, int timeout_ms, Nv12Image& out,
               std::uint64_t& frame_id, std::uint64_t& timestamp_ns) {
    hbn_vnode_image_t image{};
    const int ret = hbn_vnode_getframe(pipeline.vse, 0, timeout_ms, &image);
    if (ret != 0) {
      std::ostringstream message;
      message << "hbn_vnode_getframe(host=" << pipeline.host
              << ") failed with code " << ret;
      last_error_ = message.str();
      return false;
    }
    // cached buffer，CPU 读之前必须 invalidate
    hb_mem_invalidate_buf_with_vaddr(
        reinterpret_cast<std::uint64_t>(image.buffer.virt_addr[0]),
        image.buffer.size[0]);
    hb_mem_invalidate_buf_with_vaddr(
        reinterpret_cast<std::uint64_t>(image.buffer.virt_addr[1]),
        image.buffer.size[1]);
    copyNv12(out, image.buffer.width, image.buffer.height, image.buffer.stride,
             image.buffer.virt_addr[0], image.buffer.virt_addr[1]);
    frame_id = image.info.frame_id;
    timestamp_ns = image.info.timestamps;
    hbn_vnode_releaseframe(pipeline.vse, 0, &image);
    last_error_.clear();
    return true;
  }

  bool acquireLeft(int timeout_ms, Nv12Image& out, std::uint64_t& frame_id,
                   std::uint64_t& timestamp_ns) {
    return acquire(left_, timeout_ms, out, frame_id, timestamp_ns);
  }

  bool acquireRight(int timeout_ms, Nv12Image& out, std::uint64_t& frame_id,
                    std::uint64_t& timestamp_ns) {
    return acquire(right_, timeout_ms, out, frame_id, timestamp_ns);
  }

  void stop() noexcept {
    stopOne(left_);
    stopOne(right_);
    initialized_ = false;
  }

  const std::string& lastError() const { return last_error_; }

 private:
  bool call(const char* operation, int ret, int host) {
    if (ret == 0) return true;
    std::ostringstream message;
    message << operation << "(host=" << host << ") failed with code " << ret;
    last_error_ = message.str();
    return false;
  }

  bool prepareOne(Pipeline& pipeline) {
    if (!applyDeviceTreeRouting(pipeline)) {
      last_error_ = "failed to read vcon rx_phy/lpwm_chn for host=" +
                    std::to_string(pipeline.host);
      return false;
    }
    int sensor_address = 0;
    bool sensor_found = false;
    for (const int address : {0x30, 0x32, 0x33}) {
      if (readSc132gsChipId(pipeline.i2c_bus, address)) {
        sensor_address = address;
        sensor_found = true;
        break;
      }
    }
    if (!sensor_found) {
      last_error_ = "SC132GS chip ID 0x0132 was not found on i2c bus=" +
                    std::to_string(pipeline.i2c_bus) + " host=" +
                    std::to_string(pipeline.host);
      return false;
    }
    pipeline.camera.addr = static_cast<std::uint32_t>(sensor_address);
    int ret = hbn_camera_create(&pipeline.camera, &pipeline.camera_handle);
    if (!call("hbn_camera_create", ret, pipeline.host)) return false;
    pipeline.camera_created = true;

    ret = hbn_vnode_open(HB_VIN, pipeline.host, AUTO_ALLOC_ID, &pipeline.vin);
    if (!call("hbn_vnode_open(VIN)", ret, pipeline.host)) return false;
    pipeline.vin_created = true;
    if (!call("hbn_vnode_set_attr(VIN)",
              hbn_vnode_set_attr(pipeline.vin, &pipeline.vin_node_attr),
              pipeline.host) ||
        !call("hbn_vnode_set_ichn_attr(VIN)",
              hbn_vnode_set_ichn_attr(pipeline.vin, 0, &pipeline.vin_ichn_attr),
              pipeline.host) ||
        !call("hbn_vnode_set_ochn_attr(VIN)",
              hbn_vnode_set_ochn_attr(pipeline.vin, 0, &pipeline.vin_ochn_attr),
              pipeline.host)) {
      return false;
    }
    hbn_buf_alloc_attr_t vin_buffers{};
    vin_buffers.buffers_num = 3;
    vin_buffers.is_contig = 1;
    vin_buffers.flags = HB_MEM_USAGE_CPU_READ_OFTEN |
                        HB_MEM_USAGE_CPU_WRITE_OFTEN | HB_MEM_USAGE_CACHED |
                        HB_MEM_USAGE_HW_CIM |
                        HB_MEM_USAGE_GRAPHIC_CONTIGUOUS_BUF;
    if (!call("hbn_vnode_set_ochn_buf_attr(VIN)",
              hbn_vnode_set_ochn_buf_attr(pipeline.vin, 0, &vin_buffers),
              pipeline.host)) {
      return false;
    }
    if (hostHasMclkConfig(pipeline.host)) {
      pipeline.vin_attr_ex.ex_attr_type = VIN_STATIC_MCLK_ATTR;
      if (!call("hbn_vnode_set_attr_ex(VIN MCLK)",
                hbn_vnode_set_attr_ex(pipeline.vin, &pipeline.vin_attr_ex),
                pipeline.host)) {
        return false;
      }
    }

    ret = hbn_vnode_open(HB_ISP, 0, AUTO_ALLOC_ID, &pipeline.isp);
    if (!call("hbn_vnode_open(ISP)", ret, pipeline.host)) return false;
    pipeline.isp_created = true;
    if (!call("hbn_vnode_set_attr(ISP)",
              hbn_vnode_set_attr(pipeline.isp, &pipeline.isp_attr),
              pipeline.host) ||
        !call("hbn_vnode_set_ichn_attr(ISP)",
              hbn_vnode_set_ichn_attr(pipeline.isp, 0, &pipeline.isp_ichn_attr),
              pipeline.host) ||
        !call("hbn_vnode_set_ochn_attr(ISP)",
              hbn_vnode_set_ochn_attr(pipeline.isp, 0, &pipeline.isp_ochn_attr),
              pipeline.host)) {
      return false;
    }
    hbn_buf_alloc_attr_t isp_buffers{};
    isp_buffers.buffers_num = 3;
    isp_buffers.is_contig = 1;
    isp_buffers.flags = HB_MEM_USAGE_CPU_READ_OFTEN |
                        HB_MEM_USAGE_CPU_WRITE_OFTEN | HB_MEM_USAGE_CACHED |
                        HB_MEM_USAGE_HW_ISP |
                        HB_MEM_USAGE_GRAPHIC_CONTIGUOUS_BUF;
    if (!call("hbn_vnode_set_ochn_buf_attr(ISP)",
              hbn_vnode_set_ochn_buf_attr(pipeline.isp, 0, &isp_buffers),
              pipeline.host)) {
      return false;
    }

    ret = hbn_vnode_open(HB_VSE, 0, AUTO_ALLOC_ID, &pipeline.vse);
    if (!call("hbn_vnode_open(VSE)", ret, pipeline.host)) return false;
    pipeline.vse_created = true;
    // 采集用：VSE 全幅直通（不做 ROI 裁剪/缩放，保留原始分辨率 NV12）
    vse_attr_t vse_attr{};
    vse_ichn_attr_t vse_input{};
    vse_input.width = kSensorWidth;
    vse_input.height = kSensorHeight;
    vse_input.fmt = FRM_FMT_NV12;
    vse_input.bit_width = 8;
    vse_ochn_attr_t vse_output{};
    vse_output.chn_en = CAM_TRUE;
    vse_output.roi.x = 0;
    vse_output.roi.y = 0;
    vse_output.roi.w = kSensorWidth;
    vse_output.roi.h = kSensorHeight;
    vse_output.target_w = kSensorWidth;
    vse_output.target_h = kSensorHeight;
    vse_output.fmt = FRM_FMT_NV12;
    vse_output.bit_width = 8;
    vse_output.fps.src = 0;
    vse_output.fps.dst = 0;
    if (!call("hbn_vnode_set_attr(VSE)",
              hbn_vnode_set_attr(pipeline.vse, &vse_attr), pipeline.host) ||
        !call("hbn_vnode_set_ichn_attr(VSE)",
              hbn_vnode_set_ichn_attr(pipeline.vse, 0, &vse_input),
              pipeline.host) ||
        !call("hbn_vnode_set_ochn_attr(VSE)",
              hbn_vnode_set_ochn_attr(pipeline.vse, 0, &vse_output),
              pipeline.host)) {
      return false;
    }
    hbn_buf_alloc_attr_t vse_buffers{};
    vse_buffers.buffers_num = 3;
    vse_buffers.is_contig = 1;
    vse_buffers.flags = HB_MEM_USAGE_CPU_READ_OFTEN |
                        HB_MEM_USAGE_CPU_WRITE_OFTEN | HB_MEM_USAGE_CACHED |
                        HB_MEM_USAGE_GRAPHIC_CONTIGUOUS_BUF;
    if (!call("hbn_vnode_set_ochn_buf_attr(VSE)",
              hbn_vnode_set_ochn_buf_attr(pipeline.vse, 0, &vse_buffers),
              pipeline.host)) {
      return false;
    }

    if (!call("hbn_vflow_create", hbn_vflow_create(&pipeline.flow),
              pipeline.host)) {
      return false;
    }
    pipeline.flow_created = true;
    if (!call("hbn_vflow_add_vnode(VIN)",
              hbn_vflow_add_vnode(pipeline.flow, pipeline.vin), pipeline.host) ||
        !call("hbn_vflow_add_vnode(ISP)",
              hbn_vflow_add_vnode(pipeline.flow, pipeline.isp), pipeline.host) ||
        !call("hbn_vflow_add_vnode(VSE)",
              hbn_vflow_add_vnode(pipeline.flow, pipeline.vse), pipeline.host) ||
        !call("hbn_vflow_bind_vnode(VIN-ISP)",
              hbn_vflow_bind_vnode(pipeline.flow, pipeline.vin, 0, pipeline.isp,
                                   0),
              pipeline.host) ||
        !call("hbn_vflow_bind_vnode(ISP-VSE)",
              hbn_vflow_bind_vnode(pipeline.flow, pipeline.isp, 0, pipeline.vse,
                                   0),
              pipeline.host) ||
        !call("hbn_camera_attach_to_vin",
              hbn_camera_attach_to_vin(pipeline.camera_handle, pipeline.vin),
              pipeline.host)) {
      return false;
    }
    return true;
  }

  bool startOne(Pipeline& pipeline) {
    if (!call("hbn_vflow_start", hbn_vflow_start(pipeline.flow),
              pipeline.host)) {
      return false;
    }
    pipeline.flow_started = true;
    return true;
  }

  static void stopOne(Pipeline& pipeline) noexcept {
    if (pipeline.flow_started) hbn_vflow_stop(pipeline.flow);
    pipeline.flow_started = false;
    if (pipeline.vse_created) hbn_vnode_close(pipeline.vse);
    pipeline.vse_created = false;
    if (pipeline.isp_created) hbn_vnode_close(pipeline.isp);
    pipeline.isp_created = false;
    if (pipeline.vin_created) hbn_vnode_close(pipeline.vin);
    pipeline.vin_created = false;
    if (pipeline.camera_created) hbn_camera_destroy(pipeline.camera_handle);
    pipeline.camera_created = false;
    if (pipeline.flow_created) hbn_vflow_destroy(pipeline.flow);
    pipeline.flow_created = false;
  }

  int left_host_ = -1;
  int right_host_ = -1;
  Pipeline left_;
  Pipeline right_;
  bool initialized_ = false;
  std::string last_error_;
};

DualVioPipeline::DualVioPipeline(int left_host, int right_host)
    : impl_(std::make_unique<Impl>(left_host, right_host)) {}

DualVioPipeline::~DualVioPipeline() = default;

bool DualVioPipeline::initialize() { return impl_->initialize(); }

bool DualVioPipeline::acquireLeft(int timeout_ms, Nv12Image& out,
                                  std::uint64_t& frame_id,
                                  std::uint64_t& timestamp_ns) {
  return impl_->acquireLeft(timeout_ms, out, frame_id, timestamp_ns);
}

bool DualVioPipeline::acquireRight(int timeout_ms, Nv12Image& out,
                                   std::uint64_t& frame_id,
                                   std::uint64_t& timestamp_ns) {
  return impl_->acquireRight(timeout_ms, out, frame_id, timestamp_ns);
}

void DualVioPipeline::stop() noexcept { impl_->stop(); }

const std::string& DualVioPipeline::lastError() const {
  return impl_->lastError();
}

}  // namespace ps::vio
