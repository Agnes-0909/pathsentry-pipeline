#include "vio_pipeline.hpp"

#include "async_logger.hpp"

#include <cstring>
#include <arpa/inet.h>
#include <cstdio>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <memory>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>

#include "hb_camera_interface.h"
#include "hbn_api.h"
#include "vse_cfg.h"
#include "vp_sensors.h"

namespace deploy {
namespace {

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
  pipeline.vin_node_attr.lpwm_attr.enable = 1;
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

BufferLayout makeLayout(const hbn_vnode_image_t& image) {
  BufferLayout layout;
  layout.width = image.buffer.width;
  layout.height = image.buffer.height;
  layout.stride = image.buffer.stride;
  layout.stride_y = image.buffer.stride;
  layout.stride_uv = image.buffer.stride;
  layout.y_bytes = static_cast<std::uint32_t>(image.buffer.size[0]);
  layout.uv_bytes = static_cast<std::uint32_t>(image.buffer.size[1]);
  layout.y_phy = image.buffer.phys_addr[0];
  layout.uv_phy = image.buffer.phys_addr[1];
  layout.y_virt = image.buffer.virt_addr[0];
  layout.uv_virt = image.buffer.virt_addr[1];
  layout.y_fd = image.buffer.fd[0];
  layout.uv_fd = image.buffer.fd[1];
  layout.y_share_id = image.buffer.share_id[0];
  layout.uv_share_id = image.buffer.share_id[1];
  hb_mem_graphic_buf_t queried{};
  if (layout.y_virt && hb_mem_get_graph_buf_with_vaddr(reinterpret_cast<uint64_t>(layout.y_virt), &queried) == 0) {
    layout.y_fd = queried.fd[0]; layout.uv_fd = queried.fd[1];
    layout.y_share_id = queried.share_id[0]; layout.uv_share_id = queried.share_id[1];
    layout.y_phy = queried.phys_addr[0]; layout.uv_phy = queried.phys_addr[1];
    layout.y_bytes = static_cast<std::uint32_t>(queried.size[0]); layout.uv_bytes = static_cast<std::uint32_t>(queried.size[1]);
  }
  layout.plane_count = image.buffer.plane_cnt;
  layout.format = image.buffer.format == MEM_PIX_FMT_NV12
                      ? PixelFormat::kNv12
                      : PixelFormat::kUnknown;
  return layout;
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

bool readBigEndianWord(const std::string& path, std::size_t index,
                       int& value) {
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
  const std::string gpio_path =
      "/sys/class/gpio/gpio" + std::to_string(gpio);
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
  const SensorRouting routing = makeSensorRouting(
      pipeline.mipi.rx_attr.phy, rx_phy, lpwm_channel);
  pipeline.mipi.rx_attr.phy = routing.camera_phy;
  pipeline.vin_node_attr.cim_attr.mipi_rx = routing.vin_rx;
  pipeline.vin_node_attr.cim_attr.func.ts_src = routing.timestamp_source;
  pipeline.i2c_bus = i2c_bus;
  const bool active_high = pipeline.camera.gpio_level_bit == 0;
  return pulseSensorPower(reset_gpio, active_high);
}

}  // namespace

class LeftVioPipeline::Impl {
 public:
  Impl(int left_host, AsyncLogger* logger)
      : left_host_(left_host), logger_(logger) { left_.channel = "left"; }

  ~Impl() { stop(); }

  bool initialize() {
    if (initialized_) return true;
    cloneSensorConfig(left_, left_host_);
    if (!prepareOne(left_) || !startOne(left_)) {
      stop();
      return false;
    }
    initialized_ = true;
    return true;
  }

  bool acquire(Pipeline& pipeline, int timeout_ms, deploy::VseFrame& frame) {
    hbn_vnode_image_t image{};
    const int ret = hbn_vnode_getframe(pipeline.vse, 0, timeout_ms, &image);
    if (ret != 0) {
      std::ostringstream message;
      message << "hbn_vnode_getframe(host=" << pipeline.host
              << ") failed with code " << ret;
      last_error_ = message.str();
      return false;
    }
    auto owned_image = std::make_shared<hbn_vnode_image_t>(image);
    const hbn_vnode_handle_t handle = pipeline.vse;
    frame = deploy::VseFrame(
        image.info.frame_id, makeLayout(image),
        [this, handle, owned_image, channel = pipeline.channel,
         frame_id = image.info.frame_id] {
          const int release_result =
              hbn_vnode_releaseframe(handle, 0, owned_image.get());
          emit(release_result == 0 ? LogLevel::kDebug : LogLevel::kError,
               "frame_released", channel, static_cast<int64_t>(frame_id),
               release_result == 0 ? "VSE frame returned after processing"
                                   : "VSE frame release failed",
               {{"result", static_cast<int64_t>(release_result)}});
          return release_result;
        },
        image.info.timestamps);
    const auto layout = frame.getLayout();
    emit(LogLevel::kDebug, "frame_acquired", pipeline.channel,
         static_cast<int64_t>(image.info.frame_id), "VSE output frame acquired",
         {{"hardware_timestamp", image.info.timestamps},
          {"height", int64_t{layout.height}},
          {"physical_address", layout.y_phy},
          {"width", int64_t{layout.width}}});
    last_error_.clear();
    return true;
  }

  bool acquire(int timeout_ms, deploy::VseFrame& frame) {
    return acquire(left_, timeout_ms, frame);
  }
  void stop() noexcept {
    const auto owns_resources = [](const Pipeline& pipeline) {
      return pipeline.camera_created || pipeline.vin_created ||
             pipeline.isp_created || pipeline.vse_created ||
             pipeline.flow_created || pipeline.flow_started;
    };
    if (!owns_resources(left_)) { initialized_ = false; return; }
    emit(LogLevel::kInfo, "pipeline_stopping", "left", -1, "stopping VIO flow", {});
    stopOne(left_);
    emit(LogLevel::kInfo, "pipeline_stopped", "left", -1, "VIO flow stopped", {});
    initialized_ = false;
  }

  const std::string& lastError() const { return last_error_; }

 private:
  bool call(const char* operation, int ret, int host) {
    const std::string channel = "left";
    emit(ret == 0 ? LogLevel::kDebug : LogLevel::kError, "sdk_call", channel,
         -1, operation,
         {{"host", static_cast<int64_t>(host)},
          {"result", static_cast<int64_t>(ret)}});
    if (ret == 0) return true;
    std::ostringstream message;
    message << operation << "(host=" << host << ") failed with code " << ret;
    last_error_ = message.str();
    return false;
  }

  bool prepareOne(Pipeline& pipeline) {
    emit(LogLevel::kInfo, "pipeline_initializing", pipeline.channel, -1,
         "configuring MIPI -> VIN -> ISP -> VSE",
         {{"host", static_cast<int64_t>(pipeline.host)}});
    if (!applyDeviceTreeRouting(pipeline)) {
      last_error_ = "failed to read vcon rx_phy/lpwm_chn for host=" +
                    std::to_string(pipeline.host);
      return false;
    }
    const auto sensor_address = detectSensorAddress(
        {0x30, 0x32, 0x33}, [&](int address) {
          const bool found = readSc132gsChipId(pipeline.i2c_bus, address);
          emit(LogLevel::kDebug, "sensor_probe", pipeline.channel, -1,
               found ? "SC132GS chip ID matched" : "sensor address did not match",
               {{"address", static_cast<int64_t>(address)},
                {"i2c_bus", static_cast<int64_t>(pipeline.i2c_bus)},
                {"matched", found}});
          return found;
        });
    if (!sensor_address.has_value()) {
      last_error_ = "SC132GS chip ID 0x0132 was not found on i2c bus=" +
                    std::to_string(pipeline.i2c_bus) + " host=" +
                    std::to_string(pipeline.host);
      return false;
    }
    pipeline.camera.addr = static_cast<std::uint32_t>(*sensor_address);
    emit(LogLevel::kInfo, "sensor_selected", pipeline.channel, -1,
         "SC132GS chip ID 0x0132 detected",
         {{"address", static_cast<int64_t>(*sensor_address)},
          {"i2c_bus", static_cast<int64_t>(pipeline.i2c_bus)}});
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
              hbn_vnode_set_ichn_attr(pipeline.vin, 0,
                                      &pipeline.vin_ichn_attr),
              pipeline.host) ||
        !call("hbn_vnode_set_ochn_attr(VIN)",
              hbn_vnode_set_ochn_attr(pipeline.vin, 0,
                                      &pipeline.vin_ochn_attr),
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
    if (shouldSetMclkAttribute(hostHasMclkConfig(pipeline.host))) {
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
              hbn_vnode_set_ichn_attr(pipeline.isp, 0,
                                      &pipeline.isp_ichn_attr),
              pipeline.host) ||
        !call("hbn_vnode_set_ochn_attr(ISP)",
              hbn_vnode_set_ochn_attr(pipeline.isp, 0,
                                      &pipeline.isp_ochn_attr),
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
    vse_attr_t vse_attr{};
    vse_ichn_attr_t vse_input{};
    vse_input.width = 1088;
    vse_input.height = 1280;
    vse_input.fmt = FRM_FMT_NV12;
    vse_input.bit_width = 8;
    vse_ochn_attr_t vse_output{};
    vse_output.chn_en = CAM_TRUE;
    vse_output.roi.x = 0;
    vse_output.roi.y = 169;
    vse_output.roi.w = 1088;
    vse_output.roi.h = 942;
    vse_output.target_w = 960;
    vse_output.target_h = 832;
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
              hbn_vflow_bind_vnode(pipeline.flow, pipeline.vin, 0,
                                   pipeline.isp, 0),
              pipeline.host) ||
        !call("hbn_vflow_bind_vnode(ISP-VSE)",
              hbn_vflow_bind_vnode(pipeline.flow, pipeline.isp, 0,
                                   pipeline.vse, 0),
              pipeline.host) ||
        !call("hbn_camera_attach_to_vin",
              hbn_camera_attach_to_vin(pipeline.camera_handle, pipeline.vin),
              pipeline.host)) {
      return false;
    }
    emit(LogLevel::kInfo, "pipeline_prepared", pipeline.channel, -1,
         "hardware flow prepared for synchronized start",
         {{"host", static_cast<int64_t>(pipeline.host)}});
    return true;
  }

  bool startOne(Pipeline& pipeline) {
    if (!call("hbn_vflow_start", hbn_vflow_start(pipeline.flow),
              pipeline.host)) {
      return false;
    }
    pipeline.flow_started = true;
    const auto sync = sc132gsHardwareSyncTiming();
    emit(LogLevel::kInfo, "pipeline_started", pipeline.channel, -1,
         "hardware-triggered flow started",
         {{"host", static_cast<int64_t>(pipeline.host)},
          {"sync_period_us", static_cast<uint64_t>(sync.period_us)},
          {"vse_height", int64_t{640}}, {"vse_width", int64_t{640}}});
    return true;
  }

  void emit(LogLevel level, const std::string& event,
            const std::string& channel, int64_t frame_id,
            const std::string& message, LogFields fields) noexcept {
    if (logger_ == nullptr) return;
    try {
      logger_->log({level, "vio", event, "main", channel,
                    frame_id,
                    message, std::move(fields)});
    } catch (...) {
    }
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

  int left_host_;
  Pipeline left_;
  bool initialized_ = false;
  std::string last_error_;
  AsyncLogger* logger_ = nullptr;
};

LeftVioPipeline::LeftVioPipeline(int left_host, AsyncLogger* logger)
    : impl_(std::make_unique<Impl>(left_host, logger)) {}

LeftVioPipeline::~LeftVioPipeline() = default;

bool LeftVioPipeline::initialize() { return impl_->initialize(); }
bool LeftVioPipeline::acquire(int timeout_ms, deploy::VseFrame& frame) {
  return impl_->acquire(timeout_ms, frame);
}
void LeftVioPipeline::stop() noexcept { impl_->stop(); }
const std::string& LeftVioPipeline::lastError() const {
  return impl_->lastError();
}

}  // namespace deploy
