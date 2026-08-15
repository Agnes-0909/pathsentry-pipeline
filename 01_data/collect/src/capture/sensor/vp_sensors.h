#ifndef MIPI_VSE_BPU_ZEROCOPY_VP_SENSORS_H_
#define MIPI_VSE_BPU_ZEROCOPY_VP_SENSORS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vin_cfg.h"
#include "isp_cfg.h"
#include "hb_camera_data_config.h"
#include "cam_def.h"

typedef struct vp_sensor_config_s {
  int16_t chip_id_reg;
  int16_t chip_id;
  uint32_t sensor_i2c_addr_list[8];
  char sensor_name[128];
  char config_file[128];
  camera_config_t* camera_config;
  vin_node_attr_t* vin_node_attr;
  vin_ichn_attr_t* vin_ichn_attr;
  vin_ochn_attr_t* vin_ochn_attr;
  vin_attr_ex_t* vin_attr_ex;
  isp_attr_t* isp_attr;
  isp_ichn_attr_t* isp_ichn_attr;
  isp_ochn_attr_t* isp_ochn_attr;
  int16_t ex_chip_id;
} vp_sensor_config_t;

extern vp_sensor_config_t sc132gs_linear_1088x1280_raw10_30fps_1lane;

#ifdef __cplusplus
}
#endif

#endif
