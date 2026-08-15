# ps_collector · 双目数据采集工具

PathSentry Pipeline 的数据采集端（X5 板端运行），实现 [docs/01_data.md](../../docs/01_data.md) 的采集规范：

- 双路 MIPI SC132GS（1088×1280@30fps），**LPWM 硬件触发**保证左右目同曝；
- 取帧按**硬件时间戳配对**（容差默认 1ms），不齐则丢旧帧重对齐；
- VSE 全幅直通输出 NV12，深拷贝归还硬件 buffer 后交落盘线程；
- 后台落盘线程（有界队列，满则丢新帧不拖慢采集）做 NV12→BGR→JPEG/PNG；
- 产物目录结构与 index/meta 记录，直接对接后续去冗余抽帧与预标注。

## 产物结构

```
<output>/<scene>_<YYYYmmdd_HHMMSS>/
├── left/000001.jpg ...      # 与 right 同序号一一配对
├── right/000001.jpg ...
├── index.csv                # frame_index,左右frame_id,硬件时间戳ns,skew_ns,文件路径
└── meta.json                # 场景/格式/stride/传感器/同步方式等会话元数据
```

## 编译（宿主机）

```bash
# 依赖：aarch64-linux-gnu-gcc / g++ 交叉工具链
./build.sh                 # 默认 RDK_ROOT=../../../RDK_resource/RDK_proj，可用环境变量覆盖
```

## 部署与使用（板端）

```bash
scp build_aarch64/ps_collector root@<板子IP>:/root/

# 板端示例：草地逆光场景，抽帧 1/3，限时 5 分钟
./ps_collector -o /root/data/raw --scene grass_backlight --stride 3 --duration-sec 300

# 无损 PNG、全帧率、限定 2000 对
./ps_collector -o /root/data/raw --scene indoor --format png --max-pairs 2000
```

其他参数：`--left-host/--right-host`（MIPI host，默认 0/1）、`--jpg-quality`、`--timeout-ms`、`--max-skew-ns`、`--help`。

## 验收要点（对应 docs/01_data.md）

1. `index.csv` 中 `skew_ns` 分布应 ≤1e6ns（LPWM 硬同步生效的证据）；
2. 采集统计行 `dropped=0` 表示磁盘跟得上；若 dropped 高，加大 `--stride` 或换写盘更快的存储；
3. 结束后抽查 left/right 同序号图像应无肉眼可辨的时间错位（运动物体位置一致）。

## 实现说明

- 采集管线改编自 `RDK_resource/RDK_proj/mipi_vse_bpu_zerocopy`（去掉 BPU 推理与异步日志依赖），sensor 寄存器配置（`src/sensor/`）与 vp_sensors.h 直接 vendored；
- NV12 双 plane 带 stride，转换用 `cv::cvtColorTwoPlane`（正确处理对齐 padding）；
- 与参考工程差异：VSE 输出改为**全幅直通**（参考工程是 640×640 ROI，那是推理用的，采集要保原始分辨率）。

## IMU（规划）

SC132GS 嵌入式数据可经 `hbn_camera_parse_emb` 解析（本工程暂未启用，当前版本用硬件时间戳做帧对齐已满足双目训练数据要求）。
