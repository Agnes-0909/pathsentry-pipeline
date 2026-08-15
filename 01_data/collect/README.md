# ps_collector · 双目数据采集工具

PathSentry Pipeline **阶段 01（数据收集）** 的板端采集工具，实现 [docs/01_data.md](../../docs/01_data.md) 的采集规范：双路 MIPI SC132GS（1088×1280@30fps）、LPWM 硬件触发左右同曝、硬件时间戳配对（默认 1ms 容差）、后台线程 NV12→BGR→JPEG/PNG 落盘并记录 index.csv。

**使用文档：[USAGE.md](USAGE.md)**（编译、部署、参数、产物格式、验收与故障排查）

## 模块结构

```
01_data/collect/
├── include/ps_collect/          # 对外头文件（模块间只经由 include/ps_collect 交互）
│   ├── stereo_frame.hpp         # core    : Nv12Image / StereoFrame 数据结构
│   ├── options.hpp              # cli     : 命令行选项解析
│   ├── vio_dual.hpp             # capture : 双目 VIO 管线与帧配对决策
│   └── disk_writer.hpp          # storage : 后台落盘线程
├── src/
│   ├── app/main.cpp             # app     : 主循环（取帧→配对→抽帧→入队）
│   ├── cli/options.cpp
│   ├── capture/
│   │   ├── vio_dual.cpp         # camera→VIN→ISP→VSE 全幅直通，LPWM 硬同步
│   │   └── sensor/              # SC132GS 寄存器配置（vendored）
│   └── storage/disk_writer.cpp  # 有界队列 + cvtColorTwoPlane + imwrite
├── CMakeLists.txt               # 四个模块各自成库（core 接口库）
└── build.sh                     # aarch64 交叉编译入口
```

依赖：仓库自带的 `3rdlibrary/`（RDK_CAMERA SDK + OpenCV aarch64，已 vendor 到本仓库、不随 git 提交，见 `.gitignore`）。工程独立自包含，`RDK_ROOT` 可覆盖。
