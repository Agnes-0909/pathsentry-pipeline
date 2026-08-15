# 06 · 地平线 X5 端侧 C++ 部署

## 1. 部署形态

```
双目相机驱动(左目 RGB + 双目帧对)
   │
   ├──► [BPU] 量化检测+分割模型（HBM） ──► 障碍物框/类别 + 可行驶 mask
   ├──► [SAE/ACM] 双目深度（X5 硬件立体匹配加速，SGBM 兜底） ──► 视差/深度图
   │
   └──► CPU 后处理（C++）：NMS、mask 上采样、
            框-深度融合 → 障碍物 3D 距离 → 可行驶区域最近边界距离
   │
   └──► 感知输出 (ROS2 topic 或共享内存) → 小车避障/导航控制
```

- 运行时：X5 上的 Linux（含交叉编译环境），推理走 **BPU**（hrtcpubit/hb_rt），双目深度走 X5 的 **SAE 引擎**（若系统镜像带 stereo 加速），无 SAE 则 CPU/OpenCL SGBM；
- 语言与构建：C++17，CMake；推理用 OpenExplorer 的 C API（`libdnn`/`hb_dnn`），图像处理 OpenCV 4.x。

## 2. 工具链流程（开发机 → 板端）

1. Docker 内用 oe 工具链：ONNX → `hb_mapper makertbin` → HBM（见 05）；
2. 交叉编译应用：`aarch64` 工具链 + 板端 sysroot，产出可执行文件；
3. 板上验证顺序：`hrt_model_exec` 跑通单帧 → C++ demo 单帧对齐 → 接相机流水线 → 压测。

## 3. C++ 工程结构

```
deploy/
├── include/        # PerceptionNode, DepthFusion, PostProcess
├── src/
│   ├── main.cpp            # 采集-推理-融合主循环（多线程流水线）
│   ├── infer_runner.cpp    # hb_dnn 封装：模型加载、输入预处理(resize/NV12)、输出解析
│   ├── postprocess.cpp     # NMS、conf 阈值(INT8 重标定值)、mask argmax+上采样
│   ├── depth.cpp           # SGBM/SAE 深度，视差→深度 (f*B/disparity)
│   └── fusion.cpp          # 框底边中心采深度→3D 距离；mask 求最近不可行驶边界
├── configs/        # 模型路径、标定参数、阈值
└── CMakeLists.txt  # 交叉编译 + OpenExplorer runtime + OpenCV
```

关键工程点：
- **流水线并行**：采集线程 / BPU 推理线程 / 深度线程 / 融合后处理线程，队列有界丢弃旧帧（端侧宁可丢帧不积压）；
- **零拷贝**：相机 NV12 直接喂 BPU（模型输入配 NV12 layout），避免 CPU 转 RGB；
- 输出节流：感知结果 20Hz 发布，控制接口用时间戳同步。

## 4. 3D 融合算法（检测 × 深度）

1. 对每个检测框取**底边中点向下 10% 区域**的深度中值（抗噪，比框中心更接近接地点）；
2. `Z = f·B / d`（焦距×基线/视差）；无效视差（遮挡/弱纹理）时框标记 `range=unknown`，控制端保守处理；
3. 可行驶 mask 下采样至深度图分辨率，求"可行驶像素消失线"的最近深度作为前向可通行距离；
4. 时序滤波：3D 距离做 1€ filter / 简单 EKF，抑制双目深度抖动。

## 5. 评价指标

| 维度 | 指标 | 达标线 |
|---|---|---|
| 精度 | 板端 vs 仿真器输出一致性 | bbox 偏差 <1px、mask 像素一致率 >99% |
| 端到端性能 | 采集→感知输出延迟、FPS | ≥20 FPS（检测+分割+深度总流水线） |
| BPU 占用 | hrt_monitor 采样 | <70%（留控制余量） |
| 测距 | 实地标定板/卷尺对比，1/2/3/5m 各 20 次 | 5m 内绝对误差 <10cm，3m 内 <5cm |
| 功能 | 小车实地避障测试 | 障碍场景（含电线/坑洞）成功避让率 ≥95%，无误刹率记录 |

## 6. 性能调优 Checklist
- 若帧率不足：输入降 512、剪枝稀疏率加大、SAE 深度降分辨率/降频率（10Hz 深度足够低速避障）；
- 若 CPU 占用高：mask 后处理改查表 + 仅 ROI 处理；
- 弱纹理草地深度漂移：加大 SGBM block size、加纹理滤波，或对 drivable 区域内深度做中值时序平滑。

## 7. 交付物
- [ ] 交叉编译工程 `deploy/`
- [ ] 板端单元测试（对齐、测距标定）`deploy/tests/`
- [ ] 实地验收报告 `reports/deploy_eval.md`（FPS/BPU 占用/测距曲线/避障成功率）
