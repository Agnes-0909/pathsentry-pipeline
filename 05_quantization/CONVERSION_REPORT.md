# RDK X5 转换报告

日期：2026-10-06。工具链来自本机镜像 `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8`，版本为 `hb_mapper 1.24.3`、`hbdk 3.49.15`、`horizon_nn 1.1.0`，目标架构为 `bayes-e`。

## 输入和校准

- 输入：固定 `[1, 3, 832, 960]`，双输出检测 `[1, 6, 16380]`、分割 `[1, 1, 208, 240]`。
- ONNX：opset 11。原始 opset 17 被工具链拒绝，错误为最大支持版本为 11；使用训练环境重新导出 opset 11 后通过 checker。
- 输入配置：训练侧 RGB/NCHW，运行时 NV12，`data_scale=1/255`。
- 校准集：验证集固定随机种子 42 抽取 50 张，生成 RGB、NCHW、float32 的 `.rgbchw` 文件；文件保持 0–255 原始量程，由 mapper 的 `data_scale=1/255` 统一缩放一次。
- 编译：`O3`、`latency`、8 jobs、双输出保留。

## 为什么会有两个子图

这里的“两个图”是两个 BPU 子图，并不是模型被复制成了两个完整网络。`hb_mapper` 按算子支持情况和张量边界切图：第 0 个子图包含主干、检测/分割卷积和特征融合；第 1 个子图包含检测 DFL 解码和最终检测拼接。DFL 的部分 `Reshape/Transpose/Softmax/ReduceSum` 以及 H2 分割最后的 `Resize` 仍标记为 CPU 节点，因此端到端时间还要加上运行时 CPU 段。

这和官方 `export_monkey_patch.py` 的思路一致：固定输入、opset 11，并把检测头导出成 BPU 友好的逐尺度输出，把 DFL 解码和 NMS 放在后处理。当前自定义 `MultiTaskYOLO11s` 没有直接套用官方 `Detect` 类，所以现有结构无需为“能转换”而重训或重写；如果要减少 CPU 段、降低延迟，下一版应增加 RDK X5 export wrapper，将检测头改成 raw per-scale cls/reg 输出并把 DFL/NMS 后移，分割最后一级 Resize 也可移到 CPU 后处理。这属于导出/部署结构调整，不改变共享主干和两个任务头的训练结构。

## Checker 结果

B1 和 H2 的 opset 11 ONNX 均通过 `hb_mapper checker --march bayes-e`。工具链识别两个输出，并将大多数卷积、激活、Resize 放到 BPU。当前图仍有 CPU 节点：YOLO DFL 的部分 Softmax/ReduceSum，以及 H2 分割最终输出 Resize；这不阻塞生成 `.bin`，但必须计入板端端到端延迟。

## PTQ 结果

| 模型 | `.bin` | 检测 cosine | 分割 cosine | 编译状态 |
| --- | --- | ---: | ---: | --- |
| B1 | `compiled_b1/b1_dual_output_opset11_bayese_960x832.bin` | 0.999681 | 0.998604 | 成功 |
| H2 | `compiled_h2/h2_dual_output_opset11_bayese_960x832.bin` | 0.999631 | 0.999147 | 成功 |

余弦相似度是工具链浮点/量化输出对比，不等同于数据集 mAP 或 IoU。两个 `.bin` 都已在 `192.168.1.12` 上加载，对验证集前 20 张实际图片做了 FP32 ONNX 与板端输出对比：

| 模型 | 检测 cosine / MAE / RMSE | 分割 logits cosine / MAE / RMSE | 分割阈值一致率 / mask IoU |
| --- | --- | --- | --- |
| B1 | 0.999537 / 4.723 / 9.559 | 0.999216 / 0.377 / 0.527 | 99.978% / 99.953% |
| H2 | 0.999498 / 4.785 / 9.922 | 0.998864 / 0.510 / 0.640 | 99.958% / 99.914% |

这组数使用与板端一致的 960×832 NV12 输入路径，属于实际板端量化损失；完整数据集仍应继续计算检测 mAP、person AP、分割 IoU 和 Dice。

## Raw-head DFL 验证

参考官方导出 patch，新增了 [raw-head 导出脚本](scripts/export_raw_head_opset11.py)：每个尺度输出 `cls NHWC` 和 `bbox NHWC`，其中 bbox 为 `4×16` DFL logits；[dfl_postprocess.py](scripts/dfl_postprocess.py) 使用相同的 stride、半像素 anchor、softmax expectation 和 `dist2bbox` 规则完成后处理。

- H2 raw-head ONNX 输出：`cls_0/bbox_0 = 104×120`、`cls_1/bbox_1 = 52×60`、`cls_2/bbox_2 = 26×30`，另保留分割输出。
- 20 张真实验证图上的 ONNX raw-head + 自定义 DFL 与原始解码 ONNX：平均 cosine **0.999999994**，最大绝对误差 **1.83e-4**。
- raw-head `.bin` 已通过 X5 checker 和 `bayes-e` 编译：[h2_raw_head_opset11_bayese_960x832.bin](compiled_h2_raw/h2_raw_head_opset11_bayese_960x832.bin)，BPU 主子图估算 **30.0969 ms / 33.23 FPS**。
- raw-head `.bin` 上板后，20 张真实图片经同一 DFL 后处理得到检测输出 cosine **0.999631**、MAE **2.800**、RMSE **8.295**；分割输出与 FP32 的 cosine **0.998864**，mask IoU **99.914%**。

因此 DFL 后处理的数值实现已经验证通过；raw-head 版本可以作为下一步性能优化的部署候选，DFL 和 NMS 由端侧 C++ 后处理实现。

## `hb_perf` 结果

| 模型 | 主子图 | 第二子图 | 静态合计 | 估算 FPS |
| --- | ---: | ---: | ---: | ---: |
| B1 | 38.8621 ms | 5.8687 ms | 44.7308 ms | 25.73 / 170.4 |
| H2 | 34.3064 ms | 5.8687 ms | 40.1751 ms | 29.15 / 170.4 |

H2 相对 B1 的静态合计估算减少约 10.2%。`hb_perf` 运行在当前 x86 Docker 工具链环境，输出中的 FPS 是子图估算值；官方日志同时提示压缩访存模型的实际延迟依赖输入数据。此结果不能替代 RDK X5 实板的 BPU、CPU 后处理、NV12 输入和应用调度总延迟。

## 产物位置

- H2 `.bin`：[compiled_h2/h2_dual_output_opset11_bayese_960x832.bin](compiled_h2/h2_dual_output_opset11_bayese_960x832.bin)
- B1 `.bin`：[compiled_b1/b1_dual_output_opset11_bayese_960x832.bin](compiled_b1/b1_dual_output_opset11_bayese_960x832.bin)
- mapper 工作目录、checker 中间文件和 `hb_perf` 可视化结果已清理；需要复核时按 [README.md](README.md) 中的 Docker 命令重新生成。
- 转换脚本：[scripts/rdk_mapper.py](scripts/rdk_mapper.py)
- opset 11 导出脚本：[scripts/export_opset11.py](scripts/export_opset11.py)
