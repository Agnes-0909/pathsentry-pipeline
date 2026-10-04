# P02 剪枝与结构优化最终报告

更新日期：2026-10-04  
报告范围：以 P01-B1 为起点的结构化剪枝、分割头结构优化、检测主干敏感度验证，以及最终推理融合与双输出导出。

## 1. 最终结论

当前按**速度优先、质量门槛不退化**选择的最终模型为 H2 的推理融合版本：

- 结构：YOLO11s 检测分支保持 P3/P4/P5，分割头在 P3 生成 logits 后上采样 2 倍，移除 P2 refine 路径。
- 权重：[`best_fused.pt`](final/best_fused.pt)。它由 [`finetuned_best.pt`](experiments/p02_h2_p3_output/finetuned_best.pt) 做 Conv/BN 融合得到，原训练权重未覆盖。
- 输入：固定 `1×3×832×960`，即宽×高 `960×832`。
- 推理接口：检测原始输出 `[1, 6, 16380]`，分割 logits `[1, 1, 208, 240]`；检测解码/NMS 和分割阈值仍由调用方负责。
- 参考导出：[`best_fused.onnx`](final/best_fused.onnx)，静态 batch 1、opset 17，包含 `detection` 和 `segmentation` 两个输出。

H2 融合模型在 RTX 4090 上的模型前向、NMS 和分割阈值处理均值为 `2.680 ms`；同一进程、同一批验证图下，融合 B1 为 `3.141 ms`，公平加速 `14.69%`。它没有达到原先设定的 20% 端到端目标，因此本阶段应表述为“完成了可复现的中等加速”，不应宣称达到目标。

## 2. 质量结果

最终融合只改变推理图中的 Conv/BN 表达，指标与 H2 仅有浮点级差异。

| 模型/数据集 | mAP50 | mAP50-95 | person AP50-95 | vehicle AP50-95 | 分割 IoU | Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 融合 / val | 0.2469 | 0.1145 | 0.0184 | 0.2106 | 0.99444 | 0.99721 |
| H2 融合 / test | 0.5922 | 0.2855 | 0.2815 | 0.2894 | 0.99319 | 0.99658 |

完整结果分别见 [`val_metrics.json`](final/val_metrics.json) 和 [`test_metrics.json`](final/test_metrics.json)。测试集只用于最终候选核查，没有参与 H2 选择。数据仍按时序切分，测试集高于验证集不能解释为跨场景泛化能力。

## 3. 候选比较

| 候选 | 参数量 | GMACs | val mAP50-95 | val person AP | val IoU | 融合后延迟 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1 | 10,003,415 | 31.89 | 0.1145 | 0.0184 | 0.99425 | 3.141 ms | 压缩前基线 |
| S1 | 9,181,959 | 30.33 | 0.1111 | 0.0155 | 0.9941 | 约 3.0 ms | 质量通过，速度收益有限 |
| H1 | 9,605,399 | 24.04 | 0.1145 | 0.0184 | 0.99465 | 约 3.1 ms | 备选，保留 P2 细节 |
| H2 | 9,839,191 | 23.68 | 0.1145 | 0.0184 | 0.99444 | **2.680 ms** | **最终速度候选** |
| 联合敏感度剪枝 | 9,462,199 | 23.16 | 0.1185 | 0.0222 | 0.99408 | 约 2.67 ms | 质量研究候选，未带来额外速度 |

S2/S3 的检测质量未通过门槛；联合敏感度剪枝经过微调后检测指标较好，但与 H2 的融合延迟基本相同，且分割 IoU 略低，因此不作为最终部署权重。

## 4. 延迟与内存口径

测速使用 RTX 4090、batch 1、固定 32 张验证图、图像预加载 GPU；计时包含模型前向、检测 NMS 和分割阈值处理，不包含磁盘读取、预处理和主机到 GPU 拷贝。每个模型 4 轮交替测量，报告均值的中位数。

| 版本 | 中位延迟 | 峰值 CUDA 分配 |
| --- | ---: | ---: |
| B1 未融合 | 3.476 ms | 约 666 MiB |
| B1 融合 | 3.141 ms | 约 666 MiB |
| H2 未融合 | 3.057 ms | 约 593 MiB |
| H2 融合 | **2.680 ms** | **约 587 MiB** |

原始数据见 [`benchmark.json`](final/benchmark.json)。H2 融合相对未融合 B1 的 22.89% 不能作为纯结构收益，因为两者融合状态不同；严格比较应使用融合 B1 与融合 H2 的 14.69%。

## 5. 部署产物与复核

- PyTorch 融合权重：[`final/best_fused.pt`](final/best_fused.pt)
- 静态双输出 ONNX：[`final/best_fused.onnx`](final/best_fused.onnx)
- 权重和环境信息：[`final/artifact.json`](final/artifact.json)
- ONNX 校验记录：[`final/onnx_validation.json`](final/onnx_validation.json)
- 固定验证样本可视化：[`final/visuals/contact_00.jpg`](final/visuals/contact_00.jpg)
- 可复现脚本：[`scripts/finalize_best.py`](scripts/finalize_best.py)

ONNX 使用 ONNX Runtime CPU 校验了两个固定样本：检测输出形状均为 `[1,6,16380]`，分割输出形状均为 `[1,1,208,240]`，分割二值 mask agreement 为 `1.0`；检测最大绝对差约 `8e-4`，分割最大绝对差约 `2.2e-5`。当前环境没有 TensorRT，ONNX 只完成导出和 CPU 数值校验，不能代替 TensorRT FP16/INT8 的部署速度结论。

重新生成最终产物：

```bash
/home/huace/miniconda3/envs/yolo/bin/python \
  04_pruning/scripts/finalize_best.py --device cuda:0
```

加载融合 PyTorch 权重时使用本工程生成的可信完整 checkpoint，并设置 `weights_only=False`；部署前应固定输入尺寸、后处理阈值和类别顺序（`person`, `vehicle`）。

## 6. 收尾判断

P02 已完成既定的剪枝、结构对比、敏感度验证、推理融合和 ONNX 双输出导出。最终保留 H2 融合权重用于速度优先部署；B1 和 H2 原始训练权重、S1/H1 以及联合敏感度权重均保留用于回溯和研究。当前检测任务仍是多任务系统的主要短板，P02 的目标是压缩和加速，并未解决 P01 中远处小行人检测效果不足的问题。
