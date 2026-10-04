# 04 · 模型压缩与部署候选

本阶段以 03_training 的 P01-B1 多任务模型为基线，评估结构化通道剪枝、分割头简化、检测层敏感度和推理图融合。模型始终保留两个任务头：YOLO11s 检测头负责 person/vehicle，分割头负责二值可行驶区域 mask；检测与分割共享主干。

## 阶段结论

当前速度优先候选是 H2 的 Conv/BN 融合版本：[final/best_fused.pt](final/best_fused.pt)，并已导出双输出 ONNX：[final/best_fused.onnx](final/best_fused.onnx)。它在固定 960×832、batch 1、RTX 4090、包含 NMS 和分割阈值处理的口径下，相对融合 B1 从 3.141 ms 降到 2.680 ms，公平加速 14.69%。验证集检测与分割质量保持在门槛内，但原定的 20% 端到端时延目标尚未达到，不能把 GMACs 下降直接表述为同等时延收益。

H2 的结构是：检测分支继续使用 P3/P4/P5；分割头移除 P2 lateral 和最高分辨率 refine，在 P3 生成 logits 后双线性上采样 2 倍。H1 保留 P2 细节路径并缩窄融合宽度，是可回溯的质量优先备选。S1 通过质量门槛但实际加速有限；S2/S3 的检测质量未通过门槛。联合敏感度剪枝经过微调后质量较好，但融合后没有超过 H2 的实测速度，因此不作为部署模型。

## 目录结构

~~~text
04_pruning/
├── README.md                          # 本阶段入口
├── PRUNING_PLAN.md                    # 设计、门槛和风险
├── EXPERIMENTS.md                     # S1/S2/S3 结构化剪枝结果
├── ARCHITECTURE_EXPERIMENTS.md        # H1/H2 分割头结构结果
├── SENSITIVITY.md                     # 检测层敏感度和联合探针
├── FINAL_REPORT.md                    # 最终选择、测速和导出复核
├── architectures.py                   # H1/H2 分割头与权重迁移
├── checkpoint_compat.py               # 旧 prune.architectures 权重兼容
├── scripts/
│   ├── README.md                      # 脚本参数和执行顺序
│   ├── run_experiments.py             # S1/S2/S3 剪枝与短周期微调
│   ├── run_architecture_experiments.py# H1/H2 分割头对比
│   ├── sensitivity_scan.py            # 单层检测敏感度扫描
│   ├── joint_prune_probe.py           # 联合低敏感层探针
│   ├── finetune_joint_probe.py        # 联合探针恢复训练
│   ├── evaluate_checkpoint.py         # 完整 checkpoint 评测
│   └── finalize_best.py               # Conv/BN 融合、测速和 ONNX 导出
├── experiments/                       # S/H 实验日志、权重和可视化
├── sensitivity/                       # 单层敏感度结果
├── sensitivity_full/                  # 联合探针及恢复训练结果
└── final/                             # H2 最终 PyTorch/ONNX 产物
~~~

experiments/、sensitivity_full/ 和 final/ 中的完整 checkpoint 体积较大。报告引用的指标、日志和图片与代码位于同一阶段目录；P01 的基线权重仍位于 03_training/runs/p01/b1/best.pt，该 runs/ 目录按工程约定被 Git 忽略。

## 基线和实验口径

| 项目 | 设置 |
| --- | --- |
| 起点 | P01-B1，多任务 YOLO11s，03_training/runs/p01/b1/best.pt |
| 输入 | 宽×高 960×832，评测 batch 1 |
| 检测 | person、vehicle，YOLO 原生 P3/P4/P5 |
| 分割 | 二值可行驶区域，B1 融合 P2/P3/P4/P5 |
| 训练数据 | P01 原始划分，训练集使用审核版 SAM Copy-Paste，概率 0.3 |
| 选择指标 | 验证集 mAP50-95、person AP50-95、分割 IoU 和同口径延迟 |
| 测速 | RTX 4090，固定 32 张验证图，预加载 GPU，包含 NMS 和分割阈值处理 |

### 结构化剪枝

S1、S2、S3 都从同一个 B1 独立开始，使用 Torch-Pruning 依赖图物理删除通道，再以相同多任务配置短周期微调。结果如下：

| 候选 | 参数量 | GMACs | val mAP50-95 | person AP | val IoU | 延迟 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1 | 10,003,415 | 31.89 | 0.1145 | 0.0184 | 0.9942 | 3.50 ms | 基线 |
| S1 | 9,181,959 | 30.33 | 0.1111 | 0.0155 | 0.9941 | 3.36 ms | 质量通过，速度收益有限 |
| S2 | 8,358,943 | 28.74 | 0.1055 | 0.0109 | 0.9941 | 3.39 ms | 检测质量未通过 |
| S3 | 7,573,903 | 27.61 | 0.1034 | 0.0114 | 0.9945 | 3.38 ms | 检测质量未通过 |

详见 [EXPERIMENTS.md](EXPERIMENTS.md)。每个实验目录包含 status.json、prune_manifest.json、微调历史、验证指标、基准测速和固定样本可视化。

### 分割头结构

H1/H2 从 B1 初始化，只替换分割头并冻结检测网络。两者检测指标保持 B1 水平，H2 的计算量和配对延迟更低：

| 候选 | 分割结构 | 参数量 | GMACs | val mAP50-95 | person AP | val IoU | 配对延迟 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | P2/P3/P4/P5，融合宽度 128→64 | 9,605,399 | 24.04 | 0.1145 | 0.0184 | 0.9947 | 3.16 ms |
| H2 | P3/P4/P5，P3 输出后上采样×2 | 9,839,191 | 23.68 | 0.1145 | 0.0184 | 0.9944 | 3.01 ms |

详见 [ARCHITECTURE_EXPERIMENTS.md](ARCHITECTURE_EXPERIMENTS.md)。

### 最终产物

H2 融合后的验证集与测试集结果：

| 数据集 | mAP50-95 | person AP | vehicle AP | 分割 IoU | Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 0.1145 | 0.0184 | 0.2106 | 0.99444 | 0.99721 |
| test | 0.2855 | 0.2815 | 0.2894 | 0.99319 | 0.99658 |

ONNX 为静态 batch 1、opset 17，检测输出 [1, 6, 16380]，分割输出 [1, 1, 208, 240]。ONNX Runtime CPU 校验的分割 mask agreement 为 1.0；当前环境没有 TensorRT，因此没有给出 TensorRT FP16/INT8 速度结论。完整说明见 [FINAL_REPORT.md](FINAL_REPORT.md)。

## 复现命令

从仓库根目录执行，并将 YOLO_PY 设置为已安装 PyTorch、Ultralytics、Torch-Pruning、THOP 和 ONNX Runtime 的 Python：

~~~bash
YOLO_PY=/path/to/yolo/bin/python

# S1/S2/S3 结构化剪枝
$YOLO_PY -u 04_pruning/scripts/run_experiments.py \
  --device cuda:0

# H1/H2 分割头结构实验
$YOLO_PY -u 04_pruning/scripts/run_architecture_experiments.py \
  --device cuda:0
~~~

实验脚本会读取 03_training/runs/p01/b1/best.pt，默认将结果写入 04_pruning/experiments/；已存在的阶段 checkpoint 会用于续跑，完成阶段会自动跳过。具体参数可通过 --help 查看。

敏感度扫描、联合探针和最终收尾按以下顺序执行：

~~~bash
$YOLO_PY 04_pruning/scripts/sensitivity_scan.py \
  --source 04_pruning/experiments/p02_h2_p3_output/finetuned_best.pt \
  --output 04_pruning/sensitivity --device cuda:0

$YOLO_PY 04_pruning/scripts/joint_prune_probe.py \
  --output 04_pruning/sensitivity_full/joint_probe --device cuda:0

$YOLO_PY 04_pruning/scripts/finetune_joint_probe.py \
  --source 04_pruning/sensitivity_full/joint_probe.pt \
  --output 04_pruning/sensitivity_full/joint_finetuned --device cuda:0

$YOLO_PY 04_pruning/scripts/finalize_best.py --device cuda:0
~~~

单个完整 checkpoint 的验证/测试评测：

~~~bash
$YOLO_PY 04_pruning/scripts/evaluate_checkpoint.py \
  --checkpoint 04_pruning/final/best_fused.pt \
  --split val --device cuda:0
~~~

## 文档索引

- [剪枝方案、门槛和风险](PRUNING_PLAN.md)
- [S1/S2/S3 实验记录](EXPERIMENTS.md)
- [H1/H2 分割头结构实验](ARCHITECTURE_EXPERIMENTS.md)
- [检测层敏感度与联合探针](SENSITIVITY.md)
- [最终模型、测速和 ONNX 校验](FINAL_REPORT.md)
