# 03 · 模型训练

本目录实现 PathSentry 的检测与可行驶区域分割多任务训练。模型使用 YOLO11s 的检测主干和检测头，并在共享特征上增加二值语义分割头；两个任务在一次前向传播中共同更新共享主干。

## 任务与模型

| 分支 | 输出 | 类别/目标 | 监督 |
|---|---|---|---|
| 检测头 | 边界框、类别和置信度 | `person`、`vehicle` | YOLO 归一化框标签 |
| 分割头 | 二值可行驶区域 mask | `drivable_area` | 单通道 PNG，`0=背景`、`1=可行驶` |

检测损失沿用 Ultralytics YOLO11s 实现；分割损失为 BCE 与 Dice 的平均值。总损失为 `L_det + λ_seg · L_seg`。默认 `λ_seg=5`。分割分支融合 P2/P3/P4/P5 特征，检测分支默认使用 P3/P4/P5；P01 实验另外验证了 P2 检测路径和高分辨率输入。

## 目录结构

```text
03_training/
├── README.md                         # 本阶段入口
├── MODEL_OPTIMIZATION.md             # P01 问题、实验台账和结论
├── scripts/
│   ├── multitask.py                  # 数据集、双头模型、损失和评测
│   ├── train.py                      # 单次训练
│   ├── evaluate.py                  # 验证/测试评测
│   ├── run_experiments.py            # E0–E4 实验编排
│   ├── prepare_copy_paste.py         # SAM 目标素材提取
│   ├── curate_copy_paste.py          # 素材筛选
│   ├── preview_copy_paste.py         # Copy-Paste 可视化检查
│   └── models/yolo11-p2.yaml         # P2 检测结构
└── artifacts/p01/                    # 可纳入 Git 的轻量实验证据
```

训练运行目录默认写入 `runs/`，该目录被 Git 忽略。报告所需的指标、图片和素材摘要已经复制到 [`artifacts/p01`](artifacts/p01/README.md)，模型权重的来源、大小和 SHA256 记录在 [`checkpoints.md`](artifacts/p01/checkpoints.md)。

## 数据

多任务数据由 LabelMe 标注导出，当前本机路径为：

```text
02_prelabel/labeled_data/labelme_check/yolo_multitask/
├── images/{train,val,test}/
├── labels_det/{train,val,test}/
├── masks_seg/{train,val,test}/
├── manifest.csv
└── dataset.yaml
```

`manifest.csv` 将每张图像、检测标签和语义 mask 配对。训练集 5728 张、验证集 716 张、测试集 716 张。验证和测试只使用原图，不启用 Copy-Paste。迁移到其他机器时，应修改 `dataset.yaml` 中的 `path`，或用绝对路径传给 `--data`。

## 环境

训练脚本在 PyTorch 2.9.1、Ultralytics 8.3.233 环境中验证。需要一个包含 CUDA、PyTorch、Ultralytics、OpenCV、Pillow、NumPy、PyYAML 和 tqdm 的 Python 环境；SAM 3.1 素材提取另需可用的本机模型和 CUDA 环境。

## 单次多任务训练

从仓库根目录执行，路径按当前机器调整：

```bash
YOLO_PY=/path/to/yolo/bin/python
DATA=$PWD/02_prelabel/labeled_data/labelme_check/yolo_multitask/dataset.yaml
WEIGHTS=/path/to/yolo11s.pt

$YOLO_PY 03_training/scripts/train.py \
  --data "$DATA" \
  --weights "$WEIGHTS" \
  --output 03_training/runs/yolo11s_multitask \
  --width 960 --height 832 \
  --batch 2 --epochs 100 --seg-weight 5
```

训练会保存 `best.pt`、`last.pt` 和 `history.jsonl`。最佳权重按验证集 `(mAP50-95 + 分割 IoU) / 2` 选择；`--seg-weight 0` 可运行纯检测对照。输入宽高需要是 32 的倍数；图像、框和 mask 会同步 letterbox，mask 使用最近邻缩放。

## 评测与实验

```bash
$YOLO_PY 03_training/scripts/evaluate.py \
  --data "$DATA" \
  --checkpoint 03_training/runs/yolo11s_multitask/best.pt \
  --split val \
  --output 03_training/runs/yolo11s_multitask/val_metrics.json

$YOLO_PY 03_training/scripts/evaluate.py \
  --data "$DATA" \
  --checkpoint 03_training/runs/yolo11s_multitask/best.pt \
  --split test \
  --output 03_training/runs/yolo11s_multitask/test_metrics.json
```

E0–E4 的多任务权重、分辨率和纯检测对照可由 `run_experiments.py` 编排。P01 的高分辨率、P2 检测和 Copy-Paste 实验记录在 [`MODEL_OPTIMIZATION.md`](MODEL_OPTIMIZATION.md)。当前验证集整体检测指标最好的候选是 B1，但行人检测短板仍未解决；E0 继续作为展示基线。

## 结果与复现

- [P01 优化报告](MODEL_OPTIMIZATION.md)
- [P01 Git 可追踪实验产物](artifacts/p01/README.md)
- [脚本使用说明](scripts/README.md)

报告中的测试集只在验证集选定方案后使用。重新划分数据或修改标签后，应先重跑 E0，再比较优化实验。
